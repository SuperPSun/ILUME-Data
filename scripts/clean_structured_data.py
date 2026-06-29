"""Clean non-ILThermo structured CSV files with auditable rejection reports."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_SOURCES = ("AIonopedia", "ILBERT", "after_AIonopedia", "simulation")
EXCLUDED_SOURCES = {"ILThermo"}
IDENTIFIER_COLUMNS = ("cation", "anion", "solute", "solvent", "smiles", "SMILES", "mol_id")
SMILES_COLUMNS = ("cation", "anion", "solute", "solvent", "smiles", "SMILES")
CONDITION_COLUMNS = ("temperature_K", "pressure_kPa", "frequency_MHz", "wavelength_nm", "phase")
NON_LABEL_COLUMNS = {*IDENTIFIER_COLUMNS, *CONDITION_COLUMNS}
FRACTION_COLUMNS = {"ESP_pos_frac", "ESP_neg_frac", "q_pos_frac"}
IQR_MIN_ROWS = 20

HARD_THRESHOLDS: dict[str, tuple[float, float]] = {
    "density_g/cm^3": (0.5, 3.0),
    "density_err_g/cm^3": (0, 0.1),
    "viscosity_mPa*s_log10": (-1, 6),
    "surface_tension_mN/m": (0, 120),
    "melting_point_K": (150, 800),
    "solvation_kcal/mol": (-100, 100),
    "transfer_kcal/mol": (-50, 50),
    "transfer_organic_kcal/mol": (-50, 50),
    "partition_log10": (-10, 15),
    "electrical_conductivity_S/m": (0, 20),
    "lnEC_unitless": (-12, 5),
    "x_CO2_unitless": (0, 1),
    "ln_x_CO2_unitless": (-20, 0),
    "hc_unitless": (0, 5000),
    "lnhc_unitless": (0, 10),
    "EC50_unitless": (0, 1_000_000),
    "logEC50_unitless": (-5, 8),
    "glass_transition_temperature_K": (100, 600),
    "refractive_index_unitless": (1, 2),
    "thermal_conductivity_W/m/K": (0, 1),
    "thermal_decomposition_temperature_K": (250, 1000),
    "heat_capacity_J/mol/K": (0, 5000),
    "heat_capacity_err_J/mol/K": (0, 500),
    "heat_of_vaporization_kJ/mol": (0, 1000),
    "heat_of_vaporization_err_kJ/mol": (0, 100),
    "thermal_expansion_K^-1": (0, 0.01),
    "thermal_expansion_err_K^-1": (0, 0.005),
    "HOMO_eV": (-20, 10),
    "LUMO_eV": (-20, 10),
    "gap_eV": (0, 20),
    "charge": (-20, 20),
    "solv": (-100, 100),
}
IQR_ELIGIBLE_COLUMNS = {
    column
    for column in HARD_THRESHOLDS
    if column
    not in {
        "charge",
        "density_err_g/cm^3",
        "heat_capacity_err_J/mol/K",
        "heat_of_vaporization_err_kJ/mol",
        "thermal_expansion_err_K^-1",
    }
}

OUTPUT_ORDER = (
    "mol_id",
    "cation",
    "anion",
    "solute",
    "solvent",
    "smiles",
    "SMILES",
    "temperature_K",
    "pressure_kPa",
    "frequency_MHz",
    "wavelength_nm",
    "phase",
)


@dataclass
class CleanResult:
    source: str
    filename: str
    input_rows: int
    output_rows: int
    rejected_rows: int
    rejection_counts: dict[str, int] = field(default_factory=dict)
    unit_conversions: list[str] = field(default_factory=list)
    iqr_bounds: dict[str, dict[str, float | bool | int | None]] = field(default_factory=dict)
    conflict_key_groups: int = 0
    output_path: str = ""
    rejected_path: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "filename": self.filename,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "rejected_rows": self.rejected_rows,
            "rejection_counts": "; ".join(f"{key}={value}" for key, value in sorted(self.rejection_counts.items())),
            "unit_conversions": "; ".join(self.unit_conversions),
            "conflict_key_groups": self.conflict_key_groups,
            "output_path": self.output_path,
            "rejected_path": self.rejected_path,
        }


def clean_text(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return pd.NA if text == "" or text.lower() == "nan" else text


def source_from_path(path: Path) -> str:
    parts = path.parts
    if "structured" in parts:
        index = parts.index("structured")
        if index + 1 < len(parts):
            return parts[index + 1]
    return path.parent.name


def smiles_cache_for(df: pd.DataFrame) -> dict[str, str | None]:
    values: set[str] = set()
    for column in SMILES_COLUMNS:
        if column not in df.columns:
            continue
        values.update(str(value).strip() for value in df[column].dropna().unique() if str(value).strip())

    cache: dict[str, str | None] = {}
    for value in values:
        mol = Chem.MolFromSmiles(value)
        cache[value] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else None
    return cache


def cached_canonical_smiles(value: object, cache: dict[str, str | None]) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    canonical = cache.get(text)
    return canonical if canonical is not None else pd.NA


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if column in IDENTIFIER_COLUMNS or column == "phase":
            continue
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def ordered_frame(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [column for column in OUTPUT_ORDER if column in df.columns]
    extras = [column for column in df.columns if column not in ordered]
    return df[ordered + extras]


def label_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in NON_LABEL_COLUMNS]


def add_rejections(
    rejected_rows: list[dict[str, object]],
    df: pd.DataFrame,
    mask: pd.Series,
    reason: str,
    trigger_column: str,
    trigger_values: pd.Series | object,
) -> None:
    selected = df[mask]
    for index, row in selected.iterrows():
        trigger_value = trigger_values.loc[index] if isinstance(trigger_values, pd.Series) else trigger_values
        record = {"row_index": index, "rejection_reason": reason, "trigger_column": trigger_column, "trigger_value": trigger_value}
        record.update(row.to_dict())
        rejected_rows.append(record)


def standardize_units(df: pd.DataFrame, filename: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    conversions: list[str] = []

    if "density_kg/m^3" in out.columns:
        out["density_g/cm^3"] = pd.to_numeric(out.pop("density_kg/m^3"), errors="coerce") / 1000
        conversions.append("density_kg/m^3 -> density_g/cm^3")
    if filename == "simulated_density_structured.csv" and "density" in out.columns:
        out["density_g/cm^3"] = pd.to_numeric(out.pop("density"), errors="coerce") / 1000
        conversions.append("density -> density_g/cm^3 (/1000)")
    if filename == "simulated_density_structured.csv" and "density_err" in out.columns:
        out["density_err_g/cm^3"] = pd.to_numeric(out.pop("density_err"), errors="coerce") / 1000
        conversions.append("density_err -> density_err_g/cm^3 (/1000)")

    if "viscosity_mPa*s" in out.columns:
        values = pd.to_numeric(out.pop("viscosity_mPa*s"), errors="coerce")
        out["viscosity_mPa*s_log10"] = values.map(lambda value: math.log10(value) if pd.notna(value) and value > 0 else -999.0)
        conversions.append("viscosity_mPa*s -> viscosity_mPa*s_log10")
        if "ln_viscosity_mPa*s_unitless" in out.columns:
            out = out.drop(columns=["ln_viscosity_mPa*s_unitless"])
            conversions.append("dropped ln_viscosity_mPa*s_unitless after viscosity conversion")
    elif "ln_viscosity_mPa*s_unitless" in out.columns:
        values = pd.to_numeric(out.pop("ln_viscosity_mPa*s_unitless"), errors="coerce")
        out["viscosity_mPa*s_log10"] = values / math.log(10)
        conversions.append("ln_viscosity_mPa*s_unitless -> viscosity_mPa*s_log10")

    rename_only = {
        "solv": "solvation_kcal/mol",
        "Cp": "heat_capacity_J/mol/K",
        "Cp_err": "heat_capacity_err_J/mol/K",
        "Hvap": "heat_of_vaporization_kJ/mol",
        "Hvap_err": "heat_of_vaporization_err_kJ/mol",
        "alpha": "thermal_expansion_K^-1",
        "alpha_err": "thermal_expansion_err_K^-1",
        "HOMO": "HOMO_eV",
        "LUMO": "LUMO_eV",
        "gap": "gap_eV",
        "Gap": "gap_eV",
    }
    for source, target in rename_only.items():
        if source in out.columns and target not in out.columns:
            out = out.rename(columns={source: target})
            conversions.append(f"{source} -> {target}")
        elif source in out.columns and target in out.columns and source != target:
            out = out.drop(columns=[source])
            conversions.append(f"dropped duplicate {source}; kept {target}")

    return out, conversions


def required_label_missing_mask(df: pd.DataFrame, filename: str, labels: list[str]) -> pd.Series:
    if not labels:
        return pd.Series(False, index=df.index)
    required_labels = labels.copy()
    if filename == "simulated_charge_20260514_mapping_structured.csv":
        required_labels = [column for column in required_labels if column != "charge"]
    if not required_labels:
        return pd.Series(False, index=df.index)
    return df[required_labels].isna().all(axis=1)


def apply_hard_thresholds(
    df: pd.DataFrame,
    rejected_rows: list[dict[str, object]],
    active: pd.Series,
) -> pd.Series:
    for column, (lower, upper) in HARD_THRESHOLDS.items():
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        mask = active & values.notna() & ((values < lower) | (values > upper))
        if mask.any():
            add_rejections(rejected_rows, df, mask, "hard_threshold", column, values)
            active = active & ~mask
    for column in FRACTION_COLUMNS:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        mask = active & values.notna() & ((values < 0) | (values > 1))
        if mask.any():
            add_rejections(rejected_rows, df, mask, "hard_threshold", column, values)
            active = active & ~mask
    return active


def apply_iqr_rules(
    df: pd.DataFrame,
    labels: list[str],
    rejected_rows: list[dict[str, object]],
    active: pd.Series,
) -> tuple[pd.Series, dict[str, dict[str, float | bool | int | None]]]:
    bounds: dict[str, dict[str, float | bool | int | None]] = {}
    for column in labels:
        if column not in IQR_ELIGIBLE_COLUMNS:
            continue
        if column not in df.columns:
            continue
        values = pd.to_numeric(df.loc[active, column], errors="coerce").dropna()
        if len(values) < IQR_MIN_ROWS:
            bounds[column] = {"applied": False, "n": int(len(values)), "lower": None, "upper": None}
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            bounds[column] = {"applied": False, "n": int(len(values)), "lower": q1, "upper": q3}
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        bounds[column] = {"applied": True, "n": int(len(values)), "lower": lower, "upper": upper}
        full_values = pd.to_numeric(df[column], errors="coerce")
        mask = active & full_values.notna() & ((full_values < lower) | (full_values > upper))
        if mask.any():
            add_rejections(rejected_rows, df, mask, "iqr_outlier", column, full_values)
            active = active & ~mask
    return active, bounds


def count_conflict_key_groups(df: pd.DataFrame, labels: list[str]) -> int:
    key_columns = [column for column in (*IDENTIFIER_COLUMNS, *CONDITION_COLUMNS) if column in df.columns]
    if not key_columns or not labels:
        return 0
    grouped = df.groupby(key_columns, dropna=False)[labels].nunique(dropna=True)
    if grouped.empty:
        return 0
    return int(grouped.gt(1).any(axis=1).sum())


def clean_structured_file(input_path: Path, output_path: Path, rejected_path: Path | None = None) -> CleanResult:
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    input_path = Path(input_path)
    output_path = Path(output_path)
    if rejected_path is None:
        rejected_path = output_path.parent.parent / "rejected_rows" / output_path.parent.name / f"{output_path.stem}_rejected.csv"
    rejected_path = Path(rejected_path)

    df = pd.read_csv(input_path)
    input_rows = len(df)
    rejected_rows: list[dict[str, object]] = []

    for column in df.columns:
        if column in IDENTIFIER_COLUMNS or column == "phase":
            df[column] = df[column].map(clean_text)

    smiles_cache = smiles_cache_for(df)
    for column in SMILES_COLUMNS:
        if column not in df.columns:
            continue
        raw = df[column].astype("object")
        invalid_mask = raw.notna() & raw.map(lambda value: str(value).strip() != "" and smiles_cache.get(str(value).strip()) is None)
        if invalid_mask.any():
            add_rejections(rejected_rows, df, invalid_mask, "invalid_smiles", column, raw)
        df.loc[~invalid_mask, column] = raw[~invalid_mask].map(lambda value: cached_canonical_smiles(value, smiles_cache))

    active = pd.Series(True, index=df.index)
    if rejected_rows:
        invalid_indexes = [row["row_index"] for row in rejected_rows if row["rejection_reason"] == "invalid_smiles"]
        active.loc[invalid_indexes] = False

    for column in IDENTIFIER_COLUMNS:
        if column not in df.columns:
            continue
        missing = active & df[column].isna()
        if missing.any():
            add_rejections(rejected_rows, df, missing, "missing_identifier", column, "")
            active = active & ~missing

    df = coerce_numeric_columns(df)
    df, unit_conversions = standardize_units(df, input_path.name)
    labels = label_columns(df)

    missing_labels = active & required_label_missing_mask(df, input_path.name, labels)
    if missing_labels.any():
        add_rejections(rejected_rows, df, missing_labels, "missing_label", ",".join(labels), "")
        active = active & ~missing_labels

    duplicate_mask = active & df.duplicated()
    if duplicate_mask.any():
        add_rejections(rejected_rows, df, duplicate_mask, "duplicate_row", "all_columns", "")
        active = active & ~duplicate_mask

    active = apply_hard_thresholds(df, rejected_rows, active)
    active, iqr_bounds = apply_iqr_rules(df, labels, rejected_rows, active)

    cleaned = ordered_frame(df[active].reset_index(drop=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    if rejected_rows:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rejected_rows).to_csv(rejected_path, index=False)
    elif rejected_path.exists():
        rejected_path.unlink()

    rejection_counts = dict(Counter(row["rejection_reason"] for row in rejected_rows))
    return CleanResult(
        source=source_from_path(input_path),
        filename=input_path.name,
        input_rows=input_rows,
        output_rows=len(cleaned),
        rejected_rows=len(rejected_rows),
        rejection_counts=rejection_counts,
        unit_conversions=unit_conversions,
        iqr_bounds=iqr_bounds,
        conflict_key_groups=count_conflict_key_groups(cleaned, labels),
        output_path=str(output_path),
        rejected_path=str(rejected_path) if rejected_rows else "",
    )


def write_reports(results: list[CleanResult], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.as_dict() for result in results]).to_csv(output_root / "cleaning_report.csv", index=False)

    lines = ["# Structured Data Cleaning Report", ""]
    for result in results:
        lines.extend(
            [
                f"## {result.source}/{result.filename}",
                "",
                f"- Input rows: {result.input_rows}",
                f"- Output rows: {result.output_rows}",
                f"- Rejected rows: {result.rejected_rows}",
                f"- Conflict key groups retained: {result.conflict_key_groups}",
                f"- Unit conversions: {'; '.join(result.unit_conversions) if result.unit_conversions else 'none'}",
            ]
        )
        if result.iqr_bounds:
            bounds = []
            for column, detail in result.iqr_bounds.items():
                if detail["applied"]:
                    bounds.append(f"{column}: n={detail['n']}, lower={detail['lower']:.6g}, upper={detail['upper']:.6g}")
                else:
                    bounds.append(f"{column}: skipped, n={detail['n']}")
            lines.append(f"- IQR bounds: {'; '.join(bounds)}")
        if result.rejection_counts:
            lines.append(f"- Rejection counts: {'; '.join(f'{k}={v}' for k, v in sorted(result.rejection_counts.items()))}")
        if result.rejected_path:
            lines.append(f"- Full rejected rows CSV: `{result.rejected_path}`")
            rejected = pd.read_csv(result.rejected_path)
            display_columns = [
                column
                for column in [
                    "row_index",
                    "rejection_reason",
                    "trigger_column",
                    "trigger_value",
                    "mol_id",
                    "cation",
                    "anion",
                    "solute",
                    "solvent",
                    "SMILES",
                    "smiles",
                ]
                if column in rejected.columns
            ]
            lines.extend(["", markdown_table(rejected[display_columns]), ""])
        lines.append("")
    (output_root / "cleaning_report.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in df.iterrows():
        values = [str(row[column]).replace("\n", " ").replace("|", "\\|") for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def clean_non_ilthermo_structured(
    input_root: Path,
    output_root: Path,
    sources: tuple[str, ...] | list[str] = DEFAULT_SOURCES,
) -> list[CleanResult]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    results: list[CleanResult] = []
    for source in sources:
        if source in EXCLUDED_SOURCES:
            continue
        source_dir = input_root / source
        if not source_dir.exists():
            continue
        for input_path in sorted(source_dir.glob("*_structured.csv")):
            output_path = output_root / source / input_path.name
            rejected_path = output_root / "rejected_rows" / source / f"{input_path.stem}_rejected.csv"
            results.append(clean_structured_file(input_path, output_path, rejected_path))
    write_reports(results, output_root)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "structured")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "cleaned")
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = clean_non_ilthermo_structured(args.input_root, args.output_root, args.sources)
    for result in results:
        print(f"cleaned={result.output_path} rows={result.output_rows} rejected={result.rejected_rows}")


if __name__ == "__main__":
    main()
