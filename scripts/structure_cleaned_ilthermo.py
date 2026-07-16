"""Convert manually cleaned ILThermo CSVs to the current structured format."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raw_prep import net_formal_charge, parse_ion_pair_identity  # noqa: E402

from scripts.structure_raw_data import (  # noqa: E402
    CONDITION_COLUMNS,
    ILTHERMO_SPECS,
    SYSTEM_COLUMNS,
    clean_smiles,
    maybe_log10,
    ordered_frame,
    parse_phase,
    write_frame,
)

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "cleaned" / "ilt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "cleaned" / "ILThermo"
DISCARDED_COLUMNS = {
    "property_name",
    "property_unit",
    "property_value",
    "standard unit",
    "standard_unit",
    "parse_error",
    "note",
    "source_text",
}
STATIC_RELATIVE_PERMITTIVITY_LABEL = "static_relative_permittivity_unitless"
DYNAMIC_RELATIVE_PERMITTIVITY_LABEL = "dynamic_relative_permittivity_unitless"
STATIC_RELATIVE_PERMITTIVITY_OUTPUT = "ilt_static_relative_permittivity_structured.csv"
DYNAMIC_RELATIVE_PERMITTIVITY_OUTPUT = "ilt_dynamic_relative_permittivity_structured.csv"
SOURCE_SMILES_RE = re.compile(r"^\s*smiles\s*:\s*(\S+)", flags=re.IGNORECASE)


def clean_text(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() == "nan" else text


def extract_phase(row: pd.Series) -> str:
    explicit_phase = clean_text(row.get("phase"))
    if explicit_phase:
        return explicit_phase

    source_text = clean_text(row.get("source_text"))
    phase_match = re.search(r"(?:^|\|\s*)phase\s*=\s*([^|]+(?:\|[^|]+)*)\s*$", source_text, flags=re.I)
    if phase_match:
        return phase_match.group(1).strip()

    note = clean_text(row.get("note"))
    note_match = re.search(r"Phase:\s*([^|]+(?:\|[^|]+)*)", note, flags=re.I)
    if note_match:
        return note_match.group(1).strip()

    parsed_phase = parse_phase(source_text)
    return parsed_phase.strip()


def extract_standard_state_note(row: pd.Series) -> str:
    standard_state_note = clean_text(row.get("standard_state_note"))
    if standard_state_note:
        return standard_state_note

    note_text = clean_text(row.get("note"))
    if "reference state" in note_text.lower():
        return note_text
    return ""


def cleaned_row(row: pd.Series, label_column: str) -> dict[str, object]:
    output: dict[str, object] = {}
    for column in SYSTEM_COLUMNS:
        if column in row.index and not pd.isna(row[column]):
            output[column] = clean_smiles(row[column])
    for column in CONDITION_COLUMNS:
        if column in row.index and not pd.isna(row[column]):
            output[column] = row[column]

    phase = extract_phase(row)
    if phase:
        output["phase"] = phase

    standard_state_note = extract_standard_state_note(row)
    if standard_state_note:
        output["standard_state_note"] = standard_state_note

    output[label_column] = row["label"]
    return output


def validated_ion_pair(row: pd.Series) -> tuple[str, str, str | None]:
    source_text = clean_text(row.get("source_text"))
    source_match = SOURCE_SMILES_RE.search(source_text)
    if source_match:
        identity = parse_ion_pair_identity(source_match.group(1))
        if identity.error:
            return "", "", identity.error
        cation = clean_smiles(identity.cation)
        anion = clean_smiles(identity.anion)
    else:
        cation = clean_smiles(row.get("cation"))
        anion = clean_smiles(row.get("anion"))

    if not cation or not anion:
        return "", "", "missing_identifier"
    cation_charge = net_formal_charge(cation)
    anion_charge = net_formal_charge(anion)
    if cation_charge is None or anion_charge is None:
        return "", "", "invalid_smiles"
    if cation_charge <= 0 or anion_charge >= 0:
        return "", "", "invalid_ion_role"
    return cation, anion, None


def transform_cleaned_label(value: object, property_slug: str) -> object:
    if property_slug != "self_diffusion_coefficient":
        return value
    return maybe_log10(pd.to_numeric(value, errors="coerce"))


def drop_liquid_only_phase(df: pd.DataFrame) -> pd.DataFrame:
    if "phase" not in df.columns:
        return df

    phases = df["phase"].dropna().astype(str).str.strip()
    phases = phases[phases != ""]
    if not phases.empty and phases.str.lower().eq("liquid").all():
        return df.drop(columns=["phase"])
    return df


def structure_cleaned_ilthermo_frame(
    df: pd.DataFrame,
    property_slug: str,
    label_column: str | None = None,
    rejected_rows: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    spec = ILTHERMO_SPECS[property_slug]
    if "label" not in df.columns:
        raise ValueError("missing required label column")
    if property_slug == "electrical_conductivity" and "frequency_MHz" in df.columns:
        df = df[df["frequency_MHz"].isna()]
    if property_slug == "speed_of_sound" and "frequency_MHz" in df.columns:
        df = df.drop(columns=["frequency_MHz"])

    output_label = label_column or spec.label_column
    rows = []
    for row_index, csv_row in df.iterrows():
        csv_row = csv_row.copy()
        cation, anion, rejection_reason = validated_ion_pair(csv_row)
        if rejection_reason:
            if rejected_rows is not None:
                rejected = {
                    "row_index": row_index,
                    "rejection_reason": rejection_reason,
                    "trigger_column": "cation,anion",
                }
                rejected.update(csv_row.to_dict())
                rejected_rows.append(rejected)
            continue
        csv_row["cation"] = cation
        csv_row["anion"] = anion
        csv_row["label"] = transform_cleaned_label(csv_row["label"], property_slug)
        rows.append(cleaned_row(csv_row, output_label))
    out = ordered_frame(rows, [output_label]).drop_duplicates().reset_index(drop=True)
    out = drop_liquid_only_phase(out)
    return out


def write_rejected_rows(rejected_rows: list[dict[str, object]], rejected_path: Path) -> None:
    if rejected_rows:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rejected_rows).to_csv(rejected_path, index=False)
    elif rejected_path.exists():
        rejected_path.unlink()


def default_rejected_path(output_path: Path) -> Path:
    return (
        output_path.parent.parent
        / "rejected_rows"
        / output_path.parent.name
        / f"{output_path.stem}_rejected.csv"
    )


def structure_cleaned_ilthermo_file(
    input_path: Path,
    output_path: Path,
    property_slug: str,
    rejected_path: Path | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    rejected_rows: list[dict[str, object]] = []
    try:
        out = structure_cleaned_ilthermo_frame(df, property_slug, rejected_rows=rejected_rows)
    except ValueError as exc:
        if str(exc) == "missing required label column":
            raise ValueError(f"{input_path} missing required label column") from exc
        raise
    write_rejected_rows(rejected_rows, Path(rejected_path) if rejected_path else default_rejected_path(output_path))
    return write_frame(out, output_path)


def structure_relative_permittivity_files(input_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_path)
    rejected_rows: list[dict[str, object]] = []
    if "frequency_MHz" in df.columns:
        frequencies = pd.to_numeric(df["frequency_MHz"], errors="coerce")
        static_mask = frequencies.isna() | frequencies.eq(0)
    else:
        static_mask = pd.Series(True, index=df.index)

    static_input = df.loc[static_mask].drop(columns=["frequency_MHz"], errors="ignore")
    dynamic_input = df.loc[~static_mask].copy()
    static = structure_cleaned_ilthermo_frame(
        static_input,
        "relative_permittivity",
        STATIC_RELATIVE_PERMITTIVITY_LABEL,
        rejected_rows,
    )
    dynamic = structure_cleaned_ilthermo_frame(
        dynamic_input,
        "relative_permittivity",
        DYNAMIC_RELATIVE_PERMITTIVITY_LABEL,
        rejected_rows,
    )
    write_frame(static, output_dir / STATIC_RELATIVE_PERMITTIVITY_OUTPUT)
    write_frame(dynamic, output_dir / DYNAMIC_RELATIVE_PERMITTIVITY_OUTPUT)
    rejected_path = default_rejected_path(output_dir / "ilt_relative_permittivity_structured.csv")
    write_rejected_rows(rejected_rows, rejected_path)


def structure_cleaned_ilthermo(input_dir: Path, output_dir: Path) -> None:
    for slug, spec in ILTHERMO_SPECS.items():
        input_path = input_dir / spec.output_name
        if not input_path.exists():
            continue
        if slug == "relative_permittivity":
            structure_relative_permittivity_files(input_path, output_dir)
            (output_dir / spec.output_name).unlink(missing_ok=True)
            continue
        structure_cleaned_ilthermo_file(input_path, output_dir / spec.output_name, slug)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structure manually cleaned ILThermo CSVs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structure_cleaned_ilthermo(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
