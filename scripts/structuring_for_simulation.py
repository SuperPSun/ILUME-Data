"""Structure raw simulation datasets into ILThermo-style wide CSV outputs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raw_prep import canonicalize_smiles, disable_rdkit_logs, raw_root, structured_root


STANDARD_COLUMNS = [
    "cation",
    "anion",
    "temperature_K",
    "pressure_kPa",
    "frequency_MHz",
    "wavelength_nm",
    "property_name",
    "property_unit",
    "property_value",
    "label",
    "standard_unit",
    "parse_error",
    "note",
    "source_text",
]


def to_float(value: object) -> float | None:
    """Convert a value to float, returning None when conversion is impossible."""

    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


@lru_cache(maxsize=None)
def _canonicalize_cached(smiles: str) -> str:
    """Canonicalize one SMILES string with a cache for repeated mapping rows."""

    return canonicalize_smiles(smiles)


def clean_smiles(value: object) -> str:
    """Trim and canonicalize one SMILES string."""

    return _canonicalize_cached(str(value or "").strip())


def write_output(rows: list[dict[str, object]], output_path: Path, extra_columns: list[str] | None = None) -> None:
    """Write a structured CSV with standard columns first and source-specific columns after."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = STANDARD_COLUMNS + list(extra_columns or [])
    pd.DataFrame(rows, columns=columns).drop_duplicates().to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")


def process_density_260501(raw_dir: Path, output_dir: Path) -> None:
    """Structure density_260501.csv."""

    input_path = raw_dir / "density_260501.csv"
    output_path = output_dir / "simulated_density_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        cation = clean_smiles(row.get("cation"))
        anion = clean_smiles(row.get("anion"))
        temperature = to_float(row.get("temperature"))
        property_value = to_float(row.get("density"))
        label = property_value * 0.001 if property_value is not None else None
        rows.append(
            {
                "cation": cation,
                "anion": anion,
                "temperature_K": temperature,
                "pressure_kPa": None,
                "frequency_MHz": None,
                "wavelength_nm": None,
                "property_name": "Specific density",
                "property_unit": "kg/m^3",
                "property_value": property_value,
                "label": label,
                "standard_unit": "g/cm^3",
                "parse_error": "",
                "note": "",
                "source_text": row.to_json(force_ascii=False),
                "density_err": to_float(row.get("density_err")),
            }
        )
    write_output(rows, output_path, extra_columns=["density_err"])


def process_heat_capacity_260501(raw_dir: Path, output_dir: Path) -> None:
    """Structure heat_capacity_260501.csv."""

    input_path = raw_dir / "heat_capacity_260501.csv"
    output_path = output_dir / "simulated_heat_capacity_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        cation = clean_smiles(row.get("cation"))
        anion = clean_smiles(row.get("anion"))
        temperature = to_float(row.get("temperature"))
        property_value = to_float(row.get("Cp"))
        rows.append(
            {
                "cation": cation,
                "anion": anion,
                "temperature_K": temperature,
                "pressure_kPa": None,
                "frequency_MHz": None,
                "wavelength_nm": None,
                "property_name": "Heat capacity at constant pressure",
                "property_unit": "J/mol/K",
                "property_value": property_value,
                "label": property_value,
                "standard_unit": "J/mol/K",
                "parse_error": "",
                "note": "",
                "source_text": row.to_json(force_ascii=False),
                "Cp_err": to_float(row.get("Cp_err")),
            }
        )
    write_output(rows, output_path, extra_columns=["Cp_err"])


def process_thermal_expansion_260501(raw_dir: Path, output_dir: Path) -> None:
    """Structure thermal_expansion_260501.csv."""

    input_path = raw_dir / "thermal_expansion_260501.csv"
    output_path = output_dir / "simulated_thermal_expansion_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        cation = clean_smiles(row.get("cation"))
        anion = clean_smiles(row.get("anion"))
        temperature = to_float(row.get("temperature"))
        property_value = to_float(row.get("alpha"))
        rows.append(
            {
                "cation": cation,
                "anion": anion,
                "temperature_K": temperature,
                "pressure_kPa": None,
                "frequency_MHz": None,
                "wavelength_nm": None,
                "property_name": "Isobaric coefficient of volume expansion",
                "property_unit": "K^-1",
                "property_value": property_value,
                "label": property_value,
                "standard_unit": "K^-1",
                "parse_error": "",
                "note": "",
                "source_text": row.to_json(force_ascii=False),
                "alpha_err": to_float(row.get("alpha_err")),
            }
        )
    write_output(rows, output_path, extra_columns=["alpha_err"])


def process_heat_of_vaporization_260603(raw_dir: Path, output_dir: Path) -> None:
    """Structure heat_of_vaporization_260603.csv."""

    input_path = raw_dir / "heat_of_vaporization_260603.csv"
    output_path = output_dir / "simulated_heat_of_vaporization_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        cation = clean_smiles(row.get("cation"))
        anion = clean_smiles(row.get("anion"))
        temperature = to_float(row.get("temperature"))
        property_value = to_float(row.get("Hvap"))
        rows.append(
            {
                "cation": cation,
                "anion": anion,
                "temperature_K": temperature,
                "pressure_kPa": None,
                "frequency_MHz": None,
                "wavelength_nm": None,
                "property_name": "Enthalpy of vaporization or sublimation",
                "property_unit": "kJ/mol",
                "property_value": property_value,
                "label": property_value,
                "standard_unit": "kJ/mol",
                "parse_error": "",
                "note": "",
                "source_text": row.to_json(force_ascii=False),
                "Hvap_err": to_float(row.get("Hvap_err")),
            }
        )
    write_output(rows, output_path, extra_columns=["Hvap_err"])


def process_pbe_tzvp_anions_260103(raw_dir: Path, output_dir: Path) -> None:
    """Structure PBE_TZVP_anions_260103.csv as one wide row per anion."""

    input_path = raw_dir / "PBE_TZVP_anions_260103.csv"
    output_path = output_dir / "simulated_PBE_TZVP_anions_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        anion = clean_smiles(row.get("SMILES"))
        gap = to_float(row.get("gap"))
        rows.append(
            {
                "cation": "",
                "anion": anion,
                "temperature_K": None,
                "pressure_kPa": None,
                "frequency_MHz": None,
                "wavelength_nm": None,
                "property_name": "PBE_TZVP_gap",
                "property_unit": "eV",
                "property_value": gap,
                "label": gap,
                "standard_unit": "eV",
                "parse_error": "",
                "note": "",
                "source_text": row.to_json(force_ascii=False),
                "SMILES": anion,
                "HOMO": to_float(row.get("HOMO")),
                "LUMO": to_float(row.get("LUMO")),
                "gap": gap,
            }
        )
    write_output(rows, output_path, extra_columns=["SMILES", "HOMO", "LUMO", "gap"])


def process_pbe_tzvp_cations_260103(raw_dir: Path, output_dir: Path) -> None:
    """Structure PBE_TZVP_cations_260103.csv as one wide row per cation."""

    input_path = raw_dir / "PBE_TZVP_cations_260103.csv"
    output_path = output_dir / "simulated_PBE_TZVP_cations_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        cation = clean_smiles(row.get("SMILES"))
        gap = to_float(row.get("gap"))
        rows.append(
            {
                "cation": cation,
                "anion": "",
                "temperature_K": None,
                "pressure_kPa": None,
                "frequency_MHz": None,
                "wavelength_nm": None,
                "property_name": "PBE_TZVP_gap",
                "property_unit": "eV",
                "property_value": gap,
                "label": gap,
                "standard_unit": "eV",
                "parse_error": "",
                "note": "",
                "source_text": row.to_json(force_ascii=False),
                "SMILES": cation,
                "HOMO": to_float(row.get("HOMO")),
                "LUMO": to_float(row.get("LUMO")),
                "gap": gap,
            }
        )
    write_output(rows, output_path, extra_columns=["SMILES", "HOMO", "LUMO", "gap"])


def process_hl_gap_pbe_tzvp(raw_dir: Path, output_dir: Path) -> None:
    """Structure HL_gap_PBE_TZVP.csv as one wide row per molecule."""

    input_path = raw_dir / "HL_gap_PBE_TZVP.csv"
    output_path = output_dir / "simulated_HL_gap_PBE_TZVP_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        smiles = clean_smiles(row.get("SMILES"))
        gap = to_float(row.get("gap"))
        rows.append(
            {
                "cation": "",
                "anion": "",
                "temperature_K": None,
                "pressure_kPa": None,
                "frequency_MHz": None,
                "wavelength_nm": None,
                "property_name": "PBE_TZVP_gap",
                "property_unit": "eV",
                "property_value": gap,
                "label": gap,
                "standard_unit": "eV",
                "parse_error": "",
                "note": "",
                "source_text": row.to_json(force_ascii=False),
                "SMILES": smiles,
                "HOMO": to_float(row.get("HOMO")),
                "LUMO": to_float(row.get("LUMO")),
                "gap": gap,
            }
        )
    write_output(rows, output_path, extra_columns=["SMILES", "HOMO", "LUMO", "gap"])


def process_qm_elec_hf(raw_dir: Path, output_dir: Path) -> None:
    """Structure QM_elec_HF.csv as one wide row per molecule while retaining all raw property columns."""

    input_path = raw_dir / "QM_elec_HF.csv"
    output_path = output_dir / "simulated_QM_elec_HF_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    raw_property_columns = [col for col in df.columns if col != "SMILES"]
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        smiles = clean_smiles(row.get("SMILES"))
        gap = to_float(row.get("Gap"))
        structured_row: dict[str, object] = {
            "cation": "",
            "anion": "",
            "temperature_K": None,
            "pressure_kPa": None,
            "frequency_MHz": None,
            "wavelength_nm": None,
            "property_name": "HF_Gap",
            "property_unit": "eV",
            "property_value": gap,
            "label": gap,
            "standard_unit": "eV",
            "parse_error": "",
            "note": "",
            "source_text": row.to_json(force_ascii=False),
            "SMILES": smiles,
        }
        for col in raw_property_columns:
            structured_row[col] = to_float(row.get(col))
        rows.append(structured_row)
    write_output(rows, output_path, extra_columns=["SMILES", *raw_property_columns])


def process_combi_qm_solv(raw_dir: Path, output_dir: Path) -> None:
    """Structure combi_qm_solv.csv."""

    input_path = raw_dir / "combi_qm_solv.csv"
    output_path = output_dir / "simulated_combi_qm_solv_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        solvent = clean_smiles(row.get("solvent"))
        solute = clean_smiles(row.get("solute"))
        solv = to_float(row.get("solv"))
        rows.append(
            {
                "cation": "",
                "anion": "",
                "temperature_K": None,
                "pressure_kPa": None,
                "frequency_MHz": None,
                "wavelength_nm": None,
                "property_name": "solvation",
                "property_unit": "",
                "property_value": solv,
                "label": solv,
                "standard_unit": "",
                "parse_error": "",
                "note": "",
                "source_text": row.to_json(force_ascii=False),
                "solvent": solvent,
                "solute": solute,
            }
        )
    write_output(rows, output_path, extra_columns=["solvent", "solute"])


def process_box_mapping_20260514(raw_dir: Path, output_dir: Path) -> None:
    """Structure box_20260514/mapping.csv and ignore sibling PDB files."""

    input_path = raw_dir / "box_20260514" / "mapping.csv"
    output_path = output_dir / "simulated_box_20260514_mapping_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        cation = clean_smiles(row.get("cation_smiles"))
        anion = clean_smiles(row.get("anion_smiles"))
        rows.append(
            {
                "cation": cation,
                "anion": anion,
                "temperature_K": to_float(row.get("temperature")),
                "pressure_kPa": None,
                "frequency_MHz": None,
                "wavelength_nm": None,
                "property_name": "simulation_box_mapping",
                "property_unit": "",
                "property_value": None,
                "label": None,
                "standard_unit": "",
                "parse_error": "",
                "note": "",
                "source_text": row.to_json(force_ascii=False),
                "mol_id": row.get("mol_id", ""),
            }
        )
    write_output(rows, output_path, extra_columns=["mol_id"])


def process_charge_mapping_20260514(raw_dir: Path, output_dir: Path) -> None:
    """Structure charge_20260514/mapping.csv and ignore sibling MOL2 files."""

    input_path = raw_dir / "charge_20260514" / "mapping.csv"
    output_path = output_dir / "simulated_charge_20260514_mapping_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        smiles = clean_smiles(row.get("smiles"))
        charge = to_float(row.get("charge"))
        rows.append(
            {
                "cation": "",
                "anion": "",
                "temperature_K": None,
                "pressure_kPa": None,
                "frequency_MHz": None,
                "wavelength_nm": None,
                "property_name": "charge",
                "property_unit": "",
                "property_value": charge,
                "label": charge,
                "standard_unit": "",
                "parse_error": "",
                "note": "",
                "source_text": row.to_json(force_ascii=False),
                "mol_id": row.get("mol_id", ""),
                "SMILES": smiles,
                "charge": charge,
            }
        )
    write_output(rows, output_path, extra_columns=["mol_id", "SMILES", "charge"])


PROCESSORS: dict[str, Callable[[Path, Path], None]] = {
    "density_260501.csv": process_density_260501,
    "heat_capacity_260501.csv": process_heat_capacity_260501,
    "thermal_expansion_260501.csv": process_thermal_expansion_260501,
    "heat_of_vaporization_260603.csv": process_heat_of_vaporization_260603,
    "PBE_TZVP_anions_260103.csv": process_pbe_tzvp_anions_260103,
    "PBE_TZVP_cations_260103.csv": process_pbe_tzvp_cations_260103,
    "HL_gap_PBE_TZVP.csv": process_hl_gap_pbe_tzvp,
    "QM_elec_HF.csv": process_qm_elec_hf,
    "combi_qm_solv.csv": process_combi_qm_solv,
    "box_20260514/mapping.csv": process_box_mapping_20260514,
    "charge_20260514/mapping.csv": process_charge_mapping_20260514,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the simulation-data structuring entrypoint."""

    parser = argparse.ArgumentParser(
        description="Clean and structure raw simulation CSV files into ILThermo-style wide outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=raw_root(PROJECT_ROOT) / "simulation_data",
        help="Path to the directory that contains the raw simulation CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=structured_root(PROJECT_ROOT) / "simulation",
        help="Path to the directory where the cleaned simulation CSV files will be written.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=list(PROCESSORS),
        help="One or more supported simulation CSV filenames to process from the input directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the fixed set of simulation cleaning jobs for this repository."""

    disable_rdkit_logs()
    args = parse_args()
    for name in args.files:
        processor = PROCESSORS.get(name)
        if processor is None:
            print(f"Skipped unsupported simulation file: {name}")
            continue
        processor(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
