"""Structure raw simulation datasets into compact, dataset-native CSV outputs."""

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


def to_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


@lru_cache(maxsize=None)
def clean_smiles(value: object) -> str:
    return canonicalize_smiles(str(value or "").strip())


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")


def canonicalize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = df[column].map(clean_smiles)
    return df


def numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = df[column].map(to_float)
    return df


def process_density_260501(raw_dir: Path, output_dir: Path) -> None:
    input_path = raw_dir / "density_260501.csv"
    output_path = output_dir / "simulated_density_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["cation", "anion"])
    df = numeric_columns(df, ["temperature", "density", "density_err"])
    df = df.rename(columns={"temperature": "temperature_K"})
    write_output(df[["cation", "anion", "temperature_K", "density", "density_err"]], output_path)


def process_heat_capacity_260501(raw_dir: Path, output_dir: Path) -> None:
    input_path = raw_dir / "heat_capacity_260501.csv"
    output_path = output_dir / "simulated_heat_capacity_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["cation", "anion"])
    df = numeric_columns(df, ["temperature", "Cp", "Cp_err"])
    df = df.rename(columns={"temperature": "temperature_K"})
    write_output(df[["cation", "anion", "temperature_K", "Cp", "Cp_err"]], output_path)


def process_thermal_expansion_260501(raw_dir: Path, output_dir: Path) -> None:
    input_path = raw_dir / "thermal_expansion_260501.csv"
    output_path = output_dir / "simulated_thermal_expansion_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["cation", "anion"])
    df = numeric_columns(df, ["temperature", "alpha", "alpha_err"])
    df = df.rename(columns={"temperature": "temperature_K"})
    write_output(df[["cation", "anion", "temperature_K", "alpha", "alpha_err"]], output_path)


def process_heat_of_vaporization_260603(raw_dir: Path, output_dir: Path) -> None:
    input_path = raw_dir / "heat_of_vaporization_260603.csv"
    output_path = output_dir / "simulated_heat_of_vaporization_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["cation", "anion"])
    df = numeric_columns(df, ["temperature", "Hvap", "Hvap_err"])
    df = df.rename(columns={"temperature": "temperature_K"})
    write_output(df[["cation", "anion", "temperature_K", "Hvap", "Hvap_err"]], output_path)


def process_pbe_tzvp_anions_260103(raw_dir: Path, output_dir: Path) -> None:
    input_path = raw_dir / "PBE_TZVP_anions_260103.csv"
    output_path = output_dir / "simulated_PBE_TZVP_anions_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["SMILES"])
    df = numeric_columns(df, ["HOMO", "LUMO", "gap"])
    df["anion"] = df["SMILES"]
    write_output(df[["SMILES", "anion", "HOMO", "LUMO", "gap"]], output_path)


def process_pbe_tzvp_cations_260103(raw_dir: Path, output_dir: Path) -> None:
    input_path = raw_dir / "PBE_TZVP_cations_260103.csv"
    output_path = output_dir / "simulated_PBE_TZVP_cations_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["SMILES"])
    df = numeric_columns(df, ["HOMO", "LUMO", "gap"])
    df["cation"] = df["SMILES"]
    write_output(df[["SMILES", "cation", "HOMO", "LUMO", "gap"]], output_path)


"""def process_hl_gap_pbe_tzvp(raw_dir: Path, output_dir: Path) -> None:
    input_path = raw_dir / "HL_gap_PBE_TZVP.csv"
    output_path = output_dir / "simulated_HL_gap_PBE_TZVP_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["SMILES"])
    df = numeric_columns(df, ["HOMO", "LUMO", "gap"])
    write_output(df[["SMILES", "HOMO", "LUMO", "gap"]], output_path)
"""

def process_qm_elec_hf(raw_dir: Path, output_dir: Path) -> None:
    input_path = raw_dir / "QM_elec_HF.csv"
    output_path = output_dir / "simulated_QM_elec_HF_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["SMILES"])
    numeric = [column for column in df.columns if column != "SMILES"]
    df = numeric_columns(df, numeric)
    write_output(df[["SMILES", *numeric]], output_path)


def process_combi_qm_solv(raw_dir: Path, output_dir: Path) -> None:
    input_path = raw_dir / "combi_qm_solv.csv"
    output_path = output_dir / "simulated_combi_qm_solv_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["solvent", "solute"])
    df = numeric_columns(df, ["solv"])
    write_output(df[["solvent", "solute", "solv"]], output_path)


def process_box_mapping_20260514(raw_dir: Path, output_dir: Path) -> None:
    input_path = raw_dir / "box_20260514" / "mapping.csv"
    output_path = output_dir / "simulated_box_20260514_mapping_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    df = df.rename(columns={"cation_smiles": "cation", "anion_smiles": "anion", "temperature": "temperature_K"})
    df = canonicalize_columns(df, ["cation", "anion"])
    df = numeric_columns(df, ["temperature_K"])
    write_output(df[["mol_id", "cation", "anion", "temperature_K"]], output_path)


def process_charge_mapping_20260514(raw_dir: Path, output_dir: Path) -> None:
    input_path = raw_dir / "charge_20260514" / "mapping.csv"
    output_path = output_dir / "simulated_charge_20260514_mapping_structured.csv"
    print(f"Processing {input_path} -> {output_path}")
    df = pd.read_csv(input_path)
    df = df.rename(columns={"smiles": "SMILES"})
    df = canonicalize_columns(df, ["SMILES"])
    df = numeric_columns(df, ["charge"])
    write_output(df[["mol_id", "SMILES", "charge"]], output_path)


PROCESSORS: dict[str, Callable[[Path, Path], None]] = {
    "density_260501.csv": process_density_260501,
    "heat_capacity_260501.csv": process_heat_capacity_260501,
    "thermal_expansion_260501.csv": process_thermal_expansion_260501,
    "heat_of_vaporization_260603.csv": process_heat_of_vaporization_260603,
    "PBE_TZVP_anions_260103.csv": process_pbe_tzvp_anions_260103,
    "PBE_TZVP_cations_260103.csv": process_pbe_tzvp_cations_260103,
    #"HL_gap_PBE_TZVP.csv": process_hl_gap_pbe_tzvp,
    "QM_elec_HF.csv": process_qm_elec_hf,
    "combi_qm_solv.csv": process_combi_qm_solv,
    "box_20260514/mapping.csv": process_box_mapping_20260514,
    "charge_20260514/mapping.csv": process_charge_mapping_20260514,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and structure raw simulation CSV files into dataset-native outputs."
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
