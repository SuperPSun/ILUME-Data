"""Structure AIonopedia raw CSV files into compact, dataset-native CSV outputs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raw_prep import canonicalize_smiles, disable_rdkit_logs, raw_root, structured_root


PROMPT_KEYS = [
    ("cation", r"cation\s*\[START_SMILES\](?P<cation>.*?)\[END_SMILES\]"),
    ("anion", r"anion\s*\[START_SMILES\](?P<anion>.*?)\[END_SMILES\]"),
    ("solute", r"solute\s*\[START_SMILES\](?P<solute>.*?)\[END_SMILES\]"),
    ("solvent", r"solvent\s*\[START_SMILES\](?P<solvent>.*?)\[END_SMILES\]"),
    ("temperature", r"temperature\s*\$?(?P<temperature>[0-9]+(?:\.[0-9]+)?)\$?K"),
]


def parse_prompt(text: str) -> dict[str, str]:
    data = {key: "" for key, _ in PROMPT_KEYS}
    normalized = str(text).replace("\n", " ").replace("\r", " ")
    for key, pattern in PROMPT_KEYS:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            data[key] = match.group(key).strip()
    return data


def canonicalize_fields(row: dict[str, str], invalid_sink: set[str] | None = None) -> dict[str, str]:
    for key in ("cation", "anion", "solute", "solvent"):
        if row.get(key):
            row[key] = canonicalize_smiles(row[key], invalid_sink)
    return row


def to_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def prompt_fieldname(fieldnames: list[str] | None) -> str:
    fields = set(fieldnames or [])
    if "Prompt" in fields:
        return "Prompt"
    if "prompt" in fields:
        return "prompt"
    raise RuntimeError("missing Prompt/prompt column")


def write_output(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")


def process_prompt_file(
    raw_dir: Path,
    output_dir: Path,
    invalid_smiles: set[str],
    *,
    input_name: str,
    output_name: str,
    value_column: str,
    kind: str = "il",
) -> None:
    input_path = raw_dir / input_name
    output_path = output_dir / output_name
    print(f"Processing {input_path} -> {output_path}")
    rows: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        prompt_col = prompt_fieldname(reader.fieldnames)
        for csv_row in reader:
            parsed = canonicalize_fields(parse_prompt(csv_row.get(prompt_col, "")), invalid_smiles)
            value = to_float(csv_row.get("label", ""))
            if kind == "solvation":
                rows.append(
                    {
                        "solute": parsed.get("solute", ""),
                        "solvent": parsed.get("solvent", ""),
                        value_column: value,
                    }
                )
            else:
                rows.append(
                    {
                        "cation": parsed.get("cation", ""),
                        "anion": parsed.get("anion", ""),
                        "temperature_K": to_float(parsed.get("temperature")),
                        value_column: value,
                    }
                )
    write_output(rows, output_path)


def process_density_all(raw_dir: Path, output_dir: Path, invalid_smiles: set[str]) -> None:
    process_prompt_file(
        raw_dir,
        output_dir,
        invalid_smiles,
        input_name="density_all.csv",
        output_name="AIonopedia_density_constructed.csv",
        value_column="density",
    )


def process_melt_all(raw_dir: Path, output_dir: Path, invalid_smiles: set[str]) -> None:
    process_prompt_file(
        raw_dir,
        output_dir,
        invalid_smiles,
        input_name="melt_all.csv",
        output_name="AIonopedia_melt_constructed.csv",
        value_column="melt",
    )


def process_tension_all(raw_dir: Path, output_dir: Path, invalid_smiles: set[str]) -> None:
    process_prompt_file(
        raw_dir,
        output_dir,
        invalid_smiles,
        input_name="tension_all.csv",
        output_name="AIonopedia_tension_constructed.csv",
        value_column="tension",
    )


def process_viscosity_all(raw_dir: Path, output_dir: Path, invalid_smiles: set[str]) -> None:
    process_prompt_file(
        raw_dir,
        output_dir,
        invalid_smiles,
        input_name="viscosity_all.csv",
        output_name="AIonopedia_viscosity_constructed.csv",
        value_column="viscosity",
    )


def process_solvation_all(raw_dir: Path, output_dir: Path, invalid_smiles: set[str]) -> None:
    process_prompt_file(
        raw_dir,
        output_dir,
        invalid_smiles,
        input_name="solvation_all.csv",
        output_name="AIonopedia_solvation_constructed.csv",
        value_column="solvation",
        kind="solvation",
    )


def process_transfer_all(raw_dir: Path, output_dir: Path, invalid_smiles: set[str]) -> None:
    process_prompt_file(
        raw_dir,
        output_dir,
        invalid_smiles,
        input_name="transfer_all.csv",
        output_name="AIonopedia_transfer_constructed.csv",
        value_column="transfer",
        kind="solvation",
    )


def process_transfer_organic_all(raw_dir: Path, output_dir: Path, invalid_smiles: set[str]) -> None:
    process_prompt_file(
        raw_dir,
        output_dir,
        invalid_smiles,
        input_name="transfer_organic_all.csv",
        output_name="AIonopedia_transfer_organic_constructed.csv",
        value_column="transfer_organic",
        kind="solvation",
    )


PROCESSORS: dict[str, Callable[[Path, Path, set[str]], None]] = {
    "density_all.csv": process_density_all,
    "melt_all.csv": process_melt_all,
    "tension_all.csv": process_tension_all,
    "viscosity_all.csv": process_viscosity_all,
    "solvation_all.csv": process_solvation_all,
    "transfer_all.csv": process_transfer_all,
    "transfer_organic_all.csv": process_transfer_organic_all,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and structure AIonopedia CSV files into dataset-native outputs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=raw_root(PROJECT_ROOT) / "AIonopedia",
        help="Path to the directory that contains the raw AIonopedia CSV files to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=structured_root(PROJECT_ROOT) / "AIonopedia",
        help="Path to the directory where the structured AIonopedia CSV files will be written.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=list(PROCESSORS),
        help="One or more raw AIonopedia CSV filenames to process from the input directory.",
    )
    parser.add_argument(
        "--invalid-smiles-log",
        type=Path,
        default=structured_root(PROJECT_ROOT) / "AIonopedia" / "invalid_smiles_AIonopedia.csv",
        help="Path to the CSV file that records SMILES strings that RDKit could not canonicalize.",
    )
    return parser.parse_args()


def main() -> None:
    disable_rdkit_logs()
    args = parse_args()
    invalid_smiles: set[str] = set()

    for name in args.files:
        processor = PROCESSORS.get(name)
        if processor is None:
            print(f"Skipped unsupported AIonopedia file: {name}")
            continue
        processor(args.input_dir, args.output_dir, invalid_smiles)

    if invalid_smiles:
        args.invalid_smiles_log.parent.mkdir(parents=True, exist_ok=True)
        with args.invalid_smiles_log.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["smiles"])
            for smiles in sorted(invalid_smiles):
                writer.writerow([smiles])
        print(f"Logged {len(invalid_smiles)} invalid SMILES to {args.invalid_smiles_log}")


if __name__ == "__main__":
    main()
