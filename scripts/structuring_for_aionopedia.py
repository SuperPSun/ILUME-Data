"""Structure AIonopedia raw CSV files into ILThermo-style wide CSV outputs."""

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

PROMPT_KEYS = [
    ("cation", r"cation\s*\[START_SMILES\](?P<cation>.*?)\[END_SMILES\]"),
    ("anion", r"anion\s*\[START_SMILES\](?P<anion>.*?)\[END_SMILES\]"),
    ("solute", r"solute\s*\[START_SMILES\](?P<solute>.*?)\[END_SMILES\]"),
    ("solvent", r"solvent\s*\[START_SMILES\](?P<solvent>.*?)\[END_SMILES\]"),
    ("temperature", r"temperature\s*\$?(?P<temperature>[0-9]+(?:\.[0-9]+)?)\$?K"),
]


def parse_prompt(text: str) -> dict[str, str]:
    """Extract structured fields from a prompt-style AIonopedia row."""

    data = {key: "" for key, _ in PROMPT_KEYS}
    normalized = str(text).replace("\n", " ").replace("\r", " ")
    for key, pattern in PROMPT_KEYS:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            data[key] = match.group(key).strip()
    return data


def canonicalize_fields(row: dict[str, str], invalid_sink: set[str] | None = None) -> dict[str, str]:
    """Canonicalize every chemical field present in a parsed AIonopedia row."""

    for key in ("cation", "anion", "solute", "solvent"):
        if row.get(key):
            row[key] = canonicalize_smiles(row[key], invalid_sink)
    return row


def to_float(value: object) -> float | None:
    """Convert a value to float, returning None when conversion is impossible."""

    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def prompt_fieldname(fieldnames: list[str] | None) -> str:
    """Resolve the prompt column name used by one raw AIonopedia CSV."""

    fields = set(fieldnames or [])
    if "Prompt" in fields:
        return "Prompt"
    if "prompt" in fields:
        return "prompt"
    raise RuntimeError("missing Prompt/prompt column")


def build_wide_row(
    *,
    parsed: dict[str, str],
    label: object,
    property_name: str,
    property_unit: str,
    standard_unit: str,
    source_text: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one ILThermo-style wide row."""

    label_value = to_float(label)
    row: dict[str, object] = {
        "cation": parsed.get("cation", ""),
        "anion": parsed.get("anion", ""),
        "temperature_K": to_float(parsed.get("temperature")),
        "pressure_kPa": None,
        "frequency_MHz": None,
        "wavelength_nm": None,
        "property_name": property_name,
        "property_unit": property_unit,
        "property_value": label_value,
        "label": label_value,
        "standard_unit": standard_unit,
        "parse_error": "",
        "note": "",
        "source_text": source_text,
    }
    if extra:
        row.update(extra)
    return row


def write_output(rows: list[dict[str, object]], output_path: Path, extra_columns: list[str] | None = None) -> None:
    """Write a structured CSV with standard columns first and source-specific columns after."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = STANDARD_COLUMNS + list(extra_columns or [])
    pd.DataFrame(rows, columns=columns).drop_duplicates().to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")


def process_density_all(raw_dir: Path, output_dir: Path, invalid_smiles: set[str]) -> None:
    """Structure density_all.csv."""

    input_path = raw_dir / "density_all.csv"
    output_path = output_dir / "AIonopedia_density_constructed.csv"
    print(f"Processing {input_path} -> {output_path}")
    rows: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        prompt_col = prompt_fieldname(reader.fieldnames)
        for csv_row in reader:
            prompt = csv_row.get(prompt_col, "")
            parsed = canonicalize_fields(parse_prompt(prompt), invalid_smiles)
            rows.append(
                build_wide_row(
                    parsed=parsed,
                    label=csv_row.get("label", ""),
                    property_name="Specific density",
                    property_unit="g/cm^3",
                    standard_unit="g/cm^3",
                    source_text=prompt,
                )
            )
    write_output(rows, output_path)


def process_melt_all(raw_dir: Path, output_dir: Path, invalid_smiles: set[str]) -> None:
    """Structure melt_all.csv."""

    input_path = raw_dir / "melt_all.csv"
    output_path = output_dir / "AIonopedia_melt_constructed.csv"
    print(f"Processing {input_path} -> {output_path}")
    rows: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        prompt_col = prompt_fieldname(reader.fieldnames)
        for csv_row in reader:
            prompt = csv_row.get(prompt_col, "")
            parsed = canonicalize_fields(parse_prompt(prompt), invalid_smiles)
            rows.append(
                build_wide_row(
                    parsed=parsed,
                    label=csv_row.get("label", ""),
                    property_name="Normal melting temperature",
                    property_unit="K",
                    standard_unit="K",
                    source_text=prompt,
                )
            )
    write_output(rows, output_path)


def process_tension_all(raw_dir: Path, output_dir: Path, invalid_smiles: set[str]) -> None:
    """Structure tension_all.csv."""

    input_path = raw_dir / "tension_all.csv"
    output_path = output_dir / "AIonopedia_tension_constructed.csv"
    print(f"Processing {input_path} -> {output_path}")
    rows: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        prompt_col = prompt_fieldname(reader.fieldnames)
        for csv_row in reader:
            prompt = csv_row.get(prompt_col, "")
            parsed = canonicalize_fields(parse_prompt(prompt), invalid_smiles)
            rows.append(
                build_wide_row(
                    parsed=parsed,
                    label=csv_row.get("label", ""),
                    property_name="Surface tension liquid-gas",
                    property_unit="mN/m",
                    standard_unit="mN/m",
                    source_text=prompt,
                )
            )
    write_output(rows, output_path)


def process_viscosity_all(raw_dir: Path, output_dir: Path, invalid_smiles: set[str]) -> None:
    """Structure viscosity_all.csv."""

    input_path = raw_dir / "viscosity_all.csv"
    output_path = output_dir / "AIonopedia_viscosity_constructed.csv"
    print(f"Processing {input_path} -> {output_path}")
    rows: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        prompt_col = prompt_fieldname(reader.fieldnames)
        for csv_row in reader:
            prompt = csv_row.get(prompt_col, "")
            parsed = canonicalize_fields(parse_prompt(prompt), invalid_smiles)
            rows.append(
                build_wide_row(
                    parsed=parsed,
                    label=csv_row.get("label", ""),
                    property_name="Viscosity",
                    property_unit="mPa*s (log scale)",
                    standard_unit="mPa*s (log scale)",
                    source_text=prompt,
                )
            )
    write_output(rows, output_path)


def process_solvation_all(raw_dir: Path, output_dir: Path, invalid_smiles: set[str]) -> None:
    """Structure solvation_all.csv."""

    input_path = raw_dir / "solvation_all.csv"
    output_path = output_dir / "AIonopedia_solvation_constructed.csv"
    print(f"Processing {input_path} -> {output_path}")
    rows: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        prompt_col = prompt_fieldname(reader.fieldnames)
        for csv_row in reader:
            prompt = csv_row.get(prompt_col, "")
            parsed = canonicalize_fields(parse_prompt(prompt), invalid_smiles)
            rows.append(
                build_wide_row(
                    parsed=parsed,
                    label=csv_row.get("label", ""),
                    property_name="solvation",
                    property_unit="",
                    standard_unit="",
                    source_text=prompt,
                    extra={"solute": parsed.get("solute", ""), "solvent": parsed.get("solvent", "")},
                )
            )
    write_output(rows, output_path, extra_columns=["solute", "solvent"])


def process_transfer_all(raw_dir: Path, output_dir: Path, invalid_smiles: set[str]) -> None:
    """Structure transfer_all.csv."""

    input_path = raw_dir / "transfer_all.csv"
    output_path = output_dir / "AIonopedia_transfer_constructed.csv"
    print(f"Processing {input_path} -> {output_path}")
    rows: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        prompt_col = prompt_fieldname(reader.fieldnames)
        for csv_row in reader:
            prompt = csv_row.get(prompt_col, "")
            parsed = canonicalize_fields(parse_prompt(prompt), invalid_smiles)
            rows.append(
                build_wide_row(
                    parsed=parsed,
                    label=csv_row.get("label", ""),
                    property_name="transfer",
                    property_unit="",
                    standard_unit="",
                    source_text=prompt,
                    extra={"solute": parsed.get("solute", ""), "solvent": parsed.get("solvent", "")},
                )
            )
    write_output(rows, output_path, extra_columns=["solute", "solvent"])


def process_transfer_organic_all(raw_dir: Path, output_dir: Path, invalid_smiles: set[str]) -> None:
    """Structure transfer_organic_all.csv."""

    input_path = raw_dir / "transfer_organic_all.csv"
    output_path = output_dir / "AIonopedia_transfer_organic_constructed.csv"
    print(f"Processing {input_path} -> {output_path}")
    rows: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        prompt_col = prompt_fieldname(reader.fieldnames)
        for csv_row in reader:
            prompt = csv_row.get(prompt_col, "")
            parsed = canonicalize_fields(parse_prompt(prompt), invalid_smiles)
            rows.append(
                build_wide_row(
                    parsed=parsed,
                    label=csv_row.get("label", ""),
                    property_name="transfer_organic",
                    property_unit="",
                    standard_unit="",
                    source_text=prompt,
                    extra={"solute": parsed.get("solute", ""), "solvent": parsed.get("solvent", "")},
                )
            )
    write_output(rows, output_path, extra_columns=["solute", "solvent"])


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
    """Parse CLI arguments for the AIonopedia-data structuring entrypoint."""

    parser = argparse.ArgumentParser(
        description="Clean and structure AIonopedia CSV files into ILThermo-style wide outputs."
    )
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
    """Process every requested AIonopedia CSV and log invalid SMILES values."""

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
