"""Convert manually cleaned ILThermo CSVs to the current structured format."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.structure_raw_data import (  # noqa: E402
    CONDITION_COLUMNS,
    ILTHERMO_SPECS,
    SYSTEM_COLUMNS,
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


def cleaned_row(row: pd.Series, label_column: str) -> dict[str, object]:
    output: dict[str, object] = {}
    for column in [*SYSTEM_COLUMNS, *CONDITION_COLUMNS]:
        if column in row.index and not pd.isna(row[column]):
            output[column] = row[column]

    phase = extract_phase(row)
    if phase:
        output["phase"] = phase

    output[label_column] = row["label"]
    return output


def structure_cleaned_ilthermo_file(input_path: Path, output_path: Path, property_slug: str) -> pd.DataFrame:
    spec = ILTHERMO_SPECS[property_slug]
    df = pd.read_csv(input_path)
    if "label" not in df.columns:
        raise ValueError(f"{input_path} missing required label column")

    rows = [cleaned_row(csv_row, spec.label_column) for _, csv_row in df.iterrows()]
    out = ordered_frame(rows, [spec.label_column]).drop_duplicates().reset_index(drop=True)
    return write_frame(out, output_path)


def structure_cleaned_ilthermo(input_dir: Path, output_dir: Path) -> None:
    for slug, spec in ILTHERMO_SPECS.items():
        input_path = input_dir / spec.output_name
        if not input_path.exists():
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
