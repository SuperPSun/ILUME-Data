"""Crawl pure-compound ILThermo energetics records with ILThermoPy.

This script fetches ILThermo energetics entries through ILThermoPy and writes a
streamlined CSV for pure-compound energetics records.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from html import unescape
from pathlib import Path
from typing import Any

import ilthermopy as ilt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raw_prep import canonicalize_smiles, disable_rdkit_logs, raw_root

PROPERTY_NAMES = {
    "enthalpy": "Enthalpy",
    "entropy": "Entropy",
    "enthalpy_of_transition_or_fusion": "Enthalpy of transition or fusion",
    "enthalpy_of_vaporization_or_sublimation": "Enthalpy of vaporization or sublimation",
}

LIQUID_ONLY_PROPERTIES = {"enthalpy", "entropy"}

TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
PHASE_SPLIT_RE = re.compile(r"[;,/]")
TRAILING_PHASE_INDEX_RE = re.compile(r"\s+\d+$")


def html_to_text(value: str | None) -> str:
    """Strip HTML markup and normalize whitespace into plain text."""

    text = unescape(value or "")
    text = TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def parse_float(value: str | None) -> float | None:
    """Parse a numeric string into a float when possible."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_header_label(raw_label: str) -> tuple[str, str | None, str]:
    """Split one ILThermo header label into its name, unit, and phase text."""

    text = html_to_text(raw_label)
    text = text.replace("*", "").strip()
    phase = ""
    if "=>" in text:
        text, phase = (part.strip() for part in text.split("=>", 1))
    if "," not in text:
        return text, None, phase
    name, unit = text.rsplit(",", 1)
    return name.strip(), unit.strip() or None, phase


def search_property_results(property_slug: str) -> list[dict[str, Any]]:
    """Search ILThermoPy for one pure-compound energetics property."""

    results = ilt.Search(n_compounds=1, prop=PROPERTY_NAMES[property_slug])
    if results.empty:
        return []
    return results.to_dict(orient="records")


def fetch_entry(set_id: str) -> Any:
    """Fetch one ILThermoPy entry by its set identifier."""

    return ilt.GetEntry(set_id)


def normalize_phase_tokens(value: Any) -> set[str]:
    """Normalize phase labels into a comparable token set."""

    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        tokens: set[str] = set()
        for item in value:
            tokens.update(normalize_phase_tokens(item))
        return tokens

    text = html_to_text(str(value)).lower()
    if not text:
        return set()

    tokens: set[str] = set()
    for piece in PHASE_SPLIT_RE.split(text):
        cleaned = WHITESPACE_RE.sub(" ", piece).strip()
        cleaned = TRAILING_PHASE_INDEX_RE.sub("", cleaned).strip()
        if cleaned:
            tokens.add(cleaned)
    return tokens


def stringify_phase_value(value: Any) -> str:
    """Convert a phase payload into a readable single-line label."""

    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        parts: list[str] = []
        for item in value:
            text = stringify_phase_value(item)
            if text and text not in parts:
                parts.append(text)
        return " | ".join(parts)

    text = html_to_text(str(value))
    text = TRAILING_PHASE_INDEX_RE.sub("", text).strip()
    return text


def resolve_phase_label(search_result: dict[str, Any], entry: Any, phase_context: str) -> str:
    """Resolve the most specific phase label available for one record set."""

    for candidate in (phase_context, getattr(entry, "phases", None), search_result.get("phases")):
        text = stringify_phase_value(candidate)
        if text:
            return text
    return ""


def extract_phase_context(raw_header: str) -> str:
    """Extract the per-column phase context from one ILThermoPy header label."""

    _, _, phase = parse_header_label(raw_header)
    return phase


def is_liquid_record(search_result: dict[str, Any], entry: Any, phase_context: str) -> bool:
    """Return whether the current search result and row context represent liquid data."""

    search_tokens = normalize_phase_tokens(search_result.get("phases"))
    if search_tokens and search_tokens != {"liquid"}:
        return False

    dataset_tokens = normalize_phase_tokens(getattr(entry, "phases", None) or [])
    if dataset_tokens and dataset_tokens != {"liquid"}:
        return False

    row_tokens = normalize_phase_tokens(phase_context)
    if row_tokens and row_tokens != {"liquid"}:
        return False

    return bool(search_tokens or dataset_tokens or row_tokens)


def extract_component(
    search_result: dict[str, Any],
    entry: Any,
) -> dict[str, Any]:
    """Extract the primary component metadata for one ILThermoPy entry."""

    components = getattr(entry, "components", None) or []
    component = components[0] if components else None
    component_name = html_to_text(getattr(component, "name", None) or search_result.get("cmp1"))
    raw_smiles = getattr(component, "smiles", None) or search_result.get("cmp1_smiles") or ""
    return {
        "component_name": component_name,
        "canonical_smiles": canonicalize_smiles(str(raw_smiles)) if raw_smiles else "",
    }


def extract_property_column(
    entry: Any,
    property_slug: str,
) -> tuple[str, str, str | None, str, bool]:
    """Find the property column name and metadata inside one ILThermoPy entry."""

    expected_label = PROPERTY_NAMES[property_slug].lower()
    for column_name, raw_label in (getattr(entry, "header", None) or {}).items():
        name, unit, _ = parse_header_label(str(raw_label))
        if name.lower().startswith(expected_label):
            return column_name, str(raw_label), name, unit, "*" in str(raw_label)
    raise RuntimeError(f"Could not find property column in set {getattr(entry, 'id', '')}")


def extract_conditions(entry: Any, row: dict[str, Any], property_column: str) -> dict[str, Any]:
    """Extract the contextual condition fields for one ILThermoPy data row."""

    conditions = {
        "temperature_value": None,
        "temperature_unit": "",
        "pressure_value": None,
        "pressure_unit": "",
        "wavelength_value": None,
        "wavelength_unit": "",
        "frequency_value": None,
        "frequency_unit": "",
    }
    for column_name, raw_label in (getattr(entry, "header", None) or {}).items():
        if column_name == property_column or column_name.startswith("d"):
            continue

        label_text, unit, _ = parse_header_label(str(raw_label))
        if label_text.lower().startswith("error of"):
            continue

        value = row.get(column_name)
        lowered = label_text.lower()
        if lowered.startswith("temperature"):
            conditions["temperature_value"] = parse_float(value)
            conditions["temperature_unit"] = unit or ""
        elif lowered.startswith("pressure"):
            conditions["pressure_value"] = parse_float(value)
            conditions["pressure_unit"] = unit or ""
        elif "wave" in lowered and "length" in lowered:
            conditions["wavelength_value"] = parse_float(value)
            conditions["wavelength_unit"] = unit or ""
        elif lowered.startswith("frequency"):
            conditions["frequency_value"] = parse_float(value)
            conditions["frequency_unit"] = unit or ""
    return conditions


def build_records(
    property_slug: str,
    search_result: dict[str, Any],
    entry: Any,
) -> list[dict[str, Any]]:
    """Build structured crawler records from one ILThermoPy entry."""

    property_column, raw_property_header, property_name, property_unit, header_starred = extract_property_column(entry, property_slug)
    footer_html = getattr(entry, "footnotes", None) or ""
    footer_starred = "<SUP>*</SUP>" in str(footer_html).upper() or "*" in html_to_text(str(footer_html))
    phase_context = extract_phase_context(raw_property_header)
    if property_slug in LIQUID_ONLY_PROPERTIES and not is_liquid_record(search_result, entry, phase_context):
        return []
    phase_label = resolve_phase_label(search_result, entry, phase_context)

    records: list[dict[str, Any]] = []
    component = extract_component(search_result, entry)
    standard_state_note = html_to_text(footer_html)

    data_frame = getattr(entry, "data", None)
    if data_frame is None or data_frame.empty:
        return []

    error_column = f"d{property_column}" if str(property_column).startswith("V") else ""
    for row_index, (_, row_series) in enumerate(data_frame.iterrows()):
        row = row_series.to_dict()
        if property_column not in row:
            continue
        value_raw = row.get(property_column)
        uncertainty_raw = row.get(error_column) if error_column else None
        records.append(
            {
                "record_id": f"{search_result['id']}-{row_index:04d}",
                "set_id": search_result["id"],
                "property_slug": property_slug,
                "phase": phase_label,
                "property_name": property_name,
                "property_value": parse_float(value_raw),
                "property_unit": property_unit or "",
                "property_stddev": parse_float(uncertainty_raw),
                "property_stddev_unit": property_unit or "",
                "standard_state_note": standard_state_note,
                "note_text": standard_state_note,
                "starred": header_starred or footer_starred,
                **component,
                **extract_conditions(entry, row, property_column),
            }
        )
    return records


def write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write the streamlined crawler records to a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "set_id",
        "property_slug",
        "phase",
        "component_name",
        "canonical_smiles",
        "temperature_value",
        "temperature_unit",
        "pressure_value",
        "pressure_unit",
        "wavelength_value",
        "wavelength_unit",
        "frequency_value",
        "frequency_unit",
        "property_name",
        "property_value",
        "property_unit",
        "property_stddev",
        "property_stddev_unit",
        "standard_state_note",
        "starred",
        "note_text",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def output_csv_path(output_dir: Path, property_slug: str) -> Path:
    """Build the canonical CSV output path for one crawled property."""

    return output_dir / f"pure_compound_{property_slug}.csv"


def write_json(data: dict[str, Any], output_path: Path) -> None:
    """Write one raw entry payload to disk for traceability."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ILThermo energetics crawler."""

    parser = argparse.ArgumentParser(
        description=(
            "Crawl phase-aware pure-compound ILThermo enthalpy-family records into streamlined CSV outputs. "
            "Enthalpy and entropy remain liquid-only; transition/fusion and vaporization/sublimation keep their reported phases. "
            "Component SMILES are sourced through ILThermoPy."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=raw_root(PROJECT_ROOT) / "ILThermo",
        help="Path to the directory where the crawler should write the CSV output and optional raw set files.",
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        choices=sorted(PROPERTY_NAMES),
        default=sorted(PROPERTY_NAMES),
        help=(
            "One or more ILThermo property names to crawl. The default runs enthalpy, entropy, "
            "enthalpy_of_transition_or_fusion, and enthalpy_of_vaporization_or_sublimation."
        ),
    )
    parser.add_argument(
        "--limit-sets",
        type=int,
        default=None,
        help="Optional maximum number of result sets to crawl per property, useful for quick smoke tests.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.5,
        help="Pause duration in seconds between successive ILThermoPy entry downloads.",
    )
    parser.add_argument(
        "--write-raw-sets",
        action="store_true",
        help="Write the raw per-set JSON payloads in addition to the streamlined CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    """Crawl the requested ILThermo energetics properties and write outputs."""

    disable_rdkit_logs()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for property_slug in args.properties:
        try:
            results = search_property_results(property_slug)
        except Exception as exc:
            raise RuntimeError(f"Search failed for {property_slug}: {exc}") from exc
        if args.limit_sets is not None:
            results = results[: args.limit_sets]

        property_records: list[dict[str, Any]] = []
        raw_set_dir = args.output_dir / "raw_sets" / property_slug
        for index, result in enumerate(results):
            detail = fetch_entry(str(result["id"]))
            if args.write_raw_sets:
                write_json(detail.response, raw_set_dir / f"{result['id']}.json")
            property_records.extend(
                build_records(
                    property_slug,
                    result,
                    detail,
                )
            )
            if index + 1 < len(results):
                time.sleep(args.pause_seconds)

        csv_path = output_csv_path(args.output_dir, property_slug)
        write_csv(property_records, csv_path)
        print(
            f"[{property_slug}] sets={len(results)} rows={len(property_records)} "
            f"resolved_smiles={sum(1 for row in property_records if row['canonical_smiles'])}"
        )
        print(f"[{property_slug}] saved={csv_path}")


if __name__ == "__main__":
    main()