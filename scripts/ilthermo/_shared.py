"""Provide shared parsing and processing helpers for ILThermo property scripts."""

from __future__ import annotations

import csv
import re
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raw_prep import canonicalize_smiles, split_cation_anion, to_kelvin, to_kpa, to_mhz, to_nm

PropertyStandardizer = Callable[
    [str | None, float | None, str | None, str | None, str | None],
    tuple[float | None, str | None],
]
PropertyFieldParser = Callable[[str, str], tuple[str | None, str | None, float | None, str | None]]
StructuredRow = dict[str, object]
NoteResolver = Callable[[StructuredRow], str]
ProcessFile = Callable[[Path, Path, Path | None], pd.DataFrame]

NUM_PAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
SMILES_RE = re.compile(r"^\s*smiles\s*:\s*(\S+)", flags=re.IGNORECASE)
TEMPERATURE_RE = re.compile(
    r"(?<!equilibrium\s)(?<!melting\s)Temperature,\s*([^:=>,]+?)\s*(?:=>\s*Liquid)?\s*:\s*({})".format(NUM_PAT),
    flags=re.IGNORECASE,
)
PRESSURE_RE = re.compile(
    r"(?<!equilibrium\s)\bPressure,\s*([^:=>,]+?)\s*(?:=>\s*Liquid)?\s*:\s*({})".format(NUM_PAT),
    flags=re.IGNORECASE,
)
FREQUENCY_RE = re.compile(
    r"Frequency,\s*([^:=>,]+?)\s*(?:=>\s*Liquid)?\s*:\s*({})".format(NUM_PAT),
    flags=re.IGNORECASE,
)
WAVELENGTH_RE = re.compile(
    r"Wavelength,\s*([^:=>,]+?)\s*(?:=>\s*Liquid)?\s*:\s*({})".format(NUM_PAT),
    flags=re.IGNORECASE,
)


def resolve_energetics_csv_paths(csv_path: Path | None) -> list[Path]:
    """Resolve the available energetics CSV inputs for note lookup."""

    if csv_path is None:
        return []

    split_candidates = [
        csv_path.parent / "pure_compound_enthalpy.csv",
        csv_path.parent / "pure_compound_entropy.csv",
    ]
    legacy_split_candidates = [
        csv_path.parent / "pure_compound_energetics" / "pure_compound_enthalpy.csv",
        csv_path.parent / "pure_compound_energetics" / "pure_compound_entropy.csv",
    ]
    existing_split = [path for path in split_candidates if path.exists()]
    if not existing_split:
        existing_split = [path for path in legacy_split_candidates if path.exists()]
    if existing_split:
        return existing_split

    if csv_path.exists():
        return [csv_path]
    return []


def resolve_property_csv_paths(csv_path: Path | None, property_slug: str) -> list[Path]:
    """Resolve the available crawler CSV inputs for one ILThermo property."""

    if csv_path is None:
        return []

    direct_candidate = csv_path.parent / f"pure_compound_{property_slug}.csv"
    if direct_candidate.exists():
        return [direct_candidate]

    legacy_candidate = csv_path.parent / "pure_compound_energetics" / f"pure_compound_{property_slug}.csv"
    if legacy_candidate.exists():
        return [legacy_candidate]

    if csv_path.exists() and csv_path.name == f"pure_compound_{property_slug}.csv":
        return [csv_path]
    return []


def compile_property_pair_regex(property_name_pattern: str) -> re.Pattern[str]:
    """Compile the standard ILThermo property/value regex for one property pattern."""

    return re.compile(
        r"({})(?:,\s*([^:]+?))?\s*(?:=>\s*Liquid)?\s*:\s*({})".format(property_name_pattern, NUM_PAT),
        flags=re.IGNORECASE,
    )


def build_property_field_parser(
    property_name_pattern: str,
    *,
    skip_property_name: Callable[[str], bool] | None = None,
) -> PropertyFieldParser:
    """Build a parser that extracts one property name, unit, and numeric value."""

    property_pair_re = compile_property_pair_regex(property_name_pattern)

    def _parse_property_fields(
        _raw_text: str,
        remainder_text: str,
    ) -> tuple[str | None, str | None, float | None, str | None]:
        """Parse the property-specific remainder text from one ILThermo row."""

        for match_prop in property_pair_re.finditer(remainder_text):
            property_name = match_prop.group(1).strip()
            if property_name.lower().startswith("error of"):
                continue
            if skip_property_name is not None and skip_property_name(property_name):
                continue
            property_unit = match_prop.group(2).strip() if match_prop.group(2) is not None else None
            property_value = float(match_prop.group(3))
            return property_name, property_unit, property_value, None
        return None, None, None, "property regex not matched"

    return _parse_property_fields


def _load_energetics_note_lookup(property_slug: str, csv_path: Path | None) -> dict[bool, str]:
    """Load starred and non-starred note text for one energetics property."""

    csv_paths = resolve_energetics_csv_paths(csv_path)
    if not csv_paths:
        return {}

    grouped: dict[bool, set[str]] = {True: set(), False: set()}
    for current_csv_path in csv_paths:
        with current_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row_property_slug = (row.get("property_slug") or "").strip().lower()
                if row_property_slug != property_slug:
                    continue
                note_text = (row.get("note_text") or "").strip()
                starred_text = (row.get("starred") or "").strip().lower()
                starred = starred_text in {"true", "1", "yes"}
                grouped[starred].add(note_text)

    resolved: dict[bool, str] = {}
    for starred, values in grouped.items():
        non_empty = sorted(value for value in values if value)
        if non_empty:
            resolved[starred] = " | ".join(non_empty)
        elif values:
            resolved[starred] = ""
    return resolved


def build_energetics_note_resolver(property_slug: str, csv_path: Path | None) -> NoteResolver:
    """Build a note resolver for energetics-backed ILThermo properties."""

    note_lookup = _load_energetics_note_lookup(property_slug, csv_path)

    def _resolve_note(row: StructuredRow) -> str:
        """Resolve the note text for one raw ILThermo record."""

        raw_text = str(row.get("raw_text") or "")
        is_starred = "<sup>*</sup>" in raw_text.lower()
        if is_starred and True in note_lookup:
            return note_lookup[True]
        if False in note_lookup:
            return note_lookup[False]
        if True in note_lookup:
            return note_lookup[True]
        return ""

    return _resolve_note


def _canonicalize_lookup_smiles(raw_chem_text: str | None) -> str:
    """Canonicalize one raw smiles field for crawler-row matching."""

    if not raw_chem_text:
        return ""
    chem_text = raw_chem_text.strip()
    if chem_text.lower().startswith("smiles:"):
        chem_text = chem_text.split(":", 1)[1].strip()
    return canonicalize_smiles(chem_text)


def _normalize_lookup_float(value: object) -> float | None:
    """Normalize one numeric value into a stable lookup key component."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return round(float(text), 8)
    except ValueError:
        return None


def _normalize_lookup_text(value: object) -> str:
    """Normalize one text field into a stable lookup key component."""

    return str(value or "").strip().lower()


def _build_phase_lookup_key(
    *,
    canonical_smiles: str,
    temperature_k: float | None,
    pressure_kpa: float | None,
    frequency_mhz: float | None,
    wavelength_nm: float | None,
    property_name: str | None,
    property_value: float | None,
    property_unit: str | None,
) -> tuple[object, ...]:
    """Build the composite key used to match structured rows to crawled phase rows."""

    return (
        canonical_smiles,
        _normalize_lookup_float(temperature_k),
        _normalize_lookup_float(pressure_kpa),
        _normalize_lookup_float(frequency_mhz),
        _normalize_lookup_float(wavelength_nm),
        _normalize_lookup_text(property_name),
        _normalize_lookup_float(property_value),
        _normalize_lookup_text(property_unit),
    )


def _compose_phase_note(phase_text: str, note_text: str) -> str:
    """Compose the final phase-prefixed note string."""

    phase_text = phase_text.strip()
    note_text = note_text.strip()
    if phase_text and note_text:
        return f"Phase: {phase_text} | {note_text}"
    if phase_text:
        return f"Phase: {phase_text}"
    return note_text


def _load_phase_note_lookup(property_slug: str, csv_path: Path | None) -> dict[tuple[object, ...], str]:
    """Load row-level phase-aware notes for one crawled ILThermo property."""

    csv_paths = resolve_property_csv_paths(csv_path, property_slug)
    if not csv_paths:
        return {}

    lookup: dict[tuple[object, ...], set[str]] = {}
    for current_csv_path in csv_paths:
        with current_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = _build_phase_lookup_key(
                    canonical_smiles=str(row.get("canonical_smiles") or "").strip(),
                    temperature_k=_normalize_lookup_float(row.get("temperature_value")),
                    pressure_kpa=_normalize_lookup_float(row.get("pressure_value")),
                    frequency_mhz=_normalize_lookup_float(row.get("frequency_value")),
                    wavelength_nm=_normalize_lookup_float(row.get("wavelength_value")),
                    property_name=(row.get("property_name") or "").strip(),
                    property_value=_normalize_lookup_float(row.get("property_value")),
                    property_unit=(row.get("property_unit") or "").strip(),
                )
                note_text = _compose_phase_note(
                    str(row.get("phase") or "").strip(),
                    str(row.get("note_text") or "").strip(),
                )
                if not note_text:
                    continue
                lookup.setdefault(key, set()).add(note_text)

    return {key: " | ".join(sorted(values)) for key, values in lookup.items() if values}


def build_phase_note_resolver(property_slug: str, csv_path: Path | None) -> NoteResolver:
    """Build a row-level resolver that injects crawled phase text into the note field."""

    phase_lookup = _load_phase_note_lookup(property_slug, csv_path)

    def _resolve_note(row: StructuredRow) -> str:
        """Resolve the phase-aware note text for one structured ILThermo row."""

        key = _build_phase_lookup_key(
            canonical_smiles=_canonicalize_lookup_smiles(str(row.get("raw_chem_text") or "")),
            temperature_k=row.get("temperature_K"),
            pressure_kpa=row.get("pressure_kPa"),
            frequency_mhz=row.get("frequency_MHz"),
            wavelength_nm=row.get("wavelength_nm"),
            property_name=str(row.get("property_name") or ""),
            property_value=row.get("property_value"),
            property_unit=str(row.get("property_unit") or ""),
        )
        return phase_lookup.get(key, "")

    return _resolve_note


def _csv_truthy(value: object) -> bool:
    """Interpret one CSV cell as a boolean flag."""

    return str(value or "").strip().lower() in {"true", "1", "yes"}


def _build_crawler_source_text(row: dict[str, str]) -> str:
    """Build a readable provenance string for one crawler CSV row."""

    parts = [
        f"record_id={str(row.get('record_id') or '').strip()}",
        f"set_id={str(row.get('set_id') or '').strip()}",
        f"component_name={str(row.get('component_name') or '').strip()}",
        f"property_name={str(row.get('property_name') or '').strip()}",
        f"property_value={str(row.get('property_value') or '').strip()}",
        f"property_unit={str(row.get('property_unit') or '').strip()}",
        f"phase={str(row.get('phase') or '').strip()}",
    ]
    return " | ".join(part for part in parts if not part.endswith("="))


def _build_energetics_csv_note(row: dict[str, str]) -> str:
    """Build the final note text for one crawler energetics row."""

    note_text = str(row.get("note_text") or row.get("standard_state_note") or "").strip()
    return note_text


def _build_phase_csv_note(row: dict[str, str]) -> str:
    """Build the final phase-prefixed note text for one crawler row."""

    note_text = str(row.get("note_text") or row.get("standard_state_note") or "").strip()
    phase_text = str(row.get("phase") or "").strip()
    return _compose_phase_note(phase_text, note_text)


def build_process_file(
    *,
    standardize_value: PropertyStandardizer,
    parse_property_fields: PropertyFieldParser,
    note_resolver_factory: Callable[[Path | None], NoteResolver] | None = None,
) -> ProcessFile:
    """Build a standard process_file wrapper for one property script."""

    def _process_file(input_path: Path, output_path: Path, energetics_csv: Path | None = None):
        """Process one raw ILThermo file into its structured CSV output."""

        resolve_note = note_resolver_factory(energetics_csv) if note_resolver_factory is not None else None
        return process_property_file(
            input_path=input_path,
            output_path=output_path,
            standardize_value=standardize_value,
            parse_property_fields=parse_property_fields,
            resolve_note=resolve_note,
        )

    return _process_file


def build_energetics_process_file(
    *,
    property_slug: str,
    standardize_value: PropertyStandardizer,
) -> ProcessFile:
    """Build a process_file wrapper for energetics-backed properties."""

    return build_crawler_energetics_process_file(
        property_slug=property_slug,
        standardize_value=standardize_value,
    )


def build_phase_note_process_file(
    *,
    property_slug: str,
    standardize_value: PropertyStandardizer,
) -> ProcessFile:
    """Build a process_file wrapper for properties that need row-level phase notes."""

    return build_crawler_phase_process_file(
        property_slug=property_slug,
        standardize_value=standardize_value,
    )


def _parse_crawler_csv_row(row: dict[str, str], invalid_smiles: set[str]) -> StructuredRow:
    """Parse one crawler CSV row into the shared structured-row shape."""

    raw_smiles = str(row.get("canonical_smiles") or "").strip()
    raw_cation, raw_anion = split_cation_anion(raw_smiles if raw_smiles else None)
    cation = canonicalize_smiles(raw_cation, invalid_smiles) if raw_cation else None
    anion = canonicalize_smiles(raw_anion, invalid_smiles) if raw_anion else None

    property_name = str(row.get("property_name") or "").strip() or None
    property_unit = str(row.get("property_unit") or "").strip() or None
    property_value = _normalize_lookup_float(row.get("property_value"))

    parse_errors: list[str] = []
    if property_name is None:
        parse_errors.append("missing property_name")
    if property_value is None:
        parse_errors.append("missing property_value")

    return {
        "raw_text": _build_crawler_source_text(row),
        "raw_chem_text": f"smiles:{raw_smiles}" if raw_smiles else None,
        "temperature_unit": str(row.get("temperature_unit") or "").strip() or None,
        "temperature_value": _normalize_lookup_float(row.get("temperature_value")),
        "pressure_unit": str(row.get("pressure_unit") or "").strip() or None,
        "pressure_value": _normalize_lookup_float(row.get("pressure_value")),
        "frequency_unit": str(row.get("frequency_unit") or "").strip() or None,
        "frequency_value": _normalize_lookup_float(row.get("frequency_value")),
        "wavelength_unit": str(row.get("wavelength_unit") or "").strip() or None,
        "wavelength_value": _normalize_lookup_float(row.get("wavelength_value")),
        "property_name": property_name,
        "property_unit": property_unit,
        "property_value": property_value,
        "parse_error": "; ".join(parse_errors) if parse_errors else None,
        "remainder_text": "",
        "crawler_row": row,
        "cation": cation,
        "anion": anion,
    }


def process_crawler_csv_file(
    *,
    input_path: Path,
    output_path: Path,
    standardize_value: PropertyStandardizer,
    note_builder: Callable[[dict[str, str]], str] | None = None,
) -> pd.DataFrame:
    """Process one crawler CSV into the shared structured ILThermo output shape."""

    unit_fail_rows: list[dict[str, str]] = []
    invalid_smiles: set[str] = set()
    raw_rows = 0
    final_rows: list[dict[str, object]] = []

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
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
            "note_text",
            "standard_state_note",
            "phase",
        }
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(required_columns - fieldnames)
        if missing_columns:
            raise RuntimeError(
                f"Crawler CSV {input_path.name} is missing required columns: {', '.join(missing_columns)}"
            )
        for csv_row in reader:
            raw_rows += 1
            row = _parse_crawler_csv_row(csv_row, invalid_smiles)

            converted_temp = convert_context_value(
                to_kelvin,
                row["temperature_value"],
                row["temperature_unit"],
                "temperature",
                str(row["raw_text"]),
                unit_fail_rows,
            )
            converted_pressure = convert_context_value(
                to_kpa,
                row["pressure_value"],
                row["pressure_unit"],
                "pressure",
                str(row["raw_text"]),
                unit_fail_rows,
            )
            converted_frequency = convert_context_value(
                to_mhz,
                row["frequency_value"],
                row["frequency_unit"],
                "frequency",
                str(row["raw_text"]),
                unit_fail_rows,
            )
            converted_wavelength = convert_context_value(
                to_nm,
                row["wavelength_value"],
                row["wavelength_unit"],
                "wavelength",
                str(row["raw_text"]),
                unit_fail_rows,
            )

            try:
                label, standard_unit = standardize_value(
                    row["property_name"],
                    row["property_value"],
                    row["property_unit"],
                    row["cation"],
                    row["anion"],
                )
            except Exception as exc:
                unit_fail_rows.append({"error": str(exc), "raw_text": str(row["raw_text"])})
                label, standard_unit = None, None

            final_rows.append(
                {
                    "cation": row["cation"],
                    "anion": row["anion"],
                    "temperature_K": converted_temp,
                    "pressure_kPa": converted_pressure,
                    "frequency_MHz": converted_frequency,
                    "wavelength_nm": converted_wavelength,
                    "property_name": row["property_name"],
                    "property_unit": row["property_unit"],
                    "property_value": row["property_value"],
                    "label": label,
                    "standard_unit": standard_unit,
                    "parse_error": row["parse_error"],
                    "note": note_builder(csv_row) if note_builder is not None else "",
                    "source_text": row["raw_text"],
                }
            )

    df = pd.DataFrame(final_rows)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[{input_path.name}] raw_rows={raw_rows}")
    print(f"[{input_path.name}] final_rows={len(df)}, duplicate_rows_removed={before - len(df)}")
    print(f"[{input_path.name}] unit_convert_fail_count={len(unit_fail_rows)}")
    print(f"[{input_path.name}] invalid_smiles_count={len(invalid_smiles)}")
    print(f"[{input_path.name}] saved={output_path}")
    return df


def build_crawler_csv_process_file(
    *,
    standardize_value: PropertyStandardizer,
    note_builder: Callable[[dict[str, str]], str] | None = None,
) -> ProcessFile:
    """Build a process_file wrapper for properties that read crawler CSV inputs directly."""

    def _process_file(input_path: Path, output_path: Path, _energetics_csv: Path | None = None):
        """Process one crawler CSV file into its structured CSV output."""

        return process_crawler_csv_file(
            input_path=input_path,
            output_path=output_path,
            standardize_value=standardize_value,
            note_builder=note_builder,
        )

    return _process_file


def build_crawler_energetics_process_file(
    *,
    property_slug: str,
    standardize_value: PropertyStandardizer,
) -> ProcessFile:
    """Build a crawler-CSV-backed process_file wrapper for enthalpy and entropy."""

    del property_slug
    return build_crawler_csv_process_file(
        standardize_value=standardize_value,
        note_builder=_build_energetics_csv_note,
    )


def build_crawler_phase_process_file(
    *,
    property_slug: str,
    standardize_value: PropertyStandardizer,
) -> ProcessFile:
    """Build a crawler-CSV-backed process_file wrapper for phase-aware properties."""

    del property_slug
    return build_crawler_csv_process_file(
        standardize_value=standardize_value,
        note_builder=_build_phase_csv_note,
    )


def read_dedup_lines(input_path: Path) -> tuple[list[str], list[str], int]:
    """Read raw lines and collapse consecutive duplicates for stable processing."""

    with input_path.open("r", encoding="utf-8") as handle:
        raw_lines = [line.strip() for line in handle if line.strip()]

    dedup_lines: list[str] = []
    index = 0
    max_repeat = 1
    while index < len(raw_lines):
        current = raw_lines[index]
        next_index = index
        while next_index < len(raw_lines) and raw_lines[next_index] == current:
            next_index += 1
        max_repeat = max(max_repeat, next_index - index)
        dedup_lines.append(current)
        index = next_index
    return raw_lines, dedup_lines, max_repeat


def parse_common_fields(line: str) -> dict[str, object]:
    """Extract chemistry and shared condition fields from one ILThermo row."""

    row: dict[str, object] = {
        "raw_text": line,
        "raw_chem_text": None,
        "temperature_unit": None,
        "temperature_value": None,
        "pressure_unit": None,
        "pressure_value": None,
        "frequency_unit": None,
        "frequency_value": None,
        "wavelength_unit": None,
        "wavelength_value": None,
        "property_name": None,
        "property_unit": None,
        "property_value": None,
        "parse_error": None,
        "remainder_text": line,
    }

    match_smiles = SMILES_RE.search(line)
    if match_smiles:
        row["raw_chem_text"] = "smiles:" + match_smiles.group(1)
    else:
        row["parse_error"] = "missing smiles"

    match_temp = TEMPERATURE_RE.search(line)
    if match_temp:
        row["temperature_unit"] = match_temp.group(1).strip()
        row["temperature_value"] = float(match_temp.group(2))

    match_press = PRESSURE_RE.search(line)
    if match_press:
        row["pressure_unit"] = match_press.group(1).strip()
        row["pressure_value"] = float(match_press.group(2))

    match_freq = FREQUENCY_RE.search(line)
    if match_freq:
        row["frequency_unit"] = match_freq.group(1).strip()
        row["frequency_value"] = float(match_freq.group(2))

    match_wave = WAVELENGTH_RE.search(line)
    if match_wave:
        row["wavelength_unit"] = match_wave.group(1).strip()
        row["wavelength_value"] = float(match_wave.group(2))

    remainder_text = re.sub(r"^\s*smiles\s*:\s*\S+\s*", "", line, count=1, flags=re.IGNORECASE)
    remainder_text = TEMPERATURE_RE.sub("", remainder_text, count=1)
    remainder_text = PRESSURE_RE.sub("", remainder_text, count=1)
    remainder_text = FREQUENCY_RE.sub("", remainder_text, count=1)
    remainder_text = WAVELENGTH_RE.sub("", remainder_text, count=1).strip()
    row["remainder_text"] = remainder_text
    return row


def convert_context_value(
    converter: Callable[[float | None, str | None], float | None],
    value: float | None,
    unit: str | None,
    field_name: str,
    raw_text: str,
    unit_fail_rows: list[dict[str, str]],
) -> float | None:
    """Convert one contextual field and record any conversion failure."""

    try:
        return converter(value, unit)
    except Exception as exc:
        unit_fail_rows.append({"error": f"{field_name}: {exc}", "raw_text": raw_text})
        return None


def process_property_file(
    *,
    input_path: Path,
    output_path: Path,
    standardize_value: PropertyStandardizer,
    parse_property_fields: PropertyFieldParser,
    resolve_note: NoteResolver | None = None,
) -> pd.DataFrame:
    """Run the shared ILThermo row-processing pipeline for one property file."""

    regex_fail_rows: list[str] = []
    unit_fail_rows: list[dict[str, str]] = []
    invalid_smiles: set[str] = set()

    raw_lines, dedup_lines, max_repeat = read_dedup_lines(input_path)
    final_rows: list[dict[str, object]] = []
    for line in dedup_lines:
        row = parse_common_fields(line)
        property_name, property_unit, property_value, property_parse_error = parse_property_fields(
            str(row["raw_text"]),
            str(row["remainder_text"]),
        )
        if property_name is None:
            if row["parse_error"] is None:
                row["parse_error"] = property_parse_error or "property regex not matched"
            regex_fail_rows.append(line)
        else:
            row["property_name"] = property_name
            row["property_unit"] = property_unit
            row["property_value"] = property_value

        raw_cation, raw_anion = split_cation_anion(row["raw_chem_text"])
        cation = canonicalize_smiles(raw_cation, invalid_smiles) if raw_cation else None
        anion = canonicalize_smiles(raw_anion, invalid_smiles) if raw_anion else None

        converted_temp = convert_context_value(
            to_kelvin,
            row["temperature_value"],
            row["temperature_unit"],
            "temperature",
            row["raw_text"],
            unit_fail_rows,
        )
        converted_pressure = convert_context_value(
            to_kpa,
            row["pressure_value"],
            row["pressure_unit"],
            "pressure",
            row["raw_text"],
            unit_fail_rows,
        )
        converted_frequency = convert_context_value(
            to_mhz,
            row["frequency_value"],
            row["frequency_unit"],
            "frequency",
            row["raw_text"],
            unit_fail_rows,
        )
        converted_wavelength = convert_context_value(
            to_nm,
            row["wavelength_value"],
            row["wavelength_unit"],
            "wavelength",
            row["raw_text"],
            unit_fail_rows,
        )

        try:
            label, standard_unit = standardize_value(
                row["property_name"],
                row["property_value"],
                row["property_unit"],
                cation,
                anion,
            )
        except Exception as exc:
            unit_fail_rows.append({"error": str(exc), "raw_text": row["raw_text"]})
            label, standard_unit = None, None

        resolved_row = {
            **row,
            "temperature_K": converted_temp,
            "pressure_kPa": converted_pressure,
            "frequency_MHz": converted_frequency,
            "wavelength_nm": converted_wavelength,
            "label": label,
            "standard_unit": standard_unit,
        }

        final_rows.append(
            {
                "cation": cation,
                "anion": anion,
                "temperature_K": converted_temp,
                "pressure_kPa": converted_pressure,
                "frequency_MHz": converted_frequency,
                "wavelength_nm": converted_wavelength,
                "property_name": row["property_name"],
                "property_unit": row["property_unit"],
                "property_value": row["property_value"],
                "label": label,
                "standard_unit": standard_unit,
                "parse_error": row["parse_error"],
                "note": resolve_note(resolved_row) if resolve_note is not None else "",
                "source_text": row["raw_text"],
            }
        )

    df = pd.DataFrame(final_rows)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[{input_path.name}] raw={len(raw_lines)}, dedup={len(dedup_lines)}, max_repeat={max_repeat}")
    print(f"[{input_path.name}] final_rows={len(df)}, duplicate_rows_removed={before - len(df)}")
    print(f"[{input_path.name}] regex_fail_count={len(regex_fail_rows)}")
    print(f"[{input_path.name}] unit_convert_fail_count={len(unit_fail_rows)}")
    print(f"[{input_path.name}] invalid_smiles_count={len(invalid_smiles)}")
    print(f"[{input_path.name}] saved={output_path}")
    return df