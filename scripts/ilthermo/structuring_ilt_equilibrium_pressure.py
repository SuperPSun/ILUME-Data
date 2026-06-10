"""Structure the ILThermo equilibrium pressure raw file into a cleaned CSV output."""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _runner import main_for
from _shared import build_process_file, build_property_field_parser
from raw_prep import build_converter_standardizer, to_kpa, to_log10

PROPERTY_SLUG = "equilibrium_pressure"
PROPERTY_NAME_PAT = r"Equilibrium pressure"
_parse_property_fields = build_property_field_parser(PROPERTY_NAME_PAT)
_standardize_property_value = build_converter_standardizer(
    quantity_name="equilibrium pressure",
    output_unit="kPa (log scale)",
    converter=to_kpa,
    value_transform=lambda value: to_log10(value, "equilibrium pressure"),
)
process_file = build_process_file(
    standardize_value=_standardize_property_value,
    parse_property_fields=_parse_property_fields,
)


if __name__ == "__main__":
    main_for(PROPERTY_SLUG, process_file)