"""Structure the ILThermo electrical conductivity raw file into a cleaned CSV output."""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _runner import main_for
from _shared import build_process_file, build_property_field_parser
from raw_prep import build_linear_standardizer, to_log10

PROPERTY_SLUG = "electrical_conductivity"
PROPERTY_NAME_PAT = r"Electrical conductivity"
_parse_property_fields = build_property_field_parser(PROPERTY_NAME_PAT)
_standardize_property_value = build_linear_standardizer(
    quantity_name="electrical conductivity",
    output_unit="S/m (log scale)",
    unit_factors={
        "s/m": 1.0,
        "ms/cm": 0.1,
        "ms*cm^-1": 0.1,
        "ms*cm-1": 0.1,
    },
    value_transform=lambda value: to_log10(value, "electrical conductivity"),
)
process_file = build_process_file(
    standardize_value=_standardize_property_value,
    parse_property_fields=_parse_property_fields,
)


if __name__ == "__main__":
    main_for(PROPERTY_SLUG, process_file)