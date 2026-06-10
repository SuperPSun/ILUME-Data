"""Structure the ILThermo thermal conductivity raw file into a cleaned CSV output."""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _runner import main_for
from _shared import build_process_file, build_property_field_parser
from raw_prep import build_linear_standardizer

PROPERTY_SLUG = "thermal_conductivity"
PROPERTY_NAME_PAT = r"Thermal conductivity"
_parse_property_fields = build_property_field_parser(PROPERTY_NAME_PAT)
_standardize_property_value = build_linear_standardizer(
    quantity_name="thermal conductivity",
    output_unit="W/m/K",
    unit_factors={
        "w/m/k": 1.0,
        "w*m^-1*k^-1": 1.0,
        "w*m-1*k-1": 1.0,
        "mw/m/k": 1e-3,
        "mw*m^-1*k^-1": 1e-3,
        "mw*m-1*k-1": 1e-3,
    },
)
process_file = build_process_file(
    standardize_value=_standardize_property_value,
    parse_property_fields=_parse_property_fields,
)


if __name__ == "__main__":
    main_for(PROPERTY_SLUG, process_file)