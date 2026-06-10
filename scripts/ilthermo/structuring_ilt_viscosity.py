"""Structure the ILThermo viscosity raw file into a cleaned CSV output."""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _runner import main_for
from _shared import build_process_file, build_property_field_parser
from raw_prep import build_linear_standardizer, to_log10

PROPERTY_SLUG = "viscosity"
PROPERTY_NAME_PAT = r"(?:Viscosity|Kinematic viscosity)"
_parse_property_fields = build_property_field_parser(PROPERTY_NAME_PAT)
_standardize_property_value = build_linear_standardizer(
    quantity_name="viscosity",
    output_unit="mPa*s (log scale)",
    unit_factors={
        "mpa*s": 1.0,
        "mpas": 1.0,
        "cp": 1.0,
        "pa*s": 1000.0,
        "pas": 1000.0,
    },
    value_transform=lambda value: to_log10(value, "viscosity"),
)
process_file = build_process_file(
    standardize_value=_standardize_property_value,
    parse_property_fields=_parse_property_fields,
)


if __name__ == "__main__":
    main_for(PROPERTY_SLUG, process_file)