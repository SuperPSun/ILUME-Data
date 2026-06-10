"""Structure the ILThermo isobaric coefficient of volume expansion raw file into a cleaned CSV output."""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _runner import main_for
from _shared import build_process_file, build_property_field_parser
from raw_prep import build_linear_standardizer

PROPERTY_SLUG = "isobaric_coefficient_of_volume_expansion"
PROPERTY_NAME_PAT = r"Isobaric coefficient of volume expansion"
_parse_property_fields = build_property_field_parser(PROPERTY_NAME_PAT)
_standardize_property_value = build_linear_standardizer(
    quantity_name="isobaric coefficient of volume expansion",
    output_unit="K^-1",
    unit_factors={
        "k^-1": 1.0,
        "1/k": 1.0,
        "k-1": 1.0,
    },
)
process_file = build_process_file(
    standardize_value=_standardize_property_value,
    parse_property_fields=_parse_property_fields,
)


if __name__ == "__main__":
    main_for(PROPERTY_SLUG, process_file)