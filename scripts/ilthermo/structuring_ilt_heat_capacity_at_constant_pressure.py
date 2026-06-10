"""Structure the ILThermo heat capacity at constant pressure raw file into a cleaned CSV output."""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _runner import main_for
from _shared import build_process_file, build_property_field_parser
from raw_prep import build_linear_standardizer

PROPERTY_SLUG = "heat_capacity_at_constant_pressure"
PROPERTY_NAME_PAT = r"Heat capacity at constant pressure"


def _skip_property_name(property_name: str) -> bool:
    """Skip misleading matches that belong to another property family."""

    return property_name.lower().startswith("specific density")


_parse_property_fields = build_property_field_parser(
    PROPERTY_NAME_PAT,
    skip_property_name=_skip_property_name,
)
_standardize_property_value = build_linear_standardizer(
    quantity_name="heat capacity",
    output_unit="J/mol/K",
    unit_factors={
        "j/mol/k": 1.0,
        "j/k/mol": 1.0,
        "j*mol^-1*k^-1": 1.0,
        "j*mol-1*k-1": 1.0,
        "kj/mol/k": 1000.0,
        "kj/k/mol": 1000.0,
        "kj*mol^-1*k^-1": 1000.0,
        "kj*mol-1*k-1": 1000.0,
    },
)
process_file = build_process_file(
    standardize_value=_standardize_property_value,
    parse_property_fields=_parse_property_fields,
)


if __name__ == "__main__":
    main_for(PROPERTY_SLUG, process_file)