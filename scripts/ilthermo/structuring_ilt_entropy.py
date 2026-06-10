"""Structure the ILThermo entropy raw file into a cleaned CSV output."""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _runner import main_for
from _shared import build_energetics_process_file
from raw_prep import build_linear_standardizer

PROPERTY_SLUG = "entropy"
_standardize_property_value = build_linear_standardizer(
    quantity_name="entropy",
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
process_file = build_energetics_process_file(
    property_slug=PROPERTY_SLUG,
    standardize_value=_standardize_property_value,
)


if __name__ == "__main__":
    main_for(PROPERTY_SLUG, process_file)