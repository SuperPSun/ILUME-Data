"""Structure the ILThermo enthalpy of transition or fusion raw file into a cleaned CSV output."""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _runner import main_for
from _shared import build_phase_note_process_file
from raw_prep import build_linear_standardizer

PROPERTY_SLUG = "enthalpy_of_transition_or_fusion"
_standardize_property_value = build_linear_standardizer(
    quantity_name="enthalpy",
    output_unit="kJ/mol",
    unit_factors={
        "kj/mol": 1.0,
        "j/mol": 1e-3,
    },
)
process_file = build_phase_note_process_file(
    property_slug=PROPERTY_SLUG,
    standardize_value=_standardize_property_value,
)


if __name__ == "__main__":
    main_for(PROPERTY_SLUG, process_file)