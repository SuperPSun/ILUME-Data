"""Structure the ILThermo density raw file into a cleaned CSV output."""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _runner import main_for
from _shared import build_process_file, build_property_field_parser
from raw_prep import calc_pair_molar_mass_g_per_mol, normalize_unit

PROPERTY_SLUG = "density"
PROPERTY_NAME_PAT = r"(?:Specific density|Specific volume|Molar volume)"
_parse_property_fields = build_property_field_parser(PROPERTY_NAME_PAT)


def _to_g_per_cm3(value: float | None, unit: str | None) -> float | None:
    """Convert a density value to grams per cubic centimeter."""

    if value is None or unit is None:
        return None
    normalized = normalize_unit(unit)
    if normalized in {"kg/m^3", "kg/m3"}:
        return value * 1e-3
    if normalized in {"g/cm^3", "g/cm3"}:
        return value
    raise ValueError(f"unsupported density unit: {unit}")


def _to_m3_per_kg(value: float | None, unit: str | None) -> float | None:
    """Convert a specific-volume value to cubic meters per kilogram."""

    if value is None or unit is None:
        return None
    normalized = normalize_unit(unit)
    if normalized in {"m^3/kg", "m3/kg"}:
        return value
    raise ValueError(f"unsupported specific volume unit: {unit}")


def _to_m3_per_mol(value: float | None, unit: str | None) -> float | None:
    """Convert a molar-volume value to cubic meters per mole."""

    if value is None or unit is None:
        return None
    normalized = normalize_unit(unit)
    if normalized in {"m^3/mol", "m3/mol"}:
        return value
    raise ValueError(f"unsupported molar volume unit: {unit}")


def _standardize_property_value(
    property_name: str | None,
    property_value: float | None,
    property_unit: str | None,
    cation: str | None,
    anion: str | None,
) -> tuple[float | None, str | None]:
    """Normalize the density-family measurements into a common density label."""

    if property_name is None or property_value is None:
        return None, None

    name_lower = " ".join(property_name.strip().lower().split())
    if "specific volume" in name_lower:
        specific_volume = _to_m3_per_kg(property_value, property_unit)
        if specific_volume == 0:
            raise ValueError("specific volume is zero")
        return (1.0 / specific_volume) * 1e-3, "g/cm^3"

    if "molar volume" in name_lower:
        molar_volume = _to_m3_per_mol(property_value, property_unit)
        if molar_volume == 0:
            raise ValueError("molar volume is zero")
        density_kg_per_m3 = (calc_pair_molar_mass_g_per_mol(cation, anion) / 1000.0) / molar_volume
        return density_kg_per_m3 * 1e-3, "g/cm^3"

    if "density" in name_lower:
        return _to_g_per_cm3(property_value, property_unit), "g/cm^3"

    return None, None


process_file = build_process_file(
    standardize_value=_standardize_property_value,
    parse_property_fields=_parse_property_fields,
)


if __name__ == "__main__":
    main_for(PROPERTY_SLUG, process_file)