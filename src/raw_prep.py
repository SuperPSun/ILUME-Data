"""Provide shared raw-data preparation utilities for the repository.

This module centralizes cross-source helpers such as path resolution,
SMILES canonicalization, ILThermo property metadata, and reusable unit
conversion helpers under a flat src-level import path.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

PropertyStandardizer = Callable[
    [str | None, float | None, str | None, str | None, str | None],
    tuple[float | None, str | None],
]


@dataclass(frozen=True)
class ILThermoPropertySpec:
    """Describe the raw-input and structured-output filenames for one property."""

    slug: str
    input_filename: str
    output_filename: str


PROPERTY_SPECS = (
    ILThermoPropertySpec("density", "ilt_density_data.txt", "ilt_density_structured.csv"),
    ILThermoPropertySpec(
        "electrical_conductivity",
        "ilt_electrical_conductivity_data.txt",
        "ilt_electrical_conductivity_structured.csv",
    ),
    ILThermoPropertySpec("enthalpy", "pure_compound_enthalpy.csv", "ilt_enthalpy_structured.csv"),
    ILThermoPropertySpec(
        "enthalpy_of_transition_or_fusion",
        "pure_compound_enthalpy_of_transition_or_fusion.csv",
        "ilt_enthalpy_of_transition_or_fusion_structured.csv",
    ),
    ILThermoPropertySpec(
        "enthalpy_of_vaporization_or_sublimation",
        "pure_compound_enthalpy_of_vaporization_or_sublimation.csv",
        "ilt_enthalpy_of_vaporization_or_sublimation_structured.csv",
    ),
    ILThermoPropertySpec("entropy", "pure_compound_entropy.csv", "ilt_entropy_structured.csv"),
    ILThermoPropertySpec(
        "equilibrium_pressure",
        "ilt_equilibrium_pressure_data.txt",
        "ilt_equilibrium_pressure_structured.csv",
    ),
    ILThermoPropertySpec(
        "equilibrium_temperature",
        "ilt_equilibrium_temperature_data.txt",
        "ilt_equilibrium_temperature_structured.csv",
    ),
    ILThermoPropertySpec(
        "heat_capacity_at_constant_pressure",
        "ilt_heat_capacity_at_constant_pressure_data.txt",
        "ilt_heat_capacity_at_constant_pressure_structured.csv",
    ),
    ILThermoPropertySpec(
        "heat_capacity_at_vapor_saturation_pressure",
        "ilt_heat_capacity_at_vapor_saturation_pressure_data.txt",
        "ilt_heat_capacity_at_vapor_saturation_pressure_structured.csv",
    ),
    ILThermoPropertySpec(
        "isobaric_coefficient_of_volume_expansion",
        "ilt_isobaric_coefficient_of_volume_expansion_data.txt",
        "ilt_isobaric_coefficient_of_volume_expansion_structured.csv",
    ),
    ILThermoPropertySpec(
        "normal_melting_temperature",
        "ilt_normal_melting_temperature_data.txt",
        "ilt_normal_melting_temperature_structured.csv",
    ),
    ILThermoPropertySpec(
        "refractive_index",
        "ilt_refractive_index_data.txt",
        "ilt_refractive_index_structured.csv",
    ),
    ILThermoPropertySpec(
        "relative_permittivity",
        "ilt_relative_permittivity_data.txt",
        "ilt_relative_permittivity_structured.csv",
    ),
    ILThermoPropertySpec(
        "self_diffusion_coefficient",
        "ilt_self_diffusion_coefficient_data.txt",
        "ilt_self_diffusion_coefficient_structured.csv",
    ),
    ILThermoPropertySpec("speed_of_sound", "ilt_speed_of_sound_data.txt", "ilt_speed_of_sound_structured.csv"),
    ILThermoPropertySpec(
        "surface_tension_liquid_gas",
        "ilt_surface_tension_liquid-gas_data.txt",
        "ilt_surface_tension_liquid_gas_structured.csv",
    ),
    ILThermoPropertySpec(
        "thermal_conductivity",
        "ilt_thermal_conductivity_data.txt",
        "ilt_thermal_conductivity_structured.csv",
    ),
    ILThermoPropertySpec(
        "thermal_diffusivity",
        "ilt_thermal_diffusivity_data.txt",
        "ilt_thermal_diffusivity_structured.csv",
    ),
    ILThermoPropertySpec("viscosity", "ilt_viscosity_data.txt", "ilt_viscosity_structured.csv"),
)

SPEC_BY_SLUG = {spec.slug: spec for spec in PROPERTY_SPECS}
SPEC_BY_INPUT = {spec.input_filename: spec for spec in PROPERTY_SPECS}


def project_root() -> Path:
    """Return the repository root for the current workspace."""

    return Path(__file__).resolve().parents[1]


def data_root(root: Path | None = None) -> Path:
    """Return the top-level data directory for the repository."""

    base = root or project_root()
    return Path(base).resolve() / "data"


def raw_root(root: Path | None = None) -> Path:
    """Return the raw-data directory under the repository data tree."""

    return data_root(root) / "raw"


def structured_root(root: Path | None = None) -> Path:
    """Return the structured-data directory under the repository data tree."""

    return data_root(root) / "structured"


def disable_rdkit_logs() -> None:
    """Silence noisy RDKit warnings that are expected during parsing."""

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")


def canonicalize_smiles(smiles: str, invalid_sink: set[str] | None = None) -> str:
    """Canonicalize a SMILES string and optionally record invalid inputs."""

    smiles = (smiles or "").strip()
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        if invalid_sink is not None:
            invalid_sink.add(smiles)
        return smiles
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def default_energetics_csv(project_root_path: Path) -> Path:
    """Return the default crawler CSV path used by ILThermo energetics processors."""

    return project_root_path / "data" / "raw" / "ILThermo" / "pure_compound_enthalpy.csv"


def split_cation_anion(raw_chem_text: str | None) -> tuple[str | None, str | None]:
    """Split a paired SMILES field into cation and anion fragments."""

    if not raw_chem_text:
        return None, None
    chem = raw_chem_text.strip()
    if chem.startswith("smiles:"):
        chem = chem[len("smiles:") :]
    parts = [part for part in chem.split(".") if part]
    cation = next((part for part in parts if "+" in part), parts[0] if parts else None)
    anion = next((part for part in parts if "-" in part), parts[1] if len(parts) > 1 else None)
    return cation, anion


def normalize_unit(unit: str | None) -> str | None:
    """Normalize unit text into a lowercase ASCII-friendly canonical form."""

    if unit is None:
        return None
    normalized = unit.strip().lower()
    normalized = normalized.replace("−", "-").replace("·", "*").replace("&#8226;", "*")
    normalized = normalized.replace("<sup>", "^").replace("</sup>", "")
    normalized = normalized.replace("²", "^2").replace("³", "^3")
    return re.sub(r"\s+", "", normalized)


def convert_by_normalized_unit(
    value: float | None,
    unit: str | None,
    *,
    quantity_name: str,
    unit_factors: Mapping[str, float],
) -> float | None:
    """Convert a numeric value by looking up its normalized unit in a factor map."""

    if value is None:
        return None
    normalized_unit = normalize_unit(unit) if unit is not None else None
    if normalized_unit not in unit_factors:
        raise ValueError(f"unsupported {quantity_name} unit: {unit}")
    return value * unit_factors[normalized_unit]


def build_linear_standardizer(
    *,
    quantity_name: str,
    output_unit: str,
    unit_factors: Mapping[str, float],
    value_transform: Callable[[float], float] | None = None,
) -> PropertyStandardizer:
    """Build a property standardizer that applies a unit-factor lookup."""

    def _standardize_property_value(
        _property_name: str | None,
        property_value: float | None,
        property_unit: str | None,
        _cation: str | None,
        _anion: str | None,
    ) -> tuple[float | None, str | None]:
        """Convert a raw property value into the configured output unit."""

        standardized_value = convert_by_normalized_unit(
            property_value,
            property_unit,
            quantity_name=quantity_name,
            unit_factors=unit_factors,
        )
        if standardized_value is None:
            return None, None
        if value_transform is not None:
            standardized_value = value_transform(standardized_value)
        return standardized_value, output_unit

    return _standardize_property_value


def build_converter_standardizer(
    *,
    quantity_name: str,
    output_unit: str,
    converter: Callable[[float | None, str | None], float | None],
    value_transform: Callable[[float], float] | None = None,
) -> PropertyStandardizer:
    """Build a property standardizer around an existing converter function."""

    def _standardize_property_value(
        _property_name: str | None,
        property_value: float | None,
        property_unit: str | None,
        _cation: str | None,
        _anion: str | None,
    ) -> tuple[float | None, str | None]:
        """Convert a raw property value by delegating to the supplied converter."""

        standardized_value = converter(property_value, property_unit)
        if standardized_value is None:
            return None, None
        if value_transform is not None:
            standardized_value = value_transform(standardized_value)
        return standardized_value, output_unit

    return _standardize_property_value


def build_passthrough_standardizer(output_unit: str) -> PropertyStandardizer:
    """Build a property standardizer that forwards the value unchanged."""

    def _standardize_property_value(
        _property_name: str | None,
        property_value: float | None,
        _property_unit: str | None,
        _cation: str | None,
        _anion: str | None,
    ) -> tuple[float | None, str | None]:
        """Return the raw property value with a fixed output unit label."""

        if property_value is None:
            return None, None
        return property_value, output_unit

    return _standardize_property_value


def to_kelvin(value: float | None, unit: str | None) -> float | None:
    """Convert a temperature value to kelvin."""

    if value is None or unit is None:
        return None
    normalized = unit.strip().lower()
    if normalized == "k":
        return value
    if normalized in {"c", "degc", "celsius"}:
        return value + 273.15
    raise ValueError(f"unsupported temperature unit: {unit}")


def to_kpa(value: float | None, unit: str | None) -> float | None:
    """Convert a pressure value to kilopascals."""

    if value is None or unit is None:
        return None
    normalized = unit.strip().lower()
    if normalized == "kpa":
        return value
    if normalized == "pa":
        return value / 1000.0
    if normalized == "mpa":
        return value * 1000.0
    if normalized == "bar":
        return value * 100.0
    if normalized == "atm":
        return value * 101.325
    raise ValueError(f"unsupported pressure unit: {unit}")


def to_g_cm3(value: float | None, unit: str | None) -> float | None:
    """Convert density values to grams per cubic centimeter."""

    if value is None or unit is None:
        return None
    normalized = normalize_unit(unit)
    if normalized in {"g/cm^3", "g/ml", "g*cm^-3"}:
        return value
    if normalized in {"kg/m^3", "kg*m^-3", "g/l"}:
        return value / 1000.0
    raise ValueError(f"unsupported density unit: {unit}")


def to_mhz(value: float | None, unit: str | None) -> float | None:
    """Convert a frequency value to megahertz."""

    if value is None or unit is None:
        return None
    normalized = unit.strip().lower()
    if normalized == "mhz":
        return value
    if normalized == "hz":
        return value / 1e6
    if normalized == "khz":
        return value / 1e3
    if normalized == "ghz":
        return value * 1e3
    raise ValueError(f"unsupported frequency unit: {unit}")


def to_nm(value: float | None, unit: str | None) -> float | None:
    """Convert a wavelength value to nanometers."""

    if value is None or unit is None:
        return None
    normalized = normalize_unit(unit)
    if normalized == "nm":
        return value
    if normalized in {"a", "angstrom", "angs"}:
        return value
    if normalized in {"um", "μm"}:
        return value * 1e3
    if normalized == "m":
        return value * 1e9
    raise ValueError(f"unsupported wavelength unit: {unit}")


def to_log10(value: float | None, quantity_name: str) -> float | None:
    """Apply a base-10 logarithm after validating that the input is positive."""

    if value is None:
        return None
    if value <= 0:
        raise ValueError(f"{quantity_name} must be positive for log10: {value}")
    return math.log10(value)


def calc_pair_molar_mass_g_per_mol(cation: str | None, anion: str | None) -> float:
    """Calculate the combined molar mass for an ionic pair in grams per mole."""

    parts = [part for part in (cation, anion) if part]
    if not parts:
        raise ValueError("missing cation/anion for molar mass")
    mol = Chem.MolFromSmiles(".".join(parts))
    if mol is None:
        raise ValueError("invalid cation/anion smiles for molar mass")
    return Descriptors.MolWt(mol)


def get_spec_by_slug(slug: str) -> ILThermoPropertySpec:
    """Return the ILThermo property spec for a known property slug."""

    if slug not in SPEC_BY_SLUG:
        raise KeyError(f"Unsupported ILThermo property slug: {slug}")
    return SPEC_BY_SLUG[slug]


def get_spec_by_input_filename(input_filename: str) -> ILThermoPropertySpec:
    """Return the ILThermo property spec for a known raw input filename."""

    if input_filename not in SPEC_BY_INPUT:
        raise KeyError(f"Unsupported ILThermo input file: {input_filename}")
    return SPEC_BY_INPUT[input_filename]
