"""Structure raw IL data sources into unit-explicit CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raw_prep import (  # noqa: E402
    canonicalize_smiles,
    disable_rdkit_logs,
    normalize_unit,
    raw_root,
    split_cation_anion,
    structured_root,
    to_g_cm3,
    to_kelvin,
    to_kpa,
    to_mhz,
    to_nm,
)

NUM_PAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PROMPT_PATTERNS = {
    "cation": r"cation\s*\[START_SMILES\](?P<cation>.*?)\[END_SMILES\]",
    "anion": r"anion\s*\[START_SMILES\](?P<anion>.*?)\[END_SMILES\]",
    "solute": r"solute\s*\[START_SMILES\](?P<solute>.*?)\[END_SMILES\]",
    "solvent": r"solvent\s*\[START_SMILES\](?P<solvent>.*?)\[END_SMILES\]",
    "temperature_K": r"temperature\s*\$?(?P<temperature_K>{})\$?K".format(NUM_PAT),
}
SYSTEM_COLUMNS = ["cation", "anion", "solute", "solvent", "smiles"]
CONDITION_COLUMNS = ["temperature_K", "pressure_kPa", "frequency_MHz", "wavelength_nm"]
META_COLUMNS = ["phase", "standard_state_note"]


def to_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def maybe_log10(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return math.log10(value)


def clean_smiles(value: object, invalid_smiles: set[str] | None = None) -> str:
    return canonicalize_smiles(str(value or "").strip(), invalid_smiles)


def split_ion_pair(value: object, invalid_smiles: set[str] | None = None) -> tuple[str, str]:
    cation, anion = split_cation_anion(str(value or ""))
    return (
        canonicalize_smiles(cation or "", invalid_smiles),
        canonicalize_smiles(anion or "", invalid_smiles),
    )


def ordered_frame(rows: list[dict[str, object]], label_columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[*SYSTEM_COLUMNS, *CONDITION_COLUMNS, *META_COLUMNS, *label_columns])
    ordered = [
        column
        for column in [*SYSTEM_COLUMNS, *CONDITION_COLUMNS, *META_COLUMNS, *label_columns]
        if column in df.columns
    ]
    extras = [column for column in df.columns if column not in ordered]
    return df[ordered + extras]


def write_frame(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"saved={output_path} rows={len(df)}")
    return df


def parse_aionopedia_prompt(text: str) -> dict[str, object]:
    normalized = str(text or "").replace("\n", " ").replace("\r", " ")
    parsed: dict[str, object] = {"cation": "", "anion": "", "solute": "", "solvent": "", "temperature_K": None}
    for key, pattern in PROMPT_PATTERNS.items():
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(key).strip()
        if key == "temperature_K":
            parsed[key] = to_float(value)
        else:
            parsed[key] = clean_smiles(value)
    return parsed


AIONOPEDIA_SPECS = {
    "density": ("density_all.csv", "AIonopedia_density_structured.csv", "density_g/cm^3", None),
    "melt": ("melt_all.csv", "AIonopedia_melt_structured.csv", "melting_point_K", None),
    "solvation": ("solvation_all.csv", "AIonopedia_solvation_structured.csv", "solvation_kcal/mol", None),
    "tension": ("tension_all.csv", "AIonopedia_tension_structured.csv", "surface_tension_mN/m", None),
    "transfer": ("transfer_all.csv", "AIonopedia_transfer_structured.csv", "transfer_kcal/mol", None),
    "transfer_organic": (
        "transfer_organic_all.csv",
        "AIonopedia_transfer_organic_structured.csv",
        "transfer_organic_kcal/mol",
        None,
    ),
    "viscosity": ("viscosity_all.csv", "AIonopedia_viscosity_structured.csv", "viscosity_mPa*s_log10", None),
}


def structure_aionopedia_file(input_path: Path, output_path: Path, property_key: str) -> pd.DataFrame:
    _input_name, _output_name, label_column, _converter = AIONOPEDIA_SPECS[property_key]
    df = pd.read_csv(input_path)
    prompt_column = "Prompt" if "Prompt" in df.columns else "prompt"
    rows: list[dict[str, object]] = []
    for _, csv_row in df.iterrows():
        parsed = parse_aionopedia_prompt(str(csv_row[prompt_column]))
        value = to_float(csv_row.get("label"))
        row = {key: value for key, value in parsed.items() if value not in ("", None)}
        row[label_column] = value
        rows.append(row)
    out = ordered_frame(rows, [label_column])
    return write_frame(out, output_path)


AFTER_AIONOPEDIA_SPECS = {
    "density": ("updated_data_density", "after_AIonopedia_density_structured.csv", "density_g/cm^3"),
    "melting_point": (
        "updated_data_melting_point",
        "after_AIonopedia_melting_point_structured.csv",
        "melting_point_K",
    ),
    "part": ("updated_data_part", "after_AIonopedia_part_structured.csv", "partition_log10"),
    "solv": ("updated_data_solv", "after_AIonopedia_solv_structured.csv", "solvation_kcal/mol"),
    "surface_tension": (
        "updated_data_surface_tension",
        "after_AIonopedia_surface_tension_structured.csv",
        "surface_tension_mN/m",
    ),
    "viscosity": ("updated_data_viscosity", "after_AIonopedia_viscosity_structured.csv", "viscosity_mPa*s"),
}


def structure_after_aionopedia_file(input_path: Path, output_path: Path, property_key: str) -> pd.DataFrame:
    _input_name, _output_name, label_column = AFTER_AIONOPEDIA_SPECS[property_key]
    df = pd.read_csv(input_path)
    rows: list[dict[str, object]] = []
    value_source = {
        "density": "density",
        "melting_point": "melting_point",
        "part": "part",
        "solv": "solv",
        "surface_tension": "surface_tension",
        "viscosity": "viscosity",
    }[property_key]
    for _, csv_row in df.iterrows():
        value = to_float(csv_row.get(value_source))
        if property_key == "density":
            value = to_g_cm3(value, "g/L")
        row: dict[str, object] = {
            "cation": clean_smiles(csv_row.get("ion1")),
            "anion": clean_smiles(csv_row.get("ion2")),
            label_column: value,
        }
        if "T" in df.columns:
            row["temperature_K"] = to_float(csv_row.get("T"))
        if "solute" in df.columns:
            row["solute"] = clean_smiles(csv_row.get("solute"))
        rows.append(row)
    out = ordered_frame(rows, [label_column])
    return write_frame(out, output_path)


@dataclass(frozen=True)
class ILBERTSpec:
    input_name: str
    output_name: str
    labels: tuple[tuple[str, str, Callable[[float | None], float | None]], ...]


def identity(value: float | None) -> float | None:
    return value


def kg_m3_to_g_cm3(value: float | None) -> float | None:
    return to_g_cm3(value, "kg/m^3")


ILBERT_SPECS = {
    "EC.csv": ILBERTSpec(
        "EC.csv",
        "ILBERT_EC_structured.csv",
        (("Exp(S/m)", "electrical_conductivity_S/m", identity), ("lnEC", "lnEC_unitless", identity)),
    ),
    "MP.csv": ILBERTSpec("MP.csv", "ILBERT_MP_structured.csv", (("MP_K", "melting_point_K", identity),)),
    "Norm_C.csv": ILBERTSpec(
        "Norm_C.csv",
        "ILBERT_norm_cytotoxicity_structured.csv",
        (("logEC50", "logEC50_unitless", identity), ("EC50", "EC50_unitless", identity)),
    ),
    "Norm_CO2.csv": ILBERTSpec(
        "Norm_CO2.csv",
        "ILBERT_CO2_structured.csv",
        (("x_CO2", "x_CO2_unitless", identity), ("ln(x_CO2)", "ln_x_CO2_unitless", identity)),
    ),
    "Norm_GTT.csv": ILBERTSpec(
        "Norm_GTT.csv",
        "ILBERT_glass_transition_temperature_structured.csv",
        (("Tg/K", "glass_transition_temperature_K", identity),),
    ),
    "Norm_HC.csv": ILBERTSpec(
        "Norm_HC.csv",
        "ILBERT_HC_structured.csv",
        (("hc", "hc_unitless", identity), ("lnhc", "lnhc_unitless", identity)),
    ),
    "Norm_TDT.csv": ILBERTSpec(
        "Norm_TDT.csv",
        "ILBERT_thermal_decomposition_temperature_structured.csv",
        (("Td/K", "thermal_decomposition_temperature_K", identity),),
    ),
    "Norm_surface.csv": ILBERTSpec(
        "Norm_surface.csv",
        "ILBERT_surface_tension_structured.csv",
        (("s_mNm", "surface_tension_mN/m", identity),),
    ),
    "RR.csv": ILBERTSpec("RR.csv", "ILBERT_refractive_index_structured.csv", (("R", "refractive_index_unitless", identity),)),
    "TC_Norm.csv": ILBERTSpec(
        "TC_Norm.csv",
        "ILBERT_thermal_conductivity_structured.csv",
        (("TC/W m-1 K-1", "thermal_conductivity_W/m/K", identity),),
    ),
    "density_P.csv": ILBERTSpec(
        "density_P.csv",
        "ILBERT_density_structured.csv",
        (("d_kg m-3", "density_g/cm^3", kg_m3_to_g_cm3),),
    ),
    "viscosity_P.csv": ILBERTSpec(
        "viscosity_P.csv",
        "ILBERT_viscosity_structured.csv",
        (("n_mPas(o)", "viscosity_mPa*s", identity), ("ln(n_mPas)", "ln_viscosity_mPa*s_unitless", identity)),
    ),
}


def extract_ilbert_ions(csv_row: pd.Series, invalid_smiles: set[str]) -> tuple[str, str]:
    if "cation SMILES" in csv_row.index and "anion SMILES" in csv_row.index:
        cation = clean_smiles(csv_row.get("cation SMILES"), invalid_smiles)
        anion = clean_smiles(csv_row.get("anion SMILES"), invalid_smiles)
        if cation or anion:
            return cation, anion
    for column in ("Normalized SMILES", "Normalized SMILES1", "IL SMILES", "SMILES", "smiles"):
        if column in csv_row.index and str(csv_row.get(column) or "").strip():
            return split_ion_pair(csv_row.get(column), invalid_smiles)
    return "", ""


def structure_ilbert_file(input_path: Path, output_path: Path, input_name: str) -> pd.DataFrame:
    spec = ILBERT_SPECS[input_name]
    df = pd.read_csv(input_path)
    rows: list[dict[str, object]] = []
    invalid_smiles: set[str] = set()
    label_columns = [target for _source, target, _convert in spec.labels]
    for _, csv_row in df.iterrows():
        cation, anion = extract_ilbert_ions(csv_row, invalid_smiles)
        row: dict[str, object] = {"cation": cation, "anion": anion}
        if "T/K" in df.columns:
            row["temperature_K"] = to_float(csv_row.get("T/K"))
        if "P/bar" in df.columns:
            row["pressure_kPa"] = to_kpa(to_float(csv_row.get("P/bar")), "bar")
        elif "P/MPa" in df.columns:
            row["pressure_kPa"] = to_kpa(to_float(csv_row.get("P/MPa")), "MPa")
        for source, target, convert in spec.labels:
            row[target] = convert(to_float(csv_row.get(source)))
        rows.append(row)
    out = ordered_frame(rows, label_columns)
    if invalid_smiles:
        print(f"[{input_name}] invalid_smiles_count={len(invalid_smiles)}")
    return write_frame(out, output_path)


@dataclass(frozen=True)
class ILThermoSpec:
    slug: str
    input_name: str
    output_name: str
    label_column: str
    property_pattern: str
    unit: str
    transform: Callable[[float | None, str | None], float | None]
    csv_backed: bool = False


def by_unit(unit_factors: dict[str, float]) -> Callable[[float | None, str | None], float | None]:
    def convert(value: float | None, unit: str | None) -> float | None:
        if value is None:
            return None
        normalized = normalize_unit(unit)
        if normalized not in unit_factors:
            raise ValueError(f"unsupported unit: {unit}")
        return value * unit_factors[normalized]

    return convert


def density_transform(value: float | None, unit: str | None) -> float | None:
    return to_g_cm3(value, unit)


ILTHERMO_SPECS = {
    "density": ILThermoSpec(
        "density",
        "ilt_density_data.txt",
        "ilt_density_structured.csv",
        "density_g/cm^3",
        r"(?:Specific density|Mass density|Density)",
        "g/cm^3",
        density_transform,
    ),
    "electrical_conductivity": ILThermoSpec(
        "electrical_conductivity",
        "ilt_electrical_conductivity_data.txt",
        "ilt_electrical_conductivity_structured.csv",
        "electrical_conductivity_S/m_log10",
        r"Electrical conductivity",
        "S/m",
        lambda value, unit: maybe_log10(by_unit({"s/m": 1.0, "ms/cm": 0.1})(value, unit)),
    ),
    "enthalpy": ILThermoSpec(
        "enthalpy",
        "pure_compound_enthalpy.csv",
        "ilt_enthalpy_structured.csv",
        "enthalpy_kJ/mol",
        r"Enthalpy(?:<SUP>\*</SUP>)?",
        "kJ/mol",
        by_unit({"kj/mol": 1.0, "j/mol": 0.001}),
        True,
    ),
    "enthalpy_of_transition_or_fusion": ILThermoSpec(
        "enthalpy_of_transition_or_fusion",
        "pure_compound_enthalpy_of_transition_or_fusion.csv",
        "ilt_enthalpy_of_transition_or_fusion_structured.csv",
        "enthalpy_of_transition_or_fusion_kJ/mol",
        r"Enthalpy of transition or fusion",
        "kJ/mol",
        by_unit({"kj/mol": 1.0, "j/mol": 0.001}),
        True,
    ),
    "enthalpy_of_vaporization_or_sublimation": ILThermoSpec(
        "enthalpy_of_vaporization_or_sublimation",
        "pure_compound_enthalpy_of_vaporization_or_sublimation.csv",
        "ilt_enthalpy_of_vaporization_or_sublimation_structured.csv",
        "enthalpy_of_vaporization_or_sublimation_kJ/mol",
        r"Enthalpy of vaporization or sublimation",
        "kJ/mol",
        by_unit({"kj/mol": 1.0, "j/mol": 0.001}),
        True,
    ),
    "entropy": ILThermoSpec(
        "entropy",
        "pure_compound_entropy.csv",
        "ilt_entropy_structured.csv",
        "entropy_J/mol/K",
        r"Entropy(?:<SUP>\*</SUP>)?",
        "J/mol/K",
        by_unit({"j/k/mol": 1.0, "j/mol/k": 1.0, "kj/k/mol": 1000.0}),
        True,
    ),
    "equilibrium_pressure": ILThermoSpec(
        "equilibrium_pressure",
        "ilt_equilibrium_pressure_data.txt",
        "ilt_equilibrium_pressure_structured.csv",
        "pressure_kPa_log10",
        r"Equilibrium pressure",
        "kPa",
        lambda value, unit: maybe_log10(to_kpa(value, unit)),
    ),
    "equilibrium_temperature": ILThermoSpec(
        "equilibrium_temperature",
        "ilt_equilibrium_temperature_data.txt",
        "ilt_equilibrium_temperature_structured.csv",
        "equilibrium_temperature_K",
        "Equilibrium temperature",
        "K",
        to_kelvin,
    ),
    "heat_capacity_at_constant_pressure": ILThermoSpec(
        "heat_capacity_at_constant_pressure",
        "ilt_heat_capacity_at_constant_pressure_data.txt",
        "ilt_heat_capacity_at_constant_pressure_structured.csv",
        "heat_capacity_J/mol/K",
        r"Heat capacity at constant pressure",
        "J/mol/K",
        by_unit({"j/k/mol": 1.0, "j/mol/k": 1.0, "kj/k/mol": 1000.0}),
    ),
    "heat_capacity_at_vapor_saturation_pressure": ILThermoSpec(
        "heat_capacity_at_vapor_saturation_pressure",
        "ilt_heat_capacity_at_vapor_saturation_pressure_data.txt",
        "ilt_heat_capacity_at_vapor_saturation_pressure_structured.csv",
        "heat_capacity_at_vapor_saturation_pressure_J/mol/K",
        r"Heat capacity at vapor saturation pressure",
        "J/mol/K",
        by_unit({"j/k/mol": 1.0, "j/mol/k": 1.0, "kj/k/mol": 1000.0}),
    ),
    "isobaric_coefficient_of_volume_expansion": ILThermoSpec(
        "isobaric_coefficient_of_volume_expansion",
        "ilt_isobaric_coefficient_of_volume_expansion_data.txt",
        "ilt_isobaric_coefficient_of_volume_expansion_structured.csv",
        "isobaric_coefficient_of_volume_expansion_K^-1",
        r"Isobaric coefficient of volume expansion",
        "K^-1",
        by_unit({"k^-1": 1.0, "1/k": 1.0}),
    ),
    "normal_melting_temperature": ILThermoSpec(
        "normal_melting_temperature",
        "ilt_normal_melting_temperature_data.txt",
        "ilt_normal_melting_temperature_structured.csv",
        "melting_point_K",
        r"Normal melting temperature",
        "K",
        to_kelvin,
    ),
    "refractive_index": ILThermoSpec(
        "refractive_index",
        "ilt_refractive_index_data.txt",
        "ilt_refractive_index_structured.csv",
        "refractive_index_unitless",
        r"Refractive index(?: \([^)]+\))?",
        "unitless",
        lambda value, _unit: value,
    ),
    "relative_permittivity": ILThermoSpec(
        "relative_permittivity",
        "ilt_relative_permittivity_data.txt",
        "ilt_relative_permittivity_structured.csv",
        "relative_permittivity_unitless",
        r"Relative permittivity(?: at zero frequency)?",
        "unitless",
        lambda value, _unit: value,
    ),
    "self_diffusion_coefficient": ILThermoSpec(
        "self_diffusion_coefficient",
        "ilt_self_diffusion_coefficient_data.txt",
        "ilt_self_diffusion_coefficient_structured.csv",
        "self_diffusion_coefficient_10^-9*m^2/s",
        r"Self diffusion coefficient",
        "10^-9*m^2/s",
        by_unit({"10^-9*m^2/s": 1.0, "m^2/s": 1e9}),
    ),
    "speed_of_sound": ILThermoSpec(
        "speed_of_sound",
        "ilt_speed_of_sound_data.txt",
        "ilt_speed_of_sound_structured.csv",
        "speed_of_sound_m/s",
        r"Speed of sound",
        "m/s",
        by_unit({"m/s": 1.0}),
    ),
    "surface_tension_liquid_gas": ILThermoSpec(
        "surface_tension_liquid_gas",
        "ilt_surface_tension_liquid-gas_data.txt",
        "ilt_surface_tension_liquid_gas_structured.csv",
        "surface_tension_mN/m",
        r"Surface tension liquid-gas",
        "mN/m",
        by_unit({"n/m": 1000.0, "mn/m": 1.0}),
    ),
    "thermal_conductivity": ILThermoSpec(
        "thermal_conductivity",
        "ilt_thermal_conductivity_data.txt",
        "ilt_thermal_conductivity_structured.csv",
        "thermal_conductivity_W/m/K",
        r"Thermal conductivity",
        "W/m/K",
        by_unit({"w/m/k": 1.0, "w/(m*k)": 1.0}),
    ),
    "thermal_diffusivity": ILThermoSpec(
        "thermal_diffusivity",
        "ilt_thermal_diffusivity_data.txt",
        "ilt_thermal_diffusivity_structured.csv",
        "thermal_diffusivity_m^2/s",
        r"Thermal diffusivity",
        "m^2/s",
        by_unit({"m^2/s": 1.0}),
    ),
    "viscosity": ILThermoSpec(
        "viscosity",
        "ilt_viscosity_data.txt",
        "ilt_viscosity_structured.csv",
        "viscosity_mPa*s_log10",
        r"(?:Dynamic viscosity|Kinematic viscosity|Viscosity)",
        "mPa*s",
        lambda value, unit: maybe_log10(by_unit({"pa*s": 1000.0, "mpa*s": 1.0, "m^2/s": 1.0})(value, unit)),
    ),
}


SMILES_RE = re.compile(r"^\s*smiles\s*:\s*(\S+)", flags=re.IGNORECASE)
CONTEXT_PATTERNS = {
    "temperature_K": (
        re.compile(r"(?<!equilibrium\s)(?<!melting\s)Temperature,\s*([^:=>,]+?)\s*(?:=>\s*Liquid)?\s*:\s*({})".format(NUM_PAT), re.I),
        to_kelvin,
    ),
    "pressure_kPa": (
        re.compile(
            r"(?<!equilibrium\s)(?<!constant\s)(?<!saturation\s)\bPressure,\s*([^:=>,]+?)\s*(?:=>\s*Liquid)?\s*:\s*({})".format(
                NUM_PAT
            ),
            re.I,
        ),
        to_kpa,
    ),
    "frequency_MHz": (
        re.compile(r"Frequency,\s*([^:=>,]+?)\s*(?:=>\s*Liquid)?\s*:\s*({})".format(NUM_PAT), re.I),
        to_mhz,
    ),
    "wavelength_nm": (
        re.compile(r"Wavelength,\s*([^:=>,]+?)\s*(?:=>\s*Liquid)?\s*:\s*({})".format(NUM_PAT), re.I),
        to_nm,
    ),
}


def parse_phase(text: str) -> str:
    match = re.search(r"=>\s*([^:]+?)\s*:", text)
    return match.group(1).strip() if match else ""


def parse_ilthermo_text_line(line: str, spec: ILThermoSpec, invalid_smiles: set[str]) -> dict[str, object]:
    row: dict[str, object] = {"phase": parse_phase(line)}
    smiles_match = SMILES_RE.search(line)
    if smiles_match:
        cation, anion = split_ion_pair(smiles_match.group(1), invalid_smiles)
        row["cation"] = cation
        row["anion"] = anion
    for output_name, (pattern, converter) in CONTEXT_PATTERNS.items():
        match = pattern.search(line)
        if match:
            row[output_name] = converter(float(match.group(2)), match.group(1).strip())
    prop_re = re.compile(r"({})(?:,\s*([^:]+?))?\s*(?:=>\s*([^:]+?))?\s*:\s*({})".format(spec.property_pattern, NUM_PAT), re.I)
    for match in prop_re.finditer(line):
        matched_property = match.group(1).strip()
        if matched_property.lower().startswith("error of"):
            continue
        unit = match.group(2).strip() if match.group(2) else spec.unit
        if match.group(3):
            row["phase"] = match.group(3).strip()
        row[spec.label_column] = spec.transform(float(match.group(4)), unit)
        break
    return row


def structure_ilthermo_csv(input_path: Path, output_path: Path, spec: ILThermoSpec) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    invalid_smiles: set[str] = set()
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for csv_row in reader:
            cation, anion = split_ion_pair(csv_row.get("canonical_smiles"), invalid_smiles)
            value = spec.transform(to_float(csv_row.get("property_value")), csv_row.get("property_unit") or spec.unit)
            row = {
                "cation": cation,
                "anion": anion,
                "temperature_K": to_kelvin(to_float(csv_row.get("temperature_value")), csv_row.get("temperature_unit")),
                "pressure_kPa": to_kpa(to_float(csv_row.get("pressure_value")), csv_row.get("pressure_unit")),
                "frequency_MHz": to_mhz(to_float(csv_row.get("frequency_value")), csv_row.get("frequency_unit")),
                "wavelength_nm": to_nm(to_float(csv_row.get("wavelength_value")), csv_row.get("wavelength_unit")),
                "phase": str(csv_row.get("phase") or "").strip(),
                spec.label_column: value,
            }
            standard_state_note = str(csv_row.get("standard_state_note") or "").strip()
            if standard_state_note:
                row["standard_state_note"] = standard_state_note
            rows.append(row)
    out = ordered_frame(rows, [spec.label_column]).drop_duplicates().reset_index(drop=True)
    if invalid_smiles:
        print(f"[{input_path.name}] invalid_smiles_count={len(invalid_smiles)}")
    return write_frame(out, output_path)


def structure_ilthermo_file(input_path: Path, output_path: Path, property_slug: str) -> pd.DataFrame:
    spec = ILTHERMO_SPECS[property_slug]
    if spec.csv_backed:
        return structure_ilthermo_csv(input_path, output_path, spec)

    invalid_smiles: set[str] = set()
    raw_lines = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [parse_ilthermo_text_line(line, spec, invalid_smiles) for line in raw_lines]
    out = ordered_frame(rows, [spec.label_column]).drop_duplicates().reset_index(drop=True)
    if invalid_smiles:
        print(f"[{input_path.name}] invalid_smiles_count={len(invalid_smiles)}")
    print(f"[{input_path.name}] raw={len(raw_lines)} final_rows={len(out)}")
    return write_frame(out, output_path)


def structure_aionopedia(input_dir: Path, output_dir: Path) -> None:
    for key, (input_name, output_name, _label, _converter) in AIONOPEDIA_SPECS.items():
        structure_aionopedia_file(input_dir / input_name, output_dir / output_name, key)


def structure_after_aionopedia(input_dir: Path, output_dir: Path) -> None:
    for key, (input_name, output_name, _label) in AFTER_AIONOPEDIA_SPECS.items():
        structure_after_aionopedia_file(input_dir / input_name, output_dir / output_name, key)


def structure_ilbert(input_dir: Path, output_dir: Path) -> None:
    for input_name, spec in ILBERT_SPECS.items():
        structure_ilbert_file(input_dir / input_name, output_dir / spec.output_name, input_name)


def structure_ilthermo(input_dir: Path, output_dir: Path) -> None:
    for slug, spec in ILTHERMO_SPECS.items():
        structure_ilthermo_file(input_dir / spec.input_name, output_dir / spec.output_name, slug)


def canonicalize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = df[column].map(clean_smiles)
    return df


def numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = df[column].map(to_float)
    return df


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def structure_simulation_density(input_dir: Path, output_dir: Path) -> pd.DataFrame | None:
    input_path = input_dir / "density_260501.csv"
    if not input_path.exists():
        return None
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["cation", "anion"])
    df = numeric_columns(df, ["temperature", "density", "density_err"])
    df = df.rename(columns={"temperature": "temperature_K"})
    return write_frame(df[["cation", "anion", "temperature_K", "density", "density_err"]], output_dir / "simulated_density_structured.csv")


def structure_simulation_heat_capacity(input_dir: Path, output_dir: Path) -> pd.DataFrame | None:
    input_path = input_dir / "heat_capacity_260501.csv"
    if not input_path.exists():
        return None
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["cation", "anion"])
    df = numeric_columns(df, ["temperature", "Cp", "Cp_err"])
    df = df.rename(columns={"temperature": "temperature_K"})
    return write_frame(
        df[["cation", "anion", "temperature_K", "Cp", "Cp_err"]],
        output_dir / "simulated_heat_capacity_structured.csv",
    )


def structure_simulation_thermal_expansion(input_dir: Path, output_dir: Path) -> pd.DataFrame | None:
    input_path = input_dir / "thermal_expansion_260501.csv"
    if not input_path.exists():
        return None
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["cation", "anion"])
    df = numeric_columns(df, ["temperature", "alpha", "alpha_err"])
    df = df.rename(columns={"temperature": "temperature_K"})
    return write_frame(
        df[["cation", "anion", "temperature_K", "alpha", "alpha_err"]],
        output_dir / "simulated_thermal_expansion_structured.csv",
    )


def structure_simulation_heat_of_vaporization(input_dir: Path, output_dir: Path) -> pd.DataFrame | None:
    input_path = input_dir / "heat_of_vaporization_260603.csv"
    if not input_path.exists():
        return None
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["cation", "anion"])
    df = numeric_columns(df, ["temperature", "Hvap", "Hvap_err"])
    df = df.rename(columns={"temperature": "temperature_K"})
    return write_frame(
        df[["cation", "anion", "temperature_K", "Hvap", "Hvap_err"]],
        output_dir / "simulated_heat_of_vaporization_structured.csv",
    )


def structure_simulation_pbe_tzvp_anions(input_dir: Path, output_dir: Path) -> pd.DataFrame | None:
    input_path = input_dir / "PBE_TZVP_anions_260103.csv"
    if not input_path.exists():
        return None
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["SMILES"])
    df = numeric_columns(df, ["HOMO", "LUMO", "gap"])
    df["anion"] = df["SMILES"]
    remove_if_exists(output_dir / "simulated_PBE_TZVP_anions_structured.csv")
    return write_frame(
        df[["anion", "HOMO", "LUMO", "gap"]],
        output_dir / "simulated_HOMO+LUMO_PBE_TZVP_anions_structured.csv",
    )


def structure_simulation_pbe_tzvp_cations(input_dir: Path, output_dir: Path) -> pd.DataFrame | None:
    input_path = input_dir / "PBE_TZVP_cations_260103.csv"
    if not input_path.exists():
        return None
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["SMILES"])
    df = numeric_columns(df, ["HOMO", "LUMO", "gap"])
    df["cation"] = df["SMILES"]
    remove_if_exists(output_dir / "simulated_PBE_TZVP_cations_structured.csv")
    return write_frame(
        df[["cation", "HOMO", "LUMO", "gap"]],
        output_dir / "simulated_HOMO+LUMO_PBE_TZVP_cations_structured.csv",
    )


def structure_simulation_qm_elec_hf(input_dir: Path, output_dir: Path) -> pd.DataFrame | None:
    input_path = input_dir / "QM_elec_HF.csv"
    if not input_path.exists():
        return None
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["SMILES"])
    numeric = [column for column in df.columns if column != "SMILES"]
    df = numeric_columns(df, numeric)
    return write_frame(df[["SMILES", *numeric]], output_dir / "simulated_QM_elec_HF_structured.csv")


def structure_simulation_combi_qm_solv(input_dir: Path, output_dir: Path) -> pd.DataFrame | None:
    input_path = input_dir / "combi_qm_solv.csv"
    if not input_path.exists():
        return None
    df = pd.read_csv(input_path)
    df = canonicalize_columns(df, ["solvent", "solute"])
    df = numeric_columns(df, ["solv"])
    return write_frame(df[["solvent", "solute", "solv"]], output_dir / "simulated_combi_qm_solv_structured.csv")


def structure_simulation_box_mapping(input_dir: Path, output_dir: Path) -> pd.DataFrame | None:
    input_path = input_dir / "box_20260514" / "mapping.csv"
    if not input_path.exists():
        return None
    df = pd.read_csv(input_path)
    df = df.rename(columns={"cation_smiles": "cation", "anion_smiles": "anion", "temperature": "temperature_K"})
    df = canonicalize_columns(df, ["cation", "anion"])
    df = numeric_columns(df, ["temperature_K"])
    return write_frame(
        df[["mol_id", "cation", "anion", "temperature_K"]],
        output_dir / "simulated_box_20260514_mapping_structured.csv",
    )


def structure_simulation_charge_mapping(input_dir: Path, output_dir: Path) -> pd.DataFrame | None:
    input_path = input_dir / "charge_20260514" / "mapping.csv"
    if not input_path.exists():
        return None
    df = pd.read_csv(input_path)
    df = df.rename(columns={"smiles": "SMILES"})
    df = canonicalize_columns(df, ["SMILES"])
    df = numeric_columns(df, ["charge"])
    return write_frame(
        df[["mol_id", "SMILES", "charge"]],
        output_dir / "simulated_charge_20260514_mapping_structured.csv",
    )


SIMULATION_PROCESSORS: tuple[Callable[[Path, Path], pd.DataFrame | None], ...] = (
    structure_simulation_density,
    structure_simulation_heat_capacity,
    structure_simulation_thermal_expansion,
    structure_simulation_heat_of_vaporization,
    structure_simulation_pbe_tzvp_anions,
    structure_simulation_pbe_tzvp_cations,
    structure_simulation_qm_elec_hf,
    structure_simulation_combi_qm_solv,
    structure_simulation_box_mapping,
    structure_simulation_charge_mapping,
)


def structure_simulation(input_dir: Path, output_dir: Path) -> None:
    for processor in SIMULATION_PROCESSORS:
        processor(input_dir, output_dir)


@dataclass(frozen=True)
class SourceSpec:
    raw_subdir: str
    output_subdir: str
    runner: Callable[[Path, Path], None]


SOURCE_RUNNERS = {
    "AIonopedia": SourceSpec("AIonopedia", "AIonopedia", structure_aionopedia),
    "ILBERT": SourceSpec("ILBERT", "ILBERT", structure_ilbert),
    "ILThermo": SourceSpec("ILThermo", "ILThermo", structure_ilthermo),
    "after_AIonopedia": SourceSpec("after_AIonopedia", "after_AIonopedia", structure_after_aionopedia),
    "simulation": SourceSpec("simulation_data", "simulation", structure_simulation),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structure raw IL data sources into unit-explicit CSV outputs.")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(SOURCE_RUNNERS),
        default=list(SOURCE_RUNNERS),
        help="Raw source directories to structure.",
    )
    parser.add_argument("--raw-root", type=Path, default=raw_root(PROJECT_ROOT))
    parser.add_argument("--output-root", type=Path, default=structured_root(PROJECT_ROOT))
    return parser.parse_args()


def main() -> None:
    disable_rdkit_logs()
    args = parse_args()
    for source in args.sources:
        print(f"processing={source}")
        spec = SOURCE_RUNNERS[source]
        spec.runner(args.raw_root / spec.raw_subdir, args.output_root / spec.output_subdir)


if __name__ == "__main__":
    main()
