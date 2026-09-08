from pathlib import Path

import pandas as pd

from raw_prep import (
    canonicalize_smiles,
    net_formal_charge,
    parse_ion_pair_identity,
    reconcile_smiles_charge,
    split_cation_anion,
    to_g_cm3,
    to_kpa,
)
from scripts.structure_raw_data import (
    parse_aionopedia_prompt,
    split_ion_pair,
    structure_after_aionopedia_file,
    structure_aionopedia_file,
    structure_ilbert_file,
    structure_ilthermo_file,
    structure_simulation,
    to_float,
)


def test_split_cation_anion_and_canonicalize_smiles():
    cation, anion = split_cation_anion("smiles:CC[n+]1ccn(C)c1.F[B-](F)(F)F")

    assert canonicalize_smiles(cation) == "CC[n+]1ccn(C)c1"
    assert canonicalize_smiles(anion) == "F[B-](F)(F)F"


def test_split_cation_anion_uses_fragment_net_charge_and_preserves_stoichiometry():
    cation, anion = split_cation_anion("O=[N+]([O-])[O-].CC[n+]1ccn(C)c1")

    assert canonicalize_smiles(cation) == "CC[n+]1ccn(C)c1"
    assert canonicalize_smiles(anion) == "O=[N+]([O-])[O-]"

    cation, anion = split_cation_anion("[Cl-].[Ca+2].[Cl-]")

    assert canonicalize_smiles(cation) == "[Ca+2]"
    assert canonicalize_smiles(anion) == "[Cl-].[Cl-]"


def test_parse_ion_pair_identity_reports_ambiguous_or_invalid_pairs():
    assert parse_ion_pair_identity("CC[n+]1ccn(C)c1.[Cl-].O").error == "neutral_fragment"
    assert parse_ion_pair_identity("CC[n+]1ccn(C)c1.[Ca+2]").error == "missing_charge_role"
    assert parse_ion_pair_identity("CC[n+]1ccn(C)c1.not-a-smiles").error == "invalid_smiles"


def test_parse_ion_pair_identity_keeps_charge_imbalanced_ionic_roles():
    identity = parse_ion_pair_identity("CC[n+]1ccn(C)c1.[Cl-].[Cl-]")

    assert identity.error is None
    assert identity.cation == "CC[n+]1ccn(C)c1"
    assert identity.anion == "[Cl-].[Cl-]"


def test_reconcile_smiles_charge_uses_deterministic_proton_transfers():
    carboxylate = reconcile_smiles_charge("O=C(O)CC(=O)O", -2)
    diammonium = reconcile_smiles_charge("NCCN", 2)
    repeated = reconcile_smiles_charge("O=C(O)CC(=O)O", -2)

    assert carboxylate.status == "repaired"
    assert net_formal_charge(carboxylate.corrected_smiles) == -2
    assert carboxylate.operations == repeated.operations
    assert carboxylate.corrected_smiles == repeated.corrected_smiles
    assert diammonium.status == "repaired"
    assert net_formal_charge(diammonium.corrected_smiles) == 2
    assert len(carboxylate.operations) == 2
    assert len(diammonium.operations) == 2


def test_reconcile_smiles_charge_repairs_phosphate_and_heteroatom_sites():
    phospholipid = (
        "CCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCCN)"
        "OC(=O)CCCCCCCCCCCCCCCCC"
    )
    phosphate = reconcile_smiles_charge(phospholipid, -1)
    amine = reconcile_smiles_charge("CCN", 1)
    aromatic_nh = reconcile_smiles_charge("c1nn[nH]n1", -1)

    assert phosphate.status == "repaired"
    assert "P(=O)([O-])" in phosphate.corrected_smiles
    assert net_formal_charge(phosphate.corrected_smiles) == -1
    assert amine.corrected_smiles == "CC[NH3+]"
    assert net_formal_charge(aromatic_nh.corrected_smiles) == -1


def test_reconcile_smiles_charge_rejects_invalid_or_unrepairable_states():
    quaternary = reconcile_smiles_charge("Cc1cc(C)[n+](C)n1-c1ccccc1", 0)

    assert quaternary.status == "unrepairable_charge_state"
    assert quaternary.corrected_smiles is None
    assert reconcile_smiles_charge("not-a-smiles", 0).status == "invalid_smiles"
    assert reconcile_smiles_charge("CCO", -0.5).status == "invalid_target_charge"
    assert reconcile_smiles_charge("CCO", None).status == "invalid_target_charge"


def test_to_float_preserves_zero_and_rejects_only_missing_or_invalid_values():
    assert to_float(0) == 0.0
    assert to_float(0.0) == 0.0
    assert to_float(-0.0) == -0.0
    assert to_float("0") == 0.0
    assert to_float(None) is None
    assert to_float(float("nan")) is None
    assert to_float("") is None
    assert to_float("not-a-number") is None


def test_pressure_and_density_unit_conversions():
    assert to_kpa(1, "bar") == 100
    assert to_kpa(0.1, "MPa") == 100
    assert to_kpa(101325, "Pa") == 101.325
    assert to_g_cm3(1343, "g/L") == 1.343


def test_parse_aionopedia_prompt():
    parsed = parse_aionopedia_prompt(
        "solute [START_SMILES]C(=O)=O[END_SMILES] temperature $298.15$K "
        "cation [START_SMILES]CC[n+]1ccn(C)c1[END_SMILES] "
        "anion [START_SMILES]F[B-](F)(F)F[END_SMILES]"
    )

    assert parsed == {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "solute": "O=C=O",
        "solvent": "",
        "temperature_K": 298.15,
    }


def test_structure_aionopedia_density(tmp_path: Path):
    input_path = tmp_path / "density_all.csv"
    output_path = tmp_path / "density_structured.csv"
    pd.DataFrame(
        {
            "Prompt": [
                "temperature $348$K cation [START_SMILES]CCCC[N+]1(C)CCCC1[END_SMILES] "
                "anion [START_SMILES]F[B-](F)(F)F[END_SMILES]"
            ],
            "label": ["1.252"],
        }
    ).to_csv(input_path, index=False)

    df = structure_aionopedia_file(input_path, output_path, "density")

    assert list(df.columns) == ["cation", "anion", "temperature_K", "density_g/cm^3"]
    assert df.loc[0, "density_g/cm^3"] == 1.252
    assert output_path.exists()


def test_structure_after_aionopedia_density_converts_g_l(tmp_path: Path):
    input_path = tmp_path / "updated_data_density"
    output_path = tmp_path / "density_structured.csv"
    pd.DataFrame(
        {
            "ion1": ["CCn1cc[n+](C)c1"],
            "ion2": ["CS(=O)(=O)[N-]S(C)(=O)=O"],
            "T": [298.2],
            "density": [1343.0],
            "property": ["density"],
        }
    ).to_csv(input_path, index=False)

    df = structure_after_aionopedia_file(input_path, output_path, "density")

    assert list(df.columns) == ["cation", "anion", "temperature_K", "density_g/cm^3"]
    assert df.loc[0, "density_g/cm^3"] == 1.343


def test_after_aionopedia_part_uses_transfer_label(tmp_path: Path):
    input_path = tmp_path / "updated_data_part"
    output_path = tmp_path / "part_structured.csv"
    pd.DataFrame(
        {
            "ion1": ["COn1cc[n+](OC)c1"],
            "ion2": ["O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F"],
            "solute": ["C1COCCO1"],
            "T": [298.15],
            "part": [-0.15961],
            "property": ["part"],
        }
    ).to_csv(input_path, index=False)

    df = structure_after_aionopedia_file(input_path, output_path, "part")

    assert list(df.columns) == ["cation", "anion", "solute", "temperature_K", "transfer_kcal/mol"]
    assert df.loc[0, "transfer_kcal/mol"] == -0.15961


def test_aionopedia_solvation_and_transfer_use_kcal_mol_labels(tmp_path: Path):
    prompt = (
        "solute [START_SMILES]C(=O)=O[END_SMILES] temperature $298.15$K "
        "cation [START_SMILES]CC[n+]1ccn(C)c1[END_SMILES] "
        "anion [START_SMILES]F[B-](F)(F)F[END_SMILES]"
    )
    solvation_input = tmp_path / "solvation_all.csv"
    transfer_input = tmp_path / "transfer_all.csv"
    pd.DataFrame({"Prompt": [prompt], "label": [-4.887]}).to_csv(solvation_input, index=False)
    pd.DataFrame({"Prompt": [prompt], "label": [1.25]}).to_csv(transfer_input, index=False)

    solvation = structure_aionopedia_file(solvation_input, tmp_path / "solvation.csv", "solvation")
    transfer = structure_aionopedia_file(transfer_input, tmp_path / "transfer.csv", "transfer")

    assert "solvation_kcal/mol" in solvation.columns
    assert solvation.loc[0, "solvation_kcal/mol"] == -4.887
    assert "transfer_kcal/mol" in transfer.columns
    assert transfer.loc[0, "transfer_kcal/mol"] == 1.25


def test_ilbert_normalized_smiles_split_and_pressure_conversion(tmp_path: Path):
    input_path = tmp_path / "density_P.csv"
    output_path = tmp_path / "density_P_structured.csv"
    pd.DataFrame(
        {
            "IL": [1],
            "IL SMILES": ["CC[n+]1ccn(C)c1.F[B-](F)(F)F"],
            "T/K": [298.15],
            "d_kg m-3": [1252.0],
            "P/bar": [1.0],
            "Normalized SMILES": ["CC[n+]1ccn(C)c1.F[B-](F)(F)F"],
            "RANDOM": [1],
        }
    ).to_csv(input_path, index=False)

    df = structure_ilbert_file(input_path, output_path, "density_P.csv")

    assert list(df.columns) == ["cation", "anion", "temperature_K", "pressure_kPa", "density_g/cm^3"]
    assert split_ion_pair("CC[n+]1ccn(C)c1.F[B-](F)(F)F") == ("CC[n+]1ccn(C)c1", "F[B-](F)(F)F")
    assert df.loc[0, "pressure_kPa"] == 100
    assert df.loc[0, "density_g/cm^3"] == 1.252


def test_ilbert_co2_keeps_mole_fraction_and_log_scale_labels(tmp_path: Path):
    input_path = tmp_path / "Norm_CO2.csv"
    output_path = tmp_path / "co2_structured.csv"
    pd.DataFrame(
        {
            "cation SMILES": ["CC[n+]1ccn(C)c1"],
            "anion SMILES": ["F[B-](F)(F)F"],
            "T/K": [298.16],
            "P/bar": [36.2],
            "x_CO2": [0.655],
            "ln(x_CO2)": [-0.42312],
        }
    ).to_csv(input_path, index=False)

    df = structure_ilbert_file(input_path, output_path, "Norm_CO2.csv")

    assert list(df.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "pressure_kPa",
        "x_CO2_unitless",
        "ln_x_CO2_unitless",
    ]
    assert df.loc[0, "x_CO2_unitless"] == 0.655
    assert df.loc[0, "ln_x_CO2_unitless"] == -0.42312


def test_ilthermo_single_line_outputs_specific_label(tmp_path: Path):
    input_path = tmp_path / "ilt_density_data.txt"
    output_path = tmp_path / "ilt_density_structured.csv"
    input_path.write_text(
        "smiles:CC[n+]1ccn(C)c1.F[B-](F)(F)F Temperature, K:298.15 "
        "Pressure, kPa:101.325 Specific density, kg/m<SUP>3</SUP> => Liquid:1252.0 "
        "Error of specific density, kg/m<SUP>3</SUP> => Liquid:0.1\n",
        encoding="utf-8",
    )

    df = structure_ilthermo_file(input_path, output_path, "density")

    assert "label" not in df.columns
    assert "standard_unit" not in df.columns
    assert "property_name" not in df.columns
    assert "note" not in df.columns
    assert "source_text" not in df.columns
    assert list(df.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "pressure_kPa",
        "phase",
        "density_g/cm^3",
    ]
    assert df.loc[0, "density_g/cm^3"] == 1.252


def test_ilthermo_self_diffusion_is_log10_after_unit_conversion(tmp_path: Path):
    input_path = tmp_path / "ilt_self_diffusion_coefficient_data.txt"
    output_path = tmp_path / "ilt_self_diffusion_coefficient_structured.csv"
    input_path.write_text(
        "smiles:CC[n+]1ccn(C)c1.F[B-](F)(F)F Temperature, K:298.15 "
        "Self diffusion coefficient, 10^-9*m^2/s => Liquid:0.01\n"
        "smiles:CCC[n+]1ccn(C)c1.F[B-](F)(F)F Temperature, K:298.15 "
        "Self diffusion coefficient, m^2/s => Liquid:1e-11\n",
        encoding="utf-8",
    )

    df = structure_ilthermo_file(input_path, output_path, "self_diffusion_coefficient")

    assert list(df.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "phase",
        "self_diffusion_coefficient_10^-9*m^2/s_log10",
    ]
    assert set(df["self_diffusion_coefficient_10^-9*m^2/s_log10"]) == {-2.0}


def test_ilthermo_csv_preserves_standard_state_note(tmp_path: Path):
    input_path = tmp_path / "pure_compound_enthalpy.csv"
    output_path = tmp_path / "ilt_enthalpy_structured.csv"
    reference_note = (
        "* property value is given as the difference ( x - x ref ) from the reference state: "
        "Crystal at the temperature of 298.15 K and the pressure of 101.3 kPa"
    )
    pd.DataFrame(
        {
            "canonical_smiles": ["CCCC[n+]1ccccc1.F[B-](F)(F)F"],
            "temperature_value": [290.0],
            "temperature_unit": ["K"],
            "pressure_value": [101.325],
            "pressure_unit": ["kPa"],
            "frequency_value": [None],
            "frequency_unit": [None],
            "wavelength_value": [None],
            "wavelength_unit": [None],
            "phase": ["Liquid"],
            "property_value": [-4.329],
            "property_unit": ["kJ/mol"],
            "standard_state_note": [reference_note],
            "starred": [True],
            "note_text": [reference_note],
        }
    ).to_csv(input_path, index=False)

    df = structure_ilthermo_file(input_path, output_path, "enthalpy")

    assert list(df.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "pressure_kPa",
        "frequency_MHz",
        "wavelength_nm",
        "phase",
        "standard_state_note",
        "enthalpy_kJ/mol",
    ]
    assert df.loc[0, "standard_state_note"] == reference_note
    assert df.loc[0, "enthalpy_kJ/mol"] == -4.329
    assert "starred" not in df.columns
    assert "note_text" not in df.columns


def test_structure_simulation_pbe_tzvp_renames_outputs_and_drops_smiles(tmp_path: Path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "structured"
    input_dir.mkdir()
    pd.DataFrame({"SMILES": ["C[N-]C"], "HOMO": [-0.1], "LUMO": [0.2], "gap": [0.3]}).to_csv(
        input_dir / "PBE_TZVP_anions_260103.csv", index=False
    )
    pd.DataFrame({"SMILES": ["C[NH3+]"], "HOMO": [-9.5], "LUMO": [-4.3], "gap": [5.2]}).to_csv(
        input_dir / "PBE_TZVP_cations_260103.csv", index=False
    )
    (output_dir / "simulated_PBE_TZVP_anions_structured.csv").parent.mkdir(parents=True)
    (output_dir / "simulated_PBE_TZVP_anions_structured.csv").write_text("stale\n", encoding="utf-8")
    (output_dir / "simulated_PBE_TZVP_cations_structured.csv").write_text("stale\n", encoding="utf-8")

    structure_simulation(input_dir, output_dir)

    anions = pd.read_csv(output_dir / "simulated_HOMO+LUMO_PBE_TZVP_anions_structured.csv")
    cations = pd.read_csv(output_dir / "simulated_HOMO+LUMO_PBE_TZVP_cations_structured.csv")
    assert list(anions.columns) == ["anion", "HOMO", "LUMO", "gap"]
    assert list(cations.columns) == ["cation", "HOMO", "LUMO", "gap"]
    assert anions.loc[0, "anion"] == "C[N-]C"
    assert cations.loc[0, "cation"] == "C[NH3+]"
    assert not (output_dir / "simulated_PBE_TZVP_anions_structured.csv").exists()
    assert not (output_dir / "simulated_PBE_TZVP_cations_structured.csv").exists()


def test_structure_simulation_uses_260905_property_sources(tmp_path: Path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "structured"
    input_dir.mkdir()
    identity = {
        "cation": ["CC[n+]1ccn(C)c1"],
        "anion": ["F[B-](F)(F)F"],
        "temperature": [313],
    }
    sources = {
        "density_260905.csv": ("density", 1250.0, "density_err", 0.5),
        "heat_capacity_260905.csv": ("Cp", 510.0, "Cp_err", 2.0),
        "thermal_expansion_260905.csv": ("alpha", 0.0005, "alpha_err", 0.00001),
        "heat_of_vaporization_260905.csv": ("Hvap", 150.0, "Hvap_err", 0.3),
    }
    for filename, (label, value, error_label, error_value) in sources.items():
        pd.DataFrame(
            {**identity, label: [value], error_label: [error_value]}
        ).to_csv(input_dir / filename, index=False)

    structure_simulation(input_dir, output_dir)

    assert pd.read_csv(
        output_dir / "simulated_density_structured.csv"
    ).loc[0, "density"] == 1250.0
    assert pd.read_csv(
        output_dir / "simulated_heat_capacity_structured.csv"
    ).loc[0, "Cp"] == 510.0
    assert pd.read_csv(
        output_dir / "simulated_thermal_expansion_structured.csv"
    ).loc[0, "alpha"] == 0.0005
    assert pd.read_csv(
        output_dir / "simulated_heat_of_vaporization_structured.csv"
    ).loc[0, "Hvap"] == 150.0


def test_structure_simulation_mappings_preserve_mol_id(tmp_path: Path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "structured"
    (input_dir / "box_20260514").mkdir(parents=True)
    (input_dir / "charge_20260514").mkdir(parents=True)
    pd.DataFrame(
        {
            "mol_id": ["mol_0000001"],
            "cation_smiles": ["CC[n+]1ccn(C)c1"],
            "anion_smiles": ["F[B-](F)(F)F"],
            "temperature": [293],
        }
    ).to_csv(input_dir / "box_20260514" / "mapping.csv", index=False)
    pd.DataFrame({"mol_id": ["mol_0000002"], "smiles": ["CCO"], "charge": [0]}).to_csv(
        input_dir / "charge_20260514" / "mapping.csv", index=False
    )

    structure_simulation(input_dir, output_dir)

    box = pd.read_csv(output_dir / "simulated_box_20260514_mapping_structured.csv")
    charge = pd.read_csv(output_dir / "simulated_charge_20260514_mapping_structured.csv")
    assert list(box.columns) == ["mol_id", "cation", "anion", "temperature_K"]
    assert list(charge.columns) == ["mol_id", "SMILES", "charge"]
    assert box.loc[0, "mol_id"] == "mol_0000001"
    assert charge.loc[0, "mol_id"] == "mol_0000002"
    assert charge.loc[0, "charge"] == 0


def test_structure_cleaned_ilthermo_file_renames_label_and_drops_intermediate_columns(tmp_path: Path):
    from scripts.structure_cleaned_ilthermo import structure_cleaned_ilthermo_file

    input_path = tmp_path / "ilt_density_structured.csv"
    output_path = tmp_path / "out" / "ilt_density_structured.csv"
    pd.DataFrame(
        {
            "cation": ["C(C)[N+](C)(C)C"],
            "anion": ["CC([O-])=O"],
            "temperature_K": [298.15],
            "pressure_kPa": [101.325],
            "frequency_MHz": [None],
            "wavelength_nm": [None],
            "property_name": ["Specific density"],
            "property_unit": ["kg/m<SUP>3</SUP>"],
            "property_value": [1252.0],
            "label": [1.252],
            "standard unit": ["g/cm^3"],
            "parse_error": [None],
            "source_text": [
                "smiles:CC[n+]1ccn(C)c1.F[B-](F)(F)F Temperature, K:298.15 "
                "Pressure, kPa:101.325 Specific density, kg/m<SUP>3</SUP> => Liquid:1252.0"
            ],
        }
    ).to_csv(input_path, index=False)

    df = structure_cleaned_ilthermo_file(input_path, output_path, "density")

    assert list(df.columns) == ["cation", "anion", "temperature_K", "pressure_kPa", "density_g/cm^3"]
    assert df.loc[0, "cation"] == "CC[n+]1ccn(C)c1"
    assert df.loc[0, "anion"] == "F[B-](F)(F)F"
    assert df.loc[0, "density_g/cm^3"] == 1.252


def test_structure_cleaned_ilthermo_repairs_ions_from_source_text_and_audits_failures(tmp_path: Path):
    from scripts.structure_cleaned_ilthermo import structure_cleaned_ilthermo_file

    input_path = tmp_path / "ilt_density_structured.csv"
    output_path = tmp_path / "out" / "ilt_density_structured.csv"
    rejected_path = tmp_path / "rejected.csv"
    pd.DataFrame(
        {
            "cation": ["O=[N+]([O-])[O-]", "CC[n+]1ccn(C)c1"],
            "anion": ["O=[N+]([O-])[O-]", "[Cl-]"],
            "label": [1.1, 1.2],
            "source_text": [
                "smiles:O=[N+]([O-])[O-].CC[n+]1ccn(C)c1 Specific density, kg/m3:1100",
                "smiles:CC[n+]1ccn(C)c1.[Cl-].O Specific density, kg/m3:1200",
            ],
        }
    ).to_csv(input_path, index=False)

    df = structure_cleaned_ilthermo_file(
        input_path,
        output_path,
        "density",
        rejected_path=rejected_path,
    )

    assert len(df) == 1
    assert df.loc[0, "cation"] == "CC[n+]1ccn(C)c1"
    assert df.loc[0, "anion"] == "O=[N+]([O-])[O-]"
    rejected = pd.read_csv(rejected_path)
    assert rejected.loc[0, "rejection_reason"] == "neutral_fragment"
    for column in ["property_name", "property_unit", "property_value", "standard unit", "parse_error", "source_text"]:
        assert column not in df.columns


def test_structure_cleaned_ilthermo_file_log_transforms_self_diffusion(tmp_path: Path):
    from scripts.structure_cleaned_ilthermo import structure_cleaned_ilthermo_file

    input_path = tmp_path / "ilt_self_diffusion_coefficient_structured.csv"
    output_path = tmp_path / "out" / "ilt_self_diffusion_coefficient_structured.csv"
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1"],
            "anion": ["F[B-](F)(F)F"],
            "temperature_K": [298.15],
            "pressure_kPa": [101.325],
            "label": [0.01],
            "standard unit": ["10^-9*m^2/s"],
        }
    ).to_csv(input_path, index=False)

    df = structure_cleaned_ilthermo_file(input_path, output_path, "self_diffusion_coefficient")

    assert list(df.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "pressure_kPa",
        "self_diffusion_coefficient_10^-9*m^2/s_log10",
    ]
    assert df.loc[0, "self_diffusion_coefficient_10^-9*m^2/s_log10"] == -2.0


def test_structure_cleaned_ilthermo_applies_revised_hard_thresholds(tmp_path: Path):
    from scripts.structure_cleaned_ilthermo import structure_cleaned_ilthermo_file

    input_path = tmp_path / "ilt_density_structured.csv"
    output_path = tmp_path / "out" / "ilt_density_structured.csv"
    rejected_path = tmp_path / "rejected.csv"
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1", "CCC[n+]1ccn(C)c1"],
            "anion": ["F[B-](F)(F)F", "F[B-](F)(F)F"],
            "label": [5.0, 0.0],
        }
    ).to_csv(input_path, index=False)

    df = structure_cleaned_ilthermo_file(
        input_path,
        output_path,
        "density",
        rejected_path=rejected_path,
    )

    assert df["density_g/cm^3"].tolist() == [5.0]
    rejected = pd.read_csv(rejected_path)
    assert rejected.loc[0, "rejection_reason"] == "hard_threshold"
    assert rejected.loc[0, "trigger_column"] == "density_g/cm^3"


def test_structure_cleaned_ilthermo_file_extracts_energetics_phase_from_note_and_source_text(tmp_path: Path):
    from scripts.structure_cleaned_ilthermo import structure_cleaned_ilthermo_file

    input_path = tmp_path / "ilt_enthalpy_structured.csv"
    output_path = tmp_path / "out" / "ilt_enthalpy_structured.csv"
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1"],
            "anion": ["F[B-](F)(F)F"],
            "label": [220.0],
            "standard_unit": ["kJ/mol"],
            "parse_error": [None],
            "note": ["Phase: Crystal | Gas"],
            "source_text": [
                "record_id=1 | property_name=Enthalpy | property_value=220 | property_unit=kJ/mol | "
                "phase=Crystal | Liquid"
            ],
        }
    ).to_csv(input_path, index=False)

    df = structure_cleaned_ilthermo_file(input_path, output_path, "enthalpy_of_transition_or_fusion")

    assert list(df.columns) == ["cation", "anion", "phase", "enthalpy_of_transition_or_fusion_kJ/mol"]
    assert df.loc[0, "phase"] == "Crystal | Liquid"
    assert df.loc[0, "enthalpy_of_transition_or_fusion_kJ/mol"] == 220.0


def test_structure_cleaned_ilthermo_file_preserves_standard_state_note_without_liquid_phase(tmp_path: Path):
    from scripts.structure_cleaned_ilthermo import structure_cleaned_ilthermo_file

    input_path = tmp_path / "ilt_entropy_structured.csv"
    output_path = tmp_path / "out" / "ilt_entropy_structured.csv"
    reference_note = (
        "* property value is given as the difference ( x - x ref ) from the reference state: "
        "Crystal at the temperature of 0 K and the pressure of 0 kPa"
    )
    pd.DataFrame(
        {
            "cation": ["CCCC[n+]1ccn(C)c1"],
            "anion": ["F[P-](F)(F)(F)(F)F"],
            "temperature_K": [190.6],
            "pressure_kPa": [101.325],
            "phase": ["Liquid"],
            "label": [324.4],
            "standard_state_note": [reference_note],
        }
    ).to_csv(input_path, index=False)

    df = structure_cleaned_ilthermo_file(input_path, output_path, "entropy")

    assert list(df.columns) == ["cation", "anion", "temperature_K", "pressure_kPa", "standard_state_note", "entropy_J/mol/K"]
    assert df.loc[0, "standard_state_note"] == reference_note
    assert df.loc[0, "entropy_J/mol/K"] == 324.4


def test_structure_cleaned_ilthermo_file_requires_label_column(tmp_path: Path):
    from scripts.structure_cleaned_ilthermo import structure_cleaned_ilthermo_file

    input_path = tmp_path / "ilt_density_structured.csv"
    output_path = tmp_path / "out.csv"
    pd.DataFrame({"cation": ["CC[n+]1ccn(C)c1"], "anion": ["F[B-](F)(F)F"]}).to_csv(input_path, index=False)

    try:
        structure_cleaned_ilthermo_file(input_path, output_path, "density")
    except ValueError as exc:
        assert "missing required label column" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing label column")


def test_structure_cleaned_ilthermo_file_drops_frequency_conditioned_electrical_conductivity(tmp_path: Path):
    from scripts.structure_cleaned_ilthermo import structure_cleaned_ilthermo_file

    input_path = tmp_path / "ilt_electrical_conductivity_structured.csv"
    output_path = tmp_path / "out" / "ilt_electrical_conductivity_structured.csv"
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1", "CC[n+]1ccn(C)c1"],
            "anion": ["F[B-](F)(F)F", "F[B-](F)(F)F"],
            "temperature_K": [298.15, 298.15],
            "frequency_MHz": [None, 0.001],
            "label": [2.160168, 1.5],
            "standard unit": ["S/m (log scale)", "S/m (log scale)"],
        }
    ).to_csv(input_path, index=False)

    df = structure_cleaned_ilthermo_file(input_path, output_path, "electrical_conductivity")

    assert list(df.columns) == ["cation", "anion", "temperature_K", "electrical_conductivity_S/m_log10"]
    assert len(df) == 1
    assert df.loc[0, "electrical_conductivity_S/m_log10"] == 2.160168


def test_structure_cleaned_ilthermo_file_drops_speed_of_sound_frequency_condition(tmp_path: Path):
    from scripts.structure_cleaned_ilthermo import structure_cleaned_ilthermo_file

    input_path = tmp_path / "ilt_speed_of_sound_structured.csv"
    output_path = tmp_path / "out" / "ilt_speed_of_sound_structured.csv"
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1", "CC[n+]1ccn(C)c1"],
            "anion": ["F[B-](F)(F)F", "F[B-](F)(F)F"],
            "temperature_K": [298.15, 308.15],
            "frequency_MHz": [0.001, None],
            "label": [1200.0, 1180.0],
            "standard unit": ["m/s", "m/s"],
        }
    ).to_csv(input_path, index=False)

    df = structure_cleaned_ilthermo_file(input_path, output_path, "speed_of_sound")

    assert list(df.columns) == ["cation", "anion", "temperature_K", "speed_of_sound_m/s"]
    assert len(df) == 2
    assert df["speed_of_sound_m/s"].tolist() == [1200.0, 1180.0]


def test_structure_cleaned_ilthermo_splits_static_and_dynamic_relative_permittivity(tmp_path: Path):
    from scripts.structure_cleaned_ilthermo import structure_cleaned_ilthermo

    input_dir = tmp_path / "cleaned" / "ilt"
    output_dir = tmp_path / "cleaned" / "ILThermo"
    input_dir.mkdir(parents=True)
    legacy_path = output_dir / "ilt_relative_permittivity_structured.csv"
    output_dir.mkdir(parents=True)
    pd.DataFrame({"legacy": [True]}).to_csv(legacy_path, index=False)
    pd.DataFrame(
        {
            "cation": [
                "C[N+](C)(C)C",
                "CC[N+](C)(C)C",
                "CCC[N+](C)(C)C",
                "CCCC[N+](C)(C)C",
            ],
            "anion": ["[Cl-]", "[Br-]", "[F-]", "[I-]"],
            "temperature_K": [298.15, 298.15, 298.15, 298.15],
            "pressure_kPa": [101.325, 101.325, 101.325, 101.325],
            "frequency_MHz": [0.0, None, 10.0, 100.0],
            "label": [20.0, 21.0, 22.0, 23.0],
        }
    ).to_csv(input_dir / "ilt_relative_permittivity_structured.csv", index=False)

    structure_cleaned_ilthermo(input_dir, output_dir)

    static = pd.read_csv(output_dir / "ilt_static_relative_permittivity_structured.csv")
    dynamic = pd.read_csv(output_dir / "ilt_dynamic_relative_permittivity_structured.csv")
    assert list(static.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "pressure_kPa",
        "static_relative_permittivity_unitless",
    ]
    assert static["static_relative_permittivity_unitless"].tolist() == [20.0, 21.0]
    assert "frequency_MHz" not in static.columns
    assert list(dynamic.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "pressure_kPa",
        "frequency_MHz",
        "dynamic_relative_permittivity_unitless",
    ]
    assert dynamic["frequency_MHz"].tolist() == [10.0, 100.0]
    assert dynamic["dynamic_relative_permittivity_unitless"].tolist() == [22.0, 23.0]
    assert not legacy_path.exists()


def test_structure_cleaned_ilthermo_directory_cli(tmp_path: Path):
    import subprocess
    import sys

    input_dir = tmp_path / "cleaned" / "ilt"
    output_dir = tmp_path / "cleaned" / "ILThermo"
    input_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1"],
            "anion": ["F[B-](F)(F)F"],
            "label": [2.160168],
            "standard unit": ["S/m (log scale)"],
            "source_text": ["smiles:CC[n+]1ccn(C)c1.F[B-](F)(F)F Electrical conductivity, S/m => Liquid:144.6"],
        }
    ).to_csv(input_dir / "ilt_electrical_conductivity_structured.csv", index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/structure_cleaned_ilthermo.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    df = pd.read_csv(output_dir / "ilt_electrical_conductivity_structured.csv")
    assert list(df.columns) == ["cation", "anion", "electrical_conductivity_S/m_log10"]
    assert df.loc[0, "electrical_conductivity_S/m_log10"] == 2.160168
