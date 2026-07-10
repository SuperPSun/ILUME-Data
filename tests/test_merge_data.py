from pathlib import Path

import pandas as pd
import pytest

from scripts.merge_data import merge_data, property_slug


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_property_slug_uses_stable_filename_rules():
    assert property_slug("density_g/cm^3") == "density"
    assert property_slug("viscosity_mPa*s_log10") == "viscosity"
    assert property_slug("heat_capacity_J/mol/K") == "heat_capacity"
    assert property_slug("pEC50") == "pec50"
    assert property_slug("partition_log10") == "partition"
    assert property_slug("HOMO_eV") == "homo"
    assert property_slug("x_CO2_unitless") == "x_co2"
    assert property_slug("simulated_QM_elec_HF") == "simulated_qm_elec_hf"


def test_experiment_sources_merge_by_property_and_aggregate_identical_records(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    duplicate_row = {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "temperature_K": 298.15,
        "density_g/cm^3": 1.2,
    }
    write_csv(input_root / "AIonopedia" / "AIonopedia_density_structured.csv", [duplicate_row])
    write_csv(input_root / "ILBERT" / "ILBERT_density_structured.csv", [duplicate_row])
    write_csv(input_root / "ILThermo" / "ilt_density_structured.csv", [{**duplicate_row, "phase": "Liquid"}])
    write_csv(input_root / "after_AIonopedia" / "after_AIonopedia_density_structured.csv", [{**duplicate_row, "density_g/cm^3": 1.3}])

    results = merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "density.csv")
    assert len(out) == 2
    assert list(out.columns) == ["cation", "anion", "temperature_K", "phase", "density_g/cm^3", "source_list"]
    aggregated = out[out["density_g/cm^3"] == 1.2].iloc[0]
    assert aggregated["phase"] == "Liquid"
    assert aggregated["source_list"] == "AIonopedia; ILBERT; ILThermo"
    assert set(out["density_g/cm^3"]) == {1.2, 1.3}
    assert not (output_root / "AIonopedia").exists()
    assert not (output_root / "ILBERT").exists()
    assert not (output_root / "ILThermo").exists()
    assert not (output_root / "after_AIonopedia").exists()
    assert (output_root / "experiment").is_dir()
    assert (output_root / "simulation").is_dir()
    assert any(result["bucket"] == "experiment" and result["property_label"] == "density_g/cm^3" for result in results)


def test_refractive_index_missing_wavelength_defaults_to_sodium_d_line(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    base = {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "temperature_K": 298.15,
        "refractive_index_unitless": 1.4,
    }
    write_csv(input_root / "ILBERT" / "ILBERT_refractive_index_structured.csv", [base])
    write_csv(
        input_root / "ILThermo" / "ilt_refractive_index_structured.csv",
        [{**base, "wavelength_nm": 450.0, "refractive_index_unitless": 1.5}],
    )

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "refractive_index.csv")
    assert "wavelength_nm" in out.columns
    assert set(out["wavelength_nm"]) == {450.0, 589.0}
    assert out["wavelength_nm"].isna().sum() == 0


def test_simulation_stays_separate_from_experiment_for_same_property(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    row = {"cation": "CC[n+]1ccn(C)c1", "anion": "F[B-](F)(F)F", "temperature_K": 298.15, "density_g/cm^3": 1.2}
    write_csv(input_root / "ILBERT" / "ILBERT_density_structured.csv", [row])
    write_csv(input_root / "simulation" / "simulated_density_structured.csv", [{**row, "density_err_g/cm^3": 0.01}])

    merge_data(input_root, output_root)

    experiment = pd.read_csv(output_root / "experiment" / "density.csv")
    simulation = pd.read_csv(output_root / "simulation" / "density.csv")
    assert len(experiment) == 1
    assert len(simulation) == 1
    assert "density_err_g/cm^3" not in simulation.columns
    assert not (output_root / "simulation" / "density_err.csv").exists()


def test_multi_label_simulation_file_is_split_and_error_labels_are_dropped(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    write_csv(
        input_root / "simulation" / "simulated_HOMO+LUMO_PBE_TZVP_anions_structured.csv",
        [{"anion": "F[B-](F)(F)F", "HOMO_eV": -1.0, "LUMO_eV": 2.0, "gap_eV": 3.0}],
    )
    write_csv(
        input_root / "simulation" / "simulated_heat_capacity_structured.csv",
        [
            {
                "cation": "CC[n+]1ccn(C)c1",
                "anion": "F[B-](F)(F)F",
                "temperature_K": 298.15,
                "heat_capacity_J/mol/K": 500.0,
                "heat_capacity_err_J/mol/K": 5.0,
            }
        ],
    )

    merge_data(input_root, output_root)

    assert (output_root / "simulation" / "homo.csv").exists()
    assert (output_root / "simulation" / "lumo.csv").exists()
    assert (output_root / "simulation" / "gap.csv").exists()
    heat_capacity = pd.read_csv(output_root / "simulation" / "heat_capacity.csv")
    assert list(heat_capacity.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "heat_capacity_J/mol/K",
        "source_list",
    ]
    assert not (output_root / "simulation" / "heat_capacity_err.csv").exists()


def test_qm_elec_hf_file_outputs_one_wide_table_instead_of_split_labels(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    write_csv(
        input_root / "simulation" / "simulated_QM_elec_HF_structured.csv",
        [
            {"SMILES": "CCO", "ESP_max": 1.0, "ESP_min": -1.0, "gap_eV": 2.0},
            {"SMILES": "CCN", "ESP_max": None, "ESP_min": -2.0, "gap_eV": 3.0},
        ],
    )

    merge_data(input_root, output_root)

    wide = pd.read_csv(output_root / "simulation" / "simulated_qm_elec_hf.csv")
    assert list(wide.columns) == ["SMILES", "ESP_max", "ESP_min", "gap_eV", "source_list"]
    assert len(wide) == 2
    assert wide["ESP_max"].isna().sum() == 1
    assert not (output_root / "simulation" / "esp_max.csv").exists()
    assert not (output_root / "simulation" / "esp_min.csv").exists()
    assert not (output_root / "simulation" / "gap.csv").exists()

    manifest = pd.read_csv(output_root / "merged_manifest.csv")
    assert manifest.iloc[0]["property_label"] == "simulated_QM_elec_HF"
    assert manifest.iloc[0]["output_file"] == "simulation/simulated_qm_elec_hf.csv"


def test_same_key_different_label_values_remain_separate_rows(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    base = {"cation": "CC[n+]1ccn(C)c1", "anion": "F[B-](F)(F)F", "temperature_K": 298.15}
    write_csv(input_root / "AIonopedia" / "AIonopedia_viscosity_structured.csv", [{**base, "viscosity_mPa*s_log10": 1.0}])
    write_csv(input_root / "ILBERT" / "ILBERT_viscosity_structured.csv", [{**base, "viscosity_mPa*s_log10": 2.0}])

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "viscosity.csv")
    assert len(out) == 2
    assert set(out["viscosity_mPa*s_log10"]) == {1.0, 2.0}
    assert set(out["source_list"]) == {"AIonopedia", "ILBERT"}


def test_condition_subset_row_collapses_into_more_complete_record(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    partial = {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "temperature_K": 298.15,
        "density_g/cm^3": 1.2,
    }
    complete = {**partial, "pressure_kPa": 100.0}
    write_csv(input_root / "AIonopedia" / "AIonopedia_density_structured.csv", [partial])
    write_csv(input_root / "ILBERT" / "ILBERT_density_structured.csv", [complete])

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "density.csv")
    assert len(out) == 1
    assert out.iloc[0]["pressure_kPa"] == 100.0
    assert out.iloc[0]["source_list"] == "AIonopedia; ILBERT"


def test_condition_subset_does_not_collapse_different_label_values(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    base = {"cation": "CC[n+]1ccn(C)c1", "anion": "F[B-](F)(F)F", "temperature_K": 298.15}
    write_csv(input_root / "AIonopedia" / "AIonopedia_density_structured.csv", [{**base, "density_g/cm^3": 1.2}])
    write_csv(
        input_root / "ILBERT" / "ILBERT_density_structured.csv",
        [{**base, "pressure_kPa": 100.0, "density_g/cm^3": 1.3}],
    )

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "density.csv")
    assert len(out) == 2
    assert set(out["density_g/cm^3"]) == {1.2, 1.3}


def test_ambiguous_condition_subset_candidates_are_preserved(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    partial = {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "temperature_K": 298.15,
        "density_g/cm^3": 1.2,
    }
    write_csv(input_root / "AIonopedia" / "AIonopedia_density_structured.csv", [partial])
    write_csv(
        input_root / "ILBERT" / "ILBERT_density_structured.csv",
        [
            {**partial, "pressure_kPa": 100.0},
            {**partial, "pressure_kPa": 200.0},
        ],
    )

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "density.csv")
    assert len(out) == 3
    assert out["pressure_kPa"].isna().sum() == 1


def test_complementary_condition_rows_are_preserved(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    base = {
        "cation": "CC[n+]1ccn(C)c1",
        "anion": "F[B-](F)(F)F",
        "temperature_K": 298.15,
        "density_g/cm^3": 1.2,
    }
    write_csv(input_root / "AIonopedia" / "AIonopedia_density_structured.csv", [{**base, "phase": "Liquid"}])
    write_csv(input_root / "ILBERT" / "ILBERT_density_structured.csv", [{**base, "pressure_kPa": 100.0}])

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "density.csv")
    assert len(out) == 2
    assert set(out["source_list"]) == {"AIonopedia", "ILBERT"}


def test_standard_state_note_is_metadata_not_property(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    reference_note = (
        "* property value is given as the difference ( x - x ref ) from the reference state: "
        "Crystal at the temperature of 298.15 K and the pressure of 101.3 kPa"
    )
    write_csv(
        input_root / "ILThermo" / "ilt_enthalpy_structured.csv",
        [
            {
                "cation": "CCCC[n+]1ccccc1",
                "anion": "F[B-](F)(F)F",
                "temperature_K": 290.0,
                "pressure_kPa": 101.325,
                "standard_state_note": reference_note,
                "enthalpy_kJ/mol": -4.329,
            }
        ],
    )

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "enthalpy.csv")
    assert list(out.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "pressure_kPa",
        "standard_state_note",
        "enthalpy_kJ/mol",
        "source_list",
    ]
    assert out.loc[0, "standard_state_note"] == reference_note
    assert out.loc[0, "enthalpy_kJ/mol"] == -4.329
    assert not (output_root / "experiment" / "standard_state_note.csv").exists()


def test_ilthermo_equilibrium_pressure_uses_specific_output_filename(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    write_csv(
        input_root / "ILThermo" / "ilt_equilibrium_pressure_structured.csv",
        [
            {
                "cation": "CC[n+]1ccn(C)c1",
                "anion": "F[B-](F)(F)F",
                "temperature_K": 298.15,
                "pressure_kPa_log10": 2.0,
            }
        ],
    )

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "equilibrium_pressure.csv")
    assert list(out.columns) == ["cation", "anion", "temperature_K", "pressure_kPa_log10", "source_list"]
    assert out.loc[0, "pressure_kPa_log10"] == 2.0
    assert not (output_root / "experiment" / "pressure.csv").exists()
    manifest = pd.read_csv(output_root / "merged_manifest.csv")
    assert manifest.iloc[0]["property_label"] == "pressure_kPa_log10"
    assert manifest.iloc[0]["output_file"] == "experiment/equilibrium_pressure.csv"


def test_self_diffusion_log_label_keeps_stable_output_filename(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    write_csv(
        input_root / "ILThermo" / "ilt_self_diffusion_coefficient_structured.csv",
        [
            {
                "cation": "CC[n+]1ccn(C)c1",
                "anion": "F[B-](F)(F)F",
                "temperature_K": 298.15,
                "self_diffusion_coefficient_10^-9*m^2/s_log10": -2.0,
            }
        ],
    )

    merge_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "self_diffusion_coefficient.csv")
    assert list(out.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "self_diffusion_coefficient_10^-9*m^2/s_log10",
        "source_list",
    ]
    assert out.loc[0, "self_diffusion_coefficient_10^-9*m^2/s_log10"] == -2.0
    manifest = pd.read_csv(output_root / "merged_manifest.csv")
    assert manifest.iloc[0]["property_label"] == "self_diffusion_coefficient_10^-9*m^2/s_log10"
    assert manifest.iloc[0]["output_file"] == "experiment/self_diffusion_coefficient.csv"


def test_manifest_records_output_files_and_counts(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    stale_system_counts = output_root / "system_property_counts.csv"
    stale_system_counts.parent.mkdir(parents=True, exist_ok=True)
    stale_system_counts.write_text("stale\n")
    write_csv(
        input_root / "ILBERT" / "ILBERT_HC_structured.csv",
        [{"cation": "CC[n+]1ccn(C)c1", "anion": "F[B-](F)(F)F", "temperature_K": 298.15, "heat_capacity_J/mol/K": 500.0}],
    )

    merge_data(input_root, output_root)

    manifest = pd.read_csv(output_root / "merged_manifest.csv")
    assert list(manifest.columns) == ["bucket", "property_label", "output_file", "input_files", "input_rows", "output_rows"]
    row = manifest.iloc[0]
    assert row["bucket"] == "experiment"
    assert row["property_label"] == "heat_capacity_J/mol/K"
    assert row["output_file"] == "experiment/heat_capacity.csv"
    assert row["input_files"] == "ILBERT/ILBERT_HC_structured.csv"
    assert row["input_rows"] == 1
    assert row["output_rows"] == 1
    assert not stale_system_counts.exists()


def test_property_filename_collisions_raise_clear_error(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "merged"
    row = {"cation": "CC[n+]1ccn(C)c1", "anion": "F[B-](F)(F)F"}
    write_csv(input_root / "ILBERT" / "a.csv", [{**row, "temperature_K": 298.15, "density_g/cm^3": 1.2}])
    write_csv(input_root / "ILBERT" / "b.csv", [{**row, "temperature_K": 298.15, "density_kg/m^3": 1200.0}])

    with pytest.raises(ValueError, match="experiment/density.csv"):
        merge_data(input_root, output_root)
