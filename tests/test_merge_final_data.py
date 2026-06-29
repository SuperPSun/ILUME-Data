from pathlib import Path

import pandas as pd

from scripts.merge_final_data import merge_final_data, property_slug


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_property_slug_uses_stable_filename_rules():
    assert property_slug("density_g/cm^3") == "density_g_per_cm_pow_3"
    assert property_slug("viscosity_mPa*s_log10") == "viscosity_mpa_s_log10"
    assert property_slug("heat_capacity_J/mol/K") == "heat_capacity_j_per_mol_per_k"
    assert property_slug("pEC50") == "pec50"


def test_experiment_sources_merge_by_property_and_aggregate_identical_records(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "final"
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

    results = merge_final_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "density_g_per_cm_pow_3.csv")
    assert len(out) == 3
    aggregated = out[out["source_record_count"] == 2].iloc[0]
    assert aggregated["source_list"] == "AIonopedia; ILBERT"
    assert aggregated["source_file_list"] == "AIonopedia_density_structured.csv; ILBERT_density_structured.csv"
    assert set(out["density_g/cm^3"]) == {1.2, 1.3}
    assert not (output_root / "AIonopedia").exists()
    assert not (output_root / "ILBERT").exists()
    assert not (output_root / "ILThermo").exists()
    assert not (output_root / "after_AIonopedia").exists()
    assert (output_root / "experiment").is_dir()
    assert (output_root / "simulation").is_dir()
    assert any(result["bucket"] == "experiment" and result["property_label"] == "density_g/cm^3" for result in results)


def test_simulation_stays_separate_from_experiment_for_same_property(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "final"
    row = {"cation": "CC[n+]1ccn(C)c1", "anion": "F[B-](F)(F)F", "temperature_K": 298.15, "density_g/cm^3": 1.2}
    write_csv(input_root / "ILBERT" / "ILBERT_density_structured.csv", [row])
    write_csv(input_root / "simulation" / "simulated_density_structured.csv", [{**row, "density_err_g/cm^3": 0.01}])

    merge_final_data(input_root, output_root)

    experiment = pd.read_csv(output_root / "experiment" / "density_g_per_cm_pow_3.csv")
    simulation = pd.read_csv(output_root / "simulation" / "density_g_per_cm_pow_3.csv")
    assert len(experiment) == 1
    assert len(simulation) == 1
    assert "density_err_g/cm^3" not in simulation.columns
    assert not (output_root / "simulation" / "density_err_g_per_cm_pow_3.csv").exists()


def test_multi_label_simulation_file_is_split_and_error_labels_are_dropped(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "final"
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

    merge_final_data(input_root, output_root)

    assert (output_root / "simulation" / "homo_ev.csv").exists()
    assert (output_root / "simulation" / "lumo_ev.csv").exists()
    assert (output_root / "simulation" / "gap_ev.csv").exists()
    heat_capacity = pd.read_csv(output_root / "simulation" / "heat_capacity_j_per_mol_per_k.csv")
    assert list(heat_capacity.columns) == [
        "source_list",
        "source_file_list",
        "source_record_count",
        "cation",
        "anion",
        "temperature_K",
        "heat_capacity_J/mol/K",
    ]
    assert not (output_root / "simulation" / "heat_capacity_err_j_per_mol_per_k.csv").exists()


def test_property_split_drops_rows_missing_that_property(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "final"
    write_csv(
        input_root / "simulation" / "simulated_QM_elec_HF_structured.csv",
        [
            {"SMILES": "CCO", "ESP_max": 1.0, "ESP_min": -1.0},
            {"SMILES": "CCN", "ESP_max": None, "ESP_min": -2.0},
        ],
    )

    merge_final_data(input_root, output_root)

    esp_max = pd.read_csv(output_root / "simulation" / "esp_max.csv")
    esp_min = pd.read_csv(output_root / "simulation" / "esp_min.csv")
    assert len(esp_max) == 1
    assert len(esp_min) == 2
    assert not esp_max["ESP_max"].isna().any()


def test_same_key_different_label_values_remain_separate_rows(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "final"
    base = {"cation": "CC[n+]1ccn(C)c1", "anion": "F[B-](F)(F)F", "temperature_K": 298.15}
    write_csv(input_root / "AIonopedia" / "AIonopedia_viscosity_structured.csv", [{**base, "viscosity_mPa*s_log10": 1.0}])
    write_csv(input_root / "ILBERT" / "ILBERT_viscosity_structured.csv", [{**base, "viscosity_mPa*s_log10": 2.0}])

    merge_final_data(input_root, output_root)

    out = pd.read_csv(output_root / "experiment" / "viscosity_mpa_s_log10.csv")
    assert len(out) == 2
    assert set(out["viscosity_mPa*s_log10"]) == {1.0, 2.0}
    assert set(out["source_record_count"]) == {1}


def test_manifest_records_output_files_and_counts(tmp_path: Path):
    input_root = tmp_path / "cleaned"
    output_root = tmp_path / "final"
    write_csv(
        input_root / "ILBERT" / "ILBERT_HC_structured.csv",
        [{"cation": "CC[n+]1ccn(C)c1", "anion": "F[B-](F)(F)F", "temperature_K": 298.15, "heat_capacity_J/mol/K": 500.0}],
    )

    merge_final_data(input_root, output_root)

    manifest = pd.read_csv(output_root / "final_manifest.csv")
    assert list(manifest.columns) == ["bucket", "property_label", "output_file", "input_files", "input_rows", "output_rows"]
    row = manifest.iloc[0]
    assert row["bucket"] == "experiment"
    assert row["property_label"] == "heat_capacity_J/mol/K"
    assert row["output_file"] == "experiment/heat_capacity_j_per_mol_per_k.csv"
    assert row["input_files"] == "ILBERT/ILBERT_HC_structured.csv"
    assert row["input_rows"] == 1
    assert row["output_rows"] == 1
