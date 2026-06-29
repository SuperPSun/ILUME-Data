from pathlib import Path

import pandas as pd

from scripts.clean_structured_data import (
    clean_non_ilthermo_structured,
    clean_structured_file,
)


def test_deleted_rows_are_reported_with_reasons(tmp_path: Path):
    input_path = tmp_path / "AIonopedia_density_structured.csv"
    output_path = tmp_path / "out.csv"
    rejected_path = tmp_path / "rejected.csv"
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1", "", "CC[n+]1ccn(C)c1"],
            "anion": ["F[B-](F)(F)F", "F[B-](F)(F)F", "F[B-](F)(F)F"],
            "temperature_K": [298.15, 9999.0, 298.15],
            "density_g/cm^3": [1.1, 1.2, 1.1],
        }
    ).to_csv(input_path, index=False)

    result = clean_structured_file(input_path, output_path, rejected_path)

    cleaned = pd.read_csv(output_path)
    rejected = pd.read_csv(rejected_path)
    assert len(cleaned) == 1
    assert result.rejected_rows == 2
    assert {"row_index", "rejection_reason", "trigger_column", "trigger_value"}.issubset(rejected.columns)
    assert set(rejected["rejection_reason"]) == {"missing_identifier", "duplicate_row"}
    assert rejected.loc[rejected["rejection_reason"] == "missing_identifier", "trigger_column"].iloc[0] == "cation"


def test_temperature_and_pressure_do_not_trigger_hard_threshold_rejection(tmp_path: Path):
    input_path = tmp_path / "ILBERT_density_structured.csv"
    output_path = tmp_path / "out.csv"
    rejected_path = tmp_path / "rejected.csv"
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1"],
            "anion": ["F[B-](F)(F)F"],
            "temperature_K": [9999.0],
            "pressure_kPa": [9999999.0],
            "density_g/cm^3": [1.2],
        }
    ).to_csv(input_path, index=False)

    clean_structured_file(input_path, output_path, rejected_path)

    cleaned = pd.read_csv(output_path)
    assert len(cleaned) == 1
    assert cleaned.loc[0, "temperature_K"] == 9999.0
    assert not rejected_path.exists()


def test_hard_threshold_and_iqr_reject_label_outliers(tmp_path: Path):
    input_path = tmp_path / "after_AIonopedia_viscosity_structured.csv"
    output_path = tmp_path / "out.csv"
    rejected_path = tmp_path / "rejected.csv"
    rows = []
    for index in range(20):
        rows.append(
            {
                "cation": f"{'C' * (index + 1)}n1cc[n+](C)c1",
                "anion": "F[B-](F)(F)F",
                "temperature_K": 298.15,
                "viscosity_mPa*s": 100.0 + index % 3,
            }
        )
    rows.append(
        {
            "cation": "CCCCCCCCCCCCCCCCCCn1cc[n+](C)c1",
            "anion": "F[B-](F)(F)F",
            "temperature_K": 298.15,
            "viscosity_mPa*s": 10000.0,
        }
    )
    rows.append(
        {
            "cation": "CCCCCCCCCCCCCCCCCCCn1cc[n+](C)c1",
            "anion": "F[B-](F)(F)F",
            "temperature_K": 298.15,
            "viscosity_mPa*s": -5.0,
        }
    )
    pd.DataFrame(rows).to_csv(input_path, index=False)

    result = clean_structured_file(input_path, output_path, rejected_path)

    cleaned = pd.read_csv(output_path)
    rejected = pd.read_csv(rejected_path)
    assert "viscosity_mPa*s_log10" in cleaned.columns
    assert result.iqr_bounds["viscosity_mPa*s_log10"]["applied"] is True
    assert set(rejected["rejection_reason"]) == {"hard_threshold", "iqr_outlier"}
    assert len(cleaned) == 20


def test_iqr_skips_small_samples(tmp_path: Path):
    input_path = tmp_path / "after_AIonopedia_solv_structured.csv"
    output_path = tmp_path / "out.csv"
    rejected_path = tmp_path / "rejected.csv"
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1"] * 3,
            "anion": ["F[B-](F)(F)F"] * 3,
            "solute": ["O=C=O", "CCO", "CCCC"],
            "temperature_K": [298.15, 298.15, 298.15],
            "solvation_kcal/mol": [-1.0, -2.0, -90.0],
        }
    ).to_csv(input_path, index=False)

    result = clean_structured_file(input_path, output_path, rejected_path)

    cleaned = pd.read_csv(output_path)
    assert len(cleaned) == 3
    assert result.iqr_bounds["solvation_kcal/mol"]["applied"] is False
    assert not rejected_path.exists()


def test_units_are_unified_for_simulation_and_viscosity(tmp_path: Path):
    density_input = tmp_path / "simulated_density_structured.csv"
    density_output = tmp_path / "density_out.csv"
    density_rejected = tmp_path / "density_rejected.csv"
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1"],
            "anion": ["F[B-](F)(F)F"],
            "temperature_K": [298.15],
            "density": [1250.0],
            "density_err": [2.0],
        }
    ).to_csv(density_input, index=False)

    clean_structured_file(density_input, density_output, density_rejected)
    density = pd.read_csv(density_output)
    assert list(density.columns) == ["cation", "anion", "temperature_K", "density_g/cm^3", "density_err_g/cm^3"]
    assert density.loc[0, "density_g/cm^3"] == 1.25
    assert density.loc[0, "density_err_g/cm^3"] == 0.002

    viscosity_input = tmp_path / "ILBERT_viscosity_structured.csv"
    viscosity_output = tmp_path / "viscosity_out.csv"
    viscosity_rejected = tmp_path / "viscosity_rejected.csv"
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1"],
            "anion": ["F[B-](F)(F)F"],
            "temperature_K": [298.15],
            "viscosity_mPa*s": [100.0],
            "ln_viscosity_mPa*s_unitless": [4.605170185988092],
        }
    ).to_csv(viscosity_input, index=False)

    clean_structured_file(viscosity_input, viscosity_output, viscosity_rejected)
    viscosity = pd.read_csv(viscosity_output)
    assert list(viscosity.columns) == ["cation", "anion", "temperature_K", "viscosity_mPa*s_log10"]
    assert viscosity.loc[0, "viscosity_mPa*s_log10"] == 2.0


def test_simulation_property_columns_are_renamed(tmp_path: Path):
    input_path = tmp_path / "simulated_heat_of_vaporization_structured.csv"
    output_path = tmp_path / "out.csv"
    rejected_path = tmp_path / "rejected.csv"
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1"],
            "anion": ["F[B-](F)(F)F"],
            "temperature_K": [298.15],
            "Hvap": [120.5],
            "Hvap_err": [0.5],
        }
    ).to_csv(input_path, index=False)

    clean_structured_file(input_path, output_path, rejected_path)

    cleaned = pd.read_csv(output_path)
    assert list(cleaned.columns) == [
        "cation",
        "anion",
        "temperature_K",
        "heat_of_vaporization_kJ/mol",
        "heat_of_vaporization_err_kJ/mol",
    ]


def test_empty_charge_is_kept_but_out_of_range_charge_is_rejected(tmp_path: Path):
    input_path = tmp_path / "simulated_charge_20260514_mapping_structured.csv"
    output_path = tmp_path / "out.csv"
    rejected_path = tmp_path / "rejected.csv"
    pd.DataFrame(
        {
            "mol_id": ["mol_1", "mol_2"],
            "SMILES": ["CCO", "CCN"],
            "charge": [None, 99],
        }
    ).to_csv(input_path, index=False)

    clean_structured_file(input_path, output_path, rejected_path)

    cleaned = pd.read_csv(output_path)
    rejected = pd.read_csv(rejected_path)
    assert len(cleaned) == 1
    assert pd.isna(cleaned.loc[0, "charge"])
    assert rejected.loc[0, "rejection_reason"] == "hard_threshold"


def test_unknown_unit_simulation_columns_are_not_iqr_filtered(tmp_path: Path):
    input_path = tmp_path / "simulated_QM_elec_HF_structured.csv"
    output_path = tmp_path / "out.csv"
    rejected_path = tmp_path / "rejected.csv"
    rows = []
    for index in range(20):
        rows.append(
            {
                "SMILES": "C" * (index + 1),
                "ESP_max": float(index % 3),
                "Dipole": float(index % 4),
                "q_abs_mean": 0.1,
                "ESP_pos_frac": 0.5,
                "Gap": 5.0,
            }
        )
    rows.append({"SMILES": "CCCCCCCCCCCCCCCCCCCCC", "ESP_max": 9999.0, "Dipole": 999.0, "q_abs_mean": 99.0, "ESP_pos_frac": 0.5, "Gap": 5.0})
    pd.DataFrame(rows).to_csv(input_path, index=False)

    result = clean_structured_file(input_path, output_path, rejected_path)

    cleaned = pd.read_csv(output_path)
    assert len(cleaned) == 21
    assert "ESP_max" not in result.iqr_bounds
    assert "Dipole" not in result.iqr_bounds
    assert "q_abs_mean" not in result.iqr_bounds
    assert "gap_eV" in cleaned.columns
    assert not rejected_path.exists()


def test_directory_cleaning_skips_ilthermo_and_constructed_and_writes_reports(tmp_path: Path):
    input_root = tmp_path / "structured"
    output_root = tmp_path / "cleaned"
    (input_root / "AIonopedia").mkdir(parents=True)
    (input_root / "ILThermo").mkdir(parents=True)
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1", ""],
            "anion": ["F[B-](F)(F)F", "F[B-](F)(F)F"],
            "temperature_K": [298.15, 298.15],
            "density_g/cm^3": [1.2, 1.1],
        }
    ).to_csv(input_root / "AIonopedia" / "AIonopedia_density_structured.csv", index=False)
    pd.DataFrame({"density": [1.2]}).to_csv(input_root / "AIonopedia" / "AIonopedia_density_constructed.csv", index=False)
    pd.DataFrame({"cation": ["CC[n+]1ccn(C)c1"], "density_g/cm^3": [1.2]}).to_csv(
        input_root / "ILThermo" / "ilt_density_structured.csv", index=False
    )

    results = clean_non_ilthermo_structured(input_root, output_root)

    assert len(results) == 1
    assert (output_root / "AIonopedia" / "AIonopedia_density_structured.csv").exists()
    assert not (output_root / "AIonopedia" / "AIonopedia_density_constructed.csv").exists()
    assert not (output_root / "ILThermo").exists()
    report = (output_root / "cleaning_report.md").read_text(encoding="utf-8")
    assert "AIonopedia_density_structured.csv" in report
    assert "missing_identifier" in report
    assert (output_root / "cleaning_report.csv").exists()
