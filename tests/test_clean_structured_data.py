from pathlib import Path

import pandas as pd

from scripts.clean_structured_data import (
    BOX_3D_OUTPUT_COLUMNS,
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


def test_rare_values_within_hard_threshold_are_retained(tmp_path: Path):
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
                "viscosity_mPa*s": 100.0,
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
    pd.DataFrame(rows).to_csv(input_path, index=False)

    result = clean_structured_file(input_path, output_path, rejected_path)

    cleaned = pd.read_csv(output_path)
    assert "viscosity_mPa*s_log10" in cleaned.columns
    assert len(cleaned) == 21
    assert result.rejected_rows == 0
    assert not rejected_path.exists()


def test_report_does_not_include_iqr_fields_or_reasons(tmp_path: Path):
    input_path = tmp_path / "after_AIonopedia_solv_structured.csv"
    input_root = tmp_path / "structured"
    output_root = tmp_path / "cleaned"
    (input_root / "after_AIonopedia").mkdir(parents=True)
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1", ""],
            "anion": ["F[B-](F)(F)F", "F[B-](F)(F)F"],
            "solute": ["O=C=O", "CCO"],
            "temperature_K": [298.15, 298.15],
            "solvation_kcal/mol": [-1.0, -2.0],
        }
    ).to_csv(input_root / "after_AIonopedia" / input_path.name, index=False)

    clean_non_ilthermo_structured(input_root, output_root, sources=["after_AIonopedia"])

    report_md = (output_root / "cleaning_report.md").read_text(encoding="utf-8")
    report_csv = pd.read_csv(output_root / "cleaning_report.csv")
    assert "IQR" not in report_md
    assert "iqr_outlier" not in report_md
    assert "iqr_outlier" not in str(report_csv["rejection_counts"].iloc[0])


def test_nonpositive_values_are_rejected_before_log_conversion(tmp_path: Path):
    input_path = tmp_path / "ILBERT_EC_structured.csv"
    output_path = tmp_path / "out.csv"
    rejected_path = tmp_path / "rejected.csv"
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1", "CCC[n+]1ccn(C)c1"],
            "anion": ["F[B-](F)(F)F", "F[B-](F)(F)F"],
            "temperature_K": [298.15, 298.15],
            "electrical_conductivity_S/m": [0.1, 0.0],
            "lnEC_unitless": [-2.302585092994046, None],
        }
    ).to_csv(input_path, index=False)

    clean_structured_file(input_path, output_path, rejected_path)

    cleaned = pd.read_csv(output_path)
    rejected = pd.read_csv(rejected_path)
    assert list(cleaned.columns) == ["cation", "anion", "temperature_K", "electrical_conductivity_S/m_log10"]
    assert cleaned.loc[0, "electrical_conductivity_S/m_log10"] == -1.0
    assert rejected.loc[0, "rejection_reason"] == "nonpositive_for_log"


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


def test_3d_box_is_selected_and_cleaned_with_prepeak_presence(tmp_path: Path):
    input_root = tmp_path / "structured"
    output_root = tmp_path / "cleaned"
    input_path = input_root / "simulation" / "3d_box_structured.csv"
    input_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "mol_id": ["mol_1", "mol_2"],
            "cation": ["C[N+](C)(C)C", "C[N+](C)(C)C"],
            "anion": ["[Cl-]", "[Cl-]"],
            "temperature_K": [298.15, 308.15],
            "rdf_ca_peak1_r_A": [4.1, 4.2],
            "rdf_ca_peak1_g": [2.1, 2.2],
            "rdf_ca_coordination1": [5.1, 5.2],
            "rdf_ca_excess_area_A": [0.8, 0.9],
            "rdf_cc_peak1_g": [1.5, 1.6],
            "rdf_cc_excess_area_A": [0.3, 0.4],
            "rdf_aa_peak1_g": [1.7, 1.8],
            "rdf_aa_excess_area_A": [0.5, 0.6],
            "scc_prepeak_q_A^-1": [0.25, None],
            "scc_prepeak_height": [1.4, None],
            "scc_prepeak_area_A^-1": [0.12, None],
            "szz_peak_q_A^-1": [0.7, 0.8],
            "szz_peak_height": [1.9, 2.0],
            "szz_peak_area_A^-1": [0.2, 0.3],
            "qc_status": ["ok", "partial"],
        }
    ).to_csv(input_path, index=False)

    results = clean_non_ilthermo_structured(input_root, output_root, sources=["simulation"])

    output_path = output_root / "simulation" / "3d_box_structured.csv"
    cleaned = pd.read_csv(output_path)
    assert len(results) == 1
    assert list(cleaned.columns) == list(BOX_3D_OUTPUT_COLUMNS)
    assert cleaned["scc_prepeak_present"].tolist() == [True, False]
    assert "qc_status" not in cleaned.columns
    assert (output_root / "cleaning_report.csv").exists()


def test_3d_box_cleaning_reports_missing_required_columns(tmp_path: Path):
    input_path = tmp_path / "3d_box_structured.csv"
    pd.DataFrame({"mol_id": ["mol_1"]}).to_csv(input_path, index=False)

    try:
        clean_structured_file(input_path, tmp_path / "out.csv")
    except ValueError as error:
        assert "3d_box_structured.csv is missing columns" in str(error)
        assert "rdf_ca_peak1_r_A" in str(error)
    else:
        raise AssertionError("expected missing 3d_box columns to raise ValueError")


def test_log_or_linear_label_choices_are_applied(tmp_path: Path):
    ec_input = tmp_path / "ILBERT_EC_structured.csv"
    ec_output = tmp_path / "ec_out.csv"
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1"],
            "anion": ["F[B-](F)(F)F"],
            "temperature_K": [298.15],
            "electrical_conductivity_S/m": [0.01],
            "lnEC_unitless": [-4.605170185988092],
        }
    ).to_csv(ec_input, index=False)
    clean_structured_file(ec_input, ec_output, tmp_path / "ec_rejected.csv")
    assert list(pd.read_csv(ec_output).columns) == [
        "cation",
        "anion",
        "temperature_K",
        "electrical_conductivity_S/m_log10",
    ]

    co2_input = tmp_path / "ILBERT_CO2_structured.csv"
    co2_output = tmp_path / "co2_out.csv"
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1"],
            "anion": ["F[B-](F)(F)F"],
            "temperature_K": [298.15],
            "pressure_kPa": [100.0],
            "x_CO2_unitless": [0.25],
            "ln_x_CO2_unitless": [-1.3862943611198906],
        }
    ).to_csv(co2_input, index=False)
    clean_structured_file(co2_input, co2_output, tmp_path / "co2_rejected.csv")
    assert list(pd.read_csv(co2_output).columns) == ["cation", "anion", "temperature_K", "pressure_kPa", "x_CO2_unitless"]

    hc_input = tmp_path / "ILBERT_HC_structured.csv"
    hc_output = tmp_path / "hc_out.csv"
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1"],
            "anion": ["F[B-](F)(F)F"],
            "temperature_K": [298.15],
            "hc_unitless": [500.0],
            "lnhc_unitless": [6.214608098422191],
        }
    ).to_csv(hc_input, index=False)
    clean_structured_file(hc_input, hc_output, tmp_path / "hc_rejected.csv")
    hc = pd.read_csv(hc_output)
    assert list(hc.columns) == ["cation", "anion", "temperature_K", "heat_capacity_J/mol/K"]
    assert hc.loc[0, "heat_capacity_J/mol/K"] == 500.0

    ec50_input = tmp_path / "ILBERT_norm_cytotoxicity_structured.csv"
    ec50_output = tmp_path / "ec50_out.csv"
    pd.DataFrame(
        {
            "cation": ["CC[n+]1ccn(C)c1"],
            "anion": ["F[B-](F)(F)F"],
            "logEC50_unitless": [3.0],
            "EC50_unitless": [1000.0],
        }
    ).to_csv(ec50_input, index=False)
    clean_structured_file(ec50_input, ec50_output, tmp_path / "ec50_rejected.csv")
    ec50 = pd.read_csv(ec50_output)
    assert list(ec50.columns) == ["cation", "anion", "pEC50"]
    assert ec50.loc[0, "pEC50"] == -3.0


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
    base = {
        "ESP_abs_mean": 0.2,
        "ESP_std": 0.3,
        "ESP_pos_frac": 0.5,
        "ESP_neg_frac": 0.5,
        "ESP_pos_mean": 0.4,
        "ESP_neg_mean": -0.4,
        "Quadrupole": 1.0,
        "q_max": 0.5,
        "q_min": -0.5,
        "q_abs_mean": 0.1,
        "q_std": 0.2,
        "q_pos_sum": 1.0,
        "q_neg_sum": -1.0,
        "q_pos_frac": 0.5,
        "Gap": 5.0,
    }
    rows = [
        {
            "SMILES": "C" * (index + 1),
            **base,
            "ESP_max": float(index % 3),
            "ESP_min": -float(index % 3),
            "Dipole": float(index % 4),
        }
        for index in range(20)
    ]
    rows.append(
        {
            "SMILES": "CCCCCCCCCCCCCCCCCCCCC",
            **base,
            "ESP_max": 9999.0,
            "ESP_min": -9999.0,
            "ESP_abs_mean": 999.0,
            "ESP_std": 999.0,
            "ESP_pos_mean": 999.0,
            "ESP_neg_mean": -999.0,
            "Dipole": 999.0,
            "Quadrupole": 999.0,
            "q_max": 99.0,
            "q_min": -99.0,
            "q_abs_mean": 99.0,
            "q_std": 99.0,
            "q_pos_sum": 99.0,
            "q_neg_sum": -99.0,
        }
    )
    pd.DataFrame(rows).to_csv(input_path, index=False)

    result = clean_structured_file(input_path, output_path, rejected_path)

    cleaned = pd.read_csv(output_path)
    assert len(cleaned) == 21
    assert list(cleaned.columns) == [
        "SMILES",
        "ESP_max",
        "ESP_min",
        "ESP_std",
        "ESP_pos_frac",
        "Dipole",
        "Quadrupole",
        "q_max",
        "q_min",
        "q_std",
        "q_pos_frac",
        "gap_eV",
    ]
    assert cleaned.loc[20, "gap_eV"] == 5.0
    assert cleaned.loc[20, "ESP_max"] == 9999.0
    removed = {"ESP_pos_mean", "ESP_neg_mean", "ESP_abs_mean", "ESP_neg_frac", "q_abs_mean", "q_pos_sum", "q_neg_sum"}
    assert removed.isdisjoint(cleaned.columns)
    assert any("dropped unsupported QM labels" in conversion for conversion in result.unit_conversions)
    assert not rejected_path.exists()


def test_qm_label_filtering_precedes_missing_label_and_threshold_checks(tmp_path: Path):
    input_path = tmp_path / "simulated_QM_elec_HF_structured.csv"
    output_path = tmp_path / "out.csv"
    rejected_path = tmp_path / "rejected.csv"
    base = {
        "ESP_max": 1.0,
        "ESP_min": -1.0,
        "ESP_std": 0.3,
        "ESP_pos_frac": 0.5,
        "Dipole": 2.0,
        "Quadrupole": 3.0,
        "q_max": 0.5,
        "q_min": -0.5,
        "q_std": 0.2,
        "q_pos_frac": 0.5,
        "Gap": 5.0,
        "q_abs_mean": 0.1,
    }
    pd.DataFrame(
        [
            {"SMILES": "C", **base},
            {
                "SMILES": "CC",
                **{column: None for column in base if column != "q_abs_mean"},
                "q_abs_mean": 0.1,
            },
            {"SMILES": "CCC", **base, "ESP_pos_frac": 1.1},
            {"SMILES": "CCCC", **base, "q_pos_frac": -0.1},
            {"SMILES": "CCCCC", **base, "Gap": 31.0},
        ]
    ).to_csv(input_path, index=False)

    clean_structured_file(input_path, output_path, rejected_path)

    cleaned = pd.read_csv(output_path)
    rejected = pd.read_csv(rejected_path)
    assert cleaned["SMILES"].tolist() == ["C"]
    assert rejected["rejection_reason"].value_counts().to_dict() == {"hard_threshold": 3, "missing_label": 1}
    assert set(rejected["trigger_column"]) == {
        "ESP_pos_frac",
        "q_pos_frac",
        "gap_eV",
        ",".join(cleaned.columns[1:]),
    }


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
