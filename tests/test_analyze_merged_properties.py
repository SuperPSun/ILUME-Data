from pathlib import Path

import matplotlib.axes
import pandas as pd

from scripts.analyze_merged_properties import analyze_merged_properties, plot_system_frequency


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_analyze_merged_properties_summarizes_regular_properties(tmp_path: Path):
    input_root = tmp_path / "merged"
    output_dir = input_root / "analysis"
    write_csv(
        input_root / "merged_manifest.csv",
        [
            {
                "bucket": "experiment",
                "property_label": "density_g/cm^3",
                "output_file": "experiment/density.csv",
                "input_files": "ILBERT/ILBERT_density_structured.csv",
                "input_rows": 3,
                "output_rows": 3,
            },
            {
                "bucket": "simulation",
                "property_label": "density_g/cm^3",
                "output_file": "simulation/density.csv",
                "input_files": "simulation/simulated_density_structured.csv",
                "input_rows": 1,
                "output_rows": 1,
            },
            {
                "bucket": "experiment",
                "property_label": "electrical_conductivity_S/m_log10",
                "output_file": "experiment/electrical_conductivity.csv",
                "input_files": "ILBERT/ILBERT_EC_structured.csv",
                "input_rows": 2,
                "output_rows": 2,
            },
        ],
    )
    write_csv(
        input_root / "experiment" / "density.csv",
        [
            {
                "cation": "cat1",
                "anion": "an1",
                "temperature_K": 298.15,
                "pressure_kPa": 101.3,
                "density_g/cm^3": 1.0,
                "source_list": "ILBERT",
            },
            {
                "cation": "cat1",
                "anion": "an1",
                "temperature_K": 308.15,
                "pressure_kPa": None,
                "density_g/cm^3": 1.2,
                "source_list": "ILBERT; ILThermo",
            },
            {
                "cation": "cat2",
                "anion": "an2",
                "temperature_K": 298.15,
                "pressure_kPa": 101.3,
                "density_g/cm^3": 1.4,
                "source_list": "ILThermo",
            },
        ],
    )
    write_csv(
        input_root / "simulation" / "density.csv",
        [
            {
                "cation": "cat3",
                "anion": "an3",
                "temperature_K": 298.15,
                "density_g/cm^3": 1.1,
                "source_list": "simulation",
            }
        ],
    )
    write_csv(
        input_root / "experiment" / "electrical_conductivity.csv",
        [
            {
                "cation": "cat4",
                "anion": "an4",
                "temperature_K": 298.15,
                "frequency_MHz": 1.0,
                "electrical_conductivity_S/m_log10": -1.0,
                "source_list": "ILBERT",
            },
            {
                "cation": "cat5",
                "anion": "an5",
                "temperature_K": 308.15,
                "frequency_MHz": 10.0,
                "electrical_conductivity_S/m_log10": -0.5,
                "source_list": "ILBERT",
            },
        ],
    )

    summary = analyze_merged_properties(
        input_root,
        output_dir,
        min_holdout_systems=2,
        test_fraction=0.5,
        high_coverage_threshold=3,
        medium_coverage_threshold=2,
    )

    experiment = summary[summary["bucket"].eq("experiment")].iloc[0]
    assert experiment["property"] == "density"
    assert experiment["property_label"] == "density_g/cm^3"
    assert experiment["rows"] == 3
    assert experiment["data_points"] == 3
    assert experiment["unique_systems"] == 2
    assert experiment["points_per_system_mean"] == 1.5
    assert experiment["max_points_per_system"] == 2
    assert experiment["multi_point_systems"] == 1
    assert experiment["multi_point_system_ratio"] == 0.5
    assert experiment["condition_columns"] == "temperature_K; pressure_kPa"
    assert experiment["condition_complete_rows"] == 2
    assert experiment["unique_condition_sets"] == 2
    assert experiment["source_count"] == 2
    assert experiment["value_min"] == 1.0
    assert experiment["value_median"] == 1.2
    assert experiment["value_max"] == 1.4
    assert experiment["leakage_risk"] == "high"
    assert experiment["recommended_split"] == "system_holdout_test"
    assert experiment["recommended_test_systems"] == 1

    simulation = summary[summary["bucket"].eq("simulation")].iloc[0]
    assert simulation["source_count"] == 1
    assert simulation["recommended_split"] == "leave_one_system_out_or_descriptive_only"

    written = pd.read_csv(output_dir / "property_analysis_summary.csv")
    assert len(written) == 3
    plot_manifest = pd.read_csv(output_dir / "plot_manifest.csv")
    assert {
        "coverage",
        "property_distribution_2d",
        "system_frequency",
        "condition_availability",
    }.issubset(set(plot_manifest["figure_type"]))
    assert (output_dir / "figures" / "coverage" / "property_coverage_all.png").exists()
    assert (output_dir / "figures" / "coverage" / "property_coverage_high.png").exists()
    assert (output_dir / "figures" / "coverage" / "property_coverage_medium.png").exists()
    assert (output_dir / "figures" / "coverage" / "property_coverage_low.png").exists()
    assert (
        output_dir
        / "figures"
        / "property_distributions"
        / "high"
        / "experiment_density_temperature_k_pressure_kpa.png"
    ).exists()
    assert (
        plot_manifest["path"]
        .str.contains("experiment_density_temperature_k_density_value.png")
        .any()
    )
    assert (
        plot_manifest["path"]
        .str.contains("experiment_electrical_conductivity_log10_frequency_mhz_electrical_conductivity_value.png")
        .any()
    )
    assert (output_dir / "figures" / "system_frequency" / "high" / "experiment_density.png").exists()
    assert (
        output_dir / "figures" / "condition_availability" / "property_condition_availability_heatmap.png"
    ).exists()
    report = (output_dir / "property_analysis_report.md").read_text()
    assert "Merged Property Analysis Report" in report
    assert "High Leakage Risk Properties" in report
    assert "Generated Figures" in report


def test_analyze_merged_properties_uses_equilibrium_pressure_name(tmp_path: Path):
    input_root = tmp_path / "merged"
    output_dir = input_root / "analysis"
    write_csv(
        input_root / "merged_manifest.csv",
        [
            {
                "bucket": "experiment",
                "property_label": "pressure_kPa_log10",
                "output_file": "experiment/equilibrium_pressure.csv",
                "input_files": "ILThermo/ilt_equilibrium_pressure_structured.csv",
                "input_rows": 1,
                "output_rows": 1,
            }
        ],
    )
    write_csv(
        input_root / "experiment" / "equilibrium_pressure.csv",
        [
            {
                "cation": "cat1",
                "anion": "an1",
                "temperature_K": 298.15,
                "pressure_kPa_log10": 2.0,
                "source_list": "ILThermo",
            }
        ],
    )

    summary = analyze_merged_properties(input_root, output_dir)

    row = summary.iloc[0]
    assert row["property"] == "equilibrium_pressure"
    assert row["property_label"] == "pressure_kPa_log10"
    plot_manifest = pd.read_csv(output_dir / "plot_manifest.csv")
    assert plot_manifest["path"].str.contains("experiment_equilibrium_pressure").any()


def test_analyze_merged_properties_splits_wide_tables_by_value_column(tmp_path: Path):
    input_root = tmp_path / "merged"
    output_dir = input_root / "analysis"
    write_csv(
        input_root / "merged_manifest.csv",
        [
            {
                "bucket": "simulation",
                "property_label": "simulated_QM_elec_HF",
                "output_file": "simulation/simulated_qm_elec_hf.csv",
                "input_files": "simulation/simulated_QM_elec_HF_structured.csv",
                "input_rows": 2,
                "output_rows": 2,
            }
        ],
    )
    write_csv(
        input_root / "simulation" / "simulated_qm_elec_hf.csv",
        [
            {"SMILES": "CCO", "ESP_max": 1.0, "ESP_min": -1.0, "gap_eV": 2.0, "source_list": "simulation"},
            {"SMILES": "CCN", "ESP_max": None, "ESP_min": -2.0, "gap_eV": 3.0, "source_list": "simulation"},
        ],
    )

    summary = analyze_merged_properties(input_root, output_dir, min_holdout_systems=200, test_fraction=0.1)

    assert set(summary["property"]) == {"esp_max", "esp_min", "gap"}
    esp_max = summary[summary["property"].eq("esp_max")].iloc[0]
    assert esp_max["property_label"] == "ESP_max"
    assert esp_max["rows"] == 2
    assert esp_max["data_points"] == 1
    assert esp_max["unique_systems"] == 1
    assert esp_max["value_min"] == 1.0
    assert esp_max["recommended_split"] == "leave_one_system_out_or_descriptive_only"

    gap = summary[summary["property"].eq("gap")].iloc[0]
    assert gap["data_points"] == 2
    assert gap["unique_systems"] == 2

    plot_manifest = pd.read_csv(output_dir / "plot_manifest.csv")
    assert "property_distribution_1d" in set(plot_manifest["figure_type"])
    assert (
        output_dir / "figures" / "property_distributions" / "low" / "simulation_esp_max_esp_max_value.png"
    ).exists()


def test_analyze_merged_properties_draws_single_condition_property_distribution(tmp_path: Path):
    input_root = tmp_path / "merged"
    output_dir = input_root / "analysis"
    write_csv(
        input_root / "merged_manifest.csv",
        [
            {
                "bucket": "experiment",
                "property_label": "surface_tension_mN/m",
                "output_file": "experiment/surface_tension.csv",
                "input_files": "ILThermo/ilt_surface_tension_structured.csv",
                "input_rows": 2,
                "output_rows": 2,
            }
        ],
    )
    write_csv(
        input_root / "experiment" / "surface_tension.csv",
        [
            {
                "cation": "cat1",
                "anion": "an1",
                "temperature_K": 298.15,
                "surface_tension_mN/m": 40.0,
                "source_list": "ILThermo",
            },
            {
                "cation": "cat2",
                "anion": "an2",
                "temperature_K": 308.15,
                "surface_tension_mN/m": 35.0,
                "source_list": "ILThermo",
            },
        ],
    )

    analyze_merged_properties(input_root, output_dir)

    plot_manifest = pd.read_csv(output_dir / "plot_manifest.csv")
    assert "property_distribution_2d" in set(plot_manifest["figure_type"])
    assert (
        output_dir
        / "figures"
        / "property_distributions"
        / "low"
        / "experiment_surface_tension_temperature_k_surface_tension_value.png"
    ).exists()


def test_system_frequency_uses_linear_x_axis(tmp_path: Path, monkeypatch):
    xscale_calls: list[str] = []

    def record_xscale(self, value, *args, **kwargs):
        xscale_calls.append(value)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xscale", record_xscale)

    plot_system_frequency(
        pd.Series([1, 2, 25]),
        {
            "bucket": "experiment",
            "property": "density",
            "max_points_per_system": 25,
            "leakage_risk": "high",
        },
        tmp_path / "system_frequency.png",
        dpi=80,
    )

    assert xscale_calls == []


def test_analyze_merged_properties_counts_conditions_only_for_present_values(tmp_path: Path):
    input_root = tmp_path / "merged"
    output_dir = input_root / "analysis"
    write_csv(
        input_root / "merged_manifest.csv",
        [
            {
                "bucket": "simulation",
                "property_label": "simulated_QM_elec_HF",
                "output_file": "simulation/simulated_qm_elec_hf.csv",
                "input_files": "simulation/simulated_QM_elec_HF_structured.csv",
                "input_rows": 2,
                "output_rows": 2,
            }
        ],
    )
    write_csv(
        input_root / "simulation" / "simulated_qm_elec_hf.csv",
        [
            {
                "SMILES": "CCO",
                "temperature_K": 298.15,
                "pressure_kPa": 101.3,
                "q_max": 1.0,
                "q_min": -1.0,
                "source_list": "simulation",
            },
            {
                "SMILES": "CCN",
                "temperature_K": 308.15,
                "pressure_kPa": 101.3,
                "q_max": None,
                "q_min": -2.0,
                "source_list": "simulation",
            },
        ],
    )

    summary = analyze_merged_properties(input_root, output_dir)

    q_max = summary[summary["property"].eq("q_max")].iloc[0]
    assert q_max["data_points"] == 1
    assert q_max["condition_complete_rows"] == 1
    assert q_max["unique_condition_sets"] == 1


def test_analyze_merged_properties_recommends_grouped_cv_for_mid_sized_data(tmp_path: Path):
    input_root = tmp_path / "merged"
    output_dir = input_root / "analysis"
    write_csv(
        input_root / "merged_manifest.csv",
        [
            {
                "bucket": "experiment",
                "property_label": "melting_point_K",
                "output_file": "experiment/melting_point.csv",
                "input_files": "ILBERT/ILBERT_MP_structured.csv",
                "input_rows": 20,
                "output_rows": 20,
            }
        ],
    )
    write_csv(
        input_root / "experiment" / "melting_point.csv",
        [
            {
                "cation": f"cat{i}",
                "anion": f"an{i}",
                "melting_point_K": 250.0 + i,
                "source_list": "ILBERT",
            }
            for i in range(20)
        ],
    )

    summary = analyze_merged_properties(input_root, output_dir, min_holdout_systems=200, test_fraction=0.1)

    row = summary.iloc[0]
    assert row["unique_systems"] == 20
    assert row["recommended_split"] == "grouped_cross_validation"
    assert row["recommended_test_systems"] == 0


def test_analyze_merged_properties_can_skip_plots(tmp_path: Path):
    input_root = tmp_path / "merged"
    output_dir = input_root / "analysis"
    write_csv(
        input_root / "merged_manifest.csv",
        [
            {
                "bucket": "experiment",
                "property_label": "melting_point_K",
                "output_file": "experiment/melting_point.csv",
                "input_files": "ILBERT/ILBERT_MP_structured.csv",
                "input_rows": 1,
                "output_rows": 1,
            }
        ],
    )
    write_csv(
        input_root / "experiment" / "melting_point.csv",
        [{"cation": "cat1", "anion": "an1", "melting_point_K": 250.0, "source_list": "ILBERT"}],
    )

    analyze_merged_properties(input_root, output_dir, skip_plots=True)

    assert (output_dir / "property_analysis_summary.csv").exists()
    assert (output_dir / "property_analysis_report.md").exists()
    assert not (output_dir / "plot_manifest.csv").exists()
    assert not (output_dir / "figures").exists()
