from pathlib import Path

import matplotlib.axes
import pandas as pd
import scripts.analyze_merged_properties as analysis_module

from scripts.analyze_merged_properties import (
    analyze_merged_properties,
    normalize_property_values,
    numeric_condition_dimensions,
    plot_condition_spaces,
    plot_normalized_property_violin,
    plot_system_frequency,
)


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
        "normalized_property_violin",
        "property_distribution_1d",
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
        / "property_distributions_2d"
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
        .str.contains("experiment_electrical_conductivity_frequency_mhz_electrical_conductivity_value.png")
        .any()
    )
    assert plot_manifest["path"].str.contains("experiment_density_density_value.png").any()
    assert (
        output_dir / "figures" / "normalized_distributions" / "property_normalized_violin.png"
    ).exists()
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


def test_analyze_merged_properties_reports_experiment_il_property_system_overlap(tmp_path: Path):
    input_root = tmp_path / "merged"
    output_dir = input_root / "analysis"
    write_csv(
        input_root / "merged_manifest.csv",
        [
            {
                "bucket": "experiment",
                "property_label": "density_g/cm^3",
                "output_file": "experiment/density.csv",
                "input_files": "density.csv",
                "input_rows": 4,
                "output_rows": 4,
            },
            {
                "bucket": "experiment",
                "property_label": "viscosity_mPa*s_log10",
                "output_file": "experiment/viscosity.csv",
                "input_files": "viscosity.csv",
                "input_rows": 3,
                "output_rows": 3,
            },
            {
                "bucket": "experiment",
                "property_label": "melting_point_K",
                "output_file": "experiment/melting_point.csv",
                "input_files": "melting_point.csv",
                "input_rows": 1,
                "output_rows": 1,
            },
        ],
    )
    write_csv(
        input_root / "experiment" / "density.csv",
        [
            {"cation": "cat1", "anion": "an1", "temperature_K": 298.15, "density_g/cm^3": 1.0},
            {"cation": "cat1", "anion": "an1", "temperature_K": 318.15, "density_g/cm^3": 0.9},
            {"cation": "cat2", "anion": "an2", "temperature_K": 298.15, "density_g/cm^3": 1.1},
            {"cation": None, "anion": "an3", "temperature_K": 298.15, "density_g/cm^3": 1.2},
        ],
    )
    write_csv(
        input_root / "experiment" / "viscosity.csv",
        [
            {"cation": "cat1", "anion": "an1", "temperature_K": 333.15, "viscosity_mPa*s_log10": 2.0},
            {"cation": "cat2", "anion": "an2", "temperature_K": 298.15, "viscosity_mPa*s_log10": None},
            {"cation": "cat3", "anion": "an3", "temperature_K": 298.15, "viscosity_mPa*s_log10": 2.2},
        ],
    )
    write_csv(
        input_root / "experiment" / "melting_point.csv",
        [{"cation": "cat1", "melting_point_K": 250.0}],
    )

    analyze_merged_properties(input_root, output_dir)

    overlap = pd.read_csv(output_dir / "property_system_overlap.csv")
    density_viscosity = overlap[
        overlap["property_a"].eq("density") & overlap["property_b"].eq("viscosity")
    ].iloc[0]
    assert density_viscosity["bucket"] == "experiment"
    assert density_viscosity["property_a_label"] == "density_g/cm^3"
    assert density_viscosity["property_b_label"] == "viscosity_mPa*s_log10"
    assert density_viscosity["property_a_unique_systems"] == 2
    assert density_viscosity["property_b_unique_systems"] == 2
    assert density_viscosity["shared_systems"] == 1
    assert density_viscosity["property_a_overlap_ratio"] == 0.5
    assert density_viscosity["property_b_overlap_ratio"] == 0.5
    density_diagonal = overlap[
        overlap["property_a"].eq("density") & overlap["property_b"].eq("density")
    ].iloc[0]
    assert density_diagonal["shared_systems"] == 2
    assert len(overlap) == 4
    assert (
        output_dir
        / "figures"
        / "property_system_overlap"
        / "experiment_property_system_overlap_heatmap.png"
    ).exists()
    plot_manifest = pd.read_csv(output_dir / "plot_manifest.csv")
    assert "property_system_overlap" in set(plot_manifest["figure_type"])
    report = (output_dir / "property_analysis_report.md").read_text()
    assert "Cross-property IL System Overlap" in report


def test_analyze_merged_properties_preserves_self_diffusion_log_label(tmp_path: Path):
    input_root = tmp_path / "merged"
    output_dir = input_root / "analysis"
    write_csv(
        input_root / "merged_manifest.csv",
        [
            {
                "bucket": "experiment",
                "property_label": "self_diffusion_coefficient_10^-9*m^2/s_log10",
                "output_file": "experiment/self_diffusion_coefficient.csv",
                "input_files": "ILThermo/ilt_self_diffusion_coefficient_structured.csv",
                "input_rows": 2,
                "output_rows": 2,
            }
        ],
    )
    write_csv(
        input_root / "experiment" / "self_diffusion_coefficient.csv",
        [
            {
                "cation": "cat1",
                "anion": "an1",
                "temperature_K": 298.15,
                "self_diffusion_coefficient_10^-9*m^2/s_log10": -2.0,
                "source_list": "ILThermo",
            },
            {
                "cation": "cat2",
                "anion": "an2",
                "temperature_K": 308.15,
                "self_diffusion_coefficient_10^-9*m^2/s_log10": -1.0,
                "source_list": "ILThermo",
            },
        ],
    )

    summary = analyze_merged_properties(input_root, output_dir)

    row = summary.iloc[0]
    assert row["property"] == "self_diffusion_coefficient"
    assert row["property_label"] == "self_diffusion_coefficient_10^-9*m^2/s_log10"
    assert row["value_min"] == -2.0
    assert row["value_max"] == -1.0


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
            {
                "SMILES": "CCO",
                "ESP_max": 1.0,
                "ESP_min": -1.0,
                "gap_eV": 2.0,
                "scc_prepeak_present": True,
                "source_list": "simulation",
            },
            {
                "SMILES": "CCN",
                "ESP_max": None,
                "ESP_min": -2.0,
                "gap_eV": 3.0,
                "scc_prepeak_present": False,
                "source_list": "simulation",
            },
        ],
    )

    summary = analyze_merged_properties(input_root, output_dir, min_holdout_systems=200, test_fraction=0.1)

    assert set(summary["property"]) == {"esp_max", "esp_min", "gap", "scc_prepeak_present"}
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

    prepeak = summary[summary["property"].eq("scc_prepeak_present")].iloc[0]
    assert prepeak["value_min"] == 0.0
    assert prepeak["value_mean"] == 0.5
    assert prepeak["value_max"] == 1.0

    plot_manifest = pd.read_csv(output_dir / "plot_manifest.csv")
    assert "property_distribution_1d" in set(plot_manifest["figure_type"])
    assert (
        output_dir / "figures" / "property_distributions_1d" / "low" / "simulation_esp_max_esp_max_value.png"
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
    assert "property_distribution_1d" in set(plot_manifest["figure_type"])
    assert (
        output_dir
        / "figures"
        / "property_distributions_1d"
        / "low"
        / "experiment_surface_tension_surface_tension_value.png"
    ).exists()
    assert (
        output_dir
        / "figures"
        / "property_distributions_2d"
        / "low"
        / "experiment_surface_tension_temperature_k_surface_tension_value.png"
    ).exists()


def test_normalize_property_values_uses_min_max_and_handles_constant_values():
    assert normalize_property_values(pd.Series([10.0, 20.0, 30.0])).tolist() == [0.0, 0.5, 1.0]
    assert normalize_property_values(pd.Series([5.0, 5.0])).tolist() == [0.5, 0.5]


def test_normalized_property_violin_uses_experiment_and_simulation_subplots(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}
    original_subplots = analysis_module.plt.subplots

    def recording_subplots(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        captured["axes"] = axes
        return fig, axes

    monkeypatch.setattr(analysis_module.plt, "subplots", recording_subplots)

    plot_normalized_property_violin(
        {
            "experiment": {"experiment/density": pd.Series([0.0, 0.5, 1.0])},
            "simulation": {"simulation/density": pd.Series([0.2, 0.4, 0.8])},
        },
        tmp_path / "violin.png",
        100,
    )

    axes = captured["axes"].ravel()
    assert len(axes) == 2
    assert [axis.get_title() for axis in axes] == ["Experiment", "Simulation"]
    assert [tuple(axis.get_ylim()) for axis in axes] == [(0.0, 1.0), (0.0, 1.0)]
    assert not axes[0].get_shared_x_axes().joined(axes[0], axes[1])


def test_numeric_condition_dimensions_only_fill_temperature_and_pressure_for_plotting():
    df = pd.DataFrame(
        {
            "temperature_K": [None, 310.0, 320.0, 330.0],
            "pressure_kPa": [101.0, None, 102.0, 103.0],
            "frequency_MHz": [1.0, None, 0.0, 10.0],
            "wavelength_nm": [589.0, None, 600.0, None],
            "density_g/cm^3": [1.0, 1.1, 1.2, 1.3],
        }
    )
    present = df["density_g/cm^3"].notna()

    dimensions = {str(dimension["slug"]): dimension for dimension in numeric_condition_dimensions(df, present)}

    assert dimensions["temperature_k"]["series"].isna().sum() == 0
    assert dimensions["temperature_k"]["series"].iloc[0] == 298.15
    assert dimensions["temperature_k"]["fill_note"] == "filled: temperature_K=1"

    assert dimensions["pressure_kpa"]["series"].isna().sum() == 0
    assert dimensions["pressure_kpa"]["series"].iloc[1] == 101.325
    assert dimensions["pressure_kpa"]["fill_note"] == "filled: pressure_kPa=1"

    frequency = dimensions["frequency_mhz"]
    assert frequency["label"] == "frequency_MHz"
    assert frequency["series"].isna().sum() == 1
    assert frequency["series"].dropna().tolist() == [1.0, 0.0, 10.0]
    assert "missing_tick" not in frequency
    assert "fill_note" not in frequency

    wavelength = dimensions["wavelength_nm"]
    assert wavelength["series"].isna().sum() == 2
    assert "missing_tick" not in wavelength
    assert "fill_note" not in wavelength


def test_plot_condition_spaces_keeps_frequency_in_mhz(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}

    def record_plot(plot_df, x_col, y_col, value_column, y_label, summary_row, output_path, dpi):
        captured["frequencies"] = plot_df[y_col].tolist()
        captured["y_label"] = y_label

    monkeypatch.setattr(analysis_module, "plot_condition_scatter", record_plot)
    outputs = plot_condition_spaces(
        pd.DataFrame(
            {
                "temperature_K": [298.15, 308.15],
                "frequency_MHz": [1000.0, 2000.0],
                "dynamic_relative_permittivity_unitless": [10.0, 8.0],
            }
        ),
        "dynamic_relative_permittivity_unitless",
        {"bucket": "experiment", "property": "dynamic_relative_permittivity"},
        tmp_path,
        "experiment_dynamic_relative_permittivity",
        set(),
        dpi=80,
        max_condition_scatter_points=100,
    )

    assert [kind for kind, _path in outputs] == ["temperature_frequency"]
    assert captured == {"frequencies": [1000.0, 2000.0], "y_label": "Frequency (MHz)"}


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
    assert (output_dir / "property_system_overlap.csv").exists()
    assert (output_dir / "property_analysis_report.md").exists()
    assert not (output_dir / "plot_manifest.csv").exists()
    assert not (output_dir / "figures").exists()
