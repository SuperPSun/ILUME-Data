from pathlib import Path

import pandas as pd

from scripts.analyze_final_properties import analyze_final_properties


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_analyze_final_properties_scans_final_files_without_manifest(tmp_path: Path):
    input_root = tmp_path / "final"
    output_dir = tmp_path / "analysis"
    write_csv(
        input_root / "experiment" / "density.csv",
        [
            {"cation": "cat1", "anion": "an1", "density_g/cm^3": 1.0, "source_list": "ILBERT"},
            {"cation": "cat2", "anion": "an2", "density_g/cm^3": 1.2, "source_list": "ILThermo"},
        ],
    )
    write_csv(
        input_root / "simulation" / "simulated_qm_elec_hf.csv",
        [
            {"SMILES": "CCO", "ESP_max": 1.0, "ESP_min": -1.0, "gap_eV": 3.0, "source_list": "simulation"},
            {"SMILES": "CCN", "ESP_max": 2.0, "ESP_min": -2.0, "gap_eV": 4.0, "source_list": "simulation"},
        ],
    )
    write_csv(
        input_root / "simulation" / "gap.csv",
        [
            {"cation": "cat1", "anion": "an1", "gap_eV": 3.0, "source_list": "simulation"},
            {"cation": "cat2", "anion": "an2", "gap_eV": 4.0, "source_list": "simulation"},
        ],
    )

    summary = analyze_final_properties(input_root, output_dir)

    assert set(summary["property"]) == {"density", "esp_max", "esp_min", "gap"}
    assert len(summary[summary["property"].eq("gap")]) == 2
    assert set(summary["bucket"]) == {"experiment", "simulation"}
    assert (output_dir / "property_analysis_summary.csv").exists()
    assert (output_dir / "property_analysis_report.md").exists()
    assert (output_dir / "figures" / "normalized_distributions" / "property_normalized_violin.png").exists()
    manifest = pd.read_csv(output_dir / "plot_manifest.csv")
    assert "normalized_property_violin" in set(manifest["figure_type"])


def test_analyze_final_properties_handles_boolean_values(tmp_path: Path):
    input_root = tmp_path / "final"
    output_dir = tmp_path / "analysis"
    write_csv(
        input_root / "simulation" / "3d_box.csv",
        [
            {"SMILES": "CCO", "scc_prepeak_present": True, "source_list": "simulation"},
            {"SMILES": "CCN", "scc_prepeak_present": False, "source_list": "simulation"},
        ],
    )

    summary = analyze_final_properties(input_root, output_dir)

    row = summary.iloc[0]
    assert row["property"] == "scc_prepeak_present"
    assert row["value_min"] == 0.0
    assert row["value_mean"] == 0.5
    assert row["value_max"] == 1.0
    assert (output_dir / "plot_manifest.csv").exists()
