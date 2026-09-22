from pathlib import Path

import pandas as pd

from analysis.final_data_quality.audit import pressure_missingness


def test_pressure_missingness_returns_schema_when_no_pressure_is_missing(
    tmp_path: Path,
):
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "cation": ["C[N+](C)(C)C"],
            "anion": ["C(=O)[O-]"],
            "temperature_K": [298.15],
            "pressure_kPa": [101.325],
            "density_g/cm^3": [1.0],
        }
    ).to_csv(experiment_root / "density.csv", index=False)

    result = pressure_missingness(tmp_path)

    assert result.empty
    assert list(result.columns) == [
        "file",
        "rows",
        "missing_pressure_rows",
        "missing_pressure_rate",
    ]
