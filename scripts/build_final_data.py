"""Copy merged experiment/simulation CSVs into final, excluding selected experiments."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHARGE_DATA_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "simulation_data" / "charge_20260514"
)
BUCKETS = ("experiment", "simulation")
EXCLUDED_EXPERIMENT_FILES = (
    "thermal_diffusivity.csv",
    "enthalpy.csv",
    "entropy.csv",
    "heat_capacity_at_vapor_saturation_pressure.csv",
    "enthalpy_of_vaporization_or_sublimation.csv",
    "enthalpy_of_transition_or_fusion.csv",
    "equilibrium_temperature.csv",
)
EXCLUDED_SIMULATION_FILES = ("3d_box.csv",)


def build_final_data(
    input_root: Path,
    output_root: Path,
    charge_data_root: Path = CHARGE_DATA_ROOT,
) -> None:
    """Rebuild ``output_root`` from merged data while omitting excluded files."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    charge_data_root = Path(charge_data_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output_root.parent) as temporary_dir:
        staged_root = Path(temporary_dir) / output_root.name
        for bucket in BUCKETS:
            shutil.copytree(input_root / bucket, staged_root / bucket)
        shutil.copytree(
            charge_data_root,
            staged_root / "simulation" / charge_data_root.name,
        )
        for filename in EXCLUDED_EXPERIMENT_FILES:
            (staged_root / "experiment" / filename).unlink()
        for filename in EXCLUDED_SIMULATION_FILES:
            (staged_root / "simulation" / filename).unlink(missing_ok=True)

        if output_root.exists():
            shutil.rmtree(output_root)
        staged_root.rename(output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "merged")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "final")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_final_data(args.input_root, args.output_root)


if __name__ == "__main__":
    main()
