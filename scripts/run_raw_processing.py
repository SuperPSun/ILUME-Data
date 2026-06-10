"""Run the repository's standard raw-data preparation steps.

This entrypoint orchestrates the main reference, simulation, and ILThermo
structuring workflows, with an optional crawl step for ILThermo energetics.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEP_TO_SCRIPT = {
    "reference": "structuring_for_reference.py",
    "simulation": "structuring_for_simulation.py",
    "ilthermo": "structuring_ilthermo_data.py",
    "pure-energetics": "crawl_ilthermo_pure_energetics.py",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the repository-level raw-processing entrypoint."""

    parser = argparse.ArgumentParser(
        description="Run one or more raw-data preparation steps for the extracted AIonopedia2 workspace."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["all", *STEP_TO_SCRIPT.keys()],
        default=["reference", "simulation", "ilthermo"],
        help=(
            "Select which processing steps to run. Use 'reference' to structure reference CSV files, "
            "'simulation' to clean simulation CSV files, 'ilthermo' to structure all ILThermo property files, "
            "'pure-energetics' to crawl ILThermo pure-compound energetics, or 'all' to run every step in order."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected raw-data preparation scripts in sequence."""

    args = parse_args()
    steps = list(STEP_TO_SCRIPT.keys()) if "all" in args.steps else args.steps
    for step in steps:
        script_path = PROJECT_ROOT / "scripts" / STEP_TO_SCRIPT[step]
        print(f"Running {step}: {script_path}")
        subprocess.run([sys.executable, str(script_path)], check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()