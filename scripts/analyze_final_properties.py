"""Analyze final property CSVs using the merged-data analysis workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_merged_properties import analyze_property_manifest  # noqa: E402
from scripts.merge_data import label_columns  # noqa: E402

FINAL_BUCKETS = ("experiment", "simulation")


def build_final_manifest(input_root: Path) -> pd.DataFrame:
    input_root = Path(input_root)
    rows: list[dict[str, str]] = []
    for bucket in FINAL_BUCKETS:
        for path in sorted((input_root / bucket).glob("*.csv")):
            columns = pd.read_csv(path, nrows=0)
            for property_label in label_columns(columns):
                if property_label == "source_list":
                    continue
                rows.append(
                    {
                        "bucket": bucket,
                        "property_label": property_label,
                        "output_file": str(path.relative_to(input_root)),
                    }
                )
    return pd.DataFrame(rows, columns=["bucket", "property_label", "output_file"])


def analyze_final_properties(
    input_root: Path,
    output_dir: Path,
    *,
    min_holdout_systems: int = 200,
    test_fraction: float = 0.1,
    dpi: int = 300,
    high_coverage_threshold: int = 20000,
    medium_coverage_threshold: int = 2000,
    max_condition_scatter_points: int = 50000,
    skip_plots: bool = False,
) -> pd.DataFrame:
    input_root = Path(input_root)
    return analyze_property_manifest(
        input_root,
        output_dir,
        build_final_manifest(input_root),
        min_holdout_systems=min_holdout_systems,
        test_fraction=test_fraction,
        dpi=dpi,
        high_coverage_threshold=high_coverage_threshold,
        medium_coverage_threshold=medium_coverage_threshold,
        max_condition_scatter_points=max_condition_scatter_points,
        skip_plots=skip_plots,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "final")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "analysis")
    parser.add_argument("--min-holdout-systems", type=int, default=200)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--high-coverage-threshold", type=int, default=20000)
    parser.add_argument("--medium-coverage-threshold", type=int, default=2000)
    parser.add_argument("--max-condition-scatter-points", type=int, default=50000)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_final_properties(
        args.input_root,
        args.output_dir,
        min_holdout_systems=args.min_holdout_systems,
        test_fraction=args.test_fraction,
        dpi=args.dpi,
        high_coverage_threshold=args.high_coverage_threshold,
        medium_coverage_threshold=args.medium_coverage_threshold,
        max_condition_scatter_points=args.max_condition_scatter_points,
        skip_plots=args.skip_plots,
    )
    print(f"Wrote {len(summary)} final property analyses to {args.output_dir}")


if __name__ == "__main__":
    main()
