"""Analyze merged property CSVs for data quality and split planning."""

from __future__ import annotations

import argparse
from itertools import combinations
import math
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_data import (
    CONDITION_COLUMNS,
    IDENTIFIER_COLUMNS,
    NON_LABEL_COLUMNS,
    WIDE_TABLE_FILES,
    is_error_label,
    output_slug,
)

SUMMARY_COLUMNS = [
    "bucket",
    "property",
    "property_label",
    "output_file",
    "rows",
    "data_points",
    "unique_systems",
    "points_per_system_mean",
    "points_per_system_median",
    "points_per_system_p95",
    "max_points_per_system",
    "multi_point_systems",
    "multi_point_system_ratio",
    "condition_columns",
    "condition_complete_rows",
    "unique_condition_sets",
    "source_count",
    "value_min",
    "value_p05",
    "value_p25",
    "value_median",
    "value_mean",
    "value_p75",
    "value_p95",
    "value_max",
    "value_std",
    "leakage_risk",
    "recommended_split",
    "recommended_test_systems",
]

NUMERIC_CONDITION_COLUMNS = tuple(column for column in CONDITION_COLUMNS if column != "phase")
PLOTTING_CONDITION_DEFAULTS = {
    "temperature_K": 298.15,
    "pressure_kPa": 101.325,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/merged"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/merged/analysis"))
    parser.add_argument("--min-holdout-systems", type=int, default=200)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--high-coverage-threshold", type=int, default=20000)
    parser.add_argument("--medium-coverage-threshold", type=int, default=2000)
    parser.add_argument("--max-condition-scatter-points", type=int, default=50000)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def value_columns_for_row(df: pd.DataFrame, property_label: str) -> list[str]:
    if property_label in set(WIDE_TABLE_FILES.values()):
        return [
            column
            for column in df.columns
            if column not in NON_LABEL_COLUMNS and column != "source_list" and not is_error_label(column)
        ]
    return [property_label]


def source_count(df: pd.DataFrame) -> int:
    if "source_list" not in df.columns:
        return 0
    sources: set[str] = set()
    for value in df["source_list"].dropna():
        for part in str(value).split(";"):
            part = part.strip()
            if part:
                sources.add(part)
    return len(sources)


def numeric_property_values(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if pd.api.types.is_bool_dtype(numeric.dtype):
        return numeric.astype(float)
    return numeric


def recommendation(unique_systems: int, min_holdout_systems: int, test_fraction: float) -> tuple[str, int]:
    if unique_systems >= min_holdout_systems:
        return "system_holdout_test", max(1, math.ceil(unique_systems * test_fraction))
    if unique_systems >= 20:
        return "grouped_cross_validation", 0
    return "leave_one_system_out_or_descriptive_only", 0


def analyze_value_column(
    *,
    bucket: str,
    output_file: str,
    df: pd.DataFrame,
    value_column: str,
    min_holdout_systems: int,
    test_fraction: float,
) -> dict[str, object]:
    values = numeric_property_values(df[value_column])
    present = values.notna()
    rows_with_values = df.loc[present].copy()
    identifier_columns = [column for column in IDENTIFIER_COLUMNS if column in rows_with_values.columns]
    condition_columns = [column for column in CONDITION_COLUMNS if column in df.columns]

    if identifier_columns and not rows_with_values.empty:
        system_counts = rows_with_values.groupby(identifier_columns, dropna=False).size()
        unique_systems = int(len(system_counts))
    else:
        system_counts = pd.Series(dtype="int64")
        unique_systems = 0

    if condition_columns:
        condition_complete_rows = int(rows_with_values[condition_columns].notna().all(axis=1).sum())
        unique_condition_sets = int(rows_with_values[condition_columns].drop_duplicates().shape[0])
        condition_column_text = "; ".join(condition_columns)
    else:
        condition_complete_rows = int(present.sum())
        unique_condition_sets = 0
        condition_column_text = ""

    multi_point_systems = int((system_counts > 1).sum()) if not system_counts.empty else 0
    multi_point_system_ratio = multi_point_systems / unique_systems if unique_systems else 0.0
    max_points_per_system = int(system_counts.max()) if not system_counts.empty else 0
    leakage_risk = "high" if multi_point_system_ratio >= 0.5 or max_points_per_system >= 20 else "low"
    recommended_split, recommended_test_systems = recommendation(unique_systems, min_holdout_systems, test_fraction)
    clean_values = values[present]

    return {
        "bucket": bucket,
        "property": output_slug(value_column),
        "property_label": value_column,
        "output_file": output_file,
        "rows": int(len(df)),
        "data_points": int(present.sum()),
        "unique_systems": unique_systems,
        "points_per_system_mean": float(system_counts.mean()) if not system_counts.empty else 0.0,
        "points_per_system_median": float(system_counts.median()) if not system_counts.empty else 0.0,
        "points_per_system_p95": float(system_counts.quantile(0.95)) if not system_counts.empty else 0.0,
        "max_points_per_system": max_points_per_system,
        "multi_point_systems": multi_point_systems,
        "multi_point_system_ratio": float(multi_point_system_ratio),
        "condition_columns": condition_column_text,
        "condition_complete_rows": condition_complete_rows,
        "unique_condition_sets": unique_condition_sets,
        "source_count": source_count(df),
        "value_min": float(clean_values.min()) if not clean_values.empty else pd.NA,
        "value_p05": float(clean_values.quantile(0.05)) if not clean_values.empty else pd.NA,
        "value_p25": float(clean_values.quantile(0.25)) if not clean_values.empty else pd.NA,
        "value_median": float(clean_values.median()) if not clean_values.empty else pd.NA,
        "value_mean": float(clean_values.mean()) if not clean_values.empty else pd.NA,
        "value_p75": float(clean_values.quantile(0.75)) if not clean_values.empty else pd.NA,
        "value_p95": float(clean_values.quantile(0.95)) if not clean_values.empty else pd.NA,
        "value_max": float(clean_values.max()) if not clean_values.empty else pd.NA,
        "value_std": float(clean_values.std()) if len(clean_values) > 1 else 0.0,
        "leakage_risk": leakage_risk,
        "recommended_split": recommended_split,
        "recommended_test_systems": recommended_test_systems,
    }


def analyze_property_manifest(
    input_root: Path,
    output_dir: Path,
    manifest: pd.DataFrame,
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
    output_dir = Path(output_dir)

    rows: list[dict[str, object]] = []
    for manifest_row in manifest.itertuples(index=False):
        output_file = str(manifest_row.output_file)
        property_label = str(manifest_row.property_label)
        path = input_root / output_file
        df = pd.read_csv(path)
        for value_column in value_columns_for_row(df, property_label):
            if value_column not in df.columns:
                continue
            rows.append(
                analyze_value_column(
                    bucket=str(manifest_row.bucket),
                    output_file=output_file,
                    df=df,
                    value_column=value_column,
                    min_holdout_systems=min_holdout_systems,
                    test_fraction=test_fraction,
                )
            )

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "property_analysis_summary.csv", index=False)
    if skip_plots:
        clean_plot_outputs(output_dir)
        plot_manifest = pd.DataFrame()
    else:
        plot_manifest = generate_plots(
            input_root,
            output_dir,
            manifest,
            summary,
            dpi=dpi,
            high_coverage_threshold=high_coverage_threshold,
            medium_coverage_threshold=medium_coverage_threshold,
            max_condition_scatter_points=max_condition_scatter_points,
        )
        plot_manifest.to_csv(output_dir / "plot_manifest.csv", index=False)
    write_markdown_report(summary, output_dir / "property_analysis_report.md", plot_manifest)
    return summary


def analyze_merged_properties(
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
    manifest_path = input_root / "merged_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing merged manifest: {manifest_path}")
    return analyze_property_manifest(
        input_root,
        output_dir,
        pd.read_csv(manifest_path),
        min_holdout_systems=min_holdout_systems,
        test_fraction=test_fraction,
        dpi=dpi,
        high_coverage_threshold=high_coverage_threshold,
        medium_coverage_threshold=medium_coverage_threshold,
        max_condition_scatter_points=max_condition_scatter_points,
        skip_plots=skip_plots,
    )


def clean_plot_outputs(output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    plot_manifest = output_dir / "plot_manifest.csv"
    if plot_manifest.exists():
        plot_manifest.unlink()


def coverage_group(data_points: int, high_threshold: int, medium_threshold: int) -> str:
    if data_points >= high_threshold:
        return "high"
    if data_points >= medium_threshold:
        return "medium"
    return "low"


def format_count(value: object) -> str:
    number = float(value)
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if number >= 1_000:
        return f"{number / 1_000:.1f}k"
    return str(int(number))


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def unique_path(path: Path, used_paths: set[Path]) -> Path:
    if path not in used_paths:
        used_paths.add(path)
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if candidate not in used_paths:
            used_paths.add(candidate)
            return candidate
        index += 1


def plot_record(
    records: list[dict[str, object]],
    *,
    figure_type: str,
    bucket: str,
    property_name: str,
    path: Path,
    output_dir: Path,
    data_points: int,
    unique_systems: int,
    coverage: str,
) -> None:
    records.append(
        {
            "figure_type": figure_type,
            "bucket": bucket,
            "property": property_name,
            "path": str(path.relative_to(output_dir)),
            "data_points": data_points,
            "unique_systems": unique_systems,
            "coverage_group": coverage,
        }
    )


def generate_plots(
    input_root: Path,
    output_dir: Path,
    manifest: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    dpi: int,
    high_coverage_threshold: int,
    medium_coverage_threshold: int,
    max_condition_scatter_points: int,
) -> pd.DataFrame:
    clean_plot_outputs(output_dir)
    figures_dir = output_dir / "figures"
    plot_records: list[dict[str, object]] = []
    used_paths: set[Path] = set()
    summary = summary.copy()
    summary["coverage_group"] = summary["data_points"].map(
        lambda value: coverage_group(int(value), high_coverage_threshold, medium_coverage_threshold)
    )

    for group_name, group_df in [
        ("all", summary),
        ("high", summary[summary["coverage_group"].eq("high")]),
        ("medium", summary[summary["coverage_group"].eq("medium")]),
        ("low", summary[summary["coverage_group"].eq("low")]),
    ]:
        path = figures_dir / "coverage" / f"property_coverage_{group_name}.png"
        plot_coverage(group_df, path, dpi, title=f"Property Coverage ({group_name})")
        plot_record(
            plot_records,
            figure_type="coverage",
            bucket="all",
            property_name=group_name,
            path=path,
            output_dir=output_dir,
            data_points=int(group_df["data_points"].sum()) if not group_df.empty else 0,
            unique_systems=int(group_df["unique_systems"].sum()) if not group_df.empty else 0,
            coverage=group_name,
        )

    availability_rows: list[dict[str, object]] = []
    normalized_values: dict[str, dict[str, pd.Series]] = {
        "experiment": {},
        "simulation": {},
    }
    summary_by_key = {
        (str(row.output_file), str(row.property_label)): row._asdict()
        for row in summary.itertuples(index=False)
    }
    for manifest_row in manifest.itertuples(index=False):
        output_file = str(manifest_row.output_file)
        property_label = str(manifest_row.property_label)
        df = pd.read_csv(input_root / output_file)
        for value_column in value_columns_for_row(df, property_label):
            if value_column not in df.columns:
                continue
            summary_row = summary_by_key.get((output_file, value_column))
            if summary_row is None:
                continue
            plot_property_figures(
                df,
                value_column,
                summary_row,
                figures_dir,
                output_dir,
                plot_records,
                used_paths,
                dpi=dpi,
                max_condition_scatter_points=max_condition_scatter_points,
            )
            normalized = normalize_property_values(df[value_column])
            if not normalized.empty:
                bucket = str(summary_row["bucket"])
                bucket_values = normalized_values.setdefault(bucket, {})
                normalized_label = f"{bucket}/{summary_row['property']}"
                if normalized_label in bucket_values:
                    normalized_label = f"{normalized_label} ({Path(output_file).stem})"
                bucket_values[normalized_label] = normalized
            availability_rows.append(condition_availability_row(df, value_column, summary_row))

    violin_path = figures_dir / "normalized_distributions" / "property_normalized_violin.png"
    plot_normalized_property_violin(normalized_values, violin_path, dpi)
    plot_record(
        plot_records,
        figure_type="normalized_property_violin",
        bucket="all",
        property_name="normalized_properties",
        path=violin_path,
        output_dir=output_dir,
        data_points=sum(
            len(values)
            for bucket_values in normalized_values.values()
            for values in bucket_values.values()
        ),
        unique_systems=int(summary["unique_systems"].sum()) if not summary.empty else 0,
        coverage="all",
    )

    heatmap_path = figures_dir / "condition_availability" / "property_condition_availability_heatmap.png"
    plot_condition_availability(availability_rows, heatmap_path, dpi)
    plot_record(
        plot_records,
        figure_type="condition_availability",
        bucket="all",
        property_name="condition_availability",
        path=heatmap_path,
        output_dir=output_dir,
        data_points=int(summary["data_points"].sum()) if not summary.empty else 0,
        unique_systems=int(summary["unique_systems"].sum()) if not summary.empty else 0,
        coverage="all",
    )
    return pd.DataFrame(
        plot_records,
        columns=["figure_type", "bucket", "property", "path", "data_points", "unique_systems", "coverage_group"],
    )


def plot_coverage(summary: pd.DataFrame, output_path: Path, dpi: int, *, title: str) -> None:
    fig_width = max(10, min(24, 0.45 * max(len(summary), 1)))
    fig, ax = plt.subplots(figsize=(fig_width, 7))
    if summary.empty:
        ax.text(0.5, 0.5, "No properties in this group", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        save_figure(fig, output_path, dpi)
        return

    plot_df = summary.sort_values("data_points", ascending=False).reset_index(drop=True)
    x = list(range(len(plot_df)))
    width = 0.42
    bars_points = ax.bar([value - width / 2 for value in x], plot_df["data_points"], width, label="Data points", color="#34699A")
    bars_systems = ax.bar([value + width / 2 for value in x], plot_df["unique_systems"], width, label="Unique systems", color="#D9822B")
    if plot_df[["data_points", "unique_systems"]].to_numpy().max() / max(1, plot_df[["data_points", "unique_systems"]].to_numpy().min()) > 100:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels((plot_df["bucket"] + "/" + plot_df["property"]).tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.bar_label(bars_points, labels=[format_count(v) for v in plot_df["data_points"]], fontsize=6, padding=2, rotation=90)
    ax.bar_label(bars_systems, labels=[format_count(v) for v in plot_df["unique_systems"]], fontsize=6, padding=2, rotation=90)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path, dpi)


def plot_property_figures(
    df: pd.DataFrame,
    value_column: str,
    summary_row: dict[str, object],
    figures_dir: Path,
    output_dir: Path,
    plot_records: list[dict[str, object]],
    used_paths: set[Path],
    *,
    dpi: int,
    max_condition_scatter_points: int,
) -> None:
    bucket = str(summary_row["bucket"])
    property_name = str(summary_row["property"])
    coverage = str(summary_row["coverage_group"])
    base_name = f"{bucket}_{property_name}"

    for figure_type, distribution_path in plot_property_distributions(
        df,
        value_column,
        summary_row,
        figures_dir,
        base_name,
        coverage,
        used_paths,
        dpi=dpi,
    ):
        plot_record(
            plot_records,
            figure_type=figure_type,
            bucket=bucket,
            property_name=property_name,
            path=distribution_path,
            output_dir=output_dir,
            data_points=int(summary_row["data_points"]),
            unique_systems=int(summary_row["unique_systems"]),
            coverage=coverage,
        )

    system_counts = system_frequency(df, value_column)
    if not system_counts.empty:
        freq_path = unique_path(figures_dir / "system_frequency" / coverage / f"{base_name}.png", used_paths)
        plot_system_frequency(system_counts, summary_row, freq_path, dpi)
        plot_record(
            plot_records,
            figure_type="system_frequency",
            bucket=bucket,
            property_name=property_name,
            path=freq_path,
            output_dir=output_dir,
            data_points=int(summary_row["data_points"]),
            unique_systems=int(summary_row["unique_systems"]),
            coverage=coverage,
        )

def axis_slug(label: str) -> str:
    slug = label.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_")


def smooth_counts(counts: np.ndarray) -> np.ndarray:
    if len(counts) < 3:
        return counts.astype(float)
    window = min(5, len(counts))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(counts.astype(float), kernel, mode="same")


def histogram_bin_count(values: pd.Series, *, max_bins: int = 40) -> int:
    unique_values = int(values.nunique())
    return min(max_bins, max(5, unique_values))


def normalize_property_values(values: pd.Series) -> pd.Series:
    numeric = numeric_property_values(values).replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return numeric.astype(float)
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if maximum == minimum:
        return pd.Series(0.5, index=numeric.index, dtype=float)
    return (numeric - minimum) / (maximum - minimum)


def plot_histogram_with_density(
    ax: plt.Axes,
    values: pd.Series,
    bins: int | np.ndarray,
    *,
    orientation: str,
    color: str,
) -> None:
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    smoothed = smooth_counts(counts)
    if orientation == "horizontal":
        ax.barh(edges[:-1], counts, height=widths, align="edge", color=color, alpha=0.75, edgecolor="white")
        ax.plot(smoothed, centers, color="#222222", linewidth=1.2)
    else:
        ax.bar(edges[:-1], counts, width=widths, align="edge", color=color, alpha=0.75, edgecolor="white")
        ax.plot(centers, smoothed, color="#222222", linewidth=1.2)


def numeric_condition_dimensions(df: pd.DataFrame, present: pd.Series) -> list[dict[str, object]]:
    dimensions: list[dict[str, object]] = []
    for column in NUMERIC_CONDITION_COLUMNS:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        label = column
        slug = axis_slug(column)

        fill_note = ""
        missing_mask = present & series.isna()
        missing_count = int(missing_mask.sum())
        if column in PLOTTING_CONDITION_DEFAULTS and missing_count:
            series = series.copy()
            series.loc[missing_mask] = PLOTTING_CONDITION_DEFAULTS[column]
            fill_note = f"filled: {column}={missing_count}"

        if series.loc[present].notna().any():
            dimension: dict[str, object] = {"label": label, "slug": slug, "series": series}
            if fill_note:
                dimension["fill_note"] = fill_note
            dimensions.append(dimension)
    return dimensions


def value_dimension(df: pd.DataFrame, value_column: str, summary_row: dict[str, object]) -> dict[str, object]:
    property_name = str(summary_row["property"])
    return {
        "label": str(summary_row["property_label"]),
        "slug": f"{axis_slug(property_name)}_value",
        "series": numeric_property_values(df[value_column]),
    }


def plot_property_distributions(
    df: pd.DataFrame,
    value_column: str,
    summary_row: dict[str, object],
    figures_dir: Path,
    base_name: str,
    coverage: str,
    used_paths: set[Path],
    *,
    dpi: int,
) -> list[tuple[str, Path]]:
    outputs: list[tuple[str, Path]] = []
    value_dim = value_dimension(df, value_column, summary_row)
    value_series = value_dim["series"]
    present = value_series.notna()
    if not present.any():
        return outputs

    condition_dims = numeric_condition_dimensions(df, present)
    distribution_1d_dir = figures_dir / "property_distributions_1d" / coverage
    distribution_2d_dir = figures_dir / "property_distributions_2d" / coverage
    values = value_series.loc[present]
    path = unique_path(distribution_1d_dir / f"{base_name}_{value_dim['slug']}.png", used_paths)
    plot_property_distribution_1d(values, value_dim, summary_row, path, dpi)
    outputs.append(("property_distribution_1d", path))
    if not condition_dims:
        return outputs

    dimensions = [*condition_dims, value_dim]
    for x_dim, y_dim in combinations(dimensions, 2):
        plot_df = pd.DataFrame(
            {
                "x": x_dim["series"],
                "y": y_dim["series"],
                "value": value_series,
            }
        ).dropna()
        if plot_df.empty:
            continue
        path = unique_path(distribution_2d_dir / f"{base_name}_{x_dim['slug']}_{y_dim['slug']}.png", used_paths)
        plot_property_distribution_2d(plot_df, x_dim, y_dim, summary_row, path, dpi)
        outputs.append(("property_distribution_2d", path))
    return outputs


def plot_property_distribution_1d(
    values: pd.Series,
    value_dim: dict[str, object],
    summary_row: dict[str, object],
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = histogram_bin_count(values)
    plot_histogram_with_density(ax, values, bins, orientation="vertical", color="#34699A")
    ax.set_title(
        f"{summary_row['bucket']}/{summary_row['property']} value distribution\n"
        f"n={format_count(summary_row['data_points'])}, systems={format_count(summary_row['unique_systems'])}, "
        f"split={summary_row['recommended_split']}"
    )
    ax.set_xlabel(str(value_dim["label"]))
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path, dpi)


def plot_normalized_property_violin(
    normalized_values: dict[str, dict[str, pd.Series]],
    output_path: Path,
    dpi: int,
) -> None:
    buckets = ("experiment", "simulation")
    max_properties = max((len(normalized_values.get(bucket, {})) for bucket in buckets), default=1)
    fig_width = max(16, min(60, 0.5 * max_properties))
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 16), squeeze=False)

    for index, bucket in enumerate(buckets):
        ax = axes[index, 0]
        bucket_values = normalized_values.get(bucket, {})
        labels = list(bucket_values)
        if not labels:
            ax.text(0.5, 0.5, "No properties to plot", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylim(0, 1)
            ax.set_title(bucket.capitalize())
            continue

        variable_positions: list[int] = []
        variable_values: list[np.ndarray] = []
        constant_positions: list[int] = []
        constant_values: list[float] = []
        for position, label in enumerate(labels, start=1):
            values = bucket_values[label].to_numpy(dtype=float)
            if len(np.unique(values)) > 1:
                variable_positions.append(position)
                variable_values.append(values)
            else:
                constant_positions.append(position)
                constant_values.append(float(values[0]))

        if variable_values:
            parts = ax.violinplot(
                variable_values,
                positions=variable_positions,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor("#34699A")
                body.set_edgecolor("#1F3F5B")
                body.set_alpha(0.7)
            parts["cmedians"].set_color("#222222")

        if constant_positions:
            ax.scatter(constant_positions, constant_values, color="#D9822B", s=18, zorder=3)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(bucket.capitalize())
        ax.set_xlabel("Property")
        ax.set_ylabel("Min-Max normalized value")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Normalized Property Distributions")
    save_figure(fig, output_path, dpi)


def plot_property_distribution_2d(
    plot_df: pd.DataFrame,
    x_dim: dict[str, object],
    y_dim: dict[str, object],
    summary_row: dict[str, object],
    output_path: Path,
    dpi: int,
) -> None:
    x_bins = histogram_bin_count(plot_df["x"], max_bins=60)
    y_bins = histogram_bin_count(plot_df["y"], max_bins=60)
    _, x_edges = np.histogram(plot_df["x"], bins=x_bins)
    _, y_edges = np.histogram(plot_df["y"], bins=y_bins)

    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1.25),
        height_ratios=(1.25, 4),
        hspace=0.08,
        wspace=0.08,
    )
    ax_histx = fig.add_subplot(grid[0, 0])
    ax_main = fig.add_subplot(grid[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(grid[1, 1], sharey=ax_main)

    hist = ax_main.hist2d(plot_df["x"], plot_df["y"], bins=[x_edges, y_edges], cmap="YlGnBu", cmin=1)
    cbar = fig.colorbar(hist[3], ax=ax_main)
    cbar.set_label("Frequency")

    plot_histogram_with_density(ax_histx, plot_df["x"], x_edges, orientation="vertical", color="#34699A")
    plot_histogram_with_density(ax_histy, plot_df["y"], y_edges, orientation="horizontal", color="#D9822B")
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ax_main.set_xlabel(str(x_dim["label"]))
    ax_main.set_ylabel(str(y_dim["label"]))
    ax_histx.set_ylabel("Frequency")
    ax_histy.set_xlabel("Frequency")
    ax_main.grid(alpha=0.2)
    ax_histx.grid(axis="y", alpha=0.2)
    ax_histy.grid(axis="x", alpha=0.2)
    title = (
        f"{summary_row['bucket']}/{summary_row['property']} distribution: "
        f"{x_dim['label']} vs {y_dim['label']}"
    )
    ax_histx.set_title(title)
    save_figure(fig, output_path, dpi)


def system_frequency(df: pd.DataFrame, value_column: str) -> pd.Series:
    values = numeric_property_values(df[value_column])
    rows = df.loc[values.notna()]
    identifier_columns = [column for column in IDENTIFIER_COLUMNS if column in rows.columns]
    if not identifier_columns or rows.empty:
        return pd.Series(dtype="int64")
    return rows.groupby(identifier_columns, dropna=False).size()


def plot_system_frequency(system_counts: pd.Series, summary_row: dict[str, object], output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    max_count = int(system_counts.max())
    if max_count <= 1:
        bins = [0.5, 1.5]
    else:
        bins = min(40, max_count)
    ax.hist(system_counts, bins=bins, color="#5F8D4E", alpha=0.85, edgecolor="white")
    ax.set_title(
        f"{summary_row['bucket']}/{summary_row['property']} system frequency\n"
        f"max={summary_row['max_points_per_system']}, leakage={summary_row['leakage_risk']}"
    )
    ax.set_xlabel("Records per system")
    ax.set_ylabel("System count")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path, dpi)


def condition_availability_row(df: pd.DataFrame, value_column: str, summary_row: dict[str, object]) -> dict[str, object]:
    values = numeric_property_values(df[value_column])
    present = values.notna()
    denominator = int(present.sum())
    row = {"label": f"{summary_row['bucket']}/{summary_row['property']}"}
    for column in CONDITION_COLUMNS:
        if denominator and column in df.columns:
            row[column] = float(df.loc[present, column].notna().sum() / denominator)
        else:
            row[column] = 0.0
    return row


def plot_condition_availability(rows: list[dict[str, object]], output_path: Path, dpi: int) -> None:
    labels = [str(row["label"]) for row in rows]
    matrix = [[float(row[column]) for column in CONDITION_COLUMNS] for row in rows]
    fig_height = max(6, min(24, 0.28 * max(len(rows), 1)))
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    if not rows:
        ax.text(0.5, 0.5, "No properties to plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        save_figure(fig, output_path, dpi)
        return
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(len(CONDITION_COLUMNS)))
    ax.set_xticklabels(CONDITION_COLUMNS, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Property x Condition Availability")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Completeness")
    save_figure(fig, output_path, dpi)


def plot_condition_spaces(
    df: pd.DataFrame,
    value_column: str,
    summary_row: dict[str, object],
    figures_dir: Path,
    base_name: str,
    used_paths: set[Path],
    *,
    dpi: int,
    max_condition_scatter_points: int,
) -> list[tuple[str, Path]]:
    outputs: list[tuple[str, Path]] = []
    pairs = [
        ("temperature_pressure", "temperature_K", "pressure_kPa", "Pressure (kPa)"),
        ("temperature_frequency", "temperature_K", "frequency_MHz", "Frequency (MHz)"),
        ("temperature_wavelength", "temperature_K", "wavelength_nm", "Wavelength (nm)"),
    ]
    for kind, x_col, y_col, y_label in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        values = numeric_property_values(df[value_column])
        plot_df = pd.DataFrame(
            {
                x_col: pd.to_numeric(df[x_col], errors="coerce"),
                y_col: pd.to_numeric(df[y_col], errors="coerce"),
                value_column: values,
            }
        ).dropna()
        if plot_df.empty:
            continue
        if len(plot_df) > max_condition_scatter_points:
            plot_df = plot_df.sample(max_condition_scatter_points, random_state=0)
        path = unique_path(figures_dir / "condition_space" / f"{base_name}_{kind}.png", used_paths)
        plot_condition_scatter(plot_df, x_col, y_col, value_column, y_label, summary_row, path, dpi)
        outputs.append((kind, path))
    return outputs


def plot_condition_scatter(
    plot_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_column: str,
    y_label: str,
    summary_row: dict[str, object],
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    scatter = ax.scatter(
        plot_df[x_col],
        plot_df[y_col],
        c=plot_df[value_column],
        s=12,
        alpha=0.65,
        cmap="viridis",
        edgecolors="none",
    )
    ax.set_title(f"{summary_row['bucket']}/{summary_row['property']} condition coverage")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_label)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(value_column)
    ax.grid(alpha=0.25)
    save_figure(fig, output_path, dpi)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> list[str]:
    if df.empty:
        return ["None."]
    subset = df.loc[:, columns].head(max_rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in subset.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_markdown_report(summary: pd.DataFrame, output_path: Path, plot_manifest: pd.DataFrame | None = None) -> None:
    total_points = int(summary["data_points"].sum()) if not summary.empty else 0
    total_systems = int(summary["unique_systems"].sum()) if not summary.empty else 0
    holdout = summary[summary["recommended_split"].eq("system_holdout_test")].sort_values("data_points", ascending=False)
    cv = summary[summary["recommended_split"].eq("grouped_cross_validation")].sort_values("unique_systems")
    small = summary[summary["recommended_split"].eq("leave_one_system_out_or_descriptive_only")].sort_values("unique_systems")
    high_risk = summary[summary["leakage_risk"].eq("high")].sort_values(
        ["multi_point_system_ratio", "max_points_per_system"],
        ascending=False,
    )
    largest = summary.sort_values("data_points", ascending=False)

    lines = [
        "# Merged Property Analysis Report",
        "",
        "## Global Summary",
        "",
        f"- Analyzed properties: {len(summary)}",
        f"- Total data points across analyzed properties: {total_points}",
        f"- Sum of property-level unique systems: {total_systems}",
        f"- Properties recommended for system holdout test: {len(holdout)}",
        f"- Properties recommended for grouped CV: {len(cv)}",
        f"- Properties recommended for leave-one-system-out or descriptive analysis: {len(small)}",
        "",
        "## Largest Properties",
        "",
        *markdown_table(largest, ["bucket", "property", "data_points", "unique_systems", "recommended_split"]),
        "",
        "## High Leakage Risk Properties",
        "",
        "These properties should not be split by random rows because repeated systems can cross train/test boundaries.",
        "",
        *markdown_table(
            high_risk,
            ["bucket", "property", "unique_systems", "multi_point_system_ratio", "max_points_per_system"],
        ),
        "",
        "## System Holdout Test Candidates",
        "",
        *markdown_table(holdout, ["bucket", "property", "unique_systems", "recommended_test_systems"]),
        "",
        "## Grouped Cross-Validation Candidates",
        "",
        *markdown_table(cv, ["bucket", "property", "unique_systems", "data_points"]),
        "",
        "## Small Properties",
        "",
        *markdown_table(small, ["bucket", "property", "unique_systems", "data_points"]),
        "",
        "## Generated Figures",
        "",
    ]
    if plot_manifest is None or plot_manifest.empty:
        lines.append("Plot generation skipped or no figures were generated.")
    else:
        counts = plot_manifest["figure_type"].value_counts().sort_index()
        for figure_type, count in counts.items():
            lines.append(f"- {figure_type}: {int(count)}")
        lines.extend(
            [
                "",
                "Key summary figures:",
                "- `figures/coverage/property_coverage_all.png`",
                "- `figures/condition_availability/property_condition_availability_heatmap.png`",
                "- `figures/normalized_distributions/property_normalized_violin.png`",
            ]
        )
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    summary = analyze_merged_properties(
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
    print(f"Wrote {len(summary)} property analyses to {args.output_dir}")


if __name__ == "__main__":
    main()
