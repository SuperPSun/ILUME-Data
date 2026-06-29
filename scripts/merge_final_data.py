"""Merge cleaned IL datasets into final property-level experiment/simulation CSVs."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

EXPERIMENT_SOURCES = ("AIonopedia", "ILBERT", "ILThermo", "after_AIonopedia")
SIMULATION_SOURCES = ("simulation",)
IDENTIFIER_COLUMNS = ("mol_id", "cation", "anion", "solute", "solvent", "smiles", "SMILES")
CONDITION_COLUMNS = ("temperature_K", "pressure_kPa", "frequency_MHz", "wavelength_nm", "phase")
PROVENANCE_COLUMNS = ("source_list", "source_file_list", "source_record_count")
BASE_COLUMNS = (*IDENTIFIER_COLUMNS, *CONDITION_COLUMNS)
NON_LABEL_COLUMNS = set(BASE_COLUMNS)
ERROR_LABEL_PATTERNS = ("_err", "_error", "stddev", "stderr")


def property_slug(label: str) -> str:
    slug = label.lower()
    slug = slug.replace("/", "_per_")
    slug = slug.replace("*", "_")
    slug = slug.replace("^", "_pow_")
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_")


def is_error_label(label: str) -> bool:
    lower = label.lower()
    return any(pattern in lower for pattern in ERROR_LABEL_PATTERNS)


def label_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in NON_LABEL_COLUMNS and not is_error_label(column)]


def output_columns(df: pd.DataFrame, label: str) -> list[str]:
    return [
        *PROVENANCE_COLUMNS,
        *[column for column in BASE_COLUMNS if column in df.columns],
        label,
    ]


def join_unique(values: pd.Series) -> str:
    unique = sorted({str(value) for value in values.dropna() if str(value)})
    return "; ".join(unique)


def direct_property_frame(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    out["source_list"] = out.pop("source")
    out["source_file_list"] = out.pop("source_file")
    out["source_record_count"] = 1
    return out[output_columns(out, label)]


def aggregate_property(rows: list[pd.DataFrame], label: str) -> pd.DataFrame:
    combined = pd.concat(rows, ignore_index=True)
    grouping_columns = [column for column in BASE_COLUMNS if column in combined.columns]
    grouping_columns.append(label)
    if not combined.duplicated(subset=grouping_columns).any():
        return direct_property_frame(combined, label).reset_index(drop=True)
    if combined["source"].nunique(dropna=False) == 1 and combined["source_file"].nunique(dropna=False) == 1:
        source = str(combined["source"].iloc[0])
        source_file = str(combined["source_file"].iloc[0])
        aggregated = combined.groupby(grouping_columns, dropna=False).size().reset_index(name="source_record_count")
        aggregated["source_list"] = source
        aggregated["source_file_list"] = source_file
        return aggregated[output_columns(aggregated, label)].sort_values(output_columns(aggregated, label)).reset_index(drop=True)
    aggregated = (
        combined.groupby(grouping_columns, dropna=False)
        .agg(
            source_list=("source", join_unique),
            source_file_list=("source_file", join_unique),
            source_record_count=("source", "size"),
        )
        .reset_index()
    )
    return aggregated[output_columns(aggregated, label)].sort_values(output_columns(aggregated, label)).reset_index(drop=True)


def collect_bucket(input_root: Path, sources: tuple[str, ...]) -> dict[str, list[pd.DataFrame]]:
    properties: dict[str, list[pd.DataFrame]] = {}
    for source in sources:
        source_dir = input_root / source
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.glob("*.csv")):
            df = pd.read_csv(path)
            for label in label_columns(df):
                keep_columns = [column for column in BASE_COLUMNS if column in df.columns]
                property_df = df.loc[df[label].notna(), [*keep_columns, label]].copy()
                if property_df.empty:
                    continue
                property_df["source"] = source
                property_df["source_file"] = path.name
                properties.setdefault(label, []).append(property_df)
    return properties


def clean_output_root(output_root: Path) -> None:
    for subdir in ("experiment", "simulation"):
        path = output_root / subdir
        if path.exists():
            shutil.rmtree(path)
    manifest = output_root / "final_manifest.csv"
    if manifest.exists():
        manifest.unlink()


def write_bucket(
    bucket: str,
    properties: dict[str, list[pd.DataFrame]],
    output_root: Path,
) -> list[dict[str, object]]:
    bucket_dir = output_root / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for label, rows in sorted(properties.items()):
        merged = aggregate_property(rows, label)
        output_name = f"{property_slug(label)}.csv"
        output_path = bucket_dir / output_name
        merged.to_csv(output_path, index=False)
        input_files = sorted(
            {f"{str(row['source'].iloc[0])}/{str(row['source_file'].iloc[0])}" for row in rows}
        )
        manifest_rows.append(
            {
                "bucket": bucket,
                "property_label": label,
                "output_file": f"{bucket}/{output_name}",
                "input_files": "; ".join(input_files),
                "input_rows": int(sum(len(row) for row in rows)),
                "output_rows": int(len(merged)),
            }
        )
    return manifest_rows


def merge_final_data(input_root: Path, output_root: Path) -> list[dict[str, object]]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    clean_output_root(output_root)

    manifest_rows: list[dict[str, object]] = []
    manifest_rows.extend(write_bucket("experiment", collect_bucket(input_root, EXPERIMENT_SOURCES), output_root))
    manifest_rows.extend(write_bucket("simulation", collect_bucket(input_root, SIMULATION_SOURCES), output_root))
    manifest = pd.DataFrame(
        manifest_rows,
        columns=["bucket", "property_label", "output_file", "input_files", "input_rows", "output_rows"],
    )
    manifest.to_csv(output_root / "final_manifest.csv", index=False)
    return manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--output-root", type=Path, default=Path("data/final"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = merge_final_data(args.input_root, args.output_root)
    for row in rows:
        print(
            f"bucket={row['bucket']} property={row['property_label']} "
            f"rows={row['output_rows']} output={row['output_file']}"
        )


if __name__ == "__main__":
    main()
