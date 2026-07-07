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
BASE_COLUMNS = (*IDENTIFIER_COLUMNS, *CONDITION_COLUMNS)
NON_LABEL_COLUMNS = set(BASE_COLUMNS)
ERROR_LABEL_PATTERNS = ("_err", "_error", "stddev", "stderr")
WIDE_TABLE_FILES = {"simulated_QM_elec_HF_structured.csv": "simulated_QM_elec_HF"}
SOURCE_COLUMNS = {"source", "source_file"}
MISSING_TOKEN = "__ILUME_MISSING_CONDITION__"
UNIT_SUFFIXES = (
    "_10^-9*m^2/s",
    "_J/mol/K",
    "_kJ/mol",
    "_kcal/mol",
    "_g/cm^3",
    "_kg/m^3",
    "_mPa*s",
    "_mN/m",
    "_S/m",
    "_W/m/K",
    "_m^2/s",
    "_K^-1",
    "_unitless",
    "_kPa",
    "_MHz",
    "_nm",
    "_m/s",
    "_eV",
    "_K",
)


def property_name(label: str) -> str:
    name = re.sub(r"_log10$", "", label, flags=re.IGNORECASE)
    lower = name.lower()
    for suffix in UNIT_SUFFIXES:
        if lower.endswith(suffix.lower()):
            return name[: -len(suffix)]
    return name


def property_slug(label: str) -> str:
    slug = property_name(label).lower()
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
        *[column for column in BASE_COLUMNS if column in df.columns],
        label,
        "source_list",
    ]


def wide_output_columns(df: pd.DataFrame, labels: list[str]) -> list[str]:
    return [
        *[column for column in BASE_COLUMNS if column in df.columns],
        *labels,
        "source_list",
    ]


def join_unique(values: pd.Series) -> str:
    unique = sorted({str(value) for value in values.dropna() if str(value)})
    return "; ".join(unique)


def join_source_values(values: pd.Series) -> str:
    parts: set[str] = set()
    for value in values.dropna():
        for part in str(value).split(";"):
            part = part.strip()
            if part:
                parts.add(part)
    return "; ".join(sorted(parts))


def collapse_condition_subsets(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    condition_columns = [column for column in CONDITION_COLUMNS if column in df.columns]
    if not condition_columns or not df[condition_columns].isna().any().any():
        return df

    key_columns = [column for column in IDENTIFIER_COLUMNS if column in df.columns]
    key_columns.extend(value_columns)
    signature_columns = [*key_columns, *condition_columns]
    signatures = df[signature_columns].astype("object")
    signatures = signatures.where(signatures.notna(), MISSING_TOKEN)
    unique_signatures = signatures.drop_duplicates().reset_index(drop=True)
    unique_signatures["_signature_id"] = range(len(unique_signatures))
    target_conditions = unique_signatures[condition_columns].copy()
    missing_patterns = unique_signatures[condition_columns].eq(MISSING_TOKEN)
    pattern_tuples = pd.Series(
        list(missing_patterns.itertuples(index=False, name=None)),
        index=missing_patterns.index,
    )
    changed = False

    for pattern in missing_patterns.drop_duplicates().itertuples(index=False, name=None):
        if not any(pattern):
            continue
        missing_columns = [column for column, is_missing in zip(condition_columns, pattern) if is_missing]
        present_columns = [column for column, is_missing in zip(condition_columns, pattern) if not is_missing]
        match_columns = [*key_columns, *present_columns]
        partial_mask = pattern_tuples.map(lambda value: value == pattern)
        partial = unique_signatures.loc[partial_mask, ["_signature_id", *match_columns]]
        if partial.empty:
            continue

        candidates = unique_signatures.rename(columns={"_signature_id": "_candidate_signature_id"})
        candidates = candidates[["_candidate_signature_id", *match_columns, *missing_columns]]
        merged = partial.merge(candidates, on=match_columns, how="inner")
        merged = merged[merged["_signature_id"] != merged["_candidate_signature_id"]]
        fills_missing = merged[missing_columns].ne(MISSING_TOKEN).any(axis=1)
        merged = merged[fills_missing]
        if merged.empty:
            continue

        candidate_signatures = merged[["_signature_id", *condition_columns]].drop_duplicates()
        counts = candidate_signatures.groupby("_signature_id", sort=False).size()
        unambiguous_ids = counts[counts == 1].index
        updates = candidate_signatures[candidate_signatures["_signature_id"].isin(unambiguous_ids)]
        if updates.empty:
            continue
        target_conditions.loc[updates["_signature_id"], condition_columns] = updates[condition_columns].to_numpy()
        changed = True

    if not changed:
        return df

    signature_targets = unique_signatures[[*signature_columns, "_signature_id"]].copy()
    for column in condition_columns:
        signature_targets[f"_target_{column}"] = target_conditions[column]
    row_targets = signatures.reset_index(names="_row_index").merge(
        signature_targets,
        on=signature_columns,
        how="left",
    )
    collapsed = df.copy()
    for column in condition_columns:
        values = row_targets[f"_target_{column}"].replace(MISSING_TOKEN, pd.NA)
        values.index = row_targets["_row_index"]
        collapsed[column] = values.reindex(collapsed.index).to_numpy()
    grouping_columns = [column for column in collapsed.columns if column not in SOURCE_COLUMNS]
    return (
        collapsed.groupby(grouping_columns, dropna=False, sort=False)
        .agg(source=("source", join_source_values), source_file=("source_file", join_unique))
        .reset_index()
    )


def direct_property_frame(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    out["source_list"] = out.pop("source")
    out = out.drop(columns=["source_file"])
    return out[output_columns(out, label)]


def aggregate_property(rows: list[pd.DataFrame], label: str) -> pd.DataFrame:
    combined = pd.concat(rows, ignore_index=True)
    combined = collapse_condition_subsets(combined, [label])
    grouping_columns = [column for column in BASE_COLUMNS if column in combined.columns]
    grouping_columns.append(label)
    if not combined.duplicated(subset=grouping_columns).any():
        return direct_property_frame(combined, label).reset_index(drop=True)
    if combined["source"].nunique(dropna=False) == 1 and combined["source_file"].nunique(dropna=False) == 1:
        source = str(combined["source"].iloc[0])
        aggregated = combined[grouping_columns].drop_duplicates().copy()
        aggregated["source_list"] = source
        return aggregated[output_columns(aggregated, label)].sort_values(output_columns(aggregated, label)).reset_index(drop=True)
    aggregated = (
        combined.groupby(grouping_columns, dropna=False)
        .agg(
            source_list=("source", join_source_values),
        )
        .reset_index()
    )
    return aggregated[output_columns(aggregated, label)].sort_values(output_columns(aggregated, label)).reset_index(drop=True)


def aggregate_wide_table(rows: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(rows, ignore_index=True)
    labels = [
        column
        for column in combined.columns
        if column not in NON_LABEL_COLUMNS and column not in SOURCE_COLUMNS and not is_error_label(column)
    ]
    combined = collapse_condition_subsets(combined, labels)
    grouping_columns = [column for column in BASE_COLUMNS if column in combined.columns]
    grouping_columns.extend(labels)
    aggregated = (
        combined.groupby(grouping_columns, dropna=False)
        .agg(source_list=("source", join_source_values))
        .reset_index()
    )
    columns = wide_output_columns(aggregated, labels)
    return aggregated[columns].sort_values(columns).reset_index(drop=True)


def collect_bucket(input_root: Path, sources: tuple[str, ...]) -> dict[str, list[pd.DataFrame]]:
    properties: dict[str, list[pd.DataFrame]] = {}
    for source in sources:
        source_dir = input_root / source
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.glob("*.csv")):
            df = pd.read_csv(path)
            wide_key = WIDE_TABLE_FILES.get(path.name)
            if wide_key is not None:
                labels = label_columns(df)
                keep_columns = [column for column in BASE_COLUMNS if column in df.columns]
                wide_df = df.loc[df[labels].notna().any(axis=1), [*keep_columns, *labels]].copy()
                if wide_df.empty:
                    continue
                wide_df["source"] = source
                wide_df["source_file"] = path.name
                properties.setdefault(wide_key, []).append(wide_df)
                continue
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
    system_counts = output_root / "system_property_counts.csv"
    if system_counts.exists():
        system_counts.unlink()


def write_bucket(
    bucket: str,
    properties: dict[str, list[pd.DataFrame]],
    output_root: Path,
) -> list[dict[str, object]]:
    bucket_dir = output_root / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    output_labels: dict[str, str] = {}
    for label, rows in sorted(properties.items()):
        if label in set(WIDE_TABLE_FILES.values()):
            merged = aggregate_wide_table(rows)
        else:
            merged = aggregate_property(rows, label)
        output_name = f"{property_slug(label)}.csv"
        previous_label = output_labels.get(output_name)
        if previous_label is not None and previous_label != label:
            raise ValueError(
                f"Property filename collision for {bucket}/{output_name}: "
                f"{previous_label!r} and {label!r}"
            )
        output_labels[output_name] = label
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
