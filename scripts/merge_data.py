"""Merge cleaned IL datasets into merged property-level experiment/simulation CSVs."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import re
import shutil
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import inchi

EXPERIMENT_SOURCES = ("AIonopedia", "ILBERT", "ILThermo", "after_AIonopedia")
SIMULATION_SOURCES = ("simulation",)
IDENTIFIER_COLUMNS = ("mol_id", "cation", "anion", "solute", "solvent", "smiles", "SMILES")
CONDITION_COLUMNS = ("temperature_K", "pressure_kPa", "frequency_MHz", "wavelength_nm", "phase")
ORBITAL_AUDIT_COLUMNS = (
    "ion_role",
    "provenance_source_file",
    "provenance_source_row",
)
METADATA_COLUMNS = ("standard_state_note", *ORBITAL_AUDIT_COLUMNS)
BASE_COLUMNS = (*IDENTIFIER_COLUMNS, *CONDITION_COLUMNS, *METADATA_COLUMNS)
NON_LABEL_COLUMNS = set(BASE_COLUMNS)
ERROR_LABEL_PATTERNS = ("_err", "_error", "stddev", "stderr")
WIDE_TABLE_FILES = {
    "3d_box_structured.csv": "3d_box",
    "simulated_QM_elec_HF_structured.csv": "simulated_QM_elec_HF",
}
SINGLE_ION_ORBITAL_FILES = {
    "simulated_HOMO+LUMO_PBE_TZVP_anions_structured.csv": ("anion", "anion"),
    "simulated_HOMO+LUMO_PBE_TZVP_cations_structured.csv": ("cation", "cation"),
}
ORBITAL_OUTPUTS = {"HOMO_eV": "homo", "LUMO_eV": "lumo"}
ORBITAL_GAP_TOLERANCE_EV = 1.0e-8
ORBITAL_GAP_SUMMARY_COLUMNS = (
    "source_file",
    "identity_column",
    "checked_rows",
    "exceeded_rows",
    "max_absolute_residual_eV",
    "tolerance_eV",
)
ORBITAL_GAP_ANOMALY_COLUMNS = (
    "source_file",
    "identity_column",
    "identity",
    "HOMO_eV",
    "LUMO_eV",
    "gap_eV",
    "absolute_residual_eV",
    "tolerance_eV",
)
PROPERTY_OUTPUT_SLUGS = {"pressure_kPa_log10": "equilibrium_pressure"}
PROPERTY_LABEL_ALIASES = {
    ("after_AIonopedia", "partition_log10"): "transfer_kcal/mol",
    ("simulation", "solvation_kcal/mol"): "transfer_organic_kcal/mol",
}
SOURCE_COLUMNS = {"source", "source_file"}
MISSING_TOKEN = "__ILUME_MISSING_CONDITION__"
DEFAULT_REFRACTIVE_INDEX_WAVELENGTH_NM = 589.0
DEFAULT_EXPERIMENT_PRESSURE_KPA = 101.325
EXPERIMENT_DEFAULT_PRESSURE_LABELS = frozenset(
    {
        "density_g/cm^3",
        "electrical_conductivity_S/m_log10",
        "heat_capacity_J/mol/K",
        "refractive_index_unitless",
        "thermal_conductivity_W/m/K",
        "viscosity_mPa*s_log10",
    }
)
SCALAR_VALUE_ABSOLUTE_TOLERANCE = Decimal("5e-7")
SOLVATION_REVISION_PAIR_TOLERANCE = Decimal("0.005")
FLOAT_SIGNIFICANT_DIGITS = 15
CHEMICAL_IDENTITY_AUDIT_COLUMNS = (
    "fixed_h_inchikey",
    "representative_smiles",
    "equivalent_smiles_list",
    "occurrence_count",
    "source_list",
    "source_file_list",
)
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


@lru_cache(maxsize=None)
def canonicalize_identity_smiles(smiles: str) -> str:
    text = str(smiles).strip()
    if not text:
        raise ValueError("empty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {text}")
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )


@lru_cache(maxsize=None)
def fixed_h_inchikey(smiles: str) -> str:
    canonical = canonicalize_identity_smiles(smiles)
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(canonical)
        key = (
            inchi.MolToInchiKey(molecule, options="/FixedH")
            if molecule is not None
            else ""
        )
    if not key:
        raise ValueError(f"unable to generate Fixed-H InChIKey: {canonical}")
    return key


def normalize_chemical_identities(
    buckets: dict[str, dict[str, list[pd.DataFrame]]],
) -> pd.DataFrame:
    observations: dict[str, dict[str, object]] = {}
    for properties in buckets.values():
        for frames in properties.values():
            for frame in frames:
                identity_columns = [
                    column
                    for column in IDENTIFIER_COLUMNS
                    if column in frame.columns and column != "mol_id"
                ]
                for column in identity_columns:
                    counts = frame[column].value_counts(dropna=False)
                    for value, count in counts.items():
                        if pd.isna(value) or not str(value).strip():
                            raise ValueError(
                                "missing chemical identity in "
                                f"{frame['source'].iloc[0]}/"
                                f"{frame['source_file'].iloc[0]}, column {column}"
                            )
                        try:
                            canonical = canonicalize_identity_smiles(str(value))
                            identity_key = fixed_h_inchikey(canonical)
                        except ValueError as exc:
                            raise ValueError(
                                "chemical identity normalization failed in "
                                f"{frame['source'].iloc[0]}/"
                                f"{frame['source_file'].iloc[0]}, column {column}: "
                                f"{exc}"
                            ) from exc
                        entry = observations.setdefault(
                            identity_key,
                            {
                                "smiles": set(),
                                "occurrence_count": 0,
                                "sources": set(),
                                "source_files": set(),
                            },
                        )
                        entry["smiles"].add(canonical)
                        entry["occurrence_count"] += int(count)
                        entry["sources"].add(str(frame["source"].iloc[0]))
                        entry["source_files"].add(
                            str(frame["source_file"].iloc[0])
                        )

    representatives = {
        identity_key: min(entry["smiles"])
        for identity_key, entry in observations.items()
    }
    for properties in buckets.values():
        for frames in properties.values():
            for frame in frames:
                for column in IDENTIFIER_COLUMNS:
                    if column not in frame.columns or column == "mol_id":
                        continue
                    replacements: dict[object, str] = {}
                    for value in pd.unique(frame[column]):
                        canonical = canonicalize_identity_smiles(str(value))
                        replacements[value] = representatives[
                            fixed_h_inchikey(canonical)
                        ]
                    frame[column] = frame[column].map(replacements)

    audit_rows = []
    for identity_key, entry in observations.items():
        equivalent_smiles = sorted(entry["smiles"])
        if len(equivalent_smiles) < 2:
            continue
        audit_rows.append(
            {
                "fixed_h_inchikey": identity_key,
                "representative_smiles": representatives[identity_key],
                "equivalent_smiles_list": "; ".join(equivalent_smiles),
                "occurrence_count": int(entry["occurrence_count"]),
                "source_list": "; ".join(sorted(entry["sources"])),
                "source_file_list": "; ".join(
                    sorted(entry["source_files"])
                ),
            }
        )
    return (
        pd.DataFrame(audit_rows, columns=CHEMICAL_IDENTITY_AUDIT_COLUMNS)
        .sort_values("fixed_h_inchikey", kind="stable")
        .reset_index(drop=True)
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


def output_slug(label: str) -> str:
    return PROPERTY_OUTPUT_SLUGS.get(label, property_slug(label))


def output_label(source: str, label: str) -> str:
    return PROPERTY_LABEL_ALIASES.get((source, label), label)


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


def same_system_condition_rows(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    key_columns = [column for column in BASE_COLUMNS if column in df.columns]
    output_columns = [
        *key_columns,
        *value_columns,
        "matching_entry_count",
        "source",
        "source_file",
    ]
    if not key_columns:
        return pd.DataFrame(columns=output_columns)

    group_sizes = df.groupby(key_columns, dropna=False, sort=False)[key_columns[0]].transform("size")
    matching = df.loc[
        group_sizes.gt(1),
        [*key_columns, *value_columns, "source", "source_file"],
    ].copy()
    if matching.empty:
        return pd.DataFrame(columns=output_columns)
    matching["matching_entry_count"] = group_sizes[group_sizes.gt(1)].to_numpy()
    return matching[output_columns].sort_values(key_columns, kind="stable", na_position="first").reset_index(drop=True)


def decimal_value(value: object) -> Decimal | None:
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return decimal if decimal.is_finite() else None


def meaningful_precision(value: object) -> int:
    decimal = Decimal(format(float(value), f".{FLOAT_SIGNIFICANT_DIGITS}g")).normalize()
    return len(decimal.as_tuple().digits)


def approximate_exclusion_columns(df: pd.DataFrame, label: str) -> list[str]:
    return [
        *[column for column in BASE_COLUMNS if column in df.columns],
        label,
        "retained_value",
        "absolute_difference",
        "source",
        "source_file",
    ]


def collapse_solvation_revision_pairs(
    df: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold nearest after-AIonopedia revisions onto retained AIonopedia values."""

    key_columns = [column for column in BASE_COLUMNS if column in df.columns]
    exclusion_columns = approximate_exclusion_columns(df, label)
    if not key_columns:
        return df, pd.DataFrame(columns=exclusion_columns)

    candidates = df[df.duplicated(subset=key_columns, keep=False)]
    if candidates.empty:
        return df, pd.DataFrame(columns=exclusion_columns)

    collapsed = df.copy()
    exclusion_rows: list[dict[str, object]] = []
    for _key, group in candidates.groupby(key_columns, dropna=False, sort=False):
        aionopedia_indices = group.index[group["source"].eq("AIonopedia")].tolist()
        revised_indices = group.index[group["source"].eq("after_AIonopedia")].tolist()
        possible_pairs: list[tuple[Decimal, int, int]] = []
        for aionopedia_index in aionopedia_indices:
            aionopedia_value = decimal_value(group.at[aionopedia_index, label])
            if aionopedia_value is None:
                continue
            for revised_index in revised_indices:
                revised_value = decimal_value(group.at[revised_index, label])
                if revised_value is None:
                    continue
                difference = abs(aionopedia_value - revised_value)
                if difference <= SOLVATION_REVISION_PAIR_TOLERANCE:
                    possible_pairs.append((difference, int(aionopedia_index), int(revised_index)))

        used_aionopedia: set[int] = set()
        used_revised: set[int] = set()
        for difference, aionopedia_index, revised_index in sorted(possible_pairs):
            if aionopedia_index in used_aionopedia or revised_index in used_revised:
                continue
            retained_value = collapsed.at[aionopedia_index, label]
            revised_value = collapsed.at[revised_index, label]
            if decimal_value(retained_value) != decimal_value(revised_value):
                exclusion_rows.append(
                    {
                        **{column: collapsed.at[revised_index, column] for column in key_columns},
                        label: revised_value,
                        "retained_value": retained_value,
                        "absolute_difference": difference,
                        "source": collapsed.at[revised_index, "source"],
                        "source_file": collapsed.at[revised_index, "source_file"],
                    }
                )
            collapsed.at[revised_index, label] = retained_value
            used_aionopedia.add(aionopedia_index)
            used_revised.add(revised_index)

    return collapsed, pd.DataFrame(exclusion_rows, columns=exclusion_columns)


def collapse_close_property_values(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_columns = [column for column in BASE_COLUMNS if column in df.columns]
    exclusion_columns = approximate_exclusion_columns(df, label)
    if not key_columns:
        return df, pd.DataFrame(columns=exclusion_columns)
    candidates = df[df.duplicated(subset=key_columns, keep=False)]
    if candidates.empty:
        return df, pd.DataFrame(columns=exclusion_columns)

    collapsed = df.copy()
    exclusion_rows: list[dict[str, object]] = []

    def collapse_cluster(cluster: list[tuple[object, Decimal]]) -> None:
        if len(cluster) < 2:
            return
        representative_index = max(
            (index for index, _value in cluster),
            key=lambda index: (meaningful_precision(collapsed.at[index, label]), -int(index)),
        )
        representative_value = collapsed.at[representative_index, label]
        representative_decimal = next(value for index, value in cluster if index == representative_index)
        for index, value in cluster:
            if value == representative_decimal:
                continue
            exclusion_rows.append(
                {
                    **{column: collapsed.at[index, column] for column in key_columns},
                    label: collapsed.at[index, label],
                    "retained_value": representative_value,
                    "absolute_difference": abs(value - representative_decimal),
                    "source": collapsed.at[index, "source"],
                    "source_file": collapsed.at[index, "source_file"],
                }
            )
        collapsed.loc[[index for index, _value in cluster], label] = representative_value

    for _key, group in candidates.groupby(key_columns, dropna=False, sort=False):
        numeric_values = [
            (index, decimal)
            for index, value in group[label].items()
            if (decimal := decimal_value(value)) is not None
        ]
        numeric_values.sort(key=lambda item: item[1])
        cluster: list[tuple[object, Decimal]] = []
        cluster_min: Decimal | None = None
        for item in numeric_values:
            if cluster_min is None or item[1] - cluster_min <= SCALAR_VALUE_ABSOLUTE_TOLERANCE:
                cluster.append(item)
                if cluster_min is None:
                    cluster_min = item[1]
                continue
            collapse_cluster(cluster)
            cluster = [item]
            cluster_min = item[1]
        collapse_cluster(cluster)
    return collapsed, pd.DataFrame(exclusion_rows, columns=exclusion_columns)


def aggregate_property(rows: list[pd.DataFrame], label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(rows, ignore_index=True)
    matching_rows = same_system_condition_rows(combined, [label])
    combined = collapse_condition_subsets(combined, [label])
    revision_exclusions = pd.DataFrame(columns=approximate_exclusion_columns(combined, label))
    if label == "solvation_kcal/mol":
        combined, revision_exclusions = collapse_solvation_revision_pairs(combined, label)
    combined, close_value_exclusions = collapse_close_property_values(combined, label)
    close_value_exclusions = pd.concat(
        [revision_exclusions, close_value_exclusions],
        ignore_index=True,
    )
    grouping_columns = [column for column in BASE_COLUMNS if column in combined.columns]
    grouping_columns.append(label)
    if not combined.duplicated(subset=grouping_columns).any():
        return direct_property_frame(combined, label).reset_index(drop=True), close_value_exclusions, matching_rows
    if combined["source"].nunique(dropna=False) == 1 and combined["source_file"].nunique(dropna=False) == 1:
        source = str(combined["source"].iloc[0])
        aggregated = combined[grouping_columns].drop_duplicates().copy()
        aggregated["source_list"] = source
        merged = aggregated[output_columns(aggregated, label)].sort_values(output_columns(aggregated, label)).reset_index(drop=True)
        return merged, close_value_exclusions, matching_rows
    aggregated = (
        combined.groupby(grouping_columns, dropna=False)
        .agg(
            source_list=("source", join_source_values),
        )
        .reset_index()
    )
    merged = aggregated[output_columns(aggregated, label)].sort_values(output_columns(aggregated, label)).reset_index(drop=True)
    return merged, close_value_exclusions, matching_rows


def aggregate_wide_table(rows: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(rows, ignore_index=True)
    labels = [
        column
        for column in combined.columns
        if column not in NON_LABEL_COLUMNS and column not in SOURCE_COLUMNS and not is_error_label(column)
    ]
    matching_rows = same_system_condition_rows(combined, labels)
    combined = collapse_condition_subsets(combined, labels)
    grouping_columns = [column for column in BASE_COLUMNS if column in combined.columns]
    grouping_columns.extend(labels)
    aggregated = (
        combined.groupby(grouping_columns, dropna=False)
        .agg(source_list=("source", join_source_values))
        .reset_index()
    )
    columns = wide_output_columns(aggregated, labels)
    merged = aggregated[columns].sort_values(columns).reset_index(drop=True)
    return merged, matching_rows


def aggregate_qm_elec_hf(rows: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(rows, ignore_index=True)
    labels = [
        column
        for column in combined.columns
        if column not in NON_LABEL_COLUMNS and column not in SOURCE_COLUMNS and not is_error_label(column)
    ]
    matching_rows = same_system_condition_rows(combined, labels)
    if "SMILES" not in combined.columns:
        raise ValueError("simulated_QM_elec_HF requires an SMILES column")

    numeric = combined[["SMILES", *labels]].copy()
    for label in labels:
        numeric[label] = pd.to_numeric(numeric[label], errors="coerce")
    medians = numeric.groupby("SMILES", dropna=False, sort=False)[labels].median().reset_index()
    sources = (
        combined.groupby("SMILES", dropna=False, sort=False)
        .agg(source_list=("source", join_source_values))
        .reset_index()
    )
    aggregated = medians.merge(sources, on="SMILES", how="left", validate="one_to_one")
    columns = wide_output_columns(aggregated, labels)
    return aggregated[columns].sort_values("SMILES").reset_index(drop=True), matching_rows


def collect_bucket(
    input_root: Path,
    sources: tuple[str, ...],
    *,
    default_pressure_labels: frozenset[str] = frozenset(),
) -> tuple[dict[str, list[pd.DataFrame]], list[dict[str, object]], list[dict[str, object]]]:
    properties: dict[str, list[pd.DataFrame]] = {}
    orbital_gap_summaries: list[dict[str, object]] = []
    orbital_gap_anomalies: list[dict[str, object]] = []
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
            orbital_spec = SINGLE_ION_ORBITAL_FILES.get(path.name)
            if orbital_spec is not None:
                identity_column, ion_role = orbital_spec
                required = {identity_column, "HOMO_eV", "LUMO_eV", "gap_eV"}
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(
                        f"{path.name} is missing orbital columns: {sorted(missing)}"
                    )
                numeric = df[["HOMO_eV", "LUMO_eV", "gap_eV"]].apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                residuals = (numeric["gap_eV"] - (numeric["LUMO_eV"] - numeric["HOMO_eV"])).abs()
                anomaly_mask = residuals.gt(ORBITAL_GAP_TOLERANCE_EV) | residuals.isna()
                finite_residuals = residuals.dropna()
                orbital_gap_summaries.append(
                    {
                        "source_file": f"{source}/{path.name}",
                        "identity_column": identity_column,
                        "checked_rows": len(df),
                        "exceeded_rows": int(anomaly_mask.sum()),
                        "max_absolute_residual_eV": (
                            float(finite_residuals.max()) if not finite_residuals.empty else pd.NA
                        ),
                        "tolerance_eV": ORBITAL_GAP_TOLERANCE_EV,
                    }
                )
                for index in df.index[anomaly_mask]:
                    orbital_gap_anomalies.append(
                        {
                            "source_file": f"{source}/{path.name}",
                            "identity_column": identity_column,
                            "identity": df.at[index, identity_column],
                            "HOMO_eV": df.at[index, "HOMO_eV"],
                            "LUMO_eV": df.at[index, "LUMO_eV"],
                            "gap_eV": df.at[index, "gap_eV"],
                            "absolute_residual_eV": residuals.at[index],
                            "tolerance_eV": ORBITAL_GAP_TOLERANCE_EV,
                        }
                    )
                provenance_rows = pd.Series(
                    range(2, len(df) + 2), index=df.index, dtype="int64"
                )
                for target, task_label in ORBITAL_OUTPUTS.items():
                    orbital = df.loc[
                        df[target].notna(), [identity_column, target]
                    ].rename(columns={identity_column: "SMILES"})
                    if orbital.empty:
                        continue
                    orbital["ion_role"] = ion_role
                    orbital["provenance_source_file"] = (
                        f"{source}/{path.name}"
                    )
                    orbital["provenance_source_row"] = provenance_rows.loc[
                        orbital.index
                    ]
                    orbital["source"] = source
                    orbital["source_file"] = path.name
                    properties.setdefault(task_label, []).append(orbital)
                continue
            for label in label_columns(df):
                keep_columns = [column for column in BASE_COLUMNS if column in df.columns]
                property_df = df.loc[df[label].notna(), [*keep_columns, label]].copy()
                if property_df.empty:
                    continue
                target_label = output_label(source, label)
                if target_label != label:
                    property_df = property_df.rename(columns={label: target_label})
                if target_label in default_pressure_labels:
                    if "pressure_kPa" not in property_df.columns:
                        property_df["pressure_kPa"] = DEFAULT_EXPERIMENT_PRESSURE_KPA
                    else:
                        property_df["pressure_kPa"] = property_df["pressure_kPa"].fillna(
                            DEFAULT_EXPERIMENT_PRESSURE_KPA
                        )
                if label == "refractive_index_unitless":
                    if "wavelength_nm" not in property_df.columns:
                        property_df["wavelength_nm"] = DEFAULT_REFRACTIVE_INDEX_WAVELENGTH_NM
                    else:
                        property_df["wavelength_nm"] = property_df["wavelength_nm"].fillna(
                            DEFAULT_REFRACTIVE_INDEX_WAVELENGTH_NM
                        )
                property_df["source"] = source
                property_df["source_file"] = path.name
                properties.setdefault(target_label, []).append(property_df)
    return properties, orbital_gap_summaries, orbital_gap_anomalies


def clean_output_root(output_root: Path) -> None:
    for subdir in (
        "experiment",
        "simulation",
        "rejected_rows",
        "same_system_condition_rows",
        "_audit",
    ):
        path = output_root / subdir
        if path.exists():
            shutil.rmtree(path)
    manifest = output_root / "merged_manifest.csv"
    if manifest.exists():
        manifest.unlink()
    system_counts = output_root / "system_property_counts.csv"
    if system_counts.exists():
        system_counts.unlink()
    identity_audit = output_root / "chemical_identity_equivalences.csv"
    if identity_audit.exists():
        identity_audit.unlink()


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
        if label == "simulated_QM_elec_HF":
            merged, matching_rows = aggregate_qm_elec_hf(rows)
            close_value_exclusions = pd.DataFrame()
        elif label in set(WIDE_TABLE_FILES.values()) | set(
            ORBITAL_OUTPUTS.values()
        ):
            merged, matching_rows = aggregate_wide_table(rows)
            close_value_exclusions = pd.DataFrame()
        else:
            merged, close_value_exclusions, matching_rows = aggregate_property(rows, label)
        output_name = f"{output_slug(label)}.csv"
        previous_label = output_labels.get(output_name)
        if previous_label is not None and previous_label != label:
            raise ValueError(
                f"Property filename collision for {bucket}/{output_name}: "
                f"{previous_label!r} and {label!r}"
            )
        output_labels[output_name] = label
        output_path = bucket_dir / output_name
        merged.to_csv(output_path, index=False)
        if not close_value_exclusions.empty:
            rejected_dir = output_root / "rejected_rows" / bucket
            rejected_dir.mkdir(parents=True, exist_ok=True)
            rejected_name = f"{output_slug(label)}_approximate_values_rejected.csv"
            close_value_exclusions.to_csv(rejected_dir / rejected_name, index=False)
        if not matching_rows.empty:
            matching_dir = output_root / "same_system_condition_rows" / bucket
            matching_dir.mkdir(parents=True, exist_ok=True)
            matching_rows.to_csv(matching_dir / output_name, index=False)
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


def merge_data(input_root: Path, output_root: Path) -> list[dict[str, object]]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    clean_output_root(output_root)

    experiment, experiment_gap_summaries, experiment_gap_anomalies = collect_bucket(
        input_root,
        EXPERIMENT_SOURCES,
        default_pressure_labels=EXPERIMENT_DEFAULT_PRESSURE_LABELS,
    )
    simulation, simulation_gap_summaries, simulation_gap_anomalies = collect_bucket(
        input_root,
        SIMULATION_SOURCES,
    )
    buckets = {"experiment": experiment, "simulation": simulation}
    identity_audit = normalize_chemical_identities(buckets)

    manifest_rows: list[dict[str, object]] = []
    manifest_rows.extend(
        write_bucket("experiment", buckets["experiment"], output_root)
    )
    manifest_rows.extend(
        write_bucket("simulation", buckets["simulation"], output_root)
    )
    manifest = pd.DataFrame(
        manifest_rows,
        columns=["bucket", "property_label", "output_file", "input_files", "input_rows", "output_rows"],
    )
    manifest.to_csv(output_root / "merged_manifest.csv", index=False)
    identity_audit.to_csv(
        output_root / "chemical_identity_equivalences.csv",
        index=False,
    )
    audit_root = output_root / "_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [*experiment_gap_summaries, *simulation_gap_summaries],
        columns=ORBITAL_GAP_SUMMARY_COLUMNS,
    ).to_csv(audit_root / "orbital_gap_consistency_summary.csv", index=False)
    pd.DataFrame(
        [*experiment_gap_anomalies, *simulation_gap_anomalies],
        columns=ORBITAL_GAP_ANOMALY_COLUMNS,
    ).to_csv(audit_root / "orbital_gap_consistency_anomalies.csv", index=False)
    return manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--output-root", type=Path, default=Path("data/merged"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = merge_data(args.input_root, args.output_root)
    for row in rows:
        print(
            f"bucket={row['bucket']} property={row['property_label']} "
            f"rows={row['output_rows']} output={row['output_file']}"
        )


if __name__ == "__main__":
    main()
