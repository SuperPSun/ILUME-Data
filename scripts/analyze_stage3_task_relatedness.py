"""Analyze directed Stage-3 task relatedness before training-data splits."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TextIO

import numpy as np
import pandas as pd
import scipy
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_STAGE3_TASK_IDS = (
    "experiment/density",
    "experiment/dynamic_relative_permittivity",
    "experiment/electrical_conductivity",
    "experiment/equilibrium_pressure",
    "experiment/glass_transition_temperature",
    "experiment/heat_capacity",
    "experiment/isobaric_coefficient_of_volume_expansion",
    "experiment/melting_point",
    "experiment/pec50",
    "experiment/refractive_index",
    "experiment/self_diffusion_coefficient",
    "experiment/solvation",
    "experiment/speed_of_sound",
    "experiment/static_relative_permittivity",
    "experiment/surface_tension",
    "experiment/thermal_conductivity",
    "experiment/thermal_decomposition_temperature",
    "experiment/transfer",
    "experiment/transfer_organic",
    "experiment/viscosity",
    "experiment/x_co2",
)

ROLE_ORDER = ("cation", "anion", "solute", "solvent")
SUPPORTED_TOPOLOGIES = {
    ("cation", "anion"): "il",
    ("cation", "anion", "solute"): "il_solute",
    ("solute", "solvent"): "solute_solvent",
}
NUMERIC_CONDITIONS = (
    "temperature_K",
    "pressure_kPa",
    "frequency_MHz",
    "wavelength_nm",
)
CONDITION_ORDER = (*NUMERIC_CONDITIONS, "phase")
LOG_CONDITIONS = {"pressure_kPa", "frequency_MHz", "wavelength_nm"}
METADATA_COLUMNS = {
    "source_list",
    "mol_id",
    "ion_role",
    "provenance_source_file",
    "provenance_source_row",
}

CHEMISTRY_WEIGHT = 0.7
CONDITION_WEIGHT = 0.3
MISSING_CONDITION_SIMILARITY = 0.5
DEFAULT_EXPERIMENT_PRESSURE_KPA = 101.325
DEFAULT_REFRACTIVE_INDEX_WAVELENGTH_NM = 589.0
DEFAULT_PRESSURE_TASKS = {
    "experiment/density",
    "experiment/electrical_conductivity",
    "experiment/heat_capacity",
    "experiment/refractive_index",
    "experiment/thermal_conductivity",
    "experiment/viscosity",
}

MATCH_COLUMNS = (
    "source_task",
    "target_task",
    "source_row",
    "target_row",
    *(f"source_{role}" for role in ROLE_ORDER),
    *(f"target_{role}" for role in ROLE_ORDER),
    *(f"source_{condition}" for condition in CONDITION_ORDER),
    *(f"target_{condition}" for condition in CONDITION_ORDER),
    "source_property",
    "target_property",
    "source_value",
    "target_value",
    "chemistry_similarity",
    "condition_similarity",
    "combined_similarity",
)


class TaskRelatednessError(RuntimeError):
    """Raised when Stage-3 inputs do not satisfy the analysis contract."""


class _PairProgress:
    """Single-line task-pair progress for interactive terminals."""

    def __init__(
        self,
        total_pairs: int,
        total_work: int,
        stream: TextIO | None = None,
    ):
        self.total_pairs = total_pairs
        self.total_work = total_work
        self.stream = stream or sys.stderr
        self.enabled = self.stream.isatty()
        self.completed_pairs = 0
        self.completed_work = 0
        self.pair_work = 0
        self.pair_completed_work = 0
        self.started_at = time.monotonic()
        self.last_rendered_at = -math.inf
        self.current = ""

    def start_pair(
        self,
        source_task: str,
        target_task: str,
        pair_work: int,
    ) -> None:
        self.current = (
            f"{source_task.removeprefix('experiment/')} -> "
            f"{target_task.removeprefix('experiment/')}"
        )
        self.pair_work = pair_work
        self.pair_completed_work = 0
        self._render(force=True)

    def advance_work(self) -> None:
        self.completed_work += 1
        self.pair_completed_work += 1
        self._render(force=False)

    def finish_pair(self) -> None:
        self.completed_pairs += 1
        self._render(force=True)

    def close(self) -> None:
        if not self.enabled:
            return
        self.current = (
            "done" if self.completed_pairs == self.total_pairs else "stopped"
        )
        self._render(force=True)
        self.stream.write("\n")
        self.stream.flush()

    def _render(self, *, force: bool) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_rendered_at < 0.2:
            return
        self.last_rendered_at = now
        fraction = self.completed_work / self.total_work if self.total_work else 1.0
        filled = min(24, int(fraction * 24))
        bar = "#" * filled + "-" * (24 - filled)
        elapsed = max(0, int(now - self.started_at))
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        pair_number = min(self.completed_pairs + 1, self.total_pairs)
        self.stream.write(
            f"\rWork [{bar}] {self.completed_work:>7}/{self.total_work} "
            f"({fraction:>6.1%}) elapsed {hours:02}:{minutes:02}:{seconds:02} "
            f"pair {pair_number:>3}/{self.total_pairs} "
            f"current: {self.current} "
            f"rows {self.pair_completed_work}/{self.pair_work}\033[K"
        )
        self.stream.flush()


@dataclass
class TaskData:
    task_id: str
    path: Path
    roles: tuple[str, ...]
    topology: str
    condition_columns: tuple[str, ...]
    target_column: str
    frame: pd.DataFrame
    property_values: np.ndarray
    condition_values: dict[str, np.ndarray]
    chemistry_keys: list[tuple[str, ...]]
    chemistry_groups: list[np.ndarray]
    chemistry_fingerprints: list[tuple[object, ...]]


@dataclass(frozen=True)
class PairMatches:
    target_indices: np.ndarray
    chemistry_similarity: np.ndarray
    condition_similarity: np.ndarray
    combined_similarity: np.ndarray


def _transform_condition(column: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if column not in LOG_CONDITIONS:
        return values
    finite = np.isfinite(values)
    if np.any(values[finite] <= -1.0):
        raise TaskRelatednessError(
            f"{column} contains values outside the log1p domain"
        )
    transformed = values.copy()
    transformed[finite] = np.log1p(transformed[finite])
    return transformed


def _normalise_phase(value: object) -> object:
    if pd.isna(value):
        return None
    text = str(value).strip().casefold()
    return text or None


def _fingerprint(
    smiles: str,
    cache: dict[str, object],
    generator: object,
    *,
    task_id: str,
    role: str,
) -> object:
    if smiles in cache:
        return cache[smiles]
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise TaskRelatednessError(
            f"Invalid {role} SMILES in {task_id}: {smiles!r}"
        )
    fingerprint = generator.GetFingerprint(molecule)
    cache[smiles] = fingerprint
    return fingerprint


def _apply_semantic_defaults(task_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if task_id in DEFAULT_PRESSURE_TASKS:
        if "pressure_kPa" not in frame:
            frame["pressure_kPa"] = DEFAULT_EXPERIMENT_PRESSURE_KPA
        else:
            frame["pressure_kPa"] = frame["pressure_kPa"].fillna(
                DEFAULT_EXPERIMENT_PRESSURE_KPA
            )
    if task_id == "experiment/refractive_index":
        if "wavelength_nm" not in frame:
            frame["wavelength_nm"] = DEFAULT_REFRACTIVE_INDEX_WAVELENGTH_NM
        else:
            frame["wavelength_nm"] = frame["wavelength_nm"].fillna(
                DEFAULT_REFRACTIVE_INDEX_WAVELENGTH_NM
            )
    return frame


def _load_task(
    task_id: str,
    path: Path,
    fingerprint_cache: dict[str, object],
    fingerprint_generator: object,
) -> TaskData:
    frame = _apply_semantic_defaults(task_id, pd.read_csv(path))
    if frame.empty:
        raise TaskRelatednessError(f"Stage-3 task is empty: {task_id}")

    roles = tuple(role for role in ROLE_ORDER if role in frame.columns)
    if roles not in SUPPORTED_TOPOLOGIES:
        raise TaskRelatednessError(
            f"Unsupported chemistry topology in {task_id}: {roles}"
        )
    for role in roles:
        if frame[role].isna().any():
            raise TaskRelatednessError(f"Missing {role} value in {task_id}")
        frame[role] = frame[role].astype(str)

    condition_columns = tuple(
        column for column in CONDITION_ORDER if column in frame.columns
    )
    target_columns = [
        column
        for column in frame.columns
        if column not in roles
        and column not in CONDITION_ORDER
        and column not in METADATA_COLUMNS
    ]
    if len(target_columns) != 1:
        raise TaskRelatednessError(
            f"Expected one property column in {task_id}, found {target_columns}"
        )
    target_column = target_columns[0]
    property_values = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(property_values).all():
        raise TaskRelatednessError(
            f"Non-finite property value in {task_id}/{target_column}"
        )

    condition_values: dict[str, np.ndarray] = {}
    for column in NUMERIC_CONDITIONS:
        if column not in frame:
            condition_values[column] = np.full(len(frame), np.nan)
            continue
        numeric_series = pd.to_numeric(frame[column], errors="coerce")
        malformed = frame[column].notna() & numeric_series.isna()
        if malformed.any() or np.isinf(numeric_series.dropna().to_numpy()).any():
            raise TaskRelatednessError(
                f"Invalid numeric condition in {task_id}/{column}"
            )
        numeric = numeric_series.to_numpy(dtype=float)
        condition_values[column] = _transform_condition(column, numeric)
    if "phase" in frame:
        condition_values["phase"] = np.asarray(
            [_normalise_phase(value) for value in frame["phase"]], dtype=object
        )
    else:
        condition_values["phase"] = np.full(len(frame), None, dtype=object)

    chemistry_keys: list[tuple[str, ...]] = []
    group_rows: list[list[int]] = []
    key_to_group: dict[tuple[str, ...], int] = {}
    for row_index, key in enumerate(
        frame.loc[:, list(roles)].itertuples(index=False, name=None)
    ):
        chemistry_key = tuple(str(value) for value in key)
        group_index = key_to_group.get(chemistry_key)
        if group_index is None:
            group_index = len(chemistry_keys)
            key_to_group[chemistry_key] = group_index
            chemistry_keys.append(chemistry_key)
            group_rows.append([])
        group_rows[group_index].append(row_index)

    chemistry_fingerprints = [
        tuple(
            _fingerprint(
                smiles,
                fingerprint_cache,
                fingerprint_generator,
                task_id=task_id,
                role=role,
            )
            for role, smiles in zip(roles, key)
        )
        for key in chemistry_keys
    ]
    return TaskData(
        task_id=task_id,
        path=path,
        roles=roles,
        topology=SUPPORTED_TOPOLOGIES[roles],
        condition_columns=condition_columns,
        target_column=target_column,
        frame=frame,
        property_values=property_values,
        condition_values=condition_values,
        chemistry_keys=chemistry_keys,
        chemistry_groups=[np.asarray(rows, dtype=np.int64) for rows in group_rows],
        chemistry_fingerprints=chemistry_fingerprints,
    )


def load_stage3_tasks(input_root: Path) -> list[TaskData]:
    input_root = Path(input_root)
    experiment_root = input_root / "experiment"
    paths = sorted(experiment_root.glob("*.csv"))
    discovered = tuple(f"experiment/{path.stem}" for path in paths)
    if set(discovered) != set(EXPECTED_STAGE3_TASK_IDS):
        missing = sorted(set(EXPECTED_STAGE3_TASK_IDS) - set(discovered))
        extra = sorted(set(discovered) - set(EXPECTED_STAGE3_TASK_IDS))
        raise TaskRelatednessError(
            f"Expected exactly the 21 Stage-3 tasks; missing={missing}, extra={extra}"
        )

    fingerprint_cache: dict[str, object] = {}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    paths_by_task = {f"experiment/{path.stem}": path for path in paths}
    return [
        _load_task(task_id, paths_by_task[task_id], fingerprint_cache, generator)
        for task_id in EXPECTED_STAGE3_TASK_IDS
    ]


def compute_condition_scales(tasks: Sequence[TaskData]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for column in NUMERIC_CONDITIONS:
        observed = [
            values[np.isfinite(values)]
            for task in tasks
            if (values := task.condition_values[column]).size
        ]
        unique = np.unique(np.concatenate(observed)) if observed else np.asarray([])
        if unique.size == 0:
            scales[column] = 0.0
            continue
        q25, q75 = np.quantile(unique, (0.25, 0.75))
        scale = float(q75 - q25)
        if scale == 0.0:
            scale = float(unique[-1] - unique[0])
        scales[column] = scale
    return scales


def _numeric_condition_similarity(
    query_value: float,
    target_values: np.ndarray,
    scale: float,
) -> np.ndarray:
    result = np.full(len(target_values), MISSING_CONDITION_SIMILARITY, dtype=float)
    if not np.isfinite(query_value):
        return result
    present = np.isfinite(target_values)
    if scale > 0.0:
        result[present] = np.exp2(
            -np.abs(target_values[present] - query_value) / scale
        )
    else:
        result[present] = np.equal(target_values[present], query_value).astype(float)
    return result


def condition_similarity(
    source: TaskData,
    source_index: int,
    target: TaskData,
    target_indices: np.ndarray,
    scales: dict[str, float],
) -> np.ndarray:
    applicable = tuple(
        column
        for column in CONDITION_ORDER
        if column in source.condition_columns or column in target.condition_columns
    )
    if not applicable:
        return np.full(
            len(target_indices), MISSING_CONDITION_SIMILARITY, dtype=float
        )

    contributions: list[np.ndarray] = []
    for column in applicable:
        if column in NUMERIC_CONDITIONS:
            contributions.append(
                _numeric_condition_similarity(
                    float(source.condition_values[column][source_index]),
                    target.condition_values[column][target_indices].astype(float),
                    scales[column],
                )
            )
            continue
        query_phase = source.condition_values["phase"][source_index]
        target_phases = target.condition_values["phase"][target_indices]
        phase_similarity = np.full(
            len(target_indices), MISSING_CONDITION_SIMILARITY, dtype=float
        )
        present = np.asarray([value is not None for value in target_phases])
        if query_phase is not None:
            phase_similarity[present] = np.equal(
                target_phases[present], query_phase
            ).astype(float)
        contributions.append(phase_similarity)
    return np.mean(np.vstack(contributions), axis=0)


def chemistry_similarities(
    query_fingerprints: tuple[object, ...],
    target_fingerprints: Sequence[tuple[object, ...]],
) -> np.ndarray:
    role_similarities = [
        np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                query_fingerprint,
                [fingerprints[role_index] for fingerprints in target_fingerprints],
            ),
            dtype=float,
        )
        for role_index, query_fingerprint in enumerate(query_fingerprints)
    ]
    product = np.prod(np.vstack(role_similarities), axis=0)
    return np.power(product, 1.0 / len(query_fingerprints))


def _candidate_wins(
    score: float,
    index: int,
    best_score: float,
    best_index: int,
    *,
    source_index: int,
    same_task: bool,
) -> bool:
    if score > best_score:
        return True
    if score < best_score:
        return False
    candidate_is_self = same_task and index == source_index
    best_is_self = same_task and best_index == source_index
    if candidate_is_self != best_is_self:
        return candidate_is_self
    return best_index < 0 or index < best_index


def match_task_pair(
    source: TaskData,
    target: TaskData,
    scales: dict[str, float],
    progress_callback: Callable[[], None] | None = None,
) -> PairMatches:
    if source.roles != target.roles:
        raise TaskRelatednessError(
            f"Cannot match incompatible topologies: {source.task_id}, {target.task_id}"
        )
    row_count = len(source.frame)
    target_indices = np.full(row_count, -1, dtype=np.int64)
    chemistry_scores = np.full(row_count, np.nan)
    condition_scores = np.full(row_count, np.nan)
    combined_scores = np.full(row_count, np.nan)
    same_task = source.task_id == target.task_id

    for query_group_index, query_rows in enumerate(source.chemistry_groups):
        group_chemistry = chemistry_similarities(
            source.chemistry_fingerprints[query_group_index],
            target.chemistry_fingerprints,
        )
        group_order = np.argsort(-group_chemistry, kind="stable")
        for source_index_value in query_rows:
            source_index = int(source_index_value)
            best_index = -1
            best_chemistry = math.nan
            best_condition = math.nan
            best_combined = -math.inf
            for target_group_index_value in group_order:
                target_group_index = int(target_group_index_value)
                chemistry = float(group_chemistry[target_group_index])
                upper_bound = CHEMISTRY_WEIGHT * chemistry + CONDITION_WEIGHT
                if upper_bound < best_combined:
                    break
                rows = target.chemistry_groups[target_group_index]
                conditions = condition_similarity(
                    source, source_index, target, rows, scales
                )
                combined = CHEMISTRY_WEIGHT * chemistry + CONDITION_WEIGHT * conditions
                local_score = float(np.max(combined))
                local_candidates = rows[combined == local_score]
                if same_task and source_index in local_candidates:
                    local_index = source_index
                else:
                    local_index = int(local_candidates[0])
                if _candidate_wins(
                    local_score,
                    local_index,
                    best_combined,
                    best_index,
                    source_index=source_index,
                    same_task=same_task,
                ):
                    local_position = int(np.flatnonzero(rows == local_index)[0])
                    best_index = local_index
                    best_chemistry = chemistry
                    best_condition = float(conditions[local_position])
                    best_combined = local_score
            if best_index < 0:
                raise TaskRelatednessError(
                    f"No target found for {source.task_id} row {source_index + 2}"
                )
            target_indices[source_index] = best_index
            chemistry_scores[source_index] = best_chemistry
            condition_scores[source_index] = best_condition
            combined_scores[source_index] = best_combined
            if progress_callback is not None:
                progress_callback()

    return PairMatches(
        target_indices=target_indices,
        chemistry_similarity=chemistry_scores,
        condition_similarity=condition_scores,
        combined_similarity=combined_scores,
    )


def brute_force_match_task_pair(
    source: TaskData,
    target: TaskData,
    scales: dict[str, float],
) -> PairMatches:
    """Reference matcher used to verify the pruned exact matcher."""
    if source.roles != target.roles:
        raise TaskRelatednessError(
            f"Cannot match incompatible topologies: {source.task_id}, {target.task_id}"
        )
    row_to_group = np.empty(len(target.frame), dtype=np.int64)
    for group_index, rows in enumerate(target.chemistry_groups):
        row_to_group[rows] = group_index
    target_fingerprint_by_row = [
        target.chemistry_fingerprints[int(row_to_group[index])]
        for index in range(len(target.frame))
    ]

    selected: list[int] = []
    chemistry_selected: list[float] = []
    condition_selected: list[float] = []
    combined_selected: list[float] = []
    same_task = source.task_id == target.task_id
    source_row_to_group = np.empty(len(source.frame), dtype=np.int64)
    for group_index, rows in enumerate(source.chemistry_groups):
        source_row_to_group[rows] = group_index
    all_target_indices = np.arange(len(target.frame), dtype=np.int64)
    for source_index in range(len(source.frame)):
        query_fingerprints = source.chemistry_fingerprints[
            int(source_row_to_group[source_index])
        ]
        chemistry = chemistry_similarities(
            query_fingerprints, target_fingerprint_by_row
        )
        conditions = condition_similarity(
            source, source_index, target, all_target_indices, scales
        )
        combined = CHEMISTRY_WEIGHT * chemistry + CONDITION_WEIGHT * conditions
        best_score = float(np.max(combined))
        candidates = all_target_indices[combined == best_score]
        if same_task and source_index in candidates:
            best_index = source_index
        else:
            best_index = int(candidates[0])
        selected.append(best_index)
        chemistry_selected.append(float(chemistry[best_index]))
        condition_selected.append(float(conditions[best_index]))
        combined_selected.append(best_score)
    return PairMatches(
        target_indices=np.asarray(selected, dtype=np.int64),
        chemistry_similarity=np.asarray(chemistry_selected),
        condition_similarity=np.asarray(condition_selected),
        combined_similarity=np.asarray(combined_selected),
    )


def _spearman(
    source_values: np.ndarray, target_values: np.ndarray
) -> tuple[float, str]:
    if len(source_values) < 2:
        return math.nan, "fewer_than_two_pairs"
    if np.unique(source_values).size < 2:
        return math.nan, "constant_source_property"
    if np.unique(target_values).size < 2:
        return math.nan, "constant_target_property"
    return float(spearmanr(source_values, target_values).statistic), ""


def _similarity_summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_q05": float(quantiles[0]),
        f"{prefix}_q25": float(quantiles[1]),
        f"{prefix}_median": float(quantiles[2]),
        f"{prefix}_q75": float(quantiles[3]),
        f"{prefix}_q95": float(quantiles[4]),
        f"{prefix}_max": float(np.max(values)),
    }


def _audit_value(task: TaskData, row_index: int, column: str) -> object:
    if column not in task.frame:
        return ""
    value = task.frame.iloc[row_index][column]
    return "" if pd.isna(value) else value


def _write_pair_matches(
    writer: csv.DictWriter,
    source: TaskData,
    target: TaskData,
    matches: PairMatches,
) -> None:
    for source_index, target_index_value in enumerate(matches.target_indices):
        target_index = int(target_index_value)
        row: dict[str, object] = {
            "source_task": source.task_id,
            "target_task": target.task_id,
            "source_row": source_index + 2,
            "target_row": target_index + 2,
            "source_property": source.target_column,
            "target_property": target.target_column,
            "source_value": source.property_values[source_index],
            "target_value": target.property_values[target_index],
            "chemistry_similarity": matches.chemistry_similarity[source_index],
            "condition_similarity": matches.condition_similarity[source_index],
            "combined_similarity": matches.combined_similarity[source_index],
        }
        for role in ROLE_ORDER:
            row[f"source_{role}"] = _audit_value(source, source_index, role)
            row[f"target_{role}"] = _audit_value(target, target_index, role)
        for condition in CONDITION_ORDER:
            row[f"source_{condition}"] = _audit_value(
                source, source_index, condition
            )
            row[f"target_{condition}"] = _audit_value(
                target, target_index, condition
            )
        writer.writerow(row)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_metadata(
    output_path: Path,
    tasks: Sequence[TaskData],
    scales: dict[str, float],
) -> None:
    metadata = {
        "task_count": len(tasks),
        "task_order": [task.task_id for task in tasks],
        "input_files": {
            task.task_id: {
                "path": str(task.path.resolve()),
                "sha256": _file_sha256(task.path),
                "rows": len(task.frame),
                "topology": task.topology,
                "roles": list(task.roles),
                "property": task.target_column,
                "conditions": list(task.condition_columns),
            }
            for task in tasks
        },
        "similarity": {
            "chemistry_weight": CHEMISTRY_WEIGHT,
            "condition_weight": CONDITION_WEIGHT,
            "fingerprint": {
                "type": "Morgan/ECFP4",
                "radius": 2,
                "bits": 2048,
                "metric": "Tanimoto",
                "system_aggregation": "geometric_mean_by_role",
            },
            "numeric_condition_similarity": "2 ** (-absolute_distance / scale)",
            "numeric_condition_scales": scales,
            "scale_estimator": "IQR of unique transformed values; full-range fallback",
            "log1p_conditions": sorted(LOG_CONDITIONS),
            "phase_normalization": "strip and casefold",
            "missing_condition_similarity": MISSING_CONDITION_SIMILARITY,
            "condition_aggregation": "arithmetic_mean_of_applicable_dimensions",
            "threshold": None,
        },
        "defaults": {
            "pressure_kPa": {
                "value": DEFAULT_EXPERIMENT_PRESSURE_KPA,
                "tasks": sorted(DEFAULT_PRESSURE_TASKS),
            },
            "wavelength_nm": {
                "value": DEFAULT_REFRACTIVE_INDEX_WAVELENGTH_NM,
                "tasks": ["experiment/refractive_index"],
            },
        },
        "matching": {
            "topology_policy": "exact_role_tuple_only",
            "neighbors_per_query": 1,
            "many_to_one": True,
            "threshold_enabled": False,
            "cross_task_tie_break": "lowest_original_target_row",
            "same_task_tie_break": "query_row_then_lowest_original_target_row",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }
    output_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def analyze_stage3_task_relatedness(
    input_root: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = load_stage3_tasks(input_root)
    scales = compute_condition_scales(tasks)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_ids = [task.task_id for task in tasks]
    matrix = pd.DataFrame(np.nan, index=task_ids, columns=task_ids, dtype=float)
    matrix.index.name = "source_task"
    summaries: list[dict[str, object]] = []
    matches_path = output_dir / "matches.csv.gz"
    total_pairs = len(tasks) * len(tasks)
    total_work = sum(
        len(source.frame) if source.roles == target.roles else 1
        for source in tasks
        for target in tasks
    )
    progress = _PairProgress(total_pairs=total_pairs, total_work=total_work)
    try:
        with gzip.open(matches_path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MATCH_COLUMNS)
            writer.writeheader()
            for source in tasks:
                for target in tasks:
                    pair_work = (
                        len(source.frame) if source.roles == target.roles else 1
                    )
                    progress.start_pair(
                        source.task_id, target.task_id, pair_work=pair_work
                    )
                    base_summary: dict[str, object] = {
                        "source_task": source.task_id,
                        "target_task": target.task_id,
                        "source_topology": source.topology,
                        "target_topology": target.topology,
                        "source_rows": len(source.frame),
                        "target_rows": len(target.frame),
                        "excluded_query_count": 0,
                    }
                    if source.roles != target.roles:
                        summaries.append(
                            {
                                **base_summary,
                                "compatible": False,
                                "matched_pairs": 0,
                                "spearman_rho": math.nan,
                                "undefined_reason": "incompatible_topology",
                            }
                        )
                        progress.advance_work()
                        progress.finish_pair()
                        continue
                    matches = match_task_pair(
                        source,
                        target,
                        scales,
                        progress_callback=(
                            progress.advance_work if progress.enabled else None
                        ),
                    )
                    target_values = target.property_values[matches.target_indices]
                    rho, undefined_reason = _spearman(
                        source.property_values, target_values
                    )
                    matrix.loc[source.task_id, target.task_id] = rho
                    summary = {
                        **base_summary,
                        "compatible": True,
                        "matched_pairs": len(matches.target_indices),
                        "spearman_rho": rho,
                        "undefined_reason": undefined_reason,
                    }
                    summary.update(
                        _similarity_summary(
                            "chemistry", matches.chemistry_similarity
                        )
                    )
                    summary.update(
                        _similarity_summary(
                            "condition", matches.condition_similarity
                        )
                    )
                    summary.update(
                        _similarity_summary(
                            "combined", matches.combined_similarity
                        )
                    )
                    summaries.append(summary)
                    _write_pair_matches(writer, source, target, matches)
                    progress.finish_pair()
    finally:
        progress.close()

    summary_frame = pd.DataFrame(summaries)
    matrix.to_csv(output_dir / "spearman_matrix.csv")
    summary_frame.to_csv(output_dir / "task_pair_summary.csv", index=False)
    _write_metadata(output_dir / "run_metadata.json", tasks, scales)
    return matrix, summary_frame


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=PROJECT_ROOT / "data" / "final"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "stage3_task_relatedness",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    matrix, summary = analyze_stage3_task_relatedness(
        args.input_root, args.output_dir
    )
    compatible = int(summary["compatible"].sum())
    print(
        f"Wrote {matrix.shape[0]}x{matrix.shape[1]} directed Spearman matrix "
        f"with {compatible} compatible task pairs to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
