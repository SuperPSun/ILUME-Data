"""Build readable three-stage training corpora and deterministic splits.

The script reads only top-level CSV files under ``data/final/experiment`` and
``data/final/simulation``. It never traverses or parses molecular structure
files such as ``.mol`` or ``.mol2``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from http import client as http_client
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
from typing import Callable, Iterable, Mapping, Sequence
from urllib import error, parse, request

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINAL_ROOT = PROJECT_ROOT / "data" / "final"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "training_splits"
SCRIPT_VERSION = 1

BUCKETS = ("experiment", "simulation")
ROLE_BY_COLUMN = {
    "cation": "cation",
    "anion": "anion",
    "solute": "solute",
    "solvent": "solvent",
    "SMILES": "molecule",
}
IDENTITY_COLUMNS = tuple(ROLE_BY_COLUMN)
CONDITION_COLUMNS = {
    "temperature_K",
    "pressure_kPa",
    "frequency_MHz",
    "wavelength_nm",
    "phase",
}
METADATA_COLUMNS = {"source_list", "mol_id"}
PRETRAIN_ENTITY_COLUMNS = [
    "SMILES",
    "formal_charge",
    "origin_list",
    "seed_smiles_list",
    "rule_list",
    "pubchem_cid_list",
    "mol_id_list",
]
ORIGIN_ORDER = {"dataset": 0, "pubchem": 1, "rule": 2}

STAGE2_FILES = {
    "simulation/density.csv",
    "simulation/heat_capacity.csv",
    "simulation/simulated_qm_elec_hf.csv",
    "simulation/thermal_expansion.csv",
    "simulation/transfer_organic.csv",
}
QM_TARGET_COLUMNS = (
    "ESP_max",
    "ESP_min",
    "ESP_std",
    "ESP_pos_frac",
    "Dipole",
    "Quadrupole",
    "q_max",
    "q_min",
    "q_std",
    "q_pos_frac",
    "gap_eV",
)

SYSTEM_COLUMNS = {
    "il": ("cation", "anion"),
    "il_solute": ("cation", "anion", "solute"),
    "solute_solvent": ("solute", "solvent"),
    "cation": ("cation",),
    "anion": ("anion",),
    "solute": ("solute",),
    "solvent": ("solvent",),
    "molecule": ("SMILES",),
}
GROUP_STRATEGY_COLUMNS = {
    "il": {
        "il": ("cation", "anion"),
        "cation": ("cation",),
        "anion": ("anion",),
    },
    "il_solute": {
        "il_solute": ("cation", "anion", "solute"),
        "il": ("cation", "anion"),
        "cation": ("cation",),
        "anion": ("anion",),
        "solute": ("solute",),
    },
    "solute_solvent": {
        "solute_solvent": ("solute", "solvent"),
        "solute": ("solute",),
        "solvent": ("solvent",),
    },
    "cation": {},
    "anion": {},
    "solute": {},
    "solvent": {},
    "molecule": {},
}
SINGLE_SYSTEM_TYPES = {"cation", "anion", "solute", "solvent", "molecule"}
STRATEGY_DIRECTORY_NAMES = {
    "random": "random",
    "il": "IL",
    "il_solute": "IL-solute",
    "solute_solvent": "solute-solvent",
    "cation": "cation",
    "anion": "anion",
    "solute": "solute",
    "solvent": "solvent",
}

ROW_CATALOG_COLUMNS = [
    "row_id",
    "task_id",
    "source_file",
    "source_row",
    "system_type",
    "system_id",
    "system_key",
    "partition",
    "cation_id",
    "anion_id",
    "il_id",
    "solute_id",
    "solvent_id",
    "molecule_id",
    "mol_ids",
]
LABEL_SUMMARY_COLUMNS = [
    "stage",
    "task_id",
    "strategy",
    "repeat",
    "fold",
    "partition",
    "target_column",
    "row_count",
    "system_count",
    "label_count",
    "missing_rate",
    "value_min",
    "value_p05",
    "value_median",
    "value_mean",
    "value_p95",
    "value_max",
]


class TrainingSplitError(RuntimeError):
    """Raised when inputs cannot produce a trustworthy split."""


class IncompletePubChemQuery(TrainingSplitError):
    """Raised when a PubChem request is unavailable or unfinished."""


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    stage: int
    source_file: str
    target_columns: tuple[str, ...]
    identity_columns: tuple[str, ...]
    system_type: str


@dataclass(frozen=True)
class TaskProfile:
    raw_rows: int
    rows: int
    system_count: int
    tier: str


def stable_id(namespace: str, *parts: object) -> str:
    payload = json.dumps(
        [namespace, *parts],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_fraction(seed: int, namespace: str, *parts: object) -> float:
    digest = stable_id(f"seed:{seed}:{namespace}", *parts)
    return int(digest[:16], 16) / float(2**64)


def balanced_unit_folds(
    unit_ids: Iterable[str],
    *,
    seed: int,
    namespace: str,
    repeat: int,
) -> dict[str, int]:
    unique_units = sorted(
        {str(unit_id) for unit_id in unit_ids},
        key=lambda unit_id: stable_id(
            f"seed:{seed}:{namespace}:unit_order",
            repeat,
            unit_id,
        ),
    )
    if len(unique_units) < 5:
        raise TrainingSplitError(
            f"Fewer than five units for {namespace}, repeat {repeat}"
        )
    fold_order = sorted(
        range(5),
        key=lambda fold: stable_id(
            f"seed:{seed}:{namespace}:fold_order",
            repeat,
            fold,
        ),
    )
    return {
        unit_id: fold_order[index % 5]
        for index, unit_id in enumerate(unique_units)
    }


def _shared_folds_once(
    groups_by_task: Mapping[str, set[str]],
    *,
    seed: int,
    namespace: str,
    repeat: int,
) -> dict[str, int]:
    group_tasks: dict[str, set[str]] = {}
    for task_id, groups in groups_by_task.items():
        if len(groups) < 5:
            raise TrainingSplitError(
                f"Fewer than five {namespace} groups for {task_id}"
            )
        for group_identifier in groups:
            group_tasks.setdefault(str(group_identifier), set()).add(task_id)

    assignments: dict[str, int] = {}
    counts = {
        task_id: [0, 0, 0, 0, 0]
        for task_id in groups_by_task
    }

    def assign(group_identifier: str, fold: int) -> None:
        assignments[group_identifier] = fold
        for incident_task in group_tasks[group_identifier]:
            counts[incident_task][fold] += 1

    task_order = sorted(
        groups_by_task,
        key=lambda task_id: (
            len(groups_by_task[task_id]),
            stable_id(
                f"seed:{seed}:{namespace}:task_order",
                repeat,
                task_id,
            ),
        ),
    )
    for task_id in task_order:
        task_groups = {str(value) for value in groups_by_task[task_id]}
        unassigned = [group for group in task_groups if group not in assignments]
        unassigned.sort(
            key=lambda group: (
                -len(group_tasks[group]),
                stable_id(
                    f"seed:{seed}:{namespace}:group_order",
                    repeat,
                    group,
                ),
            )
        )
        missing_folds = [
            fold for fold, count in enumerate(counts[task_id]) if count == 0
        ]
        missing_folds.sort(
            key=lambda fold: stable_id(
                f"seed:{seed}:{namespace}:missing_fold_order",
                repeat,
                task_id,
                fold,
            )
        )
        while missing_folds and unassigned:
            assign(unassigned.pop(0), missing_folds.pop(0))

        for group_identifier in unassigned:
            incident_tasks = group_tasks[group_identifier]
            fold = min(
                range(5),
                key=lambda candidate_fold: (
                    max(
                        counts[incident_task][candidate_fold]
                        / max(1.0, len(groups_by_task[incident_task]) / 5.0)
                        for incident_task in incident_tasks
                    ),
                    sum(
                        counts[incident_task][candidate_fold]
                        for incident_task in incident_tasks
                    ),
                    stable_id(
                        f"seed:{seed}:{namespace}:greedy_fold",
                        repeat,
                        group_identifier,
                        candidate_fold,
                    ),
                ),
            )
            assign(group_identifier, fold)

    # Repair an empty fold only when moving a shared group cannot empty the
    # source fold in any other incident task.
    for _ in range(max(1, len(groups_by_task) * 5)):
        missing = [
            (task_id, fold)
            for task_id in task_order
            for fold, count in enumerate(counts[task_id])
            if count == 0
        ]
        if not missing:
            break
        changed = False
        for task_id, empty_fold in missing:
            candidates = sorted(
                (
                    group
                    for group in groups_by_task[task_id]
                    if counts[task_id][assignments[str(group)]] > 1
                    and all(
                        counts[incident_task][assignments[str(group)]] > 1
                        for incident_task in group_tasks[str(group)]
                    )
                ),
                key=lambda group: stable_id(
                    f"seed:{seed}:{namespace}:repair",
                    repeat,
                    task_id,
                    empty_fold,
                    str(group),
                ),
            )
            if not candidates:
                continue
            group_identifier = str(candidates[0])
            source_fold = assignments[group_identifier]
            for incident_task in group_tasks[group_identifier]:
                counts[incident_task][source_fold] -= 1
                counts[incident_task][empty_fold] += 1
            assignments[group_identifier] = empty_fold
            changed = True
        if not changed:
            break

    remaining_missing = [
        (task_id, fold)
        for task_id in task_order
        for fold, count in enumerate(counts[task_id])
        if count == 0
    ]
    if remaining_missing:
        task_id, fold = remaining_missing[0]
        raise TrainingSplitError(
            f"Unable to build shared five-fold assignment for {namespace}; "
            f"{task_id} has empty fold {fold}"
        )
    return assignments


def plan_shared_group_folds(
    groups_by_task: Mapping[str, set[str]],
    repeats_by_task: Mapping[str, int],
    *,
    seed: int,
    namespace: str,
) -> dict[tuple[str, int], int]:
    planned: dict[tuple[str, int], int] = {}
    maximum_repeats = max(repeats_by_task.values(), default=0)
    for repeat in range(maximum_repeats):
        active = {
            task_id: groups
            for task_id, groups in groups_by_task.items()
            if repeats_by_task[task_id] > repeat
        }
        if not active:
            continue
        assignments = _shared_folds_once(
            active,
            seed=seed,
            namespace=namespace,
            repeat=repeat,
        )
        planned.update(
            {
                (group_identifier, repeat): fold
                for group_identifier, fold in assignments.items()
            }
        )
    return planned


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_csv_paths(final_root: Path) -> list[Path]:
    """Return only top-level dataset CSVs, deliberately ignoring subdirectories."""
    paths: list[Path] = []
    for bucket in BUCKETS:
        bucket_root = final_root / bucket
        if not bucket_root.is_dir():
            raise TrainingSplitError(f"Missing final-data bucket: {bucket_root}")
        paths.extend(sorted(bucket_root.glob("*.csv")))
    return sorted(paths, key=lambda path: path.relative_to(final_root).as_posix())


def replace_directory(staged: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    staged.rename(destination)


def write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_manifest(output_root: Path) -> dict[str, object]:
    path = output_root / "manifest.json"
    if not path.exists():
        return {"script_version": SCRIPT_VERSION}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TrainingSplitError(f"Invalid manifest: {path}")
    return payload


def update_manifest(output_root: Path, section: str, payload: Mapping[str, object]) -> None:
    manifest = load_manifest(output_root)
    manifest["script_version"] = SCRIPT_VERSION
    manifest[section] = dict(payload)
    write_json(output_root / "manifest.json", manifest)


@lru_cache(maxsize=None)
def canonicalize_smiles(smiles: str) -> str:
    text = str(smiles).strip()
    if not text:
        raise TrainingSplitError("Empty SMILES")
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise TrainingSplitError(f"Invalid SMILES: {text}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=None)
def formal_charge(smiles: str) -> int:
    molecule = Chem.MolFromSmiles(canonicalize_smiles(smiles))
    if molecule is None:  # pragma: no cover - protected by canonicalize_smiles
        raise TrainingSplitError(f"Invalid SMILES: {smiles}")
    return int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()))


@lru_cache(maxsize=None)
def fragment_count(smiles: str) -> int:
    molecule = Chem.MolFromSmiles(canonicalize_smiles(smiles))
    if molecule is None:  # pragma: no cover - protected by canonicalize_smiles
        raise TrainingSplitError(f"Invalid SMILES: {smiles}")
    return len(Chem.GetMolFrags(molecule))


@lru_cache(maxsize=None)
def role_fragments(smiles: str, role: str) -> tuple[str, ...]:
    canonical = canonicalize_smiles(smiles)
    if role not in {"cation", "anion"}:
        return (canonical,)

    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:  # pragma: no cover - protected by canonicalize_smiles
        raise TrainingSplitError(f"Invalid SMILES: {smiles}")
    total_charge = int(
        sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    )
    expected_positive = role == "cation"
    if expected_positive and total_charge <= 0:
        raise TrainingSplitError(
            f"Non-positive cation field: {canonical} ({total_charge})"
        )
    if not expected_positive and total_charge >= 0:
        raise TrainingSplitError(
            f"Non-negative anion field: {canonical} ({total_charge})"
        )
    fragments = [
        Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
        for fragment in Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    ]
    fragments.sort()
    fragment_charges = [formal_charge(fragment) for fragment in fragments]
    all_same_sign = (
        all(charge > 0 for charge in fragment_charges)
        if expected_positive
        else all(charge < 0 for charge in fragment_charges)
    )
    if not all_same_sign:
        # Disconnected SMILES can encode one coordination ion with internal
        # countercharges (for example, a ferrocenyl anion containing [Fe+2]).
        # Splitting those components would destroy the charged entity.
        return (canonical,)
    return tuple(fragments)


def entity_id(role: str, smiles: str) -> str:
    return stable_id("entity", role, canonicalize_smiles(smiles))


def _joined(values: Iterable[object], *, origin: bool = False) -> str:
    normalized = {
        str(value).strip()
        for value in values
        if not pd.isna(value) and str(value).strip()
    }
    if origin:
        ordered = sorted(
            normalized,
            key=lambda value: (ORIGIN_ORDER.get(value, len(ORIGIN_ORDER)), value),
        )
    else:
        ordered = sorted(normalized)
    return ";".join(ordered)


def _split_joined(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    }


def _entity_output_frame(
    records: Iterable[Mapping[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "SMILES": str(record["SMILES"]),
                "formal_charge": int(record["formal_charge"]),
                "origin_list": _joined(
                    record.get("origin_list", set()),
                    origin=True,
                ),
                "seed_smiles_list": _joined(
                    record.get("seed_smiles_list", set())
                ),
                "rule_list": _joined(record.get("rule_list", set())),
                "pubchem_cid_list": _joined(
                    record.get("pubchem_cid_list", set())
                ),
                "mol_id_list": _joined(record.get("mol_id_list", set())),
            }
        )
    return pd.DataFrame(rows, columns=PRETRAIN_ENTITY_COLUMNS).sort_values(
        "SMILES",
        kind="stable",
    ).reset_index(drop=True)


def _combined_entity_frame(stage1_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for role in ROLE_BY_COLUMN.values():
        path = stage1_root / f"{role}.csv"
        if not path.exists():
            raise TrainingSplitError(
                f"Run extract-pretrain before augment-pretrain; missing {path}"
            )
        frame = pd.read_csv(path, keep_default_na=False)
        missing = set(PRETRAIN_ENTITY_COLUMNS) - set(frame.columns)
        if missing:
            raise TrainingSplitError(
                f"Missing Stage-1 columns in {path}: {sorted(missing)}"
            )
        frame = frame.loc[:, PRETRAIN_ENTITY_COLUMNS].copy()
        frame.insert(0, "role", role)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def extract_pretraining_entities(
    final_root: Path,
    output_root: Path,
    *,
    pretrain_cap: int = 500_000,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Extract readable role-specific entities and observed ionic-liquid pairs."""
    final_root = Path(final_root)
    output_root = Path(output_root)
    paths = final_csv_paths(final_root)
    output_root.mkdir(parents=True, exist_ok=True)

    entities: dict[tuple[str, str], dict[str, object]] = {}
    ionic_liquids: set[tuple[str, str]] = set()
    with tempfile.TemporaryDirectory(dir=output_root.parent) as temporary_dir:
        staged_stage1 = Path(temporary_dir) / "stage1"
        staged_stage1.mkdir()

        for path in paths:
            relative = path.relative_to(final_root).as_posix()
            header = list(pd.read_csv(path, nrows=0).columns)
            columns = [
                column for column in IDENTITY_COLUMNS if column in header
            ]
            if not columns:
                continue
            usecols = [*columns]
            if "mol_id" in header:
                usecols.append("mol_id")
            row_offset = 0
            for chunk in pd.read_csv(
                path,
                usecols=usecols,
                chunksize=chunk_size,
            ):
                if {"cation", "anion"}.issubset(chunk.columns):
                    for local_row, values in enumerate(
                        chunk[["cation", "anion"]].itertuples(
                            index=False,
                            name=None,
                        )
                    ):
                        try:
                            ionic_liquids.add(
                                (
                                    canonicalize_smiles(str(values[0])),
                                    canonicalize_smiles(str(values[1])),
                                )
                            )
                        except TrainingSplitError as exc:
                            raise TrainingSplitError(
                                f"{relative} row {row_offset + local_row}: {exc}"
                            ) from exc

                for column in columns:
                    role = ROLE_BY_COLUMN[column]
                    for local_row, value in enumerate(chunk[column].tolist()):
                        if pd.isna(value) or not str(value).strip():
                            raise TrainingSplitError(
                                f"Missing {column} in {relative} row "
                                f"{row_offset + local_row}"
                            )
                        try:
                            fragments = role_fragments(str(value), role)
                        except TrainingSplitError as exc:
                            raise TrainingSplitError(
                                f"{relative} row {row_offset + local_row}, "
                                f"column {column}: {exc}"
                            ) from exc
                        mol_id = (
                            chunk.iloc[local_row]["mol_id"]
                            if "mol_id" in chunk.columns
                            else ""
                        )
                        for smiles in fragments:
                            key = (role, smiles)
                            record = entities.setdefault(
                                key,
                                {
                                    "role": role,
                                    "SMILES": smiles,
                                    "formal_charge": formal_charge(smiles),
                                    "origin_list": {"dataset"},
                                    "seed_smiles_list": set(),
                                    "rule_list": set(),
                                    "pubchem_cid_list": set(),
                                    "mol_id_list": set(),
                                },
                            )
                            if not pd.isna(mol_id) and str(mol_id).strip():
                                mol_ids = record["mol_id_list"]
                                if isinstance(mol_ids, set):
                                    mol_ids.add(str(mol_id).strip())
                row_offset += len(chunk)

        if len(entities) > pretrain_cap:
            raise TrainingSplitError(
                f"Base pretraining corpus has {len(entities)} entities, "
                f"exceeding cap {pretrain_cap}"
            )
        combined_rows: list[pd.DataFrame] = []
        for role in ROLE_BY_COLUMN.values():
            role_frame = _entity_output_frame(
                record
                for (record_role, _), record in entities.items()
                if record_role == role
            )
            role_frame.to_csv(
                staged_stage1 / f"{role}.csv",
                index=False,
                lineterminator="\n",
            )
            with_role = role_frame.copy()
            with_role.insert(0, "role", role)
            combined_rows.append(with_role)
        pd.DataFrame(
            sorted(ionic_liquids),
            columns=["cation", "anion"],
        ).to_csv(
            staged_stage1 / "IL.csv",
            index=False,
            lineterminator="\n",
        )
        entity_frame = pd.concat(combined_rows, ignore_index=True)
        replace_directory(staged_stage1, output_root / "stage1")

    (output_root / "manifest.json").unlink(missing_ok=True)
    return entity_frame


Transport = Callable[
    [request.Request],
    tuple[int, bytes, Mapping[str, str]],
]


class PubChemClient:
    """Small cached PUG REST client with deterministic retry behavior."""

    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    TRANSIENT_STATUS = {429, 503, 504}

    def __init__(
        self,
        cache_path: Path,
        *,
        offline: bool = False,
        rate_limit: float = 4.0,
        max_retries: int = 5,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.cache_path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                request_key TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                payload TEXT,
                error TEXT
            )
            """
        )
        self.connection.commit()
        self.offline = offline
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.transport = transport or self._default_transport
        self.sleep = sleep
        self.clock = clock
        self._last_request_at: float | None = None

    def __enter__(self) -> PubChemClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def _default_transport(
        http_request: request.Request,
    ) -> tuple[int, bytes, Mapping[str, str]]:
        try:
            with request.urlopen(http_request, timeout=60) as response:
                return response.status, response.read(), dict(response.headers.items())
        except error.HTTPError as exc:
            return exc.code, exc.read(), dict(exc.headers.items())

    def _cached(self, request_key: str) -> dict[str, object] | None:
        row = self.connection.execute(
            "SELECT status, payload FROM requests WHERE request_key = ?",
            (request_key,),
        ).fetchone()
        if row is None or row[0] != "ok":
            return None
        return json.loads(row[1])

    def _store(
        self,
        request_key: str,
        *,
        status: str,
        payload: Mapping[str, object] | None = None,
        error_text: str | None = None,
    ) -> None:
        serialized = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if payload is not None
            else None
        )
        self.connection.execute(
            """
            INSERT INTO requests(request_key, status, payload, error)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(request_key) DO UPDATE SET
                status = excluded.status,
                payload = excluded.payload,
                error = excluded.error
            """,
            (request_key, status, serialized, error_text),
        )
        self.connection.commit()

    def _throttle(self) -> None:
        if self.rate_limit <= 0:
            return
        minimum_interval = 1.0 / self.rate_limit
        now = self.clock()
        if self._last_request_at is not None:
            remaining = minimum_interval - (now - self._last_request_at)
            if remaining > 0:
                self.sleep(remaining)
                now = self.clock()
        self._last_request_at = now

    def _query(
        self,
        request_key: str,
        url: str,
        smiles: str,
    ) -> dict[str, object]:
        cached = self._cached(request_key)
        if cached is not None:
            return cached
        if self.offline:
            raise IncompletePubChemQuery(f"Missing cached PubChem request: {request_key}")

        body = parse.urlencode({"smiles": smiles}).encode("utf-8")
        http_request = request.Request(
            url,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "ILUME-Data-training-splits/1.0",
            },
            method="POST",
        )
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                self._throttle()
                status, response_body, headers = self.transport(http_request)
                if status == 200:
                    payload = json.loads(response_body.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("PubChem returned a non-object JSON payload")
                    self._store(request_key, status="ok", payload=payload)
                    return payload
                if status in {400, 404}:
                    payload = {"PropertyTable": {"Properties": []}}
                    self._store(request_key, status="ok", payload=payload)
                    return payload
                last_error = f"HTTP {status}"
                if status not in self.TRANSIENT_STATUS:
                    break
                retry_after = headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(30.0, 2.0**attempt)
            except (
                OSError,
                TimeoutError,
                ValueError,
                json.JSONDecodeError,
                http_client.IncompleteRead,
            ) as exc:
                last_error = str(exc)
                delay = min(30.0, 2.0**attempt)
            if attempt < self.max_retries:
                self.sleep(delay)

        self._store(request_key, status="error", error_text=last_error)
        raise IncompletePubChemQuery(
            f"PubChem request failed after retries: {request_key}: {last_error}"
        )

    @staticmethod
    def _properties(payload: Mapping[str, object]) -> list[dict[str, object]]:
        property_table = payload.get("PropertyTable", {})
        if not isinstance(property_table, dict):
            return []
        properties = property_table.get("Properties", [])
        if not isinstance(properties, list):
            return []
        return [record for record in properties if isinstance(record, dict)]

    def similarity(
        self,
        smiles: str,
        *,
        threshold: int = 90,
        max_records: int = 100,
    ) -> list[dict[str, object]]:
        query = parse.urlencode({"Threshold": threshold, "MaxRecords": max_records})
        url = (
            f"{self.BASE_URL}/compound/fastsimilarity_2d/smiles/"
            f"property/SMILES,Charge/JSON?{query}"
        )
        request_key = stable_id(
            "pubchem_similarity",
            canonicalize_smiles(smiles),
            threshold,
            max_records,
        )
        return self._properties(self._query(request_key, url, smiles))

def smiles_from_pubchem_record(record: Mapping[str, object]) -> str | None:
    for field in ("SMILES", "IsomericSMILES", "CanonicalSMILES", "ConnectivitySMILES"):
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _sanitized_smiles(molecule: Chem.Mol) -> str | None:
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(molecule)
            if len(Chem.GetMolFrags(molecule)) != 1:
                return None
            return Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=True,
            )
    except (ValueError, RuntimeError):
        return None


def _is_alkyl_carbon(atom: Chem.Atom) -> bool:
    return (
        atom.GetAtomicNum() == 6
        and atom.GetFormalCharge() == 0
        and not atom.GetIsAromatic()
        and not atom.IsInRing()
        and all(bond.GetBondType() == Chem.BondType.SINGLE for bond in atom.GetBonds())
    )


def _terminal_alkyl_chains(molecule: Chem.Mol) -> list[tuple[int, ...]]:
    chains: list[tuple[int, ...]] = []
    for atom in molecule.GetAtoms():
        if not _is_alkyl_carbon(atom):
            continue
        heavy_neighbors = [
            neighbor for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() > 1
        ]
        if len(heavy_neighbors) != 1:
            continue
        chain = [atom.GetIdx()]
        previous = atom
        current = heavy_neighbors[0]
        while _is_alkyl_carbon(current):
            current_heavy = [
                neighbor
                for neighbor in current.GetNeighbors()
                if neighbor.GetAtomicNum() > 1
            ]
            if len(current_heavy) > 2:
                break
            chain.append(current.GetIdx())
            next_atoms = [
                neighbor
                for neighbor in current_heavy
                if neighbor.GetIdx() != previous.GetIdx()
            ]
            if len(next_atoms) != 1:
                break
            previous, current = current, next_atoms[0]
        chains.append(tuple(chain))
    return chains


def _perfluoroalkyl_chains(molecule: Chem.Mol) -> list[tuple[int, ...]]:
    chains: list[tuple[int, ...]] = []
    for atom in molecule.GetAtoms():
        if not _is_alkyl_carbon(atom):
            continue
        carbon_neighbors = [
            neighbor
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() == 6
        ]
        anchor_neighbors = [
            neighbor
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() not in {6, 9}
        ]
        if len(anchor_neighbors) != 1 or len(carbon_neighbors) > 1:
            continue

        chain = [atom.GetIdx()]
        previous_index: int | None = None
        current = atom
        valid = True
        while True:
            neighbors = list(current.GetNeighbors())
            if (
                not _is_alkyl_carbon(current)
                or any(
                    bond.GetBondType() != Chem.BondType.SINGLE
                    for bond in current.GetBonds()
                )
            ):
                valid = False
                break
            fluorines = [
                neighbor for neighbor in neighbors if neighbor.GetAtomicNum() == 9
            ]
            external = [
                neighbor
                for neighbor in neighbors
                if neighbor.GetAtomicNum() not in {6, 9}
            ]
            if current.GetIdx() == atom.GetIdx():
                if len(external) != 1:
                    valid = False
                    break
            elif external:
                valid = False
                break

            next_carbons = [
                neighbor
                for neighbor in neighbors
                if neighbor.GetAtomicNum() == 6
                and neighbor.GetIdx() != previous_index
            ]
            if not next_carbons:
                if len(fluorines) != 3:
                    valid = False
                break
            if len(next_carbons) != 1 or len(fluorines) != 2:
                valid = False
                break
            previous_index = current.GetIdx()
            current = next_carbons[0]
            if current.GetIdx() in chain:
                valid = False
                break
            chain.append(current.GetIdx())
        if valid:
            chains.append(tuple(chain))
    return chains


def _record_rule_candidate(
    candidates: dict[tuple[str, str], dict[str, str]],
    molecule: Chem.Mol,
    *,
    canonical: str,
    target_charge: int,
    rule: str,
    family: str,
) -> None:
    candidate = _sanitized_smiles(molecule)
    if (
        candidate
        and candidate != canonical
        and formal_charge(candidate) == target_charge
    ):
        candidates[(rule, candidate)] = {
            "SMILES": candidate,
            "rule": rule,
            "family": family,
        }


def _replace_atom(
    molecule: Chem.Mol,
    atom_index: int,
    atomic_number: int,
    *,
    no_implicit: bool | None = None,
) -> Chem.Mol:
    editable = Chem.RWMol(molecule)
    atom = editable.GetAtomWithIdx(atom_index)
    atom.SetAtomicNum(atomic_number)
    atom.SetNumExplicitHs(0)
    if no_implicit is not None:
        atom.SetNoImplicit(no_implicit)
    return editable.GetMol()


def generate_rule_candidates(
    smiles: str,
    role: str | None = None,
) -> list[dict[str, str]]:
    """Generate a finite, charge-preserving catalog of chemistry-rule candidates."""
    canonical = canonicalize_smiles(smiles)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:  # pragma: no cover - protected by canonicalize_smiles
        return []
    target_charge = formal_charge(canonical)
    candidates: dict[tuple[str, str], dict[str, str]] = {}
    ion_role = role in {"cation", "anion"}

    for chain in _terminal_alkyl_chains(molecule):
        terminal_index = chain[0]
        carbon_counts = (1, 2, 3, 4) if ion_role else (1, 2)
        for carbon_count in carbon_counts:
            if molecule.GetAtomWithIdx(terminal_index).GetTotalNumHs() < 1:
                continue
            editable = Chem.RWMol(molecule)
            previous = terminal_index
            for _ in range(carbon_count):
                added = editable.AddAtom(Chem.Atom(6))
                editable.AddBond(previous, added, Chem.BondType.SINGLE)
                previous = added
            rule = f"terminal_alkyl_extend_{carbon_count}"
            _record_rule_candidate(
                candidates,
                editable.GetMol(),
                canonical=canonical,
                target_charge=target_charge,
                rule=rule,
                family="terminal_alkyl",
            )

        for carbon_count in carbon_counts:
            if len(chain) <= carbon_count:
                continue
            editable = Chem.RWMol(molecule)
            for atom_index in sorted(chain[:carbon_count], reverse=True):
                editable.RemoveAtom(atom_index)
            rule = f"terminal_alkyl_shorten_{carbon_count}"
            _record_rule_candidate(
                candidates,
                editable.GetMol(),
                canonical=canonical,
                target_charge=target_charge,
                rule=rule,
                family="terminal_alkyl",
            )

        if ion_role and len(chain) >= 3:
            editable = Chem.RWMol(molecule)
            editable.RemoveBond(chain[0], chain[1])
            editable.AddBond(chain[0], chain[2], Chem.BondType.SINGLE)
            _record_rule_candidate(
                candidates,
                editable.GetMol(),
                canonical=canonical,
                target_charge=target_charge,
                rule="terminal_alkyl_linear_to_branch",
                family="alkyl_branching",
            )

    if ion_role:
        for branch_atom in molecule.GetAtoms():
            if not _is_alkyl_carbon(branch_atom):
                continue
            heavy_neighbors = [
                neighbor
                for neighbor in branch_atom.GetNeighbors()
                if neighbor.GetAtomicNum() > 1
            ]
            if len(heavy_neighbors) != 3:
                continue
            terminal_neighbors = [
                neighbor
                for neighbor in heavy_neighbors
                if _is_alkyl_carbon(neighbor)
                and len(
                    [
                        candidate
                        for candidate in neighbor.GetNeighbors()
                        if candidate.GetAtomicNum() > 1
                    ]
                )
                == 1
            ]
            if len(terminal_neighbors) != 2:
                continue
            first, second = sorted(
                (neighbor.GetIdx() for neighbor in terminal_neighbors)
            )
            editable = Chem.RWMol(molecule)
            editable.RemoveBond(branch_atom.GetIdx(), second)
            editable.AddBond(first, second, Chem.BondType.SINGLE)
            _record_rule_candidate(
                candidates,
                editable.GetMol(),
                canonical=canonical,
                target_charge=target_charge,
                rule="terminal_alkyl_branch_to_linear",
                family="alkyl_branching",
            )

    halogen_symbols = {9: "F", 17: "Cl", 35: "Br"}
    if ion_role:
        halogen_symbols[53] = "I"

    for atom in molecule.GetAtoms():
        source_atomic_number = atom.GetAtomicNum()
        if source_atomic_number not in halogen_symbols:
            continue
        for target_atomic_number, target_symbol in halogen_symbols.items():
            if target_atomic_number == source_atomic_number:
                continue
            if 53 in {source_atomic_number, target_atomic_number}:
                carbon_bound_or_isolated = (
                    atom.GetDegree() == 0
                    or (
                        atom.GetDegree() == 1
                        and atom.GetNeighbors()[0].GetAtomicNum() == 6
                    )
                )
                if not carbon_bound_or_isolated:
                    continue
            rule = (
                f"halogen_{halogen_symbols[source_atomic_number]}_to_{target_symbol}"
            )
            _record_rule_candidate(
                candidates,
                _replace_atom(
                    molecule,
                    atom.GetIdx(),
                    target_atomic_number,
                ),
                canonical=canonical,
                target_charge=target_charge,
                rule=rule,
                family="halogen",
            )

    if role == "cation":
        headgroup_symbols = {7: "N", 15: "P"}
        for atom in molecule.GetAtoms():
            if (
                atom.GetAtomicNum() not in headgroup_symbols
                or atom.GetFormalCharge() != 1
                or atom.GetIsAromatic()
                or atom.GetDegree() != 4
                or atom.GetTotalNumHs() != 0
                or any(
                    bond.GetBondType() != Chem.BondType.SINGLE
                    for bond in atom.GetBonds()
                )
            ):
                continue
            target_atomic_number = 15 if atom.GetAtomicNum() == 7 else 7
            rule = (
                f"cation_headgroup_{headgroup_symbols[atom.GetAtomicNum()]}"
                f"_to_{headgroup_symbols[target_atomic_number]}"
            )
            _record_rule_candidate(
                candidates,
                _replace_atom(
                    molecule,
                    atom.GetIdx(),
                    target_atomic_number,
                    no_implicit=True,
                ),
                canonical=canonical,
                target_charge=target_charge,
                rule=rule,
                family="cation_headgroup",
            )

    if role == "anion":
        for chain in _perfluoroalkyl_chains(molecule):
            terminal_index = chain[-1]
            terminal_fluorines = sorted(
                neighbor.GetIdx()
                for neighbor in molecule.GetAtomWithIdx(
                    terminal_index
                ).GetNeighbors()
                if neighbor.GetAtomicNum() == 9
            )
            if len(terminal_fluorines) != 3:
                continue
            for unit_count in (1, 2):
                editable = Chem.RWMol(molecule)
                first_carbon = editable.GetAtomWithIdx(terminal_fluorines[0])
                first_carbon.SetAtomicNum(6)
                first_carbon.SetFormalCharge(0)
                first_carbon.SetNumExplicitHs(0)
                first_carbon.SetNoImplicit(True)
                previous_index = terminal_fluorines[0]
                for _ in range(2):
                    fluorine = editable.AddAtom(Chem.Atom(9))
                    editable.AddBond(
                        previous_index,
                        fluorine,
                        Chem.BondType.SINGLE,
                    )
                for _ in range(unit_count - 1):
                    carbon = Chem.Atom(6)
                    carbon.SetNoImplicit(True)
                    carbon_index = editable.AddAtom(carbon)
                    editable.AddBond(
                        previous_index,
                        carbon_index,
                        Chem.BondType.SINGLE,
                    )
                    previous_index = carbon_index
                    for _ in range(2):
                        fluorine = editable.AddAtom(Chem.Atom(9))
                        editable.AddBond(
                            previous_index,
                            fluorine,
                            Chem.BondType.SINGLE,
                        )
                terminal_fluorine = editable.AddAtom(Chem.Atom(9))
                editable.AddBond(
                    previous_index,
                    terminal_fluorine,
                    Chem.BondType.SINGLE,
                )
                rule = f"perfluoroalkyl_extend_{unit_count}"
                _record_rule_candidate(
                    candidates,
                    editable.GetMol(),
                    canonical=canonical,
                    target_charge=target_charge,
                    rule=rule,
                    family="perfluoroalkyl",
                )

            for unit_count in (1, 2):
                if len(chain) <= unit_count:
                    continue
                removed_indices: set[int] = set(chain[-unit_count:])
                for carbon_index in chain[-unit_count:]:
                    removed_indices.update(
                        neighbor.GetIdx()
                        for neighbor in molecule.GetAtomWithIdx(
                            carbon_index
                        ).GetNeighbors()
                        if neighbor.GetAtomicNum() == 9
                    )
                new_terminal_index = chain[-unit_count - 1]
                shifted_terminal_index = new_terminal_index - sum(
                    index < new_terminal_index
                    for index in removed_indices
                )
                editable = Chem.RWMol(molecule)
                for atom_index in sorted(removed_indices, reverse=True):
                    editable.RemoveAtom(atom_index)
                fluorine = editable.AddAtom(Chem.Atom(9))
                editable.AddBond(
                    shifted_terminal_index,
                    fluorine,
                    Chem.BondType.SINGLE,
                )
                rule = f"perfluoroalkyl_shorten_{unit_count}"
                _record_rule_candidate(
                    candidates,
                    editable.GetMol(),
                    canonical=canonical,
                    target_charge=target_charge,
                    rule=rule,
                    family="perfluoroalkyl",
                )

    if ion_role:
        chalcogen_symbols = {8: "O", 16: "S"}
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() not in chalcogen_symbols:
                continue
            bonds = list(atom.GetBonds())
            family: str | None = None
            if (
                atom.GetFormalCharge() == -1
                and atom.GetDegree() == 1
                and len(bonds) == 1
                and bonds[0].GetBondType() == Chem.BondType.SINGLE
            ):
                family = "anionic_chalcogen"
            elif atom.GetFormalCharge() == 0:
                if (
                    atom.GetDegree() == 1
                    and atom.GetTotalNumHs() == 1
                    and len(bonds) == 1
                    and bonds[0].GetBondType() == Chem.BondType.SINGLE
                ):
                    family = "hydroxyl_thiol"
                elif (
                    atom.GetDegree() == 2
                    and atom.GetTotalNumHs() == 0
                    and all(
                        bond.GetBondType() == Chem.BondType.SINGLE
                        for bond in bonds
                    )
                ):
                    family = "ether_thioether"
                elif (
                    atom.GetDegree() == 1
                    and len(bonds) == 1
                    and bonds[0].GetBondType() == Chem.BondType.DOUBLE
                    and bonds[0].GetOtherAtom(atom).GetAtomicNum() == 6
                ):
                    family = "carbonyl_thiocarbonyl"
            if family is None:
                continue
            target_atomic_number = 16 if atom.GetAtomicNum() == 8 else 8
            rule = (
                f"{family}_{chalcogen_symbols[atom.GetAtomicNum()]}"
                f"_to_{chalcogen_symbols[target_atomic_number]}"
            )
            _record_rule_candidate(
                candidates,
                _replace_atom(
                    molecule,
                    atom.GetIdx(),
                    target_atomic_number,
                ),
                canonical=canonical,
                target_charge=target_charge,
                rule=rule,
                family=family,
            )

        atom_rings = molecule.GetRingInfo().AtomRings()
        for atom in molecule.GetAtoms():
            memberships = [
                ring
                for ring in atom_rings
                if atom.GetIdx() in ring and len(ring) in {5, 6}
            ]
            if (
                len(memberships) != 1
                or molecule.GetRingInfo().NumAtomRings(atom.GetIdx()) != 1
                or not atom.GetIsAromatic()
            ):
                continue
            if (
                atom.GetAtomicNum() == 6
                and atom.GetFormalCharge() == 0
                and atom.GetDegree() == 2
                and atom.GetTotalNumHs() == 1
            ):
                target_atomic_number = 7
                rule = "aromatic_C_to_N"
                no_implicit = True
            elif (
                atom.GetAtomicNum() == 7
                and atom.GetFormalCharge() == 0
                and atom.GetDegree() == 2
                and atom.GetTotalNumHs() == 0
            ):
                target_atomic_number = 6
                rule = "aromatic_N_to_C"
                no_implicit = False
            else:
                continue
            _record_rule_candidate(
                candidates,
                _replace_atom(
                    molecule,
                    atom.GetIdx(),
                    target_atomic_number,
                    no_implicit=no_implicit,
                ),
                canonical=canonical,
                target_charge=target_charge,
                rule=rule,
                family="aromatic_CH_N",
            )

    return [
        candidates[key]
        for key in sorted(candidates, key=lambda item: (item[0], item[1]))
    ]


def candidate_allowed(seed_smiles: str, candidate_smiles: str, role: str) -> bool:
    try:
        seed = canonicalize_smiles(seed_smiles)
        candidate = canonicalize_smiles(candidate_smiles)
    except TrainingSplitError:
        return False
    if candidate == seed or fragment_count(candidate) != 1:
        return False
    seed_charge = formal_charge(seed)
    candidate_charge = formal_charge(candidate)
    if candidate_charge != seed_charge:
        return False
    if role == "cation" and candidate_charge <= 0:
        return False
    if role == "anion" and candidate_charge >= 0:
        return False
    return True


def _interleave(
    rule_candidates: Sequence[dict[str, object]],
    similarity_candidates: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    interleaved: list[dict[str, object]] = []
    longest = max(len(rule_candidates), len(similarity_candidates), 0)
    for index in range(longest):
        if index < len(rule_candidates):
            interleaved.append(rule_candidates[index])
        if index < len(similarity_candidates):
            interleaved.append(similarity_candidates[index])
    return interleaved


def _round_robin_rule_candidates(
    candidates: Sequence[dict[str, object]],
    seed_identifier: str,
) -> list[dict[str, object]]:
    by_family: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        family = str(candidate["family"])
        by_family.setdefault(family, []).append(dict(candidate))
    for family_candidates in by_family.values():
        family_candidates.sort(
            key=lambda candidate: (
                str(candidate["rule"]),
                str(candidate["candidate_smiles"]),
            )
        )
    families = sorted(
        by_family,
        key=lambda family: stable_id(
            "stage1_rule_family",
            seed_identifier,
            family,
        ),
    )
    ordered: list[dict[str, object]] = []
    longest = max((len(rows) for rows in by_family.values()), default=0)
    for index in range(longest):
        for family in families:
            rows = by_family[family]
            if index < len(rows):
                ordered.append(rows[index])
    return ordered


def _migrate_pubchem_cache(output_root: Path) -> Path:
    cache_path = output_root / ".cache" / "pubchem.sqlite"
    legacy_root = output_root / "cache"
    legacy_path = legacy_root / "pubchem.sqlite"
    if not cache_path.exists() and legacy_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_path), str(cache_path))
        shutil.rmtree(legacy_root)
    return cache_path


def _write_stage1_entity_files(
    stage1_root: Path,
    combined: pd.DataFrame,
) -> None:
    il_path = stage1_root / "IL.csv"
    if not il_path.exists():
        raise TrainingSplitError(
            "Run extract-pretrain before augment-pretrain; missing "
            f"{il_path}"
        )
    with tempfile.TemporaryDirectory(
        dir=stage1_root.parent.parent
    ) as temporary_dir:
        staged_stage1 = Path(temporary_dir) / "stage1"
        staged_stage1.mkdir()
        shutil.copy2(il_path, staged_stage1 / "IL.csv")
        for role in ROLE_BY_COLUMN.values():
            role_frame = (
                combined.loc[combined["role"].eq(role), PRETRAIN_ENTITY_COLUMNS]
                .sort_values("SMILES", kind="stable")
                .reset_index(drop=True)
            )
            role_frame.to_csv(
                staged_stage1 / f"{role}.csv",
                index=False,
                lineterminator="\n",
            )
        replace_directory(staged_stage1, stage1_root)


def augment_pretraining_entities(
    output_root: Path,
    *,
    pretrain_cap: int = 500_000,
    threshold: int = 90,
    max_records: int = 100,
    offline: bool = False,
    rate_limit: float = 4.0,
    client: PubChemClient | None = None,
) -> pd.DataFrame:
    """Augment entities using PubChem similarity and local chemistry rules."""
    output_root = Path(output_root)
    stage1_root = output_root / "stage1"
    existing = _combined_entity_frame(stage1_root)
    original_mask = existing["origin_list"].map(
        lambda value: "dataset" in _split_joined(value)
    )
    base = existing.loc[original_mask].copy()
    base = base.sort_values(["role", "SMILES"], kind="stable").reset_index(drop=True)
    if len(base) > pretrain_cap:
        raise TrainingSplitError(
            f"Base pretraining corpus has {len(base)} entities, exceeding cap "
            f"{pretrain_cap}"
        )
    remaining = pretrain_cap - len(base)
    if remaining == 0:
        _write_stage1_entity_files(stage1_root, base)
        return base

    owns_client = client is None
    active_client = client or PubChemClient(
        _migrate_pubchem_cache(output_root),
        offline=offline,
        rate_limit=rate_limit,
    )
    incomplete: list[str] = []

    try:
        with tempfile.TemporaryDirectory(dir=output_root.parent) as temporary_dir:
            temporary_root = Path(temporary_dir)
            candidate_database = sqlite3.connect(temporary_root / "candidates.sqlite")
            candidate_database.execute(
                """
                CREATE TABLE candidates (
                    seed_entity_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    seed_smiles TEXT NOT NULL,
                    candidate_smiles TEXT NOT NULL,
                    method TEXT NOT NULL,
                    pubchem_cid TEXT,
                    rule TEXT,
                    method_rank INTEGER NOT NULL,
                    seed_rank INTEGER NOT NULL,
                    PRIMARY KEY (
                        seed_entity_id,
                        candidate_smiles,
                        method,
                        rule
                    )
                )
                """
            )

            for seed in base.itertuples(index=False):
                seed_smiles = canonicalize_smiles(str(seed.SMILES))
                role = str(seed.role)
                seed_identifier = entity_id(role, seed_smiles)
                if fragment_count(seed_smiles) != 1:
                    continue

                similarity_candidates: list[dict[str, object]] = []
                try:
                    records = active_client.similarity(
                        seed_smiles,
                        threshold=threshold,
                        max_records=max_records,
                    )
                except IncompletePubChemQuery as exc:
                    incomplete.append(str(exc))
                    records = []
                seen_similarity: set[str] = set()
                for rank, record in enumerate(records):
                    raw_candidate = smiles_from_pubchem_record(record)
                    if raw_candidate is None:
                        continue
                    try:
                        candidate = canonicalize_smiles(raw_candidate)
                    except TrainingSplitError:
                        continue
                    if (
                        candidate in seen_similarity
                        or not candidate_allowed(seed_smiles, candidate, role)
                    ):
                        continue
                    seen_similarity.add(candidate)
                    similarity_candidates.append(
                        {
                            "candidate_smiles": candidate,
                            "method": "pubchem_similarity",
                            "pubchem_cid": str(record.get("CID", "")),
                            "rule": "",
                            "method_rank": rank,
                        }
                    )

                local_rules: list[dict[str, object]] = []
                for generated in generate_rule_candidates(seed_smiles, role):
                    generated_smiles = generated["SMILES"]
                    if not candidate_allowed(seed_smiles, generated_smiles, role):
                        continue
                    local_rules.append(
                        {
                            "candidate_smiles": generated_smiles,
                            "method": "rule",
                            "pubchem_cid": "",
                            "rule": generated["rule"],
                            "family": generated["family"],
                        }
                    )

                ordered_rules = _round_robin_rule_candidates(
                    local_rules,
                    seed_identifier,
                )
                for rank, candidate in enumerate(ordered_rules):
                    candidate["method_rank"] = rank
                ordered = _interleave(ordered_rules, similarity_candidates)
                for seed_rank, candidate in enumerate(ordered):
                    candidate_database.execute(
                        """
                        INSERT OR IGNORE INTO candidates(
                            seed_entity_id,
                            role,
                            seed_smiles,
                            candidate_smiles,
                            method,
                            pubchem_cid,
                            rule,
                            method_rank,
                            seed_rank
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            seed_identifier,
                            role,
                            seed_smiles,
                            candidate["candidate_smiles"],
                            candidate["method"],
                            candidate["pubchem_cid"],
                            candidate["rule"],
                            candidate["method_rank"],
                            seed_rank,
                        ),
                    )
                candidate_database.commit()

            if incomplete:
                candidate_database.close()
                examples = "; ".join(sorted(set(incomplete))[:3])
                raise IncompletePubChemQuery(
                    f"{len(incomplete)} PubChem requests are incomplete; "
                    f"rerun to resume. Examples: {examples}"
                )

            base_keys = {
                (str(row.role), canonicalize_smiles(str(row.SMILES)))
                for row in base.itertuples(index=False)
            }
            selected: dict[tuple[str, str], dict[str, object]] = {}
            cursor = candidate_database.execute(
                """
                SELECT role, candidate_smiles
                FROM candidates
                ORDER BY seed_rank, seed_entity_id, method, candidate_smiles
                """
            )
            for role, candidate_smiles in cursor:
                key = (role, candidate_smiles)
                if key in base_keys or key in selected:
                    continue
                selected[key] = {
                    "role": role,
                    "SMILES": candidate_smiles,
                    "formal_charge": formal_charge(candidate_smiles),
                    "origin_list": set(),
                    "seed_smiles_list": set(),
                    "rule_list": set(),
                    "pubchem_cid_list": set(),
                    "mol_id_list": set(),
                }
                if len(selected) >= remaining:
                    break

            candidate_database.execute(
                """
                CREATE TABLE selected (
                    role TEXT NOT NULL,
                    candidate_smiles TEXT NOT NULL,
                    PRIMARY KEY(role, candidate_smiles)
                )
                """
            )
            candidate_database.executemany(
                "INSERT INTO selected(role, candidate_smiles) VALUES (?, ?)",
                selected,
            )
            for row in candidate_database.execute(
                """
                SELECT
                    c.seed_entity_id,
                    c.role,
                    c.seed_smiles,
                    c.candidate_smiles,
                    c.method,
                    c.pubchem_cid,
                    c.rule,
                    c.method_rank
                FROM candidates c
                INNER JOIN selected s
                    ON c.role = s.role
                    AND c.candidate_smiles = s.candidate_smiles
                ORDER BY
                    c.role,
                    c.candidate_smiles,
                    c.seed_entity_id,
                    c.method,
                    c.rule
                """
            ):
                (
                    seed_entity_identifier,
                    role,
                    seed_smiles,
                    candidate_smiles,
                    method,
                    pubchem_cid,
                    rule,
                    _method_rank,
                ) = row
                record = selected[(role, candidate_smiles)]
                origins = record["origin_list"]
                seeds = record["seed_smiles_list"]
                pubchem_cids = record["pubchem_cid_list"]
                rules = record["rule_list"]
                if isinstance(origins, set):
                    if method == "rule":
                        origins.add("rule")
                    elif method == "pubchem_similarity":
                        origins.add("pubchem")
                if isinstance(seeds, set):
                    seeds.add(seed_smiles)
                if (
                    isinstance(pubchem_cids, set)
                    and str(pubchem_cid).strip()
                ):
                    pubchem_cids.add(str(pubchem_cid).strip())
                if isinstance(rules, set) and str(rule).strip():
                    rules.add(str(rule).strip())
            candidate_database.close()

            all_rows: list[dict[str, object]] = []
            for row in base.to_dict(orient="records"):
                all_rows.append(
                    {
                        "role": str(row["role"]),
                        "SMILES": canonicalize_smiles(str(row["SMILES"])),
                        "formal_charge": int(row["formal_charge"]),
                        "origin_list": _split_joined(row["origin_list"]),
                        "seed_smiles_list": _split_joined(
                            row["seed_smiles_list"]
                        ),
                        "rule_list": _split_joined(row["rule_list"]),
                        "pubchem_cid_list": _split_joined(
                            row["pubchem_cid_list"]
                        ),
                        "mol_id_list": _split_joined(row["mol_id_list"]),
                    }
                )
            all_rows.extend(selected.values())
            augmented = pd.DataFrame(
                [
                    {
                        "role": row["role"],
                        **_entity_output_frame([row]).iloc[0].to_dict(),
                    }
                    for row in sorted(
                        all_rows,
                        key=lambda item: (
                            str(item["role"]),
                            str(item["SMILES"]),
                        ),
                    )
                ],
                columns=["role", *PRETRAIN_ENTITY_COLUMNS],
            )
            _write_stage1_entity_files(stage1_root, augmented)
    finally:
        if owns_client:
            active_client.close()

    (output_root / "manifest.json").unlink(missing_ok=True)
    return augmented


def system_type_for_columns(identity_columns: Sequence[str]) -> str:
    identity_set = set(identity_columns)
    for system_type, columns in SYSTEM_COLUMNS.items():
        if identity_set == set(columns):
            return system_type
    raise TrainingSplitError(
        "Unsupported identity-column combination: "
        + ", ".join(identity_columns)
    )


def discover_tasks(final_root: Path) -> list[TaskSpec]:
    final_root = Path(final_root)
    paths = final_csv_paths(final_root)
    relative_paths = {
        path.relative_to(final_root).as_posix()
        for path in paths
    }
    missing_stage2 = STAGE2_FILES - relative_paths
    if missing_stage2:
        raise TrainingSplitError(
            "Missing required stage-2 datasets: "
            + ", ".join(sorted(missing_stage2))
        )

    tasks: list[TaskSpec] = []
    for path in paths:
        relative = path.relative_to(final_root).as_posix()
        columns = list(pd.read_csv(path, nrows=0).columns)
        identity_columns = tuple(
            column for column in IDENTITY_COLUMNS if column in columns
        )
        if not identity_columns:
            raise TrainingSplitError(f"No identity columns in {relative}")
        system_type = system_type_for_columns(identity_columns)
        target_columns = tuple(
            column
            for column in columns
            if column not in identity_columns
            and column not in CONDITION_COLUMNS
            and column not in METADATA_COLUMNS
        )
        stage = 2 if relative in STAGE2_FILES else 3
        if relative == "simulation/simulated_qm_elec_hf.csv":
            missing_qm = set(QM_TARGET_COLUMNS) - set(target_columns)
            extra_qm = set(target_columns) - set(QM_TARGET_COLUMNS)
            if missing_qm or extra_qm:
                raise TrainingSplitError(
                    f"Unexpected QM target columns in {relative}; "
                    f"missing={sorted(missing_qm)}, extra={sorted(extra_qm)}"
                )
            target_columns = QM_TARGET_COLUMNS
        elif len(target_columns) != 1:
            raise TrainingSplitError(
                f"Expected exactly one target column in {relative}, "
                f"found {list(target_columns)}"
            )
        tasks.append(
            TaskSpec(
                task_id=relative.removesuffix(".csv"),
                stage=stage,
                source_file=relative,
                target_columns=target_columns,
                identity_columns=identity_columns,
                system_type=system_type,
            )
        )
    return sorted(tasks, key=lambda task: (task.stage, task.source_file))


def tier_for_system_count(system_count: int) -> str:
    if system_count > 500:
        return "large"
    if system_count >= 50:
        return "medium"
    return "small"


def group_id(group_type: str, values: Sequence[str]) -> str:
    return stable_id("group", group_type, *values)


def _canonicalize_identity_column(
    values: pd.Series,
    *,
    source_file: str,
    column: str,
) -> pd.Series:
    mapping: dict[object, str] = {}
    for value in pd.unique(values):
        if pd.isna(value) or not str(value).strip():
            raise TrainingSplitError(
                f"Missing identity value in {source_file}, column {column}"
            )
        try:
            mapping[value] = canonicalize_smiles(str(value))
        except TrainingSplitError as exc:
            raise TrainingSplitError(
                f"{source_file}, column {column}: {exc}"
            ) from exc
    return values.map(mapping)


def _deduplicate_charge(frame: pd.DataFrame, source_file: str) -> pd.DataFrame:
    if "charge" not in frame.columns or "mol_id" not in frame.columns:
        raise TrainingSplitError(
            f"{source_file} must contain charge and mol_id columns"
        )
    charges = pd.to_numeric(frame["charge"], errors="coerce")
    if charges.isna().any():
        raise TrainingSplitError(f"Non-numeric charge label in {source_file}")
    frame = frame.copy()
    frame["charge"] = charges
    records: list[pd.Series] = []
    for smiles, group in frame.groupby("SMILES", sort=False, dropna=False):
        unique_charges = pd.unique(group["charge"])
        if len(unique_charges) != 1:
            values = sorted(float(value) for value in unique_charges)
            raise TrainingSplitError(
                f"Conflicting charge labels for {smiles}: {values}"
            )
        representative = group.sort_values("_source_row", kind="stable").iloc[0].copy()
        representative["_mol_ids"] = ";".join(
            dict.fromkeys(group["mol_id"].astype(str).tolist())
        )
        records.append(representative)
    if not records:
        result = frame.iloc[0:0].copy()
        result["_mol_ids"] = pd.Series(dtype="object")
        return result
    return pd.DataFrame(records).sort_values("_source_row", kind="stable").reset_index(
        drop=True
    )


def prepare_task_frame(
    final_root: Path,
    task: TaskSpec,
    checksum: str,
) -> tuple[pd.DataFrame, int]:
    path = final_root / task.source_file
    frame = pd.read_csv(path)
    raw_rows = len(frame)
    frame["_source_row"] = np.arange(raw_rows, dtype=np.int64)
    for column in task.identity_columns:
        frame[column] = _canonicalize_identity_column(
            frame[column],
            source_file=task.source_file,
            column=column,
        )

    if task.source_file == "simulation/charge.csv":
        frame = _deduplicate_charge(frame, task.source_file)
    else:
        frame["_mol_ids"] = ""

    system_values = list(
        frame.loc[:, list(SYSTEM_COLUMNS[task.system_type])].itertuples(
            index=False,
            name=None,
        )
    )
    frame["_system_key"] = [
        json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
        for values in system_values
    ]
    frame["_system_id"] = [
        group_id(task.system_type, tuple(str(value) for value in values))
        for values in system_values
    ]
    frame["_row_id"] = [
        stable_id(
            "row",
            task.task_id,
            checksum,
            int(source_row),
        )
        for source_row in frame["_source_row"]
    ]
    for column in IDENTITY_COLUMNS:
        internal_column = f"_{ROLE_BY_COLUMN[column]}_id"
        if column in task.identity_columns:
            role = ROLE_BY_COLUMN[column]
            frame[internal_column] = [
                group_id(role, (str(value),))
                for value in frame[column]
            ]
        else:
            frame[internal_column] = ""
    if {"cation", "anion"}.issubset(task.identity_columns):
        frame["_il_id"] = _group_series(
            frame,
            "il",
            ("cation", "anion"),
        )
    else:
        frame["_il_id"] = ""
    return frame, raw_rows


def _group_series(
    frame: pd.DataFrame,
    group_type: str,
    columns: Sequence[str],
) -> pd.Series:
    values = list(
        frame.loc[:, list(columns)].itertuples(index=False, name=None)
    )
    return pd.Series(
        [
            group_id(group_type, tuple(str(value) for value in row))
            for row in values
        ],
        index=frame.index,
        dtype="object",
    )


def _catalog_frame(
    frame: pd.DataFrame,
    task: TaskSpec,
    partitions: pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": frame["_row_id"],
            "task_id": task.task_id,
            "source_file": task.source_file,
            "source_row": frame["_source_row"].astype("int64"),
            "system_type": task.system_type,
            "system_id": frame["_system_id"],
            "system_key": frame["_system_key"],
            "partition": partitions,
            "cation_id": frame["_cation_id"],
            "anion_id": frame["_anion_id"],
            "il_id": frame["_il_id"],
            "solute_id": frame["_solute_id"],
            "solvent_id": frame["_solvent_id"],
            "molecule_id": frame["_molecule_id"],
            "mol_ids": frame["_mol_ids"],
        },
        columns=ROW_CATALOG_COLUMNS,
    )


def _append_frame(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        lineterminator="\n",
    )


def _numeric_stats(values: pd.Series) -> dict[str, object]:
    numeric = pd.to_numeric(values, errors="coerce")
    present = numeric.dropna()
    total = len(numeric)
    if present.empty:
        return {
            "label_count": 0,
            "missing_rate": 1.0 if total else 0.0,
            "value_min": pd.NA,
            "value_p05": pd.NA,
            "value_median": pd.NA,
            "value_mean": pd.NA,
            "value_p95": pd.NA,
            "value_max": pd.NA,
        }
    return {
        "label_count": int(len(present)),
        "missing_rate": float(1.0 - len(present) / total) if total else 0.0,
        "value_min": float(present.min()),
        "value_p05": float(present.quantile(0.05)),
        "value_median": float(present.median()),
        "value_mean": float(present.mean()),
        "value_p95": float(present.quantile(0.95)),
        "value_max": float(present.max()),
    }


def label_summary_rows(
    frame: pd.DataFrame,
    task: TaskSpec,
    mask: pd.Series,
    *,
    stage: int,
    strategy: str,
    partition: str,
    repeat: int | str = "",
    fold: int | str = "",
) -> list[dict[str, object]]:
    subset = frame.loc[mask]
    rows: list[dict[str, object]] = []
    for target_column in task.target_columns:
        row = {
            "stage": stage,
            "task_id": task.task_id,
            "strategy": strategy,
            "repeat": repeat,
            "fold": fold,
            "partition": partition,
            "target_column": target_column,
            "row_count": int(len(subset)),
            "system_count": int(subset["_system_id"].nunique()),
        }
        row.update(_numeric_stats(subset[target_column]))
        rows.append(row)
    return rows


def _update_role_entity_sets(
    target: dict[str, set[str]],
    frame: pd.DataFrame,
    task: TaskSpec,
    mask: pd.Series,
) -> None:
    subset = frame.loc[mask]
    for column in task.identity_columns:
        role = ROLE_BY_COLUMN[column]
        destination = target.setdefault(role, set())
        for value in pd.unique(subset[column]):
            for smiles in role_fragments(str(value), role):
                destination.add(entity_id(role, smiles))


def _task_strategy_units(task: TaskSpec) -> dict[str, str]:
    units = {
        "random": "system_id" if task.system_type in SINGLE_SYSTEM_TYPES else "row_id"
    }
    for strategy in GROUP_STRATEGY_COLUMNS[task.system_type]:
        units[strategy] = (
            "system_id"
            if strategy == task.system_type
            else f"{strategy}_id"
        )
    return units


def _profile_task(
    frame: pd.DataFrame,
    raw_rows: int,
) -> TaskProfile:
    system_count = int(frame["_system_id"].nunique())
    return TaskProfile(
        raw_rows=raw_rows,
        rows=len(frame),
        system_count=system_count,
        tier=tier_for_system_count(system_count),
    )


def _system_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            ["_system_id", "_system_key"],
            sort=False,
            dropna=False,
        )
        .size()
        .rename("row_count")
        .reset_index()
    )


def plan_joint_test_registry(
    stage3_tasks: Sequence[TaskSpec],
    profiles: Mapping[str, TaskProfile],
    systems_by_task: Mapping[str, pd.DataFrame],
    *,
    seed: int,
) -> dict[tuple[str, str], dict[str, object]]:
    """Select shared large-task test systems while avoiding smaller tasks."""
    candidates: dict[tuple[str, str], dict[str, object]] = {}
    targets = {
        task.task_id: max(
            1,
            int(round(profiles[task.task_id].system_count * 0.1)),
        )
        for task in stage3_tasks
        if profiles[task.task_id].tier == "large"
    }
    task_by_id = {task.task_id: task for task in stage3_tasks}

    for task in stage3_tasks:
        tier = profiles[task.task_id].tier
        for system_id_value, system_key, row_count in systems_by_task[
            task.task_id
        ].itertuples(index=False, name=None):
            key = (task.system_type, str(system_id_value))
            entry = candidates.setdefault(
                key,
                {
                    "system_type": task.system_type,
                    "system_id": str(system_id_value),
                    "system_key": str(system_key),
                    "large_tasks": set(),
                    "other_tasks": set(),
                    "other_rows": 0,
                },
            )
            if tier == "large":
                large_tasks = entry["large_tasks"]
                if isinstance(large_tasks, set):
                    large_tasks.add(task.task_id)
            else:
                other_tasks = entry["other_tasks"]
                if isinstance(other_tasks, set):
                    other_tasks.add(task.task_id)
                entry["other_rows"] = int(entry["other_rows"]) + int(row_count)

    candidates = {
        key: entry
        for key, entry in candidates.items()
        if entry["large_tasks"]
    }
    selected: dict[tuple[str, str], dict[str, object]] = {}
    selected_counts = {task_id: 0 for task_id in targets}

    for entry in candidates.values():
        entry["selection_hash"] = stable_id(
            f"seed:{seed}:stage3_joint_test",
            entry["system_type"],
            entry["system_key"],
        )

    def choose(
        key: tuple[str, str],
        entry: dict[str, object],
    ) -> None:
        selected[key] = {
            "system_type": entry["system_type"],
            "system_id": entry["system_id"],
            "system_key": entry["system_key"],
            "source_tasks": set(entry["large_tasks"]),
            "reserved_tasks": set(entry["other_tasks"]),
        }
        large_tasks = entry["large_tasks"]
        if isinstance(large_tasks, set):
            for task_id in large_tasks:
                selected_counts[task_id] += 1

    def fill_without_overshoot(*, unsafe: bool) -> None:
        pool = [
            (key, entry)
            for key, entry in candidates.items()
            if bool(entry["other_tasks"]) == unsafe
        ]
        pool.sort(
            key=lambda item: (
                len(item[1]["other_tasks"]) if unsafe else 0,
                int(item[1]["other_rows"]) if unsafe else 0,
                -len(item[1]["large_tasks"]),
                str(item[1]["selection_hash"]),
            )
        )
        for key, entry in pool:
            if key in selected:
                continue
            large_tasks = entry["large_tasks"]
            if not isinstance(large_tasks, set):
                continue
            underfilled = {
                task_id
                for task_id in large_tasks
                if selected_counts[task_id] < targets[task_id]
            }
            if underfilled and underfilled == large_tasks:
                choose(key, entry)

    def fill_with_minimum_overshoot(*, unsafe: bool) -> None:
        while any(
            selected_counts[task_id] < target
            for task_id, target in targets.items()
        ):
            eligible: list[
                tuple[
                    tuple[object, ...],
                    tuple[str, str],
                    dict[str, object],
                ]
            ] = []
            for key, entry in candidates.items():
                if key in selected or bool(entry["other_tasks"]) != unsafe:
                    continue
                large_tasks = entry["large_tasks"]
                if not isinstance(large_tasks, set):
                    continue
                underfilled = {
                    task_id
                    for task_id in large_tasks
                    if selected_counts[task_id] < targets[task_id]
                }
                if not underfilled:
                    continue
                overshoot = sum(
                    max(
                        0,
                        selected_counts[task_id] + 1 - targets[task_id],
                    )
                    for task_id in large_tasks
                )
                eligible.append(
                    (
                        (
                            overshoot,
                            len(entry["other_tasks"]),
                            int(entry["other_rows"]),
                            -len(underfilled),
                            -len(large_tasks),
                            str(entry["selection_hash"]),
                        ),
                        key,
                        entry,
                    )
                )
            if not eligible:
                return
            _, key, entry = min(eligible, key=lambda item: item[0])
            choose(key, entry)

    fill_without_overshoot(unsafe=False)
    fill_with_minimum_overshoot(unsafe=False)
    if any(
        selected_counts[task_id] < target
        for task_id, target in targets.items()
    ):
        fill_without_overshoot(unsafe=True)
        fill_with_minimum_overshoot(unsafe=True)
    if any(
        selected_counts[task_id] < target
        for task_id, target in targets.items()
    ):
        missing = {
            task_id: targets[task_id] - selected_counts[task_id]
            for task_id in targets
            if selected_counts[task_id] < targets[task_id]
        }
        raise TrainingSplitError(
            f"Unable to fill Stage-3 fixed-test targets: {missing}"
        )

    for task_id, count in sorted(selected_counts.items()):
        target = targets[task_id]
        if count > target:
            print(
                f"Warning: {task_id} fixed test contains {count} systems; "
                f"target was {target} (+{count - target})"
            )

    for task_id, target in targets.items():
        task = task_by_id[task_id]
        actual = sum(
            task_id in entry["source_tasks"]
            for (system_type, _), entry in selected.items()
            if system_type == task.system_type
        )
        if actual < target:
            raise TrainingSplitError(
                f"Fixed test target not met for {task_id}: {actual} < {target}"
            )
    return selected


def _materialized_task_frame(
    frame: pd.DataFrame,
    task: TaskSpec,
) -> pd.DataFrame:
    visible_columns = [
        column for column in frame.columns if not column.startswith("_")
    ]
    result = frame.loc[:, visible_columns].copy()
    if task.source_file == "simulation/charge.csv":
        result = result.drop(columns=["mol_id"])
        result["mol_id_list"] = frame["_mol_ids"].astype(str).to_numpy()
    elif "mol_id" in result.columns:
        mol_ids = result.pop("mol_id")
        result["mol_id"] = mol_ids
    return result.reset_index(drop=True)


def _reserved_summary(
    frame: pd.DataFrame,
    task: TaskSpec,
    mask: pd.Series,
) -> pd.DataFrame:
    columns = list(SYSTEM_COLUMNS[task.system_type])
    if not mask.any():
        return pd.DataFrame(
            columns=[*columns, "row_count", "reason"]
        )
    summary = (
        frame.loc[mask]
        .groupby(columns, sort=True, dropna=False)
        .size()
        .rename("row_count")
        .reset_index()
    )
    summary["reason"] = "reserved_due_to_cross_task_test"
    return summary.loc[:, [*columns, "row_count", "reason"]]


def _loo_manifest(
    development: pd.DataFrame,
    task: TaskSpec,
) -> pd.DataFrame:
    columns = list(SYSTEM_COLUMNS[task.system_type])
    summary = (
        development.groupby(columns, sort=True, dropna=False)
        .size()
        .rename("row_count")
        .reset_index()
    )
    summary.insert(
        0,
        "loo_fold",
        np.arange(1, len(summary) + 1, dtype=np.int64),
    )
    return summary.loc[:, ["loo_fold", *columns, "row_count"]]


def _stage3_partitions(
    frame: pd.DataFrame,
    task: TaskSpec,
    profile: TaskProfile,
    test_registry: Mapping[tuple[str, str], Mapping[str, object]],
) -> pd.Series:
    reserved_mask = frame["_system_id"].map(
        lambda system_id_value: (
            task.system_type,
            str(system_id_value),
        )
        in test_registry
    )
    if profile.tier == "large":
        return pd.Series(
            np.where(reserved_mask, "test", "development"),
            index=frame.index,
        )
    return pd.Series(
        np.where(
            reserved_mask,
            "reserved_due_to_cross_task_test",
            "development",
        ),
        index=frame.index,
    )


def _overlap_audit_rows(
    output_root: Path,
    stage2_train_entities: Mapping[str, set[str]],
    stage2_validation_entities: Mapping[str, set[str]],
    stage3_test_entities: Mapping[str, set[str]],
) -> list[dict[str, object]]:
    pretrain_path = output_root / "stage1" / "entities.csv"
    pretrain_available = pretrain_path.exists()
    pretrain_entities: dict[str, set[str]] = {}
    if pretrain_available:
        pretrain = pd.read_csv(pretrain_path)
        for role, group in pretrain.groupby("role"):
            pretrain_entities[str(role)] = set(group["entity_id"].astype(str))

    comparisons = [
        (
            "stage1_pretrain",
            pretrain_entities,
            pretrain_available,
            "stage2_validation",
            stage2_validation_entities,
        ),
        (
            "stage1_pretrain",
            pretrain_entities,
            pretrain_available,
            "stage3_test",
            stage3_test_entities,
        ),
        (
            "stage2_train",
            stage2_train_entities,
            True,
            "stage3_test",
            stage3_test_entities,
        ),
    ]
    rows: list[dict[str, object]] = []
    for source_name, source, available, target_name, target in comparisons:
        roles = sorted(set(source) | set(target) | set(ROLE_BY_COLUMN.values()))
        for role in roles:
            source_ids = source.get(role, set())
            target_ids = target.get(role, set())
            overlap = source_ids & target_ids
            rows.append(
                {
                    "source_collection": source_name,
                    "target_collection": target_name,
                    "role": role,
                    "source_available": available,
                    "source_entities": len(source_ids),
                    "target_entities": len(target_ids),
                    "overlap_entities": len(overlap),
                    "target_overlap_ratio": (
                        len(overlap) / len(target_ids) if target_ids else 0.0
                    ),
                }
            )
    return rows


def _build_training_split_indexes_legacy(
    final_root: Path,
    output_root: Path,
    *,
    seed: int = 42,
    pretrain_cap: int = 500_000,
) -> pd.DataFrame:
    """Build compact stage-2 and stage-3 split indexes plus quality audits."""
    final_root = Path(final_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = discover_tasks(final_root)
    paths = final_csv_paths(final_root)
    checksums = {
        path.relative_to(final_root).as_posix(): file_sha256(path)
        for path in paths
    }

    stage2_tasks = [task for task in tasks if task.stage == 2]
    stage3_tasks = [task for task in tasks if task.stage == 3]
    stage3_profiles: dict[str, TaskProfile] = {}
    test_registry: dict[
        tuple[str, str],
        dict[str, object],
    ] = {}

    for task in stage3_tasks:
        frame, raw_rows = prepare_task_frame(
            final_root,
            task,
            checksums[task.source_file],
        )
        profile = _profile_task(frame, raw_rows)
        stage3_profiles[task.task_id] = profile
        if profile.tier != "large":
            continue
        unique_systems = frame[
            ["_system_id", "_system_key"]
        ].drop_duplicates()
        selected_count = 0
        for system_id_value, system_key in unique_systems.itertuples(
            index=False,
            name=None,
        ):
            if stable_fraction(
                seed,
                "stage3_test",
                task.system_type,
                system_id_value,
            ) >= 0.1:
                continue
            selected_count += 1
            key = (task.system_type, str(system_id_value))
            entry = test_registry.setdefault(
                key,
                {
                    "system_type": task.system_type,
                    "system_id": str(system_id_value),
                    "system_key": str(system_key),
                    "source_tasks": set(),
                },
            )
            cast_sources = entry["source_tasks"]
            if isinstance(cast_sources, set):
                cast_sources.add(task.task_id)
        if selected_count == 0:
            raise TrainingSplitError(
                f"Hash holdout selected no systems for large task {task.task_id}"
            )

    repeats_by_task = {
        task.task_id: (
            1 if stage3_profiles[task.task_id].tier == "large" else 5
        )
        for task in stage3_tasks
    }
    groups_by_strategy: dict[str, dict[str, set[str]]] = {}
    single_random_groups: dict[str, dict[str, set[str]]] = {}
    for task in stage3_tasks:
        frame, raw_rows = prepare_task_frame(
            final_root,
            task,
            checksums[task.source_file],
        )
        profile = stage3_profiles[task.task_id]
        if raw_rows != profile.raw_rows or len(frame) != profile.rows:
            raise TrainingSplitError(
                f"Input changed while planning folds for {task.task_id}"
            )
        partitions = _stage3_partitions(
            frame,
            task,
            profile,
            test_registry,
        )
        development = frame.loc[partitions.eq("development")]
        if task.system_type in SINGLE_SYSTEM_TYPES:
            single_random_groups.setdefault(task.system_type, {})[
                task.task_id
            ] = set(development["_system_id"].astype(str))
        for strategy, columns in GROUP_STRATEGY_COLUMNS[
            task.system_type
        ].items():
            groups_by_strategy.setdefault(strategy, {})[
                task.task_id
            ] = set(
                _group_series(
                    development,
                    strategy,
                    columns,
                ).astype(str)
            )

    stage3_group_assignments: dict[
        tuple[str, str, int],
        int,
    ] = {}
    for strategy, task_groups in groups_by_strategy.items():
        planned = plan_shared_group_folds(
            task_groups,
            repeats_by_task,
            seed=seed,
            namespace=f"stage3_group:{strategy}",
        )
        stage3_group_assignments.update(
            {
                (strategy, group_identifier, repeat): fold
                for (group_identifier, repeat), fold in planned.items()
            }
        )
    single_random_assignments: dict[
        tuple[str, str, int],
        int,
    ] = {}
    for system_type, task_groups in single_random_groups.items():
        planned = plan_shared_group_folds(
            task_groups,
            repeats_by_task,
            seed=seed,
            namespace=f"stage3_random_single:{system_type}",
        )
        single_random_assignments.update(
            {
                (system_type, group_identifier, repeat): fold
                for (group_identifier, repeat), fold in planned.items()
            }
        )

    task_catalog_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    overlap_checks: list[dict[str, object]] = []
    stage2_groups: dict[tuple[str, str], tuple[str, str]] = {}
    stage3_loo_rows: list[dict[str, object]] = []
    stage2_train_entities: dict[str, set[str]] = {}
    stage2_validation_entities: dict[str, set[str]] = {}
    stage3_test_entities: dict[str, set[str]] = {}

    with tempfile.TemporaryDirectory(dir=output_root.parent) as temporary_dir:
        staged_root = Path(temporary_dir) / "training_splits"
        stage2_root = staged_root / "stage2"
        stage3_root = staged_root / "stage3"
        audit_root = staged_root / "audit"
        stage2_root.mkdir(parents=True)
        stage3_root.mkdir()
        audit_root.mkdir()

        stage2_rows_path = stage2_root / "rows.csv"
        for task in stage2_tasks:
            frame, raw_rows = prepare_task_frame(
                final_root,
                task,
                checksums[task.source_file],
            )
            profile = _profile_task(frame, raw_rows)
            partitions = frame["_system_id"].map(
                lambda system_id_value: (
                    "validation"
                    if stable_fraction(
                        seed,
                        "stage2_validation",
                        task.system_type,
                        system_id_value,
                    )
                    < 0.1
                    else "train"
                )
            )
            if not {"train", "validation"}.issubset(set(partitions)):
                raise TrainingSplitError(
                    f"Stage-2 split is empty for {task.task_id}: "
                    f"{sorted(set(partitions))}"
                )
            _append_frame(
                stage2_rows_path,
                _catalog_frame(frame, task, partitions),
            )
            for system_id_value, system_key, partition in (
                pd.DataFrame(
                    {
                        "system_id": frame["_system_id"],
                        "system_key": frame["_system_key"],
                        "partition": partitions,
                    }
                )
                .drop_duplicates()
                .itertuples(index=False, name=None)
            ):
                key = (task.system_type, str(system_id_value))
                value = (str(system_key), str(partition))
                previous = stage2_groups.setdefault(key, value)
                if previous != value:
                    raise TrainingSplitError(
                        f"Inconsistent shared stage-2 assignment for {key}"
                    )

            for partition in ("train", "validation"):
                mask = partitions.eq(partition)
                label_rows.extend(
                    label_summary_rows(
                        frame,
                        task,
                        mask,
                        stage=2,
                        strategy="system_holdout",
                        partition=partition,
                    )
                )
                _update_role_entity_sets(
                    stage2_train_entities
                    if partition == "train"
                    else stage2_validation_entities,
                    frame,
                    task,
                    mask,
                )
            overlap_checks.append(
                {
                    "check": "stage2_train_validation_system_overlap",
                    "task_id": task.task_id,
                    "strategy": "system_holdout",
                    "repeat": "",
                    "passed": True,
                    "overlap_count": 0,
                }
            )
            task_catalog_rows.append(
                {
                    "stage": 2,
                    "task_id": task.task_id,
                    "source_file": task.source_file,
                    "target_columns": ";".join(task.target_columns),
                    "identity_columns": ";".join(task.identity_columns),
                    "system_type": task.system_type,
                    "raw_rows": profile.raw_rows,
                    "rows": profile.rows,
                    "unique_systems": profile.system_count,
                    "tier": "physics_guided",
                    "test_systems": 0,
                    "reserved_systems": 0,
                    "development_systems": profile.system_count,
                    "strategies": "system_holdout",
                    "repeats": 1,
                    "strategy_units": json.dumps(
                        {"system_holdout": "system_id"},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )

        stage2_assignment_rows = [
            {
                "system_type": system_type,
                "system_id": system_id_value,
                "system_key": system_key,
                "partition": partition,
            }
            for (system_type, system_id_value), (
                system_key,
                partition,
            ) in sorted(stage2_groups.items())
        ]
        pd.DataFrame(
            stage2_assignment_rows,
            columns=["system_type", "system_id", "system_key", "partition"],
        ).to_csv(
            stage2_root / "group_assignments.csv",
            index=False,
            lineterminator="\n",
        )

        test_group_rows = []
        for entry in sorted(
            test_registry.values(),
            key=lambda row: (str(row["system_type"]), str(row["system_id"])),
        ):
            source_tasks = entry["source_tasks"]
            test_group_rows.append(
                {
                    "system_type": entry["system_type"],
                    "system_id": entry["system_id"],
                    "system_key": entry["system_key"],
                    "source_tasks": ";".join(sorted(source_tasks))
                    if isinstance(source_tasks, set)
                    else "",
                }
            )
        pd.DataFrame(
            test_group_rows,
            columns=["system_type", "system_id", "system_key", "source_tasks"],
        ).to_csv(
            stage3_root / "test_groups.csv",
            index=False,
            lineterminator="\n",
        )

        stage3_rows_path = stage3_root / "rows.csv"
        random_assignment_path = stage3_root / "random_fold_assignments.csv"
        for task in stage3_tasks:
            frame, raw_rows = prepare_task_frame(
                final_root,
                task,
                checksums[task.source_file],
            )
            profile = stage3_profiles[task.task_id]
            if raw_rows != profile.raw_rows or len(frame) != profile.rows:
                raise TrainingSplitError(
                    f"Input changed while building task {task.task_id}"
                )
            partitions = _stage3_partitions(
                frame,
                task,
                profile,
                test_registry,
            )
            development_mask = partitions.eq("development")
            if not development_mask.any():
                raise TrainingSplitError(
                    f"No development rows remain for {task.task_id}"
                )
            _append_frame(
                stage3_rows_path,
                _catalog_frame(frame, task, partitions),
            )

            for partition in (
                "development",
                "test",
                "reserved_due_to_cross_task_test",
            ):
                mask = partitions.eq(partition)
                if not mask.any():
                    continue
                label_rows.extend(
                    label_summary_rows(
                        frame,
                        task,
                        mask,
                        stage=3,
                        strategy="fixed_test",
                        partition=partition,
                    )
                )
            test_mask = partitions.eq("test")
            if test_mask.any():
                _update_role_entity_sets(
                    stage3_test_entities,
                    frame,
                    task,
                    test_mask,
                )

            repeats = 1 if profile.tier == "large" else 5
            strategies = ["random", *GROUP_STRATEGY_COLUMNS[task.system_type]]
            development = frame.loc[development_mask]
            if len(development) < 5:
                raise TrainingSplitError(
                    f"Fewer than five development rows for {task.task_id}"
                )

            random_unit = (
                development["_system_id"]
                if task.system_type in SINGLE_SYSTEM_TYPES
                else development["_row_id"]
            )
            for repeat_index in range(repeats):
                if task.system_type in SINGLE_SYSTEM_TYPES:
                    folds = random_unit.map(
                        lambda unit_id: single_random_assignments[
                            (
                                task.system_type,
                                str(unit_id),
                                repeat_index,
                            )
                        ]
                    )
                else:
                    random_fold_map = balanced_unit_folds(
                        random_unit.astype(str),
                        seed=seed,
                        namespace=f"stage3_random:{task.task_id}",
                        repeat=repeat_index,
                    )
                    folds = random_unit.map(
                        lambda unit_id: random_fold_map[str(unit_id)]
                    )
                if set(folds) != set(range(5)):
                    raise TrainingSplitError(
                        f"Random five-fold split has empty folds for "
                        f"{task.task_id}, repeat {repeat_index}"
                    )
                random_rows = pd.DataFrame(
                    {
                        "task_id": task.task_id,
                        "row_id": development["_row_id"],
                        "repeat": repeat_index,
                        "fold": folds,
                    }
                )
                _append_frame(random_assignment_path, random_rows)
                for fold_index in range(5):
                    full_mask = pd.Series(False, index=frame.index)
                    full_mask.loc[development.index] = folds.eq(fold_index)
                    label_rows.extend(
                        label_summary_rows(
                            frame,
                            task,
                            full_mask,
                            stage=3,
                            strategy="random",
                            repeat=repeat_index,
                            fold=fold_index,
                            partition="validation",
                        )
                    )
                overlap_checks.append(
                    {
                        "check": "stage3_random_row_assignment_complete",
                        "task_id": task.task_id,
                        "strategy": "random",
                        "repeat": repeat_index,
                        "passed": int(folds.notna().sum()) == len(development),
                        "overlap_count": 0,
                    }
                )

            for strategy, columns in GROUP_STRATEGY_COLUMNS[
                task.system_type
            ].items():
                group_values = _group_series(
                    development,
                    strategy,
                    columns,
                )
                if group_values.nunique() < 5:
                    raise TrainingSplitError(
                        f"Fewer than five {strategy} groups for {task.task_id}"
                )
                for repeat_index in range(repeats):
                    folds = group_values.map(
                        lambda unit_id: stage3_group_assignments[
                            (
                                strategy,
                                str(unit_id),
                                repeat_index,
                            )
                        ]
                    )
                    if set(folds) != set(range(5)):
                        raise TrainingSplitError(
                            f"{strategy} five-fold split has empty folds for "
                            f"{task.task_id}, repeat {repeat_index}"
                        )
                    for unit_id, fold_index in (
                        pd.DataFrame(
                            {"group_id": group_values, "fold": folds}
                        )
                        .drop_duplicates()
                        .itertuples(index=False, name=None)
                    ):
                        assignment_key = (
                            strategy,
                            str(unit_id),
                            repeat_index,
                        )
                        previous = stage3_group_assignments.setdefault(
                            assignment_key,
                            int(fold_index),
                        )
                        if previous != int(fold_index):
                            raise TrainingSplitError(
                                f"Inconsistent cross-task fold for "
                                f"{assignment_key}"
                            )
                    for fold_index in range(5):
                        full_mask = pd.Series(False, index=frame.index)
                        full_mask.loc[development.index] = folds.eq(fold_index)
                        label_rows.extend(
                            label_summary_rows(
                                frame,
                                task,
                                full_mask,
                                stage=3,
                                strategy=strategy,
                                repeat=repeat_index,
                                fold=fold_index,
                                partition="validation",
                            )
                        )
                    overlap_checks.append(
                        {
                            "check": "stage3_group_cross_fold_overlap",
                            "task_id": task.task_id,
                            "strategy": strategy,
                            "repeat": repeat_index,
                            "passed": True,
                            "overlap_count": 0,
                        }
                    )

            if profile.tier == "small":
                for system_id_value, system_key in (
                    development[["_system_id", "_system_key"]]
                    .drop_duplicates()
                    .sort_values("_system_id", kind="stable")
                    .itertuples(index=False, name=None)
                ):
                    stage3_loo_rows.append(
                        {
                            "task_id": task.task_id,
                            "system_type": task.system_type,
                            "system_id": system_id_value,
                            "system_key": system_key,
                            "loo_fold": system_id_value,
                        }
                    )

            test_systems = int(frame.loc[test_mask, "_system_id"].nunique())
            reserved_systems = int(
                frame.loc[
                    partitions.eq("reserved_due_to_cross_task_test"),
                    "_system_id",
                ].nunique()
            )
            development_systems = int(
                development["_system_id"].nunique()
            )
            if profile.tier == "large":
                test_development_overlap = set(
                    frame.loc[test_mask, "_system_id"]
                ) & set(development["_system_id"])
                if test_development_overlap:
                    raise TrainingSplitError(
                        f"Test/development overlap in {task.task_id}"
                    )
            overlap_checks.append(
                {
                    "check": "stage3_test_development_system_overlap",
                    "task_id": task.task_id,
                    "strategy": "fixed_test",
                    "repeat": "",
                    "passed": True,
                    "overlap_count": 0,
                }
            )
            task_catalog_rows.append(
                {
                    "stage": 3,
                    "task_id": task.task_id,
                    "source_file": task.source_file,
                    "target_columns": ";".join(task.target_columns),
                    "identity_columns": ";".join(task.identity_columns),
                    "system_type": task.system_type,
                    "raw_rows": profile.raw_rows,
                    "rows": profile.rows,
                    "unique_systems": profile.system_count,
                    "tier": profile.tier,
                    "test_systems": test_systems,
                    "reserved_systems": reserved_systems,
                    "development_systems": development_systems,
                    "strategies": ";".join(strategies),
                    "repeats": repeats,
                    "strategy_units": json.dumps(
                        _task_strategy_units(task),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )

        if not random_assignment_path.exists():
            pd.DataFrame(
                columns=["task_id", "row_id", "repeat", "fold"]
            ).to_csv(
                random_assignment_path,
                index=False,
                lineterminator="\n",
            )
        group_assignment_rows = [
            {
                "strategy": strategy,
                "group_id": unit_id,
                "repeat": repeat_index,
                "fold": fold_index,
            }
            for (
                strategy,
                unit_id,
                repeat_index,
            ), fold_index in sorted(stage3_group_assignments.items())
        ]
        pd.DataFrame(
            group_assignment_rows,
            columns=["strategy", "group_id", "repeat", "fold"],
        ).to_csv(
            stage3_root / "group_fold_assignments.csv",
            index=False,
            lineterminator="\n",
        )
        pd.DataFrame(
            stage3_loo_rows,
            columns=[
                "task_id",
                "system_type",
                "system_id",
                "system_key",
                "loo_fold",
            ],
        ).to_csv(
            stage3_root / "loo_groups.csv",
            index=False,
            lineterminator="\n",
        )

        task_catalog = pd.DataFrame(task_catalog_rows).sort_values(
            ["stage", "task_id"],
            kind="stable",
        ).reset_index(drop=True)
        task_catalog.to_csv(
            staged_root / "task_catalog.csv",
            index=False,
            lineterminator="\n",
        )
        pd.DataFrame(
            label_rows,
            columns=LABEL_SUMMARY_COLUMNS,
        ).to_csv(
            audit_root / "label_and_fold_summary.csv",
            index=False,
            lineterminator="\n",
        )
        pd.DataFrame(
            overlap_checks,
            columns=[
                "check",
                "task_id",
                "strategy",
                "repeat",
                "passed",
                "overlap_count",
            ],
        ).to_csv(
            audit_root / "overlap_checks.csv",
            index=False,
            lineterminator="\n",
        )
        overlap_rows = _overlap_audit_rows(
            output_root,
            stage2_train_entities,
            stage2_validation_entities,
            stage3_test_entities,
        )
        pd.DataFrame(overlap_rows).to_csv(
            audit_root / "holdout_overlap.csv",
            index=False,
            lineterminator="\n",
        )

        stage3_tier_counts = {
            tier: int(
                sum(
                    profile.tier == tier
                    for profile in stage3_profiles.values()
                )
            )
            for tier in ("large", "medium", "small")
        }
        validation_payload = {
            "passed": bool(
                all(bool(row["passed"]) for row in overlap_checks)
            ),
            "stage2_tasks": len(stage2_tasks),
            "stage3_tasks": len(stage3_tasks),
            "stage3_tier_counts": stage3_tier_counts,
            "stage2_rows": int(
                sum(
                    row["rows"]
                    for row in task_catalog_rows
                    if row["stage"] == 2
                )
            ),
            "stage3_rows": int(
                sum(
                    row["rows"]
                    for row in task_catalog_rows
                    if row["stage"] == 3
                )
            ),
        }
        if not validation_payload["passed"]:
            raise TrainingSplitError("Internal split validation failed")
        write_json(audit_root / "validation.json", validation_payload)

        replace_directory(stage2_root, output_root / "stage2")
        replace_directory(stage3_root, output_root / "stage3")
        replace_directory(audit_root, output_root / "audit")
        task_catalog_destination = output_root / "task_catalog.csv"
        task_catalog_destination.unlink(missing_ok=True)
        (staged_root / "task_catalog.csv").rename(task_catalog_destination)

    output_files = [
        output_root / "task_catalog.csv",
        *sorted((output_root / "stage2").glob("*.csv")),
        *sorted((output_root / "stage3").glob("*.csv")),
        *sorted((output_root / "audit").glob("*")),
    ]
    update_manifest(
        output_root,
        "supervised_splits",
        {
            "final_root": str(final_root),
            "seed": seed,
            "pretrain_cap": pretrain_cap,
            "input_checksums": checksums,
            "stage2_tasks": len(stage2_tasks),
            "stage3_tasks": len(stage3_tasks),
            "stage3_tier_counts": {
                tier: int((task_catalog["tier"] == tier).sum())
                for tier in ("large", "medium", "small")
            },
            "output_checksums": {
                path.relative_to(output_root).as_posix(): file_sha256(path)
                for path in output_files
                if path.is_file()
            },
        },
    )
    return task_catalog


def build_training_splits(
    final_root: Path,
    output_root: Path,
    *,
    seed: int = 42,
    pretrain_cap: int = 500_000,
) -> pd.DataFrame:
    """Build readable Stage-2 and Stage-3 task datasets."""
    _ = pretrain_cap  # Kept for CLI compatibility.
    final_root = Path(final_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _migrate_pubchem_cache(output_root)

    tasks = discover_tasks(final_root)
    paths = final_csv_paths(final_root)
    checksums = {
        path.relative_to(final_root).as_posix(): file_sha256(path)
        for path in paths
    }
    stage2_tasks = [task for task in tasks if task.stage == 2]
    stage3_tasks = [task for task in tasks if task.stage == 3]

    stage3_profiles: dict[str, TaskProfile] = {}
    systems_by_task: dict[str, pd.DataFrame] = {}
    for task in stage3_tasks:
        frame, raw_rows = prepare_task_frame(
            final_root,
            task,
            checksums[task.source_file],
        )
        stage3_profiles[task.task_id] = _profile_task(frame, raw_rows)
        systems_by_task[task.task_id] = _system_table(frame)

    test_registry = plan_joint_test_registry(
        stage3_tasks,
        stage3_profiles,
        systems_by_task,
        seed=seed,
    )
    repeats_by_task = {
        task.task_id: (
            1 if stage3_profiles[task.task_id].tier == "large" else 5
        )
        for task in stage3_tasks
    }

    groups_by_strategy: dict[str, dict[str, set[str]]] = {}
    single_random_groups: dict[str, dict[str, set[str]]] = {}
    for task in stage3_tasks:
        frame, raw_rows = prepare_task_frame(
            final_root,
            task,
            checksums[task.source_file],
        )
        profile = stage3_profiles[task.task_id]
        if raw_rows != profile.raw_rows or len(frame) != profile.rows:
            raise TrainingSplitError(
                f"Input changed while planning folds for {task.task_id}"
            )
        partitions = _stage3_partitions(
            frame,
            task,
            profile,
            test_registry,
        )
        development = frame.loc[partitions.eq("development")]
        if task.system_type in SINGLE_SYSTEM_TYPES:
            single_random_groups.setdefault(task.system_type, {})[
                task.task_id
            ] = set(development["_system_id"].astype(str))
        for strategy, columns in GROUP_STRATEGY_COLUMNS[
            task.system_type
        ].items():
            groups_by_strategy.setdefault(strategy, {})[
                task.task_id
            ] = set(
                _group_series(
                    development,
                    strategy,
                    columns,
                ).astype(str)
            )

    stage3_group_assignments: dict[tuple[str, str, int], int] = {}
    for strategy, task_groups in groups_by_strategy.items():
        planned = plan_shared_group_folds(
            task_groups,
            repeats_by_task,
            seed=seed,
            namespace=f"stage3_group:{strategy}",
        )
        stage3_group_assignments.update(
            {
                (strategy, group_identifier, repeat): fold
                for (group_identifier, repeat), fold in planned.items()
            }
        )

    single_random_assignments: dict[tuple[str, str, int], int] = {}
    for system_type, task_groups in single_random_groups.items():
        planned = plan_shared_group_folds(
            task_groups,
            repeats_by_task,
            seed=seed,
            namespace=f"stage3_random_single:{system_type}",
        )
        single_random_assignments.update(
            {
                (system_type, group_identifier, repeat): fold
                for (group_identifier, repeat), fold in planned.items()
            }
        )

    task_catalog_rows: list[dict[str, object]] = []
    stage2_groups: dict[tuple[str, str], str] = {}
    with tempfile.TemporaryDirectory(dir=output_root.parent) as temporary_dir:
        staged_root = Path(temporary_dir) / "training_splits"
        stage2_root = staged_root / "stage2"
        stage3_root = staged_root / "stage3"
        stage2_root.mkdir(parents=True)
        stage3_root.mkdir()

        for task in stage2_tasks:
            frame, raw_rows = prepare_task_frame(
                final_root,
                task,
                checksums[task.source_file],
            )
            profile = _profile_task(frame, raw_rows)
            partitions = frame["_system_id"].map(
                lambda system_id_value: (
                    "validation"
                    if stable_fraction(
                        seed,
                        "stage2_validation",
                        task.system_type,
                        system_id_value,
                    )
                    < 0.1
                    else "train"
                )
            )
            if not {"train", "validation"}.issubset(set(partitions)):
                raise TrainingSplitError(
                    f"Stage-2 split is empty for {task.task_id}: "
                    f"{sorted(set(partitions))}"
                )
            assignments = pd.DataFrame(
                {
                    "system_id": frame["_system_id"],
                    "partition": partitions,
                }
            ).drop_duplicates()
            for system_id_value, partition in assignments.itertuples(
                index=False,
                name=None,
            ):
                key = (task.system_type, str(system_id_value))
                value = str(partition)
                previous = stage2_groups.setdefault(key, value)
                if previous != value:
                    raise TrainingSplitError(
                        f"Inconsistent shared Stage-2 assignment for {key}"
                    )

            task_root = stage2_root / Path(task.source_file).stem
            materialized = _materialized_task_frame(frame, task)
            write_dataframe(
                task_root / "train.csv",
                materialized.loc[
                    partitions.eq("train").to_numpy()
                ].reset_index(drop=True),
            )
            write_dataframe(
                task_root / "valid.csv",
                materialized.loc[
                    partitions.eq("validation").to_numpy()
                ].reset_index(drop=True),
            )
            task_catalog_rows.append(
                {
                    "stage": 2,
                    "task_id": task.task_id,
                    "source_file": task.source_file,
                    "target_columns": ";".join(task.target_columns),
                    "identity_columns": ";".join(task.identity_columns),
                    "system_type": task.system_type,
                    "raw_rows": profile.raw_rows,
                    "rows": profile.rows,
                    "unique_systems": profile.system_count,
                    "tier": "physics_guided",
                    "test_systems": 0,
                    "reserved_systems": 0,
                    "development_systems": profile.system_count,
                    "strategies": "system_holdout",
                    "repeats": 1,
                    "strategy_units": json.dumps(
                        {"system_holdout": "system_id"},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )

        for task in stage3_tasks:
            frame, raw_rows = prepare_task_frame(
                final_root,
                task,
                checksums[task.source_file],
            )
            profile = stage3_profiles[task.task_id]
            if raw_rows != profile.raw_rows or len(frame) != profile.rows:
                raise TrainingSplitError(
                    f"Input changed while building task {task.task_id}"
                )
            partitions = _stage3_partitions(
                frame,
                task,
                profile,
                test_registry,
            )
            development_mask = partitions.eq("development")
            if not development_mask.any():
                raise TrainingSplitError(
                    f"No development rows remain for {task.task_id}"
                )
            development = frame.loc[development_mask]
            if len(development) < 5:
                raise TrainingSplitError(
                    f"Fewer than five development rows for {task.task_id}"
                )
            task_root = stage3_root / task.task_id
            task_root.mkdir(parents=True)

            test_mask = partitions.eq("test")
            if profile.tier == "large":
                write_dataframe(
                    task_root / "test.csv",
                    _materialized_task_frame(frame.loc[test_mask], task),
                )
            reserved_mask = partitions.eq(
                "reserved_due_to_cross_task_test"
            )
            write_dataframe(
                task_root / "reserved_summary.csv",
                _reserved_summary(frame, task, reserved_mask),
            )
            if profile.tier == "small":
                write_dataframe(
                    task_root / "loo_manifest.csv",
                    _loo_manifest(development, task),
                )

            repeats = 1 if profile.tier == "large" else 5
            strategies = ["random", *GROUP_STRATEGY_COLUMNS[task.system_type]]
            random_unit = (
                development["_system_id"]
                if task.system_type in SINGLE_SYSTEM_TYPES
                else development["_row_id"]
            )
            for repeat_index in range(repeats):
                if task.system_type in SINGLE_SYSTEM_TYPES:
                    folds = random_unit.map(
                        lambda unit_id: single_random_assignments[
                            (
                                task.system_type,
                                str(unit_id),
                                repeat_index,
                            )
                        ]
                    )
                else:
                    random_fold_map = balanced_unit_folds(
                        random_unit.astype(str),
                        seed=seed,
                        namespace=f"stage3_random:{task.task_id}",
                        repeat=repeat_index,
                    )
                    folds = random_unit.map(
                        lambda unit_id: random_fold_map[str(unit_id)]
                    )
                if set(folds) != set(range(5)):
                    raise TrainingSplitError(
                        f"Random five-fold split has empty folds for "
                        f"{task.task_id}, repeat {repeat_index}"
                    )
                strategy_root = (
                    task_root / STRATEGY_DIRECTORY_NAMES["random"]
                )
                if repeats > 1:
                    strategy_root = strategy_root / f"cv{repeat_index + 1}"
                for fold_index in range(5):
                    write_dataframe(
                        strategy_root / f"fold{fold_index + 1}.csv",
                        _materialized_task_frame(
                            development.loc[folds.eq(fold_index)],
                            task,
                        ),
                    )

            for strategy, columns in GROUP_STRATEGY_COLUMNS[
                task.system_type
            ].items():
                group_values = _group_series(
                    development,
                    strategy,
                    columns,
                )
                if group_values.nunique() < 5:
                    raise TrainingSplitError(
                        f"Fewer than five {strategy} groups for {task.task_id}"
                    )
                for repeat_index in range(repeats):
                    folds = group_values.map(
                        lambda unit_id: stage3_group_assignments[
                            (
                                strategy,
                                str(unit_id),
                                repeat_index,
                            )
                        ]
                    )
                    if set(folds) != set(range(5)):
                        raise TrainingSplitError(
                            f"{strategy} five-fold split has empty folds for "
                            f"{task.task_id}, repeat {repeat_index}"
                        )
                    strategy_root = (
                        task_root / STRATEGY_DIRECTORY_NAMES[strategy]
                    )
                    if repeats > 1:
                        strategy_root = (
                            strategy_root / f"cv{repeat_index + 1}"
                        )
                    for fold_index in range(5):
                        write_dataframe(
                            strategy_root / f"fold{fold_index + 1}.csv",
                            _materialized_task_frame(
                                development.loc[folds.eq(fold_index)],
                                task,
                            ),
                        )

            test_systems = int(
                frame.loc[test_mask, "_system_id"].nunique()
            )
            reserved_systems = int(
                frame.loc[reserved_mask, "_system_id"].nunique()
            )
            development_systems = int(
                development["_system_id"].nunique()
            )
            if profile.tier == "large":
                target = max(1, int(round(profile.system_count * 0.1)))
                if test_systems < target:
                    raise TrainingSplitError(
                        f"Test target not met for {task.task_id}: "
                        f"{test_systems} < {target}"
                    )
                test_development_overlap = set(
                    frame.loc[test_mask, "_system_id"]
                ) & set(development["_system_id"])
                if test_development_overlap:
                    raise TrainingSplitError(
                        f"Test/development overlap in {task.task_id}"
                    )
            task_catalog_rows.append(
                {
                    "stage": 3,
                    "task_id": task.task_id,
                    "source_file": task.source_file,
                    "target_columns": ";".join(task.target_columns),
                    "identity_columns": ";".join(task.identity_columns),
                    "system_type": task.system_type,
                    "raw_rows": profile.raw_rows,
                    "rows": profile.rows,
                    "unique_systems": profile.system_count,
                    "tier": profile.tier,
                    "test_systems": test_systems,
                    "reserved_systems": reserved_systems,
                    "development_systems": development_systems,
                    "strategies": ";".join(strategies),
                    "repeats": repeats,
                    "strategy_units": json.dumps(
                        _task_strategy_units(task),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )

        replace_directory(stage2_root, output_root / "stage2")
        replace_directory(stage3_root, output_root / "stage3")

    (output_root / "task_catalog.csv").unlink(missing_ok=True)
    (output_root / "manifest.json").unlink(missing_ok=True)
    legacy_audit = output_root / "audit"
    if legacy_audit.exists():
        shutil.rmtree(legacy_audit)

    return pd.DataFrame(task_catalog_rows).sort_values(
        ["stage", "task_id"],
        kind="stable",
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "extract-pretrain",
            "augment-pretrain",
            "build-splits",
            "all",
        ),
    )
    parser.add_argument(
        "--final-root",
        type=Path,
        default=DEFAULT_FINAL_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-cap", type=int, default=500_000)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--pubchem-threshold", type=int, default=90)
    parser.add_argument("--pubchem-max-records", type=int, default=100)
    parser.add_argument("--pubchem-rate", type=float, default=4.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.pretrain_cap <= 0:
        raise TrainingSplitError("--pretrain-cap must be positive")
    if not 1 <= args.pubchem_threshold <= 100:
        raise TrainingSplitError("--pubchem-threshold must be between 1 and 100")
    if args.pubchem_max_records <= 0:
        raise TrainingSplitError("--pubchem-max-records must be positive")
    if args.pubchem_rate < 0:
        raise TrainingSplitError("--pubchem-rate cannot be negative")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.command in {"extract-pretrain", "all"}:
        extracted = extract_pretraining_entities(
            args.final_root,
            args.output_root,
            pretrain_cap=args.pretrain_cap,
        )
        print(f"Extracted {len(extracted):,} base pretraining entities")
    if args.command in {"augment-pretrain", "all"}:
        augmented = augment_pretraining_entities(
            args.output_root,
            pretrain_cap=args.pretrain_cap,
            threshold=args.pubchem_threshold,
            max_records=args.pubchem_max_records,
            offline=args.offline,
            rate_limit=args.pubchem_rate,
        )
        print(f"Prepared {len(augmented):,} pretraining entities")
    if args.command in {"build-splits", "all"}:
        catalog = build_training_splits(
            args.final_root,
            args.output_root,
            seed=args.seed,
            pretrain_cap=args.pretrain_cap,
        )
        stage2_count = int(catalog["stage"].eq(2).sum())
        stage3_count = int(catalog["stage"].eq(3).sum())
        print(
            f"Built readable datasets for {stage2_count} stage-2 tasks and "
            f"{stage3_count} stage-3 tasks"
        )


if __name__ == "__main__":
    main()
