"""Build readable three-stage training corpora and deterministic splits.

The script reads only top-level CSV files under ``data/final/experiment`` and
``data/final/simulation``. It never traverses or parses molecular structure
files such as ``.mol`` or ``.mol2``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from http import client as http_client
import json
import os
from pathlib import Path
import signal
import shutil
import sqlite3
import tempfile
import time
from typing import Callable, Iterable, Mapping, Sequence
from urllib import error, parse, request

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import inchi
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINAL_ROOT = PROJECT_ROOT / "data" / "final"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "training_splits"

BUCKETS = ("experiment", "simulation")
ROLE_BY_COLUMN = {
    "cation": "cation",
    "anion": "anion",
    "solute": "solute",
    "solvent": "solvent",
    "SMILES": "molecule",
}
IDENTITY_COLUMNS = tuple(ROLE_BY_COLUMN)
PRETRAIN_ROLES = ("anion", "cation", "molecule")
STAGE1_SOURCE_ROLE_BY_COLUMN = {
    "cation": "cation",
    "anion": "anion",
    "solute": "solute",
    "solvent": "solvent",
    "SMILES": "simulation_mol",
}
STAGE1_NEUTRAL_SOURCE_ROLES = ("simulation_mol", "solute", "solvent")
STAGE1_SOURCE_ROLES = ("anion", "cation", *STAGE1_NEUTRAL_SOURCE_ROLES)
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
FOLD_BALANCE_COLUMNS = [
    "task_id",
    "strategy",
    "cv",
    "fold",
    "row_count",
    "group_count",
    "total_rows",
    "mean_rows",
    "largest_group_rows",
    "max_to_min",
    "theoretical_lower_bound",
    "unavoidable_group_dominance",
]
ORIGIN_ORDER = {"dataset": 0, "pubchem": 1, "rule": 2}
DEFAULT_RESONANCE_MAX_STRUCTS = 256
DEFAULT_RESONANCE_MAX_HEAVY_ATOMS = 50
DEFAULT_RESONANCE_MAX_ABS_CHARGE = 2
AUGMENTATION_CACHE_VERSION = 1

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


class TrainingSplitError(RuntimeError):
    """Raised when inputs cannot produce a trustworthy split."""


class IncompletePubChemQuery(TrainingSplitError):
    """Raised when a PubChem request is unavailable or unfinished."""


@dataclass(frozen=True)
class ResonanceEligibility:
    eligible: bool
    heavy_atom_count: int
    formal_charge: int
    reason: str


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


def _group_fold_signature(
    group_values: pd.Series,
    folds: pd.Series,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        sorted(
            tuple(
                sorted(
                    set(
                        group_values.loc[folds.eq(fold)].astype(str)
                    )
                )
            )
            for fold in range(5)
        )
    )


def task_group_kfold_assignments(
    group_values: pd.Series,
    *,
    task_id: str,
    strategy: str,
    repeats: int,
    seed: int,
    max_candidates: int = 256,
) -> list[pd.Series]:
    """Build deterministic, task-local, distinct GroupKFold assignments."""
    groups = group_values.astype(str)
    group_count = int(groups.nunique())
    if group_count < 5:
        raise TrainingSplitError(
            f"Fewer than five {strategy} groups for {task_id}"
        )
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")

    features = np.zeros((len(groups), 1), dtype=np.uint8)
    signatures: set[tuple[tuple[str, ...], ...]] = set()
    assignments: list[pd.Series] = []
    for candidate_index in range(max_candidates):
        digest = stable_id(
            f"seed:{seed}:stage3_group_kfold",
            task_id,
            strategy,
            candidate_index,
        )
        random_state = int(digest[:8], 16)
        splitter = GroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state,
        )
        fold_values = np.full(len(groups), -1, dtype=np.int8)
        for fold, (_, validation_indices) in enumerate(
            splitter.split(features, groups=groups.to_numpy())
        ):
            fold_values[validation_indices] = fold
        folds = pd.Series(
            fold_values,
            index=group_values.index,
            dtype="int64",
        )
        if set(folds) != set(range(5)):
            raise TrainingSplitError(
                f"GroupKFold has empty folds for {task_id}, {strategy}"
            )
        signature = _group_fold_signature(groups, folds)
        if signature in signatures:
            continue
        signatures.add(signature)
        assignments.append(folds)
        if len(assignments) == repeats:
            return assignments

    raise TrainingSplitError(
        f"Unable to build {repeats} distinct GroupKFold partitions for "
        f"{task_id}, {strategy} after {max_candidates} candidates"
    )


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


_FINAL_IDENTITY_VALIDATION_CACHE: set[
    tuple[str, tuple[tuple[str, int, int], ...]]
] = set()


def validate_final_identity_consistency(final_root: Path) -> None:
    final_root = Path(final_root)
    paths = final_csv_paths(final_root)
    signature = (
        str(final_root.resolve()),
        tuple(
            (
                path.relative_to(final_root).as_posix(),
                path.stat().st_mtime_ns,
                path.stat().st_size,
            )
            for path in paths
        ),
    )
    if signature in _FINAL_IDENTITY_VALIDATION_CACHE:
        return

    identities: dict[str, tuple[str, str]] = {}
    for path in paths:
        relative = path.relative_to(final_root).as_posix()
        header = list(pd.read_csv(path, nrows=0).columns)
        columns = [
            column for column in IDENTITY_COLUMNS if column in header
        ]
        if not columns:
            continue
        for chunk in pd.read_csv(
            path,
            usecols=columns,
            chunksize=100_000,
        ):
            for column in columns:
                for value in pd.unique(chunk[column]):
                    if pd.isna(value) or not str(value).strip():
                        raise TrainingSplitError(
                            f"Missing identity in {relative}, column {column}"
                        )
                    try:
                        canonical = canonicalize_smiles(str(value))
                        identity_key = fixed_h_identity_key(canonical)
                    except TrainingSplitError as exc:
                        raise TrainingSplitError(
                            f"{relative}, column {column}: {exc}"
                        ) from exc
                    context = f"{relative}:{column}"
                    previous = identities.setdefault(
                        identity_key,
                        (canonical, context),
                    )
                    if previous[0] != canonical:
                        raise TrainingSplitError(
                            "Equivalent chemical identity has multiple SMILES "
                            f"representations: {previous[0]!r} "
                            f"({previous[1]}) and {canonical!r} ({context}). "
                            "Rebuild data/merged with scripts/merge_data.py, "
                            "then rebuild data/final."
                        )
    _FINAL_IDENTITY_VALIDATION_CACHE.add(signature)


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


@lru_cache(maxsize=None)
def canonicalize_smiles(smiles: str) -> str:
    text = str(smiles).strip()
    if not text:
        raise TrainingSplitError("Empty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise TrainingSplitError(f"Invalid SMILES: {text}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=None)
def fixed_h_identity_key(smiles: str) -> str:
    canonical = canonicalize_smiles(smiles)
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(canonical)
        key = (
            inchi.MolToInchiKey(molecule, options="/FixedH")
            if molecule is not None
            else ""
        )
    if not key:
        raise TrainingSplitError(
            f"Unable to generate Fixed-H InChIKey: {canonical}"
        )
    return key


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
    rows = [_entity_output_record(record) for record in records]
    return pd.DataFrame(rows, columns=PRETRAIN_ENTITY_COLUMNS).sort_values(
        "SMILES",
        kind="stable",
    ).reset_index(drop=True)


def _entity_output_record(
    record: Mapping[str, object],
) -> dict[str, object]:
    return {
        "SMILES": str(record["SMILES"]),
        "formal_charge": int(record["formal_charge"]),
        "origin_list": _joined(
            record.get("origin_list", set()),
            origin=True,
        ),
        "seed_smiles_list": _joined(record.get("seed_smiles_list", set())),
        "rule_list": _joined(record.get("rule_list", set())),
        "pubchem_cid_list": _joined(
            record.get("pubchem_cid_list", set())
        ),
        "mol_id_list": _joined(record.get("mol_id_list", set())),
    }


def _normalized_pretrain_role(role: str, charge: int) -> str:
    allowed_roles = set(STAGE1_SOURCE_ROLES) | set(PRETRAIN_ROLES)
    if role not in allowed_roles:
        raise TrainingSplitError(f"Unknown Stage-1 entity role: {role}")
    if charge > 0:
        return "cation"
    if charge < 0:
        return "anion"
    if role in {"cation", "anion"}:
        raise TrainingSplitError(
            f"Neutral entity cannot be assigned to {role}"
        )
    return role


def _normalize_base_entity_roles(frame: pd.DataFrame) -> pd.DataFrame:
    records: dict[tuple[str, str], dict[str, object]] = {}
    for row in frame.itertuples(index=False):
        canonical = canonicalize_smiles(str(row.SMILES))
        charge = formal_charge(canonical)
        try:
            stored_charge = float(row.formal_charge)
        except (TypeError, ValueError) as exc:
            raise TrainingSplitError(
                f"Invalid Stage-1 formal_charge for {canonical}: "
                f"{row.formal_charge!r}"
            ) from exc
        if not np.isfinite(stored_charge) or not stored_charge.is_integer():
            raise TrainingSplitError(
                f"Invalid Stage-1 formal_charge for {canonical}: "
                f"{row.formal_charge!r}"
            )
        if int(stored_charge) != charge:
            raise TrainingSplitError(
                f"Stage-1 formal_charge mismatch for {canonical}: "
                f"stored {int(stored_charge)}, calculated {charge}"
            )

        role = _normalized_pretrain_role(str(row.role), charge)
        key = (role, fixed_h_identity_key(canonical))
        record = records.setdefault(
            key,
            {
                "role": role,
                "SMILES": canonical,
                "formal_charge": charge,
                "origin_list": set(),
                "seed_smiles_list": set(),
                "rule_list": set(),
                "pubchem_cid_list": set(),
                "mol_id_list": set(),
            },
        )
        if canonical < str(record["SMILES"]):
            record["SMILES"] = canonical
        for column in (
            "origin_list",
            "seed_smiles_list",
            "rule_list",
            "pubchem_cid_list",
            "mol_id_list",
        ):
            values = record[column]
            if isinstance(values, set):
                values.update(_split_joined(getattr(row, column)))

    return pd.DataFrame(
        [
            {
                "role": record["role"],
                **_entity_output_record(record),
            }
            for record in sorted(
                records.values(),
                key=lambda item: (
                    str(item["role"]),
                    str(item["SMILES"]),
                ),
            )
        ],
        columns=["role", *PRETRAIN_ENTITY_COLUMNS],
    )


def _combined_entity_frame(stage1_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for role in PRETRAIN_ROLES:
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
        origins = frame["origin_list"].map(_split_joined)
        has_augmentation_provenance = (
            origins.map(lambda values: values != {"dataset"}).any()
            or frame[
                ["seed_smiles_list", "rule_list", "pubchem_cid_list"]
            ]
            .astype(str)
            .apply(lambda column: column.str.strip().ne(""))
            .any(axis=None)
        )
        if has_augmentation_provenance:
            raise TrainingSplitError(
                "Stage-1 root entity files contain augmentation rows or "
                "provenance; run extract-pretrain before augment-pretrain"
            )
        frame.insert(0, "role", role)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    normalized = _normalize_base_entity_roles(combined)
    if len(normalized) != len(combined):
        raise TrainingSplitError(
            "Stage-1 root entity files contain duplicate chemical identities; "
            "run extract-pretrain before augment-pretrain"
        )
    normalized_keys = list(
        normalized[["role", "SMILES"]].itertuples(index=False, name=None)
    )
    input_keys = list(
        combined[["role", "SMILES"]]
        .sort_values(["role", "SMILES"], kind="stable")
        .itertuples(index=False, name=None)
    )
    if normalized_keys != input_keys:
        raise TrainingSplitError(
            "Stage-1 root entity roles or SMILES are not normalized; "
            "run extract-pretrain before augment-pretrain"
        )
    return normalized


def extract_pretraining_entities(
    final_root: Path,
    output_root: Path,
    *,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Extract readable role-specific entities and observed ionic-liquid pairs."""
    final_root = Path(final_root)
    output_root = Path(output_root)
    paths = final_csv_paths(final_root)
    validate_final_identity_consistency(final_root)
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
                    role = STAGE1_SOURCE_ROLE_BY_COLUMN[column]
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
                            charge = formal_charge(smiles)
                            normalized_role = _normalized_pretrain_role(
                                role,
                                charge,
                            )
                            identity_key = fixed_h_identity_key(smiles)
                            key = (normalized_role, identity_key)
                            record = entities.setdefault(
                                key,
                                {
                                    "role": normalized_role,
                                    "SMILES": smiles,
                                    "formal_charge": charge,
                                    "origin_list": {"dataset"},
                                    "seed_smiles_list": set(),
                                    "rule_list": set(),
                                    "pubchem_cid_list": set(),
                                    "mol_id_list": set(),
                                },
                            )
                            if smiles < str(record["SMILES"]):
                                record["SMILES"] = smiles
                                record["formal_charge"] = formal_charge(smiles)
                            if not pd.isna(mol_id) and str(mol_id).strip():
                                mol_ids = record["mol_id_list"]
                                if isinstance(mol_ids, set):
                                    mol_ids.add(str(mol_id).strip())
                row_offset += len(chunk)

        source_frames: dict[str, pd.DataFrame] = {}
        for role in STAGE1_SOURCE_ROLES:
            role_frame = _entity_output_frame(
                record
                for (record_role, _), record in entities.items()
                if record_role == role
            )
            source_frames[role] = role_frame
            role_frame.to_csv(
                staged_stage1 / f"{role}.csv",
                index=False,
                lineterminator="\n",
            )

        neutral_records = _normalize_base_entity_roles(
            pd.DataFrame(
                [
                    {
                        "role": "molecule",
                        **_entity_output_record(record),
                    }
                    for (record_role, _), record in entities.items()
                    if record_role in STAGE1_NEUTRAL_SOURCE_ROLES
                ],
                columns=["role", *PRETRAIN_ENTITY_COLUMNS],
            )
        )
        molecule_frame = neutral_records.loc[:, PRETRAIN_ENTITY_COLUMNS]
        molecule_frame.to_csv(
            staged_stage1 / "molecule.csv",
            index=False,
            lineterminator="\n",
        )

        combined_rows: list[pd.DataFrame] = []
        for role in ("anion", "cation"):
            with_role = source_frames[role].copy()
            with_role.insert(0, "role", role)
            combined_rows.append(with_role)
        combined_rows.append(neutral_records)
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


def resonance_eligibility(
    smiles: str,
    max_heavy_atoms: int,
    max_abs_charge: int,
) -> ResonanceEligibility:
    canonical = canonicalize_smiles(smiles)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:  # pragma: no cover - protected by canonicalize_smiles
        raise TrainingSplitError(f"Invalid SMILES: {smiles}")
    charge = formal_charge(canonical)
    reasons: list[str] = []
    if len(Chem.GetMolFrags(molecule)) != 1:
        reasons.append("multiple_fragments")
    if molecule.GetNumHeavyAtoms() > max_heavy_atoms:
        reasons.append("heavy_atom_limit")
    if abs(charge) > max_abs_charge:
        reasons.append("charge_limit")
    return ResonanceEligibility(
        eligible=not reasons,
        heavy_atom_count=molecule.GetNumHeavyAtoms(),
        formal_charge=charge,
        reason=";".join(reasons),
    )


def _generate_rule_candidates_with_audit(
    smiles: str,
    role: str | None = None,
    *,
    resonance_max_structs: int = DEFAULT_RESONANCE_MAX_STRUCTS,
    resonance_max_heavy_atoms: int = DEFAULT_RESONANCE_MAX_HEAVY_ATOMS,
    resonance_max_abs_charge: int = DEFAULT_RESONANCE_MAX_ABS_CHARGE,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Generate a finite, charge-preserving catalog of chemistry-rule candidates."""
    if resonance_max_structs <= 0:
        raise ValueError("resonance_max_structs must be positive")
    canonical = canonicalize_smiles(smiles)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:  # pragma: no cover - protected by canonicalize_smiles
        return []
    target_charge = formal_charge(canonical)
    effective_role = _normalized_pretrain_role(
        role or "molecule",
        target_charge,
    )
    candidates: dict[tuple[str, str], dict[str, str]] = {}
    ion_role = target_charge != 0
    eligibility = resonance_eligibility(
        canonical,
        resonance_max_heavy_atoms,
        resonance_max_abs_charge,
    )
    resonance_examined = 0

    if eligibility.eligible:
        seed_identity = fixed_h_identity_key(canonical)
        supplier = Chem.ResonanceMolSupplier(
            molecule,
            0,
            resonance_max_structs,
        )
        for resonance_molecule in supplier:
            resonance_examined += 1
            if resonance_molecule is None:
                continue
            candidate = _sanitized_smiles(resonance_molecule)
            if (
                candidate
                and candidate != canonical
                and formal_charge(candidate) == target_charge
                and fixed_h_identity_key(candidate) == seed_identity
            ):
                candidates[("resonance_equivalent", candidate)] = {
                    "SMILES": candidate,
                    "rule": "resonance_equivalent",
                    "family": "resonance",
                }

    def result() -> tuple[list[dict[str, str]], dict[str, object]]:
        rows = [
            candidates[key]
            for key in sorted(candidates, key=lambda item: (item[0], item[1]))
        ]
        truncated = (
            eligibility.eligible
            and resonance_examined >= resonance_max_structs
        )
        status = (
            "skipped"
            if not eligibility.eligible
            else "truncated"
            if truncated
            else "complete"
        )
        return rows, {
            "heavy_atom_count": eligibility.heavy_atom_count,
            "formal_charge": eligibility.formal_charge,
            "resonance_status": status,
            "resonance_skip_reason": eligibility.reason,
            "resonance_examined": resonance_examined,
            "resonance_generated": sum(
                row["rule"] == "resonance_equivalent"
                for row in rows
            ),
            "resonance_truncated": truncated,
        }

    if target_charge == 0:
        return result()

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

    if effective_role == "cation":
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

    if effective_role == "anion":
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

    return result()


def generate_rule_candidates(
    smiles: str,
    role: str | None = None,
    *,
    resonance_max_structs: int = DEFAULT_RESONANCE_MAX_STRUCTS,
    resonance_max_heavy_atoms: int = DEFAULT_RESONANCE_MAX_HEAVY_ATOMS,
    resonance_max_abs_charge: int = DEFAULT_RESONANCE_MAX_ABS_CHARGE,
) -> list[dict[str, str]]:
    candidates, _audit = _generate_rule_candidates_with_audit(
        smiles,
        role,
        resonance_max_structs=resonance_max_structs,
        resonance_max_heavy_atoms=resonance_max_heavy_atoms,
        resonance_max_abs_charge=resonance_max_abs_charge,
    )
    return candidates


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


def _augmentation_fingerprint(
    base: pd.DataFrame,
    *,
    threshold: int,
    max_records: int,
    resonance_max_structs: int,
    resonance_max_heavy_atoms: int,
    resonance_max_abs_charge: int,
) -> tuple[str, dict[str, int]]:
    config = {
        "cache_version": AUGMENTATION_CACHE_VERSION,
        "pubchem_threshold": threshold,
        "pubchem_max_records": max_records,
        "resonance_max_structs": resonance_max_structs,
        "resonance_max_heavy_atoms": resonance_max_heavy_atoms,
        "resonance_max_abs_charge": resonance_max_abs_charge,
    }
    digest = hashlib.sha256(
        json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for row in base.itertuples(index=False):
        digest.update(
            f"\n{row.role}\t{row.SMILES}\t{int(row.formal_charge)}".encode(
                "utf-8"
            )
        )
    return digest.hexdigest(), config


def _open_augmentation_database(
    path: Path,
    *,
    fingerprint: str,
    config: Mapping[str, int],
) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    stored_version = connection.execute(
        "SELECT value FROM metadata WHERE key = 'cache_version'"
    ).fetchone()
    if stored_version is not None and stored_version[0] != str(
        AUGMENTATION_CACHE_VERSION
    ):
        connection.execute("DROP TABLE IF EXISTS candidates")
        connection.execute("DROP TABLE IF EXISTS seed_status")
        connection.execute("DELETE FROM metadata")

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            seed_entity_id TEXT NOT NULL,
            role TEXT NOT NULL,
            seed_smiles TEXT NOT NULL,
            candidate_smiles TEXT NOT NULL,
            method TEXT NOT NULL,
            pubchem_cid TEXT NOT NULL,
            rule TEXT NOT NULL,
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
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS seed_status (
            seed_entity_id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            seed_smiles TEXT NOT NULL,
            local_status TEXT NOT NULL DEFAULT 'pending',
            pubchem_status TEXT NOT NULL DEFAULT 'pending',
            pubchem_error TEXT NOT NULL DEFAULT '',
            heavy_atom_count INTEGER,
            formal_charge INTEGER,
            resonance_status TEXT NOT NULL DEFAULT 'pending',
            resonance_skip_reason TEXT NOT NULL DEFAULT '',
            resonance_examined INTEGER NOT NULL DEFAULT 0,
            resonance_generated INTEGER NOT NULL DEFAULT 0,
            resonance_truncated INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    stored_fingerprint = connection.execute(
        "SELECT value FROM metadata WHERE key = 'fingerprint'"
    ).fetchone()
    if stored_fingerprint is not None and stored_fingerprint[0] != fingerprint:
        connection.execute("DELETE FROM candidates")
        connection.execute("DELETE FROM seed_status")
    connection.executemany(
        """
        INSERT INTO metadata(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (
            ("cache_version", str(AUGMENTATION_CACHE_VERSION)),
            ("fingerprint", fingerprint),
            (
                "config",
                json.dumps(
                    dict(config),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        ),
    )
    connection.commit()
    return connection


def _replace_seed_candidates(
    connection: sqlite3.Connection,
    *,
    seed_identifier: str,
    role: str,
    seed_smiles: str,
    method: str,
    candidates: Sequence[Mapping[str, object]],
) -> None:
    connection.execute(
        "DELETE FROM candidates WHERE seed_entity_id = ? AND method = ?",
        (seed_identifier, method),
    )
    method_offset = 0 if method == "rule" else 1
    connection.executemany(
        """
        INSERT INTO candidates(
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
            (
                seed_identifier,
                role,
                seed_smiles,
                str(candidate["candidate_smiles"]),
                method,
                str(candidate.get("pubchem_cid", "")),
                str(candidate.get("rule", "")),
                int(candidate["method_rank"]),
                2 * int(candidate["method_rank"]) + method_offset,
            )
            for candidate in candidates
        ),
    )


def _augmentation_audit_frames(
    connection: sqlite3.Connection,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    failures = pd.read_sql_query(
        """
        SELECT role, seed_smiles, pubchem_error AS error
        FROM seed_status
        WHERE pubchem_status = 'failed'
        ORDER BY role, seed_smiles
        """,
        connection,
    )
    resonance = pd.read_sql_query(
        """
        SELECT
            role,
            seed_smiles,
            heavy_atom_count,
            formal_charge,
            resonance_status,
            resonance_skip_reason AS reason,
            resonance_examined,
            resonance_generated,
            resonance_truncated
        FROM seed_status
        WHERE resonance_status IN ('skipped', 'truncated')
        ORDER BY role, seed_smiles
        """,
        connection,
    )
    return failures, resonance


def _write_augmentation_files(
    stage1_root: Path,
    connection: sqlite3.Connection,
    *,
    base: pd.DataFrame,
    config: Mapping[str, int],
    failures: pd.DataFrame,
    resonance_audit: pd.DataFrame,
) -> dict[str, object]:
    connection.execute("DROP TABLE IF EXISTS temp.base_entities")
    connection.execute(
        """
        CREATE TEMP TABLE base_entities (
            role TEXT NOT NULL,
            smiles TEXT NOT NULL,
            PRIMARY KEY(role, smiles)
        )
        """
    )
    connection.executemany(
        "INSERT INTO base_entities(role, smiles) VALUES (?, ?)",
        (
            (str(row.role), canonicalize_smiles(str(row.SMILES)))
            for row in base.itertuples(index=False)
        ),
    )

    base_counts = {
        role: int(base["role"].eq(role).sum())
        for role in PRETRAIN_ROLES
    }
    method_counts = dict(
        connection.execute(
            """
            SELECT method, COUNT(*)
            FROM candidates
            GROUP BY method
            """
        )
    )
    pubchem_status_counts = dict(
        connection.execute(
            """
            SELECT pubchem_status, COUNT(*)
            FROM seed_status
            GROUP BY pubchem_status
            """
        )
    )
    resonance_generated = int(
        connection.execute(
            """
            SELECT COALESCE(SUM(resonance_generated), 0)
            FROM seed_status
            """
        ).fetchone()[0]
    )
    excluded_base_overlaps = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT c.role, c.candidate_smiles
                FROM candidates c
                INNER JOIN base_entities b
                    ON b.role = c.role
                    AND b.smiles = c.candidate_smiles
                GROUP BY c.role, c.candidate_smiles
            )
            """
        ).fetchone()[0]
    )

    role_counts = {role: 0 for role in PRETRAIN_ROLES}
    origin_counts = {"rule_only": 0, "pubchem_only": 0, "both": 0}
    resonance_exported = 0

    with tempfile.TemporaryDirectory(dir=stage1_root) as temporary_dir:
        staged_augmentation = Path(temporary_dir) / "augmentation"
        staged_augmentation.mkdir()
        handles = {
            role: (staged_augmentation / f"{role}.csv").open(
                "w",
                encoding="utf-8",
                newline="",
            )
            for role in PRETRAIN_ROLES
        }
        writers = {
            role: csv.DictWriter(
                handle,
                fieldnames=PRETRAIN_ENTITY_COLUMNS,
                lineterminator="\n",
            )
            for role, handle in handles.items()
        }
        for writer in writers.values():
            writer.writeheader()

        current_key: tuple[str, str] | None = None
        current_record: dict[str, object] | None = None

        def write_current_record() -> None:
            nonlocal resonance_exported
            if current_key is None or current_record is None:
                return
            role = current_key[0]
            origins = current_record["origin_list"]
            rules = current_record["rule_list"]
            if not isinstance(origins, set) or not isinstance(rules, set):
                raise TrainingSplitError(
                    f"Invalid augmentation provenance for {current_key}"
                )
            if origins == {"rule"}:
                origin_class = "rule_only"
            elif origins == {"pubchem"}:
                origin_class = "pubchem_only"
            elif origins == {"rule", "pubchem"}:
                origin_class = "both"
            else:
                raise TrainingSplitError(
                    f"Unknown augmentation origins for {current_key}: "
                    f"{sorted(origins)}"
                )
            writers[role].writerow(_entity_output_record(current_record))
            role_counts[role] += 1
            origin_counts[origin_class] += 1
            if "resonance_equivalent" in rules:
                resonance_exported += 1

        try:
            cursor = connection.execute(
                """
                SELECT
                    c.role,
                    c.candidate_smiles,
                    c.seed_smiles,
                    c.method,
                    c.pubchem_cid,
                    c.rule,
                    s.formal_charge
                FROM candidates c
                INNER JOIN seed_status s
                    ON s.seed_entity_id = c.seed_entity_id
                LEFT JOIN base_entities b
                    ON b.role = c.role
                    AND b.smiles = c.candidate_smiles
                WHERE b.smiles IS NULL
                ORDER BY
                    c.role,
                    c.candidate_smiles,
                    c.seed_smiles,
                    c.method,
                    c.rule,
                    c.pubchem_cid
                """
            )
            for (
                role,
                candidate_smiles,
                seed_smiles,
                method,
                pubchem_cid,
                rule,
                candidate_charge,
            ) in cursor:
                key = (str(role), str(candidate_smiles))
                if key[0] not in PRETRAIN_ROLES:
                    raise TrainingSplitError(
                        f"Unknown augmentation role in candidate cache: {key[0]}"
                    )
                if current_key != key:
                    write_current_record()
                    current_key = key
                    current_record = {
                        "SMILES": key[1],
                        "formal_charge": int(candidate_charge),
                        "origin_list": set(),
                        "seed_smiles_list": set(),
                        "rule_list": set(),
                        "pubchem_cid_list": set(),
                        "mol_id_list": set(),
                    }
                elif int(current_record["formal_charge"]) != int(
                    candidate_charge
                ):
                    raise TrainingSplitError(
                        f"Conflicting formal charges for augmentation {key}"
                    )

                origins = current_record["origin_list"]
                seeds = current_record["seed_smiles_list"]
                rules = current_record["rule_list"]
                pubchem_cids = current_record["pubchem_cid_list"]
                if not all(
                    isinstance(values, set)
                    for values in (origins, seeds, rules, pubchem_cids)
                ):
                    raise TrainingSplitError(
                        f"Invalid augmentation accumulator for {key}"
                    )
                if method == "rule":
                    origins.add("rule")
                elif method == "pubchem_similarity":
                    origins.add("pubchem")
                else:
                    raise TrainingSplitError(
                        f"Unknown augmentation method: {method}"
                    )
                seeds.add(str(seed_smiles))
                if str(rule).strip():
                    rules.add(str(rule).strip())
                if str(pubchem_cid).strip():
                    pubchem_cids.add(str(pubchem_cid).strip())
            write_current_record()
        finally:
            for handle in handles.values():
                handle.close()

        summary: dict[str, object] = {
            "augmentation_entities": sum(role_counts.values()),
            "augmentation_entities_by_origin": origin_counts,
            "augmentation_entities_by_role": role_counts,
            "base_entities": len(base),
            "base_entities_by_role": base_counts,
            "candidate_relation_rows": {
                "pubchem_similarity": int(
                    method_counts.get("pubchem_similarity", 0)
                ),
                "rule": int(method_counts.get("rule", 0)),
            },
            "completion_status": (
                "partial_pubchem" if not failures.empty else "complete"
            ),
            "config": dict(config),
            "excluded_base_overlaps": excluded_base_overlaps,
            "pubchem_failed_seeds": len(failures),
            "pubchem_ok_seeds": int(pubchem_status_counts.get("ok", 0)),
            "pubchem_skipped_seeds": int(
                pubchem_status_counts.get("skipped", 0)
            ),
            "resonance_exported": resonance_exported,
            "resonance_generated": resonance_generated,
            "resonance_skipped_seeds": int(
                resonance_audit["resonance_status"].eq("skipped").sum()
            ),
            "resonance_truncated_seeds": int(
                resonance_audit["resonance_status"].eq("truncated").sum()
            ),
        }
        audit_root = staged_augmentation / "_audit"
        audit_root.mkdir()
        (audit_root / "augmentation_summary.json").write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        failures.to_csv(
            audit_root / "pubchem_failures.csv",
            index=False,
            lineterminator="\n",
        )
        resonance_audit.to_csv(
            audit_root / "resonance_skips.csv",
            index=False,
            lineterminator="\n",
        )
        replace_directory(
            staged_augmentation,
            stage1_root / "augmentation",
        )
    return summary


def augment_pretraining_entities(
    output_root: Path,
    *,
    threshold: int = 90,
    max_records: int = 100,
    offline: bool = False,
    rate_limit: float = 4.0,
    resonance_max_structs: int = DEFAULT_RESONANCE_MAX_STRUCTS,
    resonance_max_heavy_atoms: int = DEFAULT_RESONANCE_MAX_HEAVY_ATOMS,
    resonance_max_abs_charge: int = DEFAULT_RESONANCE_MAX_ABS_CHARGE,
    client: PubChemClient | None = None,
) -> dict[str, object]:
    """Augment entities using PubChem similarity and local chemistry rules."""
    if resonance_max_structs <= 0:
        raise TrainingSplitError("resonance_max_structs must be positive")
    if resonance_max_heavy_atoms <= 0:
        raise TrainingSplitError(
            "resonance_max_heavy_atoms must be positive"
        )
    if resonance_max_abs_charge < 0:
        raise TrainingSplitError(
            "resonance_max_abs_charge cannot be negative"
        )

    output_root = Path(output_root)
    stage1_root = output_root / "stage1"
    base = _combined_entity_frame(stage1_root)

    fingerprint, config = _augmentation_fingerprint(
        base,
        threshold=threshold,
        max_records=max_records,
        resonance_max_structs=resonance_max_structs,
        resonance_max_heavy_atoms=resonance_max_heavy_atoms,
        resonance_max_abs_charge=resonance_max_abs_charge,
    )
    candidate_database = _open_augmentation_database(
        output_root / ".cache" / "stage1_augmentation.sqlite",
        fingerprint=fingerprint,
        config=config,
    )
    owns_client = client is None
    active_client = client or PubChemClient(
        _migrate_pubchem_cache(output_root),
        offline=offline,
        rate_limit=rate_limit,
    )
    previous_sigterm: object | None = None

    def interrupt_for_sigterm(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    try:
        previous_sigterm = signal.signal(
            signal.SIGTERM,
            interrupt_for_sigterm,
        )
    except ValueError:
        previous_sigterm = None

    try:
        total_seeds = len(base)
        for processed, seed in enumerate(
            base.itertuples(index=False),
            start=1,
        ):
            seed_smiles = canonicalize_smiles(str(seed.SMILES))
            role = str(seed.role)
            seed_identifier = entity_id(role, seed_smiles)
            candidate_database.execute(
                """
                INSERT OR IGNORE INTO seed_status(
                    seed_entity_id,
                    role,
                    seed_smiles
                ) VALUES (?, ?, ?)
                """,
                (seed_identifier, role, seed_smiles),
            )
            local_status, pubchem_status = candidate_database.execute(
                """
                SELECT local_status, pubchem_status
                FROM seed_status
                WHERE seed_entity_id = ?
                """,
                (seed_identifier,),
            ).fetchone()

            if fragment_count(seed_smiles) != 1:
                molecule = Chem.MolFromSmiles(seed_smiles)
                heavy_atoms = (
                    molecule.GetNumHeavyAtoms()
                    if molecule is not None
                    else 0
                )
                candidate_database.execute(
                    """
                    UPDATE seed_status
                    SET
                        local_status = 'resonance_skipped',
                        pubchem_status = 'skipped',
                        pubchem_error = '',
                        heavy_atom_count = ?,
                        formal_charge = ?,
                        resonance_status = 'skipped',
                        resonance_skip_reason = 'multiple_fragments'
                    WHERE seed_entity_id = ?
                    """,
                    (
                        heavy_atoms,
                        formal_charge(seed_smiles),
                        seed_identifier,
                    ),
                )
                candidate_database.commit()
                continue

            if local_status not in {"complete", "resonance_skipped"}:
                eligibility = resonance_eligibility(
                    seed_smiles,
                    resonance_max_heavy_atoms,
                    resonance_max_abs_charge,
                )
                if not eligibility.eligible:
                    candidate_database.execute(
                        """
                        UPDATE seed_status
                        SET
                            heavy_atom_count = ?,
                            formal_charge = ?,
                            resonance_status = 'skipped',
                            resonance_skip_reason = ?
                        WHERE seed_entity_id = ?
                        """,
                        (
                            eligibility.heavy_atom_count,
                            eligibility.formal_charge,
                            eligibility.reason,
                            seed_identifier,
                        ),
                    )
                    candidate_database.commit()

                generated_rows, resonance_audit = (
                    _generate_rule_candidates_with_audit(
                        seed_smiles,
                        role,
                        resonance_max_structs=resonance_max_structs,
                        resonance_max_heavy_atoms=(
                            resonance_max_heavy_atoms
                        ),
                        resonance_max_abs_charge=(
                            resonance_max_abs_charge
                        ),
                    )
                )
                local_rules: list[dict[str, object]] = []
                for generated in generated_rows:
                    generated_smiles = generated["SMILES"]
                    if not candidate_allowed(
                        seed_smiles,
                        generated_smiles,
                        role,
                    ):
                        continue
                    local_rules.append(
                        {
                            "candidate_smiles": generated_smiles,
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
                _replace_seed_candidates(
                    candidate_database,
                    seed_identifier=seed_identifier,
                    role=role,
                    seed_smiles=seed_smiles,
                    method="rule",
                    candidates=ordered_rules,
                )
                local_status = (
                    "resonance_skipped"
                    if resonance_audit["resonance_status"] == "skipped"
                    else "complete"
                )
                candidate_database.execute(
                    """
                    UPDATE seed_status
                    SET
                        local_status = ?,
                        heavy_atom_count = ?,
                        formal_charge = ?,
                        resonance_status = ?,
                        resonance_skip_reason = ?,
                        resonance_examined = ?,
                        resonance_generated = ?,
                        resonance_truncated = ?
                    WHERE seed_entity_id = ?
                    """,
                    (
                        local_status,
                        resonance_audit["heavy_atom_count"],
                        resonance_audit["formal_charge"],
                        resonance_audit["resonance_status"],
                        resonance_audit["resonance_skip_reason"],
                        resonance_audit["resonance_examined"],
                        resonance_audit["resonance_generated"],
                        int(resonance_audit["resonance_truncated"]),
                        seed_identifier,
                    ),
                )
                candidate_database.commit()

            if pubchem_status != "ok":
                similarity_candidates: list[dict[str, object]] = []
                try:
                    records = active_client.similarity(
                        seed_smiles,
                        threshold=threshold,
                        max_records=max_records,
                    )
                except IncompletePubChemQuery as exc:
                    candidate_database.execute(
                        """
                        UPDATE seed_status
                        SET
                            pubchem_status = 'failed',
                            pubchem_error = ?
                        WHERE seed_entity_id = ?
                        """,
                        (str(exc), seed_identifier),
                    )
                    candidate_database.commit()
                    if offline:
                        raise IncompletePubChemQuery(
                            "offline PubChem cache is incomplete for "
                            f"{seed_smiles}: {exc}"
                        ) from exc
                    records = None
                seen_similarity: set[str] = set()
                if records is not None:
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
                            or not candidate_allowed(
                                seed_smiles,
                                candidate,
                                role,
                            )
                        ):
                            continue
                        seen_similarity.add(candidate)
                        similarity_candidates.append(
                            {
                                "candidate_smiles": candidate,
                                "pubchem_cid": str(record.get("CID", "")),
                                "rule": "",
                                "method_rank": rank,
                            }
                        )
                    _replace_seed_candidates(
                        candidate_database,
                        seed_identifier=seed_identifier,
                        role=role,
                        seed_smiles=seed_smiles,
                        method="pubchem_similarity",
                        candidates=similarity_candidates,
                    )
                    candidate_database.execute(
                        """
                        UPDATE seed_status
                        SET
                            pubchem_status = 'ok',
                            pubchem_error = ''
                        WHERE seed_entity_id = ?
                        """,
                        (seed_identifier,),
                    )
                    candidate_database.commit()

            if processed % 100 == 0 or processed == total_seeds:
                candidate_count = candidate_database.execute(
                    "SELECT COUNT(*) FROM candidates"
                ).fetchone()[0]
                status_counts = dict(
                    candidate_database.execute(
                        """
                        SELECT pubchem_status, COUNT(*)
                        FROM seed_status
                        GROUP BY pubchem_status
                        """
                    )
                )
                resonance_counts = dict(
                    candidate_database.execute(
                        """
                        SELECT resonance_status, COUNT(*)
                        FROM seed_status
                        GROUP BY resonance_status
                        """
                    )
                )
                print(
                    "Stage1 augmentation: "
                    f"{processed:,}/{total_seeds:,} "
                    f"role={role} candidates={candidate_count:,} "
                    f"pubchem_ok={status_counts.get('ok', 0):,} "
                    f"pubchem_failed={status_counts.get('failed', 0):,} "
                    f"resonance_skipped="
                    f"{resonance_counts.get('skipped', 0):,} "
                    f"resonance_truncated="
                    f"{resonance_counts.get('truncated', 0):,}",
                    flush=True,
                )

        failures, resonance_audit = _augmentation_audit_frames(
            candidate_database
        )
        summary = _write_augmentation_files(
            stage1_root,
            candidate_database,
            base=base,
            config=config,
            failures=failures,
            resonance_audit=resonance_audit,
        )
        print(
            "Stage1 augmentation complete: "
            f"status={summary['completion_status']} "
            f"entities={summary['augmentation_entities']:,} "
            f"pubchem_failed={len(failures):,} "
            f"resonance_skipped="
            f"{summary['resonance_skipped_seeds']:,} "
            f"resonance_truncated="
            f"{summary['resonance_truncated_seeds']:,}",
            flush=True,
        )
    finally:
        candidate_database.close()
        if owns_client:
            active_client.close()
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)

    (output_root / "manifest.json").unlink(missing_ok=True)
    return summary


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


def _task_strategy_units(task: TaskSpec) -> dict[str, str]:
    units = {"random": "row_id"}
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


def fold_balance_audit_rows(
    task_id: str,
    strategy: str,
    repeat: int,
    group_values: pd.Series,
    folds: pd.Series,
) -> list[dict[str, object]]:
    row_counts = [
        int(folds.eq(fold).sum())
        for fold in range(5)
    ]
    group_counts = [
        int(group_values.loc[folds.eq(fold)].nunique())
        for fold in range(5)
    ]
    total_rows = sum(row_counts)
    mean_rows = total_rows / 5.0
    largest_group_rows = int(group_values.value_counts().max())
    maximum_to_minimum = (
        max(row_counts) / min(row_counts)
        if min(row_counts) > 0
        else float("inf")
    )
    theoretical_lower_bound = (
        4.0
        * largest_group_rows
        / (total_rows - largest_group_rows)
        if largest_group_rows > mean_rows
        and total_rows > largest_group_rows
        else 1.0
    )
    return [
        {
            "task_id": task_id,
            "strategy": strategy,
            "cv": repeat + 1,
            "fold": fold + 1,
            "row_count": row_counts[fold],
            "group_count": group_counts[fold],
            "total_rows": total_rows,
            "mean_rows": mean_rows,
            "largest_group_rows": largest_group_rows,
            "max_to_min": maximum_to_minimum,
            "theoretical_lower_bound": theoretical_lower_bound,
            "unavoidable_group_dominance": (
                largest_group_rows > mean_rows
            ),
        }
        for fold in range(5)
    ]


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


def build_training_splits(
    final_root: Path,
    output_root: Path,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Build readable Stage-2 and Stage-3 task datasets."""
    final_root = Path(final_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _migrate_pubchem_cache(output_root)
    validate_final_identity_consistency(final_root)

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

    task_catalog_rows: list[dict[str, object]] = []
    fold_balance_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(dir=output_root.parent) as temporary_dir:
        staged_root = Path(temporary_dir) / "training_splits"
        stage2_root = staged_root / "stage2"
        stage3_root = staged_root / "stage3"
        audit_root = staged_root / "_audit"
        stage2_root.mkdir(parents=True)
        stage3_root.mkdir()
        audit_root.mkdir()

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
                        task.task_id,
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
            partition_counts = assignments.groupby(
                "system_id",
                sort=False,
            )["partition"].nunique()
            if partition_counts.gt(1).any():
                raise TrainingSplitError(
                    f"Stage-2 system crosses partitions for {task.task_id}"
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
            random_unit = development["_row_id"]
            for repeat_index in range(repeats):
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
                group_folds = task_group_kfold_assignments(
                    group_values,
                    task_id=task.task_id,
                    strategy=strategy,
                    repeats=repeats,
                    seed=seed,
                )
                for repeat_index, folds in enumerate(group_folds):
                    if strategy in {"cation", "anion"}:
                        fold_balance_rows.extend(
                            fold_balance_audit_rows(
                                task.task_id,
                                strategy,
                                repeat_index,
                                group_values,
                                folds,
                            )
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

        write_dataframe(
            audit_root / "fold_balance.csv",
            pd.DataFrame(
                fold_balance_rows,
                columns=FOLD_BALANCE_COLUMNS,
            ),
        )
        replace_directory(stage2_root, output_root / "stage2")
        replace_directory(stage3_root, output_root / "stage3")
        replace_directory(audit_root, output_root / "_audit")

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
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--pubchem-threshold", type=int, default=90)
    parser.add_argument("--pubchem-max-records", type=int, default=100)
    parser.add_argument("--pubchem-rate", type=float, default=4.0)
    parser.add_argument(
        "--resonance-max-structs",
        type=int,
        default=DEFAULT_RESONANCE_MAX_STRUCTS,
    )
    parser.add_argument(
        "--resonance-max-heavy-atoms",
        type=int,
        default=DEFAULT_RESONANCE_MAX_HEAVY_ATOMS,
    )
    parser.add_argument(
        "--resonance-max-abs-charge",
        type=int,
        default=DEFAULT_RESONANCE_MAX_ABS_CHARGE,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.pubchem_threshold <= 100:
        raise TrainingSplitError("--pubchem-threshold must be between 1 and 100")
    if args.pubchem_max_records <= 0:
        raise TrainingSplitError("--pubchem-max-records must be positive")
    if args.pubchem_rate < 0:
        raise TrainingSplitError("--pubchem-rate cannot be negative")
    if args.resonance_max_structs <= 0:
        raise TrainingSplitError("--resonance-max-structs must be positive")
    if args.resonance_max_heavy_atoms <= 0:
        raise TrainingSplitError(
            "--resonance-max-heavy-atoms must be positive"
        )
    if args.resonance_max_abs_charge < 0:
        raise TrainingSplitError(
            "--resonance-max-abs-charge cannot be negative"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.command in {"extract-pretrain", "all"}:
        extracted = extract_pretraining_entities(
            args.final_root,
            args.output_root,
        )
        print(f"Extracted {len(extracted):,} base pretraining entities")
    if args.command in {"augment-pretrain", "all"}:
        augmented = augment_pretraining_entities(
            args.output_root,
            threshold=args.pubchem_threshold,
            max_records=args.pubchem_max_records,
            offline=args.offline,
            rate_limit=args.pubchem_rate,
            resonance_max_structs=args.resonance_max_structs,
            resonance_max_heavy_atoms=args.resonance_max_heavy_atoms,
            resonance_max_abs_charge=args.resonance_max_abs_charge,
        )
        counts = augmented["augmentation_entities_by_role"]
        print(
            "Prepared "
            f"{augmented['augmentation_entities']:,} augmentation entities "
            f"(anion={counts['anion']:,}, cation={counts['cation']:,}, "
            f"molecule={counts['molecule']:,})"
        )
    if args.command in {"build-splits", "all"}:
        catalog = build_training_splits(
            args.final_root,
            args.output_root,
            seed=args.seed,
        )
        stage2_count = int(catalog["stage"].eq(2).sum())
        stage3_count = int(catalog["stage"].eq(3).sum())
        print(
            f"Built readable datasets for {stage2_count} stage-2 tasks and "
            f"{stage3_count} stage-3 tasks"
        )


if __name__ == "__main__":
    main()
