"""Build one-row-per-box structural fingerprints for ionic-liquid PDB data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from box_features import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    QC_COLUMNS,
    empty_feature_row,
    extract_box_features,
    parse_mol2_info,
)
from raw_prep import canonicalize_smiles, disable_rdkit_logs  # noqa: E402

DEFAULT_BOX_DIR = PROJECT_ROOT / "data" / "raw" / "simulation_data" / "box_20260514"
DEFAULT_CHARGE_DIR = PROJECT_ROOT / "data" / "raw" / "simulation_data" / "charge_20260514"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "final" / "simulation" / "3d_box.csv"
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "data" / "structured" / "simulation" / "3d_box_audit"
ID_COLUMNS = ("mol_id", "cation", "anion", "temperature_K", "feature_version")
OUTPUT_COLUMNS = (*ID_COLUMNS, *FEATURE_COLUMNS, *QC_COLUMNS)
CURVE_SHAPES = {
    "rdf_r_A": (200,),
    **{f"rdf_{key}": (200,) for key in ("cc", "ca", "aa", "pp", "pn", "nn")},
    "q": (30,),
    "scc": (30,),
    "scc_modes": (30,),
    "q_z": (30,),
    "szz": (30,),
    "szz_modes": (30,),
}

_WORKER_BOX_DIR: Path | None = None
_WORKER_MOL2_PATHS: dict[str, str] = {}
_WORKER_MOL2_CACHE: dict[str, Any] = {}
_WORKER_TOPOLOGY_CACHE: dict[Any, Any] = {}


def mapping_digest(mapping_path: Path) -> str:
    """Return a streaming SHA-256 digest for resume compatibility checks."""

    digest = hashlib.sha256()
    with Path(mapping_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mapping(mapping_path: Path) -> pd.DataFrame:
    """Load and canonicalize the raw box mapping while preserving row order."""

    frame = pd.read_csv(mapping_path)
    expected = {"mol_id", "cation_smiles", "anion_smiles", "temperature"}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"mapping is missing columns: {sorted(missing)}")
    if frame["mol_id"].duplicated().any():
        raise ValueError("mapping contains duplicate mol_id values")
    frame = frame.rename(
        columns={"cation_smiles": "cation", "anion_smiles": "anion", "temperature": "temperature_K"}
    )
    frame["cation"] = frame["cation"].map(canonicalize_smiles)
    frame["anion"] = frame["anion"].map(canonicalize_smiles)
    frame["temperature_K"] = pd.to_numeric(frame["temperature_K"], errors="raise")
    return frame[["mol_id", "cation", "anion", "temperature_K"]]


def build_mol2_lookup(charge_dir: Path, ion_smiles: set[str]) -> dict[str, str]:
    """Map only ions used by the box dataset to optional MOL2 paths."""

    mapping_path = Path(charge_dir) / "mapping.csv"
    if not mapping_path.exists():
        return {}
    charge_mapping = pd.read_csv(mapping_path, usecols=["mol_id", "smiles"])
    exact = {
        str(smiles): str(Path(charge_dir) / f"{mol_id}.mol2")
        for mol_id, smiles in charge_mapping.itertuples(index=False, name=None)
        if str(smiles) in ion_smiles
    }
    missing = ion_smiles.difference(exact)
    if not missing:
        return exact
    for mol_id, smiles in charge_mapping.itertuples(index=False, name=None):
        canonical = canonicalize_smiles(str(smiles))
        if canonical in missing:
            exact[canonical] = str(Path(charge_dir) / f"{mol_id}.mol2")
    return exact


def _init_worker(box_dir: str, mol2_paths: dict[str, str]) -> None:
    global _WORKER_BOX_DIR, _WORKER_MOL2_PATHS, _WORKER_MOL2_CACHE, _WORKER_TOPOLOGY_CACHE
    disable_rdkit_logs()
    _WORKER_BOX_DIR = Path(box_dir)
    _WORKER_MOL2_PATHS = mol2_paths
    _WORKER_MOL2_CACHE = {}
    _WORKER_TOPOLOGY_CACHE = {}


def _mol2_info(smiles: str) -> Any | None:
    path = _WORKER_MOL2_PATHS.get(smiles)
    if not path or not Path(path).exists():
        return None
    if smiles not in _WORKER_MOL2_CACHE:
        try:
            _WORKER_MOL2_CACHE[smiles] = parse_mol2_info(Path(path))
        except (OSError, ValueError):
            _WORKER_MOL2_CACHE[smiles] = None
    return _WORKER_MOL2_CACHE[smiles]


def process_mapping_record(record: tuple[str, str, str, float]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Worker-safe conversion of one mapping record."""

    mol_id, cation, anion, temperature = record
    result = {
        "mol_id": mol_id,
        "cation": cation,
        "anion": anion,
        "temperature_K": temperature,
        "feature_version": FEATURE_VERSION,
    }
    try:
        assert _WORKER_BOX_DIR is not None
        features, curves = extract_box_features(
            _WORKER_BOX_DIR / f"{mol_id}.pdb",
            cation,
            anion,
            cation_mol2=_mol2_info(cation),
            anion_mol2=_mol2_info(anion),
            topology_cache=_WORKER_TOPOLOGY_CACHE,
        )
    except Exception as error:  # one corrupt box must not discard the dataset
        features = empty_feature_row()
        features["qc_pdb_present"] = (_WORKER_BOX_DIR / f"{mol_id}.pdb").exists()
        features["qc_flags"] = f"processing_error:{type(error).__name__}"
        features["qc_error"] = str(error)[:500]
        curves = {}
    result.update(features)
    return result, curves


def _curve_batch(results: list[tuple[dict[str, Any], dict[str, np.ndarray]]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "mol_ids": np.asarray([row["mol_id"] for row, _curves in results], dtype="U16")
    }
    for name, shape in CURVE_SHAPES.items():
        values = []
        for _row, curves in results:
            value = curves.get(name)
            if value is None or value.shape != shape:
                value = np.full(shape, np.nan, dtype=np.float32)
            values.append(np.asarray(value, dtype=np.float32))
        arrays[name] = np.stack(values)
    return arrays


def _write_batch(
    batch_index: int,
    results: list[tuple[dict[str, Any], dict[str, np.ndarray]]],
    batch_dir: Path,
    curve_dir: Path,
) -> None:
    batch_path = batch_dir / f"batch_{batch_index:05d}.csv"
    curve_path = curve_dir / f"batch_{batch_index:05d}.npz"
    batch_tmp = batch_path.with_suffix(".csv.tmp")
    curve_tmp = curve_path.with_suffix(".npz.tmp")
    frame = pd.DataFrame([row for row, _curves in results]).reindex(columns=OUTPUT_COLUMNS)
    frame.to_csv(batch_tmp, index=False)
    with curve_tmp.open("wb") as handle:
        np.savez_compressed(handle, **_curve_batch(results))
    os.replace(batch_tmp, batch_path)
    os.replace(curve_tmp, curve_path)


def _write_manifest(
    path: Path,
    *,
    mapping_sha256: str,
    batch_size: int,
    row_count: int,
    mol2_coverage: int,
) -> None:
    payload = {
        "feature_version": FEATURE_VERSION,
        "mapping_sha256": mapping_sha256,
        "batch_size": batch_size,
        "row_count": row_count,
        "mol2_ion_coverage": mol2_coverage,
        "scientific_parameters": {
            "rdf_dr_A": 0.1,
            "rdf_r_max_A": 20.0,
            "hbond_primary": {"HA_A": 2.5, "DA_A": 3.5, "DHA_deg": 150.0},
            "structure_factor_grid": 64,
            "structure_factor_dq_A^-1": 0.05,
        },
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate_resume_manifest(path: Path, mapping_sha256: str, batch_size: int, row_count: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = (FEATURE_VERSION, mapping_sha256, batch_size, row_count)
    actual = (
        payload.get("feature_version"),
        payload.get("mapping_sha256"),
        payload.get("batch_size"),
        payload.get("row_count"),
    )
    if actual != expected:
        raise ValueError(f"resume manifest does not match current run: expected={expected}, actual={actual}")


def build_dataset(
    *,
    box_dir: Path = DEFAULT_BOX_DIR,
    charge_dir: Path = DEFAULT_CHARGE_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    jobs: int = 1,
    resume: bool = False,
    batch_size: int = 100,
    limit: int | None = None,
) -> pd.DataFrame:
    """Build the fingerprint CSV with deterministic, resumable batch checkpoints."""

    disable_rdkit_logs()
    box_dir = Path(box_dir)
    mapping_path = box_dir / "mapping.csv"
    mapping = load_mapping(mapping_path)
    if limit is not None:
        mapping = mapping.iloc[:limit].copy()
    output_path = Path(output_path)
    audit_dir = Path(audit_dir)
    batch_dir = audit_dir / "batches"
    curve_dir = audit_dir / "curves"
    manifest_path = audit_dir / "run_manifest.json"
    batch_dir.mkdir(parents=True, exist_ok=True)
    curve_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    digest = mapping_digest(mapping_path)
    ion_smiles = set(mapping["cation"]).union(mapping["anion"])
    mol2_paths = build_mol2_lookup(charge_dir, ion_smiles)
    if resume and manifest_path.exists():
        _validate_resume_manifest(manifest_path, digest, batch_size, len(mapping))
    else:
        for directory, pattern in ((batch_dir, "batch_*.csv"), (curve_dir, "batch_*.npz")):
            for stale in directory.glob(pattern):
                stale.unlink()
        _write_manifest(
            manifest_path,
            mapping_sha256=digest,
            batch_size=batch_size,
            row_count=len(mapping),
            mol2_coverage=len(mol2_paths),
        )

    records = list(mapping.itertuples(index=False, name=None))
    executor: ProcessPoolExecutor | None = None
    if jobs > 1:
        executor = ProcessPoolExecutor(
            max_workers=jobs, initializer=_init_worker, initargs=(str(box_dir), mol2_paths)
        )
    else:
        _init_worker(str(box_dir), mol2_paths)
    try:
        for batch_index, start in enumerate(range(0, len(records), batch_size)):
            batch_records = records[start : start + batch_size]
            batch_path = batch_dir / f"batch_{batch_index:05d}.csv"
            if resume and batch_path.exists():
                existing_ids = pd.read_csv(batch_path, usecols=["mol_id"])["mol_id"].tolist()
                expected_ids = [record[0] for record in batch_records]
                if existing_ids != expected_ids:
                    raise ValueError(f"resume batch {batch_index} has unexpected mol_id values")
                print(f"batch={batch_index} skipped={len(batch_records)}")
                continue
            started = time.monotonic()
            if executor is None:
                results = [process_mapping_record(record) for record in batch_records]
            else:
                results = list(executor.map(process_mapping_record, batch_records, chunksize=1))
            _write_batch(batch_index, results, batch_dir, curve_dir)
            elapsed = time.monotonic() - started
            print(f"batch={batch_index} saved={len(results)} elapsed_s={elapsed:.2f}")
    finally:
        if executor is not None:
            executor.shutdown()

    batch_paths = sorted(batch_dir.glob("batch_*.csv"))
    final = pd.concat([pd.read_csv(path) for path in batch_paths], ignore_index=True)
    final = mapping.merge(final, on=["mol_id", "cation", "anion", "temperature_K"], how="left", validate="one_to_one")
    final = final.reindex(columns=OUTPUT_COLUMNS)
    if len(final) != len(mapping) or final["mol_id"].duplicated().any():
        raise RuntimeError("final row count or mol_id uniqueness validation failed")
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    final.to_csv(temporary_output, index=False)
    os.replace(temporary_output, output_path)
    failures = final[final["qc_status"].ne("ok")]
    failures.to_csv(audit_dir / "qc_non_ok.csv", index=False)
    print(
        f"saved={output_path} rows={len(final)} ok={(final['qc_status'] == 'ok').sum()} "
        f"partial={(final['qc_status'] == 'partial').sum()} failed={(final['qc_status'] == 'failed').sum()}"
    )
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-dir", type=Path, default=DEFAULT_BOX_DIR)
    parser.add_argument("--charge-dir", type=Path, default=DEFAULT_CHARGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--jobs", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, help="process only the first N rows for validation")
    args = parser.parse_args()
    if args.jobs < 1 or args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--jobs, --batch-size and --limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    build_dataset(
        box_dir=args.box_dir,
        charge_dir=args.charge_dir,
        output_path=args.output,
        audit_dir=args.audit_dir,
        jobs=args.jobs,
        resume=args.resume,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
