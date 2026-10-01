"""Snapshot binary/ternary ILThermo entries as raw JSON and long-form Parquet.

The two unfiltered Search results are frozen per output directory. Bronze entry
responses are the authority for every Silver rebuild and offline verification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import ilthermopy as ilt
import ilthermopy.requests as ilt_requests
import ilthermopy.data_structs as ilt_data_structs
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "ilthermo_mixtures"
PIPELINE_VERSION = "1.0.1"
ILTHERMOPY_VERSION = "1.1.2"
ENTRY_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
UPDATE_RE = re.compile(r"Updated on ([a-zA-Z]+ +\d+, +\d+)")

ENTRY_COLUMNS = [
    "entry_id", "mixture_size", "manifest_property", "property_type",
    "detail_property", "phases_json", "reference_full", "reference_title",
    "experimental_method", "solvent", "constraints_json", "footer",
    "manifest_component_count", "detail_component_count",
    "manifest_data_point_count", "detail_data_point_count", "validation_status",
    "fetched_at_utc", "raw_sha256",
]
COMPONENT_COLUMNS = [
    "entry_id", "component_index", "component_id", "name", "formula_raw",
    "molecular_weight_raw", "sample_json",
]
OBSERVATION_COLUMNS = [
    "entry_id", "data_point_index", "variable_index", "raw_header",
    "variable_name", "unit", "phase", "value_raw", "uncertainty_raw",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1)
def pipeline_source_sha256() -> str:
    return sha256(Path(__file__).read_bytes())


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json_bytes(value))


def atomic_parquet(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        pd.DataFrame(rows, columns=columns).to_parquet(tmp, index=False)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@lru_cache(maxsize=1)
def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def database_update_date() -> str | None:
    try:
        response = requests.get("https://ilthermo.boulder.nist.gov/", timeout=(10, 30))
        response.raise_for_status()
        match = UPDATE_RE.search(response.text)
        if match:
            return datetime.strptime(match.group(1), "%B %d, %Y").date().isoformat()
    except (requests.RequestException, ValueError):
        pass
    return None


class _TimedRequests:
    """Supply timeouts to ILThermoPy 1.1.2's request wrapper."""

    def __init__(self, original: Any):
        self.original = original

    def get(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", (10, 60))
        return self.original.get(*args, **kwargs)


@contextmanager
def timed_ilthermopy_requests():
    original = ilt_requests._requests
    ilt_requests._requests = _TimedRequests(original)
    try:
        yield
    finally:
        ilt_requests._requests = original


def retry(call: Callable[[], Any], label: str, attempts: int, sleep: Callable[[float], None] = time.sleep) -> Any:
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            if attempt + 1 == attempts:
                raise RuntimeError(f"{label} failed after {attempts} attempts: {exc}") from exc
            sleep(2**attempt)
    raise AssertionError("attempts must be positive")


def fetch_entry(entry_id: str) -> Any:
    """Use GetEntry's request while preserving responses its table parser rejects."""
    original = ilt_data_structs.ResponseToEntry

    def response_to_entry(_code: str, response: dict[str, Any]) -> Any:
        return SimpleNamespace(response=response)

    ilt_data_structs.ResponseToEntry = response_to_entry
    try:
        return ilt.GetEntry(entry_id)
    finally:
        ilt_data_structs.ResponseToEntry = original


def manifest_path(root: Path, size: int) -> Path:
    return root / "bronze" / "manifests" / ("binary.json" if size == 2 else "ternary.json")


def entry_paths(root: Path, entry_id: str) -> tuple[Path, Path, Path]:
    if not ENTRY_ID_RE.fullmatch(entry_id):
        raise ValueError(f"Unsafe entry ID: {entry_id!r}")
    bronze = root / "bronze"
    return (
        bronze / "entries" / f"{entry_id}.json",
        bronze / "metadata" / f"{entry_id}.json",
        bronze / "failures" / f"{entry_id}.json",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(manifest: dict[str, Any], size: int) -> None:
    if manifest.get("n_compounds") != size or not isinstance(manifest.get("rows"), list):
        raise ValueError(f"Invalid {size}-component Search manifest")
    if not manifest["rows"]:
        raise ValueError(f"Empty {size}-component Search manifest; refusing to freeze it")
    for row in manifest["rows"]:
        if not isinstance(row, dict) or not ENTRY_ID_RE.fullmatch(str(row.get("id", ""))):
            raise ValueError(f"Invalid entry ID in {size}-component manifest: {row!r}")
        if int(row["num_components"]) != size or int(row["num_data_points"]) < 0:
            raise ValueError(f"Invalid counts for entry {row['id']}")
        if not isinstance(row.get("property"), str):
            raise ValueError(f"Missing property for entry {row['id']}")


def load_or_search_manifests(root: Path, attempts: int) -> dict[int, dict[str, Any]]:
    run_info_path = root / "run_info.json"
    if run_info_path.exists():
        run_info = load_json(run_info_path)
    else:
        run_info = {
            "created_at_utc": utc_now(),
            "ilthermo_database_updated_on": database_update_date(),
            "ilthermopy_version": version("ilthermopy"),
            "pipeline_version": PIPELINE_VERSION,
            "pipeline_source_sha256": pipeline_source_sha256(),
            "git_commit": git_commit(),
        }
        atomic_json(run_info_path, run_info)

    manifests = {}
    for size in (2, 3):
        path = manifest_path(root, size)
        if path.exists():
            manifest = load_json(path)
        else:
            frame = retry(lambda: ilt.Search(n_compounds=size), f"Search({size})", attempts)
            manifest = {
                "n_compounds": size,
                "searched_at_utc": utc_now(),
                "ilthermo_database_updated_on": run_info["ilthermo_database_updated_on"],
                "ilthermopy_version": version("ilthermopy"),
                "rows": json.loads(frame.to_json(orient="records")),
            }
            validate_manifest(manifest, size)
            atomic_json(path, manifest)
        validate_manifest(manifest, size)
        manifests[size] = manifest
    return manifests


def unique_entries(manifests: dict[int, dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    entries: dict[str, tuple[int, dict[str, Any]]] = {}
    for size in (2, 3):
        for row in manifests[size]["rows"]:
            entry_id = str(row["id"])
            if entry_id in entries:
                old_size, old = entries[entry_id]
                keys = ("num_components", "num_data_points", "property")
                if old_size != size or any(old[key] != row[key] for key in keys):
                    raise ValueError(f"Conflicting Search rows for entry {entry_id}")
            else:
                entries[entry_id] = (size, row)
    return [(size, row) for size, row in entries.values()]


def load_complete_entry(root: Path, entry_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    raw_path, meta_path, _ = entry_paths(root, entry_id)
    if not raw_path.exists() or not meta_path.exists():
        return None, None, "missing Bronze JSON or metadata"
    try:
        raw_bytes = raw_path.read_bytes()
        meta = load_json(meta_path)
        raw = json.loads(raw_bytes)
        if not isinstance(raw, dict) or meta.get("entry_id") != entry_id:
            raise ValueError("invalid Bronze response or entry ID")
        if meta.get("response_sha256") != sha256(raw_bytes):
            raise ValueError("Bronze SHA-256 mismatch")
        return raw, meta, None
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return None, None, str(exc)


def save_entry(root: Path, entry_id: str, response: dict[str, Any]) -> None:
    if not isinstance(response, dict):
        raise ValueError(f"GetEntry({entry_id}) returned a non-object response")
    raw_path, meta_path, failure_path = entry_paths(root, entry_id)
    data = json_bytes(response)
    atomic_json(meta_path, {
        "entry_id": entry_id,
        "fetched_at_utc": utc_now(),
        "response_sha256": sha256(data),
        "ilthermopy_version": version("ilthermopy"),
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_source_sha256": pipeline_source_sha256(),
        "git_commit": git_commit(),
    })
    atomic_bytes(raw_path, data)
    failure_path.unlink(missing_ok=True)


def raw_text(value: Any) -> str | None:
    return None if value is None else str(value)


def parse_entry(
    entry_id: str, size: int, search_row: dict[str, Any],
    response: dict[str, Any], meta: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    components = response["components"]
    data = response["data"]
    headers = response["dhead"]
    if not isinstance(components, list) or not isinstance(data, list) or not isinstance(headers, list):
        raise ValueError("components, data and dhead must be lists")
    issues = []
    warnings = []
    if len(components) != int(search_row["num_components"]):
        issues.append(f"component count: Search={search_row['num_components']} detail={len(components)}")
    if len(data) != int(search_row["num_data_points"]):
        issues.append(f"data point count: Search={search_row['num_data_points']} detail={len(data)}")

    title = response.get("title") or ""
    property_type, _, detail_property = title.partition(":")
    reference = response.get("ref") or {}
    entry = {
        "entry_id": entry_id,
        "mixture_size": size,
        "manifest_property": search_row["property"],
        "property_type": property_type.strip(),
        "detail_property": detail_property.strip(),
        "phases_json": json.dumps(response.get("phases"), ensure_ascii=False),
        "reference_full": reference.get("full"),
        "reference_title": reference.get("title"),
        "experimental_method": response.get("expmeth"),
        "solvent": response.get("solvent"),
        "constraints_json": json.dumps(response.get("constr"), ensure_ascii=False),
        "footer": response.get("footer"),
        "manifest_component_count": int(search_row["num_components"]),
        "detail_component_count": len(components),
        "manifest_data_point_count": int(search_row["num_data_points"]),
        "detail_data_point_count": len(data),
        "validation_status": "count_mismatch" if issues else "ok",
        "fetched_at_utc": meta["fetched_at_utc"],
        "raw_sha256": meta["response_sha256"],
    }
    component_rows = []
    for index, component in enumerate(components, start=1):
        if not isinstance(component, dict):
            raise ValueError(f"component {index} is not an object")
        component_rows.append({
            "entry_id": entry_id,
            "component_index": index,
            "component_id": component.get("idout"),
            "name": component.get("name"),
            "formula_raw": component.get("formula"),
            "molecular_weight_raw": raw_text(component.get("mw")),
            "sample_json": json.dumps(component.get("sample"), ensure_ascii=False),
        })

    observation_rows = []
    for point_index, cells in enumerate(data, start=1):
        if not isinstance(cells, list) or len(cells) > len(headers):
            raise ValueError(f"data point {point_index} has more cells than dhead")
        if len(cells) < len(headers):
            warnings.append(f"data point {point_index} has {len(cells)} of {len(headers)} variables")
        for variable_index, (header, cell) in enumerate(zip(headers, cells), start=1):
            if not isinstance(header, list) or not header or not isinstance(header[0], str):
                raise ValueError(f"invalid dhead variable {variable_index}")
            if cell is None:
                warnings.append(f"data point {point_index} variable {variable_index} is null")
                cell_values = (None, None)
            elif isinstance(cell, list) and len(cell) in (1, 2):
                cell_values = (raw_text(cell[0]), raw_text(cell[1]) if len(cell) == 2 else None)
            else:
                raise ValueError(f"invalid cell at point {point_index}, variable {variable_index}")
            raw_header = header[0]
            name, separator, unit = raw_header.rpartition(",")
            observation_rows.append({
                "entry_id": entry_id,
                "data_point_index": point_index,
                "variable_index": variable_index,
                "raw_header": raw_header,
                "variable_name": name.strip() if separator else raw_header,
                "unit": unit.strip() if separator else None,
                "phase": raw_text(header[1]) if len(header) > 1 else None,
                "value_raw": cell_values[0],
                "uncertainty_raw": cell_values[1],
            })
    if warnings and not issues:
        entry["validation_status"] = "source_irregular"
    return entry, component_rows, observation_rows, issues, warnings


def collect(root: Path, manifests: dict[int, dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tables: dict[str, list[dict[str, Any]]] = {"entries": [], "components": [], "observations": []}
    failures = []
    warnings = []
    summary = {
        "binary": {"manifest_entries": 0, "downloaded_entries": 0, "data_points": 0, "pending_entries": 0, "failed_entries": 0},
        "ternary": {"manifest_entries": 0, "downloaded_entries": 0, "data_points": 0, "pending_entries": 0, "failed_entries": 0},
    }
    coverage: dict[tuple[int, str], dict[str, Any]] = {}
    for size, search_row in unique_entries(manifests):
        entry_id = str(search_row["id"])
        group = summary["binary" if size == 2 else "ternary"]
        group["manifest_entries"] += 1
        key = (size, search_row["property"])
        item = coverage.setdefault(key, {
            "mixture_size": size, "property": search_row["property"],
            "manifest_entries": 0, "manifest_data_points": 0,
            "downloaded_entries": 0, "data_points": 0, "pending_entries": 0, "failed_entries": 0,
        })
        item["manifest_entries"] += 1
        item["manifest_data_points"] += int(search_row["num_data_points"])
        raw, meta, error = load_complete_entry(root, entry_id)
        if error is not None:
            raw_path, meta_path, failure_path = entry_paths(root, entry_id)
            if not raw_path.exists() and not meta_path.exists() and not failure_path.exists():
                group["pending_entries"] += 1
                item["pending_entries"] += 1
                continue
            if failure_path.exists():
                try:
                    error = load_json(failure_path).get("error", error)
                except (OSError, ValueError, AttributeError):
                    error = f"invalid failure log; {error}"
            failures.append({"entry_id": entry_id, "mixture_size": size, "property": search_row["property"], "error": error})
            group["failed_entries"] += 1
            item["failed_entries"] += 1
            continue
        group["downloaded_entries"] += 1
        item["downloaded_entries"] += 1
        raw_points = raw.get("data")
        if isinstance(raw_points, list):
            group["data_points"] += len(raw_points)
            item["data_points"] += len(raw_points)
        try:
            entry, components, observations, issues, entry_warnings = parse_entry(entry_id, size, search_row, raw, meta)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            issues = [f"parse error: {exc}"]
            entry_warnings = []
        else:
            tables["entries"].append(entry)
            tables["components"].extend(components)
            tables["observations"].extend(observations)
            warnings.extend({
                "entry_id": entry_id,
                "mixture_size": size,
                "property": search_row["property"],
                "warning": warning,
            } for warning in entry_warnings)
        if issues:
            failures.append({"entry_id": entry_id, "mixture_size": size, "property": search_row["property"], "error": "; ".join(issues)})
            group["failed_entries"] += 1
            item["failed_entries"] += 1
    summary["coverage"] = [coverage[key] for key in sorted(coverage)]
    summary["warning_count"] = len(warnings)
    summary["status"] = "complete" if not failures and not any(summary[name]["pending_entries"] for name in ("binary", "ternary")) else "incomplete"
    return tables, failures, warnings, summary


def write_silver(root: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    silver = root / "silver"
    for name, columns in (
        ("entries", ENTRY_COLUMNS),
        ("components", COMPONENT_COLUMNS),
        ("observations", OBSERVATION_COLUMNS),
    ):
        atomic_parquet(silver / f"{name}.parquet", tables[name], columns)
    atomic_json(silver / "build_info.json", {
        "built_at_utc": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_source_sha256": pipeline_source_sha256(),
        "git_commit": git_commit(),
        "manifest_sha256": {
            "binary": sha256(manifest_path(root, 2).read_bytes()),
            "ternary": sha256(manifest_path(root, 3).read_bytes()),
        },
    })


def silver_issues(root: Path, expected: dict[str, list[dict[str, Any]]]) -> list[str]:
    issues = []
    try:
        build_info = load_json(root / "silver" / "build_info.json")
        manifest_hashes = {
            "binary": sha256(manifest_path(root, 2).read_bytes()),
            "ternary": sha256(manifest_path(root, 3).read_bytes()),
        }
        if build_info.get("manifest_sha256") != manifest_hashes:
            issues.append("silver: build manifest hashes differ from current manifests")
    except (OSError, ValueError, AttributeError) as exc:
        issues.append(f"silver: cannot validate build metadata: {exc}")
    for name, keys, columns in (
        ("entries", ["entry_id"], ENTRY_COLUMNS),
        ("components", ["entry_id", "component_index"], COMPONENT_COLUMNS),
        ("observations", ["entry_id", "data_point_index", "variable_index"], OBSERVATION_COLUMNS),
    ):
        path = root / "silver" / f"{name}.parquet"
        try:
            actual = pd.read_parquet(path)
            if actual.columns.tolist() != columns:
                issues.append(f"{name}: columns differ from schema")
                continue
            wanted = pd.DataFrame(expected[name], columns=columns)
            if len(actual) != len(wanted):
                issues.append(f"{name}: row count {len(actual)} != {len(wanted)}")
            if actual.duplicated(keys).any():
                issues.append(f"{name}: duplicate primary keys")
            if set(map(tuple, actual[keys].itertuples(index=False, name=None))) != set(map(tuple, wanted[keys].itertuples(index=False, name=None))):
                issues.append(f"{name}: primary keys differ from Bronze")
            try:
                pd.testing.assert_frame_equal(actual, wanted, check_dtype=False)
            except AssertionError:
                issues.append(f"{name}: values differ from Bronze")
        except (OSError, ValueError, KeyError, ImportError) as exc:
            issues.append(f"{name}: cannot validate Parquet: {exc}")
    return issues


def write_reports(root: Path, summary: dict[str, Any], failures: list[dict[str, Any]], warnings: list[dict[str, Any]], issues: list[str]) -> None:
    report = root / "reports"
    run_info = load_json(root / "run_info.json")
    build_info_path = root / "silver" / "build_info.json"
    try:
        build_info = load_json(build_info_path)
    except (OSError, ValueError):
        build_info = None
    output = {
        **run_info,
        "verified_at_utc": utc_now(),
        "verifier_pipeline_version": PIPELINE_VERSION,
        "verifier_source_sha256": pipeline_source_sha256(),
        "silver_build": build_info,
        "status": "complete" if summary["status"] == "complete" and not issues else "incomplete",
        "binary": summary["binary"],
        "ternary": summary["ternary"],
        "warning_count": summary["warning_count"],
        "silver_issues": issues,
        "manifest_sha256": {
            "binary": sha256(manifest_path(root, 2).read_bytes()),
            "ternary": sha256(manifest_path(root, 3).read_bytes()),
        },
    }
    atomic_json(report / "summary.json", output)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures)
    atomic_bytes(report / "failures.jsonl", text.encode("utf-8"))
    warning_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in warnings)
    atomic_bytes(report / "warnings.jsonl", warning_text.encode("utf-8"))
    buffer = io.StringIO()
    columns = ["mixture_size", "property", "manifest_entries", "manifest_data_points", "downloaded_entries", "data_points", "pending_entries", "failed_entries"]
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(summary["coverage"])
    atomic_bytes(report / "coverage_by_property.csv", buffer.getvalue().encode("utf-8"))


def verify(root: Path, manifests: dict[int, dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], list[str], int]:
    if manifests is None:
        manifests = {size: load_json(manifest_path(root, size)) for size in (2, 3)}
        for size, manifest in manifests.items():
            validate_manifest(manifest, size)
    tables, failures, warnings, summary = collect(root, manifests)
    issues = silver_issues(root, tables)
    write_reports(root, summary, failures, warnings, issues)
    pending = summary["binary"]["pending_entries"] + summary["ternary"]["pending_entries"]
    print(json.dumps({"status": "complete" if not failures and not issues and not pending else "incomplete", "binary": summary["binary"], "ternary": summary["ternary"], "silver_issues": issues}, ensure_ascii=False))
    return failures, issues, pending


def crawl(root: Path, max_per_mixture: int | None, attempts: int, pause_seconds: float) -> bool:
    if version("ilthermopy") != ILTHERMOPY_VERSION:
        raise RuntimeError(f"Install ilthermopy=={ILTHERMOPY_VERSION}; found {version('ilthermopy')}")
    with timed_ilthermopy_requests():
        manifests = load_or_search_manifests(root, attempts)
        selected: list[tuple[int, dict[str, Any]]] = []
        for size in (2, 3):
            rows = [(s, row) for s, row in unique_entries(manifests) if s == size]
            selected.extend(rows[:max_per_mixture])
        selected_ids = {str(row["id"]) for _, row in selected}
        for index, (_, row) in enumerate(selected):
            entry_id = str(row["id"])
            if load_complete_entry(root, entry_id)[2] is None:
                continue
            if index:
                time.sleep(pause_seconds)
            try:
                entry = retry(lambda: fetch_entry(entry_id), f"GetEntry({entry_id})", attempts)
                save_entry(root, entry_id, entry.response)
            except Exception as exc:
                _, _, failure_path = entry_paths(root, entry_id)
                atomic_json(failure_path, {"entry_id": entry_id, "failed_at_utc": utc_now(), "error": str(exc)})
    tables, _, _, _ = collect(root, manifests)
    write_silver(root, tables)
    failures, issues, _ = verify(root, manifests)
    return not issues and not any(failure["entry_id"] in selected_ids for failure in failures) and all(
        load_complete_entry(root, entry_id)[2] is None for entry_id in selected_ids
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("crawl", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
        if name == "crawl":
            command.add_argument("--max-per-mixture", type=int, help="Fetch only this many binary and ternary entries for a smoke run")
            command.add_argument("--attempts", type=int, default=5)
            command.add_argument("--pause-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.command == "crawl":
        if args.attempts < 1 or args.pause_seconds < 0 or (args.max_per_mixture is not None and args.max_per_mixture < 1):
            parser.error("attempts and max-per-mixture must be positive; pause-seconds must be nonnegative")
        success = crawl(args.output_root, args.max_per_mixture, args.attempts, args.pause_seconds)
    else:
        failures, issues, pending = verify(args.output_root)
        success = not failures and not issues and not pending
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
