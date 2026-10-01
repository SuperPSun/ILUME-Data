"""Contract checks for the independent ILThermo mixture snapshot pipeline."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import crawl_ilthermo_mixtures as crawler


def search_row(entry_id: str, size: int, points: int, prop: str = "Density") -> dict:
    return {
        "id": entry_id,
        "reference": "Reference",
        "property": prop,
        "phases": "Liquid",
        "num_components": size,
        "num_data_points": points,
    }


def response(size: int, points: int) -> dict:
    return {
        "title": "Volumetric properties: Density",
        "phases": ["Liquid"],
        "ref": {"full": "Reference", "title": "Paper"},
        "expmeth": "method",
        "solvent": None,
        "constr": [],
        "footer": "footnote",
        "components": [
            {"idout": f"c{i}", "name": f"component {i}", "formula": f"C{i}", "mw": str(i), "sample": [["Source:", "test"]]}
            for i in range(1, size + 1)
        ],
        "dhead": [["Temperature, K", None], ["Density, kg/m3", "Liquid"]],
        "data": [[[str(300 + i)], [str(1000 + i), "0.2"]] for i in range(points)],
    }


def install_fakes(monkeypatch, details):
    search_calls = []
    entry_calls = []

    def fake_search(*, n_compounds):
        search_calls.append(n_compounds)
        if n_compounds == 2:
            return pd.DataFrame([search_row("b1", 2, 2), search_row("b1", 2, 2), search_row("b2", 2, 1)])
        return pd.DataFrame([search_row("t1", 3, 1)])

    def fake_entry(entry_id):
        entry_calls.append(entry_id)
        return SimpleNamespace(response=details[entry_id])

    monkeypatch.setattr(crawler.ilt, "Search", fake_search)
    monkeypatch.setattr(crawler.ilt, "GetEntry", fake_entry)
    monkeypatch.setattr(crawler, "database_update_date", lambda: "2026-05-01")
    monkeypatch.setattr(crawler, "version", lambda _: crawler.ILTHERMOPY_VERSION)
    monkeypatch.setattr(crawler, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(crawler.time, "sleep", lambda _: None)
    return search_calls, entry_calls


def test_snapshot_resume_and_long_form(tmp_path, monkeypatch):
    details = {"b1": response(2, 2), "b2": response(2, 1), "t1": response(3, 1)}
    search_calls, entry_calls = install_fakes(monkeypatch, details)

    assert crawler.crawl(tmp_path, max_per_mixture=1, attempts=2, pause_seconds=0)
    first_report = crawler.load_json(tmp_path / "reports" / "summary.json")
    assert first_report["status"] == "incomplete"
    assert first_report["binary"] == {
        "manifest_entries": 2, "downloaded_entries": 1, "data_points": 2,
        "pending_entries": 1, "failed_entries": 0,
    }
    assert entry_calls == ["b1", "t1"]
    assert crawler.load_json(tmp_path / "bronze" / "entries" / "t1.json") == details["t1"]

    assert crawler.crawl(tmp_path, max_per_mixture=None, attempts=2, pause_seconds=0)
    assert entry_calls == ["b1", "t1", "b2"]
    assert search_calls == [2, 3]
    report = crawler.load_json(tmp_path / "reports" / "summary.json")
    assert report["status"] == "complete"
    assert report["ternary"]["data_points"] == 1
    assert report["ilthermo_database_updated_on"] == "2026-05-01"
    assert report["ilthermopy_version"] == "1.1.2"
    assert report["pipeline_version"] == crawler.PIPELINE_VERSION

    components = pd.read_parquet(tmp_path / "silver" / "components.parquet")
    assert components.loc[components.entry_id == "t1", "component_index"].tolist() == [1, 2, 3]
    observations = pd.read_parquet(tmp_path / "silver" / "observations.parquet")
    assert len(observations) == 8  # four ILThermo data points, two variables each
    density = observations.loc[(observations.entry_id == "t1") & (observations.variable_index == 2)].iloc[0]
    assert (density.raw_header, density.variable_name, density.unit, density.phase) == (
        "Density, kg/m3", "Density", "kg/m3", "Liquid",
    )
    assert (density.value_raw, density.uncertainty_raw) == ("1000", "0.2")
    assert crawler.verify(tmp_path) == ([], [], 0)

    assert crawler.crawl(tmp_path, max_per_mixture=None, attempts=2, pause_seconds=0)
    assert entry_calls == ["b1", "t1", "b2"]
    assert search_calls == [2, 3]


def test_count_mismatch_keeps_bronze_and_skips_redownload(tmp_path, monkeypatch):
    details = {"b1": response(2, 2), "b2": response(2, 1), "t1": response(3, 2)}
    _, entry_calls = install_fakes(monkeypatch, details)
    assert not crawler.crawl(tmp_path, max_per_mixture=None, attempts=2, pause_seconds=0)
    failures = [json.loads(line) for line in (tmp_path / "reports" / "failures.jsonl").read_text().splitlines()]
    assert len(failures) == 1
    assert failures[0]["entry_id"] == "t1"
    assert "data point count" in failures[0]["error"]
    assert crawler.load_json(tmp_path / "bronze" / "entries" / "t1.json") == details["t1"]
    entries = pd.read_parquet(tmp_path / "silver" / "entries.parquet")
    assert entries.loc[entries.entry_id == "t1", "validation_status"].iloc[0] == "count_mismatch"

    assert not crawler.crawl(tmp_path, max_per_mixture=None, attempts=2, pause_seconds=0)
    assert entry_calls == ["b1", "b2", "t1"]


def test_failed_download_is_logged_and_resumed(tmp_path, monkeypatch):
    details = {"b1": response(2, 2), "b2": response(2, 1), "t1": response(3, 1)}
    _, entry_calls = install_fakes(monkeypatch, details)
    good_get_entry = crawler.ilt.GetEntry

    def fail_b2(entry_id):
        if entry_id == "b2":
            entry_calls.append(entry_id)
            raise ConnectionError("temporary failure")
        return good_get_entry(entry_id)

    monkeypatch.setattr(crawler.ilt, "GetEntry", fail_b2)
    assert not crawler.crawl(tmp_path, max_per_mixture=None, attempts=1, pause_seconds=0)
    assert crawler.load_json(tmp_path / "bronze" / "failures" / "b2.json")["entry_id"] == "b2"
    report = crawler.load_json(tmp_path / "reports" / "summary.json")
    assert report["binary"]["failed_entries"] == 1
    assert report["binary"]["pending_entries"] == 0

    monkeypatch.setattr(crawler.ilt, "GetEntry", good_get_entry)
    assert crawler.crawl(tmp_path, max_per_mixture=None, attempts=1, pause_seconds=0)
    assert not (tmp_path / "bronze" / "failures" / "b2.json").exists()
    assert entry_calls.count("b1") == 1
    assert entry_calls.count("t1") == 1
    assert entry_calls.count("b2") == 2


def test_sparse_source_row_is_retained_and_reported_as_warning(tmp_path, monkeypatch):
    details = {"b1": response(2, 2), "b2": response(2, 1), "t1": response(3, 1)}
    details["t1"]["data"][0] = details["t1"]["data"][0][:1]
    install_fakes(monkeypatch, details)
    assert crawler.crawl(tmp_path, max_per_mixture=None, attempts=1, pause_seconds=0)
    report = crawler.load_json(tmp_path / "reports" / "summary.json")
    assert report["status"] == "complete"
    assert report["warning_count"] == 1
    assert (tmp_path / "reports" / "failures.jsonl").read_text() == ""
    warning = json.loads((tmp_path / "reports" / "warnings.jsonl").read_text())
    assert warning["entry_id"] == "t1"
    entries = pd.read_parquet(tmp_path / "silver" / "entries.parquet")
    assert entries.loc[entries.entry_id == "t1", "validation_status"].iloc[0] == "source_irregular"
    observations = pd.read_parquet(tmp_path / "silver" / "observations.parquet")
    assert len(observations.loc[observations.entry_id == "t1"]) == 1


def test_getentry_preserves_values_rejected_by_table_parser(monkeypatch):
    raw = response(2, 1)
    raw["data"][0][0] = None

    def fake_get_entry(_entry_id):
        return crawler.ilt_data_structs.ResponseToEntry("b1", raw)

    monkeypatch.setattr(crawler.ilt, "GetEntry", fake_get_entry)
    entry = crawler.fetch_entry("b1")
    assert entry.response["data"][0][0] is None


def test_verify_detects_silver_value_change(tmp_path, monkeypatch):
    details = {"b1": response(2, 2), "b2": response(2, 1), "t1": response(3, 1)}
    install_fakes(monkeypatch, details)
    assert crawler.crawl(tmp_path, max_per_mixture=None, attempts=1, pause_seconds=0)
    path = tmp_path / "silver" / "observations.parquet"
    observations = pd.read_parquet(path)
    observations.loc[0, "value_raw"] = "wrong"
    observations.to_parquet(path, index=False)
    failures, issues, pending = crawler.verify(tmp_path)
    assert not failures and not pending
    assert "observations: values differ from Bronze" in issues


def test_retry_uses_exponential_backoff():
    attempts = []
    delays = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("temporary")
        return "ok"

    assert crawler.retry(flaky, "test", 3, delays.append) == "ok"
    assert delays == [1, 2]


def test_atomic_write_preserves_existing_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "entry.json"
    path.write_text("old")

    def fail_replace(*_):
        raise OSError("interrupted publication")

    monkeypatch.setattr(crawler.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        crawler.atomic_bytes(path, b"new")
    assert path.read_text() == "old"
    assert list(tmp_path.iterdir()) == [path]


def test_conflicting_duplicate_id_is_rejected():
    manifests = {
        2: {"rows": [search_row("same", 2, 1)]},
        3: {"rows": [search_row("same", 3, 1)]},
    }
    with pytest.raises(ValueError, match="Conflicting Search rows"):
        crawler.unique_entries(manifests)
