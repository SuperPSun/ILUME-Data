import hashlib
import io
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import pytest
from rdkit import Chem

from scripts import crawl_public_properties as crawl
from scripts.analyze_final_properties import analyze_final_properties
from scripts.clean_structured_data import clean_non_ilthermo_structured
from scripts.merge_data import merge_data
from scripts.structure_public_properties import structure_source

CATION = "CC[n+]1ccn(C)c1"
ANION = "F[B-](F)(F)F"


def public_payload(url: str) -> bytes:
    filename = unquote(url.rsplit("/", 1)[1])
    if filename.startswith("unique_"):
        gas = filename.removeprefix("unique_").removesuffix("_data.csv")
        rows = [{"T/K": 298.15, "P/bar": 1.01325, f"x_{gas}": .1,
                 "IL_SMILES": f"{ANION}.{CATION}", "data_index": 17, "IL_index": 4,
                 f"ln(x_{gas})": -2.302585}]
    else:
        rows = [{"Sl-No": index, "Mole_frac_water": concentration, "Temperature": 298.15,
                 "Pressure": 100.0, "Activityt_water": gamma, "Ref": "original paper",
                 "Cation_SMILES": CATION, "Anion_SMILES": ANION}
                for index, (concentration, gamma) in enumerate(((0., .0546), (.2, 2.), (1., 9.36)))]
    return pd.DataFrame(rows).to_csv(index=False).encode()


def archive(tmp_path, monkeypatch, source):
    monkeypatch.setattr(crawl, "fetch_bytes", public_payload)
    return crawl.crawl_source(source, tmp_path / "raw")


def test_snapshot_contains_only_experimental_files_and_verifiable_provenance(tmp_path, monkeypatch):
    manifest = archive(tmp_path, monkeypatch, "IL4GAS")
    assert len(manifest["files"]) == 10
    assert {entry["filename"] for entry in manifest["files"]} == {
        f"unique_{gas}_data.csv" for gas in crawl.GAS_SMILES
    }
    assert manifest["commit"] == crawl.SNAPSHOTS["IL4GAS"]["commit"]
    assert manifest["downloaded_at_utc"]
    for entry in manifest["files"]:
        payload = (tmp_path / "raw" / "IL4GAS" / entry["filename"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
        assert entry["rows"] == 1
        assert manifest["commit"] in entry["url"]


@pytest.mark.parametrize("failure", ("network", "schema", "empty"))
def test_failed_download_preserves_complete_previous_source(tmp_path, monkeypatch, failure):
    archive(tmp_path, monkeypatch, "IL4GAS")
    before = {path.name: path.read_bytes() for path in (tmp_path / "raw/IL4GAS").iterdir()}

    def fail(url):
        if "unique_CH4" in url:
            if failure == "network":
                raise OSError("interrupted download")
            if failure == "schema":
                return b"wrong\n1\n"
            return public_payload(url).split(b"\n")[0] + b"\n"
        return public_payload(url)

    monkeypatch.setattr(crawl, "fetch_bytes", fail)
    with pytest.raises((OSError, ValueError)):
        crawl.crawl_source("IL4GAS", tmp_path / "raw")
    assert before == {path.name: path.read_bytes() for path in (tmp_path / "raw/IL4GAS").iterdir()}


def test_all_gases_roles_pressure_conversion_and_legacy_co2_merge(tmp_path, monkeypatch):
    archive(tmp_path, monkeypatch, "IL4GAS")
    structured = structure_source("IL4GAS", tmp_path / "raw", tmp_path / "structured")
    assert len(structured) == 10
    assert structured["cation"].eq(CATION).all()
    assert structured["anion"].eq(ANION).all()
    assert structured["pressure_kPa"].tolist() == pytest.approx([101.325] * 10)
    assert structured["x_gas_unitless"].eq(.1).all()
    assert set(structured["solute"]) == set(crawl.GAS_SMILES.values())
    assert all(Chem.MolFromSmiles(smiles) is not None for smiles in structured["solute"])
    clean_non_ilthermo_structured(tmp_path / "structured", tmp_path / "cleaned")
    legacy = tmp_path / "cleaned/AIonopedia/x_CO2_structured.csv"
    legacy.parent.mkdir()
    pd.DataFrame([{"cation": CATION, "anion": ANION, "temperature_K": 298.15,
                   "pressure_kPa": 101.325, "x_CO2_unitless": .1}]).to_csv(legacy, index=False)
    merge_data(tmp_path / "cleaned", tmp_path / "merged")
    output = pd.read_csv(tmp_path / "merged/experiment/gas_solubility.csv")
    assert list(output) == ["cation", "anion", "solute", "temperature_K", "pressure_kPa",
                            "x_gas_unitless", "source_list"]
    assert len(output) == 10
    assert output.loc[output.solute.eq("O=C=O"), "source_list"].item() == "AIonopedia; IL4GAS"
    assert not (tmp_path / "merged/experiment/x_co2.csv").exists()
    trace = pd.read_csv(tmp_path / "merged/_audit/public_properties/IL4GAS/gas_solubility_structured_provenance.csv")
    assert trace["raw_row_number"].eq(2).all()
    assert trace["status"].eq("accepted").all()
    assert trace["data_index"].eq(17).all()
    legacy_trace = pd.read_csv(tmp_path / "merged/_audit/gas_solubility_legacy_co2_conversion.csv")
    assert legacy_trace["source_row_number"].tolist() == [2]
    assert legacy_trace["solute"].tolist() == ["O=C=O"]
    summary_root = tmp_path / "analysis"
    analyze_final_properties(tmp_path / "merged", summary_root, skip_plots=True)
    summary = pd.read_csv(summary_root / "gas_solubility_by_gas.csv")
    assert len(summary) == 10
    sources = pd.read_csv(summary_root / "gas_solubility_by_gas_source.csv")
    assert sources.loc[sources.solute.eq("O=C=O"), "source"].tolist() == ["AIonopedia", "IL4GAS"]


def test_water_gamma_is_direct_and_concentration_endpoints_remain_distinct(tmp_path, monkeypatch):
    archive(tmp_path, monkeypatch, "WaterActivity")
    structure_source("WaterActivity", tmp_path / "raw", tmp_path / "structured")
    clean_non_ilthermo_structured(tmp_path / "structured", tmp_path / "cleaned")
    merge_data(tmp_path / "cleaned", tmp_path / "merged")
    output = pd.read_csv(tmp_path / "merged/experiment/water_activity_coefficient.csv")
    assert list(output) == ["cation", "anion", "x_water_unitless", "temperature_K", "pressure_kPa",
                            "water_activity_coefficient_unitless", "source_list"]
    assert output["x_water_unitless"].tolist() == [0., .2, 1.]
    assert output["water_activity_coefficient_unitless"].tolist() == [.0546, 2., 9.36]
    assert output["pressure_kPa"].eq(100.).all()
    assert "Ref" not in output
    audit = tmp_path / "merged/_audit/public_properties/WaterActivity"
    trace = pd.read_csv(audit / "water_activity_coefficient_structured_provenance.csv")
    assert trace["reference"].eq("original paper").all()
    summary = analyze_final_properties(tmp_path / "merged", tmp_path / "analysis", skip_plots=True)
    assert summary["property_label"].tolist() == ["water_activity_coefficient_unitless"]
    assert "x_water_unitless" in summary["condition_columns"].item()


def test_new_sources_reuse_existing_equivalent_identity_representatives(tmp_path):
    old = "C[NH+]1C=CN=C1"
    new = "C[N@H+]1C=CN=C1"
    for source, row in (
        ("ILBERT", {"cation": old, "anion": ANION, "density_g/cm^3": 1.2}),
        ("WaterActivity", {"cation": new, "anion": ANION, "x_water_unitless": .5,
                           "temperature_K": 298.15, "pressure_kPa": 100.,
                           "water_activity_coefficient_unitless": 2.}),
    ):
        directory = tmp_path / "cleaned" / source
        directory.mkdir(parents=True)
        pd.DataFrame([row]).to_csv(directory / "property_structured.csv", index=False)
    merge_data(tmp_path / "cleaned", tmp_path / "merged")
    density = pd.read_csv(tmp_path / "merged/experiment/density.csv")
    water = pd.read_csv(tmp_path / "merged/experiment/water_activity_coefficient.csv")
    assert density["cation"].tolist() == [old]
    assert water["cation"].tolist() == [old]
    audit = pd.read_csv(tmp_path / "merged/chemical_identity_equivalences.csv")
    assert audit.loc[audit.representative_smiles.eq(old), "equivalent_smiles_list"].str.contains(new, regex=False).all()


@pytest.mark.parametrize(("column", "value"), (
    ("Temperature", 0), ("Pressure", -1), ("Pressure", "bad"),
    ("Mole_frac_water", -.1), ("Mole_frac_water", 1.1), ("Mole_frac_water", None),
    ("Activityt_water", 0), ("Activityt_water", -1),
    ("Activityt_water", float("inf")), ("Activityt_water", "bad"),
))
def test_numeric_rejections_keep_raw_provenance(tmp_path, monkeypatch, column, value):
    def payload(url):
        frame = pd.read_csv(io.BytesIO(public_payload(url)))
        frame[column] = frame[column].astype(object)
        frame.loc[0, column] = value
        return frame.to_csv(index=False).encode()
    monkeypatch.setattr(crawl, "fetch_bytes", payload)
    crawl.crawl_source("WaterActivity", tmp_path / "raw")
    structure_source("WaterActivity", tmp_path / "raw", tmp_path / "structured")
    reports = clean_non_ilthermo_structured(tmp_path / "structured", tmp_path / "cleaned")
    assert reports[0].output_rows == 2
    rejected = pd.read_csv(reports[0].rejected_path)
    assert rejected["raw_row_number"].eq(2).all()
    assert rejected["reference"].eq("original paper").all()
    assert rejected["source_commit"].eq(crawl.SNAPSHOTS["WaterActivity"]["commit"]).all()


def test_invalid_ion_and_modified_snapshot_are_audited_or_rejected(tmp_path, monkeypatch):
    def payload(url):
        frame = pd.read_csv(io.BytesIO(public_payload(url)))
        frame.loc[0, "Anion_SMILES"] = "CCO"
        return frame.to_csv(index=False).encode()
    monkeypatch.setattr(crawl, "fetch_bytes", payload)
    crawl.crawl_source("WaterActivity", tmp_path / "raw")
    output = structure_source("WaterActivity", tmp_path / "raw", tmp_path / "structured")
    assert len(output) == 2
    audit = pd.read_csv(tmp_path / "structured/WaterActivity/_audit/structure_rejected.csv")
    assert audit["raw_row_number"].tolist() == [2]
    path = tmp_path / "raw/WaterActivity" / Path(crawl.SNAPSHOTS["WaterActivity"]["files"][0]).name
    path.write_bytes(path.read_bytes() + b"\n")
    before = (tmp_path / "structured/WaterActivity/water_activity_coefficient_structured.csv").read_bytes()
    with pytest.raises(ValueError, match="hash mismatch"):
        structure_source("WaterActivity", tmp_path / "raw", tmp_path / "structured")
    assert (tmp_path / "structured/WaterActivity/water_activity_coefficient_structured.csv").read_bytes() == before


@pytest.mark.parametrize("value", (-.1, 1.1, float("inf"), None))
def test_invalid_gas_mole_fractions_are_rejected(tmp_path, monkeypatch, value):
    def payload(url):
        frame = pd.read_csv(io.BytesIO(public_payload(url)))
        if "unique_CH4" in url:
            frame["x_CH4"] = value
        return frame.to_csv(index=False).encode()
    monkeypatch.setattr(crawl, "fetch_bytes", payload)
    crawl.crawl_source("IL4GAS", tmp_path / "raw")
    structure_source("IL4GAS", tmp_path / "raw", tmp_path / "structured")
    report = clean_non_ilthermo_structured(tmp_path / "structured", tmp_path / "cleaned")[0]
    assert report.output_rows == 9
    rejected = pd.read_csv(report.rejected_path)
    assert rejected["gas"].eq("CH4").all()


def test_ecw_archive_failure_retains_si_and_does_not_publish_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(crawl, "fetch_bytes", lambda url: b"%PDF-test")
    crawl.crawl_source("ECW", tmp_path / "raw")
    def unavailable(url):
        raise OSError("HTTP 403")
    monkeypatch.setattr(crawl, "fetch_bytes", unavailable)
    manifest = crawl.crawl_source("ECW", tmp_path / "raw")
    assert manifest["training_status"] == "deferred"
    assert manifest["files"][0]["status"] == "unavailable"
    assert (tmp_path / "raw/ECW/d3ta04310j1.pdf").read_bytes() == b"%PDF-test"
    assert (tmp_path / "raw/ECW/gap.md").is_file()
    assert not list((tmp_path / "raw/ECW").glob("*.csv"))
