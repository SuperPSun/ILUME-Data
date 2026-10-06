"""Archive pinned public experimental datasets (ECW is archive-only)."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAS_SMILES = {
    "CO2": "O=C=O", "C2H6": "CC", "CH4": "C", "H2S": "S",
    "H2": "[H][H]", "N2O": "N#[N+][O-]", "N2": "N#N",
    "NH3": "N", "O2": "O=O", "SO2": "O=S=O",
}
SNAPSHOTS = {
    "IL4GAS": {
        "repository": "Yu-Xin-Qiu/IL4GAS",
        "commit": "a3a16fd0c2a186119efe231b570aa9ebbc38d6c8",
        "files": [f"deep learning/data/unique_{gas}_data.csv" for gas in GAS_SMILES],
    },
    "WaterActivity": {
        "repository": "MohanMood/NLP_Ionic-Liquid_Properties",
        "commit": "ba9768dbdd1c8f2dd90c091b27babc9c22db0c0c",
        "files": ["Dataset/Dataset-IL-Water-Activity_SMILES_Atom-Count.csv"],
    },
}
EXPERIMENT_ARCHIVES = {
    "Lethesh2022": {
        "doi": "10.3389/fchem.2022.859304",
        "files": {
            "article.html": "https://www.frontiersin.org/journals/chemistry/articles/10.3389/fchem.2022.859304/full",
            "article.xml": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9247390/fullTextXML",
            "supplementary.zip": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9247390/supplementaryFiles",
        },
    },
    "Toots2025_QDB266": {
        "doi": "10.15152/QDB.266",
        "article_doi": "10.1016/j.molliq.2025.128184",
        "files": {
            "article4.zip": "https://qsardb.org/repository/bitstream/10967/266/1/article4.zip",
            "repository.html": "https://qsardb.org/repository/handle/10967/266",
            "toots_thesis.pdf": "https://dspace.ut.ee/bitstreams/de05b3cf-5f1d-4b8c-a7ea-36982e236ebb/download",
        },
    },
}
PUBLIC_SOURCES = (*SNAPSHOTS, *EXPERIMENT_ARCHIVES)
ECW_URL = "https://www.rsc.org/suppdata/d3/ta/d3ta04310j/d3ta04310j1.pdf"
ECW_GAP = (
    "Training task deferred: SI Table S1 contains only 14 values with mixed "
    "experimental/AIMD provenance, not the complete 50-label training table. "
    "The 660 ML predictions are not ground truth. No ECW labels are published."
)


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "ILUME-Data/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def required_columns(source: str, filename: str) -> set[str]:
    if source == "IL4GAS":
        gas = filename.removeprefix("unique_").removesuffix("_data.csv")
        return {"T/K", "P/bar", f"x_{gas}", "IL_SMILES"}
    return {"Mole_frac_water", "Temperature", "Pressure", "Activityt_water",
            "Cation_SMILES", "Anion_SMILES", "Ref"}


def crawl_source(source: str, raw_root: Path) -> dict:
    """Publish a complete source snapshot, preserving the old one on failure."""
    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    snapshot = SNAPSHOTS.get(source, EXPERIMENT_ARCHIVES.get(source, {}))
    manifest = {"source": source, **{k: v for k, v in snapshot.items() if k != "files"},
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "files": []}
    with tempfile.TemporaryDirectory(dir=raw_root) as temporary_dir:
        staged = Path(temporary_dir) / source
        staged.mkdir()
        paths = snapshot.get("files", ["d3ta04310j1.pdf"])
        for path in paths:
            filename = Path(path).name
            url = (paths[path] if isinstance(paths, dict) else ECW_URL if source == "ECW" else
                   f"https://raw.githubusercontent.com/{snapshot['repository']}/"
                   f"{snapshot['commit']}/{quote(path)}")
            entry = {"filename": filename, "repository_path": path, "url": url}
            try:
                payload = fetch_bytes(url)
                if source in EXPERIMENT_ARCHIVES:
                    if filename.endswith(".zip"):
                        archive = zipfile.ZipFile(io.BytesIO(payload))
                        if source == "Toots2025_QDB266":
                            values = pd.read_csv(io.BytesIO(archive.read("properties/logK/values")), sep="\t")
                            entry["rows"] = len(values)
                        elif not any(name.lower().endswith(".pdf") for name in archive.namelist()):
                            raise ValueError("Supplementary archive has no PDF")
                    elif filename == "article.xml":
                        article = ET.fromstring(payload)
                        tables = [w for w in article.iter("table-wrap") if w.findtext("label", "").strip().upper() == "TABLE 1"]
                        if len(tables) != 1:
                            raise ValueError("Lethesh snapshot has no unique Table 1")
                        table = tables[0]
                        temperatures = re.findall(r"\d+\.\d+\s*K", " ".join(table.find(".//thead/tr").itertext()))
                        entry["rows"] = len(table.findall(".//tbody/tr")) * len(temperatures)
                    elif filename.endswith(".pdf") and not payload.startswith(b"%PDF"):
                        raise ValueError("Response is not a PDF")
                elif source == "ECW":
                    if not payload.startswith(b"%PDF"):
                        raise ValueError("ECW response is not a PDF")
                else:
                    frame = pd.read_csv(io.BytesIO(payload))
                    missing = required_columns(source, filename) - set(frame)
                    if missing or frame.empty:
                        raise ValueError(f"{filename}: missing={sorted(missing)}, rows={len(frame)}")
                    entry["rows"] = len(frame)
                (staged / filename).write_bytes(payload)
                entry.update(status="downloaded", sha256=hashlib.sha256(payload).hexdigest(),
                             bytes=len(payload))
            except Exception as error:
                if source != "ECW":
                    raise
                entry.update(status="unavailable", error=str(error))
                # Keep a previously downloaded SI when the archive endpoint is unavailable.
                previous = raw_root / source / filename
                if previous.is_file():
                    shutil.copy2(previous, staged / filename)
                    entry["retained_sha256"] = hashlib.sha256(previous.read_bytes()).hexdigest()
            manifest["files"].append(entry)
        if source == "ECW":
            manifest.update(doi="10.1039/D3TA04310J", training_status="deferred", gap=ECW_GAP)
            (staged / "gap.md").write_text(ECW_GAP + "\n", encoding="utf-8")
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        destination = raw_root / source
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staged, destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--sources", nargs="+", choices=[*PUBLIC_SOURCES, "ECW"],
                        default=list(SNAPSHOTS))
    args = parser.parse_args()
    for source in args.sources:
        manifest = crawl_source(source, args.raw_root)
        print(f"{source}: {len(manifest['files'])} archived files")


if __name__ == "__main__":
    main()
