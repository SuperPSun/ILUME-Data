"""Archive pinned public experimental datasets (ECW is archive-only)."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
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
    snapshot = SNAPSHOTS.get(source, {})
    manifest = {"source": source, **{k: v for k, v in snapshot.items() if k != "files"},
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "files": []}
    with tempfile.TemporaryDirectory(dir=raw_root) as temporary_dir:
        staged = Path(temporary_dir) / source
        staged.mkdir()
        paths = snapshot.get("files", ["d3ta04310j1.pdf"])
        for path in paths:
            filename = Path(path).name
            url = (ECW_URL if source == "ECW" else
                   f"https://raw.githubusercontent.com/{snapshot['repository']}/"
                   f"{snapshot['commit']}/{quote(path)}")
            entry = {"filename": filename, "repository_path": path, "url": url}
            try:
                payload = fetch_bytes(url)
                if source == "ECW":
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
    parser.add_argument("--sources", nargs="+", choices=[*SNAPSHOTS, "ECW"],
                        default=list(SNAPSHOTS))
    args = parser.parse_args()
    for source in args.sources:
        manifest = crawl_source(source, args.raw_root)
        print(f"{source}: {len(manifest['files'])} archived files")


if __name__ == "__main__":
    main()
