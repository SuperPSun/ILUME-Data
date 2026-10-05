"""Structure the archived IL4GAS and water activity experimental snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from raw_prep import net_formal_charge, parse_ion_pair_identity  # noqa: E402
from crawl_public_properties import GAS_SMILES, SNAPSHOTS, required_columns  # noqa: E402


def structure_source(source: str, raw_root: Path, output_root: Path) -> pd.DataFrame:
    source_root = Path(raw_root) / source
    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    expected = {Path(path).name for path in SNAPSHOTS[source]["files"]}
    entries = {entry["filename"]: entry for entry in manifest["files"]}
    if set(entries) != expected:
        raise ValueError(f"{source}: incomplete snapshot manifest")
    rows, provenance, rejected = [], [], []
    for filename in sorted(expected):
        entry = entries[filename]
        path = source_root / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"{path}: snapshot hash mismatch")
        frame = pd.read_csv(path)
        missing = required_columns(source, filename) - set(frame)
        if missing:
            raise ValueError(f"{filename}: missing columns {sorted(missing)}")
        for index, row in frame.iterrows():
            trace = {"raw_file": filename, "raw_row_number": index + 2,
                     "source_url": entry["url"], "source_commit": manifest["commit"],
                     "source_sha256": entry["sha256"]}
            if source == "IL4GAS":
                gas = filename.removeprefix("unique_").removesuffix("_data.csv")
                identity = parse_ion_pair_identity(str(row["IL_SMILES"]))
                cation, anion = identity.cation, identity.anion
                trace.update(gas=gas, raw_IL_SMILES=row["IL_SMILES"],
                             data_index=row.get("data_index", ""),
                             IL_index=row.get("IL_index", ""),
                             conversion="P/bar * 100 -> pressure_kPa; raw x retained")
                record = {"cation": cation, "anion": anion, "solute": GAS_SMILES[gas],
                          "temperature_K": pd.to_numeric(row["T/K"], errors="coerce"),
                          "pressure_kPa": pd.to_numeric(row["P/bar"], errors="coerce") * 100,
                          "x_gas_unitless": pd.to_numeric(row[f"x_{gas}"], errors="coerce")}
            else:
                cation, anion = row["Cation_SMILES"], row["Anion_SMILES"]
                trace.update(reference=row["Ref"], source_record_id=row.get("Sl-No", ""),
                             raw_cation=cation, raw_anion=anion,
                             conversion="Activityt_water -> gamma directly; pressure already kPa")
                record = {"cation": cation, "anion": anion,
                          "x_water_unitless": pd.to_numeric(row["Mole_frac_water"], errors="coerce"),
                          "temperature_K": pd.to_numeric(row["Temperature"], errors="coerce"),
                          "pressure_kPa": pd.to_numeric(row["Pressure"], errors="coerce"),
                          "water_activity_coefficient_unitless": pd.to_numeric(row["Activityt_water"], errors="coerce")}
            cation_charge = net_formal_charge(str(cation)) if pd.notna(cation) else None
            anion_charge = net_formal_charge(str(anion)) if pd.notna(anion) else None
            if cation_charge is None or anion_charge is None or cation_charge <= 0 or anion_charge >= 0:
                rejected.append({**trace, "rejection_reason": "invalid_ion_pair"})
                continue
            trace.update(structured_row_index=len(rows), **record)
            rows.append(record)
            provenance.append(trace)
    columns = (["cation", "anion", "solute", "temperature_K", "pressure_kPa", "x_gas_unitless"]
               if source == "IL4GAS" else
               ["cation", "anion", "x_water_unitless", "temperature_K", "pressure_kPa",
                "water_activity_coefficient_unitless"])
    output = pd.DataFrame(rows, columns=columns)
    stem = "gas_solubility" if source == "IL4GAS" else "water_activity_coefficient"
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_root) as temporary_dir:
        staged = Path(temporary_dir) / source
        audit = staged / "_audit"
        audit.mkdir(parents=True)
        output.to_csv(staged / f"{stem}_structured.csv", index=False)
        pd.DataFrame(provenance, columns=["structured_row_index"] if not provenance else None).to_csv(
            audit / f"{stem}_structured_provenance.csv", index=False)
        pd.DataFrame(rejected, columns=["rejection_reason"] if not rejected else None).to_csv(
            audit / "structure_rejected.csv", index=False)
        shutil.copy2(source_root / "manifest.json", audit / "manifest.json")
        destination = output_root / source
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staged, destination)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/structured")
    parser.add_argument("--sources", nargs="+", choices=list(SNAPSHOTS), default=list(SNAPSHOTS))
    args = parser.parse_args()
    for source in args.sources:
        output = structure_source(source, args.raw_root, args.output_root)
        print(f"{source}: {len(output)} structured rows")


if __name__ == "__main__":
    main()
