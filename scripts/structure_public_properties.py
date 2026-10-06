"""Structure archived public experimental properties and partition audits."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from decimal import Decimal
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from raw_prep import net_formal_charge, parse_ion_pair_identity  # noqa: E402
from crawl_public_properties import GAS_SMILES, SNAPSHOTS, EXPERIMENT_ARCHIVES, PUBLIC_SOURCES, required_columns  # noqa: E402
from merge_data import canonicalize_identity_smiles, fixed_h_inchikey  # noqa: E402


def verified_archive(source: str, raw_root: Path) -> tuple[Path, dict]:
    root = Path(raw_root) / source
    manifest = json.loads((root / "manifest.json").read_text())
    expected = set(EXPERIMENT_ARCHIVES[source]["files"])
    entries = {entry["filename"]: entry for entry in manifest["files"]}
    if len(entries) != len(manifest["files"]) or set(entries) != expected:
        raise ValueError(f"{source}: incomplete snapshot manifest")
    for name, entry in entries.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"{source}/{name}: snapshot hash mismatch")
    return root, manifest


def normalize_il_name(name: str) -> str:
    return re.sub(r"[\s_–−-]", "", name).lower()


def parse_lethesh_table(payload: bytes) -> tuple[list[dict], list[dict]]:
    """Read the measured Table 1, using the methods text as the temperature gate."""
    root = ET.fromstring(payload)
    text = " ".join(root.itertext())
    method = re.search(r"CV scans were performed at (.*?)in this study", text, re.S)
    if not method:
        raise ValueError("Lethesh: measured-temperature method statement missing")
    measured = {Decimal(x) for x in re.findall(r"\d+\.\d+", method[1])}
    tables = [w for w in root.iter("table-wrap") if w.findtext("label", "").strip().upper() == "TABLE 1"]
    if len(tables) != 1:
        raise ValueError("Lethesh: expected a unique Table 1")
    table = tables[0]
    headers = table.findall(".//thead/tr")
    temperatures = [Decimal(x) for x in re.findall(r"(\d+\.\d+)\s*K", " ".join(headers[0].itertext()))]
    labels = [re.sub(r"\s", "", "".join(c.itertext())) for c in headers[1]]
    if not temperatures or set(temperatures) != measured or labels != ["Ec", "Ea", "ESW"] * len(temperatures):
        raise ValueError("Lethesh: temperature or Ec/Ea/ESW header mismatch")
    records, rejected = [], []
    for row_number, row in enumerate(table.findall(".//tbody/tr"), 1):
        cells = ["".join(c.itertext()).strip() for c in row]
        if len(cells) != 2 + 3 * len(temperatures):
            raise ValueError(f"Lethesh Table 1 row {row_number}: unexpected column count")
        for offset, temperature in enumerate(temperatures):
            trace = {"table": "Table 1", "entry": cells[0], "raw_IL_name": cells[1],
                     "raw_row_number": row_number, "temperature_K": float(temperature)}
            raw = cells[2 + offset * 3:5 + offset * 3]
            trace.update(raw_Ec=raw[0], raw_Ea=raw[1], raw_ESW=raw[2])
            try:
                ec, ea, esw = [Decimal(v.replace("−", "-")) for v in raw]
                if not all(v.is_finite() for v in (ec, ea, esw)):
                    raise ValueError("nonfinite potential")
            except (ValueError, ArithmeticError):
                rejected.append({**trace, "rejection_reason": "invalid_potential"})
                continue
            delta = ea - ec - esw
            records.append({**trace, "cathodic_potential_limit_V": float(ec),
                            "anodic_potential_limit_V": float(ea), "reported_ESW_V": float(esw),
                            "derived_ESW_V": float(ea - ec), "ESW_difference_V": float(delta),
                            "ESW_status": "exact" if delta == 0 else
                                "within_rounding" if abs(delta) <= Decimal("0.15") else "inconsistent"})
    return records, rejected


def structure_lethesh(root: Path, manifest: dict, staged: Path) -> pd.DataFrame:
    mapping_path = ROOT / "configs/lethesh2022_ion_map.json"
    mapping = json.loads(mapping_path.read_text())
    supplementary = zipfile.ZipFile(root / "supplementary.zip")
    pdf = supplementary.read("DataSheet1.pdf")
    if hashlib.sha256(pdf).hexdigest() != mapping["supplement_pdf_sha256"]:
        raise ValueError("Lethesh: supplementary identity evidence changed; re-review mapping")
    (staged / "_audit" / "DataSheet1.pdf").write_bytes(pdf)
    records, rejected = parse_lethesh_table((root / "article.xml").read_bytes())
    entries = {e["filename"]: e for e in manifest["files"]}
    rows, traces, identities = [], [], []
    for record in records:
        name = normalize_il_name(record["raw_IL_name"])
        identity = mapping["ionic_liquids"].get(name)
        trace = {**record, "source_doi": manifest["doi"], "source_url": entries["article.xml"]["url"],
                 "source_sha256": entries["article.xml"]["sha256"], "mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
                 "water_content": "<20 ppm (supplier specification)", "pressure_description": "standard pressure; numeric pressure not specified",
                 "purity_description": "99.9% in Table S1; EMim/FSI absent from S1, purity not individually specified" if name == "[emim][fsi]" else "99.9% (Table S1)",
                 "replicate_count": 2, "selected_CV_cycle": 4,
                 "reference_description": "AgCl-coated Ag wire, Ag/Ag+ quasi-reference; no Fc or SHE conversion",
                 "conversion": "published Ea/Ec retained; ESW derived as Ea-Ec"}
        if identity is None:
            rejected.append({**trace, "rejection_reason": "unresolved_IL_name",
                             "identity_issue": mapping.get("unresolved", {}).get(name, "No verified mapping")})
            continue
        cation = canonicalize_identity_smiles(identity["cation"])
        anion = canonicalize_identity_smiles(identity["anion"])
        if net_formal_charge(cation) != 1 or net_formal_charge(anion) != -1:
            raise ValueError(f"Invalid ion mapping for {name}")
        conditions = {"reference_electrode": "Ag/Ag+ quasi-reference (AgCl-coated Ag)",
                      "working_electrode": "Pt wire", "scan_rate_mV/s": 50.0}
        output = {"cation": cation, "anion": anion, "temperature_K": record["temperature_K"],
                  **conditions, "anodic_potential_limit_V": record["anodic_potential_limit_V"],
                  "cathodic_potential_limit_V": record["cathodic_potential_limit_V"]}
        traces.append({**trace, **output, "structured_row_index": len(rows), "mapping_evidence": identity["evidence"],
                       "source_conflict": identity.get("source_conflict", ""),
                       "identity_resolution": identity.get("identity_resolution", "")})
        identities.append({"raw_IL_name": record["raw_IL_name"], "cation": cation, "anion": anion,
                           "evidence": identity["evidence"]})
        rows.append(output)
    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("Lethesh: no resolvable experimental rows")
    output.to_csv(staged / "electrochemical_limits_structured.csv", index=False)
    audit = staged / "_audit"
    pd.DataFrame(traces).to_csv(audit / "electrochemical_limits_structured_provenance.csv", index=False)
    pd.DataFrame(records).to_csv(audit / "ESW_consistency.csv", index=False)
    pd.DataFrame(identities).drop_duplicates().to_csv(audit / "IL_structure_mapping.csv", index=False)
    pd.DataFrame(rejected, columns=None if rejected else ["rejection_reason"]).to_csv(audit / "structure_rejected.csv", index=False)
    summary = {"parsed_conditions": len(records), "published_conditions": len(output),
               "unique_ILs": output[["cation", "anion"]].drop_duplicates().shape[0],
               "complete_Ea_Ec": int(output[["anodic_potential_limit_V", "cathodic_potential_limit_V"]].notna().all(axis=1).sum()),
               "ESW_status_counts": pd.Series([r["ESW_status"] for r in records]).value_counts().to_dict(),
               "rejected_conditions": len(rejected), "mapping_file": str(mapping_path),
               "identity_policy": mapping.get("identity_policy", "verified unambiguous source identities only"),
               "source_conflict_conditions": sum(bool(row["source_conflict"]) for row in traces)}
    (audit / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output


def logk_to_solvation(logk: float, temperature: float, concentration_ratio: float = 1.0) -> float:
    if not all(math.isfinite(v) for v in (logk, temperature, concentration_ratio)) or temperature <= 0 or concentration_ratio <= 0:
        raise ValueError("Invalid logK conversion inputs")
    return 8.31446261815324 * temperature / 4184 * (-math.log(10) * logk + math.log(concentration_ratio))


def parse_qdb(payload: bytes) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    archive = zipfile.ZipFile(io.BytesIO(payload))
    registry = ET.fromstring(archive.read("compounds/compounds.xml"))
    ns = {"q": "http://www.qsardb.org/QDB"}
    values = pd.read_csv(io.BytesIO(archive.read("properties/logK/values")), sep="\t", dtype=str)
    if list(values.columns) != ["Compound Id", "Gas-ionic liquid partition coefficient"]:
        raise ValueError("QDB: unexpected experimental property header")
    archive_sha = hashlib.sha256(payload).hexdigest()
    compounds, duplicate_registry_ids, rejected = {}, set(), []
    for registry_row, node in enumerate(registry, 1):
        id_ = node.findtext("q:Id", namespaces=ns)
        if not id_:
            rejected.append({"registry_member": "compounds/compounds.xml", "registry_row_number": registry_row,
                             "source_doi": "10.15152/QDB.266", "source_sha256": archive_sha,
                             "raw_name": node.findtext("q:Name", default="", namespaces=ns),
                             "rejection_reason": "missing_registry_ID"})
            continue
        if id_ in compounds:
            duplicate_registry_ids.add(id_)
        compounds[id_] = node
    duplicate_ids = set(values.loc[values.iloc[:, 0].duplicated(False)].iloc[:, 0])
    rows = []
    for index, (id_, value) in enumerate(values.itertuples(index=False, name=None), 2):
        member = f"compounds/{id_}/daylight-smiles"
        trace = {"compound_id": id_, "raw_row_number": index, "property_member": "properties/logK/values",
                 "source_doi": "10.15152/QDB.266", "article_doi": "10.1016/j.molliq.2025.128184",
                 "source_url": EXPERIMENT_ARCHIVES["Toots2025_QDB266"]["files"]["article4.zip"],
                 "temperature_origin": "author thesis Chapter 2: compilation at 298.15 K; archive has no per-row temperature",
                 "candidate_conversion": "-R*T*ln(10)*logK/4184; equal concentration standard states assumed for audit only",
                 "source_sha256": archive_sha,
                 "structure_member": member, "raw_logK": value}
        try:
            if id_ in duplicate_ids:
                raise ValueError("duplicate_property_ID")
            if id_ in duplicate_registry_ids:
                raise ValueError("duplicate_registry_ID")
            node = compounds.get(id_)
            if node is None:
                raise ValueError("missing_compound_ID")
            raw_smiles = archive.read(member).decode().strip()
            inchi = node.findtext("q:InChI", default="", namespaces=ns)
            trace.update(raw_SMILES=raw_smiles, raw_InChI=inchi,
                         raw_name=node.findtext("q:Name", default="", namespaces=ns),
                         raw_description=node.findtext("q:Description", default="", namespaces=ns))
            molecule = Chem.MolFromSmiles(raw_smiles)
            if molecule is None:
                raise ValueError("invalid_SMILES")
            if not inchi or Chem.MolToInchiKey(molecule) != Chem.InchiToInchiKey(inchi):
                raise ValueError("SMILES_InChI_conflict")
            roles = {r: [] for r in ("cation", "anion", "solute")}
            for fragment in Chem.GetMolFrags(molecule, asMols=True):
                charge = Chem.GetFormalCharge(fragment)
                role = "cation" if charge > 0 else "anion" if charge < 0 else "solute"
                roles[role].append(canonicalize_identity_smiles(Chem.MolToSmiles(fragment)))
            if any(len(v) != 1 for v in roles.values()):
                raise ValueError("ambiguous_component_roles")
            logk = float(value)
            record = {**trace, **{role: parts[0] for role, parts in roles.items()}, "logK": logk,
                      "temperature_K": 298.15, "candidate_solvation_kcal/mol": logk_to_solvation(logk, 298.15)}
            rows.append(record)
        except (ValueError, KeyError, UnicodeError) as error:
            rejected.append({**trace, "rejection_reason": str(error)})
    return pd.DataFrame(rows), pd.DataFrame(rejected, columns=None if rejected else ["rejection_reason"]), len(values)


def residual_stats(values: pd.Series) -> dict:
    values = values.dropna()
    if values.empty:
        return {"n": 0}
    return {"n": len(values), "bias": float(values.mean()), "median": float(values.median()),
            "MAE": float(values.abs().mean()), "RMSE": float((values.pow(2).mean()) ** 0.5),
            "max_absolute": float(values.abs().max()),
            "quantiles": {str(p): float(values.quantile(p)) for p in (0, .05, .25, .5, .75, .95, 1)}}


def qdb_merge_evidence(evidence: dict | None, archive_sha: str, baseline_sha: str, candidates: pd.DataFrame) -> tuple[bool, list[str]]:
    if evidence is None:
        return False, ["Existing solvation standard state not documented; candidate is equal-concentration only",
                       "Base-10 and infinite-dilution convention need source confirmation for compatibility",
                       "Overlap numerical anomalies require source-level resolution", "No reviewed compatibility evidence"]
    required = {"approved": True, "archive_sha256": archive_sha, "baseline_sha256": baseline_sha,
                "log_base": 10, "partition_definition": "c_IL/c_gas", "temperature_K": 298.15,
                "infinite_dilution": True,
                "unresolved_issues": []}
    if any(evidence.get(k) != v for k, v in required.items()):
        return False, ["Compatibility evidence missing, unresolved, or bound to different inputs"]
    if not all(evidence.get(k) for k in ("reviewer", "existing_standard_state_evidence", "qdb_definition_evidence", "anomaly_resolution")):
        return False, ["Physical definitions and anomaly resolution need reviewed citations"]
    ratio = evidence.get("liquid_to_gas_standard_concentration_ratio")
    try:
        logk_to_solvation(0.0, 298.15, float(ratio))
    except (TypeError, ValueError):
        return False, ["Invalid documented standard-state concentration ratio"]
    exclusions = evidence.get("excluded_compounds", {})
    if not isinstance(exclusions, dict) or any(not reason for reason in exclusions.values()) or not set(exclusions) <= set(candidates.compound_id):
        return False, ["Excluded compound IDs require valid IDs and documented reasons"]
    return True, []


def structure_qdb(root: Path, manifest: dict, staged: Path, solvation_file: Path, evidence_file: Path | None) -> pd.DataFrame:
    archive = (root / "article4.zip").read_bytes()
    candidates, rejected, raw_count = parse_qdb(archive)
    if candidates.empty:
        raise ValueError("QDB: no valid records")
    baseline = pd.read_csv(solvation_file)
    required = ["cation", "anion", "solute", "temperature_K", "solvation_kcal/mol", "source_list"]
    if set(required) - set(baseline):
        raise ValueError("QDB: incomplete solvation baseline")
    baseline = baseline[required].copy()
    for column in ("temperature_K", "solvation_kcal/mol"):
        baseline[column] = pd.to_numeric(baseline[column], errors="raise")
        if not baseline[column].map(math.isfinite).all():
            raise ValueError(f"QDB: invalid baseline {column}")
    baseline.insert(0, "baseline_row_number", range(2, len(baseline) + 2))
    keys = ["cation", "anion", "solute", "temperature_K"]
    cache = {}
    def identity_key(value):
        if value not in cache:
            cache[value] = fixed_h_inchikey(value)
        return cache[value]
    for frame in (candidates, baseline):
        for col in keys[:3]:
            frame[col] = frame[col].map(canonicalize_identity_smiles)
            frame[col + "_key"] = frame[col].map(identity_key)
    chemical_keys = [col + "_key" for col in keys[:3]] + ["temperature_K"]
    direct = candidates.merge(baseline, on=keys)
    paired = candidates.merge(baseline, on=chemical_keys, suffixes=("", "_existing"))
    paired["residual_kcal/mol"] = paired["candidate_solvation_kcal/mol"] - paired["solvation_kcal/mol"]
    audit = staged / "_audit"
    candidates.to_csv(audit / "qdb_candidates.csv", index=False)
    aliases = candidates[["cation", "anion", "cation_key", "anion_key", "raw_description"]].copy()
    aliases["raw_IL_alias"] = aliases.raw_description.str.partition(" [")[2].map(lambda s: "[" + s if s else "")
    aliases.drop(columns="raw_description").drop_duplicates().to_csv(audit / "source_IL_aliases.csv", index=False)
    rejected.to_csv(audit / "structure_rejected.csv", index=False)
    paired.sort_values("residual_kcal/mol", key=lambda x: x.abs(), ascending=False).to_csv(audit / "solvation_overlap_pairs.csv", index=False)
    # System medians are statistical summaries only; original supervision stays unaggregated.
    system_delta = paired.groupby(chemical_keys)["residual_kcal/mol"].median()
    grouped = []
    for kind, columns in (("IL", chemical_keys[:2]), ("solute", chemical_keys[2:3]), ("existing_source", ["source_list"])):
        group_frame = (paired.assign(source_list=paired.source_list.str.split(r"\s*;\s*")).explode("source_list")
                       if kind == "existing_source" else paired)
        for group, frame in group_frame.groupby(columns, dropna=False):
            grouped.append({"group_type": kind, "group": json.dumps(group), **residual_stats(frame["residual_kcal/mol"])})
    pd.DataFrame(grouped).to_csv(audit / "residual_groups.csv", index=False)
    archive_sha = hashlib.sha256(archive).hexdigest()
    baseline_sha = hashlib.sha256(Path(solvation_file).read_bytes()).hexdigest()
    evidence = json.loads(evidence_file.read_text()) if evidence_file else None
    allowed, issues = qdb_merge_evidence(evidence, archive_sha, baseline_sha, candidates)
    def counts(frame):
        return {"ILs": frame[chemical_keys[:2]].drop_duplicates().shape[0],
                "solutes": frame[chemical_keys[2]].nunique(),
                "systems": frame[chemical_keys[:3]].drop_duplicates().shape[0],
                "conditions": frame[chemical_keys].drop_duplicates().shape[0]}
    union = pd.concat([baseline[chemical_keys], candidates[chemical_keys]])
    before = counts(baseline)
    variants = {"equal_concentration": paired["residual_kcal/mol"]}
    for name, pressure_Pa in (("1atm_gas_to_1M_liquid", 101325.0), ("1bar_gas_to_1M_liquid", 100000.0)):
        ratio = 1000 * 8.31446261815324 * 298.15 / pressure_Pa
        variants[name] = paired["residual_kcal/mol"] + logk_to_solvation(0.0, 298.15, ratio)
    variants["opposite_sign_diagnostic"] = -paired["candidate_solvation_kcal/mol"] - paired["solvation_kcal/mol"]
    variants["natural_log_diagnostic"] = paired["candidate_solvation_kcal/mol"] / math.log(10) - paired["solvation_kcal/mol"]
    x, y = paired["candidate_solvation_kcal/mol"], paired["solvation_kcal/mol"]
    slope = float(x.cov(y) / x.var()) if len(x) > 1 and x.var() > 0 else None
    summary = {"raw_records": raw_count, "valid_records": len(candidates), "rejected_records": len(rejected),
               "raw_description_IL_aliases": int(aliases.raw_IL_alias.nunique()),
               "QDB": counts(candidates), "baseline": before,
               "direct_SMILES_overlap_systems": direct[keys[:3]].drop_duplicates().shape[0],
               "chemical_identity_overlap": counts(paired), "overlap_pairs": len(paired),
               "candidate_new_systems": counts(union)["systems"] - before["systems"],
               "candidate_new_conditions": counts(union)["conditions"] - before["conditions"],
               "baseline_at_298_15_K": counts(baseline.loc[baseline.temperature_K.eq(298.15)]),
               "hypothetical_union": counts(union), "merge_allowed": allowed, "unresolved_issues": issues,
               "actual_after": before, "actual_new_systems": 0,
               "equal_concentration_conversion": "-R*T*ln(10)*logK/4184; R=8.31446261815324 J/mol/K",
               "residual_record_weighted": residual_stats(paired["residual_kcal/mol"]),
               "residual_system_weighted": residual_stats(system_delta),
               "diagnostic_variants_not_applied": {name: residual_stats(delta) for name, delta in variants.items()},
               "diagnostic_fit_not_applied": {"existing_vs_candidate_slope": slope,
                   "intercept": float(y.mean() - slope * x.mean()) if slope is not None else None},
               "archive_sha256": archive_sha, "baseline_sha256": baseline_sha,
               "baseline_file": str(solvation_file), "physical_evidence": evidence}
    if allowed:
        selected = candidates.loc[~candidates.compound_id.isin(evidence.get("excluded_compounds", {}))].copy()
        ratio = float(evidence["liquid_to_gas_standard_concentration_ratio"])
        selected["solvation_kcal/mol"] = selected.logK.map(lambda v: logk_to_solvation(v, 298.15, ratio))
        output = selected[[*keys, "solvation_kcal/mol"]]
        output.to_csv(staged / "solvation_structured.csv", index=False)
        selected["structured_row_index"] = range(len(selected))
        selected["source_doi"] = manifest["doi"]
        selected["source_sha256"] = archive_sha
        selected["conversion_standard_concentration_ratio"] = ratio
        selected.to_csv(audit / "solvation_structured_provenance.csv", index=False)
        summary["approved_structured_union"] = counts(pd.concat([baseline, selected]))
    else:
        output = pd.DataFrame()
    (audit / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output


def structure_archive(source: str, raw_root: Path, output_root: Path, solvation_file: Path, evidence_file: Path | None) -> pd.DataFrame:
    root, manifest = verified_archive(source, raw_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_root) as temporary_dir:
        staged = Path(temporary_dir) / source
        (staged / "_audit").mkdir(parents=True)
        output = (structure_lethesh(root, manifest, staged) if source == "Lethesh2022" else
                  structure_qdb(root, manifest, staged, solvation_file, evidence_file))
        shutil.copy2(root / "manifest.json", staged / "_audit/manifest.json")
        destination = output_root / source
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staged, destination)
    return output


def structure_source(source: str, raw_root: Path, output_root: Path,
                     solvation_file: Path = ROOT / "data/final/experiment/solvation.csv",
                     evidence_file: Path | None = None) -> pd.DataFrame:
    if source in EXPERIMENT_ARCHIVES:
        return structure_archive(source, raw_root, output_root, solvation_file, evidence_file)
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
    parser.add_argument("--sources", nargs="+", choices=list(PUBLIC_SOURCES), default=list(SNAPSHOTS))
    parser.add_argument("--solvation-file", type=Path, default=ROOT / "data/final/experiment/solvation.csv")
    parser.add_argument("--qdb-compatibility-evidence", type=Path)
    args = parser.parse_args()
    for source in args.sources:
        output = structure_source(source, args.raw_root, args.output_root, args.solvation_file, args.qdb_compatibility_evidence)
        print(f"{source}: {len(output)} structured rows")


if __name__ == "__main__":
    main()
