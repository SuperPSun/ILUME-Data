"""Structure frozen ILThermo mixtures offline, retaining every raw observation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from functools import lru_cache
from html import unescape
from importlib.metadata import version
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from ilthermopy.compound_list import _compounds

# Allow both direct CLI execution and imports from repository tests.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import crawl_ilthermo_mixtures as crawl
from structure_raw_data import ILTHERMO_SPECS
from raw_prep import normalize_unit, to_kelvin, to_kpa, to_mhz, to_nm

PROGRAM_VERSION = "1.0.0"
PROPERTIES = {
    "Density": "density", "Electrical conductivity": "electrical_conductivity",
    "Enthalpy": "enthalpy", "Equilibrium pressure": "equilibrium_pressure",
    "Equilibrium temperature": "equilibrium_temperature",
    "Heat capacity at constant pressure": "heat_capacity_at_constant_pressure",
    "Heat capacity at vapor saturation pressure": "heat_capacity_at_vapor_saturation_pressure",
    "Refractive index": "refractive_index", "Relative permittivity": "relative_permittivity",
    "Speed of sound": "speed_of_sound", "Surface tension liquid-gas": "surface_tension_liquid_gas",
    "Thermal conductivity": "thermal_conductivity", "Thermal diffusivity": "thermal_diffusivity",
    "Viscosity": "viscosity",
}
COMPOSITION = re.compile(r"^(?:(Solvent):\s*)?(Mole fraction|Weight fraction|MolaLity|MolaRity|Volume fraction) of (.+)$", re.I)
KINDS = {"mole fraction": "mole_fraction", "weight fraction": "mass_fraction", "molality": "molality", "molarity": "molarity", "volume fraction": "volume_fraction"}
CONDITIONS = {"Temperature": ("temperature_K", to_kelvin), "Pressure": ("pressure_kPa", to_kpa), "Frequency": ("frequency_MHz", to_mhz), "Wavelength": ("wavelength_nm", to_nm)}
COMP_COLS = crawl.COMPONENT_COLUMNS + ["smiles_raw", "smiles", "smiles_match_source", "smiles_status"]
OBS_COLS = crawl.OBSERVATION_COLUMNS + ["source_cell_json", "variable_name_raw", "unit_raw", "value_numeric", "component_index", "composition_kind", "composition_scope", "metadata_status"]
BASE_COLS = ["sample_id", "entry_id", "data_point_index", "target_variable_index", "mixture_size", "phase"]
for i in (1, 2, 3):
    BASE_COLS += [f"component_{i}_{name}" for name in ("id", "name", "smiles", "mole_fraction", "mass_fraction", "molality_mol/kg")]
BASE_COLS += [c[0] for c in CONDITIONS.values()] + ["target_value_raw", "target_unit_raw", "target_uncertainty_raw", "target_uncertainty_unit_raw", "context_json", "constraints_json", "quality_flags", "reference_full", "reference_title", "raw_sha256"]


def plain(value):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", unescape(value or ""))).strip()


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=None)
def molecule(component_id, name):
    raw = _compounds.id2smiles.get(component_id)
    source = "component_id" if raw else "name"
    if not raw:
        raw = _compounds.name2smiles.get(name)
    if not raw:
        return {"smiles_raw": None, "smiles": None, "smiles_match_source": None, "smiles_status": "unmatched"}
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(raw)
    return {"smiles_raw": raw, "smiles": Chem.MolToSmiles(mol) if mol else None,
            "smiles_match_source": source, "smiles_status": "ok" if mol else "invalid"}


def header_metadata(raw_header, components):
    """Match complete compound names before splitting a possible unit suffix."""
    match = COMPOSITION.fullmatch(plain(re.sub(r"<SUP>\*</SUP>", "", raw_header, flags=re.I)))
    if match:
        scope, basis, tail = match.groups()
        hits = []
        for component in components:
            name = plain(component["name"])
            if tail == name:
                hits.append((component["component_index"], None))
            elif tail.startswith(name + ", "):
                hits.append((component["component_index"], tail[len(name) + 2:]))
        kind = KINDS[basis.lower()]
        if len(hits) == 1 or sum(hit[1] is None for hit in hits) == 1:
            exact = [hit for hit in hits if hit[1] is None]
            index, unit = (exact or hits)[0]
            if unit:
                variable_raw, unit = raw_header.rsplit(",", 1)
                unit = unit.strip()
            else:
                variable_raw = raw_header
            return plain(variable_raw), variable_raw, unit, index, kind, scope, "ok"
        # Fraction labels have no implied unit; commas belong to compound names.
        if kind.endswith("fraction"):
            variable_raw, unit = raw_header, None
        else:
            before, sep, after = raw_header.rpartition(",")
            variable_raw, unit = (before, after.strip()) if sep else (raw_header, None)
        return plain(variable_raw), variable_raw, unit, None, kind, scope, "ambiguous_component" if hits else "unmatched_component"
    name, sep, unit = raw_header.rpartition(",")
    variable_raw = name.strip() if sep else raw_header
    return plain(variable_raw).rstrip("*").strip(), variable_raw, unit.strip() if sep else None, None, None, None, "ok"


class ParquetSink:
    def __init__(self, path, columns, integer=(), floating=()):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = pa.schema([(c, pa.int64() if c in integer else pa.float64() if c in floating else pa.string()) for c in columns])
        self.writer = pq.ParquetWriter(path, self.schema, compression="zstd")
        self.rows = []
        self.count = 0

    def add(self, rows):
        self.rows.extend(rows)
        self.count += len(rows)
        if len(self.rows) >= 10000:
            self.flush()

    def flush(self):
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows.clear()

    def close(self):
        self.flush()
        self.writer.close()


def csv_writer(stack, path, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = stack.enter_context(path.open("w", encoding="utf-8", newline=""))
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    return writer


def converted(observation, converter, flags, issue, field):
    value = observation["value_numeric"]
    if value is None:
        flags.add(f"missing_or_invalid:{field}")
        issue(observation, f"missing_or_invalid:{field}")
        return None
    try:
        result = converter(value, observation["unit"])
        if result is None or not math.isfinite(result):
            raise ValueError("conversion returned no finite value")
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        flags.add(f"conversion_failed:{field}")
        issue(observation, f"{field}: {exc}")
        return None


def target_conversion(spec, observation):
    name, unit = observation["variable_name"], normalize_unit(observation["unit"])
    if spec.slug == "viscosity" and (name.startswith("Kinematic") or unit not in {"pa*s", "mpa*s"}):
        raise ValueError("kinematic/unsupported viscosity cannot be a dynamic-viscosity label")
    if spec.slug in {"refractive_index", "relative_permittivity"} and unit not in {None, "unitless", "1"}:
        raise ValueError("unsupported dimensionless target unit")
    # Reject variants such as volumetric heat capacity under a molar label.
    if not re.fullmatch(spec.property_pattern, name, re.I):
        raise ValueError("target variant is outside the scalar definition")
    return spec.transform(observation["value_numeric"], observation["unit"])


def training_row(entry, components, point, target, spec, issue):
    flags = set()
    if entry["validation_status"] != "ok":
        flags.add(entry["validation_status"])
    if json.loads(entry.get("constraints_json", "[]")):
        flags.add("constraints_not_standardized")
    phases = json.loads(entry["phases_json"]) or []
    phase = target["phase"] or (phases[0] if len(phases) == 1 else None)
    row = {"sample_id": f"{entry['entry_id']}:{target['data_point_index']}:{target['variable_index']}",
           "entry_id": entry["entry_id"], "mixture_size": entry["mixture_size"],
           "data_point_index": target["data_point_index"], "target_variable_index": target["variable_index"], "phase": phase,
           "target_value_raw": target["value_raw"], "target_unit_raw": target["unit_raw"],
           "target_uncertainty_raw": target["uncertainty_raw"], "target_uncertainty_unit_raw": target["unit_raw"],
           "reference_full": entry["reference_full"], "reference_title": entry["reference_title"], "raw_sha256": entry["raw_sha256"], "constraints_json": entry.get("constraints_json", "[]")}
    for component in components:
        i = component["component_index"]
        for key in ("id", "name", "smiles"):
            row[f"component_{i}_{key}"] = component["component_id" if key == "id" else key]
        if component["smiles_status"] != "ok":
            flags.add(f"smiles_{component['smiles_status']}:{i}")
    fields = defaultdict(list)
    composition_kinds = set()
    for obs in point:
        if obs is target:
            continue
        if obs["phase"] and obs["phase"] != phase:
            if phase is None:
                flags.add("ambiguous_phase_context")
            continue
        if obs["variable_name"] in CONDITIONS:
            field, converter = CONDITIONS[obs["variable_name"]]
            if field == "wavelength_nm" and normalize_unit(obs["unit"]) in {"a", "angstrom", "angs"}:
                converter = lambda value, _unit: value * 0.1
            fields[field].append(converted(obs, converter, flags, issue, field))
        elif obs["composition_kind"]:
            kind, index = obs["composition_kind"], obs["component_index"]
            if obs["metadata_status"] != "ok":
                flags.add(obs["metadata_status"])
                continue
            if obs["composition_scope"]:
                flags.add("solvent_scope_composition")
                continue
            if kind in {"mole_fraction", "mass_fraction", "molality"}:
                composition_kinds.add(kind)
                suffix = "molality_mol/kg" if kind == "molality" else kind
                field = f"component_{index}_{suffix}"
                def converter(value, unit):
                    normalized = normalize_unit(unit)
                    if kind == "molality" and normalized != "mol/kg":
                        raise ValueError(f"unsupported molality unit: {unit}")
                    if kind != "molality" and normalized not in {None, "1", "unitless"}:
                        raise ValueError(f"unsupported fraction unit: {unit}")
                    return value
                value = converted(obs, converter, flags, issue, field)
                if value is not None and (value < 0 or (kind != "molality" and value > 1)):
                    flags.add(f"composition_out_of_range:{field}")
                    issue(obs, f"composition_out_of_range:{field}")
                fields[field].append(value)
            else:
                flags.add(f"composition_not_standardized:{kind}")
    for field, values in fields.items():
        if len(values) != 1:
            flags.add(f"ambiguous_context:{field}")
        else:
            row[field] = values[0]
    if not composition_kinds:
        flags.add("missing_standard_composition")
    for kind in composition_kinds:
        suffix = "molality_mol/kg" if kind == "molality" else kind
        if kind != "molality" and any(row.get(f"component_{i}_{suffix}") is None for i in range(1, entry["mixture_size"] + 1)):
            flags.add(f"incomplete_reported_composition:{kind}")
    if row.get("temperature_K") is None and spec.slug != "equilibrium_temperature":
        flags.add("missing_temperature")
    if row.get("pressure_kPa") is None:
        flags.add("missing_pressure")
    row[spec.label_column] = converted(target, lambda _value, _unit: target_conversion(spec, target), flags, issue, spec.label_column)
    row["context_json"] = json.dumps([{k: o[k] for k in ("variable_index", "raw_header", "phase", "value_raw", "uncertainty_raw")} for o in point if o is not target], ensure_ascii=False)
    row["quality_flags"] = "|".join(sorted(flags))
    return row


def run(input_root, output_root):
    if version("ilthermopy") != crawl.ILTHERMOPY_VERSION:
        raise RuntimeError(f"Requires ilthermopy=={crawl.ILTHERMOPY_VERSION}")
    if input_root.resolve() == output_root.resolve() or input_root.resolve() in output_root.resolve().parents:
        raise ValueError("Output must be independent of the crawl snapshot")
    manifests = {n: crawl.load_json(crawl.manifest_path(input_root, n)) for n in (2, 3)}
    for n, manifest in manifests.items():
        crawl.validate_manifest(manifest, n)
    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".structure-", dir=output_root))
    coverage = defaultdict(Counter)
    quality = Counter()
    fingerprint = hashlib.sha256()
    sample_ids = set()
    expected_observations = expected_points = 0
    try:
        with ExitStack() as stack:
            entries_sink = ParquetSink(stage / "all/entries.parquet", crawl.ENTRY_COLUMNS + ["dhead_json"], integer=[c for c in crawl.ENTRY_COLUMNS if c.endswith("count") or c == "mixture_size"])
            components_sink = ParquetSink(stage / "all/components.parquet", COMP_COLS, integer=["component_index"])
            observations_sink = ParquetSink(stage / "all/observations.parquet", OBS_COLS, integer=["data_point_index", "variable_index", "component_index"], floating=["value_numeric"])
            for sink in (entries_sink, components_sink, observations_sink):
                stack.callback(sink.close)
            writers = {name: csv_writer(stack, stage / "properties" / f"{slug}.csv", BASE_COLS + [ILTHERMO_SPECS[slug].label_column]) for name, slug in PROPERTIES.items()}
            issues = csv_writer(stack, stage / "audit/issues.csv", ["entry_id", "data_point_index", "variable_index", "raw_header", "issue"])
            unmatched = csv_writer(stack, stage / "audit/unmatched_structures.csv", ["entry_id", "component_index", "component_id", "name", "smiles_status"])
            inputs = csv_writer(stack, stage / "audit/inputs.csv", ["entry_id", "raw_sha256", "fetched_at_utc"])
            complex_entries = csv_writer(stack, stage / "audit/complex_properties.csv", ["entry_id", "property", "reason"])
            for index, (size, search) in enumerate(crawl.unique_entries(manifests), 1):
                eid = str(search["id"])
                raw, meta, error = crawl.load_complete_entry(input_root, eid)
                if error:
                    raise ValueError(f"Bronze {eid}: {error}")
                entry, components, observations, count_issues, warnings = crawl.parse_entry(eid, size, search, raw, meta)
                entry["dhead_json"] = json.dumps(raw["dhead"], ensure_ascii=False)
                fingerprint.update(f"{eid}:{meta['response_sha256']}\n".encode())
                inputs.writerow({"entry_id": eid, "raw_sha256": meta["response_sha256"], "fetched_at_utc": meta["fetched_at_utc"]})
                for component in components:
                    component.update(molecule(component["component_id"], component["name"]))
                    if component["smiles_status"] != "ok":
                        unmatched.writerow({k: component[k] for k in unmatched.fieldnames})
                def issue(obs, message):
                    issues.writerow({"entry_id": eid, "data_point_index": obs.get("data_point_index"), "variable_index": obs.get("variable_index"), "raw_header": obs.get("raw_header"), "issue": message})
                for warning in count_issues + warnings:
                    issue({}, warning)
                by_point = defaultdict(list)
                for obs in observations:
                    name, name_raw, unit, ci, kind, scope, status = header_metadata(obs["raw_header"], components)
                    obs["source_cell_json"] = json.dumps(raw["data"][obs["data_point_index"] - 1][obs["variable_index"] - 1], ensure_ascii=False)
                    obs.update(variable_name=name, variable_name_raw=name_raw, unit=unit, unit_raw=unit,
                               value_numeric=number(obs["value_raw"]), component_index=ci,
                               composition_kind=kind, composition_scope=scope, metadata_status=status)
                    if status != "ok":
                        issue(obs, status)
                    by_point[obs["data_point_index"]].append(obs)
                entries_sink.add([entry])
                components_sink.add(components)
                observations_sink.add(observations)
                expected_observations += sum(len(cells) for cells in raw["data"])
                expected_points += len(raw["data"])
                prop = search["property"].strip()
                stats = coverage[(size, prop)]
                stats["entries"] += 1
                stats["data_points"] += len(raw["data"])
                stats["observations"] += len(observations)
                if prop not in PROPERTIES:
                    complex_entries.writerow({"entry_id": eid, "property": prop, "reason": "property outside scalar registry"})
                else:
                    spec = ILTHERMO_SPECS[PROPERTIES[prop]]
                    targets = [o for o in observations if not o["composition_kind"] and (re.fullmatch(spec.property_pattern, o["variable_name"], re.I) or o["variable_name"].startswith(prop + " per "))]
                    if not targets:
                        complex_entries.writerow({"entry_id": eid, "property": prop, "reason": "no unambiguous scalar target header"})
                    for target in targets:
                        row = training_row(entry, components, by_point[target["data_point_index"]], target, spec, issue)
                        if row["sample_id"] in sample_ids:
                            raise ValueError(f"Duplicate sample ID {row['sample_id']}")
                        sample_ids.add(row["sample_id"])
                        writers[prop].writerow(row)
                        stats["training_rows"] += 1
                        stats["numeric_labels"] += row[spec.label_column] is not None
                        for flag in row["quality_flags"].split("|"):
                            if flag:
                                quality[flag] += 1
                if index % 5000 == 0:
                    print(f"structured {index} entries", flush=True)
        assert observations_sink.count == expected_observations
        report = {"program_version": PROGRAM_VERSION, "program_sha256": crawl.sha256(Path(__file__).read_bytes()),
                  "git_commit": crawl.git_commit(), "built_at_utc": crawl.utc_now(), "input_root": str(input_root.resolve()),
                  "ilthermopy_version": version("ilthermopy"), "rdkit_version": rdBase.rdkitVersion,
                  "mapping_sha256": crawl.sha256(crawl.json_bytes({"id": _compounds.id2smiles, "name": _compounds.name2smiles})),
                  "converter_sha256": {p.name: crawl.sha256(p.read_bytes()) for p in (SCRIPT_DIR / "structure_raw_data.py", SCRIPT_DIR.parent / "src/raw_prep.py")},
                  "input_sha256": fingerprint.hexdigest(), "manifest_metadata": {str(n): {k: v for k, v in manifest.items() if k != "rows"} for n, manifest in manifests.items()},
                  "manifest_sha256": {str(n): crawl.sha256(crawl.manifest_path(input_root, n).read_bytes()) for n in (2, 3)},
                  "entries": entries_sink.count, "components": components_sink.count, "data_points": expected_points,
                  "observations": observations_sink.count, "training_rows": len(sample_ids), "quality_flags": dict(quality)}
        crawl.atomic_json(stage / "audit/summary.json", report)
        with (stage / "audit/coverage_by_property.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["mixture_size", "property", "entries", "data_points", "observations", "training_rows", "numeric_labels"])
            writer.writeheader()
            for (size, prop), counts in sorted(coverage.items()):
                writer.writerow({"mixture_size": size, "property": prop, **{key: counts[key] for key in writer.fieldnames[2:]}})
        # Every completed file is durable before publication; source files are untouched.
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                target = output_root / path.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(path, target)
                crawl.fsync_dir(target.parent)
        print(json.dumps({k: report[k] for k in ("entries", "components", "data_points", "observations", "training_rows")}, ensure_ascii=False))
        return report
    finally:
        shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=crawl.DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=crawl.PROJECT_ROOT / "data/structured/ILThermo_mixtures")
    args = parser.parse_args()
    run(args.input_root, args.output_root)


if __name__ == "__main__":
    main()
