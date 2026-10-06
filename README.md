# ILUME-Data

Structures raw ionic-liquid datasets into unit-explicit CSVs and materialized training splits.

## Usage and data layout

Run from the repository root:

```bash
pip install -r requirements.txt
python scripts/structure_raw_data.py
```

The existing `structure_raw_data.py` sources are
`data/raw/{AIonopedia,ILBERT,ILThermo,after_AIonopedia,simulation_data}/`.
Outputs use the same names under `data/structured/`, except `simulation_data` becomes
`simulation`. Select sources with `--sources AIonopedia ILBERT ILThermo after_AIonopedia simulation`.

Columns follow identity → conditions → metadata → labels. Conditions retain names
such as `temperature_K` and `pressure_kPa`; labels include units, e.g. `density_g/cm^3`.
During experimental merging, missing pressure defaults to `101.325` kPa for density,
electrical conductivity, heat capacity, refractive index, thermal conductivity and
viscosity, before condition matching. Explicit pressures, cleaned inputs and
simulation pressure fields are unchanged.

Rebuild merged, final and supervised Stage1/Stage2/Stage3 data in dependency order:

```bash
python scripts/merge_data.py
python scripts/build_final_data.py
python scripts/build_training_splits.py build-splits --seed 42
```

Final publication replaces experimental `transfer.csv` with `hydration.csv`.
It pairs solvation and transfer on `(cation, anion, solute, temperature_K)` and
uses the existing labels' sign convention: hydration = solvation + transfer.
Each `(solute, temperature_K)` receives the median of all matched candidate
values, including multiple solvation values for the same pairing key; contributing
sources are combined in `source_list`. The output has no ionic-liquid identity.
`_audit/hydration_pairs.csv`, `hydration_summary.csv` and
`hydration_unmatched_transfer.csv` retain paired values and sources, per-group
counts/minima/maxima/medians, and excluded unmatched transfer rows respectively.
Merged inputs, final solvation and organic transfer are preserved. Missing one
input, missing required fields or invalid numbers stop publication.
After final publication, rebuild training splits and analysis outputs to propagate
the hydration contract; `build_final_data.py` does not update those downstream products.

This migration is required for legacy final data with role-separated orbital files
or no `charge_20260514/structure_manifest.csv`; mixed contracts fail discovery.
For code-only validation, use temporary output roots (`build-splits --output-root ...`).
Replacing official generated datasets requires separate authorization. Git restoration
does not restore generated outputs; an authorized rollback must also verify their
file set, hashes and catalog.

## Stage1 Pretraining Entities

Run Stage1 operations in this order:

```bash
python scripts/build_training_splits.py extract-pretrain
python scripts/build_training_splits.py augment-pretrain
python scripts/build_training_splits.py import-zinc-diversity
```

- `extract-pretrain` reads top-level final experiment/simulation CSVs and atomically
  replaces Stage1 entities, including removal of prior augmentation, while preserving
  `data/training_splits/stage1/properties/`.
  `anion.csv`, `cation.csv` and `molecule.csv` contain dataset entities only;
  `IL.csv` contains canonical pairs from both sources, `experiment_IL.csv` only
  experimental pairs. `simulation_mol.csv`, `solute.csv` and `solvent.csv` preserve
  neutral source records; `molecule.csv` deduplicates their training identities.
- `augment-pretrain` publishes uncapped rule/PubChem candidates to
  `stage1/augmentation/{anion,cation,molecule}.csv` with `_audit/`. It can query
  PubChem and resume after interruption. `--offline` requires complete cached
  responses and preserves the last complete output on failure. Neutral candidates
  use charge-preserving generic rules, remain sanitized single fragments with zero
  net charge, and exclude ion-specific rules; see [ADR 0002](docs/adr/0002-neutral-shared-structural-rules.md).
- `import-zinc-diversity` is a separate offline import. `all` runs extraction,
  rule/PubChem augmentation and supervised Stage1/Stage2/Stage3 splits, but excludes ZINC. Once ZINC is
  published, `augment-pretrain` refuses to erase it: a full rebuild restarts at extraction.

### Local ZINC22 diversity augmentation

Defaults are `data/raw/ZINC/zinc22_smi_data/`, `zinc22_smi_urls.txt` and
`zinc22_smi_failed.log` (the latter two under `data/raw/ZINC/`). During download:

```bash
python scripts/build_training_splits.py import-zinc-diversity --scan-only
```

This freezes finalized `.smi.gz` files at startup and updates only
`data/training_splits/.cache/stage1_zinc_diversity.sqlite`. After download, run the
formal import shown above. Neutral `M`/`O` parents become one deterministic
anion/cation representative each; bounded stable-hash sampling precedes chemistry.

`--minimum-per-ion-role` (alias `--target-per-ion-role`) defaults to 1,000,000 for
each ion role across base, rule/PubChem and ZINC entities. It is a publication gate:
all eligible unique cached candidates are published, without truncating excess.
Logged missing/empty shards may be skipped; a nonempty final shard overrides old
failure records. `.part`, unlogged missing/empty shards, corrupt gzip or inconsistent
cache abort formal publication and preserve previous output.

Published paths/columns stay stable. ZINC rows have `origin_list=zinc`, parent
`seed_smiles_list` and named `rule_list`; IDs/shards live in `_audit/zinc_provenance.csv`.
Other audits cover availability, rejection samples/counts, rules and balance.
Keep large ZINC payloads local; do not commit or redistribute them. Chemistry,
cache and publication decisions are in [ADR 0001](docs/adr/0001-zinc22-diversity-augmentation.md),
with links to source terms.

## Supervised Property Splits

`build-splits` rebuilds `stage1/properties`, Stage2, Stage3 and `_audit`, leaving
Stage1 entity CSVs and augmentation untouched. Nine simulation tasks are registered:
pooled PBE/TZVP HOMO/LUMO, partial atomic charge and HF molecular QM properties now
belong to Stage1; density, heat capacity, thermal expansion, heat of vaporization
and organic transfer belong to Stage2.
`isobaric_coefficient_of_volume_expansion` is excluded during merged-data
publication, so it is unavailable to final-data, training and analysis
workflows. Other experiment tasks enter Stage3. Unknown top-level simulation
CSVs fail; adding a simulation task requires a registry entry and tests.

### Identity and partitions

Each task splits complete systems independently: `(cation, anion)` for ILs,
`(solute, solvent)` for organic transfer, canonical `SMILES` for molecular/atom
supervision. Temperature and pressure never define identity; all condition rows
for one system stay together within that task.

| Supervised simulation tasks | Partition rule (default seed 42) |
|---|---|
| Thermal expansion, heat of vaporization, partial atomic charge | Sorted groups, task-local seeded shuffle, grouped 80/10/10 train/validation/test |
| HOMO, LUMO | Shared inherited legacy role-specific 80/10/10 mapping; [ADR 0004](docs/adr/0004-stage2-homo-lumo-scalar-tasks.md) |
| Remaining tasks | Stable-hash assignment, approximately 90/10 train/validation |

Identical inputs and seed reproduce assignments. Cross-task entity quarantine is
not applied. Test labels are for final evaluation, never tuning or early stopping;
these tests do not measure catastrophic forgetting because Stage3 does not modify
the Stage2 model.

Before splitting, density, heat capacity, heat of vaporization and organic transfer exclude systems in
their respective experiment references. Exclusion is property-local; missing
references abort.
Heat of vaporization references experimental
`enthalpy_of_vaporization_or_sublimation.csv`. The existing package contains 218
liquid–gas observations across 137 ILs; these are vaporization labels despite the
historical combined name. All temperatures of shared IL systems are excluded from
the simulation split; upstream records remain.
`_audit/stage2_overlap_exclusions.csv` records identities and excluded/matching row counts.

### Orbital and partial-charge resources

HOMO/LUMO pool both ion roles into scalar tasks, preserving role and source-row
audit columns, never used as features/targets. Their shared inherited assignment
is recorded in `_audit/stage2_orbital_split_inheritance.csv`; `gap_eV` is checked
against `LUMO_eV - HOMO_eV` at `1e-8 eV` tolerance but is not a target.
See [ADR 0004](docs/adr/0004-stage2-homo-lumo-scalar-tasks.md) for exact columns and compatibility.

`simulation/charge.csv` indexes atom-level supervision, not total-charge regression.
Each `mol_id` remains a sample; canonical SMILES defines partitions. Split columns
are `mol_id,SMILES,role,formal_charge,source_list`. ILUME parses atom labels from
structure resources. The manifest records relative paths, sizes, SHA-256 and
whether files are referenced; unreferenced files are allowed. Missing manifest
entries exclude charge rows into `_audit/partial_atomic_charge_resource_exclusions.csv`.

The complete `charge_20260514/` resource is copied once beside the partial-charge
`train.csv`, `valid.csv` and `test.csv`. All three index that shared copy, and the
catalog points to its manifest. Staging precedes replacement; allow disk space
and build time for this duplicated payload.

Stage1 property paths are `stage1/properties/{homo,lumo,partial_atomic_charge,simulated_qm_elec_hf}/`.
Task IDs remain `simulation/...`; catalog `stage` becomes 1 and resource paths follow
the new location. Partition algorithms, legacy orbital namespaces and seed remain
unchanged, preserving membership for unchanged inputs. Final files retain their
simulation provenance bucket. Rebuilding removes the four former Stage2 directories.
Existing consumer prepared/checkpoint outputs remain read-only until consumer
support is migrated separately.

### Producer interface

`data/training_splits/task_catalog.csv` is the versioned contract: qualified task
IDs, materialized paths, task/target levels, identity/condition columns, split/sample
units, method, experiment reference, label source, optional resource manifest and
statistics. Schema v2 includes supervised Stage1/Stage2 `partitions` and per-partition row/system
counts; consumers must not infer test availability by probing files.
`target_columns` names CSV columns for object tasks and logical labels for atom
tasks, interpreted with `task_kind`, `target_level` and `label_source`.
[ADR 0003](docs/adr/0003-stage2-physics-supervision-contract.md) records the original
producer decision; ADR 0004 supersedes its orbital clauses. The partition table
above describes current supervised simulation splitting.

## Public gas/water expansion

The two public sources use `crawl_public_properties.py` and
`structure_public_properties.py`; all later steps extend the existing pipeline.
Raw snapshots under `data/raw/{IL4GAS,WaterActivity}/` record original URLs, pinned
commit, download time, SHA-256, size and row count in `manifest.json`. Failed
experimental downloads preserve the previous complete source. Structure processing
verifies hashes and splits ionic roles by charge rather than fragment order.

- [IL4GAS](https://github.com/Yu-Xin-Qiu/IL4GAS), commit
  `a3a16fd0c2a186119efe231b570aa9ebbc38d6c8`: ten experimental CSVs, 19,745 raw rows.
  Gas SMILES use `solute`, and bar converts to kPa. Merge converts existing
  `x_CO2_unitless` into CO₂ gas records and publishes only
  `experiment/gas_solubility.csv`, target `x_gas_unitless`. Raw x is used directly;
  ln(x), COSMO and screening predictions are not labels. Identity is
  `(cation, anion, solute)`; T and P remain conditions.
- [Water activity dataset](https://github.com/MohanMood/NLP_Ionic-Liquid_Properties),
  commit `ba9768dbdd1c8f2dd90c091b27babc9c22db0c0c`: 3,578 raw rows. The
  [paper](https://pubs.rsc.org/en/content/articlehtml/2025/gc/d5gc02803e) defines
  `Activityt_water` as γ, so it maps directly to
  `water_activity_coefficient_unitless`, without division by mole fraction.
  `experiment/water_activity_coefficient.csv` retains `x_water_unitless`, T and P
  as conditions. x=0/1 and γ>1 are valid; γ must be finite and positive.

Both tasks enter Stage3 using existing splits. Structure/clean rejections and raw
references/record IDs stay in audits, without becoming labels or deduplication keys.
Equivalent structures from these new sources reuse existing sources' representative
SMILES when available, preserving the old tasks' chemical identity strings.
Final `_audit/public_properties/` preserves source traces;
`gas_solubility_legacy_co2_conversion.csv` records the old CO₂ conversion.
Analysis adds `gas_solubility_by_gas.csv` and `gas_solubility_by_gas_source.csv`;
source counts include each contributing source. Gas/water relationship formulas
remain `pending_formula` until separately approved.
These two tasks publish no relationship signatures, including for single raw points,
until their gas/concentration reference formulas are approved.

Existing pEC50 remains. No ILToxDB ingestion is added because it mixes IC50, EC50
and CC50 without reliably preserving their original designation. ECW is archive-only:
`python scripts/crawl_public_properties.py --sources ECW` downloads the SI when
available and records gaps/failures under `data/raw/ECW/`. No ECW task is published;
the incomplete hybrid table and 660 predictions are not used as a complete set of labels.

### User-run rebuild and validation

Use temporary output roots for implementation checks. The commands below rebuild
official datasets and analyses in an environment with `requirements.txt` installed.
Existing Stage1 entities and augmentation are preserved; extraction and augmentation
are not part of this rebuild. Consumer data and model configuration require a separate migration.

```bash
cd /data/pengs/ILUME-Data
set -euo pipefail

ILUME_EXPANSION_RUN=$(date +%Y%m%d-%H%M%S)
ILUME_EXPANSION_BACKUP="data/backups/expansion-$ILUME_EXPANSION_RUN"
mkdir -p "$ILUME_EXPANSION_BACKUP"
for ILUME_EXPANSION_TREE in merged final training_splits; do
  cp -a "data/$ILUME_EXPANSION_TREE" "$ILUME_EXPANSION_BACKUP/"
done
for ILUME_EXPANSION_TREE in raw/IL4GAS raw/WaterActivity \
  structured/IL4GAS structured/WaterActivity \
  cleaned/IL4GAS cleaned/WaterActivity cleaned/ILThermo; do
  if [ -d "data/$ILUME_EXPANSION_TREE" ]; then
    cp -a --parents "data/$ILUME_EXPANSION_TREE" "$ILUME_EXPANSION_BACKUP/"
  fi
done

python scripts/crawl_public_properties.py --sources IL4GAS WaterActivity
python scripts/structure_public_properties.py --sources IL4GAS WaterActivity
python scripts/structure_cleaned_ilthermo.py
python scripts/clean_structured_data.py
python scripts/merge_data.py
python scripts/build_final_data.py
python scripts/build_training_splits.py build-splits --seed 42

python scripts/analyze_final_properties.py --input-root data/final \
  --output-dir "analysis/dataset-expansion-$ILUME_EXPANSION_RUN"
python scripts/analyze_dataset_relationship_graph.py audit \
  --input-root data/training_splits \
  --output-dir "data/analysis/dataset-expansion-audit-$ILUME_EXPANSION_RUN" --seed 42
python scripts/analyze_dataset_relationship_graph.py compute \
  --input-root data/training_splits \
  --output-dir "data/analysis/dataset-expansion-graph-$ILUME_EXPANSION_RUN" --seed 42
```

Verify the contract and migration after rebuilding:

```bash
export ILUME_EXPANSION_BACKUP
python - <<'PY'
import os
from pathlib import Path
import pandas as pd

root = Path('data/training_splits')
catalog = pd.read_csv(root / 'task_catalog.csv').set_index('task_id')
names = ('homo', 'lumo', 'partial_atomic_charge', 'simulated_qm_elec_hf')
assert set(catalog.loc[catalog.stage.eq(1)].index) == {f'simulation/{n}' for n in names}
assert catalog.stage.eq(2).sum() == 5
backup = Path(os.environ['ILUME_EXPANSION_BACKUP'])
for name in names:
    record = catalog.loc[f'simulation/{name}']
    assert record.materialized_path == f'stage1/properties/{name}'
    assert not (root / 'stage2' / name).exists()
    old = backup / 'training_splits/stage2' / name
    new = root / record.materialized_path
    if old.is_dir():
        old_files = {p.relative_to(old): p.read_bytes() for p in old.rglob('*') if p.is_file()}
        new_files = {p.relative_to(new): p.read_bytes() for p in new.rglob('*') if p.is_file()}
        assert old_files == new_files, name
for name in ('gas_solubility', 'water_activity_coefficient', 'enthalpy_of_vaporization_or_sublimation'):
    assert catalog.loc[f'experiment/{name}', 'stage'] == 3
assert not Path('data/final/experiment/x_co2.csv').exists()
assert not Path('data/final/experiment/transfer.csv').exists()
assert 'x_water_unitless' in catalog.loc['experiment/water_activity_coefficient', 'condition_columns']
audit = pd.read_csv(root / '_audit/stage2_overlap_exclusions.csv')
hvap = audit.loc[audit.stage2_task_id.eq('simulation/heat_of_vaporization')]
print('Hvap excluded systems/rows:', len(hvap), hvap.excluded_stage2_rows.sum())
print(catalog.groupby('stage').size().to_dict())
print('Contract and unchanged-input partition migration verified.')
PY
```

### Consumer data migration

Consumer `data/` corresponds to `data/training_splits/` in this repository.
The consumer destinations are `/data/pengs/ILUME/data`,
`szx:/home/sunp/ILUME/data` and `h100:/workspace/pengs/ILUME/data`.
Back up each destination on its own host before replacement. Sync
`stage1/properties/`, `stage2/`, `stage3/` and `_audit/`, preserving existing Stage1
entity CSVs and `stage1/augmentation/`; publish `task_catalog.csv` last.
Remove obsolete task directories through the backed-up sync so that former
Stage2 property directories and the old experiment transfer task cannot survive
beside the new contract. Use `rsync -acni --delete` on each synchronized tree
after copying to verify file contents and obsolete-file removal.
Data transfer does not migrate consumer model configuration or existing prepared
and checkpoint artifacts; those artifacts retain their original dataset identities.

## 3D Box Fingerprints

```bash
python scripts/structure_3d_box_features.py --jobs 8
# Resume completed batches after interruption:
python scripts/structure_3d_box_features.py --jobs 8 --resume
```

Input is `box_20260514`; output is `data/structured/simulation/3d_box_structured.csv`.
Curves, checkpoints, parameters and non-OK rows remain in `analysis/3d_box_audit/`.
These are single-snapshot finite-box fingerprints, not trajectory averages;
`qc_` columns describe provenance/quality and are not model inputs by default.

## Dataset Relationship Graph

```bash
python scripts/analyze_dataset_relationship_graph.py compute \
  --output-dir data/analysis/dataset_relationship_graph_v1 --seed 42
```

Reads development labels without changing splits/training. Use `audit` to fit
signatures and inspect approvals without pairwise metrics. Output directories must
be new or empty. Stage3×Stage3 and Stage2→Stage3 outputs include overlap counts,
provenance, NA/stability tables, annotated heatmaps and five relationship graphs:
Spearman, distance correlation, directed binary I/H, multiclass MI and CV-NMAE.
Dynamic permittivity uses the approved 10 GHz selection. Unified gas solubility
and water activity await formula approval. Viscosity, electrical conductivity and self
diffusion use the approved Arrhenius temperature-pressure formula. Experimental
transfer-organic is included only for shared-solute comparisons with solvation
and transfer; its other Stage3 matrix cells are marked `incompatible_topology`.
See the [analysis contract](docs/dataset_relationship_graph.md) for definitions,
limitations and validation.

## Legacy Scripts

Older structuring scripts are in `trash/`; crawlers and plotting utilities remain
in `scripts/`.

## ILThermo binary and ternary snapshot

Install `requirements.txt`, then run the independent Bronze/Silver crawler:

```bash
python scripts/crawl_ilthermo_mixtures.py crawl --output-root data/ilthermo_mixtures
python scripts/crawl_ilthermo_mixtures.py verify --output-root data/ilthermo_mixtures
```

The first command searches all binary and ternary entries once, without property
filters. It freezes both Search manifests in `bronze/manifests/` and resumes missing
entries on later runs. Use a **new output directory** for a new database snapshot.
`bronze/entries/<entry_id>.json` contains the complete `GetEntry(...).response`
without added fields. Separate `bronze/metadata/` files record retrieval time and
SHA-256. A valid JSON and matching metadata checksum prevent another download.
Requests run serially with a one-second interval, timeouts, and five attempts with
exponential backoff. `bronze/failures/` records exhausted requests.

`silver/{entries,components,observations}.parquet` is rebuilt from Bronze on each
crawl. Component indices are the ILThermo component order (1, 2, 3), with no
assumed chemical roles. Each observation is one original variable within one
ILThermo data point; raw header, name, unit, phase, value and uncertainty are
retained. Data-point totals count original data rows, **not** long-format
observation rows. No unit conversion, ion decomposition or training preparation is
performed. `reports/summary.json`, `failures.jsonl`, and
`coverage_by_property.csv` distinguish pending entries, failures, and count
mismatches. A mismatch keeps its Bronze response for audit and is reported as a
failure. `silver/build_info.json` records the code hash and manifest hashes used
to materialize the tables; offline verification checks their values against Bronze.
Sparse source rows are retained in Silver and described in `reports/warnings.jsonl`.

For a small live check, use an otherwise empty temporary output directory:

```bash
python scripts/crawl_ilthermo_mixtures.py crawl \
  --output-root /tmp/ilthermo_mixtures_smoke --max-per-mixture 2
```

This still freezes the **complete** two Search manifests, but downloads at most
two entries of each size. Its report remains `incomplete` by design. Rerunning
without `--max-per-mixture` resumes the same snapshot; it does not refresh Search.
The separate `verify` command is offline and exits nonzero until all manifest
entries and Silver tables pass validation.

### Offline mixture structuring

From the repository root, structure the existing complete Bronze snapshot:

```bash
python scripts/structure_ilthermo_mixtures.py \
  --input-root data/ilthermo_mixtures \
  --output-root data/structured/ILThermo_mixtures
python -m pytest tests/test_structure_ilthermo_mixtures.py -q
```

This command makes no network requests and leaves Bronze/Silver unchanged. It
requires checksum-valid Bronze for every manifest entry; an invalid input aborts
before publishing outputs. Output files are staged and individually atomically
replaced after processing completes. Rerun the command to rebuild them.

- `all/{entries,components,observations}.parquet` covers every property. Entries
  retain original header definitions and constraints. Observations preserve each
  actual cell (including null), raw values, uncertainties, phases and units;
  short rows remain short. Compound names containing commas are matched against
  complete component names before separating units.
- `properties/*.csv` provides scalar labels using existing property conversion
  functions and column names. `sample_id` is `entry_id:data_point_index:variable_index`.
  Component order is unchanged. SMILES come only from the pinned ilthermopy local
  mapping, by ID then full name, with RDKit validation; ions are not split.
- Reported mole fractions, mass fractions and molalities are separate fields;
  no composition complement or default pressure is invented. Conditions and
  compositions are assigned only to matching phases or phase-neutral context.
  Constraints are retained in `constraints_json` and flagged, rather than
  interpreted as standardized conditions. Raw context stays in `context_json`.
- Raw uncertainty and its unit remain separate from transformed labels, including
  log10 labels. Unsupported units, kinematic viscosity, volumetric heat capacity,
  missing structures, ambiguous context and source irregularities keep their raw
  records and quality flags; failed standardized fields are empty.
- `audit/coverage_by_property.csv`, `issues.csv`, `unmatched_structures.csv` and
  `complex_properties.csv` describe coverage and problems. `summary.json` records
  input hashes, manifest metadata, library/code versions and conversion/mapping
  hashes; `inputs.csv` records each Bronze checksum and retrieval timestamp.

These independent mixture outputs are not consumed by merge, final-data or split
commands. Training-shaped CSV rows can have missing labels or quality flags;
inspect the audit files before choosing a downstream filtering policy.
