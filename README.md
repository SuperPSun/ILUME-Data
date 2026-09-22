# ILUME-Data

Structures raw ionic-liquid datasets into unit-explicit CSVs and materialized training splits.

## Usage and data layout

Run from the repository root:

```bash
pip install -r requirements.txt
python scripts/structure_raw_data.py
```

Raw sources are `data/raw/{AIonopedia,ILBERT,ILThermo,after_AIonopedia,simulation_data}/`.
Outputs use the same names under `data/structured/`, except `simulation_data` becomes
`simulation`. Select sources with `--sources AIonopedia ILBERT ILThermo after_AIonopedia simulation`.

Columns follow identity → conditions → metadata → labels. Conditions retain names
such as `temperature_K` and `pressure_kPa`; labels include units, e.g. `density_g/cm^3`.
During experimental merging, missing pressure defaults to `101.325` kPa for density,
electrical conductivity, heat capacity, refractive index, thermal conductivity and
viscosity, before condition matching. Explicit pressures, cleaned inputs and
simulation pressure fields are unchanged.

Rebuild merged, final and Stage2/Stage3 data in dependency order:

```bash
python scripts/merge_data.py
python scripts/build_final_data.py
python scripts/build_training_splits.py build-splits --seed 42
```

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
  replaces `data/training_splits/stage1/`, including removal of prior augmentation.
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
  rule/PubChem augmentation and Stage2/Stage3, but excludes ZINC. Once ZINC is
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

## Stage2 Property Splits

`build-splits` rebuilds Stage2, Stage3 and `_audit`, leaving Stage1 untouched.
Simulation supervision is explicitly registered as nine Stage2 tasks: pooled
PBE/TZVP HOMO and LUMO, partial atomic charge, HF molecular QM properties, density,
heat capacity, thermal expansion, heat of vaporization and organic transfer.
`isobaric_coefficient_of_volume_expansion` is excluded during merged-data
publication, so it is unavailable to final-data, training and analysis
workflows. Other experiment tasks enter Stage3. Unknown top-level simulation
CSVs fail; adding a simulation task requires a registry entry and tests.

### Identity and partitions

Each task splits complete systems independently: `(cation, anion)` for ILs,
`(solute, solvent)` for organic transfer, canonical `SMILES` for molecular/atom
supervision. Temperature and pressure never define identity; all condition rows
for one system stay together within that task.

| Stage2 tasks | Partition rule (default seed 42) |
|---|---|
| Heat of vaporization, partial atomic charge | Sorted groups, task-local seeded shuffle, grouped 80/10/10 train/validation/test |
| HOMO, LUMO | Shared inherited legacy role-specific 80/10/10 mapping; [ADR 0004](docs/adr/0004-stage2-homo-lumo-scalar-tasks.md) |
| Remaining tasks | Stable-hash assignment, approximately 90/10 train/validation |

Identical inputs and seed reproduce assignments. Cross-task entity quarantine is
not applied. Test labels are for final evaluation, never tuning or early stopping;
these tests do not measure catastrophic forgetting because Stage3 does not modify
the Stage2 model.

Before splitting, density, heat capacity and organic transfer exclude systems in
their respective experiment references. Exclusion is property-local; missing
references abort.
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

### Producer interface

`data/training_splits/task_catalog.csv` is the versioned contract: qualified task
IDs, materialized paths, task/target levels, identity/condition columns, split/sample
units, method, experiment reference, label source, optional resource manifest and
statistics. Schema v2 includes Stage2 `partitions` and per-partition row/system
counts; consumers must not infer test availability by probing files.
`target_columns` names CSV columns for object tasks and logical labels for atom
tasks, interpreted with `task_kind`, `target_level` and `label_source`.
[ADR 0003](docs/adr/0003-stage2-physics-supervision-contract.md) records the original
producer decision; ADR 0004 supersedes its orbital clauses. The partition table
above describes current Stage2 splitting.

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
Dynamic permittivity and CO₂ use the approved 10 GHz selection and
reference-solubility formulas. Viscosity, electrical conductivity and self
diffusion use the approved Arrhenius temperature-pressure formula. Experimental
transfer-organic is included only for shared-solute comparisons with solvation
and transfer; its other Stage3 matrix cells are marked `incompatible_topology`.
See the [analysis contract](docs/dataset_relationship_graph.md) for definitions,
limitations and validation.

## Legacy Scripts

Older structuring scripts are in `trash/`; crawlers and plotting utilities remain
in `scripts/`.
