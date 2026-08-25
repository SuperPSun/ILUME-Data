# ILUME-Data

This repository structures raw ionic-liquid datasets into unit-explicit CSV files.

## Raw Sources

The current structuring entrypoint processes these raw directories:

- `data/raw/AIonopedia/`
- `data/raw/ILBERT/`
- `data/raw/ILThermo/`
- `data/raw/after_AIonopedia/`
- `data/raw/simulation_data/`

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all supported sources:

```bash
python scripts/structure_raw_data.py
```

Run selected sources:

```bash
python scripts/structure_raw_data.py --sources AIonopedia ILBERT ILThermo after_AIonopedia simulation
```

## Output Layout

Structured files are written under:

- `data/structured/AIonopedia/`
- `data/structured/ILBERT/`
- `data/structured/ILThermo/`
- `data/structured/after_AIonopedia/`
- `data/structured/simulation/`

Output columns are ordered as system identifiers first, experimental conditions second, metadata next, and unit-explicit labels last. Condition columns keep stable names such as `temperature_K` and `pressure_kPa`; label columns preserve unit symbols in the unit suffix, for example `density_g/cm^3` and `surface_tension_mN/m`.

When `scripts/merge_data.py` merges experimental density, electrical conductivity, heat capacity, refractive index, thermal conductivity, or viscosity data, a missing `pressure_kPa` is interpreted as the standard-pressure default `101.325` kPa. This default is applied before cross-source condition matching and aggregation, and it never replaces an explicitly reported pressure. Cleaned source data and simulation tasks retain their original pressure fields.

## Stage1 Pretraining Entities

Rebuild the Stage1 dataset entities from the top-level CSV files under `data/final/experiment/` and `data/final/simulation/`:

```bash
python scripts/build_training_splits.py extract-pretrain
```

This atomically replaces `data/training_splits/stage1/`. The root `anion.csv`, `cation.csv`, and `molecule.csv` files contain only dataset entities. `IL.csv` contains the unique canonical ionic-liquid pairs observed across both experiment and simulation datasets, while `experiment_IL.csv` contains only those observed in experiment datasets. `simulation_mol.csv`, `solute.csv`, and `solvent.csv` retain neutral source-level records for auditing, while `molecule.csv` is their identity-deduplicated training entrypoint. Running extraction also removes any previous `stage1/augmentation/` directory.

Generate the complete augmentation candidate pool separately:

```bash
python scripts/build_training_splits.py augment-pretrain
```

This command may query PubChem and can be resumed with the same command after interruption. Strict cache-only execution is available with `--offline`; a missing cached PubChem response aborts without replacing the last complete augmentation output. Neutral molecules now use the same charge-preserving generic structural rules as ions: terminal alkyl-chain extension/shortening and branching, F/Cl/Br/I substitution, O/S substitution for supported functional groups, and aromatic C/N substitution. Ion-specific charged-headgroup and perfluoroalkyl rules remain excluded from neutral molecules, and every exported neutral candidate must remain a sanitized, single-fragment, net-zero structure. Candidates and audits are written under:

```text
data/training_splits/stage1/augmentation/
├── anion.csv
├── cation.csv
├── molecule.csv
└── _audit/
```

The rule/PubChem builder does not cap Stage1 entities. Use the two explicit commands above when rebuilding only those sources; the `all` command also rebuilds Stage2 and Stage3.

### Local ZINC22 diversity augmentation

ZINC diversity is a separate, offline import step. The charge-specific ZINC22 SMILES files contain neutral parent structures; their `M` and `O` classes are routing evidence, not charged structures ready for export. This project maps `M` parents to anion rules and `O` parents to cation rules, creates exactly one deterministic `-1` or `+1` representative per accepted parent, and publishes every eligible unique candidate from the bounded stable-hash sample after exact overlap exclusion.

During download, scan only the `.smi.gz` files that already completed their `.part` to final-name rename:

```bash
python scripts/build_training_splits.py import-zinc-diversity --scan-only
```

This freezes the finalized-file snapshot at startup, validates and incrementally caches that snapshot, and does not change `stage1/augmentation/`. The default local inputs are:

```text
data/raw/ZINC/zinc22_smi_data/
data/raw/ZINC/zinc22_smi_urls.txt
data/raw/ZINC/zinc22_smi_failed.log
```

After the download is complete, validate every manifest entry and atomically publish the merged result:

```bash
python scripts/build_training_splits.py import-zinc-diversity
```

The default requirement is at least `1,000,000` unique anions and at least `1,000,000` unique cations after combining base entities, rule/PubChem augmentation, and ZINC. This value is a publication gate, not a selection target: once both roles satisfy it, every eligible unique ZINC candidate in the bounded cache is published, so final totals may substantially exceed one million. Existing rows above the minimum are retained rather than truncated. `--target-per-ion-role` remains a compatibility alias for `--minimum-per-ion-role`.

Formal import treats the failed-download log as authoritative only for a currently missing or empty manifest source: those entries are audited as unavailable and skipped. A non-empty final shard always wins over its historical failed-log entry; a `.part`, an unlogged missing/empty source, a bad gzip stream, or inconsistent cache state still aborts without replacing the previous augmentation. The command never probes remote ZINC, never queries PubChem, and is intentionally not part of `all`.

The independent resumable cache is `data/training_splits/.cache/stage1_zinc_diversity.sqlite`. Published CSV columns and paths remain unchanged. ZINC rows use `origin_list=zinc`, retain the neutral parent in `seed_smiles_list`, and record the named ionization rule in `rule_list`; ZINC IDs and source shards are kept in `_audit/zinc_provenance.csv`. Additional ZINC audits cover the source manifest, unavailable sources, bounded rejection samples, rule counts, selection balance, and final minimum/excess counts.

Run Stage1 operations in this order: `extract-pretrain`, `augment-pretrain`, then `import-zinc-diversity`. Once ZINC has been formally published, `augment-pretrain` refuses to erase that layer. A complete rebuild must start again with `extract-pretrain`, then rebuild rule/PubChem augmentation and re-run the ZINC import. Large ZINC payloads remain local and must not be committed or redistributed. See the [ZINC terms and conditions](https://wiki.docking.org/index.php/Terms_And_Conditions), the [ZINC22 directory description](https://wiki.docking.org/index.php/ZINC22%3ADirectory_structure), [ADR 0001](docs/adr/0001-zinc22-diversity-augmentation.md), and [ADR 0002](docs/adr/0002-neutral-shared-structural-rules.md) for the chemical and architectural boundaries.

## Stage2 Property Splits

`python scripts/build_training_splits.py build-splits` builds the registered simulation tasks under `data/training_splits/stage2/` and experiment tasks under `data/training_splits/stage3/`. The registered Stage2 tasks are pooled PBE/TZVP HOMO and LUMO, partial atomic charge, HF molecular QM properties, density, heat capacity, thermal expansion, heat of vaporization, and organic transfer. An unregistered top-level simulation CSV aborts discovery instead of silently becoming a Stage3 task. Adding a simulation property therefore requires an explicit registry entry and tests; automatic task discovery is intentionally deferred.

The Stage2 contract depends on final data produced by the current merge and finalization code. When migrating a checkout whose `data/final/simulation/` still contains the four legacy `cation_homo`, `cation_lumo`, `anion_homo`, and `anion_lumo` files or lacks `charge_20260514/structure_manifest.csv`, rebuild in this order:

```bash
python scripts/merge_data.py
python scripts/build_final_data.py
python scripts/build_training_splits.py build-splits --seed 42
```

The last command intentionally fails rather than consuming a mixed old/new final-data contract. Use temporary output roots when validating code-only changes; replacing the official generated datasets requires separate authorization.

Each Stage2 task is split independently by its complete chemical system: `(cation, anion)` for IL properties, `(solute, solvent)` for organic transfer, and canonical `SMILES` for molecular QM and partial-charge data. Conditions such as temperature and pressure are not part of the system identity, so every condition row for one system stays entirely within one partition for that task. Heat of vaporization, pooled HOMO, pooled LUMO, and partial atomic charge use grouped `80/10/10` train/validation/test splits. The other Stage2 tasks retain grouped train/validation splits.

The role-separated structured/cleaned orbital inputs remain ingestion sources. Merge publishes `simulation/homo.csv` and `simulation/lumo.csv`, each pooling cation and anion rows and retaining `ion_role`, the structured source filename, and its 2-based CSV record number. These audit columns are validated against canonical formal charge and never become features or targets. HOMO and LUMO do not use their new task IDs to resample systems: the builder reconstructs each legacy cation/anion pool with the original task ID, role-specific system type, seed, and grouped three-way algorithm, then applies that inherited mapping to both scalar tasks. The exact mapping is published at `_audit/stage2_orbital_split_inheritance.csv`. `gap_eV` is still checked against `LUMO_eV - HOMO_eV` with a `1e-8 eV` tolerance but is not a target. See [ADR 0004](docs/adr/0004-stage2-homo-lumo-scalar-tasks.md).

`simulation/charge.csv` is an index for the `simulation/partial_atomic_charge` task, not a total-charge regression dataset. Every `mol_id` remains a separate sample, while all rows with the same canonical `SMILES` remain in one partition. Its split files contain `mol_id,SMILES,role,formal_charge,source_list`; atom-level partial charges are parsed later from the structure resource by ILUME. `data/final/simulation/charge_20260514/structure_manifest.csv` inventories each MOL/MOL2 file by relative path, size, SHA-256, and whether `charge.csv` references it. Unreferenced structure files are allowed. A charge row with no manifest entry is excluded without aborting and recorded in `_audit/partial_atomic_charge_resource_exclusions.csv`.

During `build-splits`, the complete final-data `charge_20260514/` directory is recursively copied once to `data/training_splits/stage2/partial_atomic_charge/charge_20260514/`, next to `train.csv`, `valid.csv`, and `test.csv`. All three CSVs are indexes into that shared resource. The partial-charge catalog row points `resource_manifest` at this copied manifest, so the Stage2 dataset is self-contained. This intentionally duplicates the structure payload and increases build time and disk usage; the copy is staged before the Stage2 directory is atomically replaced.

Before the Stage2 train/validation split, four simulated property tasks exclude every complete system found in the corresponding experimental task: simulation density against experiment density, simulation heat capacity against experiment heat capacity, simulation thermal expansion against experiment isobaric volume expansion, and simulation organic transfer against experiment organic transfer. These exclusions are property-local rather than a union across properties. Missing reference datasets abort the build, and the excluded systems and row counts are recorded in `data/training_splits/_audit/stage2_overlap_exclusions.csv`; its `cation`/`anion` or `solute`/`solvent` columns identify the affected system type.

The default seed `42` reproducibly assigns systems rather than rows. The four test-bearing tasks sort their groups and apply a task-local seeded shuffle before an `80/10/10` split; the other Stage2 tasks retain the existing stable-hash assignment of approximately 10% of systems to validation. Changing the seed changes the assignment, while rerunning with the same inputs and seed reproduces the same files. A system shared by different tasks is assigned independently in each task. These Stage2 test sets prevent target-supervision leakage only within the same task: Stage1, other Stage2 tasks, and Stage3 may contain the same chemical entities without that task's target label. Test data is reserved for final evaluation and must not be used for hyperparameter selection or early stopping. It is not a catastrophic-forgetting benchmark because Stage3 does not modify the Stage2 model.

`data/training_splits/task_catalog.csv` is the machine-readable producer contract. Globally unique task IDs point to their materialized paths and record task/target levels, identity and condition columns, split/sample units, method, experiment reference, label source, optional resource manifest, and row/system statistics. Catalog schema version 2 explicitly records Stage2 `partitions` and per-partition row/system counts, so consumers do not infer test availability from filesystem probing. For object tasks, `target_columns` names materialized CSV columns; for atom tasks it names the logical target and must be interpreted with `task_kind`, `target_level`, and `label_source`.

The `/data/pengs/ILUME` consumer uses this catalog as its Stage2 registry. This command also rebuilds Stage3; use a temporary `--output-root` when validating split changes without replacing generated datasets. See [ADR 0003](docs/adr/0003-stage2-physics-supervision-contract.md) and [ADR 0004](docs/adr/0004-stage2-homo-lumo-scalar-tasks.md) for the contract and compatibility boundary.

## 3D Box Fingerprints

Build one-row-per-snapshot structural fingerprints from `box_20260514` with:

```bash
python scripts/structure_3d_box_features.py --jobs 8
```

The structured table is written to `data/structured/simulation/3d_box_structured.csv`. RDF and structure-factor curves, batch checkpoints, the parameter manifest, and non-OK rows are retained under `analysis/3d_box_audit/`. Interrupted runs can continue without recomputing completed batches:

```bash
python scripts/structure_3d_box_features.py --jobs 8 --resume
```

These values are single-snapshot finite-box fingerprints, not trajectory averages. Columns prefixed with `qc_` describe provenance and calculation quality and should not be used as model inputs by default.

## Legacy Scripts

Older structuring scripts have been moved to `trash/`. Non-structuring scripts such as crawlers and plotting utilities remain in `scripts/`.
