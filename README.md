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

`python scripts/build_training_splits.py build-splits` builds the Stage2 property datasets under `data/training_splits/stage2/`. Each task is split independently by its complete chemical system: `(cation, anion)` for IL properties, `(solute, solvent)` for organic transfer, and `SMILES` for molecular QM properties. Conditions such as temperature and pressure are not part of the system identity, so every condition row for one system stays entirely in either `train.csv` or `valid.csv` within that task.

Before the Stage2 train/validation split, three simulated IL-property tasks exclude every complete IL system found in the corresponding experimental task: simulation density against experiment density, simulation heat capacity against experiment heat capacity, and simulation thermal expansion against experiment isobaric volume expansion. These exclusions are property-local rather than a union across properties. Missing reference datasets abort the build, and the excluded systems and row counts are recorded in `data/training_splits/_audit/stage2_overlap_exclusions.csv`.

The default seed `42` deterministically assigns approximately 10% of systems, rather than rows, to validation. Changing the seed changes the assignment, while rerunning with the same inputs and seed reproduces the same files. A system shared by different property tasks is assigned independently in each task. The command also rebuilds Stage3; use a temporary `--output-root` when validating split changes without replacing the current generated datasets.

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
