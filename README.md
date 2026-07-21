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

This atomically replaces `data/training_splits/stage1/`. The root `anion.csv`, `cation.csv`, and `molecule.csv` files contain only dataset entities. `simulation_mol.csv`, `solute.csv`, and `solvent.csv` retain neutral source-level records for auditing, while `molecule.csv` is their identity-deduplicated training entrypoint. Running extraction also removes any previous `stage1/augmentation/` directory.

Generate the complete augmentation candidate pool separately:

```bash
python scripts/build_training_splits.py augment-pretrain
```

This command may query PubChem and can be resumed with the same command after interruption. Strict cache-only execution is available with `--offline`; a missing cached PubChem response aborts without replacing the last complete augmentation output. Candidates and audits are written under:

```text
data/training_splits/stage1/augmentation/
├── anion.csv
├── cation.csv
├── molecule.csv
└── _audit/
```

There is no Stage1 entity cap in this builder. Selecting or sampling augmented entities for a particular pretraining run is a separate downstream step. Use the two explicit commands above when rebuilding only Stage1; the `all` command also rebuilds Stage2 and Stage3.

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
