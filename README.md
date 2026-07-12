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
