# ILUME-Data

This repository structures raw ionic-liquid datasets into unit-explicit CSV files.

## Raw Sources

The current structuring entrypoint processes these raw directories:

- `data/raw/AIonopedia/`
- `data/raw/ILBERT/`
- `data/raw/ILThermo/`
- `data/raw/after_AIonopedia/`

`data/raw/simulation_data/` is not processed by the current structuring workflow.

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
python scripts/structure_raw_data.py --sources AIonopedia ILBERT ILThermo after_AIonopedia
```

## Output Layout

Structured files are written under:

- `data/structured/AIonopedia/`
- `data/structured/ILBERT/`
- `data/structured/ILThermo/`
- `data/structured/after_AIonopedia/`

Output columns are ordered as system identifiers first, experimental conditions second, metadata next, and unit-explicit labels last. Column names use underscore unit suffixes, for example `temperature_K`, `pressure_kPa`, and `density_g_cm3`.

## Legacy Scripts

Older structuring scripts have been moved to `trash/`. Non-structuring scripts such as crawlers and plotting utilities remain in `scripts/`.
