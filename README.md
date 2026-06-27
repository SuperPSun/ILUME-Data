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

## Legacy Scripts

Older structuring scripts have been moved to `trash/`. Non-structuring scripts such as crawlers and plotting utilities remain in `scripts/`.
