# ilagent2data

This repository extracts the pre-split data preparation portion of AIonopedia2 into the standalone raw-data workspace `ilagent2data`. It only handles raw data ingestion, simple cleaning, de-duplication, and structured export.

## Included data sources

- `data/raw/1.0/`: raw reference CSV files from the extracted AIonopedia2 snapshot.
- `data/raw/ILThermo/`: raw ILThermo text exports.
- `data/raw/simulation_data/`: raw simulation CSV files.
- `data/manual/`: manual review outputs and intermediate human-checked results.

## Included scripts

- `scripts/structuring_for_reference.py`: cleans and structures reference CSV files.
- `scripts/structuring_for_simulation.py`: cleans and structures simulation CSV files.
- `scripts/structuring_ilthermo_data.py`: batch entrypoint that runs all ILThermo property-specific structuring scripts.
- `scripts/ilthermo/`: one script per ILThermo property, each writing a single structured CSV.
- `scripts/crawl_ilthermo_pure_energetics.py`: crawls pure-compound ILThermo energetics records and resolves canonical SMILES via PubChem with OPSIN fallback.
- `scripts/run_raw_processing.py`: lightweight orchestrator for the local processing steps.

## Output layout

- `data/structured/reference/`
- `data/structured/simulation/`
- `data/structured/ILThermo/`

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the local cleaning and structuring steps:

```bash
conda activate ilagent2
python scripts/run_raw_processing.py
```

Run a single step:

```bash
python scripts/structuring_ilthermo_data.py
python scripts/structuring_for_reference.py
python scripts/structuring_for_simulation.py
```

ILThermo processing is split into one script per property under `scripts/ilthermo/`, and the shared batch runner remains available at `scripts/structuring_ilthermo_data.py`.

Example single-property runs:

```bash
python scripts/ilthermo/structuring_ilt_density.py
python scripts/ilthermo/structuring_ilt_electrical_conductivity.py
python scripts/ilthermo/structuring_ilt_enthalpy.py
```

All ILThermo structured CSVs now include a `note` column immediately before `source_text`. For most properties it is empty; for enthalpy and entropy it is backfilled from the pure-compound energetics crawler output when available.

Run the ILThermo pure energetics crawler:

```bash
python scripts/crawl_ilthermo_pure_energetics.py --limit-sets 2 --write-raw-sets
```

By default, crawler outputs are written directly under `data/raw/ILThermo/` so the enthalpy and entropy structuring scripts can reuse the crawled reference-state notes without extra CLI overrides. The crawler writes one file per property as `pure_compound_enthalpy.csv`, `pure_compound_entropy.csv`, `pure_compound_enthalpy_of_transition_or_fusion.csv`, and `pure_compound_enthalpy_of_vaporization_or_sublimation.csv`; optional raw per-set JSON files are written under `data/raw/ILThermo/raw_sets/` when `--write-raw-sets` is enabled. Canonical SMILES are resolved during crawling through PubChem first, and the crawler automatically falls back to OPSIN when PubChem does not return a usable structure.

## Scope boundary

This repository intentionally excludes dataset splitting, cross-validation generation, augmentation, training, evaluation, and downstream feature engineering.