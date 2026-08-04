# Project Guidelines

## Scope

- This repository is a Python raw-data preparation workspace for ingesting, cleaning, de-duplicating, and exporting structured datasets.
- Keep changes focused on raw data processing only. Do not introduce dataset splitting, training, evaluation, or downstream feature-engineering logic unless the user explicitly asks for it.

## Terminal Usage

- Before running any command in the terminal, always activate the Conda environment with `conda activate ilagent2` in that terminal session first.
- Treat this as a hard requirement for every terminal session, including install, test, data-processing, and one-off inspection commands.
- If a command is executed in a fresh non-interactive shell, prefix it so activation happens in the same command flow before the real command runs.
- If the environment cannot be activated, stop and report the blocker instead of running the command outside `ilagent2`.

## Execution Paths

- Prefer repository entrypoints under `scripts/` over ad-hoc inline Python.
- Use `python scripts/structure_raw_data.py` for the full local raw-data structuring flow.
- Use `python scripts/structure_raw_data.py --sources ...` when processing only selected raw sources.
- Use `python scripts/build_training_splits.py extract-pretrain` to rebuild the dataset-only Stage1 entity files. Do not start `augment-pretrain`, which may query PubChem, unless the user explicitly requests network augmentation.
- Use `python scripts/build_training_splits.py build-splits --seed 42` only when the user explicitly requests generated split replacement; it atomically rebuilds Stage2, Stage3, and `_audit` outputs.
- Do not add or rely on notebook-based processing paths when an equivalent script exists under `scripts/`.

## Data Layout

- Read raw inputs from `data/raw/AIonopedia/`, `data/raw/ILBERT/`, `data/raw/ILThermo/`, `data/raw/after_AIonopedia/`, and `data/raw/simulation_data/`.
- Write structured outputs under the matching source directory in `data/structured/`; simulation outputs use `data/structured/simulation/`.
- Use `data/manual/` for manual review results that are not yet part of the automated structured outputs.
- Preserve the existing separation between raw inputs and structured outputs.
- Keep every Stage2 chemical system within one task-local partition: IL properties group by `(cation, anion)`, organic transfer by `(solute, solvent)`, and molecular QM by `SMILES`. Conditions such as temperature and pressure never define a new system, and different property tasks assign shared systems independently.

## Dependencies

- Keep Python dependencies aligned with `requirements.txt` unless the user requests a dependency change.
- When code depends on local packages, preserve the existing `src/`-based import pattern used by the scripts.

## Documentation and Comments

- Every Python script must keep a short English module docstring that explains the script's purpose.
- Every Python function must include a short English docstring that explains what the function does.
- Use short English inline comments only when a data-cleaning rule or control-flow branch is not self-evident from the code itself.
- Do not add filler comments or docstrings that only restate the function name without explaining its role.
