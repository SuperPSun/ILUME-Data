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
- Use `python scripts/run_raw_processing.py` for the full local raw-data processing flow.
- Use the dedicated scripts when a task is scoped to one data source, such as `python scripts/structuring_for_reference.py`, `python scripts/structuring_for_simulation.py`, `python scripts/structuring_ilthermo_data.py`, or a single script under `scripts/ilthermo/`.
- Do not add or rely on notebook-based processing paths when an equivalent script exists under `scripts/`.

## Data Layout

- Read raw inputs from `data/raw/1.0/`, `data/raw/ILThermo/`, and `data/raw/simulation_data/`.
- Write cleaned outputs under `data/structured/reference/`, `data/structured/simulation/`, or `data/structured/ILThermo/` according to the source.
- Use `data/manual/` for manual review results that are not yet part of the automated structured outputs.
- Preserve the existing separation between raw inputs and structured outputs.

## Dependencies

- Keep Python dependencies aligned with `requirements.txt` unless the user requests a dependency change.
- When code depends on local packages, preserve the existing `src/`-based import pattern used by the scripts.

## Documentation and Comments

- Every Python script must keep a short English module docstring that explains the script's purpose.
- Every Python function must include a short English docstring that explains what the function does.
- Use short English inline comments only when a data-cleaning rule or control-flow branch is not self-evident from the code itself.
- Do not add filler comments or docstrings that only restate the function name without explaining its role.