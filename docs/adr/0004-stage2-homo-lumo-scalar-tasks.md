# ADR 0004: Stage2 HOMO/LUMO scalar tasks

- Status: Accepted
- Date: 2026-08-25
- Supersedes: ADR 0003's role-oriented multi-target orbital task and split clauses

## Context

The previous catalog represented PBE/TZVP cation and anion orbitals as two tasks, each with HOMO and LUMO targets. That couples two physical properties inside one head while preventing the same property from sharing supervision across ion roles. Renaming the tasks would also change the task-seeded three-way split unless the old assignment is reproduced explicitly.

## Decision

The role-separated structured and cleaned CSVs remain ingestion sources. Merge publishes two property-oriented files: `simulation/homo.csv` with `HOMO_eV` and `simulation/lumo.csv` with `LUMO_eV`. Both pool cation and anion observations and contain exactly `SMILES`, `ion_role`, `provenance_source_file`, `provenance_source_row`, their scalar target, and `source_list`. The source row is the 2-based CSV record number including the header offset. Role must be `cation` or `anion`, match the expected structured source file, and agree with the sign of the canonical molecule's formal charge.

Catalog schema remains version 2. The new `simulation/homo` and `simulation/lumo` records are molecule `object_property` tasks with `identity_columns=SMILES`, `simulation_method=PBE/TZVP`, `has_test=true`, and materialized paths `stage2/homo` and `stage2/lumo`. The audit columns are an explicit exception allowed only for these tasks and are not features or targets.

Split assignment is inherited, not regenerated from the new task IDs. For the canonical `(ion_role, SMILES)` union, the builder separately reproduces the legacy cation and anion grouped `80/10/10` partitions using the old task ID, old role-specific system type, original seed, and existing algorithm. The resulting role/system mapping is shared by HOMO and LUMO and published as `_audit/stage2_orbital_split_inheritance.csv`.

## Consequences

The active catalog still contains nine Stage2 tasks, but the orbital portion is now two independent scalar properties. Consumers can pool train normalization while retaining role diagnostics and row-level auditability. The old merged/final/materialized orbital outputs are removed from the active tree after temporary end-to-end validation; structured/cleaned sources and Git history remain intact. The task/catalog identity changes and old consumer artifacts or checkpoints are not compatible.
