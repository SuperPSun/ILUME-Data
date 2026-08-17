# ADR 0003: Stage2 physics-supervision data contract

- Status: Accepted
- Date: 2026-08-17

## Context

The split pipeline previously routed only five whitelisted simulation files to Stage2. Other simulation outputs, including single-ion orbitals, total-charge metadata, and heat of vaporization, fell through to the experiment-oriented Stage3 workflow. HOMO and LUMO were also split into independent tasks, and `charge.csv` was treated as a scalar total-charge label even though its associated MOL2 resources contain the intended atom-level supervision.

ILUME-Data must publish one unambiguous producer contract without parsing structure chemistry or defining model atom indices. The current ILUME consumer remains hard-coded to the old five-task contract, so producer and consumer migration cannot be presented as one compatible release.

## Decision

All currently supported simulation supervision is registered explicitly as one of nine Stage2 tasks: PBE/TZVP cation orbitals, PBE/TZVP anion orbitals, partial atomic charge, HF molecular QM properties, density, heat capacity, thermal expansion, heat of vaporization, and organic transfer. Experiment CSVs remain Stage3 tasks. Unknown top-level simulation CSVs fail discovery; dynamic task metadata and automatic future-property discovery are deferred.

Stage2 uses deterministic task-local system hashing with approximately 90% train and 10% validation systems. Conditions never define identity. The split units are cation, anion, canonical molecule SMILES, complete ordered `(cation, anion)`, or ordered `(solute, solvent)` according to the task. Cross-task quarantine is not applied.

PBE/TZVP HOMO and LUMO are one multi-target task per ion role. The source gap is checked with an absolute tolerance of `1e-8 eV`, then omitted as a target. A mismatch is audited but does not remove or block the HOMO/LUMO row.

`charge.csv` is the identity and provenance index for `simulation/partial_atomic_charge`, not a total-charge target. Every mol_id is retained as a sample, duplicate canonical SMILES are not aggregated, and canonical SMILES is the split unit. RDKit formal charge must equal the source charge and determines cation/anion/neutral role without restricting charge magnitude. ILUME-Data does not parse MOL/MOL2 atoms or map them to Stage1 graph nodes. It inventories structure files in `structure_manifest.csv`; unreferenced files are allowed, while a charge row with no structure-manifest entry is excluded and audited without aborting the build.

`build-splits` recursively copies the complete final-data `charge_20260514/` directory into `stage2/partial_atomic_charge/charge_20260514/` during staged publication. The catalog advertises the copied `structure_manifest.csv`, making the materialized Stage2 partial-charge dataset self-contained rather than dependent on the final-data directory.

Property-local experiment exclusion remains limited to density, heat capacity, thermal expansion versus isobaric volume expansion, and organic transfer. Heat of vaporization and unrelated simulation objectives are not filtered against other experimental properties.

The root `task_catalog.csv` is the producer interface. Task IDs remain globally qualified (`simulation/...` or `experiment/...`) and map to explicit materialized paths. `target_columns` is intentionally conditional: it contains physical CSV columns for object properties and the logical label name for atom properties. Consumers must use `task_kind`, `target_level`, and `label_source` to distinguish those cases. The catalog also records condition and identity columns, split and sample units, simulation method, experiment reference, optional resource manifest, and materialized statistics.

## Consequences

Benefits:

- Simulation supervision can no longer silently enter Stage3.
- HOMO/LUMO identities cannot leak across independent splits.
- Partial-charge instances and structure provenance remain lossless without parsing structure payloads.
- The catalog makes producer output discoverable and versioned for a future ILUME migration.

Limitations:

- Adding a new simulation task requires a registry and test change until dynamic metadata is designed.
- Gap mismatches do not block publication.
- Missing structure references reduce the partial-charge task instead of failing the full build.
- Each split build duplicates the complete structure resource, increasing publication time and disk usage.
- The current ILUME Stage2 code cannot consume this contract; its catalog-driven migration is a separate architectural change.

## Alternatives considered

- Route every simulation CSV dynamically: deferred because task level, method, resource semantics, and logical atom targets cannot be inferred safely from filenames alone.
- Keep four scalar orbital tasks: rejected because it permits HOMO/LUMO split divergence for the same ion.
- Train total molecular charge from `charge.csv`: rejected because it is metadata for atom-level structure labels.
- Require exact equality between charge rows and structure files: rejected because cleaned-out source rows legitimately leave unreferenced structure resources.
- Parse MOL2 charges in ILUME-Data: rejected because atom ordering and Stage1 graph mapping belong to the ILUME preparation contract.
