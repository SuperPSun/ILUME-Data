# ADR 0002: Shared generic structural rules for neutral molecules

- Status: Accepted
- Date: 2026-08-09

## Context

Stage1 local augmentation originally applied structural analogue rules mainly to ions, while neutral molecules received resonance candidates and PubChem similarity candidates. This left neutral rule coverage unnecessarily conservative and made generic, charge-preserving transformations depend on network-derived candidates.

The requested expansion must increase local neutral diversity without turning the rule engine into an unrestricted molecule generator. Every neutral result must remain attributable to one named, single-step transformation and must preserve the Stage1 molecule contract.

## Decision

The `augment-pretrain` rule engine applies the following generic rule families to neutral molecules as well as ions:

- terminal alkyl-chain extension or shortening by one to four carbons;
- deterministic linear/branched alkyl interchange;
- substitution among F, Cl, Br, and I;
- O/S substitution for supported hydroxyl/thiol, ether/thioether, and carbonyl/thiocarbonyl environments;
- C/N substitution at supported positions in five- or six-membered aromatic rings.

Each generated candidate is sanitized and canonicalized with RDKit, deduplicated by canonical SMILES, and accepted only when it is a single fragment with the same formal charge as its seed. A candidate exported under the `molecule` role must therefore have formal charge zero.

Cation-specific N/P charged-headgroup replacement and anion-specific perfluoroalkyl-chain rules are not applied to neutral molecules. The rule engine continues to avoid multi-step transformations, unrestricted scaffold editing, and automatic relaxation after a shortfall. Neutral local augmentation has no total-count cap; PubChem behavior and its resumable cache remain unchanged.

The local cache stores a separate neutral-rule-set version. Changing this version removes only cached `rule` candidates for neutral seeds and resets their local status. Completed PubChem status, PubChem candidates, and HTTP cache entries are preserved.

## Consequences

Benefits:

- Neutral Stage1 coverage gains deterministic structural diversity without requiring PubChem.
- The same generic transformation has consistent semantics across charge roles.
- Cache migration avoids repeating completed network requests when neutral rules change.
- Named rule provenance and strict charge/fragment validation remain auditable.

Limitations:

- The transformations are graph analogues, not predictions of synthetic accessibility, stability, abundance, or experimental relevance.
- Aromatic C/N and O/S substitutions can materially change electronic properties even when formal charge is preserved.
- Single-step enumeration does not cover combinations of otherwise allowed transformations.

## Alternatives considered

- Keep neutral molecules PubChem-only: rejected because it leaves generic local diversity unused and ties expansion to network availability.
- Apply every ion rule to neutral molecules: rejected because charged-headgroup and perfluoroanion rules encode role-specific chemistry.
- Compose multiple rules recursively: rejected because it would greatly enlarge the search space and weaken direct seed-to-candidate interpretability.
