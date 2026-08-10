# ADR 0001: ZINC22 diversity augmentation from neutral parents

- Status: Accepted
- Date: 2026-08-09
- Amended: 2026-08-10

## Context

Stage1 already combines dataset entities, local resonance/chemical-rule candidates, and PubChem similarity candidates. The anion and cation pools remain below the desired one-million-per-role scale and need a broader structural-diversity source. Locally downloaded ZINC22 charge-specific `.smi.gz` shards provide that source, but their SMILES are neutral parent structures. The `M` and `O` classes describe the target charge space built by ZINC and must not be mistaken for validated explicit `-1` and `+1` SMILES.

The source can contain billions of rows and is downloaded concurrently through `.part` files followed by atomic renames. A correct importer therefore needs bounded chemistry work, deterministic recovery, source-level auditability, and an all-or-nothing publication boundary.

## Decision

Add the explicit `import-zinc-diversity` command. It is offline, is not included in `all`, and reads `data/raw/ZINC/zinc22_smi_data/` against `data/raw/ZINC/zinc22_smi_urls.txt` by default.

The chemical flow is:

```text
neutral ZINC parent -> project IL ionization rule -> explicit +/-1 candidate
-> exact deduplication and overlap exclusion -> publish all eligible candidates
-> Stage1 augmentation
```

`M` routes to `anion/-1`; `O` routes to `cation/+1`. Accepted structures must be a single non-radical fragment, contain 4-50 heavy atoms, and use only H/B/C/N/O/F/Si/P/S/Cl/Se/Br/I. The importer preserves an already explicit target charge and otherwise applies one supported protonation or deprotonation. It enumerates feasible single sites, then retains one representative by fixed functional-group priority and canonical-SMILES order. It does not perform alkylation, scaffold replacement, multi-step reactions, or automatic rule relaxation.

Before RDKit processing, each layer/HAC/logP/charge shard is bounded by a stable-hash sample. The initial role budget is four times the current minimum deficit. Cache entries are keyed by source path, SHA-256, sampling quota, and diversity seed; new shards are incremental and a changed shard invalidates only its own candidates. If a formal run is short, the importer increases the sampled population and rescans rather than weakening chemistry.

The configured per-role minimum is only a publication gate. After bounded presampling, ionization, canonical deduplication, and overlap exclusion, formal import publishes every remaining eligible candidate; it does not truncate the pool to the minimum. HAC, logP tranche, charged-center family, and Murcko scaffold remain audit dimensions rather than final selection quotas. Existing augmentation overlaps gain `zinc` provenance; base overlaps remain unchanged in the base-only CSV contract and are excluded from publication.

`--scan-only` freezes the finalized `.smi.gz` snapshot visible at process start, ignores `.part` files, updates only `stage1_zinc_diversity.sqlite`, and never changes formal augmentation output. Formal mode classifies every manifest entry against the finalized file, any `.part`, and `zinc22_smi_failed.log`. A non-empty final file always takes precedence over historical failure records. A missing or empty source is skipped only when its URL is present in the failure log; `.part` files and unlogged missing or empty sources still abort. Duplicate failure records are collapsed, non-manifest records are audited and ignored, and every processed gzip stream must validate completely.

Formal publication replaces `stage1/augmentation/` atomically only when both recomputed totals meet or exceed the configured minimum after including all eligible cached candidates. Existing excess and all eligible ZINC candidates are retained rather than truncated. CSV paths and columns remain stable. ZINC IDs and shard paths live in audit sidecars rather than the training schema. Unavailable sources are written to `zinc_unavailable_sources.csv`; rejections are stored as complete reason counts plus bounded examples, not as a row-level rejection export. The summary records `selection_mode=all_eligible`, and the balance audit retains its existing columns with `available=quota=selected` in every stratum.

## Consequences

Benefits:

- Adds a broad diversity layer while preserving an explainable connection from every ion to a neutral parent and a named rule.
- Avoids running RDKit and ionization rules over the entire ZINC population.
- Supports download-time incremental scans, interruption recovery, local shard invalidation, deterministic all-eligible publication, and atomic rollback behavior.
- Keeps ZINC import independent from PubChem availability and prevents `all` from unexpectedly starting a massive local import.

Limitations:

- The generated ion is a deterministic project representative, not a claim about the dominant protomer, tautomer, or solution speciation.
- Fixed functional-group priorities may underrepresent alternative charge sites.
- Strict rules can legitimately fail to reach the configured minimum; the command reports the shortage and retains the previous output.
- ZINC data is retained locally and is subject to the [ZINC terms and conditions](https://wiki.docking.org/index.php/Terms_And_Conditions).

## Alternatives considered

- Treat charge-specific source SMILES as explicit ions: rejected because the downloaded records are neutral parents.
- Use a general-purpose protonation package: rejected for this stage because the selected project rules are narrower, auditable, and directly testable for IL-relevant groups.
- Generate more molecules through alkylation or scaffold transformations: rejected because it turns source import into an insufficiently validated molecular generator.
- Sample only after fully ionizing every parent: rejected because its CPU and storage cost is inappropriate for ZINC22 scale.
- Retain the square-root/scaffold quota after meeting the minimum: rejected because the minimum is a validity threshold rather than a desired final size, and discarding otherwise eligible cached candidates wastes the completed chemistry work.
- Publish one ion role when the other is short: rejected because it breaks the paired minimum requirement and complicates reproducibility.
