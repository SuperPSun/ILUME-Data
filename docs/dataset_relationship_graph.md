# Dataset Relationship Graph v1

This independent analysis reads development labels from `data/training_splits`.
It does not change splits, training, groups, or existing analyses. Arrows describe
prediction, not causation. No Stage2–Stage2 matrix, chemical matching, clustering,
or aggregate relationship score is produced. Visualizations report the existing
overlap and relationship values; they do not define another metric.

From the repository root, using an environment with `requirements.txt` installed:

```sh
python scripts/analyze_dataset_relationship_graph.py audit \
  --input-root data/training_splits \
  --output-dir data/analysis/relationship_audit_v1 --seed 42

python scripts/analyze_dataset_relationship_graph.py compute \
  --input-root data/training_splits \
  --output-dir data/analysis/relationship_compute_v1 --seed 42 --dpi 300
```

Each output directory must be new or empty and outside the input tree. `compute`
includes the audit, signature construction, and all stability computations;
`audit` still fits signatures but skips pair metrics. Existing outputs are not
resumed or overwritten. Quadratic-memory dCor calculations can be expensive;
progress reports identify the active dataset pair.

## Inputs and approvals

The catalog defines `(source_file, target_columns)` nodes. Stage2 uses train and
valid only. Stage3 uses all five IL folds of cv1 for repeated tasks, otherwise
all five IL folds directly. Other strategies, repeats, test, summary and resource
files are not concatenated. Existing materialized cation/anion strings define
identities; this analysis does not recanonicalize or merge chemical equivalents.
Non-IL nodes are inventoried but excluded from matrices, except the experimental
`transfer_organic` node described below.
Invalid target/identity rows do not contribute observation overlap counts.

`configs/dataset_relationship_formulas_v2.json` records the user-approved formulas,
reference conditions and stability counts. An unknown included dataset is explicitly
pending, never assigned a default model. Approval means choosing an approximation,
not certifying that it is valid over every observed condition range.

Signatures use 298.15 K and 101.325 kPa, with 589 nm for refractive index. Observed
constant conditions remain at their actual values with `reference_mismatch`.
Varying conditions are extrapolated when necessary; coordinate distances and
ranges are recorded. `extrapolated` refers to departure from individual coordinate
ranges, not a guarantee that the reference lies inside a multivariate convex hull.
Rank-deficient designs return NA. Every varying condition in the approved formula
is retained for the general linear and Arrhenius formulas; pressure is not silently
dropped there. The x_CO2 exception follows its explicit identifiable-term rule
below. Two-point linear fits have no residual degrees of freedom and cannot
estimate uncertainty.

Single observations remain raw, including those in pending datasets. No-condition
replicates and same-condition replicates use medians; the latter remain at their
observed conditions. Density is fitted in natural-log space and restored to its
original units. Existing log10 targets are never logged a second time. Viscosity,
electrical conductivity and self diffusion use
`y = a + b(1/T - 1/298.15) + c(P - 101.325)` on their existing log10 targets.
Constant condition terms are not fitted; varying terms must be jointly identifiable.
The signature is `a`. If this design is underdetermined but an exact 298.15 K
observation exists, the closest-pressure observation at that temperature is kept
as `reference_temperature_observation`; its pressure mismatch remains explicit.
Without an exact reference-temperature observation, the signature remains NA.

Dynamic relative permittivity uses only observations at exactly 10 GHz
(`frequency_MHz == 10000`); repeated reference-frequency observations use their
median. A system without that exact frequency is `missing_reference_frequency`.
No Debye, Cole-Cole, or other frequency extrapolation is used.

For multi-observation x_CO2 systems, the response is
`ln(P_kPa/x_CO2) = a + b(1/T - 1/298.15) + c(P_kPa - 101.325)/T`.
Only varying terms that increase the design rank are retained, in formula order.
The reference signature is `101.325 exp(-a)`. Inputs require positive temperature
and pressure and `0 < x_CO2 < 1`; single-observation systems retain the raw value.

Solvation temperature fits operate on IL-solute sequences; single-observation
sequences enter the additive solute model unchanged. Transfer has no temperature
correction and rejects unexpectedly varying temperature. Unit signatures have
equal weight in the IL+solute model, constrained to mean(solute effect)=0.
Filtering precedes connectivity checking. Only the eligible connected component
with the most ILs is retained (stable component order breaks ties); a component
must contain at least two ILs. Unit provenance, fitted IL/solute effects and residuals are saved separately.
IL rows also record whether their retained units mix raw and corrected signatures;
that mixture propagates to the pair audit flag. The IL effect
standard error is conditional on those unit signatures and does not propagate
all temperature-fit errors.

Solvation-transfer and comparisons of either dataset with ordinary IL properties
use the solute-controlled IL-level `(cation, anion)` effects.

Experimental `transfer_organic` is read only from `random/fold1..5.csv`; test,
summary and alternate development strategies are not concatenated. Replicates
are aggregated by `(solute, solvent)` at constant temperature. An equal-weight
`solute + organic-solvent` additive model controls the solvent effect and extracts
solute effects constrained to mean zero. Only the largest connected component
containing at least two solutes is retained. The node is compared only with
solvation and transfer by exact shared solute strings. Those two datasets use
their solute effects from the existing `IL + solute` additive fits. Models are fit
once per dataset before intersection. Every other transfer-organic pair is kept
as `incompatible_topology` with NA counts and metrics. The solvation-transfer
edge remains an IL-level comparison.

## Metrics and stability

Every metric uses the current pair's valid-signature intersection. `n_shared`
counts that intersection; `n_observation_shared` separately counts shared
observation identities with usable targets, including failed/pending signatures.
The identity is normally an IL and is a solute only for the two allowed
transfer-organic pairs. All pairs remain present. EE diagonal relationship values are NA; its count diagonal is
the number of systems in that node. Directed SE outputs run simulation to
experiment only. Continuous z-scores use ddof=0.

| Metric | Minimum shared count | Definition |
|---|---:|---|
| Spearman | 3 | Rank correlation; constants are NA |
| Distance correlation | 5 | Biased doubly centered distance-matrix estimator |
| Binary I/H | 2 | MI divided by target-label entropy; zero entropy is NA |
| Multiclass MI | 20 | 3 bins at 20, 4 at 80, 5 at 150; pair quantiles |
| CV-NMAE | 20 | Model OOF absolute error divided by training-median baseline OOF absolute error |

`binary_thresholds` is empty in the approved config; optional entries are keyed
by exact node ID and must use the signature's units. Otherwise each side uses
its own median on the current shared subset. Quantile boundaries must be distinct,
all requested classes must exist, and every marginal bin must contain at least
five systems. A failure is `degenerate_quantile_bins` and does not fall back to
fewer bins. Both MI metrics use natural logarithms and discrete pair-specific labels,
not complete continuous mutual information.

Continuous KSG MI is excluded: its nearest-neighbor estimator is incompatible
with ordinary paired bootstrap in this pipeline.

Predictability uses StandardScaler, SplineTransformer(degree=3, n_knots=3), and
Ridge(alpha=1), all fitted within each of five shuffled training folds. At least
four distinct input values are required in each training fold. No tuning or
fallback model is used; a zero median-baseline error is NA. NMAE=1 means matching
the baseline's error; smaller is better. B→A is fitted separately from A→B.

Nonpredictive metrics use 200 paired-system bootstrap replicates and 199 target
permutations. Scaling and discretization are rebuilt in each replicate. Bootstrap
intervals require at least 100 finite replicates; failure counts are explicit.
Discrete MI may remain zero for constant labels, but I/H is unavailable for a
constant target. Spearman permutation tests use absolute correlation; other
nonpredictive tests use the upper tail. CV uses 20 repeated fold partitions for
split stability and 99 target permutations with the original folds for its lower-tail
test. P-values use the plus-one correction; no multiple-testing adjustment is
applied. Failed replicates are excluded and valid counts are reported.

All pair stability results are conditional on estimated system signatures; they
are not full uncertainty propagation. Repeated-CV percentile ranges describe split
stability and are not parameter confidence intervals. The manifest records this.

## Outputs and inspection

Top-level CSVs inventory nodes, formula approval/condition coverage, IL signatures,
IL-solute or solute-solvent unit signatures, and `comparison_signatures.csv` with
the three auditable solute-effect pools. Signature rows include formula version, source
files, fitted parameters, actual/requested conditions, uncertainty, residual degrees
of freedom and status. Inventory counts expose invalid observations.

`G_EE/` and `G_SE/` contain one CSV per metric, both count matrices, `pairs.csv`,
`na_reasons.csv`, and `confidence.csv`. Each also contains
`signature_overlap_matrix.csv`, whose short labels omit `.csv` and disambiguate
multi-target sources, plus an integer-annotated `signature_overlap_heatmap.png`.
Rows are sources and columns are targets.
EE symmetric values and their confidence rows are mirrored exactly. Pair records
include exact shared systems and flags for mixed signature kinds, reference
mismatches, heterogeneous actual conditions, extrapolation and pending formulas.
`comparison_space`, `signature_type`, `source_signature_type`, and
`target_signature_type` record whether an edge uses IL-system signatures,
solute-controlled IL effects, shared-solute effects, or an incompatible topology.
`system_identity` is `cation_anion`, `solute`, or `incompatible_topology`.
These flags are not automatic filters: inspect them before interpreting metrics.
Missing numeric values are serialized as `NA`; statuses explain why.

`knowledge_graphs/` contains one combined EE+SE PNG for each of the five metrics.
All finite relationships are drawn. Raw permutation p-values at or below 0.05
are highlighted; other edges remain gray. Symmetric metrics use undirected edges,
while I/H and CV-NMAE retain their legal directions. All figures use the same
seeded spring layout. Edge color is descriptive and is not an FDR-adjusted decision.
Spearman uses solid positive and dashed negative edges. Edge width represents
absolute Spearman, the raw nonnegative dCor/I/H/multiclass value, or
`1/(1+CV-NMAE)`, rescaled independently within each figure.

`manifest.json` records input/config/script hashes, dependency versions, seed,
settings, output CSV/PNG hashes and completion/failure status. Treat a running or
failed manifest as incomplete output. No source data is modified.

Validation:

```sh
python -m pytest -q tests/test_dataset_relationship_graph.py
```
