# Dataset Relationship Graph v1

This independent analysis reads development labels from `data/training_splits`.
It does not change splits, training, groups, or existing analyses. Arrows describe
prediction, not causation. No Stage2–Stage2 matrix, chemical matching, clustering,
plots, or aggregate relationship score is produced.

From the repository root, using an environment with `requirements.txt` installed:

```sh
python scripts/analyze_dataset_relationship_graph.py audit \
  --input-root data/training_splits \
  --output-dir data/analysis/relationship_audit_v1 --seed 42

python scripts/analyze_dataset_relationship_graph.py compute \
  --input-root data/training_splits \
  --output-dir data/analysis/relationship_compute_v1 --seed 42
```

Each output directory must be new or empty and outside the input tree. `compute`
includes the audit, signature construction, and all stability computations;
`audit` still fits signatures but skips pair metrics. Existing outputs are not
resumed or overwritten. VFT fitting and quadratic-memory dCor calculations can
be expensive; progress reports identify the active dataset pair.

## Inputs and approvals

The catalog defines `(source_file, target_columns)` nodes. Stage2 uses train and
valid only. Stage3 uses all five IL folds of cv1 for repeated tasks, otherwise
all five IL folds directly. Other strategies, repeats, test, summary and resource
files are not concatenated. Existing materialized cation/anion strings define
identities; this analysis does not recanonicalize or merge chemical equivalents.
Non-IL and transfer_organic nodes are inventoried but excluded from matrices.
Invalid target/identity rows do not contribute observation overlap counts.

`configs/dataset_relationship_formulas_v1.json` records the user-approved formulas,
reference conditions and stability counts. An unknown included dataset is explicitly
pending, never assigned a default model. Dynamic permittivity and x_CO2 currently
remain pending. Approval means choosing an approximation, not certifying that it
is valid over every observed temperature/pressure range.

Signatures use 298.15 K and 101.325 kPa, with 589 nm for refractive index. Observed
constant conditions remain at their actual values with `reference_mismatch`.
Varying conditions are extrapolated when necessary; coordinate distances and
ranges are recorded. `extrapolated` refers to departure from individual coordinate
ranges, not a guarantee that the reference lies inside a multivariate convex hull.
Rank-deficient designs return NA. Every varying condition in the approved formula
is retained; pressure is not silently dropped. Two-point linear fits have no
residual degrees of freedom and cannot estimate uncertainty.

Single observations remain raw, including those in pending datasets. No-condition
replicates and same-condition replicates use medians; the latter remain at their
observed conditions. Density is fitted in natural-log space and restored to its
original units. Existing log10 targets are never logged a second time. VFT uses
three starting T0 values, 0 <= T0 < min(observed temperatures, 298.15), and rejects
boundary/nonconverged/unidentifiable solutions. It requires at least four distinct
temperatures and more observations than the number of fitted parameters. With
constant temperature it can estimate only the approved pressure term at the
actual temperature, explicitly marked as a reference mismatch.

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

## Metrics and stability

Every metric uses the current pair's valid-signature intersection. `n_shared`
counts that intersection; `n_observation_shared` separately counts shared ILs
with usable target observations, including failed/pending signatures. All pairs
remain present. EE diagonal relationship values are NA; its count diagonal is
the number of systems in that node. Directed SE outputs run simulation to
experiment only. Continuous z-scores use ddof=0.

| Metric | Minimum shared count | Definition |
|---|---:|---|
| Spearman | 3 | Rank correlation; constants are NA |
| Distance correlation | 5 | Biased doubly centered distance-matrix estimator |
| Binary MI | 2 | Natural-log MI, high class is strictly above pair median |
| Binary I/H | 2 | MI divided by target-label entropy; zero entropy is NA |
| Multiclass MI | 30 | 3 bins at 30, 4 at 80, 5 at 150; pair quantiles |
| CV-NMAE | 20 | Model OOF absolute error divided by training-median baseline OOF absolute error |

`binary_thresholds` is empty in the approved config; optional entries are keyed
by exact node ID and must use the signature's units. Otherwise each side uses
its own median on the current shared subset. Degenerate quantile cuts do not
fall back to fewer bins. MI uses two discrete resolutions (binary and multiclass)
and directed normalized MI (`I/H`), all with natural logarithms; these do not
measure complete continuous mutual information.

Continuous KSG MI was intentionally removed because its nearest-neighbor estimator
and ordinary paired bootstrap were incompatible in the current auditable pipeline.
Dependency evidence is retained through Spearman, distance correlation, binary MI,
multiclass MI, directed I/H, and CV-NMAE.

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
and IL-solute unit signatures. Signature rows include formula version, source
files, fitted parameters, actual/requested conditions, uncertainty, residual degrees
of freedom and status. Inventory counts expose invalid observations.

`G_EE/` and `G_SE/` contain one CSV per metric, both count matrices, `pairs.csv`,
`na_reasons.csv`, and `confidence.csv`. Rows are sources and columns are targets.
EE symmetric values and their confidence rows are mirrored exactly. Pair records
include exact shared systems and flags for mixed signature kinds, reference
mismatches, heterogeneous actual conditions, extrapolation and pending formulas.
These flags are not automatic filters: inspect them before interpreting metrics.
Missing numeric values are serialized as `NA`; statuses explain why.

`manifest.json` records input/config/script hashes, dependency versions, seed,
settings, output CSV hashes and completion/failure status. Treat a running or
failed manifest as incomplete output. No source data is modified.

Validation:

```sh
python -m pytest -q tests/test_dataset_relationship_graph.py
```
