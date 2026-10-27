# Encoding API and result contract

## Lifecycle

`encoding.init_pipeline()` binds an analysis to one prepared or preprocessed
mveeg dataset root. Initialization validates its manifest and provenance but
does not load EEG data.

```python
from mveeg import encoding

pipeline = encoding.init_pipeline("data/preprocessed")
pipeline.transform_metadata(
    color=lambda x: x["color_count"].gt(0).astype(float),
    number=lambda x: x["number_count"].gt(0).astype(float),
    load=lambda x: (x["color_count"] + x["number_count"] - 1).astype(float),
    interaction=lambda x: (x["color"].eq(1) & x["number"].eq(1)).astype(float),
)
pipeline.select_trials(qc="final_status", keep=["accepted"], exclude={})
pipeline.prepare_epochs(
    crop=(-0.3, 2.2),
    time_bin=50,
    drop_channel_types=("eog", "eyegaze", "pupil", "misc"),
    drop_channels=(),
)
pipeline.setup_model(penalty={"component": 1.0, "condition": 0.1})
pipeline.setup_cv(folds=5, seed=None)
pipeline.encode(
    formula="1 + color + number + load + interaction",
    target="condition",
    conditions=TRAINING_CONDITIONS,
    expression=EXPRESSION_CONDITIONS,
    file="results/condition_variables.duckdb",
    recompute="never",
    n_jobs=5,
    progress=True,
)
```

`transform_metadata()` evaluates its keyword functions in order after artifact
fields are merged by `subject_index + epoch_index`. Each function returns one
scalar or trial-aligned column. Derived values, not callable names or manually
maintained versions, enter each subject's input fingerprint.

`select_trials()` handles only QC eligibility and generic exclusions.
`prepare_epochs()` crops in seconds, averages EEG into millisecond bins, and
removes requested channels. The fitted response remains in the original MNE
EEG units; neither EEG channels nor formula predictors are z-scored.

`setup_model()` and `setup_cv()` are optional. Their defaults are component
penalty `1.0`, condition penalty `0.1`, five folds, and a generated seed.

## Model and formula

For every subject and time bin, mveeg fits

```text
Y = X_component B_component + Z_condition B_condition + E
```

by minimizing the unscaled sum of squared residuals plus separate component
and condition ridge penalties. The intercept is always present and unpenalized.
The penalty is part of the component decomposition, not a hyperparameter chosen
by cross-validation.

The formula contains RHS component terms only. Numeric additive terms, `a:b`,
and two-way `a * b` are supported. A response, `0`/`-1`, formula functions,
automatic categorical coding, and `(1 | condition)` are rejected. Create
numeric component codes explicitly with `transform_metadata()`.

`conditions` and `expression` map named groups to raw `target` values:

```python
conditions = {
    "A": ["A"],
    "B": ["B"],
    "A+B": ["A+B"],
}
expression = {
    **conditions,
    "probe": ["probe_1", "probe_2"],
}
```

Every raw value maps to at most one group within each mapping. `expression`
must contain all training values but may use different group labels and add
values that never enter fitting. `expression=None` copies `conditions`.

The condition keys define the complete one-hot condition basis. Exact overlap
between component and condition spaces is intentional and handled by ridge.
Exact dependency among the theoretical component columns is handled by the
same positive ridge penalties. Component and full-model rank deficiency remain
visible in `design_diagnostics` rather than blocking estimation.

## Cross-validation, covariance, and expression

Training trials are split with one shuffled stratified K-fold partition. Every
training condition must contain at least `folds` eligible trials. Nontraining
expression trials are distributed across folds, and every selected trial is
evaluated exactly once. There are no repeats and no `output="mean" | "all"`
dimension. Subjects run one at a time; `n_jobs` controls parallel folds within
the active subject.

Within a fold, the complete component-plus-condition model is fit only on its
training trials. Channel covariance is estimated from that model's training
residuals at each time bin. Residuals are converted to correlation scale,
Ledoit-Wolf shrinkage is applied, and the result is transformed back to raw
covariance and precision scale. The estimator is fixed and has no public
selector.

Held-out EEG is centered by the training-fold channel/time mean. Signed
component expression is

```text
s = y' inv(Sigma) beta / sqrt(beta' inv(Sigma) beta)
```

After held-out expression is complete, mveeg fits the same model once to all
training trials. Only this full-data raw-scale beta is written to
`coefficients`; fold betas remain internal.

## DuckDB tables

Supporting tables:

- `analysis(schema_version, mveeg_version, seed, config, fingerprint, created, updated)`
- `subjects(subject_index, status, fingerprint, reason, updated)`
- `trials(subject_index, epoch_index, training_group, expression_group, <post-transform metadata>)`
- `predictors(predictor, term, role, penalty)`
- `channels(channel, x, y)`
- `time_bins(time, start, end)` in milliseconds

Primary results:

- `coefficients(subject_index, time, channel, predictor, beta)`
- `pattern_expression(subject_index, epoch_index, time, component, expression_group, expression, fold)`
- `design_diagnostics(subject_index, fold, diagnostic, predictor, value, threshold, status, message)`
- `covariance_diagnostics(subject_index, fold, time, n_train_trials, n_channels, rank, condition_number, log_determinant, shrinkage, status)`

`predictors.role` is `intercept`, `component`, or `condition`. Coefficients
include all three roles; pattern expression includes components only.

`condition_pattern_expression` is a DuckDB view over the epoch expression rows.
It reports `subject_index`, expression group as `condition`, component, time,
mean, SD, SE, and distinct epoch count. Other summaries are intentionally not stored.
Time-window averaging, GFP normalization, group topographies, inferential
models, and plotting belong in downstream R code using `coefficients`,
`predictors`, and `channels`.

## Incremental execution

With `seed=None`, the first run generates and stores a seed; compatible later
runs reuse it. Configuration changes error unless `recompute="all"` atomically
replaces the analysis.

- `never` computes missing or failed subjects and reuses completed subjects.
- `changed` additionally recomputes subjects whose manifest files or derived
  metadata values changed.
- `all` recomputes every manifest subject.

Each subject is committed transactionally. A failure records its reason in
`subjects` without partial result rows, then raises immediately with the
subject identifier; later subjects are not attempted. A later call reuses
completed subjects and retries failed and not-yet-attempted subjects. Time
smoothing, Bayesian estimation, and cross-subject
hierarchical models are outside the 0.3 contract.

Common transaction, version, fixed-column, and unitless channel-coordinate
rules are documented once in the [DuckDB result contract](results.md).

## Dataset-level subject selection

Record shared exclusions once after preprocessing and manual review with
`prep.open_pipeline(dataset).select_subjects(exclude={"4001": "Poor signal"})`.
This pipeline automatically reads the saved exclusions on every execution;
no additional analysis configuration is needed. Excluded subjects remain in
`subject_indices`, but are skipped before loading or fitting. Existing results
are retained with `recompute="never"` and deleted with `"changed"` or `"all"`.
See [the review workflow](quality.md#subject-selection-after-review) and
[the result contract](results.md#dataset-level-subject-exclusions).

## Crossvalidated theoretical-subspace strength

`compare()` is an independent analysis on the same prepared dataset. It uses
the metadata transformations, QC selection, and epoch preparation above, but
does not use `setup_model()`, ridge coefficients, expression groups, or pattern
expression. Write comparison results to a separate DuckDB file.

```python
pipeline.setup_cv(folds=5, seed=42)
pipeline.compare(
    models={
        "color": "color",
        "color_and_number": "color + number",
        "load": "load",
    },
    target="condition",
    conditions=TRAINING_CONDITIONS,
    groups="run",  # independently acquired runs/blocks, not arbitrary trial bins
    nuisance=None,  # intercept is always removed; optional common numeric RHS terms
    weights=None,   # equal condition weights; or a positive mapping with these keys
    file="results/subspace_strength.duckdb",
    recompute="never",
    progress=True,
)
```

Model names define the order of paired comparisons: `delta_strength` is the
second model's strength minus the first model's strength. Numeric formula
syntax matches `encode()`, with `"1"` additionally allowed as a zero-rank null
model. Predictors and common nuisance terms must be constant within each
training condition. Grouping multiple raw conditions with different codes is
rejected rather than averaged silently. Correlated or aliased model columns
are allowed: only their joint span is scored, not individual coefficient energy.
No observed all-zero condition, full factorial, or orthogonal component codes
are required.

### Estimator and meaning

Let `U_r` contain unbiased raw condition means from independent group `r`, and
let `W` be the fixed diagonal condition weights normalized to sum to one. With
common nuisance design `N` (including the intercept), define

```text
A = (I - P[sqrt(W) N]) sqrt(W)
Z_r = A U_r
B_M = A X_M
P_M = orthogonal projector onto col(B_M)
S_rs(M) = trace(Z_r' P_M Z_s Q_{-rs})
S(M) = mean over unordered pairs r < s of S_rs(M)
```

`P[...]` denotes a joint Euclidean projector. There is no divisor by model
rank or sum of separate component energies. Negative finite-sample strengths
and increments are retained. All models share weights, nuisance removal,
evaluation groups, channel support, and precision within each pair/time bin.

For nested spaces, `P_2 - P_1` projects onto the additional directions, so the
expected paired difference measures signal energy in those directions. Null
additional directions have zero expected increment **conditional on the
complement-trained metric**, assuming zero cross-group error covariance and
fixed theory/nuisance coding. Low SNR increases uncertainty; the estimator
does not impose positive estimates. Estimated shrinkage precision is not an
exactly unbiased estimator of population inverse noise covariance; the null
crossproduct property does not make its absolute scale population-unbiased.

For nonnested spaces, the difference compares captured signal energy; it is
not a unique added-component effect. Same-span theories necessarily tie.
Larger spaces can capture genuine signal outside a favored theory: absence of
null dimension bias is not general complexity-neutral theory selection.
Do not select models, weights, or nuisance terms on these same evaluation
estimates and then claim the fixed-model null expectation.

### EEG noise, rank, and independent groups

Every evaluation group must contain every requested condition, with at least
three groups overall. Unequal trial counts are allowed; condition means remain
unbiased, and fixed weights determine the scientific condition population.
The estimator does not weight a condition or pair by observed EEG strength.
With `groups=None`, shuffled stratified trial folds from `setup_cv()` are used;
this explicitly assumes exchangeable, independent trial errors across folds.
If temporally blocked trials share drift or condition-dependent run artifacts,
use an independent run/block column. Crossvalidation cannot distinguish a
condition effect completely confounded with run; missing group-condition
cells raise an error rather than being imputed.

For each pair, all remaining groups supply a pooled within-condition residual
noise estimate. At least two residual degrees of freedom are required. Neither
evaluation group's EEG enters its precision estimate. Numerical channel
support is determined by SVD of the complement residuals, accommodating
average-reference/interpolation rank loss. Ledoit-Wolf eigenvalue shrinkage
is applied within that support; covariance scale is corrected by residual
degrees of freedom. Precision is zero outside the estimated support.
This is a complement-trained noise metric: when trials are fewer than channels,
the support is also sample-limited and can vary across pairs, reducing power
and changing the effective metric. Saved rank, trial-count, shrinkage, and
conditioning diagnostics expose this limitation; shrinkage cannot restore
unobserved independent channels.

No additional dataset is needed. Five groups require ten pair-specific noise
fits per time bin, each from three groups. Subjects and pairs execute serially.
Nearby time bins and overlapping group pairs are dependent; use participants
as inferential units, not pair products or time bins as independent replicates.
The package does not compute group inference or pair-based standard errors.

### Comparison result contract

The separate file uses the common `analysis` and `subjects` state layout,
schema version 1, and `analysis.config.analysis_kind="subspace_strength"`.
Configuration records ordered model names/formulas, common nuisance, normalized
weights, group column, seed, preparation, and estimator/precision conventions.
`recompute`, input fingerprints, exclusions, and transactional failure behavior
match `encode()`. An existing encoding result cannot be reused as a comparison file.

Saved tables:

- `trials`, `channels`, `time_bins`: the selected metadata and prepared EEG geometry.
- `trial_groups(subject_index, epoch_index, condition, group_label)`:
  actual evaluation assignments, including randomly generated partitions.
- `model_designs(subject_index, model, condition, term, value, weight)`:
  the numeric theoretical coding used for each subject.
- `pair_strengths(subject_index, group_a, group_b, time, model, strength)`:
  paired estimates retained for checking, not independent inferential samples.
- `strengths(subject_index, time, model, strength, model_rank)`:
  one pair-averaged estimate per participant/model/time bin.
- `comparisons(subject_index, time, model_1, model_2, delta_strength, nested, same_span)`:
  `nested` means model 1's space is contained in model 2's space.
- `covariance_diagnostics(subject_index, group_a, group_b, time, n_noise_trials,
  noise_df, channel_rank, n_channels, shrinkage, condition_number)`.

The estimator follows crossvalidated pattern-energy reasoning; see
[Allefeld and Haynes (2014)](https://arxiv.org/abs/1401.4122) and
[Diedrichsen and Kriegeskorte (2017)](https://doi.org/10.1371/journal.pcbi.1005508).
Its weighted projector and complement-trained shrinkage metric are specified
above rather than claiming identity to every published estimator.
