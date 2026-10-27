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

### Denoised orthogonal condition decomposition

To estimate a condition-space decomposition without competing theoretical and
full-condition columns, choose an explicit alternative method:

```python
pipeline.setup_model(
    method="orthogonal",
    penalty={"component": 1.0, "condition": 0.1},
    weights=None,  # equal condition weights; or positive weights for all condition keys
)
```

The default remains `method="ridge"` with the existing trial-level estimator.
This preserves its ability to model trial-varying predictors. `weights` is
specific to the orthogonal method and is normalized to sum to one, then saved
in `analysis.config.model` with the method and penalties. Orthogonal codes must
be constant within every training condition; pooling different predictor codes
under one condition key raises an error instead of silently averaging them.

Let `U` contain training condition means, `A=[1, X]` the condition-level theory
design, and `W` the fixed positive diagonal condition weights. The estimator is

```text
minimize ||sqrt(W) (U - A Theta - R)||_F^2
         + lambda_component ||Theta_components||_F^2
         + lambda_condition ||sqrt(W) R||_F^2
subject to A' W R = 0
```

The intercept is included in `A` and unpenalized. Theoretical components can be
correlated: they are not sequentially orthogonalized. With the weighted theory
projector `P_A`, the solution has independent parts:

```text
Theta = weighted ridge fit of U to A
R = (I - P_A) U / (1 + lambda_condition)
M = A Theta + R
```

QR projection and an augmented least-squares solve avoid unnecessary normal
equations. Residual patterns are orthogonal to the full theoretical space,
including intercept. Their penalty cannot change theoretical coefficients;
the theoretical penalty still changes their amplitudes. `M` is a denoised
estimate, not an exact decomposition of observed condition means. Because this
loss averages over conditions with total weight one, its penalty magnitudes
are **not interchangeable** with the trial-sum ridge penalties. No automatic
truth-based or held-out-expression-based penalty tuning is performed.

The theory design must have full column rank including intercept to report
identified individual components. Exact aliases raise a targeted error;
near-collinear full-rank codes remain supported with regularization and a
`weighted_theory_condition_number` diagnostic. Regularization improves
stability but cannot create independently identified effects from a redundant
experimental design. No observed zero condition or complete factorial is
required. For A/B/AB conditions, intercept+a+b can be full rank and leave
exactly zero residual condition space; adding an unidentifiable interaction
does not create new information.

Unequal trial counts affect precision of the condition means, not their fixed
scientific weighting. Equal weights define an equal-condition population;
custom positive weights define another stated population. The same weights
apply to every fold and subject. Each training fold must contain every condition.

The result tables and `predictors.role` values are unchanged. In this mode,
`condition[...]` coefficients are constrained residual **condition patterns**,
not coefficients from the overlapping ridge solve or an arbitrary residual
basis. Their weighted orthogonality can be checked from saved condition codes
in `trials` and weights in configuration. Model-rank diagnostics use the effective
condition-space design. Full-data coefficients and fold-only covariance/
expression follow the existing lifecycle. Expression-only conditions are
allowed, and expression still uses the existing training-mean reference:
this change does not include the separately evaluated expression redesign.

The method is an allocation convention, not proof of distinct neural sources.
If a generative omitted effect overlaps the theoretical condition space, that
overlap is assigned to theory by definition. Neither method can identify the
original latent attribution from these data alone. Low trial counts and low
SNR remain sources of variance; strongly shrinking theory can increase bias.

### Paired comparison with current ridge

The reproducible synthetic comparison is
`tests/encoding/benchmark_orthogonal.py`. It uses seed 271032, 100 replicates
per scenario, six incomplete-factorial conditions with no observed zero,
correlated predictors (r approximately .954), eight correlated channels with
average-reference rank loss, and either six trials per condition or uneven
counts `[4,8,5,9,6,4]`. Noise scales are .2, .8 and 1.6. Both estimators see
identical training EEG and independent held-out condition means (20 trials per
condition), on unchanged raw code scales. The predeclared penalty grid is
component `[.1,1,10]` by condition `[.01,.1,1]`; there is no truth-selected winner.

One residual scenario is orthogonal by construction, which favors the new
allocation assumption. A second intentionally has overlapping omitted signal.
Theoretical-pattern error then uses the explicitly projection-defined target,
not the unknowable original latent attribution. At component=1, condition=.1,
with uneven counts, the results are:

| Residual generation | Noise | Theory MSE ridge / orthogonal | Held-out MSE ridge / orthogonal |
| --- | ---: | ---: | ---: |
| Orthogonal | .2 | .01730 / .01198 | .00561 / .01518 |
| Orthogonal | .8 | .03085 / .02721 | .09127 / .07816 |
| Orthogonal | 1.6 | .07186 / .07249 | .35392 / .26922 |
| Overlapping | .2 | .03111 / .02652 | .00557 / .03003 |
| Overlapping | .8 | .04474 / .04161 | .09086 / .09125 |
| Overlapping | 1.6 | .08220 / .08431 | .36284 / .28610 |

Across all 108 scenario/grid combinations, orthogonal mode has lower
theoretical-pattern error in 70, held-out error in 57, and original-generative
coefficient error in 72. These are sensitivity counts from a bounded simulation,
not confidence probabilities or universal superiority. The practical benefit is
removal of residual-penalty allocation dependence and, in several noisy cases,
better recovery/prediction. Low-noise full prediction and some high-noise theory
estimates are worse. The method is therefore explicit rather than a replacement
default. A separate near-collinearity test confirms regularization reduces
coefficient variance relative to unregularized projection; it does not promise
that every setting is more stable than the current, heavily shrunk ridge model.
On a small 32-trial/five-condition/four-channel/two-bin fit, median numerical
fit time over five batches of 1,000 was about 10 microseconds for ridge and
23 microseconds for orthogonal mode. This adds a small QR/least-squares cost;
it is not an end-to-end EEG performance benchmark.

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
completed subjects and retries failed and not-yet-attempted subjects. Model
comparison, time smoothing, Bayesian estimation, and cross-subject
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
