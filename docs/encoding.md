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
)
pipeline.select_trials(qc="final_status", keep=["accepted"], exclude={})
pipeline.prepare_epochs(
    crop=(-0.3, 2.2),
    time_bin=50,
    drop_channel_types=("eog", "eyegaze", "pupil", "misc"),
    drop_channels=(),
)
pipeline.setup_model()  # orthogonal condition decomposition, the default
pipeline.setup_cv(folds=5, seed=None)
pipeline.encode(
    formula="1 + color + number + load",
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

`setup_model()` and `setup_cv()` are optional. The default model is the
orthogonal condition decomposition (`method="orthogonal"`, component penalty
`0`, condition penalty `0.1`, equal condition weights); the default CV is five
folds with a generated seed. `setup_model(penalty={"component": 10})` changes
one penalty and keeps the method's default for the other.

## Choosing a method

- **`method="orthogonal"` (default)** fits condition means. Use it whenever
  every training condition has one fixed code for each component, which covers
  factorial, incomplete, and parametric condition designs. Its theoretical
  patterns are set by the data, not by a penalty ratio.
- **`method="ridge"`** fits single trials. Use it when a component varies from
  trial to trial within a condition, for example a per-trial covariate or an
  unbinned continuous stimulus value. With condition-constant codes, ridge's
  split between component and condition columns depends on the penalties, so
  its component coefficients are not identified by the data.

The orthogonal method stops with an error that points to ridge when a code
varies within a condition. Results from releases where ridge was the default
are reproduced with `setup_model(method="ridge")`; their result files must be
recomputed with `recompute="all"` (see
[zero-code references](#zero-code-expression-references)).

## Ridge model

With `method="ridge"`, for every subject and time bin, mveeg fits

```text
Y = X_component B_component + Z_condition B_condition + E
```

by minimizing the unscaled sum of squared residuals plus separate component
and condition ridge penalties. The intercept is always present and unpenalized.
The penalty is part of the component decomposition, not a hyperparameter chosen
by cross-validation. Ridge defaults to component penalty `1.0` and condition
penalty `0.1`. The component penalty may be `0`; the condition penalty must stay
positive because the condition one-hot block spans the intercept.

## Formula and conditions

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

The condition keys define the complete condition basis. In ridge, exact overlap
between component and condition spaces is intentional and handled by the
penalties.
For both methods, the theoretical components together with the intercept must
be linearly independent in the training trials of every subject and every fold.
If they are aliased, the data cannot separate their patterns: ridge would still
return coefficients, but the split among the aliased terms would be set by the
penalties. `encode()` therefore stops with an error that names each relation,
for example `interaction = color + number - intercept` when every condition
contains color, number, or both. Remove or recode one term in each relation.
Near-dependence is not an error; the component condition number in
`design_diagnostics` records it.

## Orthogonal condition decomposition (default)

The default method estimates a condition-space decomposition without competing
theoretical and full-condition columns:

```python
pipeline.setup_model(
    method="orthogonal",
    penalty={"component": 0.0, "condition": 0.1},  # the defaults
    weights=None,  # equal condition weights; or positive weights for all condition keys
)
```

`weights` is
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
including intercept.

The two penalties have different roles:

- `component` shrinks the theoretical patterns. It defaults to 0 because the
  theory design must be identifiable, so no shrinkage is needed to estimate
  it. With 0, the patterns do not depend on how codes are scaled: multiplying
  a code by 10 divides its pattern by 10. A positive value is useful only for
  near-collinear codes. It then depends on code scale, and on this mean-scale
  loss it is strong: a 0/1 code with variance .25 keeps 1/(1 + 4 * penalty)
  of its amplitude, about 20% at penalty 1.
- `condition` only rescales the residual condition patterns by
  `1 / (1 + lambda_condition)`. It never changes the theoretical patterns.
  It does change the fitted trial predictions, and therefore the training
  residuals from which the noise covariance, and so the expression metric,
  are estimated.

With `component=0` and weights proportional to training-trial counts, the
theoretical patterns equal ordinary least squares of the trials on the codes. `M` is a denoised
estimate, not an exact decomposition of observed condition means. Because this
loss averages over conditions with total weight one, its penalty magnitudes
are **not interchangeable** with the trial-sum ridge penalties. No automatic
truth-based or held-out-expression-based penalty tuning is performed.

The theory design must have full column rank including intercept to report
identified individual components. Exact aliases stop `encode()` with the same
error as ridge, which names each relation;
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

### Paired comparison with ridge

The reproducible synthetic comparison is
`tests/encoding/benchmark_orthogonal.py`. It uses seed 271032, 100 replicates
per scenario, six incomplete-factorial conditions with no observed zero,
correlated predictors (r approximately .954), eight correlated channels with
average-reference rank loss, and either six trials per condition or uneven
counts `[4,8,5,9,6,4]`. Noise scales are .2, .8 and 1.6. Both estimators see
identical training EEG and independent held-out condition means (20 trials per
condition), on unchanged raw code scales. The predeclared penalty grid is
component `[.1,1,10]` by condition `[.01,.1,1]`; there is no truth-selected winner.
The grid predates the component default of 0 and does not include it.

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
estimates are worse. A separate near-collinearity test confirms regularization reduces
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

## Zero-code expression references

Raw `pattern_expression` is measured from the training-mean trial, so its zero
depends on the mix of training trials. For figures, a more natural zero is often
a baseline condition. `encode()` adds this automatically, with no argument:

- A component is **referenced** when its code is constant within each training
  condition and at least one training condition has code exactly `0`. For
  `presence` coded 0/1, the reference is every trial with `presence == 0`.
- Components without a code-0 condition (for example set size coded 2/4, or a
  -0.5/0.5 contrast) and components that vary within a condition keep the raw
  zero only.
- The same rule covers interactions: `a:b` is referenced to every trial whose
  product is 0, that is, every trial without both `a` and `b`.
- Matching is exact. A code computed as `0.1 + 0.2 - 0.3` is not 0; round or
  define codes explicitly in `transform_metadata()`.

Every subject has every training condition, so every subject gets the same
referenced components. If they differ anyway, encoding stops with an error.

### What the reference does

The fitted model, filters, and noise estimates are unchanged. For component `k`,
fold `f`, and time `t`:

```text
R_fk         = code-0 trials that did not train the fold-f model
offset_fk(t) = mean raw expression of R_fk at t (equal trial weights)
referenced   = raw expression - offset_fk(t)
```

So 0 means "same as code-0 trials at that time point". One offset is subtracted
from every trial of a fold, so differences between conditions are unchanged
within a fold; across folds they can move slightly when conditions are unevenly
spread over folds. Referencing changes where 0 sits in figures, not condition
differences or tests on them. Compare conditions; tests of a single condition
against 0 are not meaningful for raw or referenced expression.

Training-condition reference trials contribute only to the fold that held them
out. Expression-only code-0 trials never train any model, so they are projected
under every fold as **auxiliary** reference evidence. They appear only in the
evidence table and never duplicate public rows. A fold without held-out code-0
trials, or a nonfinite reference projection, stops with an error.

### Result tables

`pattern_expression` and `condition_pattern_expression` are unchanged. When at
least one component is referenced, encoding also writes:

- `referenced_pattern_expression(subject_index, epoch_index, time, component,
  expression_group, raw_expression, reference_expression, expression, fold)`:
  referenced components only; `expression = raw_expression -
  reference_expression`.
- `expression_references(subject_index, fold, component, time, reference_value,
  reference_expression, n_trials)`: the offset of each fold model.
- `expression_reference_trials(subject_index, epoch_index, fold, component, time,
  reference_value, weight, raw_expression, auxiliary)`: which trials formed each
  offset, with `weight = 1/N`.

The views `condition_referenced_pattern_expression` (trial-weighted `mean`,
`n_trials`, `n_folds`) and `fold_condition_referenced_pattern_expression`
summarize them. Neither reports a trial-based SE, because a shared offset makes
trial errors dependent; use participants as the inferential unit.

`analysis.config.expression_reference` records the rule. Result files written
before automatic references lack it, so they report incompatible settings and
need `recompute="all"`.

### Synthetic check

`tests/encoding/benchmark_expression_reference.py` is a fixed-seed synthetic
comparison through the actual CV fitting path: 200 paired replicates at each
of noise-only, weak (.25) and strong (1) signal levels, four folds, uneven
condition counts `[7,11,8,5]`, expression-only counts `[1,3]`, correlated
predictors and 12 correlated average-referenced channels. The same fitted
models supply raw and referenced scores; there is no truth-based parameter
tuning or extra half-training split.

| Signal | Displayed zero-code A mean, raw / referenced | A SD, raw / referenced | D−A mean, raw / referenced |
| --- | ---: | ---: | ---: |
| Weak .25 | −.4594 / .0194 | .5502 / .5062 | .9466 / .9427 |
| Strong 1 | −2.8842 / −.0042 | .5937 / .5938 | 6.3438 / 6.3126 |

Here A and the expression-only reference observations have the same generative
mean, so the approximately zero referenced A mean demonstrates the intended
observed-reference origin in this case, not a guarantee for every zero-coded
condition under other components or omitted effects. Noise-only D−A means
were .0058 ± .0744 Monte Carlo SE raw and −.0011 ± .0738 referenced: there was
no demonstrated new systematic gradient. Within-fold differences were
preserved within `1.8e-15`; the pooled reference evidence centered itself
within `1.1e-15`. Aggregate D−A can change because folds have unequal mixtures.
These bounded results show useful origin calibration without the previous
independent-half fitting variance cost. They do not establish universal
variance improvement or neural-source identification.

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
