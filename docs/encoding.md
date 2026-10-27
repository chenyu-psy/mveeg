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
    expression_reference={"color": 0},
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

## Observed-code expression references

`encode(expression_reference={"color": 0})` adds an observed reference for
`color` within the encoding pipeline. The mapping is sparse: `number`, `load`,
and any other unlisted components retain their current raw expression.
Omitting the mapping or passing `{}` preserves the existing configuration,
scores, tables, fold assignments and cached-result compatibility exactly.

Reference settings sit beside `expression` because they describe this analysis's
expression output. They do not change the fitted model or numeric coding.
`expression` still maps display groups to raw `target` values;
`expression_reference` maps **expanded numeric component names to observed
numeric code values**, not to those display groups. For example,
`{"color": 0, "number": -1}` configures two different references without
requiring settings for every component. Nonzero and contrast-coded references
are supported. There is no compulsory zero, centering rule, or reference parameter
on `setup_model()`.

Membership uses each selected trial's actual design value after metadata
transformations and formula expansion. Thus a reference for `a:b` matches the
actual product column; it is not a counterfactual intervention on `a` or `b`.
Trial-varying numeric codes remain supported. Matching is exact numerical
equality, without an implicit tolerance. If an expression such as `.3 - .2`
does not exactly equal `.1`, define or round the code explicitly in
`transform_metadata()` before fitting. Unknown components, nonnumeric/nonfinite
references, and values not observed among eligible expression trials fail
explicitly. No fitted, extrapolated or minimum-observed fallback is used.

### Reference statistic and interpretation

The encoder, precision and normalized matched filter are learned solely from
the same training fold as before. Write the existing score as `s_raw`. For
component `k` with reference code `r`, the pipeline calculates

```text
R_fk = {j: actual component code x_jk == r, and j was not used to train model f}
offset_fk(t) = mean of s_raw_jk(t) over j in R_fk
s_referenced_ik(t) = s_raw_ik(t) - offset_fk(t)
```

**Every eligible matching reference trial has equal weight `1 / len(R_fk)`.**
Trials are pooled directly across raw conditions and expression groups. There
is no averaging of condition means, condition balancing, or group-weight option.
For example, if three matching trials in condition A have scores `[1,2,3]` and
one in condition B has score `9`, the reference is `3.75`, not `5.5`. Regrouping
the same eligible trials under different display labels does not change it.
Changing which trials survive QC or changing their number can change this
empirical trial-population reference, as intended.

Negative, zero and positive mean below, equal to and above this observed
reference projection. They do not mean absence/presence of a latent process,
uniquely isolated components, or an EEG state predicted by setting a design
variable to zero. No other-component predictions or condition dummy patterns
are subtracted. The original training mean cancels algebraically from the
referenced difference; it remains the coordinate of saved raw scores.

One common offset is subtracted from all public scores of that component,
fold and time. Differences **within the same fold** are unchanged. Across folds,
filters and offsets differ; trial-weighted condition summaries can have different
fold mixtures, so their aggregate differences are not promised unchanged.
The operation supplies a meaningful empirical origin, not universal lower
variance or independent single-trial evidence.

### Held-out coverage and auxiliary reference evidence

Training folds and public test rows are unchanged. A training-derived reference
trial contributes only to its own held-out model. If a fold has no legal matching
reference observations, that subject fails with component/code/fold context.
Missing individual raw conditions do not matter while matching trials exist;
there is no fixed condition-stratum population or per-condition coverage rule.
The pipeline does not silently repartition training data, borrow fitted scores
from another model, or renormalize a condition-weight scheme.

Expression-only trials never enter any fold's training set. Matching reference
trials of this kind may therefore supply **auxiliary** projections under each
fold model, even when their ordinary public expression was assigned to another
fold. This makes a sparse observed control usable without extra training splits
or data leakage. It follows the eligibility principle used for nontraining
classifier evidence in decoding, while retaining encoding's own estimator.

Auxiliary projections live only in the reference evidence table. They do not
duplicate public expression rows, increase trial counts, or change an unlisted
component's outputs. Within a fold each reference epoch contributes once.
A nonfinite reference projection (including an undefined near-null pattern)
fails explicitly rather than dropping trials or folds.

The pooled reference evidence has zero mean after subtracting its own offset.
An arbitrary displayed group's mean need not be exactly zero: it may mix codes,
contain only a subset of the reference population, or use different public
fold assignments from auxiliary evidence. A sole reference observation can
also center itself to zero by construction; that is not noiseless evidence.
Shared reference trials induce dependence among scores and across folds.
Use participants as inferential units, not reference prediction rows.

### Native result tables and summaries

The original `pattern_expression` and `condition_pattern_expression` remain
unchanged. With a nonempty reference mapping, the pipeline additionally writes:

- `referenced_pattern_expression(subject_index, epoch_index, time, component,
  expression_group, raw_expression, reference_expression, expression, fold)`:
  selected components only, with the same once-only public trial/fold rows;
  `expression = raw_expression - reference_expression`.
- `expression_references(subject_index, fold, component, time, reference_value,
  reference_expression, n_trials)`: actual per-model offsets and reference counts.
- `expression_reference_trials(subject_index, epoch_index, fold, component, time,
  reference_value, weight, raw_expression, auxiliary)`: exact membership,
  `weight=1/N`, projections and auxiliary flags. Join to `trials` for raw metadata.
  Summing `weight * raw_expression` reproduces the offset.

`condition_referenced_pattern_expression` reports participant/condition/component/
time, **trial-weighted** `mean`, distinct `n_trials`, and observed `n_folds`.
`fold_condition_referenced_pattern_expression` reports the corresponding fold
means and distinct trial counts, so fold mixtures remain inspectable. Neither
view provides a trial-based inferential SE; subtracting a shared reference
does not preserve independent errors. The original raw summary view is retained
with its existing descriptive fields; it is not a reference-aware uncertainty
estimator.

These are additive tables/views with explicit fixed columns, preserving result
schema version 1's existing tables. The scientific extension is identified by
`analysis.config.expression_reference` and `expression_reference_estimator`
(version, exact matching, equal-trial weighting and summary conventions).
An existing raw-only file cannot be silently reused for referenced results:
use a different file or `recompute="all"`. Omitting the reference keeps the old
fingerprint. Subject transactions, failure recording, resume and exclusions
also cover the new tables; no fold checkpoint or separate result helper is added.

### Paired validation against raw expression

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
