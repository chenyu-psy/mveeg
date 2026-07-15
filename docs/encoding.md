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

- `analysis(version, seed, config, fingerprint, created, updated)`
- `subjects(subject, status, fingerprint, reason, updated)`
- `trials(subject, trial, training_group, expression_group, <post-transform metadata>)`
- `predictors(predictor, term, role, penalty)`
- `channels(channel, x, y)`
- `time_bins(time, start, end)` in milliseconds

Primary results:

- `coefficients(subject, time, channel, predictor, beta)`
- `pattern_expression(subject, trial, time, component, expression_group, expression, fold)`
- `design_diagnostics(subject, fold, diagnostic, predictor, value, threshold, status, message)`
- `covariance_diagnostics(subject, fold, time, n_train_trials, n_channels, rank, condition_number, log_determinant, shrinkage, status)`

`predictors.role` is `intercept`, `component`, or `condition`. Coefficients
include all three roles; pattern expression includes components only.

`condition_pattern_expression` is a DuckDB view over the trial expression rows.
It reports subject, expression group as `condition`, component, time, mean, SD,
SE, and distinct trial count. Other summaries are intentionally not stored.
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
`subjects` without partial result rows. Model comparison, time smoothing,
Bayesian estimation, and cross-subject hierarchical models are outside the 0.3
contract.
