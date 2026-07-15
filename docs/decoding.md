# Decoding API and result contract

The decoding API has its own pipeline initialized from a manifest-backed
dataset root. Configuration is lazy: initialization validates the dataset
without loading signal data, the setup methods record an analysis, and
`decode()` refreshes the manifest, loads subjects, fits models, and writes one
DuckDB file. The terminal method returns `None`.

## Lifecycle

```python
from mveeg import decoding

pipeline = decoding.init_pipeline("data/preprocessed")
pipeline.transform_metadata(
    condition=lambda x: x["raw_condition"].map(condition_map),
    load=lambda x: x["set_size"].astype(float),
)
pipeline.select_trials(qc="final_status", keep=["accepted"], exclude={})
pipeline.prepare_epochs(crop=(-0.2, 0.8), time_bin=50)
pipeline.setup_classifier(classifier="logistic_regression", solver="lbfgs", max_iter=1000)
pipeline.setup_cv(
    folds=5,
    repeats=20,
    trial_averaging=5,
    permutations=0,
    seed=None,
)
pipeline.decode(
    target="condition",
    classes={"low": ["SS2"], "high": ["SS4"]},
    evidence={"low": ["SS2"], "high": ["SS4"], "probe": ["probe"]},
    generalization={
        "low": ["SS2", "probe_low"],
        "high": ["SS4", "probe_high"],
    },
    output="mean",
    file="results/decoding.duckdb",
    recompute="never",
    n_jobs=6,
    progress=True,
)
```

`target` names a column after the optional metadata transform. `classes` is an
ordered mapping from classifier labels to raw target values and must define at
least two labels. When `evidence=None`, evidence groups exactly inherit
`classes`. An explicit evidence mapping may add non-training conditions, but it
must include every raw value used by `classes`.

`generalization` is independent of `evidence`. It accepts `None` or a mapping
from classifier labels to raw values in the same post-transform `target`
column. Its keys must exist in `classes`; a subset is allowed. Each raw value
produces its own temporal-generalization matrix and may be absent from
`evidence`. A raw value may appear under only one generalization label, but its
training label may differ because the two mappings describe different
scientific roles. Combine raw conditions with `transform_metadata()` before
decoding when they should form one scientific condition. Boolean values and an
implicit "all" mode are not accepted.

`transform_metadata()` accepts only named variable definitions. Each value is
a function that receives the current metadata DataFrame and returns one scalar
or one trial-aligned column. Variables run in written order, so later variables
may use columns created earlier in the same call. The transform cannot replace
`subject_index` or `epoch_index`. mveeg fingerprints the resulting variable
values per subject; callers do not provide transform names or versions.

`trials.subject` is the canonical `subject_index`. A metadata column named
`subject` is omitted only when it contains the same identities; a conflicting
column remains a schema error.

`prepare_epochs()` expresses `crop` in seconds and `time_bin` in milliseconds.
The result tables use milliseconds without a unit suffix. `time` is the bin
center; exact `start` and `end` values are stored in `time_bins`.

## Classifiers

`setup_classifier()` accepts three built-in linear classifiers:

| Name | Estimator | Default parameters |
| --- | --- | --- |
| `logistic_regression` | `sklearn.linear_model.LogisticRegression` | `solver="lbfgs"`, `max_iter=1000` |
| `lda` | `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` | estimator defaults |
| `linear_svm` | `sklearn.svm.SVC` | `kernel="linear"`, `probability=False` |

Additional keyword arguments pass to the selected estimator. Arbitrary
estimator objects are not accepted. mveeg stores `decision_function()` output
without truncation, reordering, normalization, or sign changes. Binary output
is a scalar `DOUBLE`; multiclass output is a `DOUBLE[]` whose shape is recorded
in `classifier.evidence_shape`. Users who alter classifier parameters are
responsible for interpreting that estimator's native decision geometry.

## Cross-validation

For every repeat, mveeg randomly downsamples all training classes to the
smallest class and applies stratified K-fold CV. Every selected training trial
is held out exactly once in that repeat. Trial averaging occurs only inside the
training fold: complete same-class groups are averaged, incomplete groups are
dropped, and `trial_averaging=1` keeps single-trial training. The scaler is fit
only on these training observations. Held-out observations always remain
single trials.

Subjects are prepared, decoded, and committed one at a time. Within the active
subject, `n_jobs` controls the maximum number of complete repeats evaluated in
parallel; balancing, folds, and permutation labels remain contained within
their repeat. `progress=True` shows one standard repeat-level progress bar for
each subject that is actually computed. Progress display and `n_jobs` do not
enter the analysis fingerprint.

At each fold and training time, one fitted model supplies observed accuracy,
raw confusion counts, classifier evidence, and Haufe patterns. When a
`generalization` mapping is present, that same model is additionally evaluated
at every test time for each listed target condition. Generalization produces
accuracy only; it does not create a second evidence output.

Training-class trials receive evidence only from the model that held them out.
Evidence-only trials are never used for fitting and receive evidence from all
fold models. With `output="mean"`, evidence is averaged element by element and
`n_models` records the number of contributing models.

Trial selection uses the union of class, evidence, and generalization target
values. Generalization conditions that are also training conditions receive
predictions only from their held-out fold model. Generalization-only conditions
receive predictions from every fold model. Conditions listed only in evidence
never enter generalization, and conditions listed only in generalization never
enter `classifier_evidence`.

## Patterns and permutations

Haufe patterns are computed on the original sensor scale as

```text
A = Cov(X, S) @ pinv(Cov(S))
```

where `X` is the unstandardized training data and `S` is the fitted component
score. Logistic regression and LDA multiclass components correspond to fitted
classes. Linear SVM components correspond to the estimator's one-versus-one
coefficient pairs. Component mappings are stored in `pattern_components`.
Patterns are neither sign-flipped nor combined into predefined time windows.
See [Haufe et al. (2014)](https://pubmed.ncbi.nlm.nih.gov/24239590/) and the
[scikit-learn SVM guide](https://scikit-learn.org/stable/modules/svm.html).

`permutations=N` adds permutation IDs `1..N`; ID `0` is always the observed
model. Each permutation uses one repeat-level label shuffle consistently at all
times and reuses that repeat's balanced trials and folds. The same shuffled
labels define training and same-time held-out testing. Generalization targets
always remain fixed by the generalization mapping, including for trials that
also belong to the training pool. A shuffle that cannot form every requested
training average is redrawn. Permutations produce only accuracy and, when
enabled, generalization accuracy.

## DuckDB tables

Supporting tables:

- `analysis(version, output, generalization, seed, config, fingerprint, created, updated)`, where `generalization` is JSON or `NULL`
- `subjects(subject, status, fingerprint, reason, updated)`
- `trials(subject, trial, class, evidence_group, <selected post-transform metadata>)`
- `classifier(name, parameters, classes, evidence_shape)`
- `pattern_components(component, classes)`
- `channels(channel, x, y)`
- `time_bins(time, start, end)`

Result tables for `output="mean"`:

- `accuracy(subject, time, permutation, accuracy, n_correct, n_trials)`
- `classifier_evidence(subject, trial, time, evidence, n_models)`
- `confusion_matrix(subject, time, actual, predicted, count)`
- `patterns(subject, time, channel, component, pattern)`
- `generalization(subject, condition, train_time, test_time, permutation, accuracy, n_correct, n_trials)`

For `output="all"`, `accuracy`, `confusion_matrix`, `patterns`, and
`generalization` add `repeat, fold`; `classifier_evidence` uses
`subject, trial, repeat, fold, time, evidence`. The `generalization` table is
not created when `generalization=None`. Its `condition` column preserves the
post-transform target value, and mean accuracy is weighted from accumulated
`n_correct` and `n_trials`. Confusion values are accumulated raw counts and are
never normalized.

## Incremental execution

When `seed=None`, the first run generates a seed and stores it in `analysis`.
Later compatible runs reuse that seed. Configuration changes produce an error
unless `recompute="all"` replaces the full analysis.

- `never` reuses complete subjects even when their input fingerprint changed
  and computes only missing, pending, failed, or newly added subjects.
- `changed` also recomputes subjects whose inputs changed.
- `all` recomputes every requested subject; with a changed configuration, it
  atomically resets the result schema first.

Each subject's result rows are committed together. `subjects.status` is
`pending`, `complete`, or `failed`; failures keep their reason. There is no
fold-level checkpoint. Subjects that do not need recomputation are reused
without loading their epochs or displaying a progress bar.
