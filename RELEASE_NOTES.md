# Unreleased

## Changed

- **Breaking:** encoding result files written by earlier releases must be
  recomputed with `recompute="all"`. Running an existing script against one
  reports incompatible analysis settings, because the analysis configuration
  now records the orthogonal default and the automatic reference rule below.
  Encoding refits quickly (about 2 s per subject in a 939-trial, 31-channel
  dataset, most of it loading).
- **Breaking:** encoding now uses `method="orthogonal"` by default. It fits
  condition means, so its component patterns are identified by the data rather
  than by the ridge penalty ratio. Ridge remains available with
  `setup_model(method="ridge")` and is required when a component varies within
  a condition; the orthogonal method stops with an error that says so. Add
  `setup_model(method="ridge")` to reproduce results from earlier releases.
- Encoding now stops with an error when formula components are aliased with
  each other or with the intercept in a subject's training trials or in any
  training fold. The message names each relation, such as
  `interaction = color + number - intercept`. Earlier releases fitted these
  designs and only recorded the rank deficiency in `design_diagnostics`, but the
  patterns of aliased components were set by the ridge penalties rather than by
  the data. Formulas without aliased components give the same results as
  before.

## New

- Add explicit `setup_model(method="orthogonal", weights=...)` for denoised
  condition-space decomposition with theoretical-orthogonal residual patterns.
  The orthogonal method defaults to an
  unpenalized component block (`component=0.0`), so its theoretical patterns do
  not depend on code scale; its condition penalty only rescales residual
  condition patterns. Document estimator-specific penalty scales, identifiable
  condition codes, and paired synthetic accuracy tradeoffs.
- Encoding automatically references every component whose condition design
  contains code 0: per fold and time, the mean held-out expression of code-0
  trials is subtracted, so 0 means "same as code-0 trials". Raw expression and
  condition differences are unchanged. Three new tables
  (`referenced_pattern_expression`, `expression_references`,
  `expression_reference_trials`) and two views hold the referenced scores,
  offsets, and the trials behind each offset.

# mveeg 0.4.0

mveeg 0.4.0 adds dataset-level subject exclusions, accepts many inputs that
earlier releases rejected, and moves to AutoReject 0.5.2. It requires Python
3.11 or newer. Inputs that worked in 0.3.1 give the same results and
fingerprints, so existing datasets, artifact labels, and result files stay
current.

## Requirements

- Python 3.11 or newer; Python 3.10 is no longer supported.
- AutoReject 0.5.2 or newer, which fits much faster than 0.4.x. The AutoReject
  version is recorded as provenance only, so upgrading does not mark existing
  preprocessed datasets as changed.

## New

- `DatasetPipeline.select_subjects(exclude={subject_index: reason})` saves a
  dataset-level exclusion list in `subject_selection.tsv`. Decoding and Encoding
  skip excluded subjects without changing the analysis fingerprint.

## More flexible inputs

- Epoch metadata with missing values (NaN, `pd.NA`) can now be fingerprinted,
  and integer mapping keys such as `{2: "low"}` are accepted.
- `classes`, `evidence`, `conditions`, `exclude` rules, `store_metadata`, and
  `hf_noise.band` accept NumPy arrays and pandas objects, for example
  `df["load"].unique()`.
- `n_jobs=-1` uses every CPU; NumPy integers are accepted.
- `prepare_epochs()` accepts `time_bin=50.0` and `None` for either `crop` bound.
- `setup_model(penalty={"component": 10})` keeps the default for the omitted
  penalty, and the component penalty may be `0`.
- Result paths without a `.duckdb` suffix get one appended, so dotted names such
  as `load_tmin-0.2` work.
- `progress` accepts any true/false value, and `ReviewSession` accepts NumPy
  integers for its window settings.
- `load_eyelink()` reads monocular recordings, taking the tracked eyes from the
  `SAMPLES` header, and ignores extra sample columns.
- `load_behavior()` stacks several matched files (for example one per block) in
  sorted order.
- `merge_metadata()` and `ReviewSession` ignore metadata rows without a
  matching epoch, such as practice trials.
- Study-wide `ignore_channels`, `hide_channels`, and
  `autoreject.exclude_channels` lists warn on and skip channels a subject lacks.

## Results and compatibility

- Analysis fingerprints no longer include the mveeg version or the dataset
  location. Upgrading mveeg or moving a dataset keeps result files compatible.
  Files from earlier releases are checked against their stored configuration,
  and their fingerprint is updated in place.
- Subjects may store channels in a different order, and channel tables are
  compared without dtype or float-rounding differences.
- Epochs without electrode positions run with NaN topography coordinates.
- A saved exclusion for a subject no longer in the dataset is ignored with a
  warning instead of stopping model runs.

## Fixes

- Recomputed artifact tables are written only when their contents change, and
  applied labeling rules are saved even when the table is unchanged.
- `recompute="never"` preserves the provenance of existing artifact tables, so a
  later `recompute="changed"` still applies pending rule changes.
- An array given as a `select_trials()` `exclude` rule no longer fails.
- AutoReject no longer warns when `consensus` is omitted; the default grid is
  still saved in the diagnostics.
