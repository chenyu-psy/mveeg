# mveeg

`mveeg` is a Python package for preparing EEG datasets and running multivariate
decoding or encoding analyses. It keeps trial metadata and processing history
with the data, makes subject-level analyses easy to resume, and writes results
to portable DuckDB files.

Preparation, decoding, and encoding are independent workflows. Use only the
parts your study needs.

## What mveeg helps you do

- Build consistent datasets from continuous recordings or existing MNE Epochs.
- Apply eligibility checks, AutoReject, artifact rules, and manual review.
- Run multivariate decoding, temporal generalization, or linear encoding.
- Keep trial identity and provenance consistent from preparation to results.

## Installation

Install mveeg 0.3.0 directly from its GitHub Release:

```bash
python -m pip install https://github.com/chenyu-psy/mveeg/releases/download/v0.3.0/mveeg-0.3.0-py3-none-any.whl
```

Python 3.10 or newer is required. Runtime dependencies, including MNE 1.12 or
newer, are installed automatically.

Check the installed version:

```bash
python -c "import mveeg; print(mveeg.__version__)"
```

## Quickstart

Choose the starting point that matches your data.

### Start with continuous recordings

Create a prepared dataset from subject folders containing BrainVision files:

```python
from mveeg import prep

pipeline = prep.init_pipeline("data/raw", subject_pattern="sub*")
pipeline.load_eeg("*.vhdr", preload=False)
pipeline.make_epochs(
    event_id={"stimulus": 1},
    time_window=(-0.2, 1.0),
    baseline=(-0.2, 0),
)
prepared = pipeline.build_epochs("data/prepared", task="memory")
```

The raw pipeline can also load EyeLink and behavioral data, synchronize trials,
and create derived metadata before building the dataset.

### Start with existing MNE Epochs

If preprocessing begins in project code, add each subject's `mne.Epochs`
directly:

```python
from mveeg import prep

pipeline = prep.init_external(subject_index="4001", data=epochs)
prepared = pipeline.build_epochs("data/prepared", task="memory")
```

Use `merge_metadata()` before building when behavior is stored separately.

### Preprocess and review the dataset

Open a prepared dataset and write preprocessing output to a new dataset root:

```python
from mveeg import prep

prepared = prep.open_pipeline("data/prepared")
preprocessed = prepared.preprocess_epochs(
    "data/preprocessed",
    eligibility=eligibility_config,
    autoreject=autoreject_config,
    recompute="changed",
)
preprocessed.label_artifacts(reject=reject_rules, review=review_rules)
```

Eligibility thresholds and artifact rules are study-specific and should remain
explicit in the study's analysis code.

### Run decoding

```python
from mveeg import decoding

pipeline = decoding.init_pipeline("data/preprocessed")
pipeline.prepare_epochs(crop=(-0.2, 0.8), time_bin=50)
pipeline.setup_cv(folds=5, repeats=20, trial_averaging=5, seed=1)
pipeline.decode(
    target="condition",
    classes={"low": ["SS2"], "high": ["SS4"]},
    file="results/decoding.duckdb",
)
```

The decoding pipeline also supports QC-based trial selection, classifier
evidence, permutations, and temporal generalization.

### Run encoding

```python
from mveeg import encoding

pipeline = encoding.init_pipeline("data/preprocessed")
pipeline.transform_metadata(
    color=lambda frame: frame["color_count"].gt(0).astype(float),
    number=lambda frame: frame["number_count"].gt(0).astype(float),
)
pipeline.encode(
    formula="1 + color + number",
    target="condition",
    conditions={"color": ["color"], "number": ["number"]},
    file="results/encoding.duckdb",
)
```

Encoding results include predictors, coefficients, pattern expression, and
model diagnostics in the output DuckDB file.

## Getting help

If something is unclear or does not work as expected, open a
[GitHub issue](https://github.com/chenyu-psy/mveeg/issues). Include the mveeg,
Python, and MNE versions, a minimal example, and the full error message when
possible.

To work on mveeg itself, see [Contributing](CONTRIBUTING.md).

## License

mveeg is available under the [MIT License](LICENSE).
