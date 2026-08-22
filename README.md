# mveeg

`mveeg` helps EEG researchers prepare trial-level data and run multivariate
decoding or encoding analyses in Python. It keeps each trial linked to its
metadata and saved processing settings, resumes analyses without repeating
completed subjects, and stores model results in DuckDB files that can be read
from Python or R.

Preparation, decoding, and encoding are independent workflows. Use only the
parts your study needs.

## What mveeg helps you do

- Build consistent trial datasets from continuous recordings or existing MNE
  Epochs.
- Identify unusable trials with study-defined checks, AutoReject, and manual
  review.
- Run multivariate decoding, temporal generalization, or linear encoding.
- Keep trial IDs, metadata, and processing settings together from preparation
  to results.

## Installation

Install the latest stable version from the GitHub repository's default branch:

```bash
python -m pip install git+https://github.com/chenyu-psy/mveeg.git
```

This installation requires Git. Python 3.10 or newer is required. Runtime
dependencies, including MNE 1.12 or newer, are installed automatically.

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

Open a prepared dataset folder and save cleaned signals and quality information
to a separate output folder:

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
preprocessed.artifact_counts()
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

The decoding pipeline can select accepted trials, save classifier scores for
individual trials, run permutation analyses, and test temporal generalization.

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

The result file contains model coefficients, trial-level component scores, and
diagnostics that help identify unstable fits.

## Documentation

- [Prepare continuous recordings](docs/preprocessing-raw.md)
- [Prepare existing MNE Epochs](docs/preprocessing-external.md)
- [Preprocess, label, and review artifacts](docs/quality.md)
- [Run decoding analyses](docs/decoding.md)
- [Run encoding analyses](docs/encoding.md)

For storage details and exact result tables, see the advanced
[dataset](docs/dataset.md) and [result-file](docs/results.md) references.

## Getting help

If something is unclear or does not work as expected, open a
[GitHub issue](https://github.com/chenyu-psy/mveeg/issues). Include the mveeg,
Python, and MNE versions, a minimal example, and the full error message when
possible.

To work on mveeg itself, see [Contributing](CONTRIBUTING.md).

## License

mveeg is available under the [MIT License](LICENSE).
