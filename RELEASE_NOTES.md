# mveeg 0.3.0

mveeg 0.3.0 is the first public release of a Python package for preparing EEG
datasets and running multivariate decoding and encoding analyses. It provides
manifest-backed workflows that keep trial metadata, processing history, and
analysis results together from preparation through publication.

## Highlights

- Build prepared datasets from continuous recordings or existing MNE Epochs,
  with explicit metadata transforms, trial identity, and provenance.
- Apply eligibility checks, AutoReject, artifact rules, saved quality state,
  and interactive manual review.
- Run binary or multiclass decoding with cross-validation, trial averaging,
  classifier evidence, Haufe patterns, permutations, and temporal
  generalization with accuracy and target-directed evidence.
- Fit formula-based encoding models with coefficients, pattern expression, and
  model diagnostics.
- Resume subject-level analyses and write results transactionally to portable
  DuckDB files with explicit schemas and package versions.

## Installation

Python 3.10 or newer and MNE 1.12 or newer are required. Install the release
wheel directly from GitHub:

```bash
python -m pip install https://github.com/chenyu-psy/mveeg/releases/download/v0.3.0/mveeg-0.3.0-py3-none-any.whl
```
