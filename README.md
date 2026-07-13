# mveeg

**Multivariate encoding and decoding models for EEG research.**

`mveeg` is a reusable Python package for building and evaluating
multivariate encoding and decoding models, with a focus on EEG analysis
workflows in psychology and cognitive neuroscience.

---

## What it includes

| Sub-package | Purpose |
|---|---|
| `mveeg.encoding` | Trial-metadata regression models, pattern expression, and model comparison |
| `mveeg.decoding` | LDA, logistic regression, and cross-validated classification |
| `mveeg.prep` | EEG helpers that produce model-ready arrays |
| `mveeg.io` | Loading and saving model inputs / outputs |
| `mveeg.summaries` | Group-level summaries and reporting helpers |
| `mveeg.validation` | Input validation (trial counts, array shapes, …) |

## What it intentionally does *not* include

- RSA / representational similarity analysis
- Project-specific constants, condition maps, or file paths
- Notebook logic or one-off dataset conversion scripts

---

## Installation (editable mode)

```bash
# from the repository root
uv pip install -e .
```

Or add it as an editable path dependency from another project's
`pyproject.toml`:

```toml
[tool.uv.sources]
mveeg = { path = "../mveeg", editable = true }
```

---

## Quick start

```python
import mveeg
print(mveeg.__version__)

# Validate trial count before fitting
from mveeg.validation import check_trial_count
check_trial_count(n_trials=80)   # passes silently; raises ValueError if too few
```

## Encoding regression models

`mveeg.encoding.workflow.run_regression_model` fits cross-validated EEG
regression models from trial metadata. Use `assign_metadata` to add reusable
numeric predictors before the formula is evaluated.

```python
from mveeg.encoding.metadata import assign_metadata
from mveeg.encoding.workflow import run_regression_model

metadata_assign = assign_metadata(
    load=lambda df: df["model_condition"].eq("high_load").astype(float),
)

tables = run_regression_model(
    data_dir="data/preprocessed/exp1",
    subject_ids=["001", "002"],
    trial_filters={
        "qc_col": "qc_pass",
        "keep_qc": [True],
        "exclude_metadata": {},
    },
    encoding_params={
        "crop_time": (-0.2, 1.0),
        "drop_channel_types": [],
        "drop_channels": [],
        "time_window_ms": 50,
    },
    condition_label_map={"high_load": ["SS4"], "low_load": ["SS2"]},
    metadata_assign=metadata_assign,
    formula="pattern ~ 1 + load + (1 | model_condition)",
    penalty={"fixed": 1.0, "random": 0.1},
    overwrite=False,
    name="load_model",
)
```

Formulas select numeric trial-level metadata columns. Additive terms,
interactions such as `load * cue`, and random intercepts such as
`(1 | model_condition)` are supported.

---

## Development

```bash
uv sync              # create / update the virtual environment
uv run pytest        # run the test suite
```
