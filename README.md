# mvmodels

**Multivariate encoding and decoding models for EEG research.**

`mvmodels` is a reusable Python package for building and evaluating
multivariate encoding and decoding models, with a focus on EEG analysis
workflows in psychology and cognitive neuroscience.

---

## What it includes

| Sub-package | Purpose |
|---|---|
| `mvmodels.encoding` | Temporal response functions and linear encoding models |
| `mvmodels.decoding` | LDA, logistic regression, and cross-validated classification |
| `mvmodels.preprocessing` | EEG helpers that produce model-ready arrays |
| `mvmodels.io` | Loading and saving model inputs / outputs |
| `mvmodels.summaries` | Group-level summaries and reporting helpers |
| `mvmodels.validation` | Input validation (trial counts, array shapes, …) |

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
mvmodels = { path = "../mvmodels", editable = true }
```

---

## Quick start

```python
import mvmodels
print(mvmodels.__version__)

# Validate trial count before fitting
from mvmodels.validation import check_trial_count
check_trial_count(n_trials=80)   # passes silently; raises ValueError if too few
```

---

## Development

```bash
uv sync              # create / update the virtual environment
uv run pytest        # run the test suite
```