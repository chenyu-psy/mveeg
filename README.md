# mveeg

`mveeg` 0.3.0 provides manifest-backed EEG preparation, multivariate decoding,
and multivariate encoding. The three scientific pipelines are independent and
share only dataset, trial-selection, epoch-window, topography, provenance, and
result-storage mechanics whose semantics are identical.

Every dataset and result uses `subject_index + epoch_index` as trial identity.
Columns such as `subject`, `participant`, and `trial` remain ordinary experiment
metadata.

## Installation

```bash
uv pip install -e .
```

Python 3.10 or newer is required.

## Quickstart

Build a prepared dataset lazily from continuous recordings:

```python
from mveeg import prep

raw = prep.init_pipeline("data/raw")
raw.load_eeg("*.vhdr")
raw.make_epochs(
    event_id={"stimulus": 1},
    time_window=(-0.2, 1.0),
    baseline=(-0.2, 0),
)
raw.transform_metadata(is_target=lambda frame: frame["event_name"].eq("stimulus"))
prepared = raw.build_epochs("data/prepared", task="memory")
```

Or normalize externally prepared epochs eagerly:

```python
external = prep.init_external(subject_index="4001", data=epochs)
external.merge_metadata(metadata)
prepared = external.build_epochs("data/prepared", task="memory")
```

Preprocess a prepared dataset through its dataset pipeline:

```python
prepared = prep.open_pipeline("data/prepared")
preprocessed = prepared.preprocess_epochs(
    "data/preprocessed",
    eligibility=eligibility_config,
    autoreject=autoreject_config,
    recompute="changed",
)
preprocessed.label_artifacts(reject=reject_rules, review=review_rules)
```

Run decoding or encoding from an explicit dataset root:

```python
from mveeg import decoding, encoding

decoder = decoding.init_pipeline("data/preprocessed")
decoder.prepare_epochs(crop=(-0.2, 0.8), time_bin=50)
decoder.setup_cv(folds=5, repeats=20, seed=1)
decoder.decode(
    target="condition",
    classes={"low": ["SS2"], "high": ["SS4"]},
    file="results/decoding.duckdb",
)

encoder = encoding.init_pipeline("data/preprocessed")
encoder.transform_metadata(
    color=lambda frame: frame["color_count"].gt(0).astype(float),
    number=lambda frame: frame["number_count"].gt(0).astype(float),
)
encoder.encode(
    formula="1 + color + number",
    target="condition",
    conditions={"color": ["color"], "number": ["number"]},
    file="results/encoding.duckdb",
)
```

## Documentation

- [Dataset and metadata contract](docs/dataset.md)
- [Raw-data preparation](docs/preprocessing-raw.md)
- [External-data preparation](docs/preprocessing-external.md)
- [Quality, AutoReject, and review](docs/quality.md)
- [Common DuckDB result contract](docs/results.md)
- [Decoding](docs/decoding.md)
- [Encoding](docs/encoding.md)

The root namespace intentionally exposes only `prep`, `decoding`, `encoding`,
and `__version__`.
