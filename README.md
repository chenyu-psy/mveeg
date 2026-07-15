# mveeg

`mveeg` provides manifest-backed EEG preprocessing plus multivariate encoding
and decoding workflows. Version 0.3 uses one trial identity everywhere:
`subject_index + epoch_index`.

## Installation

```bash
uv pip install -e .
```

## Raw-data quickstart

The raw pipeline is dataset-level and lazy. Calls register steps; data are read
only by `build_epochs()`.

```python
from mveeg import prep

pipeline = prep.init_pipeline("data/raw")
pipeline.load_eeg("*.vhdr")
pipeline.load_eyelink()
pipeline.configure_gaze(
    viewing_distance_cm=80,
    screen_width_cm=53.2,
    screen_width_px=1920,
)
pipeline.load_behavior(
    "*_beh.csv",
    include={"trial_type": ["exp", "pra"], "rejection": "no"},
)
pipeline.filter_eeg(l_freq=None, h_freq=80)
pipeline.make_epochs(
    event_id={"stimulus": 1},
    time_window=(-0.2, 1.0),
    baseline=(-0.2, 0),
)
pipeline.sync_eyelink()
pipeline.align_behavior()  # strict row-count and order contract
pipeline.select_epochs(include={"trial_type": "exp"})

prepared = pipeline.build_epochs(
    "data/prepared",
    task="memory",
    exclude_subjects=None,
    recompute="never",
)
```

This is the simple event mode: each matching `event_id` entry defines an
epoch. Pass `trial_sequences` when an epoch is valid only if an expected
ordered event sequence is present. Each mapping key is its time-zero event by
default; use `time_zero=event_code` or a per-trial mapping only when another
event defines zero. The sequence validates the trial and supplies relative
event timing metadata. `sync_eyelink()` reuses the registered epoch definition.

See [the raw-data recipe](docs/preprocessing-raw.md).

## External-data quickstart

The external pipeline is subject-level and eager. Project code reads any
external format into memory; `mveeg` converts it to MNE epochs, attaches
metadata, and writes the same prepared-dataset contract as the raw pipeline.

```python
from mveeg import prep

external = prep.init_external(subject_index="4001", data=data_in_memory)
external.make_epochs(
    sampling_rate=500,
    ch_names=channel_names,
    tmin=-0.2,
    events=event_codes,
)
external.merge_metadata(metadata)  # strict row mode
external.select_epochs(include={"trial_type": "exp"})
prepared = external.build_epochs("data/prepared", task="memory")
```

If `data_in_memory` is already an `mne.Epochs` object, `make_epochs()` can be
skipped. See [the external-data recipe](docs/preprocessing-external.md).

If external epochs contain pixel-space eye-gaze channels, register their
display geometry before `build_epochs()`:

```python
external.configure_gaze(
    viewing_distance_cm=80,
    screen_width_cm=53.2,
    screen_width_px=1920,
)
```

The external gaze coordinates must already use that screen's pixel coordinate
system. `mveeg` stores the geometry; it does not infer or convert an external
source format.

## Signal preprocessing and artifact review

```python
from mveeg import prep

prepared = prep.open_pipeline("data/prepared")
preprocessed = prep.preprocess_epochs(
    prepared,
    "data/preprocessed",
    eligibility={
        "time_window": (-0.2, 1.0),
        "gaze": {
            "deviation_deg": 1.25,
            "shift_deg": 0.75,
            "max_missing_fraction": 0.10,
        },
        "eeg": eligibility_eeg_config,
    },
    autoreject=autoreject_config,  # use None to skip AutoReject
    recompute="never",
)

reject_config = {
    "time_window": (-0.2, 1.0),
    "hf_noise": {
        "band": (25, 45),
        "window_duration": 0.25,
        "z_threshold": 6,
        "min_noisy_fraction": 0.20,
        "bad_channels": 5,
    },
}
review_config = {
    "time_window": (-0.2, 1.0),
    "eeg": {
        "p2p": 120e-6,
        "step": 60e-6,
        "absolute_value": 120e-6,
        "bad_channels": 2,
    },
    "hf_noise": {
        "band": (25, 45),
        "window_duration": 0.25,
        "z_threshold": 4,
        "min_noisy_fraction": 0.20,
        "bad_channels": 4,
    },
}
preprocessed.label_artifacts(
    reject=reject_config,
    review=review_config,
    ignore_channels=["Fp1", "Fp2"],
)
preprocessed.review_artifacts(
    subject_index="4001",
    group_by="initial_status",
    label="review",
)
```

Artifact labeling prints one row per subject with automatic `accepted`,
`rejected`, and `review` counts. The manual-review window uses a compact layout,
and the first and last displayed epochs meet the plotting bounds without added
horizontal padding.

Set `autoreject.exclude_channels` to a list of EEG channel names that should
remain in the saved epochs but not participate in AutoReject threshold fitting,
cross-validation, voting, or interpolation. Their AutoReject labels are stored
as `-1`; channels already listed in `epochs.info["bads"]` follow the same model
exclusion behavior.

Before the window opens, manual review prints the complete count table for
the requested `group_by` field (or `all: N` when no group is selected). The
browser uses black traces, an orange background for rejected trials, and red
overlays for their flagged channels. `Show flags on accepted` extends only the
red channel overlays to non-rejected trials. Channel names remain plain on the
y-axis; `r` toggles all visible epoch and channel reason codes.

`DatasetPipeline.review_artifacts()` is blocking and returns `None` after the
window closes. Closing also disconnects callbacks and releases the preloaded
epochs and GUI references, so notebook output history cannot retain the full
review dataset through this call.

Clicking a trial switches it directly between `accepted` and `rejected`.
Unreviewed trials whose automatic `final_status` is `review` are staged as
`rejected` when first displayed. The upper-right `Progress current / total`
counts saved reviews together with trials displayed in the current session.
Arrow keys navigate and `w` saves the accumulated edits while marking only
visited trials reviewed. Closing without `w` writes nothing.

Prepared and preprocessed data use separate dataset roots. Later scripts open
the intended root explicitly; there is no implicit active or latest dataset.
Gaze deviation and shift thresholds are specified in visual degrees. The
optional `max_missing_fraction` marks a sample missing only when neither eye
has finite `xpos` and `ypos`; it uses the rule's `time_window` and does not
require display geometry. Users provide geometry once with `configure_gaze()`
when a degree-based rule is configured.

`hf_noise` is a residual high-frequency outlier label applied downstream of
the optional AutoReject stage. It band-filters the signal, measures log
mean-square power in complete overlapping windows, and compares each window
with a robust reference for that subject and channel. The reference excludes
ineligible epochs and AutoReject-bad epochs; when AutoReject is enabled, it
uses only channel/epoch cells labeled good (`0`), not bad (`1`) or interpolated
(`2`) cells. Channels in `epochs.info["bads"]` are not scored. Window overlap is
fixed at 50%. A channel is
labeled when the fraction of windows at or above `z_threshold` reaches
`min_noisy_fraction`; an epoch reaches that rule's status when at least
`bad_channels` channels are labeled. All five keys shown above are required;
there is no implicit window or coverage default.

This rule only writes artifact labels and statuses. Its measurement filter is
not applied to the saved signal; it does not delete epochs or claim that an
outlier is muscle activity. It also does not detect a subject or channel whose
high-frequency level remains persistently elevated across most trials, because
such a level can become the within-subject reference distribution. Diagnose
persistent channel- or subject-level noise separately.

## Configuration boundaries

- A raw or external pipeline holds dataset-construction choices before
  `build_epochs()`: source loading, epoch construction, metadata selection, and
  optional gaze geometry.
- After build, `preprocess_epochs()` and `label_artifacts()` receive the signal
  and artifact-rule configuration. If an already-built prepared dataset lacks
  gaze geometry, `prepared.configure_gaze(...)` writes it atomically to that
  dataset's provenance.
- `open_pipeline(root)` reopens one explicit dataset root and restores its
  stored gaze geometry. It does not restore an active raw pipeline, infer a
  latest stage, or require geometry to be entered again.

## Encoding

Encoding is an independent manifest-backed pipeline. It fits a time-resolved
multivariate linear encoding model with separate component and condition ridge
penalties, then writes one public DuckDB result.

```python
from mveeg import encoding

pipeline = encoding.init_pipeline("data/preprocessed")
pipeline.transform_metadata(
    color=lambda x: x["color_count"].gt(0).astype(float),
    number=lambda x: x["number_count"].gt(0).astype(float),
    load=lambda x: (x["color_count"] + x["number_count"] - 1).astype(float),
    interaction=lambda x: (x["color"].eq(1) & x["number"].eq(1)).astype(float),
)
pipeline.select_trials(
    qc="final_status",
    keep=["accepted"],
    exclude={},
)
pipeline.prepare_epochs(crop=(-0.3, 2.2), time_bin=50)
pipeline.setup_model(
    penalty={"component": 1.0, "condition": 0.1},
)
pipeline.setup_cv(folds=5, seed=None)
pipeline.encode(
    formula="1 + color + number + load + interaction",
    target="condition",
    conditions={"0C1N": ["0C1N"], "1C0N": ["1C0N"], "1C1N": ["1C1N"]},
    expression=None,
    file="results/encoding.duckdb",
    recompute="never",
    n_jobs=5,
    progress=True,
)
```

Metadata variables are evaluated in order, so later variables may use earlier
ones. `conditions` defines the training cells and complete penalized condition
basis; `expression` may add nontraining cells and defaults to `conditions`.
Every selected trial receives one held-out expression value per component and
time bin. Full-data raw-scale coefficients are stored separately for later R
summaries and topographies. `encode()` returns `None`; query the documented
DuckDB tables directly. See [Encoding API and result contract](docs/encoding.md).

## Decoding

Initialize decoding from a prepared dataset root, register the analysis
choices, and let `decode()` write the public DuckDB tables. Classifier and CV
setup are optional; their defaults are logistic regression and 5-fold CV with
20 repeats.

```python
from mveeg import decoding

pipeline = decoding.init_pipeline("data/preprocessed")
pipeline.select_trials(
    qc="final_status",
    keep=["accepted"],
    exclude={"behavior_status": ["incorrect"]},
)
pipeline.prepare_epochs(crop=(-0.2, 0.8), time_bin=50)
pipeline.setup_classifier(
    classifier="logistic_regression",
    solver="lbfgs",
    max_iter=1000,
)
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
    evidence=None,
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

`classes`, `evidence`, and `generalization` all map labels to raw target values.
`classes` must define at least two labels. An explicit `evidence` mapping may
add output groups evaluated by every fold model but not used for training.
Generalization keys must reuse labels from `classes`, while their raw conditions
remain independent of both training membership and evidence output. The same raw
condition may therefore have different training and generalization labels when
the scientific roles differ. Use `None` to skip generalization.
Subjects run one at a time; `n_jobs` controls parallel CV repeats within the
active subject, and each computed subject gets
one repeat-level progress bar. Query the documented DuckDB tables directly;
`decode()` returns `None`. See [Decoding API and result contract](docs/decoding.md)
for the CV, evidence, pattern, permutation, and schema definitions.

## Development

```bash
uv sync
uv run --with pytest python -m pytest
```
