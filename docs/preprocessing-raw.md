# Raw-data preparation

`prep.init_pipeline()` creates a dataset-level lazy pipeline. Its methods
register work; subject files are not read until `build_epochs()`.

```python
from mveeg import prep

pipeline = prep.init_pipeline("data/raw", subject_pattern="sub*")
pipeline.load_eeg("*.vhdr", preload=False)
pipeline.load_eyelink()
pipeline.load_behavior("*_beh.csv", include={"rejection": "no"})
pipeline.filter_eeg(l_freq=None, h_freq=80)
pipeline.make_epochs(
    event_id=event_id,
    trial_sequences=trial_sequences,
    time_window=(-0.25, 3.0),
    baseline=(-0.25, 0),
    sampling_rate=500,
)
pipeline.sync_eyelink()
pipeline.align_behavior()
pipeline.transform_metadata(
    load=lambda frame: frame["set_size"].astype(float),
)
pipeline.select_epochs(include={"trial_type": "exp"})
prepared = pipeline.build_epochs(
    "data/prepared",
    task="exp1",
    recompute="changed",
)
```

## Sources and events

`load_eeg()` accepts a file pattern plus keyword arguments for MNE's public
`mne.io.read_raw()` dispatcher. It has no custom reader hook. Project-specific
formats should be read by project code and passed through the external pipeline.

`load_eyelink()` discovers ASC and EDF files. mveeg reads binocular samples and
message markers directly because MNE's current EyeLink reader misclassifies some
signed samples as status columns. EDF files without an ASC counterpart require
the SR Research `edf2asc` executable. The stable mveeg reader entry point keeps
the backend internal so a validated future MNE release can replace it without
changing preprocessing pipelines.

Without `trial_sequences`, every matching `event_id` event starts one epoch.
With sequences, the mapping value defines the required ordered event codes and
the mapping key is time zero unless `time_zero` overrides it. Relative event
times are added to metadata. `sync_eyelink()` reuses this registered epoch
definition without removing EEG epochs. Unmatched EyeLink trials produce NaN
gaze channels and `gaze_available=False`; ambiguous trial-code alignment still
raises an error.

Behavior alignment is deliberately strict: preprocessing may filter the table,
but `align_behavior()` then requires one row per current epoch in the existing
order. Reserved identity columns cannot come from behavior.

## Metadata and custom steps

`transform_metadata(**variables)` evaluates variables in written order. Each
function receives the current metadata and returns a scalar or one value per
epoch. It cannot replace `subject_index` or `epoch_index`. For
`recompute="changed"`, actual derived values enter the subject input
fingerprint, so changing a function's output rebuilds the affected subject.

Use `add_raw_step()` and `add_epoch_step()` only for reusable work that cannot
be expressed by built-in steps. Custom steps require stable `name`, `version`,
and JSON-like `params` for provenance.

## Eye-gaze geometry and output

`configure_gaze()` stores viewing distance, screen width, and pixel width in
dataset provenance. It does not transform EyeLink channels. Degree-based
quality rules use this geometry later.

`build_epochs()` publishes the complete dataset atomically. `never` reuses
completed subjects, `changed` rebuilds changed inputs, and `all` rebuilds the
selected cohort. See the [dataset contract](dataset.md) and the downstream
[quality workflow](quality.md).
