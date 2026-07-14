# Raw preprocessing recipe

Use this path when continuous recordings and optional EyeLink/behavior files
are organized under one folder per subject.

```python
from mveeg import prep

pipeline = prep.init_pipeline("data/raw", subject_pattern="sub*")
pipeline.load_eeg("*.vhdr")
pipeline.load_eyelink("*.asc")
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
    event_id=event_id,
    trial_sequences=trial_sequences,
    time_window=(-0.25, 3.0),
    baseline=(-0.25, 0),
    sampling_rate=500,
)
pipeline.sync_eyelink()
pipeline.align_behavior()
pipeline.select_epochs(include={"trial_type": "exp"})
pipeline.drop_channels(["HEOG", "VEOG"])

prepared = pipeline.build_epochs(
    "data/prepared",
    task="exp1",
    recompute="never",
)
```

`load_behavior(include=...)` is the pre-alignment filter. `align_behavior()`
requires exactly one behavior row per epoch and only attaches rows in their
existing order. `select_epochs()` is the post-alignment selection and updates
the epochs, events, and metadata together.

`trial_sequences` is optional. Without it, `make_epochs()` uses simple event
mode and creates one epoch for each matching `event_id` event. With it, each
mapping key is the time-zero event by default and its value is the required
ordered event sequence (or allowed alternative sequences). Use
`time_zero=event_code` or a per-trial mapping only to override that default.
Only valid sequences become epochs, and relative event timings are added to
metadata. `sync_eyelink()` reuses this registered definition, so it takes no
repeated epoch arguments.

All optional steps may be skipped when their input is absent. For reusable
custom work, use `add_raw_step()` or `add_epoch_step()` with explicit
`name`, `version`, and `params`; the pure operations are also available under
`prep.steps`.

`configure_gaze()` stores physical display geometry in dataset provenance. It
does not change the EyeLink channels. Later eligibility and artifact rules use
degree-valued keys such as `deviation_deg` and `shift_deg`; `mveeg` performs the
pixel-threshold conversion internally.

Construction settings belong on `pipeline` before `build_epochs()`. Signal
preprocessing and artifact rules belong after build:

```python
preprocessed = prep.preprocess_epochs(
    prepared,
    "data/preprocessed",
    eligibility={
        "time_window": (-0.25, 3.0),
        "gaze": {"deviation_deg": 1.25, "shift_deg": 0.75},
        "eeg": eligibility_eeg_config,
    },
    autoreject=autoreject_config,
)

preprocessed.label_artifacts(
    reject={
        "time_window": (-0.25, 3.0),
        "hf_noise": {
            "band": (25, 45),
            "window_duration": 0.25,
            "z_threshold": 6,
            "min_noisy_fraction": 0.20,
            "bad_channels": 5,
        },
    },
    review={
        "time_window": (-0.25, 3.0),
        "eeg": review_eeg_config,
        "hf_noise": {
            "band": (25, 45),
            "window_duration": 0.25,
            "z_threshold": 4,
            "min_noisy_fraction": 0.20,
            "bad_channels": 4,
        },
    },
    ignore_channels=["Fp1", "Fp2"],
)
```

The `hf_noise` rule labels residual high-frequency outliers downstream of the
optional AutoReject stage. It band-filters each epoch and measures log
mean-square power in complete sliding windows. `window_duration` is in seconds;
the sample count is rounded for the dataset's sampling rate, windows overlap by
a fixed 50%, and the last possible complete window is included. Its robust
subject/channel reference excludes ineligible epochs and AutoReject-bad
epochs. When AutoReject is enabled, only channel/epoch cells labeled good
(`0`) enter the reference; bad (`1`) and interpolated (`2`) cells do not.
Channels in `epochs.info["bads"]` are not scored.

A channel is labeled when the fraction of windows at or above `z_threshold`
reaches `min_noisy_fraction`; an epoch reaches the rule's status when at least
`bad_channels` channels are labeled. `ignore_channels` retains their sparse
channel reasons but excludes them from this epoch-level count. `band`,
`window_duration`, `z_threshold`, `min_noisy_fraction`, and `bad_channels` are
all required.

This is label-only QC: its measurement filter is not applied to saved signal
and it does not remove trials. A high-frequency label is not evidence that the
source was muscle activity. Because the comparison is within subject and
channel, it is not designed to identify a subject or channel that is
persistently elevated across most trials. Such persistent noise requires a
separate dataset-level diagnostic.

Open the result in a later script with:

```python
prepared = prep.open_pipeline("data/prepared")
epochs = prepared.load_epochs("4001")
```

The reopened handle reads gaze geometry from provenance, so it does not need
another `configure_gaze()` call. If geometry was omitted during construction,
`prepared.configure_gaze(...)` may add it after build and persist it for later
reopens.
