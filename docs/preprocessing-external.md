# External prepared-data recipe

Use this path when another project has already preprocessed the signal. The
project remains responsible for reading its source format; `mveeg` starts from
an in-memory array or `mne.Epochs` object.

## Epoch array

```python
from mveeg import prep

external = prep.init_external(subject_index="4001", data=epoch_array)
external.make_epochs(
    sampling_rate=500,
    ch_names=channel_names,
    ch_types="eeg",
    tmin=-0.25,
    events=event_codes,
)
external.merge_metadata(metadata)
external.select_epochs(include={"trial_type": "exp"})
external.build_epochs("data/prepared", task="exp1")
```

The array axis order is `(epochs, channels, times)` and signal values must use
MNE units.

## Existing MNE epochs and keyed metadata

```python
external = prep.init_external(subject_index="4002", data=epochs)
external.merge_metadata(
    behavior,
    epoch_key="source_trial",
    metadata_key="behavior_trial",
)
external.transform_metadata(
    add_analysis_columns,
    name="add_analysis_columns",
    version="1",
)
external.build_epochs("data/prepared", task="exp1")
```

Row mode requires equal counts. Key mode requires unique, one-to-one keys and
preserves the epoch order. Neither mode silently overwrites existing columns.
`transform_metadata()` may add or modify non-identity columns, but cannot
change the row count, order, `subject_index`, or `epoch_index`.

Calling `build_epochs()` from each subject-level external pipeline updates the
same manifest-backed dataset root. The final format is identical to the raw
pipeline output.

Call `configure_gaze()` only when the external epochs contain pixel-space
eye-gaze channels:

```python
external.configure_gaze(
    viewing_distance_cm=80,
    screen_width_cm=53.2,
    screen_width_px=1920,
)
```

Those coordinates must already use the coordinate system described by
`screen_width_px`; `mveeg` does not infer external units or convert the source
data. The stored geometry lets later quality rules accept `deviation_deg` and
`shift_deg`, so users do not calculate pixel thresholds.

External construction and metadata choices belong before `build_epochs()`.
Eligibility, AutoReject, and artifact-rule configuration belong after build.
If geometry was not available during construction, it can be persisted later
with `prepared.configure_gaze(...)`. A later
`prep.open_pipeline("data/prepared")` reads the stored geometry automatically
and never guesses an active or latest dataset stage.

## Residual high-frequency outlier labels

Raw and external prepared datasets use the same artifact API downstream of the
optional AutoReject stage:

```python
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
)
```

All five `hf_noise` keys are required. Windows use a fixed 50% overlap, and a
channel is labeled when at least `min_noisy_fraction` of its windows reach the
robust within-subject/channel threshold. The measured quantity is log
mean-square power after band filtering. Only complete windows are used;
`window_duration` is in seconds, its sample count is rounded for the dataset's
sampling rate, and the last possible complete window is included. The
reference excludes ineligible epochs and AutoReject-bad epochs. When
AutoReject is enabled, only channel/epoch cells labeled good (`0`) enter the
reference; bad (`1`) and interpolated (`2`) cells do not. Channels in
`epochs.info["bads"]` are not scored.

An epoch reaches a rule's status when at least `bad_channels` channels are
labeled. `ignore_channels` retains their sparse channel reasons while excluding
them from that count. The rule only changes artifact labels and status
aggregation; its measurement filter is not applied to saved epochs, and it
never removes a trial.

The label means residual high-frequency outlier, not muscle artifact. It also
does not detect persistently high subject- or channel-level noise that defines
its own reference distribution. Diagnose that failure mode separately before
interpreting otherwise clean epoch labels.
