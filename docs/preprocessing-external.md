# External-data preparation

`prep.init_external()` creates a subject-level eager pipeline for an in-memory
array or `mne.Epochs`. Project code remains responsible for reading its source
format.

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
external.transform_metadata(
    load=lambda frame: frame["set_size"].astype(float),
    centered_load=lambda frame: frame["load"] - frame["load"].mean(),
)
external.select_epochs(include={"trial_type": "exp"})
prepared = external.build_epochs("data/prepared", task="exp1")
```

Array axes are `(epochs, channels, times)` and values use MNE units. When the
input is already `mne.Epochs`, `make_epochs()` is unnecessary.

## Metadata

Row-wise `merge_metadata()` requires equal counts. Keyed mode requires unique,
one-to-one keys and preserves epoch order:

```python
external.merge_metadata(
    behavior,
    epoch_key="source_trial",
    metadata_key="behavior_trial",
)
```

Incoming metadata cannot define `subject_index` or `epoch_index` and cannot
silently overwrite existing columns. `transform_metadata(**variables)` uses
the same ordered scalar-or-trial-aligned contract as Raw, Decoding, and
Encoding. Identity is generated and validated by mveeg; after selection the
saved `epoch_index` is sequential within the subject.

## Gaze and publication

For pixel-space eye-gaze channels, register the acquisition geometry before
building:

```python
external.configure_gaze(
    viewing_distance_cm=80,
    screen_width_cm=53.2,
    screen_width_px=1920,
)
```

The coordinates must already use that screen's pixel system. mveeg stores the
geometry but does not infer units or convert a project format.

Calling `build_epochs()` for additional subjects updates the same dataset root
through an atomic transaction. The resulting format is identical to raw
preparation output. See the [dataset contract](dataset.md) and the downstream
[quality workflow](quality.md).
