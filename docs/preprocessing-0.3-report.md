# mveeg 0.3 preprocessing implementation report

Date: 2026-07-13

## Scope and outcome

Version 0.3 replaces the 0.2 preprocessing architecture instead of wrapping
it. The package now has a dataset-level lazy raw pipeline and a subject-level
eager external pipeline. Both write the same manifest-backed prepared dataset,
which is then consumed by signal preprocessing, artifact labeling, manual
review, encoding, and decoding.

This report covers preprocessing and the dataset/metadata contract used by
model analyses. Decoding now has a separate public API and DuckDB contract;
encoding remains unchanged pending its own complete redesign.

## Public architecture

### Raw data

`prep.init_pipeline()` records the input root and lightweight discovery
configuration. Loader calls and processing methods register ordered work;
`build_epochs()` executes it per subject. Only real type dependencies are
enforced. Behavior, EyeLink synchronization, metadata transformation, epoch
selection, channel removal, and custom steps remain optional.

`configure_gaze(viewing_distance_cm, screen_width_cm, screen_width_px)` records
dataset-level display geometry. It does not transform EyeLink channels. Gaze
eligibility and review rules independently support `deviation_deg`,
`shift_deg`, and `max_missing_fraction`. Degree thresholds are converted to
pixels internally and require geometry. The missing-fraction rule does not:
within the configured time window, a sample is missing only when neither eye
has finite `xpos` and `ypos`. Implicit structural dropout checks EEG only.

Behavior `include` rules run while loading behavior and therefore before
alignment. `align_behavior()` only supports strict row-order alignment and
requires equal EEG and behavior row counts. `select_epochs()` runs after
alignment and selects signal, events, and metadata together.

Raw `make_epochs()` uses `time_window=(tmin, tmax)` and the optional target
`sampling_rate`. In simple event mode, every matching `event_id` event defines
an epoch. Optional `trial_sequences` instead requires an ordered valid event
sequence; each mapping key is the time-zero event by default, while
`time_zero` can override it with one shared event code or a per-trial mapping.
Relative event timings are written to metadata. `sync_eyelink()` reuses the
registered epoch definition without repeating its arguments.

Custom raw and epoch steps require a stable `name` and `version`, and their
parameters participate in provenance and fingerprints.

### External data

`prep.init_external()` accepts one subject and data already in memory. It
accepts either an MNE Epochs object or a three-dimensional
epoch-by-channel-by-time array. Project-specific file reading remains project
code rather than package behavior.

`merge_metadata()` supports two explicit modes:

- row mode requires exactly one metadata row per epoch;
- key mode requires unique keys and a one-to-one match.

Neither mode silently truncates, reorders, or overwrites existing columns.
`transform_metadata()` and `select_epochs()` are shared with the raw path.
`build_epochs()` assigns stable identity and writes the same prepared format as
the raw pipeline.

External pixel-space eye-gaze channels must already use the coordinate system
identified by `screen_width_px`. The package does not infer an external source
unit or reformat those coordinates. External pipelines may register the same
dataset-level gaze geometry before build.

For an external epoch array, `make_epochs()` uses `sampling_rate` and retains
`tmin` to locate the first already-epoched sample relative to time zero.

### Prepared and preprocessed datasets

`prep.open_pipeline()` reads dataset metadata and loads subjects only when
requested. It does not infer an active or latest stage. Prepared and
preprocessed data use separate roots, and later analyses must name the root
they intend to use.

Gaze geometry configured on a raw or external pipeline is stored in dataset
provenance when build succeeds. It may also be added atomically to an existing
prepared dataset with `DatasetPipeline.configure_gaze()`. Reopening that root
restores the stored geometry without requiring another configuration call.
This keeps three boundaries explicit: raw/external construction before build,
signal and artifact configuration after build, and path-only dataset reopening
in later scripts.

The standard per-subject files are:

```text
dataset_root/
  dataset_description.json
  manifest.tsv
  provenance.json
  sub-<subject_index>/
    eeg/
      sub-<subject_index>_task-<task>_desc-<stage>_epo.fif
      sub-<subject_index>_task-<task>_events.tsv
      sub-<subject_index>_task-<task>_eeg.json
      sub-<subject_index>_task-<task>_desc-artifacts.tsv
```

The immutable trial key is `subject_index + epoch_index`. `epoch_index` is
generated from zero after the final prepared-data selection and is preserved
through later preprocessing. Base metadata are stored in FIF and mirrored in
`events.tsv`; opening a subject verifies that the two copies agree.

`recompute` accepts `never`, `changed`, or `all`. A global pipeline fingerprint
prevents mixed configuration within one dataset. Per-subject input
fingerprints allow `changed` to rebuild only changed inputs when the global
configuration is unchanged.

Updates to an existing dataset are transactional: changed files are written to
a staging root and the manifest is published last. A failed update rolls back
to the prior readable dataset rather than exposing a mixed result.

## Signal preprocessing and artifact state

`prep.preprocess_epochs()` executes eligibility first and optional AutoReject
second. AutoReject is fitted only on eligible epochs. Neither stage deletes or
reorders trials. Eligibility and AutoReject state are stored in a compressed
NPZ; the saved FIF contains the complete preprocessed epochs rather than a
separate cleaned intermediate.

The AutoReject adapter keeps its dense log aligned to EEG channels even when
the Epochs object contains EOG or other channel types. A globally bad channel
is represented explicitly rather than being treated as a dropped epoch.

### Residual high-frequency outlier labels

The optional `hf_noise` rule runs only during `label_artifacts()`, downstream
of the optional AutoReject stage. Reject and review rules use independent
configurations. The full configuration for each rule contains exactly these
required keys:

```python
reject_config = {
    "band": (25, 45),
    "window_duration": 0.25,
    "z_threshold": 6,
    "min_noisy_fraction": 0.20,
    "bad_channels": 5,
}

review_config = {
    "band": (25, 45),
    "window_duration": 0.25,
    "z_threshold": 4,
    "min_noisy_fraction": 0.20,
    "bad_channels": 4,
}
```

For each epoch and channel, the rule band-filters the signal and computes log
mean-square power in complete sliding windows. `window_duration` is measured
in seconds; the sample count is `round(window_duration * sampling_rate)`, the
step is fixed at 50% of the window, and the final possible complete window is
included when the regular grid does not land on it.

The robust reference is calculated separately for each subject and channel.
It excludes ineligible epochs and AutoReject-bad epochs. When AutoReject is
enabled, only channel/epoch cells labeled good (`0`) enter the reference; bad
(`1`) and interpolated (`2`) cells do not. Channels in
`epochs.info["bads"]` are not scored. The scale is
`1.4826 * MAD` and requires at least two usable reference epochs plus a
positive finite scale. Comparisons are inclusive: a window is noisy at
`z >= z_threshold`, a channel is labeled when its noisy-window fraction is
`>= min_noisy_fraction`, and the epoch reaches the rule's status when the
number of labeled channels is `>= bad_channels`. `ignore_channels` preserves
their sparse channel reason but excludes them from the epoch-level count.

This rule is label-only: its measurement filter is not applied to saved signal
data, it does not delete trials, and it does not change eligibility or
AutoReject. Its interpretation is limited to a residual high-frequency outlier
label. It is not a muscle-artifact claim or a causal classification. Because
the reference is within subject and channel, the rule is not designed to
detect a subject or channel whose high-frequency level is persistently
elevated across most trials; that requires a separate dataset- or channel-level
diagnostic.

`label_artifacts()` runs after signal preprocessing. It combines eligibility,
AutoReject, and configured post-processing rules into the sidecar fields:

```text
subject_index
epoch_index
initial_status
final_status
epoch_reasons
reviewed
channel_<name>...
```

Statuses are limited to `accepted`, `review`, and `rejected`. Clean channel
cells are missing; labeled cells contain stable snake-case reason codes.
`ignore_channels` prevents selected channel labels from contributing to the
post-AutoReject epoch status. It does not retroactively change structural,
gaze, or extreme-EEG eligibility decisions.

Relabeling updates `final_status` only for rows that have not been manually
reviewed. Reviewed rows retain the human decision; automatic status, reasons,
and channel labels are regenerated and remain non-editable.

Manual review is one-subject-at-a-time. `group_by` may refer to a fixed
artifact field or Epochs metadata, with `label` supplying the requested value.
Opening the GUI prints all values and counts for that grouping once; an
ungrouped session prints `all: N`. Missing metadata values are shown as
`<NA>`.

The GUI uses black traces, orange backgrounds for rejected trials, and red
overlays for flagged channels in rejected trials. The optional `Show flags on
accepted` checkbox also shows those red overlays on non-rejected trials but
does not change their status or background. The y-axis contains channel names
only; `r` toggles the visible epoch- and channel-level reason codes. There is
no selected-epoch state.

Clicking a trial switches it directly between `accepted` and `rejected`.
Unreviewed `review` trials are staged as `rejected` only when first displayed.
The GUI records visited epochs as the union of every window displayed in the
session and reports saved-or-visited progress against the current target
group. Status edits stay in memory until `w`; that key writes the edits and
marks only visited epochs reviewed. Unshown epochs remain unchanged, and
closing without `w` writes nothing.

## Model-analysis bridge

Both downstream loaders now consume an explicit prepared/preprocessed dataset
root through the shared manifest loader. Artifact summaries are joined only on
`subject_index + epoch_index`; wide `channel_*` columns remain in the sidecar.
Missing or duplicate identity keys fail rather than falling back to DataFrame
index or row position.

The encoding-only `assign_metadata` implementation was removed. A modeling
transform runs after artifact merge and before filtering or condition
construction, preserves identity, and affects only the current analysis. Its
required `name` and `version` participate in the analysis fingerprint.

## Removed 0.2 surface

The following preprocessing implementations were deleted rather than retained
as aliases or compatibility wrappers:

- the giant `prep.core` workflow;
- workflow checkpoint, status, path, and subject facades;
- the old QC, visualizer, logging, and subject-selection layers;
- the source-specific MATLAB epoched-data loader;
- old epoch helper/re-export modules and their compatibility tests;
- encoding-only `assign_metadata`.

The existing project directory `data/preprocessed/exp4` was not deleted. It is
marked as legacy and is intentionally unreadable through the 0.3 dataset
loader; a future full run must write a new dataset root.

## CL41 migration and real-data validation

`scripts/exp4/CL41_preprocessing.qmd` now uses only the 0.3 API. Its ordering is
behavior pre-filter, epoch/EyeLink synchronization, strict behavior alignment,
experimental-trial selection, signal preprocessing, and artifact labeling.
It registers the exp4 display geometry immediately after `load_eyelink()` and
expresses all gaze thresholds directly in visual degrees; `SCREEN` and the
per-rule `screen` dictionaries were removed without changing threshold values.
Its existing `TRIAL_SEQUENCES` definition remains unchanged. The sequence key
now supplies the time-lock event directly, so the separate time-lock mapping
and repeated EyeLink synchronization arguments were removed.
The real-data smoke hooks were temporary acceptance scaffolding and were
removed from the production CL41 script after validation; the counts below are
retained as the historical implementation record.

The real sub4001 validation produced the required counts:

- EEG trials before alignment: 1,212;
- behavior rows after the `exp + pra` and `rejection == no` pre-filter: 1,212;
- EyeLink trials synchronized: 1,212;
- experimental trials after selection: 1,200;
- smoke-only named/versioned selection: first 120 trials;
- prepared and preprocessed shape: 120 epochs, 38 channels, 1,626 samples;
- sampling rate and time range: 500 Hz, -0.25 to 3.0 s;
- smoke condition counts: `2C0N=22`, `2C2N=15`, `2C4N=20`, `4C0N=21`,
  `4C2N=23`, and `4C4N=19`.

Eligibility retained 106 eligible epochs and marked 14 ineligible. AutoReject
used the requested `n_interpolate=[1, 2, 3]`, `cv=10`, `n_jobs=3`, and
`random_state=0`; it marked 15 epochs in its reject log. The dense log had
shape 120 by 31 EEG channels. Selected interpolation counts were 0 channels
for 27 epochs, 1 for 21, 2 for 14, and 3 for 58.

The artifact counts below are retained as historical acceptance results from
the earlier pipeline smoke. They validate trial identity, sidecar storage, and
the manual-review save/resume mechanics, but they are not golden expectations
for the finalized residual high-frequency rule and must not be used as
detector regression counts.

Initial artifact labeling produced 87 accepted, 4 review, and 29 rejected
epochs, with 38 sparse `channel_*` columns. The headless GUI spot check then
displayed two five-epoch windows, modified one status in each window, and
pressed the GUI `w` handler. Exactly epochs 0 through 9 became reviewed; epochs
10 through 119 remained unreviewed. Reopening resumed at `epoch_index=10`.
The manual check changed epoch 0 from accepted to rejected and epoch 5 from
accepted to review, resulting in final counts of 85 accepted, 5 review, and 30
rejected for this disposable smoke dataset.

The finalized residual high-frequency detector was then run label-only on the
same saved 120-epoch, 31-EEG-channel sub4001 data; AutoReject was not rerun.
With the final CL41 hard configuration, it labeled 0 channel cells, 0 epochs
with any contributing channel, and 0 epochs at the five-channel aggregation
threshold. The review configuration likewise labeled 0 channel cells, 0
epochs with any contributing channel, and 0 epochs at the four-channel
aggregation threshold. The complete automatic status result remained 87
accepted, 4 review, and 29 rejected epochs because those decisions came from
eligibility, AutoReject, and the other review rules. Artifact rule evaluation
took 0.557 seconds after loading (the separate FIF/state load took 0.107
seconds). These machine-specific counts and timings verify the label-only
runtime path; they are not detector golden expectations.

During real-data validation, the first 12 EyeLink warm-up messages were found
to precede the first available sample. Mapping all of them to sample zero would
have created duplicate events. The general fallback now excludes annotations
outside the available sample clock and preserves every in-range event without
clipping or reordering. This yielded 1,212 unique synchronized trial events and
did not introduce a project-specific format adapter.

MNE 1.12.1 can also mistake a signed first sample such as `-94.4` for an
EyeLink status column. Until [MNE PR #13571](https://github.com/mne-tools/mne-python/pull/13571)
is released, mveeg silently handles only the resulting 7-to-6 binocular column
error with its validated reader. Other MNE failures still warn, and fallback is
refused when MNE-specific reader options were supplied rather than silently
ignoring those options.

Raw dataset builds now show one subject-level progress bar by default, while
package-owned raw loading suppresses MNE's uninformative `Reading 0 ...` log.
No per-step or external-pipeline progress UI was added.

## Verification

- package tests: 207 passed, including 50 focused signal-quality tests for the
  finalized residual high-frequency detector;
- artifact/review focused tests: 18 passed, including the staged default
  rejection, direct-click GUI, reason and flag toggles, progress, group counts,
  visited-window save, close-without-save, and resume contracts;
- current CL41 Python-cell static compilation: passed all six cells;
- sub4001 event-only validation: 1,449 condition markers reduced to 1,212
  complete trial sequences, matching 1,212 pre-filtered behavior rows and
  yielding 1,200 experimental rows after selection;
- prior CL41 Quarto render: passed all six Python cells and produced HTML;
- historical real 120-epoch eligibility, AutoReject, preprocessing, and
  labeling smoke: passed; its artifact-status counts are not golden
  expectations for the finalized residual high-frequency rule;
- finalized detector label-only smoke on the saved 120 epochs: passed without
  rerunning AutoReject;
- real sidecar keyed downstream merge: 120 rows, no wide channel columns
  leaked into analysis metadata;
- one temporary headless GUI rendering was visually checked and deleted;
- source distribution and wheel build: passed;
- isolated wheel import and legacy prep-module absence check: passed;
- patch whitespace validation: passed.

The smoke artifacts were written under
`/private/tmp/mveeg-cl41-smoke`; the rendered notebook was written under
`/private/tmp/mveeg-cl41-render`.

## Small implementation decisions

- BrainVision fingerprints include the `.vhdr`, `.eeg`, and `.vmrk` files so a
  sidecar change invalidates the subject input.
- Prepared subject fingerprints include FIF, `events.tsv`, and EEG JSON before
  signal preprocessing.
- Gaze geometry is dataset-level provenance; users specify quality thresholds
  in degrees and never calculate pixel thresholds.
- Raw epoch construction exposes `time_window`, `sampling_rate`, and optional
  `trial_sequences`; sequence keys define time zero by default, with explicit
  `time_zero` overrides only for other event anchors.
- External arrays and manifests use `sampling_rate`; MNE-native `sfreq` and
  standard EEG JSON `SamplingFrequency` remain internal/storage-standard names.
- The manifest is authoritative. A blank `artifacts_path` means that no
  artifact table exists; loaders do not guess a neighboring filename.
- External single-subject writes update that subject without declaring other
  manifest subjects deleted.
- AutoReject quality-state files are validated against epoch count and EEG
  channel identity before artifact labeling.
- Reject and review `hf_noise` rules have independent required configurations:
  `band`, `window_duration`, `z_threshold`, `min_noisy_fraction`, and
  `bad_channels`. Window overlap is fixed at 50% rather than being configurable.
- Display hiding and plotting scales are GUI-only; they never modify automatic
  labels or saved signal data.
- No new runtime dependency was added for the refactor.

## Model API status

Decoding now uses the pipeline API and documented DuckDB tables described in
`docs/decoding.md`. Encoding is intentionally untouched by that redesign and
will be replaced independently rather than being forced into decoding's
structure.
