# Common DuckDB result contract

Decoding and Encoding write independent DuckDB files and keep their scientific
schemas separate. They share only transaction handling, subject state,
failure recording, exact-column checks, and support-table mechanics.

## Versions and state

Each `analysis` table contains separate columns:

- `schema_version`, the incompatible result-format version;
- `mveeg_version`, the package version that created the analysis.

Decoding result schema is currently 2 and Encoding result schema is currently
1. Schema versions change only for incompatible stored-format changes.

Both files contain:

```text
subjects(subject_index, status, fingerprint, reason, updated)
```

Status is `pending`, `complete`, or `failed`. Before recomputation, stale rows
for that subject are deleted and the subject is marked pending. All trials,
scientific results, and the final complete state are committed in one
transaction. Failures retain a reason and no partial result rows.

## Trials and fixed tables

`trials` always begins with `subject_index, epoch_index` followed by the
analysis-specific role columns. Remaining experimental metadata is dynamic and
preserves its post-transform order. Decoding may restrict these dynamic columns
with `store_metadata`; the fixed identity and role columns remain unchanged.

Scientific result tables have explicit, analysis-specific columns and order.
Writers reject missing, extra, or reordered fixed columns. Support tables such
as channels, time bins, predictors, and classifier geometry must match exactly
between subjects in the same result file.

See [Decoding](decoding.md) and [Encoding](encoding.md) for their full tables.

## Channel coordinates

`channels(channel, x, y)` uses MNE's public `mne.channels.make_eeg_layout()`
API. Coordinates are layout-box centers, reordered to the saved EEG channel
order, checked for finite values, and normalized independently to `[0, 1]` on
both axes. A constant axis is centered at `0.5`. They are unitless plotting
coordinates, not head-space distances.
Epochs must contain usable electrode positions.

## Incremental execution

Each analysis fingerprints its complete scientific configuration. A changed
configuration requires `recompute="all"` or a different result file.

- `never` reuses complete subjects and computes missing/failed subjects;
- `changed` also recomputes subjects whose dataset files or derived metadata changed;
- `all` recomputes every requested subject and may reset an incompatible configuration.

Generated seeds are stored and reused by later compatible runs. Decoding and
Encoding deliberately retain their own subject-seed/RNG algorithms because
they are independent analyses.
