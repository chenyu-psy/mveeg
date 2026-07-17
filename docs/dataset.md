# Dataset and metadata contract

Prepared and preprocessed data live in explicit, independent dataset roots.
There is no active or latest dataset. `prep.open_pipeline(root)` returns a
lightweight `DatasetPipeline` for one root.

## Identity

Every epoch is identified by:

```text
subject_index + epoch_index
```

`subject_index` is stored as a string and `epoch_index` as a zero-based integer
within each saved subject. These names are reserved. Experiment columns named
`subject`, `participant`, or `trial` are ordinary metadata and are retained. A
redundant `subject` alias may be omitted from model trials only when it exactly
duplicates `subject_index`.

The same identity appears in FIF metadata, events TSV files, artifact sidecars,
the manifest, and DuckDB outputs. Dataset loading validates the FIF/events
mirror and joins artifact state by identity rather than row position.

## Files and schema

Dataset schema 2 contains:

- `manifest.tsv`, one row per subject;
- `provenance.json`, dataset schema, pipeline, task, stage, and optional gaze geometry;
- `dataset_description.json`, task and stage summary;
- one epochs FIF, events TSV, and EEG JSON per subject;
- an optional quality/artifact TSV per subject.

The manifest records `subject_index`, task, stage, epoch/channel/time geometry,
relative paths, the subject input fingerprint, and the dataset pipeline
fingerprint. Paths in the manifest are authoritative; readers do not discover
unlisted sidecars.

`DatasetPipeline` provides subject enumeration, path lookup, epoch loading,
gaze configuration, preprocessing, artifact labeling, and artifact review.
`DatasetBuilder` remains private because partial dataset construction is not a
public workflow.

## Metadata transforms

Raw, External, Decoding, and Encoding expose the same API:

```python
pipeline.transform_metadata(
    a=lambda frame: frame["condition"].isin(["A", "AB"]).astype(float),
    b=lambda frame: frame["condition"].isin(["B", "AB"]).astype(float),
    interaction=lambda frame: frame["a"] * frame["b"],
)
```

Variables run in keyword order, so a later variable may use an earlier one.
Each function must return either one scalar or one trial-aligned column. A
DataFrame, a differently sized sequence, or an attempt to replace
`subject_index`/`epoch_index` is rejected.

Model pipelines fingerprint the actual derived values per subject. The raw
pipeline also computes transformed values before a `changed` decision, so a
same-named transform with different output is rebuilt.

## Publication and recomputation

Dataset writes stage all changed subjects and root metadata before replacing
the live root. A failed multi-subject build leaves the previous dataset intact.
Interrupted publication is recovered when the dataset is next opened.

- `never` reuses existing complete subjects and warns about changed inputs;
- `changed` writes subjects whose input fingerprint changed;
- `all` writes the selected cohort regardless of saved fingerprints.

Changes to a global pipeline cannot silently mix incompatible subjects. A
cohort-level rebuild must cover every existing subject in that root.

## Provenance

Fingerprints accept only stable JSON-like values, `pathlib.Path`, and NumPy
arrays/scalars. Callables, sets, arbitrary objects, non-string mapping keys,
and non-finite numbers are rejected; there is no `repr()` fallback.

The package version has one source of truth, `pyproject.toml`. Dataset schema
changes only when the stored format becomes incompatible; it does not track
ordinary package releases.
