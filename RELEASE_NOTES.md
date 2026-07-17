# mveeg 0.3.0

mveeg 0.3.0 is a breaking architecture release that establishes the package's
manifest-backed preparation, decoding, and encoding APIs. It replaces the 0.2
workflow facades and intermediate compatibility code rather than preserving a
transition layer.

## Highlights

- Prepared and preprocessed datasets use explicit roots, atomic publication,
  provenance fingerprints, and `subject_index + epoch_index` trial identity.
- Raw, External, and Dataset preparation pipelines separate lazy raw-data
  workflows, eager external data normalization, and dataset-level processing.
- Quality processing includes eligibility, AutoReject, artifact rules, saved
  quality state, and manual review sessions under one documented contract.
- Decoding and encoding have independent scientific pipelines while sharing
  only dataset, trial-selection, epoch-window, topography, provenance, and
  transactional result mechanics with identical semantics.
- Decoding supports classifier evidence, Haufe patterns, permutations, and
  temporal generalization with accuracy and target-directed evidence; encoding
  supports formula-based predictors, coefficients, pattern expression, and
  model diagnostics.
- Analysis results are written transactionally to caller-provided DuckDB files
  with explicit schema and package versions.
- Ordered `transform_metadata(**variables)` is available across Raw, External,
  Decoding, and Encoding pipelines.

## Compatibility

- Python 3.10 or newer and MNE 1.12 or newer are required.
- The root namespace exposes only `prep`, `decoding`, `encoding`, and
  `__version__`.
- Datasets, quality sidecars, DuckDB results, and internal imports created with
  earlier unreleased APIs are not compatible and must be regenerated.
- Decoding result schema 2 adds `generalization.target_evidence`; schema-1
  decoding files require `recompute="all"` or regeneration at a new path.
