# Unreleased

- Add `EncodingPipeline.compare()` for crossvalidated theoretical-subspace
  strength, with paired nested/nonnested differences, independent-group noise
  estimation, full-channel correlation shrinkage, sample-rank diagnostics, and
  a separate documented DuckDB contract. Low complement noise df produces an
  explicit warning; covariance uncertainty is not treated as solved by shrinkage.
  Existing ridge encoding and pattern expression are unchanged.

# mveeg 0.3.2

mveeg 0.3.2 fixes artifact-labeling persistence without changing the public API
or result schemas.

## What changed

- Recomputed artifact tables are written only when their contents change,
  preserving file modification times when new rules produce identical labels.
- Applied labeling rules and their fingerprint are saved even when the resulting
  artifact table is unchanged, avoiding repeated computation on later runs.
- `recompute="never"` preserves the provenance of existing artifact tables, so
  a later `recompute="changed"` still applies pending rule changes.
