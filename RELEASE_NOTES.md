# mveeg 0.3.1

mveeg 0.3.1 fixes subject-level result persistence when dynamic trial metadata
columns appear in a different order across subjects.

## Fixes

- Treat reordered trial-metadata columns as the same DuckDB schema while
  continuing to reject missing or additional columns.
- Preserve name-based insertion into the existing table order so values remain
  aligned across subjects.
- Add `decode(store_metadata=...)` to retain all, none, or selected trial
  metadata columns while always preserving decoding identity and role columns.
- Validate stored trial columns before fitting and reject names that collide
  under DuckDB's case-insensitive identifier rules.
- Stop Decoding and Encoding on the first unhandled subject failure after
  recording its reason, matching preprocessing's existing fail-fast behavior.
- Preserve EEG trials when EyeLink data are incomplete, marking unmatched gaze
  channels as missing instead of silently dropping epochs.
- Add `artifact_counts()` for notebook-friendly automatic or final artifact
  summaries without coupling reporting to relabeling.
