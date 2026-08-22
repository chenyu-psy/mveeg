# mveeg 0.3.1

mveeg 0.3.1 improves result reliability, artifact review, and EyeLink
synchronization while keeping the 0.3 analysis workflows unchanged.

## What changed

- Result files now match trial metadata by column name, so subjects remain
  aligned even when their metadata columns arrive in a different order.
- `decode(store_metadata=...)` can save all, none, or selected trial metadata;
  invalid or conflicting column names are reported before model fitting starts.
- Decoding and Encoding stop at the first failed subject, save the reason, and
  leave completed subjects ready to reuse on the next run.
- Artifact labeling reuses unchanged label files without repeating expensive
  calculations. Manual decisions are preserved, and reviewed labels are
  protected before preprocessing replaces a subject.
- `artifact_counts()` returns automatic or final label counts as a DataFrame
  for direct display in notebooks.
- Artifact review handles short display windows and empty review groups without
  opening a broken figure.
- EyeLink synchronization keeps every EEG trial when gaze data are incomplete;
  unmatched gaze samples are marked missing instead of dropping EEG epochs.
- EyeLink ASC files use mveeg's reader directly for consistent handling of
  signed sample values and message markers.
