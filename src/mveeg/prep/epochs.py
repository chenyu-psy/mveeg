"""Small epoch-table helpers used by preprocessing workflows."""

from __future__ import annotations

import mne
import numpy as np


def keep_epochs_by_metadata_value(epochs: mne.Epochs, column: str, value: str) -> mne.Epochs:
    """Keep epochs whose metadata column matches one requested value.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object with metadata attached.
    column : str
        Metadata column used for selecting rows.
    value : str
        Metadata value to keep.

    Returns
    -------
    mne.Epochs
        Copy of ``epochs`` containing only matching rows.
    """
    if epochs.metadata is None or column not in epochs.metadata.columns:
        raise ValueError(f"Aligned epochs metadata must contain a '{column}' column.")

    keep_trials = epochs.metadata[column].eq(value).to_numpy()
    if not np.any(keep_trials):
        raise ValueError(f"No rows matched {column} == {value!r}.")
    epochs_kept = epochs.copy()
    drop_ix = np.flatnonzero(~keep_trials).tolist()
    if len(drop_ix) > 0:
        epochs_kept.drop(drop_ix, verbose="ERROR")
    return epochs_kept


def reset_epoch_event_samples(epochs: mne.Epochs) -> mne.Epochs:
    """Rewrite epoch event sample numbers so they are strictly increasing."""
    epochs_reset = epochs.copy()
    epochs_reset.events[:, 0] = np.arange(len(epochs_reset), dtype=int)
    return epochs_reset
