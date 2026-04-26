"""Data-preparation helpers for the EEG decoding workflow."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd

from .._shared.io_filters import channels_to_drop_by_rule
from .._shared.time_windows import (
    average_time_windows as shared_average_time_windows,
)
from .._shared.time_windows import build_time_windows as shared_build_time_windows
from .config import DecodingConfig


def build_time_windows(times_s: np.ndarray, window_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """Build time windows for averaging the EEG signal.

    This wrapper keeps the decoding API stable while using the shared helper.
    """

    return shared_build_time_windows(times_s, window_ms)


def average_time_windows(data: np.ndarray, window_masks: np.ndarray) -> np.ndarray:
    """Average EEG data within each time window.

    This wrapper keeps the decoding API stable while using the shared helper.
    """

    return shared_average_time_windows(data, window_masks)


def make_balanced_trial_bins(
    data: np.ndarray,
    labels: np.ndarray,
    trial_bin_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance conditions and average trials into bins."""

    unique_labels = np.unique(labels)
    counts = {label: int(np.sum(labels == label)) for label in unique_labels}
    min_count = min(counts.values())

    if min_count < trial_bin_size:
        raise ValueError(
            "At least one condition has fewer trials than trial_bin_size. "
            f"Counts were: {counts}"
        )

    usable_trials_per_label = min_count - (min_count % trial_bin_size)
    if usable_trials_per_label == 0:
        raise ValueError(
            "No complete trial bins can be formed. "
            f"Counts were: {counts}, trial_bin_size={trial_bin_size}"
        )

    binned_blocks = []
    binned_labels = []
    for label in unique_labels:
        label_idx = np.where(labels == label)[0]
        chosen_idx = rng.choice(label_idx, size=usable_trials_per_label, replace=False)
        shuffled_idx = rng.permutation(chosen_idx)
        label_data = data[shuffled_idx]

        if trial_bin_size == 1:
            bins = label_data
        else:
            n_bins = usable_trials_per_label // trial_bin_size
            bins = label_data.reshape(n_bins, trial_bin_size, data.shape[1], data.shape[2]).mean(axis=1)

        binned_blocks.append(bins)
        binned_labels.extend([label] * len(bins))

    binned_data = np.concatenate(binned_blocks, axis=0)
    binned_labels = np.asarray(binned_labels, dtype=object)

    order = rng.permutation(len(binned_labels))
    return binned_data[order], binned_labels[order]


def apply_exclusion_rule(column: pd.Series, rule: tuple | str) -> np.ndarray:
    """Apply one metadata exclusion rule."""

    keep_mask = np.ones(len(column), dtype=bool)

    if rule == "notna":
        excluded_values = ()
        exclude_non_missing = True
    else:
        excluded_values = tuple(rule)
        exclude_non_missing = False

    if len(excluded_values) > 0:
        keep_mask &= ~column.isin(excluded_values).to_numpy()

    if exclude_non_missing:
        keep_mask &= column.isna().to_numpy()

    return keep_mask


def sample_balanced_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample the same number of trials from each class."""

    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = counts.min()
    sampled = []
    for label in unique_labels:
        label_idx = np.where(labels == label)[0]
        sampled.extend(rng.choice(label_idx, size=min_count, replace=False))
    sampled = np.asarray(sampled, dtype=int)
    return rng.permutation(sampled)


def training_row_mask(labels: np.ndarray, label_order: list[str]) -> np.ndarray:
    """Return rows that belong to the configured training labels."""

    return np.isin(labels, label_order)


def channels_to_drop(epochs: mne.Epochs, cfg: DecodingConfig) -> list[str]:
    """Return channels that should be removed before decoding.

    This wrapper keeps the decoding API stable while using the shared helper.
    """

    return channels_to_drop_by_rule(epochs, cfg.decode)
