"""Shared epoch crop, channel-drop, and time-bin mechanics."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd


def channels_to_drop(
    epochs: mne.BaseEpochs,
    *,
    drop_channel_types: list[str] | tuple[str, ...],
    drop_channels: list[str] | tuple[str, ...],
) -> list[str]:
    """Resolve configured channel names and MNE channel types."""

    selected = {
        name
        for name, kind in zip(epochs.ch_names, epochs.get_channel_types(), strict=True)
        if kind in drop_channel_types
    }
    selected.update(name for name in drop_channels if name in epochs.ch_names)
    return sorted(selected)


def build_time_bins(times_s: np.ndarray, window_ms: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Build millisecond center/start/end rows and their sample masks."""

    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("times_s must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("times_s must contain finite, strictly increasing values.")
    if not isinstance(window_ms, (int, np.integer)) or int(window_ms) < 1:
        raise ValueError("window_ms must be a positive integer.")
    times_ms = np.round(times * 1000).astype(int)
    start, stop = int(times_ms[0]), int(times_ms[-1])
    starts = np.arange(start, stop, int(window_ms), dtype=int)
    if len(starts) == 0:
        raise ValueError("The epoch time range is shorter than one time bin.")
    ends = np.minimum(starts + int(window_ms), stop)
    centers = np.round(starts + (ends - starts) / 2).astype(int)
    masks = np.asarray(
        [
            (times_ms >= left) & (times_ms <= right)
            if index == len(starts) - 1
            else (times_ms >= left) & (times_ms < right)
            for index, (left, right) in enumerate(zip(starts, ends, strict=True))
        ],
        dtype=bool,
    )
    if np.any(masks.sum(axis=1) == 0):
        raise ValueError("At least one time bin contains no EEG samples.")
    return pd.DataFrame({"time": centers, "start": starts, "end": ends}), masks


def average_time_bins(data: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Average ``trials × channels × times`` data within each mask."""

    if data.ndim != 3 or masks.ndim != 2 or data.shape[2] != masks.shape[1]:
        raise ValueError("Data and time-bin masks have incompatible shapes.")
    return np.stack([data[:, :, mask].mean(axis=2) for mask in masks], axis=2)
