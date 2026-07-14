"""Time-window helpers shared by analysis workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_time_windows(times_s: np.ndarray, window_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """Build equally spaced time-window masks from a time axis.

    Parameters
    ----------
    times_s : np.ndarray
        Sample times in seconds.
    window_ms : int
        Width of each window in milliseconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Window center times in milliseconds and a boolean mask per window.

    Examples
    --------
    >>> centers, masks = build_time_windows(np.array([0.0, 0.05, 0.10]), 50)
    >>> centers.tolist()
    [25, 75]
    """

    bins, masks = build_time_bins(times_s, window_ms)
    return bins["time"].to_numpy(dtype=int), masks


def build_time_bins(times_s: np.ndarray, window_ms: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the public center/start/end contract and sample masks.

    All table values are milliseconds. Non-final bins are left-closed and
    right-open; the final bin includes its right edge.
    """

    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("times_s must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("times_s must contain finite, strictly increasing values.")
    if not isinstance(window_ms, (int, np.integer)) or int(window_ms) < 1:
        raise ValueError("window_ms must be a positive integer.")

    times_ms = np.round(times * 1000).astype(int)
    start = int(times_ms[0])
    stop = int(times_ms[-1])
    starts = np.arange(start, stop, int(window_ms), dtype=int)
    if len(starts) == 0:
        raise ValueError("The epoch time range is shorter than one time bin.")
    ends = np.minimum(starts + int(window_ms), stop)
    centers = np.round(starts + (ends - starts) / 2).astype(int)

    masks = []
    for index, (left, right) in enumerate(zip(starts, ends)):
        if index == len(starts) - 1:
            masks.append((times_ms >= left) & (times_ms <= right))
        else:
            masks.append((times_ms >= left) & (times_ms < right))
    masks = np.asarray(masks, dtype=bool)
    if np.any(masks.sum(axis=1) == 0):
        raise ValueError("At least one time bin contains no EEG samples.")

    return pd.DataFrame({"time": centers, "start": starts, "end": ends}), masks


def average_time_windows(data: np.ndarray, window_masks: np.ndarray) -> np.ndarray:
    """Average EEG data within each time window.

    Parameters
    ----------
    data : np.ndarray
        EEG data with shape ``(n_trials, n_channels, n_times)``.
    window_masks : np.ndarray
        Boolean mask matrix with shape ``(n_windows, n_times)``.

    Returns
    -------
    np.ndarray
        Window-averaged data with shape ``(n_trials, n_channels, n_windows)``.
    """

    n_trials, n_channels, _ = data.shape
    n_windows = window_masks.shape[0]
    averaged = np.empty((n_trials, n_channels, n_windows), dtype=float)
    for window_ix, mask in enumerate(window_masks):
        averaged[:, :, window_ix] = data[:, :, mask].mean(axis=2)
    return averaged
