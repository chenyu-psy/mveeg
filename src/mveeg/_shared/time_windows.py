"""Time-window helpers shared by analysis workflows."""

from __future__ import annotations

import numpy as np


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

    times_ms = np.round(times_s * 1000).astype(int)
    start_ms = int(times_ms[0])
    end_ms = int(times_ms[-1])

    bin_starts = np.arange(start_ms, end_ms, window_ms, dtype=int)
    bin_ends = np.minimum(bin_starts + window_ms, end_ms)
    centers_ms = np.round(bin_starts + (bin_ends - bin_starts) / 2).astype(int)

    window_rows = []
    for window_ix, (bin_start, bin_end) in enumerate(zip(bin_starts, bin_ends)):
        is_last_window = window_ix == len(bin_starts) - 1
        if is_last_window:
            mask = (times_ms >= bin_start) & (times_ms <= bin_end)
        else:
            mask = (times_ms >= bin_start) & (times_ms < bin_end)
        window_rows.append(mask)

    window_masks = np.array(window_rows, dtype=bool)
    return centers_ms, window_masks


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
