"""Tests for shared time-window helpers used by encoding and decoding."""

import numpy as np

from mveeg._shared.time_windows import average_time_windows, build_time_windows


def test_build_time_windows_returns_centers_and_masks():
    """Window masks should group sample times into clear millisecond bins."""
    centers_ms, masks = build_time_windows(np.array([0.0, 0.05, 0.10]), 50)

    assert centers_ms.tolist() == [25, 75]
    assert masks.tolist() == [[True, False, False], [False, True, True]]


def test_average_time_windows_keeps_trial_and_channel_axes():
    """Averaging should only collapse the time samples selected by each mask."""
    data = np.array([[[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]])
    masks = np.array([[True, True, False], [False, False, True]])

    averaged = average_time_windows(data, masks)

    expected = np.array([[[2.0, 5.0], [3.0, 6.0]]])
    np.testing.assert_allclose(averaged, expected)
