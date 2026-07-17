"""Tests for shared analysis epoch mechanics."""

import numpy as np

from mveeg._analysis.epochs import average_time_bins, build_time_bins


def test_build_time_bins_returns_public_table_and_masks():
    bins, masks = build_time_bins(np.array([0.0, 0.05, 0.10]), 50)

    assert bins.to_dict(orient="list") == {
        "time": [25, 75],
        "start": [0, 50],
        "end": [50, 100],
    }
    assert masks.tolist() == [[True, False, False], [False, True, True]]


def test_average_time_bins_keeps_trial_and_channel_axes():
    data = np.array([[[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]])
    masks = np.array([[True, True, False], [False, False, True]])

    averaged = average_time_bins(data, masks)

    expected = np.array([[[2.0, 5.0], [3.0, 6.0]]])
    np.testing.assert_allclose(averaged, expected)
