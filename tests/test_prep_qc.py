"""Tests for simple preparation/QC helpers."""

import numpy as np

from mveeg.prep.qc import summarize_trial_mask


def test_summarize_trial_mask_reports_count_and_percent():
    """Trial-mask summaries should be readable in preprocessing logs."""
    summary = summarize_trial_mask(np.array([True, False, True, False]))

    assert summary == "2 (50.0%)"
