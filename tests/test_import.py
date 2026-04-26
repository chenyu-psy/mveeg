"""Minimal tests that verify the package imports and placeholder helpers."""

import pytest
import mveeg
from mveeg.validation import check_trial_count


def test_version_string():
    """__version__ should be a non-empty string."""
    assert isinstance(mveeg.__version__, str)
    assert mveeg.__version__ != ""


def test_subpackages_importable():
    """Every sub-package should be importable without error."""
    import mveeg.encoding
    import mveeg.decoding
    import mveeg.preprocessing
    import mveeg.io
    import mveeg.summaries
    import mveeg.validation


def test_check_trial_count_passes():
    """check_trial_count should not raise when n_trials >= min_trials."""
    # Default threshold is 20; passing 20 or more should be silent.
    check_trial_count(20)
    check_trial_count(100)


def test_check_trial_count_raises():
    """check_trial_count should raise ValueError when there are too few trials."""
    with pytest.raises(ValueError, match="Too few trials"):
        check_trial_count(10)


def test_check_trial_count_custom_threshold():
    """Custom min_trials threshold should be respected."""
    check_trial_count(5, min_trials=5)   # exactly at threshold — should pass
    with pytest.raises(ValueError):
        check_trial_count(4, min_trials=5)
