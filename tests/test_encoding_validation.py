"""Tests for encoding design validation helpers."""

import numpy as np

from mveeg.encoding.validation import validate_encoding


def test_validate_encoding_accepts_full_rank_design():
    """A full-rank design should pass estimable-independent validation."""
    X = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    result = validate_encoding(X, ["intercept", "load", "cue"])

    assert result.is_valid
    assert result.rank == 3
    assert result.aliased_columns == []


def test_validate_encoding_reports_rank_deficiency():
    """A duplicated predictor should be reported as non-estimable."""
    X = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )

    result = validate_encoding(X, ["intercept", "load", "load_copy"])

    assert not result.is_valid
    assert result.rank == 2
    assert result.aliased_columns == ["load", "load_copy"]
