"""Tests for BIDS-style path helpers exposed through ``mveeg.io``."""

from pathlib import Path

from mveeg.io import (
    build_derivative_stem,
    build_subject_label,
    derivative_file_path,
    normalize_subject_id,
)


def test_subject_id_helpers_accept_common_labels():
    """Subject helpers should accept raw, compact, and BIDS-style labels."""
    assert normalize_subject_id("101") == "101"
    assert normalize_subject_id("sub101") == "101"
    assert normalize_subject_id("sub-101") == "101"
    assert build_subject_label("sub-101") == "sub-101"


def test_derivative_path_uses_configured_naming():
    """Derivative paths should keep dataset-specific names configurable."""
    path = derivative_file_path(
        "/data/project",
        "sub-101",
        "memory",
        "epo",
        ".fif",
        derivative_label="clean",
    )

    expected = Path(
        "/data/project/derivatives/sub-101/eeg/sub-101_memory_desc-clean_epo.fif"
    )
    assert path == expected
    assert build_derivative_stem("101", "memory") == "sub-101_memory_desc-preprocessed"
