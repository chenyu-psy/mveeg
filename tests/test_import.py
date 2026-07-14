"""Minimal tests that verify package imports and metadata."""

import importlib

import tomllib

import pytest
import mveeg
from mveeg.validation import check_trial_count


def test_version_string():
    """__version__ should be a non-empty string."""
    assert isinstance(mveeg.__version__, str)
    assert mveeg.__version__ != ""


def test_package_version_matches_project_metadata():
    """Runtime and build metadata versions should stay synchronized."""
    with open("pyproject.toml", "rb") as f:
        project_version = tomllib.load(f)["project"]["version"]

    assert mveeg.__version__ == project_version


def test_subpackages_importable():
    """Every sub-package should be importable without error."""
    import mveeg.encoding
    import mveeg.decoding
    import mveeg.prep
    import mveeg.io
    import mveeg.summaries
    import mveeg.validation


def test_preprocessing_public_api_is_03_only():
    """The package should expose the 0.3 entry points without legacy modules."""
    expected = {
        "init_pipeline",
        "init_external",
        "open_pipeline",
        "preprocess_epochs",
        "steps",
    }
    assert expected.issubset(set(mveeg.prep.__all__))
    assert callable(mveeg.transform_metadata)

    for module in ("core", "workflow", "qc", "epoched_mat", "visualizer"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"mveeg.prep.{module}")


def test_decoding_public_api_has_no_legacy_facade():
    pipeline_methods = {
        "transform_metadata",
        "select_trials",
        "prepare_epochs",
        "setup_classifier",
        "setup_cv",
        "decode",
    }
    assert pipeline_methods.issubset(set(dir(mveeg.prep.DatasetPipeline)))
    for module in (
        "config",
        "io",
        "run",
        "summaries",
        "workflow",
        "workflow_outputs",
        "workflow_paths",
        "workflow_subjects",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"mveeg.decoding.{module}")


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
