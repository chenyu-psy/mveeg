"""Tests for encoding design validation helpers."""

import numpy as np

from mveeg.encoding.workflow import (
    export_encoding_outputs,
    prepare_encoding_paths,
    run_encoding,
    run_encoding_design_check,
    run_pattern_expression,
    run_pattern_expression_workflow,
    validate_glm_formula,
)
from mveeg.encoding.workflow_model import _saved_result_matches_current_settings
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


def test_encoding_workflow_facade_keeps_public_imports():
    """Existing encoding workflow imports should remain available."""
    assert callable(prepare_encoding_paths)
    assert callable(run_pattern_expression)
    assert callable(run_pattern_expression_workflow)
    assert callable(export_encoding_outputs)
    assert callable(run_encoding_design_check)
    assert callable(validate_glm_formula)
    assert callable(run_encoding)


def test_prepare_encoding_paths_uses_general_defaults(tmp_path):
    """Default encoding output paths should not name a specific experiment."""
    paths = prepare_encoding_paths(tmp_path, "run_a")

    assert paths["results_dir"] == tmp_path / "results" / "main" / "encoding" / "run_a"
    assert paths["log_path"].name == "encoding.log"


def test_validate_glm_formula_parses_additive_terms():
    """Formula parsing should keep the supported additive model explicit."""
    parsed = validate_glm_formula("~ 0 + load + cue", allowed_predictors={"load", "cue"})

    assert parsed == {"add_intercept": False, "predictors": ["load", "cue"]}


def test_saved_result_matching_checks_current_settings():
    """Cache matching should accept saved results created with the same settings."""
    saved = _make_saved_encoding_payload()

    assert _saved_result_matches_current_settings(
        saved,
        standardize_data=True,
        time_window_ms=50,
        n_null_repeats=2,
        source_condition_col="label",
        source_to_condition={"raw_a": "A", "raw_b": "B"},
        train_condition_labels=("A",),
    )


def _make_saved_encoding_payload():
    """Build the minimal saved payload needed by the cache-matching helper."""
    return {
        "subject": np.asarray("001", dtype=object),
        "times_s": np.asarray([0.0]),
        "ch_names": np.asarray(["Cz"], dtype=object),
        "n_trials": np.asarray(4),
        "n_channels": np.asarray(1),
        "n_times": np.asarray(1),
        "n_folds": np.asarray(2),
        "time_window_ms": np.asarray(50),
        "condition_levels": np.asarray(["A", "B"], dtype=object),
        "standardize_data": np.asarray(True),
        "test_method_name": np.asarray("coefficient_reconstruction", dtype=object),
        "test_method_version": np.asarray("v2_condition_shuffled_null_2026-04-17", dtype=object),
        "predictor_names": np.asarray(["intercept", "load"], dtype=object),
        "raw_beta_patterns": np.zeros((1, 1, 1, 1)),
        "pattern_strength_pattern": np.zeros((1, 1, 1)),
        "pattern_strength_null_draws": np.zeros((1, 1, 1, 1)),
        "n_null_repeats": np.asarray(2),
        "coef_predictor_names": np.asarray(["intercept", "load"], dtype=object),
        "coef_values": np.zeros((1, 2)),
        "coef_fold": np.asarray([1]),
        "coef_condition": np.asarray(["A"], dtype=object),
        "coef_trial_index": np.asarray([0]),
        "coef_time_ms": np.asarray([0.0]),
        "source_condition_col": np.asarray("label", dtype=object),
        "source_condition_keys": np.asarray(["raw_a", "raw_b"], dtype=object),
        "source_condition_values": np.asarray(["A", "B"], dtype=object),
        "train_condition_labels": np.asarray(["A"], dtype=object),
    }
