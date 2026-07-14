"""Tests for encoding design validation helpers."""

import mne
import numpy as np
import pandas as pd
import pytest

from mveeg.encoding.workflow import (
    compare_encoding_models,
    export_encoding_outputs,
    prepare_encoding_paths,
    run_regression_model,
    run_encoding_design_check,
    run_pattern_expression,
    run_pattern_expression_workflow,
    validate_glm_formula,
)
from mveeg._shared.metadata import transform_metadata
from mveeg.encoding.workflow_design import build_formula_metadata_design
from mveeg.encoding.workflow_model import (
    _fit_time_resolved_multivariate_ridge,
    _penalty_vector,
)
from mveeg.encoding.metrics import estimate_channel_covariance
from mveeg.encoding.workflow_outputs import (
    build_encoding_topography_value_table,
    export_encoding_model_outputs,
)
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
    assert callable(compare_encoding_models)
    assert callable(run_encoding_design_check)
    assert callable(validate_glm_formula)
    assert callable(run_regression_model)


def test_prepare_encoding_paths_uses_general_defaults(tmp_path):
    """Default encoding output paths should not name a specific experiment."""
    paths = prepare_encoding_paths(tmp_path, "run_a")

    assert paths["results_dir"] == tmp_path / "results" / "main" / "encoding" / "run_a"
    assert paths["log_path"].name == "encoding.log"


def test_run_encoding_facade_groups_script_settings(tmp_path, monkeypatch):
    """Public encoding facade should convert grouped settings to workflow args."""
    from mveeg.encoding import workflow

    captured = {}

    def fake_run_regression_model_workflow(**kwargs):
        captured.update(kwargs)
        return {
            "subject_summary_df": pd.DataFrame({"subject": ["001", "002"]}),
            "skipped_subjects_df": pd.DataFrame(columns=["subject", "reason"]),
        }

    monkeypatch.setattr(workflow, "run_regression_model_workflow", fake_run_regression_model_workflow)
    metadata_transform = lambda df: df.assign(  # noqa: E731
        item=df["label"].eq("correct").astype(float)
    )

    result = workflow.run_regression_model(
        data_dir=tmp_path / "data" / "preprocessed" / "exp1",
        subject_ids=["001", "002"],
        trial_filters={
            "qc_col": "final_status",
            "keep_qc": ["accepted"],
            "exclude_metadata": None,
        },
        encoding_params={
            "crop_time": (1.7, 2.8),
            "time_window_ms": 50,
            "drop_channel_types": ("eog",),
            "drop_channels": (),
            "n_splits": 2,
            "shuffle": False,
            "random_state": 1,
            "covariance": "identity",
        },
        condition_label_map={"same": ["correct"], "new": ["new"]},
        formula="pattern ~ 1 + item",
        metadata_transform=metadata_transform,
        metadata_transform_name="derive_item",
        metadata_transform_version="1",
        penalty={"fixed": 1.0, "random": 0.1},
        overwrite=False,
        name="probe_encoding",
        topography={"time_window_ms": (1900, 2800)},
    )

    assert captured["source_to_condition"] == {"correct": "same", "new": "new"}
    assert captured["loader_cfg"].dataset.experiment_name == "exp1"
    assert captured["loader_cfg"].conditions.cond_col == "label"
    assert captured["cv_n_splits"] == 2
    assert captured["cv_shuffle"] is False
    assert captured["covariance"] == "identity"
    assert captured["topography"] == {"time_window_ms": (1900, 2800)}
    assert captured["metadata_transform"] is metadata_transform
    assert captured["formula"] == "pattern ~ 1 + item"
    assert captured["penalty"] == {"fixed": 1.0, "random": 0.1}
    assert captured["subject_results_dir"] is None
    assert captured["results_dir"] is None
    assert captured["config_payload"] is None
    assert captured["log_path"] is None
    assert result["subject_summary"]["subject"].tolist() == ["001", "002"]
    assert result["skipped"].empty


def test_run_encoding_appends_new_subjects_to_result_store(tmp_path, monkeypatch):
    """Encoding DuckDB stores should only process requested missing subjects."""

    from mveeg.encoding import workflow

    processed = []
    info = mne.create_info(["Cz"], sfreq=250, ch_types="eeg")
    info.set_montage("standard_1020")

    def fake_run_regression_model_workflow(**kwargs):
        subject_ids = [str(subject_id) for subject_id in kwargs["subject_ids"]]
        processed.extend(subject_ids)
        return _make_encoding_store_output(subject_ids)

    monkeypatch.setattr(workflow, "run_regression_model_workflow", fake_run_regression_model_workflow)
    monkeypatch.setattr(workflow, "load_subject_info", lambda _subject, _cfg: info)

    file = tmp_path / "encoding"
    workflow.run_regression_model(
        **_encoding_kwargs(
            tmp_path,
            subject_ids=["001", "002"],
            overwrite=False,
            file=file,
            topography={"time_window_ms": (0, 50)},
        )
    )
    tables = workflow.run_regression_model(
        **_encoding_kwargs(
            tmp_path,
            subject_ids=["003"],
            overwrite=False,
            file=file,
            topography={"time_window_ms": (0, 50)},
        )
    )

    assert processed == ["001", "002", "003"]
    assert tables["subject_summary"]["subject"].tolist() == ["001", "002", "003"]
    assert tables["topography_values"]["n_subjects"].tolist() == [3, 3]

    tables = workflow.run_regression_model(
        **_encoding_kwargs(
            tmp_path,
            subject_ids=["002"],
            overwrite=True,
            file=file,
            topography={"time_window_ms": (0, 50)},
        )
    )

    assert processed == ["001", "002", "003", "002"]
    assert tables["subject_summary"]["subject"].tolist() == ["001", "002", "003"]
    assert sorted(tables["pattern_expression_trial"]["subject"].unique()) == ["001", "002", "003"]
    assert "testing_effect_coefficients" not in tables


def test_run_pattern_expression_appends_new_subjects_to_result_store(tmp_path):
    """Pattern-expression stores should keep old subjects and append new ones."""

    file = tmp_path / "pattern_expression"
    run_pattern_expression(
        subject_ids=["001"],
        subject_inputs={"001": _make_pattern_expression_input(1.0)},
        overwrite=False,
        name="expr",
        file=file,
    )
    tables = run_pattern_expression(
        subject_ids=["002"],
        subject_inputs={"002": _make_pattern_expression_input(2.0)},
        overwrite=False,
        name="expr",
        file=file,
    )

    assert tables["trials"]["subject"].tolist() == ["001", "002"]
    assert sorted(tables["trial_expression"]["subject"].unique()) == ["001", "002"]

    tables = run_pattern_expression(
        subject_ids=["001"],
        subject_inputs={"001": _make_pattern_expression_input(5.0)},
        overwrite=True,
        name="expr",
        file=file,
    )

    assert tables["trials"]["subject"].tolist() == ["001", "002"]
    refreshed = tables["trial_expression"].loc[tables["trial_expression"]["subject"] == "001"]
    assert refreshed["pattern_expression"].min() == 5.0


def test_build_encoding_topography_value_table_exports_all_predictors():
    """Encoding topography should summarize every saved predictor map."""
    payloads = {
        "001": _make_topography_payload(raw_offset=0.0),
        "002": _make_topography_payload(raw_offset=10.0),
    }

    table = build_encoding_topography_value_table(
        subject_payloads=payloads,
        time_window_ms=(0, 50),
    )

    assert table.columns.tolist() == [
        "channel",
        "effect",
        "raw_value",
        "z_value",
        "window_start_ms",
        "window_end_ms",
        "n_subjects",
    ]
    assert table["effect"].tolist() == ["intercept", "intercept", "item", "item"]
    assert table["channel"].tolist() == ["Fz", "Cz", "Fz", "Cz"]
    assert table["raw_value"].tolist() == [8.0, 10.0, 16.0, 18.0]
    assert table["z_value"].tolist() == [-1.0, 1.0, -1.0, 1.0]
    assert table["window_start_ms"].tolist() == [0, 0, 0, 0]
    assert table["window_end_ms"].tolist() == [50, 50, 50, 50]
    assert table["n_subjects"].tolist() == [2, 2, 2, 2]


def test_build_encoding_topography_value_table_exports_named_windows():
    """Named topography windows should be exported in the requested order."""
    payloads = {
        "001": _make_topography_payload(raw_offset=0.0),
        "002": _make_topography_payload(raw_offset=10.0),
    }

    table = build_encoding_topography_value_table(
        subject_payloads=payloads,
        time_windows_ms={
            "early": (0, 50),
            "late": (100, 100),
        },
    )

    assert table.columns.tolist() == [
        "channel",
        "effect",
        "window_name",
        "raw_value",
        "z_value",
        "window_start_ms",
        "window_end_ms",
        "n_subjects",
    ]
    assert table["window_name"].tolist() == ["early"] * 4 + ["late"] * 4
    assert table["effect"].tolist() == [
        "intercept",
        "intercept",
        "item",
        "item",
        "intercept",
        "intercept",
        "item",
        "item",
    ]
    assert table["channel"].tolist() == ["Fz", "Cz", "Fz", "Cz"] * 2
    assert table["raw_value"].tolist() == [
        8.0,
        10.0,
        16.0,
        18.0,
        104.0,
        104.0,
        104.0,
        104.0,
    ]
    assert table["z_value"].tolist() == [
        -1.0,
        1.0,
        -1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert table["window_start_ms"].tolist() == [0, 0, 0, 0, 100, 100, 100, 100]
    assert table["window_end_ms"].tolist() == [50, 50, 50, 50, 100, 100, 100, 100]


def test_build_encoding_topography_value_table_rejects_bad_windows():
    """Invalid windows should fail before writing misleading topography files."""
    payloads = {"001": _make_topography_payload(raw_offset=0.0)}

    with pytest.raises(ValueError, match="start must be <= end"):
        build_encoding_topography_value_table(
            subject_payloads=payloads,
            time_windows_ms={"bad": (50, 0)},
        )

    with pytest.raises(ValueError, match="No encoding beta pattern time bins"):
        build_encoding_topography_value_table(
            subject_payloads=payloads,
            time_windows_ms={"empty": (200, 300)},
        )


def test_export_encoding_model_outputs_writes_topography_csvs(tmp_path, monkeypatch):
    """Encoding export should write R-ready variable topography CSV files."""
    from mveeg.encoding import workflow_outputs

    info = mne.create_info(["Fz", "Cz"], sfreq=250, ch_types="eeg")
    info.set_montage("standard_1020")
    monkeypatch.setattr(
        workflow_outputs,
        "load_subject_info",
        lambda _subject, _cfg: info,
    )

    output_files = {
        "pattern_expression_trial": "pattern_expression_trial.csv",
        "condition_pattern_expression": "condition_pattern_expression.csv",
        "effect_slope": "effect_slope.csv",
        "design_diagnostics": "design_diagnostics.csv",
        "covariance_diagnostics": "covariance_diagnostics.csv",
        "subject_summary": "subject_summary.csv",
        "run_summary": "run_summary.csv",
        "skipped_subjects": "skipped_subjects.csv",
        "topography_values": "topography/topography_values.csv",
        "topography_coords": "topography/topography_coords.csv",
    }
    outputs = export_encoding_model_outputs(
        output_dir=tmp_path,
        output_files=output_files,
        subject_summary_df=pd.DataFrame({"subject": ["001"]}),
        skipped_subjects_df=pd.DataFrame(columns=["subject", "reason"]),
        run_summary_df=pd.DataFrame({"name": ["run"]}),
        pattern_expression_trial_df=pd.DataFrame({"subject": ["001"]}),
        condition_pattern_expression_df=pd.DataFrame({"subject": ["001"]}),
        effect_slope_df=pd.DataFrame({"subject": ["001"]}),
        design_diagnostics_df=pd.DataFrame({"subject": ["001"]}),
        covariance_diagnostics_df=pd.DataFrame({"subject": ["001"]}),
        config_payload={"run_name": "run"},
        topography={"time_window_ms": (0, 50)},
        subject_payloads={"001": _make_topography_payload(raw_offset=0.0)},
        loader_cfg=object(),
    )

    assert (tmp_path / "topography" / "topography_values.csv").exists()
    assert (tmp_path / "topography" / "topography_coords.csv").exists()
    assert outputs["topography_values_df"]["effect"].tolist() == [
        "intercept",
        "intercept",
        "item",
        "item",
    ]


def test_export_encoding_model_outputs_writes_named_topography_windows(tmp_path, monkeypatch):
    """Encoding export should preserve names for multi-window topography CSVs."""
    from mveeg.encoding import workflow_outputs

    info = mne.create_info(["Fz", "Cz"], sfreq=250, ch_types="eeg")
    info.set_montage("standard_1020")
    monkeypatch.setattr(
        workflow_outputs,
        "load_subject_info",
        lambda _subject, _cfg: info,
    )

    output_files = {
        "pattern_expression_trial": "pattern_expression_trial.csv",
        "condition_pattern_expression": "condition_pattern_expression.csv",
        "effect_slope": "effect_slope.csv",
        "design_diagnostics": "design_diagnostics.csv",
        "covariance_diagnostics": "covariance_diagnostics.csv",
        "subject_summary": "subject_summary.csv",
        "run_summary": "run_summary.csv",
        "skipped_subjects": "skipped_subjects.csv",
        "topography_values": "topography/topography_values.csv",
        "topography_coords": "topography/topography_coords.csv",
    }
    outputs = export_encoding_model_outputs(
        output_dir=tmp_path,
        output_files=output_files,
        subject_summary_df=pd.DataFrame({"subject": ["001"]}),
        skipped_subjects_df=pd.DataFrame(columns=["subject", "reason"]),
        run_summary_df=pd.DataFrame({"name": ["run"]}),
        pattern_expression_trial_df=pd.DataFrame({"subject": ["001"]}),
        condition_pattern_expression_df=pd.DataFrame({"subject": ["001"]}),
        effect_slope_df=pd.DataFrame({"subject": ["001"]}),
        design_diagnostics_df=pd.DataFrame({"subject": ["001"]}),
        covariance_diagnostics_df=pd.DataFrame({"subject": ["001"]}),
        config_payload={"run_name": "run"},
        topography={
            "time_windows_ms": {
                "early": (0, 50),
                "late": (100, 100),
            }
        },
        subject_payloads={"001": _make_topography_payload(raw_offset=0.0)},
        loader_cfg=object(),
    )

    saved_df = pd.read_csv(tmp_path / "topography" / "topography_values.csv")
    assert saved_df["window_name"].tolist() == ["early"] * 4 + ["late"] * 4
    assert outputs["topography_values_df"]["window_name"].tolist() == [
        "early"
    ] * 4 + ["late"] * 4


def test_validate_glm_formula_parses_additive_terms():
    """Formula parsing should keep the supported additive model explicit."""
    parsed = validate_glm_formula("~ 0 + load + cue", allowed_predictors={"load", "cue"})

    assert parsed == {
        "add_intercept": False,
        "predictors": ["load", "cue"],
        "interactions": [],
        "random_terms": [],
    }


def test_validate_glm_formula_expands_interactions():
    """Formula parsing should expand simple interaction shorthand."""
    parsed = validate_glm_formula("~ 1 + load * cue", allowed_predictors={"load", "cue"})

    assert parsed == {
        "add_intercept": True,
        "predictors": ["load", "cue", "load:cue"],
        "interactions": ["load:cue"],
        "random_terms": [],
    }


def test_covariance_modes_return_finite_precision():
    """Supported covariance modes should produce finite precision matrices."""
    residuals = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )

    for method in ["identity", "diagonal", "shrinkage"]:
        estimate = estimate_channel_covariance(residuals, method=method)
        assert estimate.precision.shape == (2, 2)
        assert np.isfinite(estimate.precision).all()


def test_sample_covariance_errors_when_rank_deficient():
    """Sample covariance should fail instead of silently pseudo-inverting."""
    residuals = np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=float)

    with pytest.raises(ValueError, match="rank deficient"):
        estimate_channel_covariance(residuals, method="sample")


def test_transform_metadata_accepts_analysis_columns():
    """Metadata transforms should add analysis columns without mutating input."""

    metadata = pd.DataFrame(
        {
            "subject_index": ["001", "001"],
            "epoch_index": [0, 1],
            "label": ["SS2/A", "SS4/B"],
        }
    )

    def derive_columns(frame):
        frame["setsize"] = frame["label"].str.split("/").str[0]
        frame["load"] = frame["setsize"].eq("SS4").astype(float)
        return frame

    output = transform_metadata(metadata, derive_columns)

    assert output["setsize"].tolist() == ["SS2", "SS4"]
    assert output["load"].tolist() == [0.0, 1.0]
    assert "setsize" not in metadata.columns


def test_formula_metadata_design_expands_random_intercept_from_training_levels():
    """Formula design should use metadata columns and train-observed random levels."""

    metadata = pd.DataFrame(
        {
            "condition": ["A", "B", "A+B", "A"],
            "a": [1.0, 0.0, 1.0, 1.0],
            "b": [0.0, 1.0, 1.0, 0.0],
        }
    )

    design, names, trial_design, parsed = build_formula_metadata_design(
        metadata,
        "pattern ~ a + b + (1 | condition)",
        fit_indices=[0, 1],
    )

    assert names == ["intercept", "a", "b", "random_condition_A", "random_condition_B"]
    assert parsed["term_types"] == ["intercept", "fixed", "fixed", "random", "random"]
    assert design[:, 3].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert design[:, 4].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert trial_design["a"].tolist() == [1.0, 0.0, 1.0, 1.0]


def test_formula_random_intercept_columns_ignore_training_row_order():
    """Random intercept column order should be stable across CV folds."""

    metadata = pd.DataFrame(
        {
            "condition": ["B", "A", "C", "A", "B", "C"],
            "color": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        }
    )

    _, names_a, _, parsed_a = build_formula_metadata_design(
        metadata,
        "pattern ~ color + (1 | condition)",
        fit_indices=[0, 1, 2],
    )
    _, names_b, _, parsed_b = build_formula_metadata_design(
        metadata,
        "pattern ~ color + (1 | condition)",
        fit_indices=[3, 4, 5],
    )

    assert names_a == names_b
    assert parsed_a["random_terms"] == parsed_b["random_terms"]


def test_ridge_penalty_vector_leaves_intercept_unpenalized():
    """Fixed and random penalties should not penalize the intercept."""

    penalty = _penalty_vector(
        ["intercept", "fixed", "random"],
        {"fixed": 4.0, "random": 0.25},
    )

    assert penalty.tolist() == [0.0, 2.0, 0.5]


def test_ridge_fits_rank_deficient_random_design():
    """Ridge augmented least squares should fit random-effect-like columns."""

    design = np.asarray(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )
    data = np.ones((4, 2, 1), dtype=float)

    betas, design_aug = _fit_time_resolved_multivariate_ridge(
        data=data,
        design_matrix=design,
        penalty_sqrt=np.asarray([0.0, 1.0, 0.1, 0.1]),
    )

    assert np.linalg.matrix_rank(design) < design.shape[1]
    assert np.linalg.matrix_rank(design_aug) == design.shape[1]
    assert betas.shape == (4, 2, 1)
    assert np.isfinite(betas).all()


def _encoding_kwargs(tmp_path, **overrides):
    """Return minimal public run_regression_model kwargs for facade/store tests."""

    kwargs = {
        "data_dir": tmp_path / "missing-data",
        "subject_ids": ["001"],
        "trial_filters": {
            "qc_col": "qc",
            "keep_qc": ["accepted"],
            "exclude_metadata": {},
        },
        "encoding_params": {
            "crop_time": (0.0, 1.0),
            "time_window_ms": 50,
            "drop_channel_types": [],
            "drop_channels": [],
            "covariance": "identity",
        },
        "condition_label_map": {"a": ["a"], "b": ["b"]},
        "metadata_transform": lambda df: df.assign(
            x=df["label"].eq("b").astype(float)
        ),
        "metadata_transform_name": "derive_x",
        "metadata_transform_version": "1",
        "formula": "pattern ~ 1 + x",
        "overwrite": False,
        "name": "encoding",
    }
    kwargs.update(overrides)
    return kwargs


def _make_encoding_store_output(subject_ids):
    """Build minimal run_regression_model_workflow output for cumulative-store tests."""

    subject_ids = [str(subject_id) for subject_id in subject_ids]
    n_subjects = len(subject_ids)
    return {
        "subject_summary_df": pd.DataFrame(
            {
                "subject": subject_ids,
                "n_trials": [2] * n_subjects,
                "n_channels": [1] * n_subjects,
                "n_times": [2] * n_subjects,
                "n_folds": [1] * n_subjects,
                "condition_levels": ["a,b"] * n_subjects,
            }
        ),
        "skipped_subjects_df": pd.DataFrame(columns=["subject", "reason"]),
        "run_summary_df": pd.DataFrame({"name": ["encoding"]}),
        "pattern_expression_trial_df": pd.DataFrame(
            {
                "subject": subject_ids,
                "fold": [1] * n_subjects,
                "trial_index": [0] * n_subjects,
                "condition": ["a"] * n_subjects,
                "effect": ["x"] * n_subjects,
                "time_ms": [0.0] * n_subjects,
                "expression": [0.2] * n_subjects,
                "covariance_method": ["identity"] * n_subjects,
            }
        ),
        "condition_pattern_expression_df": pd.DataFrame(
            {
                "subject": subject_ids,
                "condition": ["a"] * n_subjects,
                "effect": ["x"] * n_subjects,
                "time_ms": [0.0] * n_subjects,
                "expression_mean": [0.2] * n_subjects,
                "expression_sd": [0.0] * n_subjects,
                "expression_se": [0.0] * n_subjects,
                "n_trials": [1] * n_subjects,
                "covariance_method": ["identity"] * n_subjects,
            }
        ),
        "effect_slope_df": pd.DataFrame(
            {
                "subject": subject_ids,
                "effect": ["x"] * n_subjects,
                "time_ms": [0.0] * n_subjects,
                "slope": [0.2] * n_subjects,
                "intercept": [0.0] * n_subjects,
                "n_trials": [1] * n_subjects,
                "n_folds": [1] * n_subjects,
                "covariance_method": ["identity"] * n_subjects,
            }
        ),
        "design_diagnostics_df": pd.DataFrame(
            {"subject": subject_ids, "diagnostic": ["rank_X"] * n_subjects}
        ),
        "covariance_diagnostics_df": pd.DataFrame(
            {"subject": subject_ids, "covariance_method": ["identity"] * n_subjects}
        ),
        "subject_payloads": {
            subject: _make_encoding_store_payload(subject, offset=subject_ix)
            for subject_ix, subject in enumerate(subject_ids)
        },
    }


def _make_encoding_store_payload(subject: str, offset: int) -> dict[str, np.ndarray]:
    """Build a compact subject beta payload for DuckDB long-table storage."""

    return {
        "subject": np.asarray(subject, dtype=object),
        "times_s": np.asarray([0.0, 0.05], dtype=float),
        "ch_names": np.asarray(["Cz"], dtype=object),
        "predictor_names": np.asarray(["intercept", "x"], dtype=object),
        "raw_beta_patterns": np.asarray(
            [[[[1.0 + offset, 2.0 + offset]], [[3.0 + offset, 4.0 + offset]]]],
            dtype=float,
        ),
    }


def _make_pattern_expression_input(offset: float) -> dict[str, object]:
    """Build minimal per-subject pattern-expression input arrays."""

    return {
        "condition_labels": np.asarray(["a", "b"], dtype=object),
        "times": np.asarray([0.0, 0.1], dtype=float),
        "expression_by_effect": {
            "x": np.asarray(
                [
                    [offset, offset + 1.0],
                    [offset + 2.0, offset + 3.0],
                ],
                dtype=float,
            )
        },
        "trial_index": np.asarray([0, 1], dtype=int),
    }


def _make_topography_payload(raw_offset: float) -> dict[str, np.ndarray]:
    """Build a small saved encoding payload with fold/predictor/channel/time betas."""

    base = np.asarray(
        [
            [
                [[1.0, 3.0, 99.0], [3.0, 5.0, 99.0]],
                [[9.0, 11.0, 99.0], [11.0, 13.0, 99.0]],
            ],
            [
                [[3.0, 5.0, 99.0], [5.0, 7.0, 99.0]],
                [[11.0, 13.0, 99.0], [13.0, 15.0, 99.0]],
            ],
        ],
        dtype=float,
    )
    return {
        "subject": np.asarray("001", dtype=object),
        "times_s": np.asarray([0.0, 0.05, 0.10], dtype=float),
        "ch_names": np.asarray(["Fz", "Cz"], dtype=object),
        "predictor_names": np.asarray(["intercept", "item"], dtype=object),
        "raw_beta_patterns": base + float(raw_offset),
    }


class _ProgressStub:
    """Minimal progress-bar stub for direct subject-workflow tests."""

    def set_postfix_str(self, _text):
        """Accept progress text without doing any UI work."""

    def update(self, _amount):
        """Accept progress updates without doing any UI work."""
