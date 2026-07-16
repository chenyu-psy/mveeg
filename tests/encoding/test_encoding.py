"""Encoding pipeline, numerical model, and DuckDB contract."""

from __future__ import annotations

import json

import duckdb
import mne
import numpy as np
import pandas as pd
import pytest

from mveeg._dataset.store import DatasetBuilder
from mveeg.encoding import EncodingPipeline, init_pipeline
from mveeg.encoding._analysis import encode_subject
from mveeg.encoding._metrics import (
    compute_pattern_expression,
    estimate_channel_covariance,
    fit_ridge,
)
from mveeg.encoding._prepare import build_design, parse_formula, validate_groups
from mveeg.prep import open_pipeline


def test_formula_and_design_use_component_condition_roles():
    metadata = pd.DataFrame(
        {
            "a": [1.0, 0.0, 1.0] * 3,
            "b": [0.0, 1.0, 1.0] * 3,
        }
    )
    labels = np.tile(["A", "B", "AB"], 3).astype(object)
    parsed = parse_formula("1 + a * b", allowed_predictors=set(metadata.columns))
    assert parsed == {
        "predictors": ["a", "b", "a:b"],
        "interactions": ["a:b"],
    }

    design = build_design(
        metadata,
        formula="1 + a + b",
        training_labels=labels,
        condition_order=["A", "B", "AB"],
        penalty={"component": 1.0, "condition": 0.1},
    )
    assert design.predictors["role"].tolist() == [
        "intercept",
        "component",
        "component",
        "condition",
        "condition",
        "condition",
    ]
    assert design.predictors["penalty"].tolist() == [0.0, 1.0, 1.0, 0.1, 0.1, 0.1]
    assert np.linalg.matrix_rank(design.matrix) < design.matrix.shape[1]


@pytest.mark.parametrize(
    "formula, message",
    [
        ("pattern ~ 1 + a", "right-hand-side"),
        ("1 + a + (1 | condition)", "Unsupported"),
        ("0 + a", "always includes an intercept"),
        ("1", "at least one component"),
    ],
)
def test_formula_rejects_historical_or_unsupported_syntax(formula, message):
    with pytest.raises(ValueError, match=message):
        parse_formula(formula, allowed_predictors={"a", "condition"})


def test_component_exact_alias_is_handled_by_ridge():
    metadata = pd.DataFrame({"a": [0.0, 1.0] * 4, "duplicate": [0.0, 1.0] * 4})
    labels = np.array(["A", "B"] * 4, dtype=object)
    design = build_design(
        metadata,
        formula="1 + a + duplicate",
        training_labels=labels,
        condition_order=["A", "B"],
        penalty={"component": 1.0, "condition": 0.1},
    )
    assert np.linalg.matrix_rank(design.matrix) < design.matrix.shape[1]
    beta = fit_ridge(
        np.ones((len(metadata), 1, 1)),
        design.matrix,
        design.predictors["penalty"].to_numpy(),
    )
    assert np.isfinite(beta).all()


def test_ridge_preserves_original_channel_scale():
    x = np.repeat([0.0, 1.0], 10)
    design = np.column_stack([np.ones(len(x)), x])
    data = np.empty((len(x), 2, 1))
    data[:, 0, 0] = 2.0 + 3.0 * x
    data[:, 1, 0] = 20.0 + 30.0 * x
    beta = fit_ridge(data, design, np.array([0.0, 1e-9]))
    np.testing.assert_allclose(beta[:, 1], beta[:, 0] * 10.0, rtol=1e-10)


def test_correlation_shrinkage_is_equivariant_to_channel_scaling():
    rng = np.random.default_rng(4)
    residuals = rng.multivariate_normal(
        np.zeros(3),
        np.array([[1.0, 0.5, 0.2], [0.5, 2.0, 0.1], [0.2, 0.1, 0.7]]),
        size=80,
    )
    scales = np.array([0.2, 5.0, 2.0])
    first = estimate_channel_covariance(residuals)
    second = estimate_channel_covariance(residuals * scales)
    np.testing.assert_allclose(
        second.covariance,
        scales[:, None] * first.covariance * scales[None, :],
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        second.precision,
        (1 / scales)[:, None] * first.precision * (1 / scales)[None, :],
        rtol=1e-10,
        atol=1e-12,
    )

    data = rng.normal(size=(5, 3, 1))
    beta = rng.normal(size=(2, 3, 1))
    expression_a, _ = compute_pattern_expression(
        data,
        beta,
        first.precision[None, :, :],
    )
    expression_b, _ = compute_pattern_expression(
        data * scales[None, :, None],
        beta * scales[None, :, None],
        second.precision[None, :, :],
    )
    np.testing.assert_allclose(expression_a, expression_b, rtol=1e-10, atol=1e-12)


def test_default_penalties_recover_a_b_and_combined_expression():
    rng = np.random.default_rng(15)
    condition = np.repeat(["A", "B", "AB"], 30)
    a = np.isin(condition, ["A", "AB"]).astype(float)
    b = np.isin(condition, ["B", "AB"]).astype(float)
    metadata = pd.DataFrame({"a": a, "b": b})
    pattern_a = np.array([1.5, 0.2, 0.0])
    pattern_b = np.array([0.1, 1.4, 0.2])
    data = (
        0.4
        + a[:, None, None] * pattern_a[None, :, None]
        + b[:, None, None] * pattern_b[None, :, None]
        + rng.normal(scale=0.12, size=(len(condition), 3, 1))
    )
    result = encode_subject(
        subject="001",
        data=data,
        metadata=metadata,
        training_labels=condition.astype(object),
        expression_labels=condition.astype(object),
        trial_ids=np.arange(len(condition)),
        times=np.array([25]),
        condition_order=["A", "B", "AB"],
        formula="1 + a + b",
        penalty={"component": 1.0, "condition": 0.1},
        folds=5,
        seed=3,
        progress=False,
    )
    summary = (
        result.tables["pattern_expression"]
        .groupby(["expression_group", "component"])["expression"]
        .mean()
        .unstack()
    )
    assert summary.loc["A", "a"] > summary.loc["B", "a"]
    assert summary.loc["AB", "a"] > summary.loc["B", "a"]
    assert summary.loc["B", "b"] > summary.loc["A", "b"]
    assert summary.loc["AB", "b"] > summary.loc["A", "b"]
    assert summary.loc["A", "a"] > 0
    assert summary.loc["B", "b"] > 0
    assert (
        result.tables["pattern_expression"].groupby(["epoch_index", "component"]).size().eq(1).all()
    )
    assert "fold" not in result.tables["coefficients"]


def test_serial_and_parallel_folds_are_identical():
    data, metadata, labels = _subject_arrays()
    settings = {
        "subject": "001",
        "data": data,
        "metadata": metadata,
        "training_labels": labels,
        "expression_labels": labels,
        "trial_ids": np.arange(len(labels)),
        "times": np.array([25, 75]),
        "condition_order": ["A", "B", "AB"],
        "formula": "1 + a + b",
        "penalty": {"component": 1.0, "condition": 0.1},
        "folds": 3,
        "seed": 8,
        "progress": False,
    }
    serial = encode_subject(**settings, n_jobs=1)
    parallel = encode_subject(**settings, n_jobs=2)
    pd.testing.assert_frame_equal(serial.predictors, parallel.predictors)
    for table in serial.tables:
        pd.testing.assert_frame_equal(serial.tables[table], parallel.tables[table])


def test_heldout_trial_cannot_change_its_fold_model():
    data, metadata, labels = _subject_arrays()
    settings = {
        "subject": "001",
        "metadata": metadata,
        "training_labels": labels,
        "expression_labels": labels,
        "trial_ids": np.arange(len(labels)),
        "times": np.array([25, 75]),
        "condition_order": ["A", "B", "AB"],
        "formula": "1 + a + b",
        "penalty": {"component": 1.0, "condition": 0.1},
        "folds": 3,
        "seed": 8,
        "progress": False,
    }
    first = encode_subject(data=data, **settings).tables["pattern_expression"]
    heldout_fold = int(first.loc[first["epoch_index"].eq(0), "fold"].iloc[0])

    changed = data.copy()
    changed[0] += 100
    second = encode_subject(data=changed, **settings).tables["pattern_expression"]
    same_fold = first["fold"].eq(heldout_fold) & first["epoch_index"].ne(0)
    pd.testing.assert_frame_equal(
        first.loc[same_fold].reset_index(drop=True),
        second.loc[same_fold].reset_index(drop=True),
    )
    assert not np.allclose(
        first.loc[first["epoch_index"].eq(0), "expression"],
        second.loc[second["epoch_index"].eq(0), "expression"],
    )


def test_pipeline_is_lazy_chained_and_validates_model_settings(tmp_path, monkeypatch):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])

    def fail_read(*args, **kwargs):
        raise AssertionError("init_pipeline must not load epochs")

    monkeypatch.setattr(mne, "read_epochs", fail_read)
    pipeline = init_pipeline(dataset)
    assert isinstance(pipeline, EncodingPipeline)
    assert pipeline.subject_indices == ("001",)
    assert pipeline.transform_metadata(a=lambda frame: frame["condition"].eq("A")) is pipeline
    assert pipeline.select_trials(qc=None) is pipeline
    assert pipeline.prepare_epochs(crop=None) is pipeline
    assert pipeline.setup_model() is pipeline
    assert pipeline.setup_cv() is pipeline
    with pytest.raises(TypeError):
        init_pipeline(open_pipeline(dataset))
    with pytest.raises(ValueError, match="exactly"):
        pipeline.setup_model(penalty={"component": 1.0})
    with pytest.raises(ValueError, match="positive"):
        pipeline.setup_model(penalty={"component": 0.0, "condition": 0.1})


def test_pipeline_writes_public_duckdb_and_condition_view(tmp_path):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])
    result_file = tmp_path / "encoding.duckdb"
    pipeline = _pipeline(dataset)
    assert (
        pipeline.encode(
            formula="1 + a + b",
            target="condition",
            conditions={"A": ["A"], "B": ["B"], "AB": ["AB"]},
            expression={"A": ["A"], "B": ["B"], "AB": ["AB"], "probe": ["P"]},
            file=result_file,
            progress=False,
        )
        is None
    )

    with duckdb.connect(str(result_file), read_only=True) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='main' AND table_type='BASE TABLE'"
            ).fetchall()
        }
        views = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.views WHERE table_schema='main'"
            ).fetchall()
        }
        assert tables == {
            "analysis",
            "subjects",
            "trials",
            "predictors",
            "channels",
            "time_bins",
            "coefficients",
            "pattern_expression",
            "design_diagnostics",
            "covariance_diagnostics",
        }
        assert "condition_pattern_expression" in views
        assert connection.execute("SELECT status FROM subjects").fetchone()[0] == "complete"
        assert connection.execute(
            "SELECT DISTINCT role FROM predictors ORDER BY role"
        ).fetchall() == [("component",), ("condition",), ("intercept",)]
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM coefficients WHERE predictor LIKE 'condition[%]'"
            ).fetchone()[0]
            > 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM trials WHERE condition='P' AND training_group IS NULL"
            ).fetchone()[0]
            == 3
        )
        n_expression = connection.execute("SELECT COUNT(*) FROM pattern_expression").fetchone()[0]
        n_trials = connection.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
        n_times = connection.execute("SELECT COUNT(*) FROM time_bins").fetchone()[0]
        assert n_expression == n_trials * n_times * 2
        assert (
            connection.execute(
                "SELECT MAX(n_trials) FROM condition_pattern_expression WHERE condition='probe'"
            ).fetchone()[0]
            == 3
        )
        config = json.loads(
            connection.execute("SELECT config::VARCHAR FROM analysis").fetchone()[0]
        )
        assert config["metadata_variables"] == ["a", "b"]
        assert config["model"]["penalty"] == {"component": 1.0, "condition": 0.1}


def test_pipeline_incrementally_adds_subject_and_rejects_config_change(tmp_path):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])
    result_file = tmp_path / "encoding.duckdb"
    pipeline = _pipeline(dataset)
    arguments = {
        "formula": "1 + a + b",
        "target": "condition",
        "conditions": {"A": ["A"], "B": ["B"], "AB": ["AB"]},
        "file": result_file,
        "progress": False,
    }
    pipeline.encode(**arguments)
    with duckdb.connect(str(result_file), read_only=True) as connection:
        original = connection.execute(
            "SELECT * FROM coefficients WHERE subject_index='001' ORDER BY ALL"
        ).df()

    _extend_dataset(dataset, "002")
    pipeline.encode(**arguments)
    with duckdb.connect(str(result_file), read_only=True) as connection:
        assert connection.execute(
            "SELECT subject_index, status FROM subjects ORDER BY subject_index"
        ).fetchall() == [("001", "complete"), ("002", "complete")]
        saved = connection.execute(
            "SELECT * FROM coefficients WHERE subject_index='001' ORDER BY ALL"
        ).df()
    pd.testing.assert_frame_equal(original, saved)

    with pytest.raises(ValueError, match="incompatible analysis settings"):
        pipeline.encode(**{**arguments, "formula": "1 + a"})
    pipeline.encode(**{**arguments, "formula": "1 + a", "recompute": "all"})


def test_failed_subject_has_no_partial_result_rows(tmp_path):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])
    result_file = tmp_path / "failed.duckdb"
    pipeline = _pipeline(dataset)
    with pytest.raises(RuntimeError, match="No subject completed encoding"):
        pipeline.encode(
            formula="1 + a + b",
            target="condition",
            conditions={"A": ["A"], "missing": ["missing"]},
            file=result_file,
            progress=False,
        )

    with duckdb.connect(str(result_file), read_only=True) as connection:
        status, reason = connection.execute(
            "SELECT status, reason FROM subjects WHERE subject_index='001'"
        ).fetchone()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='main' AND table_type='BASE TABLE'"
            ).fetchall()
        }
    assert status == "failed"
    assert reason
    assert tables == {"analysis", "subjects"}


def test_group_validation_keeps_training_and_expression_independent():
    conditions, expression = validate_groups(
        {"training": ["A", "B"]},
        {"low": ["A"], "high": ["B"], "probe": ["P"]},
    )
    assert conditions == {"training": ["A", "B"]}
    assert expression == {"low": ["A"], "high": ["B"], "probe": ["P"]}
    with pytest.raises(ValueError, match="must include every"):
        validate_groups({"training": ["A", "B"]}, {"only_a": ["A"]})


def _subject_arrays():
    rng = np.random.default_rng(12)
    condition = np.repeat(["A", "B", "AB"], 12)
    a = np.isin(condition, ["A", "AB"]).astype(float)
    b = np.isin(condition, ["B", "AB"]).astype(float)
    metadata = pd.DataFrame({"a": a, "b": b})
    data = rng.normal(scale=0.2, size=(len(condition), 3, 2))
    data += a[:, None, None] * np.array([1.0, 0.2, 0.0])[None, :, None]
    data += b[:, None, None] * np.array([0.1, 1.0, 0.2])[None, :, None]
    return data, metadata, condition.astype(object)


def _epochs(subject: str) -> mne.EpochsArray:
    rng = np.random.default_rng(int(subject))
    labels = np.repeat(["A", "B", "AB", "P"], [6, 6, 6, 3])
    a = np.isin(labels, ["A", "AB"]).astype(float)
    b = np.isin(labels, ["B", "AB"]).astype(float)
    data = rng.normal(scale=0.2, size=(len(labels), 3, 11))
    data += a[:, None, None] * np.array([1.0, 0.2, 0.0])[None, :, None]
    data += b[:, None, None] * np.array([0.1, 1.0, 0.2])[None, :, None]
    info = mne.create_info(["Fz", "Cz", "Pz"], sfreq=100, ch_types="eeg")
    info.set_montage("standard_1020")
    metadata = pd.DataFrame(
        {
            "subject_index": [subject] * len(labels),
            "epoch_index": np.arange(len(labels)),
            "condition": labels,
            "final_status": ["accepted"] * len(labels),
        }
    )
    return mne.EpochsArray(
        data,
        info,
        events=np.column_stack(
            [
                np.arange(len(labels)) * 20,
                np.zeros(len(labels), dtype=int),
                np.ones(len(labels), dtype=int),
            ]
        ),
        event_id={"epoch_index": 1},
        tmin=0,
        metadata=metadata,
        verbose="ERROR",
    )


def _build_dataset(root, subjects):
    builder = DatasetBuilder(
        root,
        task="task",
        stage="preprocessed",
        pipeline_fingerprint="pipeline",
        pipeline_spec={"stage": "test"},
        recompute="never",
        subject_indices=subjects,
        complete_subject_set=True,
    )
    for subject in subjects:
        builder.write_subject(subject, _epochs(subject), input_fingerprint=f"input-{subject}")
    builder.finish()
    return root


def _extend_dataset(root, subject):
    builder = DatasetBuilder(
        root,
        task="task",
        stage="preprocessed",
        pipeline_fingerprint="pipeline",
        pipeline_spec={"stage": "test"},
        recompute="never",
        subject_indices=["001", subject],
        complete_subject_set=True,
    )
    builder.record_reused("001")
    builder.write_subject(subject, _epochs(subject), input_fingerprint=f"input-{subject}")
    builder.finish()


def _pipeline(dataset):
    pipeline = init_pipeline(dataset)
    pipeline.transform_metadata(
        a=lambda frame: frame["condition"].isin(["A", "AB"]).astype(float),
        b=lambda frame: frame["condition"].isin(["B", "AB"]).astype(float),
    )
    pipeline.prepare_epochs(crop=None, time_bin=50)
    pipeline.setup_cv(folds=2, seed=4)
    return pipeline
