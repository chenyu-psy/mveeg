"""Orthogonal condition decomposition and legacy behavior contracts."""

import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from mveeg.encoding._analysis import encode_subject
from mveeg.encoding._metrics import fit_orthogonal
from mveeg.encoding._prepare import build_design
from tests.encoding.test_encoding import _build_dataset, _pipeline, _subject_arrays


def _condition_problem():
    rng = np.random.default_rng(18)
    theory = np.column_stack([np.ones(5), [1, 2, 3, 4, 5], [1, 3, 2, 5, 4]])
    weights = np.array([1, 2, 1, 3, 2]) / 9
    labels = np.repeat(np.arange(5), [4, 8, 5, 9, 6])
    indicators = np.eye(5)[labels]
    data = rng.normal(size=(len(labels), 4, 2))
    data += (theory[labels] @ rng.normal(size=(3, 4)))[:, :, None]
    return data, indicators, theory, weights


def test_weighted_orthogonality_and_separate_penalty_roles():
    data, indicators, theory, weights = _condition_problem()
    first = fit_orthogonal(data, indicators, theory, weights, 1, 0.1)
    second = fit_orthogonal(data, indicators, theory, weights, 1, 10)
    np.testing.assert_allclose(first[:3], second[:3], atol=1e-12)
    np.testing.assert_allclose(
        np.einsum("ca,c,cbt->abt", theory, weights, first[3:]), 0, atol=1e-12
    )
    np.testing.assert_allclose(first[3:] * 1.1, second[3:] * 11, atol=1e-12)
    means = indicators.T @ data.reshape(len(data), -1) / indicators.sum(axis=0)[:, None]
    # Independent normal-equation characterization of the constrained minimizer.
    rhs = theory.T @ (weights[:, None] * means)
    system = theory.T @ (weights[:, None] * theory) + np.diag([0, 1, 1])
    np.testing.assert_allclose(first[:3].reshape(3, -1), np.linalg.solve(system, rhs), atol=1e-12)


def test_no_observed_zero_and_saturated_incomplete_factorial():
    theory = np.array([[1.0, 1, 0], [1, 0, 1], [1, 1, 1]])
    indicators = np.repeat(np.eye(3), [4, 7, 5], axis=0)
    data = (indicators @ theory @ np.array([[2.0, 3], [1, 0.2], [0.3, 1]]))[:, :, None]
    beta = fit_orthogonal(data, indicators, theory, np.ones(3) / 3, 0.1, 0.1)
    assert np.isfinite(beta).all()
    np.testing.assert_allclose(beta[3:], 0, atol=1e-12)


def test_aliases_and_trial_varying_codes_have_explicit_method_boundary():
    labels = np.array(["A", "B"] * 4, dtype=object)
    metadata = pd.DataFrame({"a": [0.0, 1] * 4, "alias": [0.0, 1] * 4})
    settings = dict(
        formula="a+alias",
        training_labels=labels,
        condition_order=["A", "B"],
        penalty={"component": 1.0, "condition": 0.1},
    )
    assert build_design(metadata, **settings).method == "ridge"
    with pytest.raises(ValueError, match="not identifiable"):
        build_design(metadata, **settings, method="orthogonal")
    metadata.loc[0, "a"] = 0.5
    with pytest.raises(ValueError, match="condition-constant"):
        build_design(metadata, **{**settings, "formula": "a"}, method="orthogonal")


def test_regularization_stabilizes_near_collinear_low_snr_patterns():
    rng = np.random.default_rng(43)
    theory = np.column_stack(
        [np.ones(5), np.arange(5), np.arange(5) + 0.001 * np.array([0, 1, -1, 1, -1])]
    )
    indicators = np.repeat(np.eye(5), 4, axis=0)
    estimates = [[], []]
    for _ in range(100):
        data = rng.normal(size=(20, 3, 1))
        for i, penalty in enumerate([0, 1]):
            estimates[i].append(
                fit_orthogonal(data, indicators, theory, np.ones(5) / 5, penalty, 0.1)[:3]
            )
    assert np.var(estimates[1]) < np.var(estimates[0]) / 100
    assert np.isfinite(estimates[1]).all()


def test_actual_cv_modes_and_legacy_outputs_are_unchanged():
    data, metadata, labels = _subject_arrays()
    settings = dict(
        subject="001",
        data=data,
        metadata=metadata,
        training_labels=labels,
        expression_labels=labels,
        trial_ids=np.arange(len(labels)),
        times=np.array([25, 75]),
        condition_order=["A", "B", "AB"],
        formula="a+b",
        penalty={"component": 1.0, "condition": 0.1},
        folds=3,
        seed=4,
        progress=False,
    )
    original = encode_subject(**settings)
    explicit = encode_subject(**settings, method="ridge")
    for name in original.tables:
        pd.testing.assert_frame_equal(original.tables[name], explicit.tables[name])
    orthogonal = encode_subject(**settings, method="orthogonal")
    assert len(orthogonal.tables["pattern_expression"]) == len(
        original.tables["pattern_expression"]
    )
    assert np.isfinite(orthogonal.tables["coefficients"].beta).all()
    assert (
        orthogonal.tables["design_diagnostics"]
        .query("diagnostic == 'model_rank'")
        .status.eq("ok")
        .all()
    )


def test_pipeline_saves_method_weights_and_rejects_reusing_other_estimand(tmp_path):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])
    pipeline = _pipeline(dataset)
    file = tmp_path / "orthogonal.duckdb"
    settings = dict(
        formula="a",
        target="condition",
        conditions={"A": ["A"], "B": ["B"], "AB": ["AB"]},
        expression={"A": ["A"], "B": ["B"], "AB": ["AB"], "probe": ["P"]},
        file=file,
        progress=False,
    )
    pipeline.setup_model(method="orthogonal", weights={"A": 1, "B": 2, "AB": 3})
    pipeline.encode(**settings)
    with duckdb.connect(str(file), read_only=True) as connection:
        config = json.loads(connection.execute("SELECT config FROM analysis").fetchone()[0])
        assert config["model"]["method"] == "orthogonal"
        assert config["model"]["weights"] == {"A": 1 / 6, "B": 2 / 6, "AB": 3 / 6}
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM pattern_expression WHERE expression_group='probe'"
            ).fetchone()[0]
            > 0
        )
        residual = connection.execute(
            "SELECT predictor, channel, time, beta FROM coefficients "
            "WHERE predictor LIKE 'condition[%]' ORDER BY predictor, channel, time"
        ).fetchdf()
        assert np.isfinite(residual.beta).all()
    pipeline.setup_model()
    with pytest.raises(ValueError, match="incompatible"):
        pipeline.encode(**settings)


@pytest.mark.parametrize("weights", [{"A": 1}, {"A": 1, "B": 0, "AB": 1}])
def test_invalid_condition_measure_is_rejected_before_results(tmp_path, weights):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])
    pipeline = _pipeline(dataset).setup_model(method="orthogonal", weights=weights)
    file = tmp_path / "invalid.duckdb"
    with pytest.raises(ValueError, match="weights|positive"):
        pipeline.encode(
            formula="a",
            target="condition",
            conditions={"A": ["A"], "B": ["B"], "AB": ["AB"]},
            file=file,
            progress=False,
        )
    assert not file.exists()
