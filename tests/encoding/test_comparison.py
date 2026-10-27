"""Scientific contracts for independent condition-subspace strength."""

import json
import warnings

import duckdb
import numpy as np
import pandas as pd
import pytest

from mveeg import prep
from mveeg.encoding._comparison import (
    build_projectors,
    compare_subject,
    condition_weights,
    noise_precision,
    partition_trials,
)
from tests.encoding.test_encoding import _build_dataset, _pipeline


def _arrays(seed=3):
    rng = np.random.default_rng(seed)
    labels = np.tile(np.repeat(["A", "B", "AB", "other"], [8, 5, 7, 4]), 4)
    group = np.repeat(np.arange(4), 24)
    a = np.isin(labels, ["A", "AB"]).astype(float)
    b = np.isin(labels, ["B", "AB"]).astype(float)
    metadata = pd.DataFrame({"a": a, "b": b, "scaled_a": 3 * a})
    data = rng.normal(size=(len(labels), 5, 2))
    data[:, 0, :] += 2 * a[:, None]
    data[:, 1, :] += 2 * b[:, None]
    return data, metadata, labels, group


def _compare(data, metadata, labels, group, **overrides):
    settings = dict(
        subject="001",
        data=data,
        metadata=metadata,
        labels=labels,
        trial_ids=np.arange(len(labels)),
        times=np.array([25, 75]),
        conditions=["A", "B", "AB", "other"],
        models={"a": "a", "ab": "a + b", "scaled": "scaled_a", "b": "b"},
        nuisance=None,
        weights=condition_weights(["A", "B", "AB", "other"], None),
        groups=group,
        folds=4,
        seed=7,
    )
    settings.update(overrides)
    return compare_subject(**settings)


def test_joint_projector_nested_same_span_and_non_nested():
    data, metadata, labels, groups = _arrays()
    tables = _compare(data, metadata, labels, groups)
    comparison = tables["comparisons"]
    same = comparison.query("model_1 == 'a' and model_2 == 'scaled'")
    assert same["same_span"].all()
    np.testing.assert_allclose(same.delta_strength, 0, atol=1e-12)
    added = comparison.query("model_1 == 'a' and model_2 == 'ab'")
    assert added.nested.all() and (added.delta_strength > 0).all()
    assert not comparison.query("model_1 == 'a' and model_2 == 'b'").nested.any()
    assert tables["strengths"].query("model == 'ab'").model_rank.eq(2).all()
    assert len(tables["trial_groups"]) == len(data)


def test_weighted_nuisance_and_aliases_are_joint_spaces():
    _, metadata, labels, _ = _arrays()
    weights = condition_weights(["A", "B", "AB", "other"], {"A": 1, "B": 3, "AB": 2, "other": 4})
    transform, p, _ = build_projectors(
        metadata,
        labels,
        list(weights),
        {"null": "1", "a": "a", "alias": "a + scaled_a", "all": "a+b"},
        "b",
        weights,
    )
    np.testing.assert_allclose(p["null"], 0, atol=1e-12)
    np.testing.assert_allclose(p["a"], p["alias"], atol=1e-12)
    np.testing.assert_allclose(p["a"], p["all"], atol=1e-12)
    np.testing.assert_allclose(transform @ np.ones(4), 0, atol=1e-12)


def test_null_increment_not_rank_reward_and_shared_artifact_negative_control():
    _, metadata, labels, _ = _arrays()
    weights = condition_weights(["A", "B", "AB", "other"], None)
    transform, p, _ = build_projectors(
        metadata, labels, list(weights), {"small": "a", "large": "a+b"}, None, weights
    )
    delta = p["large"] - p["small"]
    rng = np.random.default_rng(80)
    # Independent condition-mean errors with unequal trial-count variances.
    error_a = rng.normal(size=(3000, 4, 5)) / np.sqrt(np.array([8, 5, 7, 4]))[None, :, None]
    error_b = rng.normal(size=(3000, 4, 5)) / np.sqrt(np.array([8, 5, 7, 4]))[None, :, None]
    za = np.einsum("ij,rjk->rik", transform, error_a)
    zb = np.einsum("ij,rjk->rik", transform, error_b)
    estimates = np.einsum("rik,ij,rjk->r", za, delta, zb)
    assert abs(estimates.mean()) < 4 * estimates.std() / np.sqrt(len(estimates))
    assert (estimates < 0).any()
    nuisance = rng.normal(size=(3000, 4, 5))
    nuisance = np.einsum("ij,rjk->rik", delta, nuisance)
    contaminated = np.einsum("rik,ij,rjk->r", za + nuisance, delta, zb + nuisance)
    assert contaminated.mean() > 1


def test_reference_and_interpolation_rank_loss():
    rng = np.random.default_rng(15)
    residuals = rng.normal(scale=1e-6, size=(24, 12))
    residuals -= residuals.mean(axis=1, keepdims=True)
    precision, rank, _, _ = noise_precision(residuals)
    assert rank == 11
    assert np.isfinite(precision).all()
    assert np.linalg.eigvalsh(precision).min() > 0
    # Sample rank diagnoses reference/interpolation loss but does not define the
    # metric support: shrinkage operates in retained-channel coordinates.
    duplicate = np.column_stack([residuals, residuals[:, 0]])
    _, rank, _, _ = noise_precision(duplicate)
    assert rank == 11


def test_precision_excludes_both_evaluation_groups():
    data, metadata, labels, groups = _arrays()
    first = _compare(data, metadata, labels, groups)
    data[groups == 0] += np.random.default_rng(1).normal(scale=40, size=data[groups == 0].shape)
    second = _compare(data, metadata, labels, groups)
    query = "group_a == '0' and group_b == '1'"
    pd.testing.assert_frame_equal(
        first["covariance_diagnostics"].query(query), second["covariance_diagnostics"].query(query)
    )


def test_actual_complement_precision_null_increment_with_low_snr():
    _, metadata, labels, groups = _arrays()
    rng = np.random.default_rng(30)
    increments = []
    for _ in range(200):
        data = rng.normal(size=(len(labels), 5, 1))
        data[:, 0, 0] += 0.2 * metadata.a.to_numpy()
        result = _compare(
            data,
            metadata,
            labels,
            groups,
            models={"small": "a", "large": "a+b"},
            times=np.array([25]),
        )
        increments.append(result["comparisons"].delta_strength.iloc[0])
    increments = np.array(increments)
    assert abs(increments.mean()) < 4 * increments.std() / np.sqrt(len(increments))
    assert (increments < 0).any()


def test_under_sampled_channel_support_remains_finite():
    residuals = np.random.default_rng(12).normal(scale=1e-6, size=(9, 64))
    precision, rank, shrinkage, condition_number = noise_precision(residuals)
    assert rank == 8
    assert shrinkage > 0
    assert np.isfinite(precision).all() and np.isfinite(condition_number)


def test_sample_null_direction_retains_finite_signal_energy():
    residuals = np.random.default_rng(12).normal(size=(9, 64))
    centered = residuals - residuals.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=True)
    pattern = vt[len(singular)]
    precision, rank, _, _ = noise_precision(residuals)
    assert rank == 8
    assert pattern @ precision @ pattern > 0.1
    np.testing.assert_allclose(centered @ pattern, 0, atol=1e-12)


def test_noise_metric_preserves_channel_scale_and_rejects_flat_noise():
    noise = np.random.default_rng(13).normal(size=(20, 8))
    scales = np.array([10.0, 1, 3, 0.5, 2, 0.2, 1, 4])
    first, _, _, _ = noise_precision(noise)
    second, _, _, _ = noise_precision(noise * scales)
    np.testing.assert_allclose(
        second, first / scales[:, None] / scales[None, :], rtol=1e-10, atol=1e-12
    )
    noise[:, 0] = 0
    with pytest.raises(ValueError, match="flat channels"):
        noise_precision(noise)


def test_complement_df_warning_and_no_noise_information_failure():
    _, metadata, labels, groups = _arrays()
    data = np.random.default_rng(19).normal(size=(len(labels), 64, 2))
    with pytest.warns(RuntimeWarning, match="may be sample-limited") as recorded:
        result = _compare(data, metadata, labels, groups)
    assert len(recorded) == 1
    assert result["covariance_diagnostics"].noise_df.max() < 64
    assert np.isfinite(result["strengths"].strength).all()
    # Three complete groups with one trial per condition leave zero residual df
    # after holding two groups out. Shrinkage cannot manufacture repetitions.
    selected = np.concatenate(
        [
            np.flatnonzero((groups == group) & (labels == condition))[:1]
            for group in range(3)
            for condition in ["A", "B", "AB", "other"]
        ]
    )
    with pytest.raises(ValueError, match="noise degrees of freedom"):
        _compare(data[selected], metadata.iloc[selected], labels[selected], groups[selected])


def _few_channel_comparison(noise_df):
    counts = [2, noise_df]
    labels = np.tile(np.repeat(["A", "B"], counts), 3).astype(object)
    groups = np.repeat(np.arange(3), sum(counts))
    data = np.random.default_rng(71).normal(size=(len(labels), 1, 1))
    return compare_subject(
        subject="001",
        data=data,
        metadata=pd.DataFrame({"a": labels == "B"}),
        labels=labels,
        trial_ids=np.arange(len(labels)),
        times=np.array([25]),
        conditions=["A", "B"],
        models={"null": "1", "a": "a"},
        nuisance=None,
        weights={"A": 0.5, "B": 0.5},
        groups=groups,
        folds=3,
        seed=5,
    )


def test_exactly_two_noise_df_rejected_even_with_one_channel():
    with pytest.raises(ValueError, match="inverse-variance mean"):
        _few_channel_comparison(2)


@pytest.mark.parametrize("noise_df", [3, 4])
def test_three_four_noise_df_warn_even_with_one_channel(noise_df):
    with pytest.warns(RuntimeWarning, match="second moments can diverge") as recorded:
        result = _few_channel_comparison(noise_df)
    assert len(recorded) == 1
    assert result["covariance_diagnostics"].noise_df.eq(noise_df).all()


def test_five_noise_df_few_channels_has_no_boundary_warning():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = _few_channel_comparison(5)
    assert not recorded
    assert result["covariance_diagnostics"].noise_df.eq(5).all()


def test_sparse_confounded_and_trial_varying_designs_rejected():
    data, metadata, labels, groups = _arrays()
    with pytest.raises(ValueError, match="at least three"):
        partition_trials(labels, groups=groups % 2, folds=4, seed=3)
    keep = ~((groups == 0) & (labels == "A"))
    with pytest.raises(ValueError, match="missing conditions"):
        _compare(data[keep], metadata[keep], labels[keep], groups[keep])
    metadata.loc[0, "a"] = 0.5
    with pytest.raises(ValueError, match="condition-constant"):
        _compare(data, metadata, labels, groups)


def test_comparison_store_paired_metadata_resume_and_config(tmp_path):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])
    pipeline = _pipeline(dataset)
    pipeline.setup_cv(folds=3, seed=4)
    pipeline.transform_metadata(
        a=lambda x: x.condition.isin(["A", "AB"]).astype(float),
        b=lambda x: x.condition.isin(["B", "AB"]).astype(float),
        run=lambda x: x.epoch_index % 3,
    )
    file = tmp_path / "comparison.duckdb"
    settings = dict(
        models={"small": "a", "large": "a+b"},
        target="condition",
        conditions={"A": ["A"], "B": ["B"], "AB": ["AB"]},
        file=file,
        groups="run",
        progress=False,
    )
    pipeline.compare(**settings)
    with duckdb.connect(str(file), read_only=True) as connection:
        expected = connection.execute("SELECT * FROM comparisons ORDER BY time").fetchdf()
        config = json.loads(connection.execute("SELECT config FROM analysis").fetchone()[0])
        assert config["analysis_kind"] == "subspace_strength"
        assert config["estimator_version"] == 2
        assert config["minimum_noise_df"] == 3
        assert config["precision"] == "leave_two_groups_out_full_channel_correlation_shrinkage"
        assert config["weights"] == {"A": 1 / 3, "AB": 1 / 3, "B": 1 / 3}
        assert connection.execute("SELECT COUNT(*) FROM trial_groups").fetchone()[0] == 18
        assert (
            connection.execute("SELECT COUNT(DISTINCT group_label) FROM trial_groups").fetchone()[0]
            == 3
        )
        assert connection.execute("SELECT COUNT(*) FROM model_designs").fetchone()[0] > 0
    pipeline.compare(**settings)
    with duckdb.connect(str(file), read_only=True) as connection:
        pd.testing.assert_frame_equal(
            expected, connection.execute("SELECT * FROM comparisons ORDER BY time").fetchdf()
        )
    with pytest.raises(ValueError, match="incompatible"):
        pipeline.compare(**{**settings, "weights": {"A": 1, "B": 2, "AB": 1}})


def test_failed_subject_has_no_partial_comparison(tmp_path):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])
    pipeline = _pipeline(dataset)
    file = tmp_path / "failed.duckdb"
    with pytest.raises(RuntimeError, match="at least three"):
        pipeline.compare(
            models={"small": "a", "large": "a+b"},
            target="condition",
            conditions={"A": ["A"], "B": ["B"], "AB": ["AB"]},
            file=file,
            progress=False,
        )
    with duckdb.connect(str(file), read_only=True) as connection:
        assert connection.execute("SELECT status FROM subjects").fetchone()[0] == "failed"
        assert not connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name='strengths'"
        ).fetchall()


def test_comparison_write_failure_rolls_back_all_tables(tmp_path, monkeypatch):
    from mveeg.encoding import _comparison_pipeline

    dataset = _build_dataset(tmp_path / "dataset", ["001"])
    pipeline = _pipeline(dataset).setup_cv(folds=3, seed=4)
    file = tmp_path / "failed_write.duckdb"
    original = _comparison_pipeline.insert_frame

    def fail_strength(connection, name, frame):
        if name == "strengths":
            raise RuntimeError("simulated write failure")
        original(connection, name, frame)

    monkeypatch.setattr(_comparison_pipeline, "insert_frame", fail_strength)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        pipeline.compare(
            models={"small": "a", "large": "a+b"},
            target="condition",
            conditions={"A": ["A"], "B": ["B"], "AB": ["AB"]},
            file=file,
            progress=False,
        )
    with duckdb.connect(str(file), read_only=True) as connection:
        assert connection.execute("SELECT status FROM subjects").fetchone()[0] == "failed"
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        assert tables == {"analysis", "subjects"}


def test_comparison_obeys_subject_exclusions(tmp_path):
    dataset = _build_dataset(tmp_path / "dataset", ["001", "002"])
    pipeline = _pipeline(dataset).setup_cv(folds=3, seed=4)
    settings = dict(
        models={"small": "a", "large": "a+b"},
        target="condition",
        conditions={"A": ["A"], "B": ["B"], "AB": ["AB"]},
        file=tmp_path / "selection.duckdb",
        progress=False,
    )
    pipeline.compare(**settings)
    prep.open_pipeline(dataset).select_subjects(exclude={"001": "test exclusion"})
    pipeline.compare(**settings)
    with duckdb.connect(str(settings["file"]), read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM subjects").fetchone()[0] == 2
    pipeline.compare(**{**settings, "recompute": "changed"})
    with duckdb.connect(str(settings["file"]), read_only=True) as connection:
        assert connection.execute("SELECT subject_index FROM subjects").fetchall() == [("002",)]
        assert connection.execute("SELECT DISTINCT subject_index FROM strengths").fetchall() == [
            ("002",)
        ]
