"""Automatic zero-code references through the actual encoding workflow."""

import json

import duckdb
import numpy as np
import pandas as pd
import pytest

import mveeg.encoding._analysis as analysis
from mveeg._provenance import settings_fingerprint
from mveeg.encoding._analysis import encode_subject
from mveeg.encoding._prepare import build_design
from mveeg.encoding._reference import zero_code_references
from mveeg.prep import open_pipeline
from tests.encoding.test_encoding import _build_dataset, _pipeline


def _settings(seed=8):
    rng = np.random.default_rng(51)
    labels = np.repeat(np.array(["A", "B", "AB", "P", "Q"], object), [8, 12, 8, 1, 3])
    a = np.isin(labels, ["A", "AB"]).astype(float)
    b = np.isin(labels, ["B", "AB"]).astype(float)
    p = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0, -0.1, -0.2])
    q = 0.9 * p + np.array([0, 0.1, -0.1, 0, 0.1, 0, 0, 0])
    data = rng.normal(scale=0.5, size=(len(labels), 8, 2))
    data += a[:, None, None] * p[None, :, None]
    data += b[:, None, None] * q[None, :, None]
    data -= data.mean(axis=1, keepdims=True)
    return dict(
        subject="001",
        data=data * 1e-6,
        metadata=pd.DataFrame({"a": a, "b": b, "condition": labels}),
        training_labels=np.array([c if c in ["A", "B", "AB"] else None for c in labels], object),
        expression_labels=labels,
        trial_ids=np.arange(len(labels)),
        times=np.array([25, 75]),
        condition_order=["A", "B", "AB"],
        formula="a+b",
        penalty={"component": 1.0, "condition": 0.1},
        folds=4,
        seed=seed,
        progress=False,
    )


def _run(**overrides):
    settings = _settings()
    settings.update(overrides)
    return encode_subject(**settings)


def test_zero_coded_components_are_referenced_to_their_code_zero_trials():
    result = _run()
    metadata = _settings()["metadata"]
    offsets = result.tables["expression_references"]
    assert set(offsets.component) == {"a", "b"}
    assert offsets.reference_value.eq(0).all()
    audit = result.tables["expression_reference_trials"]
    for component in ("a", "b"):
        rows = audit.query("component == @component")
        assert metadata.iloc[rows.epoch_index][component].eq(0).all()
    assert len(result.tables["referenced_pattern_expression"]) == len(
        result.tables["pattern_expression"]
    )


def test_components_without_zero_or_with_trial_varying_codes_keep_the_default_zero():
    settings = _settings()
    contrast = settings["metadata"].copy()
    contrast.a = contrast.a * 2 - 1
    result = encode_subject(**{**settings, "metadata": contrast})
    assert set(result.tables["expression_references"].component) == {"b"}
    # Ridge with a code that varies within condition A: a is not referenced.
    varied = settings["metadata"].copy()
    varied.loc[varied.condition == "A", "a"] = np.linspace(0.2, 1.4, 8)
    result = encode_subject(**{**settings, "metadata": varied})
    assert set(result.tables["expression_references"].component) == {"b"}
    # No component has code 0: no referenced tables are written.
    shifted = settings["metadata"].copy()
    shifted.a += 1
    shifted.b += 1
    result = encode_subject(**{**settings, "metadata": shifted})
    assert "referenced_pattern_expression" not in result.tables


def test_interaction_reference_pools_every_trial_with_interaction_code_zero():
    labels = np.repeat(np.array(["N", "A", "B", "AB"], object), 2)
    metadata = pd.DataFrame(
        {
            "a": np.isin(labels, ["A", "AB"]).astype(float),
            "b": np.isin(labels, ["B", "AB"]).astype(float),
        }
    )
    design = build_design(
        metadata,
        formula="a*b",
        training_labels=labels,
        condition_order=["N", "A", "B", "AB"],
        penalty={"component": 0.0, "condition": 0.1},
    )
    plans = {plan.component: plan.rows.tolist() for plan in zero_code_references(design, labels)}
    assert plans == {"a": [0, 1, 4, 5], "b": [0, 1, 2, 3], "a:b": [0, 1, 2, 3, 4, 5]}


def test_zero_matching_is_exact():
    labels = np.array(["A", "B"] * 2, object)
    design = build_design(
        pd.DataFrame({"a": [0.1 + 0.2 - 0.3, 1.0] * 2}),
        formula="a",
        training_labels=labels,
        condition_order=["A", "B"],
        penalty={"component": 0.0, "condition": 0.1},
    )
    assert zero_code_references(design, labels) == ()


def test_near_null_pattern_cannot_define_a_reference(monkeypatch):
    settings = _settings()
    original = analysis.compute_pattern_expression

    def undefined_expression(*args, **kwargs):
        expression, warnings = original(*args, **kwargs)
        return np.full_like(expression, np.nan), warnings

    monkeypatch.setattr(analysis, "compute_pattern_expression", undefined_expression)
    with pytest.raises(ValueError, match="nonfinite"):
        encode_subject(**settings)


def test_reference_plans_and_auxiliary_projections_are_identical_in_parallel():
    serial = _run()
    parallel = _run(n_jobs=2)
    for name in serial.tables:
        pd.testing.assert_frame_equal(serial.tables[name], parallel.tables[name])


def test_reference_pools_trials_across_unequal_conditions_and_display_regrouping():
    settings = _settings()
    result = encode_subject(**settings)
    audit = result.tables["expression_reference_trials"]
    offsets = result.tables["expression_references"].set_index(["fold", "component", "time"])
    differs_from_equal_conditions = []
    for key, rows in audit.groupby(["fold", "component", "time"]):
        np.testing.assert_allclose(rows.weight, 1 / len(rows))
        expected = rows.raw_expression.mean()
        np.testing.assert_allclose(offsets.loc[key].reference_expression, expected)
        assert rows.epoch_index.is_unique
        conditions = settings["metadata"].iloc[rows.epoch_index].condition.to_numpy()
        equal_conditions = (
            rows.assign(condition=conditions).groupby("condition").raw_expression.mean().mean()
        )
        differs_from_equal_conditions.append(abs(expected - equal_conditions))
    assert max(differs_from_equal_conditions) > 1e-6
    # Label pooling changes display only, not filter fitting or reference membership.
    regrouped = settings["expression_labels"].copy()
    regrouped[np.isin(regrouped, ["P", "Q"])] = "control"
    other = encode_subject(**{**settings, "expression_labels": regrouped})
    pd.testing.assert_frame_equal(audit, other.tables["expression_reference_trials"])
    pd.testing.assert_frame_equal(
        result.tables["expression_references"], other.tables["expression_references"]
    )


def test_auxiliary_reference_projection_never_duplicates_public_trials_or_uses_training_rows():
    settings = _settings()
    result = encode_subject(**settings)
    public = result.tables["pattern_expression"]
    referenced = result.tables["referenced_pattern_expression"]
    assert not public.duplicated(["epoch_index", "component", "time"]).any()
    assert not referenced.duplicated(["epoch_index", "component", "time"]).any()
    audit = result.tables["expression_reference_trials"]
    # The singleton P observation never trains, so supplies every fold's reference.
    p_row = int(np.flatnonzero(settings["expression_labels"] == "P")[0])
    assert audit.loc[audit.epoch_index == p_row].fold.nunique() == settings["folds"]
    assert audit.loc[audit.epoch_index == p_row].auxiliary.any()
    training_ids = set(np.flatnonzero(pd.notna(settings["training_labels"])))
    for row in audit.itertuples():
        if row.epoch_index in training_ids:
            heldout_fold = public.query(
                "epoch_index == @row.epoch_index and component == 'a'"
            ).fold.unique()
            assert heldout_fold.tolist() == [row.fold]
            assert not row.auxiliary


def test_referencing_preserves_within_fold_differences_and_exposes_common_offset():
    result = _run()
    scores = result.tables["referenced_pattern_expression"]
    np.testing.assert_allclose(
        scores.expression, scores.raw_expression - scores.reference_expression
    )
    for _, rows in scores.groupby(["fold", "component", "time"]):
        assert rows.reference_expression.nunique() == 1
        np.testing.assert_allclose(
            np.diff(rows.expression), np.diff(rows.raw_expression), atol=1e-12
        )
    evidence = result.tables["expression_reference_trials"]
    offsets = result.tables["expression_references"]
    for key, rows in evidence.groupby(["fold", "component", "time"]):
        expected = offsets.set_index(["fold", "component", "time"]).loc[key].reference_expression
        np.testing.assert_allclose(
            np.sum(rows.weight * (rows.raw_expression - expected)), 0, atol=1e-12
        )


def test_heldout_eeg_and_auxiliary_reference_never_refit_encoder_or_noise():
    settings = _settings()
    first = encode_subject(**settings)
    public = first.tables["pattern_expression"]
    fold = int(public.loc[public.epoch_index == 0, "fold"].iloc[0])
    changed = settings["data"].copy()
    changed[0] += np.linspace(-20, 20, changed.shape[1])[:, None] * 1e-6
    second = encode_subject(**{**settings, "data": changed})
    # Trial 0 has a = 1, so it is outside a's reference pool; within its
    # held-out fold, nothing about component a may change except trial 0 itself.
    for table in ("pattern_expression", "referenced_pattern_expression"):
        old = first.tables[table]
        new = second.tables[table]
        keep = (old.fold == fold) & (old.epoch_index != 0) & (old.component == "a")
        pd.testing.assert_frame_equal(
            old.loc[keep].reset_index(drop=True), new.loc[keep].reset_index(drop=True)
        )
    a_offsets = "fold == @fold and component == 'a'"
    pd.testing.assert_frame_equal(
        first.tables["expression_references"].query(a_offsets).reset_index(drop=True),
        second.tables["expression_references"].query(a_offsets).reset_index(drop=True),
    )
    # Changing never-trained reference EEG legitimately changes the observed
    # offset, but must not change any fitted coefficient or covariance estimate.
    changed = settings["data"].copy()
    probe = int(np.flatnonzero(settings["expression_labels"] == "P")[0])
    changed[probe] += np.linspace(-10, 10, changed.shape[1])[:, None] * 1e-6
    auxiliary = encode_subject(**{**settings, "data": changed})
    for table in ("coefficients", "covariance_diagnostics"):
        pd.testing.assert_frame_equal(first.tables[table], auxiliary.tables[table])
    assert not np.allclose(
        first.tables["expression_references"].reference_expression,
        auxiliary.tables["expression_references"].reference_expression,
    )


def _pipeline_case(tmp_path, subjects=("001",)):
    dataset = _build_dataset(tmp_path / "dataset", list(subjects))
    pipeline = _pipeline(dataset)
    settings = dict(
        formula="a+b",
        target="condition",
        conditions={"A": ["A"], "B": ["B"], "AB": ["AB"]},
        expression={"A": ["A"], "B": ["B"], "AB": ["AB"], "probe": ["P"]},
        file=tmp_path / "reference.duckdb",
        progress=False,
    )
    return dataset, pipeline, settings


def test_encode_has_no_reference_argument(tmp_path):
    _, pipeline, settings = _pipeline_case(tmp_path)
    with pytest.raises(TypeError, match="expression_reference"):
        pipeline.encode(**settings, expression_reference={"a": 0})


def test_persistence_views_resume_and_rule_in_config(tmp_path):
    _, pipeline, settings = _pipeline_case(tmp_path)
    pipeline.encode(**settings)
    with duckdb.connect(str(settings["file"]), read_only=True) as connection:
        config = json.loads(connection.execute("SELECT config FROM analysis").fetchone()[0])
        assert config["expression_reference"]["rule"] == (
            "condition_constant_components_with_code_zero"
        )
        original = connection.execute(
            "SELECT * FROM referenced_pattern_expression ORDER BY epoch_index,component,time"
        ).fetchdf()
        assert set(original.component) == {"a", "b"}
        observed = connection.execute(
            "SELECT * FROM condition_referenced_pattern_expression "
            "ORDER BY condition,component,time"
        ).fetchdf()
        direct = connection.execute(
            "SELECT expression_group AS condition,component,time,AVG(expression) AS mean,"
            "COUNT(DISTINCT epoch_index) AS n_trials FROM referenced_pattern_expression "
            "GROUP BY expression_group,component,time ORDER BY condition,component,time"
        ).fetchdf()
        pd.testing.assert_frame_equal(
            observed[["condition", "component", "time", "mean", "n_trials"]], direct
        )
        folds = connection.execute(
            "SELECT * FROM fold_condition_referenced_pattern_expression"
        ).fetchdf()
        assert len(folds) > 0
    pipeline.encode(**settings)
    with duckdb.connect(str(settings["file"]), read_only=True) as connection:
        pd.testing.assert_frame_equal(
            original,
            connection.execute(
                "SELECT * FROM referenced_pattern_expression ORDER BY epoch_index,component,time"
            ).fetchdf(),
        )


def test_result_files_without_automatic_references_must_be_recomputed(tmp_path):
    _, pipeline, settings = _pipeline_case(tmp_path)
    pipeline.encode(**settings)
    # Simulate a file written before automatic references existed.
    with duckdb.connect(str(settings["file"])) as connection:
        config = json.loads(connection.execute("SELECT config FROM analysis").fetchone()[0])
        config.pop("expression_reference")
        connection.execute(
            "UPDATE analysis SET config = ?, fingerprint = ?",
            [json.dumps(config), settings_fingerprint(config)],
        )
    with pytest.raises(ValueError, match="incompatible"):
        pipeline.encode(**settings)
    pipeline.encode(**settings, recompute="all")


def test_subjects_with_different_references_are_rejected(tmp_path, monkeypatch):
    _, pipeline, settings = _pipeline_case(tmp_path, ("001", "002"))
    original = analysis.zero_code_references
    calls = []

    def second_subject_without_b(design, labels):
        calls.append(1)
        plans = original(design, labels)
        return plans if len(calls) == 1 else tuple(p for p in plans if p.component != "b")

    monkeypatch.setattr(analysis, "zero_code_references", second_subject_without_b)
    with pytest.raises(RuntimeError, match="zero-code references differ between subjects"):
        pipeline.encode(**settings)


def test_reference_transaction_rollback_and_subject_exclusions(tmp_path, monkeypatch):
    from mveeg.encoding import _storage

    dataset, pipeline, settings = _pipeline_case(tmp_path, ("001", "002"))
    original = _storage.insert_frame

    def fail_reference(connection, name, frame):
        if name == "expression_reference_trials":
            raise RuntimeError("simulated reference write failure")
        original(connection, name, frame)

    monkeypatch.setattr(_storage, "insert_frame", fail_reference)
    with pytest.raises(RuntimeError, match="simulated reference write failure"):
        pipeline.encode(**settings)
    with duckdb.connect(str(settings["file"]), read_only=True) as connection:
        assert connection.execute("SELECT status FROM subjects").fetchone()[0] == "failed"
        assert not connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name='pattern_expression'"
        ).fetchall()
    monkeypatch.setattr(_storage, "insert_frame", original)
    pipeline.encode(**settings)
    open_pipeline(dataset).select_subjects(exclude={"001": "test exclusion"})
    pipeline.encode(**{**settings, "recompute": "changed"})
    with duckdb.connect(str(settings["file"]), read_only=True) as connection:
        for table in (
            "subjects",
            "pattern_expression",
            "referenced_pattern_expression",
            "expression_references",
            "expression_reference_trials",
        ):
            assert connection.execute(f"SELECT DISTINCT subject_index FROM {table}").fetchall() == [
                ("002",)
            ]
