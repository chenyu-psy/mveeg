"""Observed-code reference behavior through the actual encoding workflow."""

import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from mveeg.encoding._analysis import encode_subject
from mveeg.encoding._prepare import build_design
from mveeg.encoding._reference import build_expression_references
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


def test_omitted_empty_and_partial_references_preserve_all_legacy_results():
    original = _run()
    empty = _run(expression_reference={})
    partial = _run(expression_reference={"a": 0})
    for name in original.tables:
        pd.testing.assert_frame_equal(original.tables[name], empty.tables[name])
        pd.testing.assert_frame_equal(original.tables[name], partial.tables[name])
    assert set(partial.tables["referenced_pattern_expression"].component) == {"a"}
    assert len(partial.tables["referenced_pattern_expression"]) * 2 == len(
        original.tables["pattern_expression"]
    )


def test_multiple_components_use_different_observed_numeric_codes():
    result = _run(expression_reference={"a": 0, "b": 1})
    audit = result.tables["expression_reference_trials"]
    metadata = _settings()["metadata"]
    for component, expected in (("a", 0), ("b", 1)):
        rows = audit.query("component == @component")
        assert metadata.iloc[rows.epoch_index][component].eq(expected).all()
    offset = result.tables["expression_references"]
    assert set(zip(offset.component, offset.reference_value)) == {("a", 0.0), ("b", 1.0)}


def test_reference_plans_and_auxiliary_projections_are_identical_in_parallel():
    serial = _run(expression_reference={"a": 0, "b": 1})
    parallel = _run(expression_reference={"a": 0, "b": 1}, n_jobs=2)
    for name in serial.tables:
        pd.testing.assert_frame_equal(serial.tables[name], parallel.tables[name])


def test_reference_pools_trials_across_unequal_conditions_and_display_regrouping():
    settings = _settings()
    result = encode_subject(**settings, expression_reference={"a": 0})
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
    other = encode_subject(
        **{**settings, "expression_labels": regrouped}, expression_reference={"a": 0}
    )
    pd.testing.assert_frame_equal(audit, other.tables["expression_reference_trials"])
    pd.testing.assert_frame_equal(
        result.tables["expression_references"], other.tables["expression_references"]
    )


def test_auxiliary_reference_projection_never_duplicates_public_trials_or_uses_training_rows():
    settings = _settings()
    result = encode_subject(**settings, expression_reference={"a": 0})
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
    result = _run(expression_reference={"a": 0})
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
    first = encode_subject(**settings, expression_reference={"a": 0})
    public = first.tables["pattern_expression"]
    fold = int(public.loc[public.epoch_index == 0, "fold"].iloc[0])
    changed = settings["data"].copy()
    changed[0] += np.linspace(-20, 20, changed.shape[1])[:, None] * 1e-6
    second = encode_subject(**{**settings, "data": changed}, expression_reference={"a": 0})
    for table in ("pattern_expression", "referenced_pattern_expression"):
        old = first.tables[table]
        new = second.tables[table]
        keep = (old.fold == fold) & (old.epoch_index != 0)
        pd.testing.assert_frame_equal(
            old.loc[keep].reset_index(drop=True), new.loc[keep].reset_index(drop=True)
        )
    pd.testing.assert_frame_equal(
        first.tables["expression_references"].query("fold == @fold").reset_index(drop=True),
        second.tables["expression_references"].query("fold == @fold").reset_index(drop=True),
    )
    # Changing never-trained reference EEG legitimately changes the observed
    # offset, but must not change any fitted coefficient or covariance estimate.
    changed = settings["data"].copy()
    probe = int(np.flatnonzero(settings["expression_labels"] == "P")[0])
    changed[probe] += np.linspace(-10, 10, changed.shape[1])[:, None] * 1e-6
    auxiliary = encode_subject(**{**settings, "data": changed}, expression_reference={"a": 0})
    for table in ("coefficients", "covariance_diagnostics"):
        pd.testing.assert_frame_equal(first.tables[table], auxiliary.tables[table])
    assert not np.allclose(
        first.tables["expression_references"].reference_expression,
        auxiliary.tables["expression_references"].reference_expression,
    )


def test_membership_uses_expanded_interactions_and_trial_varying_codes():
    metadata = pd.DataFrame({"a": [-1.0, 0, 1, 2], "b": [0.0, 2, 0, 3]})
    labels = np.array(["one"] * 4, object)
    design = build_design(
        metadata,
        formula="a*b",
        training_labels=labels,
        condition_order=["one"],
        penalty={"component": 1.0, "condition": 0.1},
    )
    plans = build_expression_references(design, {"a": 2.0, "a:b": 0.0})
    assert plans[0].rows.tolist() == [3]
    assert plans[1].rows.tolist() == [0, 1, 2]
    # A numerical code can be observed without any condition-constant design constraint.
    assert design.matrix[3, design.component_indices[0]] == 2
    varied = _settings()["metadata"].copy()
    varied.loc[varied.condition == "A", "a"] = np.linspace(0.2, 1.4, 8)
    legacy = _run(metadata=varied, formula="a*b")
    result = _run(metadata=varied, formula="a*b", expression_reference={"a:b": 0})
    for name in legacy.tables:
        pd.testing.assert_frame_equal(legacy.tables[name], result.tables[name])
    audit = result.tables["expression_reference_trials"]
    metadata = varied.iloc[audit.epoch_index]
    assert (metadata.a * metadata.b).eq(0).all()
    assert set(result.tables["referenced_pattern_expression"].component) == {"a:b"}


def test_contrast_coding_nonzero_reference_and_exact_floating_match_policy():
    settings = _settings()
    metadata = settings["metadata"].copy()
    metadata.a = metadata.a * 2 - 1
    result = encode_subject(**{**settings, "metadata": metadata}, expression_reference={"a": -1})
    assert result.tables["expression_references"].reference_value.eq(-1).all()
    with pytest.raises(ValueError, match="exact code"):
        encode_subject(**{**settings, "metadata": metadata}, expression_reference={"a": 0})
    design = build_design(
        pd.DataFrame({"a": [0.3 - 0.2, 0.2]}),
        formula="a",
        training_labels=np.array(["A", "B"], object),
        condition_order=["A", "B"],
        penalty={"component": 1.0, "condition": 0.1},
    )
    with pytest.raises(ValueError, match="exact code"):
        build_expression_references(design, {"a": 0.1})


def test_no_heldout_reference_and_near_null_pattern_fail_without_fallback():
    settings = _settings()
    # This observed code occurs in one training trial, so cannot be held out
    # from every fold model. It must not be reused in the models that trained it.
    rare_code = settings["metadata"].copy()
    rare_code.loc[0, "a"] = 2
    with pytest.raises(ValueError, match="no matching reference trials are held out"):
        encode_subject(**{**settings, "metadata": rare_code}, expression_reference={"a": 2})
    zero_component = settings["metadata"].copy()
    zero_component.a = 0
    with pytest.raises(ValueError, match="nonfinite"):
        encode_subject(**{**settings, "metadata": zero_component}, expression_reference={"a": 0})


def _pipeline_case(tmp_path, subjects=("001",)):
    dataset = _build_dataset(tmp_path / "dataset", list(subjects))
    pipeline = _pipeline(dataset)
    settings = dict(
        formula="a+b",
        target="condition",
        conditions={"A": ["A"], "B": ["B"], "AB": ["AB"]},
        expression={"A": ["A"], "B": ["B"], "AB": ["AB"], "probe": ["P"]},
        expression_reference={"a": 0},
        file=tmp_path / "reference.duckdb",
        progress=False,
    )
    return dataset, pipeline, settings


@pytest.mark.parametrize("reference", [{"a": "B"}, {"a": np.nan}, {"a": np.inf}, {"unknown": 0}])
def test_pipeline_rejects_non_numeric_and_unknown_component_reference(tmp_path, reference):
    _, pipeline, settings = _pipeline_case(tmp_path)
    with pytest.raises((ValueError, RuntimeError), match="numeric|unknown"):
        pipeline.encode(**{**settings, "expression_reference": reference})


def test_persistence_offsets_trial_weighted_views_resume_and_config_change(tmp_path):
    _, pipeline, settings = _pipeline_case(tmp_path)
    pipeline.encode(**settings)
    with duckdb.connect(str(settings["file"]), read_only=True) as connection:
        config = json.loads(connection.execute("SELECT config FROM analysis").fetchone()[0])
        assert config["expression_reference"] == {"a": 0.0}
        assert (
            config["expression_reference_estimator"]["weighting"]
            == "equal_eligible_matching_trials"
        )
        original = connection.execute(
            "SELECT * FROM referenced_pattern_expression ORDER BY epoch_index,time"
        ).fetchdf()
        offset = connection.execute("SELECT * FROM expression_references").fetchdf()
        assert len(offset) > 0
        observed = connection.execute(
            "SELECT * FROM condition_referenced_pattern_expression ORDER BY condition,time"
        ).fetchdf()
        assert "expression_se" not in observed and "se" not in observed
        direct = connection.execute(
            "SELECT expression_group AS condition,time,AVG(expression) AS mean,"
            "COUNT(DISTINCT epoch_index) AS n_trials FROM referenced_pattern_expression "
            "GROUP BY expression_group,time ORDER BY condition,time"
        ).fetchdf()
        pd.testing.assert_frame_equal(observed[["condition", "time", "mean", "n_trials"]], direct)
        folds = connection.execute(
            "SELECT * FROM fold_condition_referenced_pattern_expression"
        ).fetchdf()
        assert len(folds) > 0
    pipeline.encode(**settings)
    with duckdb.connect(str(settings["file"]), read_only=True) as connection:
        pd.testing.assert_frame_equal(
            original,
            connection.execute(
                "SELECT * FROM referenced_pattern_expression ORDER BY epoch_index,time"
            ).fetchdf(),
        )
    with pytest.raises(ValueError, match="incompatible"):
        pipeline.encode(**{**settings, "expression_reference": {"b": 0}})
    pipeline.encode(**{**settings, "expression_reference": {"b": 0}, "recompute": "all"})
    with duckdb.connect(str(settings["file"]), read_only=True) as connection:
        assert connection.execute(
            "SELECT DISTINCT component FROM referenced_pattern_expression"
        ).fetchall() == [("b",)]


def test_empty_reference_preserves_existing_default_fingerprint_and_tables(tmp_path):
    _, pipeline, settings = _pipeline_case(tmp_path)
    settings.pop("expression_reference")
    pipeline.encode(**settings)
    with duckdb.connect(str(settings["file"]), read_only=True) as connection:
        original = connection.execute("SELECT fingerprint,config FROM analysis").fetchall()
        tables = connection.execute(
            "SELECT table_name FROM information_schema.tables ORDER BY table_name"
        ).fetchall()
    pipeline.encode(**settings, expression_reference={})
    with duckdb.connect(str(settings["file"]), read_only=True) as connection:
        assert connection.execute("SELECT fingerprint,config FROM analysis").fetchall() == original
        assert (
            connection.execute(
                "SELECT table_name FROM information_schema.tables ORDER BY table_name"
            ).fetchall()
            == tables
        )
        assert "expression_reference" not in json.loads(original[0][1])


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
