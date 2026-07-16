"""Public decoding API, scientific execution, and DuckDB contracts."""

from __future__ import annotations

import json

import duckdb
import mne
import numpy as np
import pandas as pd
import pytest

import mveeg.decoding._analysis as decoding_analysis
from mveeg._dataset.store import DatasetBuilder
from mveeg.decoding import DecodingPipeline, init_pipeline
from mveeg.decoding._analysis import decode_subject
from mveeg.decoding._prepare import (
    average_training_trials,
    sample_balanced,
    select_trials,
    validate_generalization,
)
from mveeg.prep import open_pipeline


def _subject_data(classes: int = 3):
    rng = np.random.default_rng(12)
    labels = np.repeat([chr(97 + index) for index in range(classes)], 4).astype(object)
    data = rng.normal(size=(len(labels) + 2, max(classes, 2), 2))
    for index, label in enumerate(labels):
        data[index, ord(label) - 97, :] += 2
    class_labels = np.append(labels, [None, None])
    return data, class_labels, np.arange(len(class_labels))


def _trial_roles(labels: np.ndarray, *, generalization: bool) -> dict[str, np.ndarray]:
    training = pd.notna(labels)
    evidence = labels.copy()
    evidence[~training] = ["evidence", None]
    conditions = labels.copy()
    conditions[~training] = ["evidence", "generalization"]
    generalization_labels = np.full(len(labels), None, dtype=object)
    if generalization:
        generalization_labels[training] = labels[training]
        generalization_labels[-1] = labels[training][0]
    return {
        "evidence_labels": evidence,
        "generalization_labels": generalization_labels,
        "condition_values": conditions,
    }


def test_multiclass_all_output_preserves_native_geometry_and_cv_rows():
    data, labels, trials = _subject_data()
    result = decode_subject(
        subject="001",
        data=data,
        class_labels=labels,
        **_trial_roles(labels, generalization=True),
        trial_ids=trials,
        times=np.array([25, 75]),
        class_order=["a", "b", "c"],
        classifier="logistic_regression",
        classifier_parameters={"solver": "lbfgs", "max_iter": 1000},
        folds=2,
        repeats=1,
        trial_averaging=1,
        permutations=1,
        seed=8,
        output="all",
    )

    assert result.classifier["classes"] == ["a", "b", "c"]
    assert result.classifier["evidence_shape"] == [3]
    assert result.pattern_components == [["a"], ["b"], ["c"]]
    assert all(len(value) == 3 for value in result.tables["classifier_evidence"]["evidence"])
    evidence_only = result.tables["classifier_evidence"].query("epoch_index == 12")
    assert len(evidence_only) == 2 * 2  # folds x time bins
    class_trial = result.tables["classifier_evidence"].query("epoch_index == 0")
    assert len(class_trial) == 2  # one held-out model x time bins
    assert result.tables["confusion_matrix"]["count"].sum() == 12 * 2
    assert set(result.tables["accuracy"]["permutation"]) == {0, 1}
    assert set(result.tables["generalization"]["permutation"]) == {0, 1}
    assert set(result.tables["generalization"]["condition"]) == {"a", "b", "c", "generalization"}
    assert set(
        result.tables["generalization"].query("condition == 'generalization'")["n_trials"]
    ) == {1}
    assert set(result.tables["generalization"].query("condition == 'a'")["n_trials"]) == {2}
    assert result.tables["classifier_evidence"].query("epoch_index == 13").empty
    assert "permutation" not in result.tables["patterns"]


def test_binary_mean_output_keeps_scalar_evidence_and_weighted_accuracy():
    data, labels, trials = _subject_data(classes=2)
    result = decode_subject(
        subject="001",
        data=data,
        class_labels=labels,
        **_trial_roles(labels, generalization=False),
        trial_ids=trials,
        times=np.array([25, 75]),
        class_order=["a", "b"],
        classifier="logistic_regression",
        classifier_parameters={"solver": "lbfgs", "max_iter": 1000},
        folds=3,
        repeats=2,
        trial_averaging=1,
        permutations=0,
        seed=9,
        output="mean",
    )

    evidence = result.tables["classifier_evidence"]
    assert result.classifier["evidence_shape"] == []
    assert evidence["evidence"].map(np.isscalar).all()
    assert set(evidence["n_models"]) == {2, 6}
    accuracy = result.tables["accuracy"]
    assert accuracy.columns.tolist() == [
        "subject_index",
        "time",
        "permutation",
        "accuracy",
        "n_correct",
        "n_trials",
    ]
    assert accuracy["n_trials"].tolist() == [16, 16]
    assert result.tables["confusion_matrix"]["count"].sum() == 16 * 2
    assert "generalization" not in result.tables


@pytest.mark.parametrize("output", ["mean", "all"])
def test_repeat_parallelism_preserves_all_results(output):
    data, labels, trials = _subject_data()
    settings = {
        "subject": "001",
        "data": data,
        "class_labels": labels,
        **_trial_roles(labels, generalization=True),
        "trial_ids": trials,
        "times": np.array([25, 75]),
        "class_order": ["a", "b", "c"],
        "classifier": "logistic_regression",
        "classifier_parameters": {"solver": "lbfgs", "max_iter": 1000},
        "folds": 2,
        "repeats": 2,
        "trial_averaging": 1,
        "permutations": 1,
        "seed": 13,
        "output": output,
        "progress": False,
    }

    serial = decode_subject(**settings, n_jobs=1)
    parallel = decode_subject(**settings, n_jobs=2)

    assert serial.classifier == parallel.classifier
    assert serial.pattern_components == parallel.pattern_components
    assert serial.tables.keys() == parallel.tables.keys()
    for name in serial.tables:
        pd.testing.assert_frame_equal(serial.tables[name], parallel.tables[name])


def test_incremental_mean_matches_fold_level_results():
    data, labels, trials = _subject_data()
    settings = {
        "subject": "001",
        "data": data,
        "class_labels": labels,
        **_trial_roles(labels, generalization=True),
        "trial_ids": trials,
        "times": np.array([25, 75]),
        "class_order": ["a", "b", "c"],
        "classifier": "logistic_regression",
        "classifier_parameters": {"solver": "lbfgs", "max_iter": 1000},
        "folds": 2,
        "repeats": 2,
        "trial_averaging": 1,
        "permutations": 1,
        "seed": 15,
        "n_jobs": 1,
        "progress": False,
    }
    detailed = decode_subject(**settings, output="all").tables
    mean = decode_subject(**settings, output="mean").tables

    repeat_accuracy = (
        detailed["accuracy"]
        .groupby(["subject_index", "repeat", "time", "permutation"], as_index=False)[
            ["n_correct", "n_trials"]
        ]
        .sum()
    )
    repeat_accuracy["accuracy"] = repeat_accuracy["n_correct"] / repeat_accuracy["n_trials"]
    expected_accuracy = repeat_accuracy.groupby(
        ["subject_index", "time", "permutation"], as_index=False
    )["accuracy"].mean()
    np.testing.assert_allclose(mean["accuracy"]["accuracy"], expected_accuracy["accuracy"])

    detailed_evidence = detailed["classifier_evidence"].query("epoch_index == 12 and time == 25")
    expected_evidence = np.mean(np.stack(detailed_evidence["evidence"].map(np.asarray)), axis=0)
    observed_evidence = np.asarray(
        mean["classifier_evidence"].query("epoch_index == 12 and time == 25")["evidence"].iloc[0]
    )
    np.testing.assert_allclose(observed_evidence, expected_evidence)
    assert (
        mean["classifier_evidence"].query("epoch_index == 12 and time == 25")["n_models"].iloc[0]
        == 4
    )

    expected_confusion = (
        detailed["confusion_matrix"]
        .groupby(["subject_index", "time", "actual", "predicted"], as_index=False)["count"]
        .sum()
    )
    pd.testing.assert_frame_equal(mean["confusion_matrix"], expected_confusion)

    expected_patterns = (
        detailed["patterns"]
        .groupby(["subject_index", "time", "channel_index", "component"], as_index=False)["pattern"]
        .mean()
    )
    pd.testing.assert_frame_equal(mean["patterns"], expected_patterns)

    expected_generalization = (
        detailed["generalization"]
        .groupby(
            ["subject_index", "condition", "train_time", "test_time", "permutation"],
            as_index=False,
        )[["n_correct", "n_trials"]]
        .sum()
    )
    expected_generalization["accuracy"] = (
        expected_generalization["n_correct"] / expected_generalization["n_trials"]
    )
    expected_generalization = expected_generalization[
        [
            "subject_index",
            "condition",
            "train_time",
            "test_time",
            "permutation",
            "accuracy",
            "n_correct",
            "n_trials",
        ]
    ]
    pd.testing.assert_frame_equal(mean["generalization"], expected_generalization)


def test_subject_progress_uses_standard_tqdm_arguments(monkeypatch):
    calls = []

    def capture_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return iterable

    monkeypatch.setattr("mveeg.decoding._analysis.tqdm", capture_tqdm)
    data, labels, trials = _subject_data(classes=2)
    decode_subject(
        subject="001",
        data=data,
        class_labels=labels,
        **_trial_roles(labels, generalization=False),
        trial_ids=trials,
        times=np.array([25, 75]),
        class_order=["a", "b"],
        classifier="logistic_regression",
        classifier_parameters={"solver": "lbfgs", "max_iter": 1000},
        folds=2,
        repeats=2,
        trial_averaging=1,
        permutations=0,
        seed=14,
        output="mean",
        n_jobs=1,
        progress=False,
    )

    assert calls == [
        {
            "total": 2,
            "desc": "Decoding sub-001",
            "unit": "repeat",
            "disable": True,
        }
    ]


def test_multiclass_linear_svm_patterns_follow_native_pairs():
    data, labels, trials = _subject_data()
    result = decode_subject(
        subject="001",
        data=data,
        class_labels=labels,
        **_trial_roles(labels, generalization=False),
        trial_ids=trials,
        times=np.array([25, 75]),
        class_order=["a", "b", "c"],
        classifier="linear_svm",
        classifier_parameters={"kernel": "linear", "probability": False},
        folds=2,
        repeats=1,
        trial_averaging=1,
        permutations=0,
        seed=10,
        output="mean",
    )

    assert result.classifier["evidence_shape"] == [3]
    assert result.pattern_components == [["a", "b"], ["a", "c"], ["b", "c"]]
    assert sorted(result.tables["patterns"]["component"].unique()) == [0, 1, 2]


def test_multiclass_lda_uses_class_components():
    data, labels, trials = _subject_data()
    result = decode_subject(
        subject="001",
        data=data,
        class_labels=labels,
        **_trial_roles(labels, generalization=False),
        trial_ids=trials,
        times=np.array([25, 75]),
        class_order=["a", "b", "c"],
        classifier="lda",
        classifier_parameters={},
        folds=2,
        repeats=1,
        trial_averaging=1,
        permutations=0,
        seed=11,
        output="mean",
    )

    assert result.classifier["evidence_shape"] == [3]
    assert result.pattern_components == [["a"], ["b"], ["c"]]


@pytest.mark.parametrize(
    ("classifier", "parameters"),
    [
        ("logistic_regression", {"solver": "lbfgs", "max_iter": 1000}),
        ("lda", {}),
        ("linear_svm", {"kernel": "linear", "probability": False}),
    ],
)
@pytest.mark.parametrize(("n_classes", "shape"), [(2, []), (3, [3])])
def test_builtin_classifiers_keep_native_binary_and_multiclass_evidence(
    classifier, parameters, n_classes, shape
):
    data, labels, trials = _subject_data(classes=n_classes)
    class_order = [chr(97 + index) for index in range(n_classes)]
    result = decode_subject(
        subject="001",
        data=data,
        class_labels=labels,
        **_trial_roles(labels, generalization=False),
        trial_ids=trials,
        times=np.array([25, 75]),
        class_order=class_order,
        classifier=classifier,
        classifier_parameters=parameters,
        folds=2,
        repeats=1,
        trial_averaging=1,
        permutations=0,
        seed=12,
        output="mean",
    )

    assert result.classifier["evidence_shape"] == shape
    evidence = result.tables["classifier_evidence"]["evidence"]
    assert bool(evidence.map(np.isscalar).all()) == (shape == [])


def test_balancing_and_training_averages_drop_only_incomplete_training_groups():
    labels = np.array(["a"] * 6 + ["b"] * 4, dtype=object)
    balanced = sample_balanced(labels, ["a", "b"], np.random.default_rng(1))
    assert np.unique(labels[balanced], return_counts=True)[1].tolist() == [4, 4]

    data = np.arange(10 * 2 * 2, dtype=float).reshape(10, 2, 2)
    averaged, averaged_labels = average_training_trials(
        data,
        labels,
        class_order=["a", "b"],
        size=3,
        rng=np.random.default_rng(2),
    )
    assert averaged.shape == (2, 2, 2)
    assert sorted(averaged_labels.tolist()) == ["a", "b"]


def test_trial_selection_drops_only_a_redundant_subject_alias():
    metadata = pd.DataFrame(
        {
            "subject_index": ["001", "001"],
            "subject": [1, 1],
            "epoch_index": [0, 1],
            "condition": ["a", "b"],
            "final_status": ["accepted", "accepted"],
        }
    )
    selected, *_ = select_trials(
        metadata,
        target="condition",
        classes={"a": ["a"], "b": ["b"]},
        evidence={"a": ["a"], "b": ["b"]},
        generalization=None,
        qc="final_status",
        keep=["accepted"],
        exclude={},
    )

    assert "subject" not in selected.columns

    metadata.loc[1, "subject"] = 2
    selected, *_ = select_trials(
        metadata,
        target="condition",
        classes={"a": ["a"], "b": ["b"]},
        evidence={"a": ["a"], "b": ["b"]},
        generalization=None,
        qc="final_status",
        keep=["accepted"],
        exclude={},
    )
    assert selected["subject"].tolist() == [1, 2]


def test_generalization_mapping_selects_conditions_independently():
    classes = {"SS2": ["2C2N"], "SS4": ["4C2N"]}
    generalization = validate_generalization(
        classes,
        {"SS2": ["4C2N"], "SS4": ["2C2N", "2C4N"]},
    )
    metadata = pd.DataFrame(
        {
            "subject_index": ["001"] * 3,
            "epoch_index": [0, 1, 2],
            "condition": ["2C2N", "4C2N", "2C4N"],
            "final_status": ["accepted"] * 3,
        }
    )

    selected, class_labels, evidence_labels, generalization_labels, rows = select_trials(
        metadata,
        target="condition",
        classes=classes,
        evidence={
            "2C2N": ["2C2N"],
            "4C2N": ["4C2N"],
            "2C4N": ["2C4N"],
        },
        generalization=generalization,
        qc="final_status",
        keep=["accepted"],
        exclude={},
    )

    assert selected["condition"].tolist() == ["2C2N", "4C2N", "2C4N"]
    assert class_labels.tolist() == ["SS2", "SS4", None]
    assert evidence_labels.tolist() == ["2C2N", "4C2N", "2C4N"]
    assert generalization_labels.tolist() == ["SS4", "SS2", "SS4"]
    assert rows.tolist() == [0, 1, 2]

    with pytest.raises(ValueError, match="unknown"):
        validate_generalization(classes, {"missing": ["2C4N"]})
    with pytest.raises(ValueError, match="duplicated"):
        validate_generalization(classes, {"SS2": ["x"], "SS4": ["x"]})
    with pytest.raises(ValueError, match="duplicated"):
        validate_generalization(classes, {"SS2": ["x", "x"]})
    with pytest.raises(ValueError, match="at least 1"):
        validate_generalization(classes, True)


def test_permutations_keep_generalization_targets_fixed(monkeypatch):
    data, labels, trials = _subject_data(classes=2)
    training = pd.notna(labels)
    generalization_labels = np.full(len(labels), None, dtype=object)
    generalization_labels[training] = np.where(
        labels[training] == "a",
        "b",
        "a",
    )
    captured = []

    def capture_generalization(output, *, permutation, actual_labels, **kwargs):
        captured.append((permutation, actual_labels.copy()))

    monkeypatch.setattr(
        decoding_analysis,
        "_append_generalization",
        capture_generalization,
    )
    decoding_analysis._decode_repeat(
        repeat=1,
        subject="001",
        data=data,
        labels=labels,
        evidence_mask=training,
        generalization_labels=generalization_labels,
        generalization_mask=training,
        condition_values=np.where(labels == "a", "condition_a", "condition_b"),
        generalization_conditions=["condition_a", "condition_b"],
        training_mask=training,
        training_rows=np.flatnonzero(training),
        training_labels=labels[training],
        trial_ids=trials,
        times=np.array([25, 75]),
        class_order=["a", "b"],
        classifier="logistic_regression",
        classifier_parameters={"solver": "lbfgs", "max_iter": 1000},
        folds=2,
        trial_averaging=1,
        permutations=1,
        subject_seed=17,
    )

    assert {permutation for permutation, _ in captured} == {0, 1}
    for _, actual_labels in captured:
        np.testing.assert_array_equal(actual_labels, generalization_labels)


def test_pipeline_init_is_lazy_chained_and_does_not_accept_prep_handles(
    tmp_path,
    monkeypatch,
):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])

    def fail_read(*args, **kwargs):
        raise AssertionError("init_pipeline must not load epochs")

    monkeypatch.setattr(mne, "read_epochs", fail_read)
    pipeline = init_pipeline(dataset)

    assert isinstance(pipeline, DecodingPipeline)
    assert pipeline.subject_indices == ("001",)
    assert (
        pipeline.transform_metadata(condition_upper=lambda frame: frame["condition"].str.upper())
        is pipeline
    )
    assert pipeline.select_trials(qc=None) is pipeline
    assert pipeline.prepare_epochs(crop=None) is pipeline
    assert pipeline.setup_classifier() is pipeline
    assert pipeline.setup_cv() is pipeline
    with pytest.raises(TypeError):
        init_pipeline(open_pipeline(dataset))


def test_pipeline_transform_metadata_builds_ordered_trial_variables(tmp_path):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])
    result_file = tmp_path / "decoding.duckdb"
    pipeline = init_pipeline(dataset)
    pipeline.transform_metadata(
        condition_upper=lambda frame: frame["condition"].str.upper(),
        is_a=lambda frame: frame["condition_upper"].eq("A"),
    )
    pipeline.prepare_epochs(crop=None, time_bin=50)
    pipeline.setup_cv(folds=2, repeats=1, trial_averaging=1, seed=4)
    pipeline.decode(
        target="condition_upper",
        classes={"A": ["A"], "B": ["B"], "C": ["C"]},
        file=result_file,
        progress=False,
    )

    with duckdb.connect(str(result_file), read_only=True) as connection:
        trials = connection.execute(
            "SELECT condition, condition_upper, is_a FROM trials ORDER BY epoch_index"
        ).df()
        config = json.loads(
            connection.execute("SELECT config::VARCHAR FROM analysis").fetchone()[0]
        )

    assert trials["condition_upper"].tolist() == [value.upper() for value in trials["condition"]]
    assert trials["is_a"].tolist() == trials["condition_upper"].eq("A").tolist()
    assert config["metadata_variables"] == ["condition_upper", "is_a"]


def test_pipeline_metadata_variable_values_enter_subject_fingerprint(tmp_path):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])
    result_file = tmp_path / "decoding.duckdb"

    first = init_pipeline(dataset).transform_metadata(
        model_condition=lambda frame: frame["condition"]
    )
    first.prepare_epochs(crop=None, time_bin=50)
    first.setup_cv(folds=2, repeats=1, trial_averaging=1, seed=4)
    first.decode(
        target="model_condition",
        classes={"a": ["a"], "b": ["b"], "c": ["c"]},
        file=result_file,
        progress=False,
    )
    with duckdb.connect(str(result_file), read_only=True) as connection:
        first_fingerprint = connection.execute(
            "SELECT fingerprint FROM subjects WHERE subject_index='001'"
        ).fetchone()[0]

    changed = init_pipeline(dataset).transform_metadata(
        model_condition=lambda frame: frame["condition"].replace({"a": "b", "b": "a"})
    )
    changed.prepare_epochs(crop=None, time_bin=50)
    changed.setup_cv(folds=2, repeats=1, trial_averaging=1, seed=4)
    changed.decode(
        target="model_condition",
        classes={"a": ["a"], "b": ["b"], "c": ["c"]},
        file=result_file,
        recompute="changed",
        progress=False,
    )
    with duckdb.connect(str(result_file), read_only=True) as connection:
        second_fingerprint = connection.execute(
            "SELECT fingerprint FROM subjects WHERE subject_index='001'"
        ).fetchone()[0]

    assert second_fingerprint != first_fingerprint


def test_pipeline_transform_metadata_rejects_invalid_variable_definitions(tmp_path):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])
    pipeline = init_pipeline(dataset)

    with pytest.raises(TypeError, match="must be defined by a callable"):
        pipeline.transform_metadata(load=1)
    with pytest.raises(ValueError, match="cannot replace subject_index"):
        pipeline.transform_metadata(subject_index=lambda frame: frame["subject_index"])


def test_pipeline_init_rejects_inconsistent_dataset_identity(tmp_path):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])
    path = dataset / "provenance.json"
    provenance = json.loads(path.read_text())
    provenance["task"] = "different"
    path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="disagree on task or processing stage"):
        init_pipeline(dataset)


def test_pipeline_writes_public_duckdb_and_reuses_completed_subjects(tmp_path):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])
    result_file = tmp_path / "decoding.duckdb"
    pipeline = init_pipeline(dataset)
    pipeline.prepare_epochs(crop=None, time_bin=50)
    pipeline.setup_cv(folds=2, repeats=1, trial_averaging=1, permutations=1, seed=4)
    assert (
        pipeline.decode(
            target="condition",
            classes={"a": ["a"], "b": ["b"], "c": ["c"]},
            evidence={"a": ["a"], "b": ["b"], "c": ["c"]},
            generalization={"a": ["b", "x"], "b": ["a"], "c": ["c"]},
            output="mean",
            file=result_file,
        )
        is None
    )

    with duckdb.connect(str(result_file), read_only=True) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
        assert tables == {
            "analysis",
            "subjects",
            "trials",
            "classifier",
            "pattern_components",
            "channels",
            "time_bins",
            "accuracy",
            "classifier_evidence",
            "confusion_matrix",
            "patterns",
            "generalization",
        }
        assert connection.execute("DESCRIBE time_bins").df()["column_name"].tolist() == [
            "time",
            "start",
            "end",
        ]
        assert connection.execute("SELECT status FROM subjects").fetchone()[0] == "complete"
        evidence_schema = connection.execute("DESCRIBE classifier_evidence").df()
        assert (
            evidence_schema.loc[evidence_schema["column_name"].eq("evidence"), "column_type"].item()
            == "DOUBLE[]"
        )
        assert (
            connection.execute(
                "SELECT column_type FROM (DESCRIBE analysis) WHERE column_name='generalization'"
            ).fetchone()[0]
            == "JSON"
        )
        assert connection.execute(
            "SELECT DISTINCT condition FROM generalization ORDER BY condition"
        ).fetchall() == [("a",), ("b",), ("c",), ("x",)]
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM classifier_evidence e "
                "JOIN trials t USING (subject_index, epoch_index) WHERE t.condition='x'"
            ).fetchone()[0]
            == 0
        )
        assert connection.execute(
            "SELECT DISTINCT evidence_group FROM trials WHERE condition='x'"
        ).fetchall() == [(None,)]
        assert (
            connection.execute(
                "SELECT n_trials FROM generalization "
                "WHERE condition='x' AND permutation=0 ORDER BY train_time, test_time LIMIT 1"
            ).fetchone()[0]
            == 8
        )
        first_accuracy = connection.execute(
            "SELECT * FROM accuracy WHERE subject_index='001' ORDER BY time, permutation"
        ).df()

    _extend_dataset(dataset, "002")
    pipeline.decode(
        target="condition",
        classes={"a": ["a"], "b": ["b"], "c": ["c"]},
        evidence={"a": ["a"], "b": ["b"], "c": ["c"]},
        generalization={"a": ["b", "x"], "b": ["a"], "c": ["c"]},
        output="mean",
        file=result_file,
        recompute="never",
    )

    with duckdb.connect(str(result_file), read_only=True) as connection:
        assert connection.execute(
            "SELECT subject_index, status FROM subjects ORDER BY subject_index"
        ).fetchall() == [("001", "complete"), ("002", "complete")]
        saved_accuracy = connection.execute(
            "SELECT * FROM accuracy WHERE subject_index='001' ORDER BY time, permutation"
        ).df()
    pd.testing.assert_frame_equal(first_accuracy, saved_accuracy)


def test_pipeline_shows_one_progress_bar_per_computed_subject(tmp_path, monkeypatch):
    calls = []

    def capture_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return iterable

    monkeypatch.setattr("mveeg.decoding._analysis.tqdm", capture_tqdm)
    dataset = _build_dataset(tmp_path / "dataset", ["001", "002"])
    result_file = tmp_path / "decoding.duckdb"
    pipeline = init_pipeline(dataset).prepare_epochs(crop=None, time_bin=50)
    pipeline.setup_cv(folds=2, repeats=2, trial_averaging=1, seed=4)
    pipeline.decode(
        target="condition",
        classes={"a": ["a"], "b": ["b"], "c": ["c"]},
        file=result_file,
        progress=False,
    )

    assert calls == [
        {
            "total": 2,
            "desc": "Decoding sub-001",
            "unit": "repeat",
            "disable": True,
        },
        {
            "total": 2,
            "desc": "Decoding sub-002",
            "unit": "repeat",
            "disable": True,
        },
    ]

    calls.clear()
    pipeline.decode(
        target="condition",
        classes={"a": ["a"], "b": ["b"], "c": ["c"]},
        file=result_file,
        progress=True,
    )
    assert calls == []


def test_pipeline_rejects_config_change_without_full_recompute(tmp_path):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])
    result_file = tmp_path / "decoding.duckdb"
    pipeline = init_pipeline(dataset).prepare_epochs(crop=None, time_bin=50)
    pipeline.setup_cv(folds=2, repeats=1, trial_averaging=1, seed=4)
    pipeline.decode(
        target="condition",
        classes={"a": ["a"], "b": ["b"], "c": ["c"]},
        file=result_file,
    )
    with pytest.raises(ValueError, match="incompatible analysis settings"):
        pipeline.decode(
            target="condition",
            classes={"a": ["a"], "b": ["b"], "c": ["c"]},
            output="all",
            file=result_file,
        )

    pipeline.decode(
        target="condition",
        classes={"a": ["a"], "b": ["b"], "c": ["c"]},
        output="all",
        file=result_file,
        recompute="all",
    )
    with duckdb.connect(str(result_file), read_only=True) as connection:
        assert connection.execute("SELECT output FROM analysis").fetchone()[0] == "all"
        columns = connection.execute("DESCRIBE accuracy").df()["column_name"].tolist()
    assert columns == [
        "subject_index",
        "repeat",
        "fold",
        "time",
        "permutation",
        "accuracy",
        "n_correct",
        "n_trials",
    ]


def test_generated_seed_is_reused_and_changed_inputs_are_recomputed(tmp_path):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])
    result_file = tmp_path / "decoding.duckdb"
    pipeline = init_pipeline(dataset).prepare_epochs(crop=None, time_bin=50)
    pipeline.setup_cv(folds=2, repeats=1, trial_averaging=1, seed=None)
    pipeline.decode(
        target="condition",
        classes={"a": ["a"], "b": ["b"], "c": ["c"]},
        file=result_file,
    )
    with duckdb.connect(str(result_file), read_only=True) as connection:
        first_seed = connection.execute("SELECT seed FROM analysis").fetchone()[0]
        first_fingerprint = connection.execute(
            "SELECT fingerprint FROM subjects WHERE subject_index='001'"
        ).fetchone()[0]

    pipeline = init_pipeline(dataset).prepare_epochs(crop=None, time_bin=50)
    pipeline.setup_cv(folds=2, repeats=1, trial_averaging=1, seed=None)
    pipeline.decode(
        target="condition",
        classes={"a": ["a"], "b": ["b"], "c": ["c"]},
        file=result_file,
    )
    eeg_json = open_pipeline(dataset).path_for_subject("001", "eeg_json")
    eeg_json.write_text(eeg_json.read_text() + "\n")
    pipeline.decode(
        target="condition",
        classes={"a": ["a"], "b": ["b"], "c": ["c"]},
        file=result_file,
        recompute="changed",
    )

    with duckdb.connect(str(result_file), read_only=True) as connection:
        assert connection.execute("SELECT seed FROM analysis").fetchone()[0] == first_seed
        status, second_fingerprint = connection.execute(
            "SELECT status, fingerprint FROM subjects WHERE subject_index='001'"
        ).fetchone()
    assert status == "complete"
    assert second_fingerprint != first_fingerprint


def test_failed_subject_is_recorded_without_partial_results(tmp_path):
    dataset = _build_dataset(tmp_path / "dataset", ["001"])
    result_file = tmp_path / "decoding.duckdb"
    pipeline = init_pipeline(dataset).prepare_epochs(crop=None, time_bin=50)
    pipeline.setup_cv(folds=2, repeats=1, trial_averaging=1, seed=3)

    with pytest.raises(RuntimeError, match="No subject completed decoding"):
        pipeline.decode(
            target="condition",
            classes={"a": ["a"], "b": ["b"], "missing": ["missing"]},
            evidence={
                "a": ["a"],
                "b": ["b"],
                "missing": ["missing"],
                "probe": ["x"],
            },
            file=result_file,
        )

    with duckdb.connect(str(result_file), read_only=True) as connection:
        status, reason = connection.execute(
            "SELECT status, reason FROM subjects WHERE subject_index='001'"
        ).fetchone()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
    assert status == "failed"
    assert "missing" in reason
    assert tables == {"analysis", "subjects"}


def _epochs(subject: str) -> mne.EpochsArray:
    rng = np.random.default_rng(int(subject))
    labels = np.repeat(["a", "b", "c", "x"], 4)
    data = rng.normal(size=(len(labels), 2, 11))
    for row, label in enumerate(labels):
        if label in {"a", "b"}:
            data[row, ord(label) - 97, :] += 1.5
    info = mne.create_info(["Fz", "Cz"], sfreq=100, ch_types="eeg")
    info.set_montage("standard_1020")
    metadata = pd.DataFrame(
        {
            "subject_index": [subject] * len(labels),
            "epoch_index": np.arange(len(labels)),
            "condition": labels,
            "final_status": ["accepted"] * len(labels),
            "load": np.tile([1, 2, 3, 4], 4),
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
    subjects = ["001", subject]
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
    builder.record_reused("001")
    builder.write_subject(subject, _epochs(subject), input_fingerprint=f"input-{subject}")
    builder.finish()
