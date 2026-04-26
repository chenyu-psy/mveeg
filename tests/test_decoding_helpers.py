"""Tests for lightweight decoding helper functions."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mveeg.decoding.config import (
    ConditionConfig,
    DatasetConfig,
    DecodeParamConfig,
    DecodingConfig,
    ModelConfig,
    TrialFilterConfig,
)
from mveeg.decoding.models import build_classifier_from_spec
from mveeg.decoding.summaries import build_accuracy_table
from mveeg.decoding.workflow import (
    build_generalization_accuracy_table,
    export_decoding_outputs,
    infer_experiment_settings,
    prepare_decoding_paths,
    run_decoding,
    run_decoding_workflow,
    run_generalization_decoding,
    run_generalization_workflow,
    save_decoding_config,
)
from mveeg.decoding.workflow_subjects import _build_subject_run_plan


def test_build_classifier_from_spec_creates_logistic_regression():
    """Classifier specs should build concrete sklearn estimators."""
    clf = build_classifier_from_spec(
        {
            "backend": "sklearn",
            "model_name": "logistic_regression",
            "model_params": {"max_iter": 200},
        }
    )

    assert isinstance(clf, LogisticRegression)
    assert clf.solver == "lbfgs"
    assert clf.max_iter == 200


def test_build_accuracy_table_adds_subject_and_time_columns():
    """Repeat-level decoding results should become a tidy long table."""
    repeat_df = pd.DataFrame(
        {
            "time_ix": [0, 1],
            "cv_repeat": [0, 0],
            "data_type": ["observed", "observed"],
            "perm_id": [0, 0],
            "accuracy": [0.50, 0.75],
            "balanced_accuracy": [0.50, 0.75],
            "n_correct": [2, 3],
            "n_test_trials": [4, 4],
            "chance_level": [0.50, 0.50],
        }
    )

    table = build_accuracy_table({"101": {"accuracy_by_repeat": repeat_df}}, np.array([0, 50]))

    assert table["subject"].tolist() == ["101", "101"]
    assert table["time_ms"].tolist() == [0, 50]
    assert table["accuracy"].tolist() == [0.50, 0.75]


def test_prepare_decoding_paths_uses_general_defaults(tmp_path):
    """Default decoding output paths should not name a specific experiment."""
    paths = prepare_decoding_paths(tmp_path, "run_a")

    assert paths["results_dir"] == tmp_path / "results" / "main" / "decoding" / "run_a"
    assert paths["log_path"].name == "decoding.log"


def test_decoding_workflow_facade_keeps_public_imports():
    """Existing decoding workflow imports should remain available."""
    assert callable(prepare_decoding_paths)
    assert callable(save_decoding_config)
    assert callable(infer_experiment_settings)
    assert callable(run_decoding)
    assert callable(run_generalization_decoding)
    assert callable(run_decoding_workflow)
    assert callable(run_generalization_workflow)
    assert callable(export_decoding_outputs)
    assert callable(build_generalization_accuracy_table)


def test_infer_experiment_settings_uses_data_folder_defaults(tmp_path):
    """Missing experiment names should be inferred from the data folder."""
    data_dir = tmp_path / "data" / "preprocessed" / "exp2"

    experiment_name, results_subdir = infer_experiment_settings(data_dir, None, None)

    assert experiment_name == "exp2"
    assert results_subdir == "exp2"


def test_save_decoding_config_writes_json(tmp_path):
    """Decoding config helpers should write a reusable JSON payload."""
    cfg = _make_decoding_config(tmp_path)

    payload = save_decoding_config(tmp_path, cfg)

    assert payload["dataset"]["experiment_name"] == "task"
    assert (tmp_path / "config.json").exists()


def test_build_subject_run_plan_keeps_cached_partial_outputs():
    """Partial reruns should keep requested and cached subject outputs."""
    plan = _build_subject_run_plan(
        requested_subject_ids=["sub-002", "003"],
        available_subject_ids=["001", "002", "003"],
        cached_subject_ids=["001"],
    )

    assert plan["subjects_to_process"] == ["002", "003"]
    assert plan["keep_seed_subjects"] == ["001", "002", "003"]
    assert plan["is_full_run"] is False


def _make_decoding_config(tmp_path):
    """Create a minimal decoding config for helper tests."""
    return DecodingConfig(
        dataset=DatasetConfig(data_dir=tmp_path, experiment_name="task"),
        conditions=ConditionConfig(
            train_cond={"left": ["left"], "right": ["right"]},
            test_cond={"left": ["left"], "right": ["right"]},
            cond_col="condition",
        ),
        filters=TrialFilterConfig(),
        decode=DecodeParamConfig(n_repeats=1),
        model=ModelConfig(),
    )
