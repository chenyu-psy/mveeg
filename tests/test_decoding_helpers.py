"""Tests for lightweight decoding helper functions."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mveeg.decoding.models import build_classifier_from_spec
from mveeg.decoding.summaries import build_accuracy_table
from mveeg.decoding.workflow import prepare_decoding_paths


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
