"""Tests for lightweight decoding helper functions."""

import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mveeg.decoding import workflow as decoding_workflow
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
from mveeg.decoding.workflow_outputs import (
    build_topography_coord_table,
    build_topography_value_table,
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
    assert "figures_dir" not in paths


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


def test_build_topography_value_table_averages_and_z_scores_windows():
    """Topography value export should match WT13-style R plotting inputs."""
    pattern_df = pd.DataFrame(
        {
            "subject": ["001", "001", "001", "001", "002", "002", "002", "002"],
            "channel": ["Fz", "Fz", "Cz", "Cz", "Fz", "Fz", "Cz", "Cz"],
            "time_ms": [0, 50, 0, 50, 0, 50, 0, 50],
            "value": [1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0, 9.0],
        }
    )

    table = build_topography_value_table(
        pattern_df=pattern_df,
        windows_ms={"early": (0, 50)},
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
    assert table["channel"].tolist() == ["Cz", "Fz"]
    assert table["effect"].tolist() == ["Decoding pattern", "Decoding pattern"]
    assert table["raw_value"].tolist() == [6.0, 4.0]
    assert table["z_value"].tolist() == [1.0, -1.0]
    assert table["window_start_ms"].tolist() == [0, 0]
    assert table["window_end_ms"].tolist() == [50, 50]
    assert table["n_subjects"].tolist() == [2, 2]


def test_build_topography_value_table_rejects_empty_windows():
    """Requested windows with no data should fail before R plotting."""
    pattern_df = pd.DataFrame(
        {
            "subject": ["001"],
            "channel": ["Fz"],
            "time_ms": [0],
            "value": [1.0],
        }
    )

    try:
        build_topography_value_table(
            pattern_df=pattern_df,
            windows_ms={"missing": (100, 200)},
        )
    except ValueError as err:
        assert "No decoding pattern values" in str(err)
    else:
        raise AssertionError("Expected empty topography window to raise ValueError.")


def test_build_topography_coord_table_exports_projected_mne_positions():
    """Coordinate export should produce finite millimeter x/y values for R."""
    info = mne.create_info(["Fz", "Cz"], sfreq=250, ch_types="eeg")
    info.set_montage("standard_1020")

    table = build_topography_coord_table(info=info, channels=["Fz", "Cz"])

    assert table.columns.tolist() == ["channel", "x", "y"]
    assert table["channel"].tolist() == ["Fz", "Cz"]
    assert np.isfinite(table["x"]).all()
    assert np.isfinite(table["y"]).all()


def test_build_topography_coord_table_rejects_missing_positions():
    """Coordinate export should fail clearly when epochs lack a montage."""
    info = mne.create_info(["Fz", "Cz"], sfreq=250, ch_types="eeg")

    try:
        build_topography_coord_table(info=info, channels=["Fz", "Cz"])
    except ValueError as err:
        assert "electrode montage positions" in str(err)
    else:
        raise AssertionError("Expected missing montage positions to raise ValueError.")


def test_export_decoding_outputs_writes_r_ready_topography_csvs(tmp_path, monkeypatch):
    """Decoding export should write CSV topography data instead of PNG manifests."""
    from mveeg.decoding import workflow_outputs

    info = mne.create_info(["Fz", "Cz"], sfreq=250, ch_types="eeg")
    info.set_montage("standard_1020")
    monkeypatch.setattr(workflow_outputs, "load_subject_info", lambda _subject, _cfg: info)

    run_output = {
        "trial_summary_df": pd.DataFrame({"subject": ["001"]}),
        "skipped_subjects_df": pd.DataFrame(columns=["subject", "reason"]),
        "accuracy_df": pd.DataFrame({"subject": ["001"], "accuracy": [0.5]}),
        "hyperplane_df": pd.DataFrame({"subject": ["001"], "distance": [0.1]}),
        "pattern_df": pd.DataFrame(
            {
                "subject": ["001", "001"],
                "channel": ["Fz", "Cz"],
                "time_ms": [0, 0],
                "value": [1.0, 3.0],
            }
        ),
        "reference_ch_names": ["Fz", "Cz"],
        "topography_subject_id": "001",
    }

    topography_df = export_decoding_outputs(
        run_output=run_output,
        cfg=_make_decoding_config(tmp_path),
        results_dir=tmp_path,
        topo_windows_ms={"early": (0, 0)},
    )

    assert (tmp_path / "topography_values.csv").exists()
    assert (tmp_path / "topography_coords.csv").exists()
    assert not (tmp_path / "topo_manifest.csv").exists()
    assert topography_df["channel"].tolist() == ["Cz", "Fz"]
    assert topography_df["z_value"].tolist() == [1.0, -1.0]


def test_run_decoding_appends_new_subjects_to_result_store(tmp_path, monkeypatch):
    """Existing DuckDB stores should only process requested missing subjects."""

    processed = []

    def fake_discover_subject_ids(_data_dir):
        return ["001", "002", "003"]

    def fake_run_decoding_workflow(**kwargs):
        subject_ids = [str(subject_id) for subject_id in kwargs["subject_ids"]]
        processed.extend(subject_ids)
        return {
            "accuracy_df": pd.DataFrame(
                {
                    "subject": subject_ids,
                    "time_ms": [0] * len(subject_ids),
                    "accuracy": [0.75] * len(subject_ids),
                }
            ),
            "hyperplane_df": pd.DataFrame(
                {
                    "subject": subject_ids,
                    "trial_id": [1] * len(subject_ids),
                    "condition": ["a"] * len(subject_ids),
                    "time_ms": [0] * len(subject_ids),
                    "distance": [0.1] * len(subject_ids),
                }
            ),
            "pattern_df": pd.DataFrame(
                {
                    "subject": subject_ids,
                    "channel": ["Cz"] * len(subject_ids),
                    "time_ms": [0] * len(subject_ids),
                    "value": [1.0] * len(subject_ids),
                }
            ),
            "trial_summary_df": pd.DataFrame({"subject": subject_ids}),
            "skipped_subjects_df": pd.DataFrame(columns=["subject", "reason"]),
        }

    def fake_build_decoding_result_tables(**kwargs):
        run_output = kwargs["run_output"]
        return {
            "accuracy": run_output["accuracy_df"],
            "hyperplane": run_output["hyperplane_df"],
            "topography_values": pd.DataFrame({"channel": ["Cz"], "raw_value": [1.0]}),
            "topography_coords": pd.DataFrame({"channel": ["Cz"], "x": [0.0], "y": [0.0]}),
            "trials": run_output["trial_summary_df"],
            "skipped": run_output["skipped_subjects_df"],
        }

    monkeypatch.setattr(decoding_workflow, "discover_subject_ids", fake_discover_subject_ids)
    monkeypatch.setattr(decoding_workflow, "run_decoding_workflow", fake_run_decoding_workflow)
    monkeypatch.setattr(decoding_workflow, "build_decoding_result_tables", fake_build_decoding_result_tables)

    file = tmp_path / "decode"
    run_decoding(
        **_decoding_kwargs(
            tmp_path,
            subject_ids=["001", "002"],
            overwrite=False,
            topo_windows_ms={"early": (0, 50)},
            file=file,
        ),
    )
    tables = run_decoding(
        **_decoding_kwargs(
            tmp_path,
            subject_ids=["003"],
            overwrite=False,
            topo_windows_ms={"early": (0, 50)},
            file=file,
        ),
    )

    assert processed == ["001", "002", "003"]
    assert tables["accuracy"]["subject"].tolist() == ["001", "002", "003"]

    tables = run_decoding(
        **_decoding_kwargs(
            tmp_path,
            subject_ids=["002"],
            overwrite=True,
            topo_windows_ms={"early": (0, 50)},
            file=file,
        ),
    )

    assert processed == ["001", "002", "003", "002"]
    assert tables["accuracy"]["subject"].tolist() == ["001", "003", "002"]


def test_run_generalization_decoding_appends_new_subjects_to_result_store(tmp_path, monkeypatch):
    """Generalization decoding should share incremental store behavior."""

    processed = []

    def fake_discover_subject_ids(_data_dir):
        return ["001", "002", "003"]

    def fake_run_generalization_workflow(**kwargs):
        subject_ids = [str(subject_id) for subject_id in kwargs["subject_ids"]]
        processed.extend(subject_ids)
        return {
            "accuracy_df": pd.DataFrame(
                {
                    "subject": subject_ids,
                    "train_time_ms": [0] * len(subject_ids),
                    "test_time_ms": [0] * len(subject_ids),
                    "accuracy": [0.75] * len(subject_ids),
                }
            ),
            "trial_summary_df": pd.DataFrame({"subject": subject_ids}),
            "skipped_subjects_df": pd.DataFrame(columns=["subject", "reason"]),
        }

    monkeypatch.setattr(decoding_workflow, "discover_subject_ids", fake_discover_subject_ids)
    monkeypatch.setattr(decoding_workflow, "run_generalization_workflow", fake_run_generalization_workflow)

    file = tmp_path / "generalization"
    run_generalization_decoding(
        **_decoding_kwargs(
            tmp_path,
            subject_ids=["001", "002"],
            overwrite=False,
            file=file,
        ),
    )
    tables = run_generalization_decoding(
        **_decoding_kwargs(
            tmp_path,
            subject_ids=["003"],
            overwrite=False,
            file=file,
        ),
    )

    assert processed == ["001", "002", "003"]
    assert tables["accuracy"]["subject"].tolist() == ["001", "002", "003"]

    tables = run_generalization_decoding(
        **_decoding_kwargs(
            tmp_path,
            subject_ids=["002"],
            overwrite=True,
            file=file,
        ),
    )

    assert processed == ["001", "002", "003", "002"]
    assert tables["accuracy"]["subject"].tolist() == ["001", "003", "002"]


def _decoding_kwargs(tmp_path, **overrides):
    kwargs = {
        "data_dir": tmp_path / "missing-data",
        "trial_filters": {"qc_col": "qc", "keep_qc": ["accepted"], "exclude_metadata": {}},
        "decoding_params": {
            "crop_time": (0.0, 1.0),
            "time_window_ms": 50,
            "trial_bin_size": 1,
            "n_splits": 2,
            "n_repeats": 1,
            "n_jobs": 1,
            "drop_channel_types": [],
            "drop_channels": [],
        },
        "classifier": {"backend": "sklearn", "model_name": "lda"},
        "name": "decode",
        "train_conditions": {"a": ["a"], "b": ["b"]},
        "test_conditions": {"a": ["a"], "b": ["b"]},
    }
    kwargs.update(overrides)
    return kwargs


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
