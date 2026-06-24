"""Tests for general preprocessing workflow defaults and facade imports."""

import mne
import numpy as np
import pandas as pd
import pytest

from mveeg.prep.core import EpochConfig, EventConfig, IOConfig, Preprocess
from mveeg.prep import workflow_subjects
from mveeg.prep.workflow import (
    PreprocessWorkflow,
    create_flow,
    get_subject_selection,
    keep_epochs_by_metadata_value,
    redirect_output_to_file,
    reset_epoch_event_samples,
    write_run_log_header,
    write_subject_log_header,
)
from mveeg.prep.workflow_status import load_status_table


def test_preprocess_workflow_does_not_assume_reference_channel():
    """A new workflow should not assume project-specific defaults."""
    flow = PreprocessWorkflow()

    assert flow.reref_channels is None
    assert flow.behavior_name_pattern is None


def test_configure_behavior_accepts_glob_pattern():
    """configure_behavior should store an explicit behavior glob pattern."""
    flow = PreprocessWorkflow()

    flow.configure_behavior(name_pattern="*_beh.csv")

    assert flow.behavior_name_pattern == "*_beh.csv"


def test_configure_behavior_maps_legacy_suffix():
    """Older suffix-based scripts should keep working for one version."""
    flow = PreprocessWorkflow()

    flow.configure_behavior(behavior_suffix="_beh.csv")

    assert flow.behavior_name_pattern == "*_beh.csv"
    assert flow.behavior_suffix == "_beh.csv"


def test_configure_behavior_stores_transform():
    """configure_behavior should store an optional behavior transform."""
    flow = PreprocessWorkflow()

    def add_column(behavior_data):
        """Return behavior rows with an added marker column."""
        behavior_data = behavior_data.copy()
        behavior_data["marker"] = "ok"
        return behavior_data

    flow.configure_behavior(name_pattern="*_beh.csv", behavior_transform=add_column)

    assert flow.behavior_transform is add_column


def test_workflow_facade_keeps_public_imports():
    """Existing imports from mveeg.prep.workflow should remain available."""
    assert isinstance(create_flow(), PreprocessWorkflow)
    assert callable(get_subject_selection)
    assert callable(keep_epochs_by_metadata_value)
    assert callable(reset_epoch_event_samples)
    assert callable(write_run_log_header)
    assert callable(write_subject_log_header)
    assert callable(redirect_output_to_file)


def test_status_table_adds_selected_subjects(tmp_path):
    """Status helpers should create one row per selected subject."""
    subject_dirs = [tmp_path / "sub-001", tmp_path / "sub-002"]
    for subject_dir in subject_dirs:
        subject_dir.mkdir()

    status = load_status_table(tmp_path / "status.tsv", subject_dirs)

    assert status["subject_number"].tolist() == ["sub-001", "sub-002"]
    assert status["status"].tolist() == ["pending", "pending"]
    assert status["final_saved"].tolist() == [False, False]


def test_workflow_paths_use_configured_io(tmp_path):
    """Workflow path methods should delegate to the shared path builders."""
    flow = PreprocessWorkflow()
    flow.configure_io(
        data_dir=tmp_path / "bids",
        root_dir=tmp_path / "raw",
        experiment_name="task",
        derivative_label="clean",
    )

    assert flow.final_epochs_path("sub-001") == (
        tmp_path
        / "bids"
        / "derivatives"
        / "sub-001"
        / "eeg"
        / "sub-001_task_desc-clean_epo.fif"
    )
    assert flow.trial_state_path("001").name == "sub-001_task_desc-clean_trial_state.tsv"


def test_prepare_subject_epochs_skips_behavior_alignment_by_default(monkeypatch):
    """Epoch preparation should not require behavior data unless configured."""
    flow = PreprocessWorkflow()
    called = {"align": False}

    def load_streams(subject_number):
        """Return streams without behavior data for the workflow helper."""
        return "eeg", "events", None, None, False, None

    def build_epochs(subject_number, eeg, eeg_events, eye, eye_events, *, has_eye_data):
        """Return a stand-in epochs object for this behavior-routing test."""
        return "epochs"

    def align_epochs(subject_number, epochs, behavior_data):
        """Record accidental behavior alignment calls."""
        called["align"] = True
        return epochs

    monkeypatch.setattr(flow, "load_subject_streams", load_streams)
    monkeypatch.setattr(flow, "build_subject_epochs", build_epochs)
    monkeypatch.setattr(flow, "keep_aligned_experimental_trials", align_epochs)

    epochs = workflow_subjects.prepare_subject_epochs(flow, "sub-001")

    assert epochs == "epochs"
    assert called["align"] is False


def test_load_subject_streams_applies_behavior_transform(monkeypatch):
    """Behavior transforms should run before pre-filtering and alignment."""
    flow = PreprocessWorkflow()
    flow.behavior_name_pattern = "*_beh.csv"
    flow.pre_filter_rules = {"trial_types": ["exp"], "rejection": "no"}

    behavior_rows = pd.DataFrame(
        {
            "trial_type": ["exp", "pra"],
            "rejection": ["no", "no"],
            "label": ["A", "B"],
        }
    )

    class DummyPre:
        """Minimal preprocessor stand-in for stream loading."""

        def import_eeg(self, subject_number, overwrite=True):
            """Return dummy EEG data and events."""
            return "eeg", "events"

        def import_eyetracker(self, subject_number, overwrite=True):
            """Signal that no eye-tracking file is available."""
            raise FileNotFoundError("no eye")

        def import_behavior(self, subject_number, name_pattern):
            """Record behavior import without writing files."""
            return None

        def load_behavior_table(self, subject_number, name_pattern):
            """Return behavior rows that need a transform before filtering."""
            return behavior_rows

    def add_marker(behavior_data):
        """Add a column before workflow pre-filters rows."""
        behavior_data = behavior_data.copy()
        behavior_data["marker"] = behavior_data["label"].str.lower()
        return behavior_data

    flow.pre = DummyPre()
    flow.behavior_transform = add_marker

    _, _, _, _, _, behavior_data = workflow_subjects.load_subject_streams(flow, "sub-001")

    assert behavior_data["label"].tolist() == ["A"]
    assert behavior_data["marker"].tolist() == ["a"]


def test_behavior_file_matching_uses_glob_and_subject_label(tmp_path):
    """Behavior lookup should support prefix/suffix-like glob patterns."""
    pre = _make_minimal_preprocess(tmp_path)
    subject_dir = tmp_path / "raw" / "sub-001"
    subject_dir.mkdir(parents=True)
    (subject_dir / "task_memory_sub-001_beh.csv").write_text("label\nA\n")
    (subject_dir / "task_memory_sub-002_beh.csv").write_text("label\nB\n")

    behavior_file = pre._find_behavior_file("001", "task_memory_*_beh.csv")

    assert behavior_file.name == "task_memory_sub-001_beh.csv"


def test_behavior_file_matching_rejects_ambiguous_matches(tmp_path):
    """Behavior lookup should fail clearly when a pattern matches twice."""
    pre = _make_minimal_preprocess(tmp_path)
    subject_dir = tmp_path / "raw" / "sub-001"
    subject_dir.mkdir(parents=True)
    (subject_dir / "run1_sub-001_beh.csv").write_text("label\nA\n")
    (subject_dir / "run2_sub-001_beh.csv").write_text("label\nA\n")

    with pytest.raises(ValueError, match="more than one behavior file"):
        pre._find_behavior_file("001", "*_beh.csv")


def test_behavior_file_matching_reports_missing_file(tmp_path):
    """Behavior lookup should fail clearly when no file matches."""
    pre = _make_minimal_preprocess(tmp_path)
    subject_dir = tmp_path / "raw" / "sub-001"
    subject_dir.mkdir(parents=True)
    (subject_dir / "notes.csv").write_text("label\nA\n")

    with pytest.raises(FileNotFoundError, match="Could not find a behavior file"):
        pre._find_behavior_file("001", "*_beh.csv")


def test_eyelink_fallback_reads_numeric_event_messages(tmp_path):
    """EyeLink fallback parsing should keep numeric task event messages."""
    pre = _make_exp2_event_preprocess(tmp_path)
    asc_path = _write_minimal_asc(tmp_path, ["MSG\t100\t21", "MSG\t200\tDISPLAY_COORDS 0 0 1920 1080"])

    eye = pre._read_eyelink_ascii_fallback(asc_path)

    assert eye.annotations.description.tolist() == ["21"]


def test_eyelink_fallback_reads_label_event_messages(tmp_path):
    """EyeLink fallback parsing should map configured label messages to codes."""
    pre = _make_exp2_event_preprocess(tmp_path)
    asc_path = _write_minimal_asc(tmp_path, ["MSG\t100\t2C1N", "MSG\t200\tDELAY"])

    eye = pre._read_eyelink_ascii_fallback(asc_path)

    assert eye.annotations.description.tolist() == ["21", "2"]


def test_import_eyetracker_reports_no_task_events(tmp_path, monkeypatch):
    """Eyetracker import should fail clearly when no task messages are present."""
    pre = _make_exp2_event_preprocess(tmp_path)
    eye = mne.io.RawArray(
        np.zeros((1, 10)),
        mne.create_info(["xpos_left"], sfreq=1000, ch_types=["eyegaze"]),
        verbose="ERROR",
    )
    (tmp_path / "sub-001.asc").write_text("SAMPLES\tRATE\t1000.00\n0\t0\t0\t1\t0\t0\t1\n")

    monkeypatch.setattr(pre, "_resolve_subject_dir", lambda subject_number: tmp_path)
    monkeypatch.setattr(pre, "_convert_edf_to_asc", lambda subject_dir, subject_number: ["sub-001.asc"])
    monkeypatch.setattr(pre, "_read_eyelink_ascii", lambda asc_path: eye)
    monkeypatch.setattr(pre, "_copy_sidecar_template", lambda *args, **kwargs: None)
    monkeypatch.setattr(pre, "_warn_manual_sidecar_once", lambda: None)

    with pytest.raises(ValueError, match="No task eyetracking events were found"):
        pre.import_eyetracker("sub-001", overwrite=True)


def _make_minimal_preprocess(tmp_path):
    """Create a minimal preprocessor for path and file-matching tests."""
    return Preprocess(
        io_config=IOConfig(
            data_dir=tmp_path / "bids",
            root_dir=tmp_path / "raw",
            experiment_name="task",
        ),
        epoch_config=EpochConfig(trial_start=-0.2, trial_end=0.8),
        event_config=EventConfig(event_dict={"A": 1}, event_code_dict={1: [1]}),
    )


def _make_exp2_event_preprocess(tmp_path):
    """Create a preprocessor with exp2-style event labels."""
    event_dict = {
        "TRIAL_START": 1,
        "DELAY": 2,
        "2C1N": 21,
    }
    return Preprocess(
        io_config=IOConfig(
            data_dir=tmp_path / "bids",
            root_dir=tmp_path / "raw",
            experiment_name="exp2",
        ),
        epoch_config=EpochConfig(trial_start=-0.2, trial_end=0.8),
        event_config=EventConfig(
            event_dict=event_dict,
            event_code_dict={21: [1, 21, 2]},
            event_names=event_dict,
        ),
    )


def _write_minimal_asc(tmp_path, messages):
    """Write the smallest EyeLink ASC snippet needed by the fallback parser."""
    asc_path = tmp_path / "test.asc"
    sample_lines = [
        "SAMPLES\tGAZE\tLEFT\tRIGHT\tRATE\t1000.00",
        "0\t0\t0\t1\t0\t0\t1",
        "100\t1\t1\t1\t1\t1\t1",
        "200\t2\t2\t1\t2\t2\t1",
    ]
    asc_path.write_text("\n".join(sample_lines + messages) + "\n")
    return asc_path
