"""Focused contracts for the mveeg 0.3 preparation core."""

from __future__ import annotations

import json

import mne
import numpy as np
import pandas as pd
import pytest

from mveeg.prep.dataset import DatasetBuilder, MANIFEST_COLUMNS, fingerprint, open_pipeline
from mveeg.prep.external import init_external
from mveeg.prep.pipeline import (
    _read_eyelink,
    _read_eyelink_fallback,
    _source_dependencies,
    init_pipeline,
)
from mveeg.prep.steps import (
    _ordered_code_alignment,
    align_behavior,
    assign_identity,
    extract_events,
    filter_eeg,
    merge_metadata,
    sync_eyelink,
    transform_metadata,
)


def _epochs(n_epochs: int = 3) -> mne.EpochsArray:
    """Build small deterministic epochs for metadata tests."""
    info = mne.create_info(["Cz", "Pz"], sfreq=100, ch_types="eeg")
    events = np.column_stack(
        [np.arange(n_epochs), np.zeros(n_epochs, dtype=int), np.ones(n_epochs, dtype=int)]
    )
    return mne.EpochsArray(
        np.zeros((n_epochs, 2, 20)),
        info,
        events=events,
        event_id={"trial": 1},
        verbose="ERROR",
    )


def _write_raw_subject(root, subject: str, n_events: int) -> None:
    """Write one tiny annotated FIF recording for raw-pipeline tests."""
    subject_dir = root / f"sub{subject}"
    subject_dir.mkdir(parents=True)
    raw = mne.io.RawArray(
        np.zeros((1, 600)), mne.create_info(["Cz"], 100, "eeg"), verbose="ERROR"
    )
    raw.set_annotations(
        mne.Annotations(
            onset=np.arange(1, n_events + 1, dtype=float),
            duration=np.zeros(n_events),
            description=["stim"] * n_events,
        )
    )
    raw.save(subject_dir / f"sub{subject}_raw.fif", overwrite=True, verbose="ERROR")


def test_behavior_alignment_is_strict_count_and_order() -> None:
    """Behavior labels never drive alignment or reordering."""
    epochs = _epochs(2)
    behavior = pd.DataFrame({"label": ["second", "first"]})
    aligned = align_behavior(epochs, behavior)
    assert aligned.metadata["label"].tolist() == ["second", "first"]
    with pytest.raises(ValueError, match="equal row counts"):
        align_behavior(epochs, behavior.iloc[:1])


def test_eyelink_sync_removes_leading_warmup_trials() -> None:
    """Trial-code synchronization handles extra leading EyeLink warm-ups."""
    eeg_raw = mne.io.RawArray(
        np.zeros((1, 600)), mne.create_info(["Cz"], 100, "eeg"), verbose="ERROR"
    )
    eeg_raw.set_annotations(
        mne.Annotations([2.0, 4.0], [0.0, 0.0], ["condition_a", "condition_b"])
    )
    eeg = mne.Epochs(
        eeg_raw,
        np.asarray([[200, 0, 1], [400, 0, 2]]),
        event_id={"condition_a": 1, "condition_b": 2},
        tmin=-0.1,
        tmax=0.2,
        preload=True,
        verbose="ERROR",
    )
    eye = mne.io.RawArray(
        np.zeros((1, 600)),
        mne.create_info(["xpos_left"], 100, "eyegaze"),
        verbose="ERROR",
    )
    eye.set_annotations(
        mne.Annotations(
            [1.0, 2.0, 4.0],
            [0.0, 0.0, 0.0],
            ["condition_b", "condition_a", "condition_b"],
        )
    )
    synced = sync_eyelink(
        eeg,
        eye,
        event_id={"condition_a": 1, "condition_b": 2},
    )
    assert len(synced) == 2
    assert synced.events[:, 2].tolist() == [1, 2]
    assert synced.ch_names == ["Cz", "xpos_left"]


def test_eyelink_internal_duplicate_alignment_is_rejected() -> None:
    """Repeated internal condition codes are never resolved by guessing."""
    with pytest.raises(RuntimeError, match="ambiguous"):
        _ordered_code_alignment([1, 2, 3], [1, 2, 2, 3])


def test_eyelink_fallback_preserves_missing_sample_clock(tmp_path) -> None:
    """Fallback parsing keeps explicit and absent sample rows as NaN time points."""
    asc = tmp_path / "eye.asc"
    asc.write_text(
        "SAMPLES GAZE LEFT RIGHT RATE 1000\n"
        "MSG 999 out_of_range_warmup\n"
        "1000 1 2 3 4 5 6\n"
        "1001 . . . . . .\n"
        "1003 7 8 9 10 11 12\n"
        "MSG 1003 condition_a\n"
        "MSG 2000 out_of_range_after_recording\n"
    )
    raw = _read_eyelink_fallback(asc)
    assert raw.n_times == 4
    assert np.isnan(raw.get_data()[:, 1:3]).all()
    assert raw.annotations.onset.tolist() == pytest.approx([0.003])
    assert raw.annotations.description.tolist() == ["condition_a"]
    events = extract_events(raw, event_id={"condition_a": 1})
    assert len(np.unique(events[:, 0])) == len(events) == 1


@pytest.mark.filterwarnings("error")
def test_known_mne_eyelink_status_error_is_silent(tmp_path, monkeypatch) -> None:
    """A validated fallback hides MNE's known signed-value/status parsing bug."""
    asc = tmp_path / "eye.asc"
    asc.write_text("SAMPLES GAZE LEFT RIGHT RATE 1000\n1000 1 2 3 4 -94.4 6 .....\n")

    def fail_mne(*args, **kwargs):
        raise ValueError(
            "Expected the samples data in this file to have 7 columns of data, but got 6. "
            "Expected columns: ['time', 'xpos_left', 'ypos_left', 'pupil_left', "
            "'xpos_right', 'ypos_right', 'pupil_right']."
        )

    monkeypatch.setattr(mne.io, "read_raw_eyelink", fail_mne)
    raw = _read_eyelink(asc)
    assert raw.get_data()[4, 0] == pytest.approx(-94.4)


def test_unexpected_mne_eyelink_error_still_warns(tmp_path, monkeypatch) -> None:
    """Unexpected MNE failures remain visible even when the fallback succeeds."""
    asc = tmp_path / "eye.asc"
    asc.write_text("SAMPLES GAZE LEFT RIGHT RATE 1000\n1000 1 2 3 4 5 6 .....\n")

    def fail_mne(*args, **kwargs):
        raise ValueError("unexpected parser failure")

    monkeypatch.setattr(mne.io, "read_raw_eyelink", fail_mne)
    with pytest.warns(UserWarning, match="MNE could not read"):
        _read_eyelink(asc)


def test_eyelink_fallback_never_ignores_reader_options(tmp_path, monkeypatch) -> None:
    """MNE-specific reader options cannot silently disappear in the fallback."""

    def fail_mne(*args, **kwargs):
        raise ValueError("unexpected parser failure")

    def fail_fallback(*args, **kwargs):
        raise AssertionError("fallback must not run when reader options were supplied")

    monkeypatch.setattr(mne.io, "read_raw_eyelink", fail_mne)
    monkeypatch.setattr("mveeg.prep.pipeline._read_eyelink_fallback", fail_fallback)
    with pytest.raises(ValueError, match="cannot apply reader options: create_annotations"):
        _read_eyelink(tmp_path / "eye.asc", create_annotations=False)


def test_eyelink_reports_both_reader_failures(tmp_path, monkeypatch) -> None:
    """A wholly unreadable file reports both attempted readers."""
    asc = tmp_path / "eye.asc"
    asc.write_text("not EyeLink data\n")

    def fail_mne(*args, **kwargs):
        raise ValueError("unexpected parser failure")

    monkeypatch.setattr(mne.io, "read_raw_eyelink", fail_mne)
    with pytest.raises(ValueError, match="MNE reader failed:.*mveeg reader failed:"):
        _read_eyelink(asc)


def test_event_extraction_restricts_numeric_messages_to_configured_codes() -> None:
    """Numeric non-task messages such as display dimensions are ignored."""
    raw = mne.io.RawArray(
        np.zeros((1, 300)), mne.create_info(["Cz"], 100, "eeg"), verbose="ERROR"
    )
    raw.set_annotations(
        mne.Annotations(
            [1.0, 2.0],
            [0.0, 0.0],
            ["DISPLAY_COORDS 0 0 1920 1080", "Stimulus/S 20"],
        )
    )
    assert extract_events(raw, event_id={"condition": 20})[:, 2].tolist() == [20]


def test_external_merge_and_transform_protect_rows() -> None:
    """External metadata merges are one-to-one and transforms cannot reorder."""
    epochs = _epochs(3)
    epochs.metadata = pd.DataFrame({"source_trial": [2, 1, 3]})
    merged = merge_metadata(
        epochs,
        pd.DataFrame({"behavior_trial": [1, 2, 3], "condition": ["b", "a", "c"]}),
        epoch_key="source_trial",
        metadata_key="behavior_trial",
    )
    assert merged.metadata["condition"].tolist() == ["a", "b", "c"]
    with pytest.raises(ValueError, match="unique"):
        merge_metadata(
            epochs,
            pd.DataFrame({"behavior_trial": [1, 1, 3]}),
            epoch_key="source_trial",
            metadata_key="behavior_trial",
        )
    with pytest.raises(ValueError, match="row order"):
        transform_metadata(
            merged,
            lambda frame: frame.iloc[::-1].reset_index(drop=True),
        )
    with pytest.raises(ValueError, match="reserved identity"):
        merge_metadata(epochs, pd.DataFrame({"subject_index": [1, 1, 1]}))
    with pytest.raises(ValueError, match="reserved column"):
        transform_metadata(merged, lambda frame: frame.assign(epoch_index=range(len(frame))))


def test_external_pipeline_writes_standard_dataset(tmp_path) -> None:
    """Array and MNE inputs converge on the same standard storage contract."""
    output = tmp_path / "prepared"
    metadata = pd.DataFrame(
        {
            "trial_type": ["exp", "pra", "exp"],
            "score": [1, 2, 3],
            "continuous_value": [
                0.8275291992661313,
                0.593530612259512,
                0.5481671573183453,
            ],
        }
    )
    pipeline = init_external(
        subject_index="4001", data=np.zeros((3, 2, 20), dtype=float)
    )
    dataset = (
        pipeline.make_epochs(sampling_rate=100, ch_names=["Cz", "Pz"])
        .merge_metadata(metadata)
        .transform_metadata(
            lambda frame: frame.assign(centered=frame["score"] - frame["score"].mean()),
            name="center_score",
            version="1",
        )
        .select_epochs(include={"trial_type": "exp"})
        .build_epochs(output, task="exp4")
    )

    assert list(dataset.manifest.columns) == MANIFEST_COLUMNS
    assert dataset.subject_indices == ("4001",)
    assert dataset.path_for_subject("sub4001").name == "sub-4001_task-exp4_desc-prepared_epo.fif"
    assert dataset.path_for_subject("4001", "artifacts").name == "sub-4001_task-exp4_desc-artifacts.tsv"
    assert dataset.path_for_subject("4001", "events").exists()
    saved = dataset.load_epochs("4001")
    assert saved.metadata["subject_index"].astype(str).tolist() == ["4001", "4001"]
    assert saved.metadata["epoch_index"].tolist() == [0, 1]
    assert saved.metadata["trial_type"].tolist() == ["exp", "exp"]
    assert saved.metadata["continuous_value"].tolist() == [
        0.8275291992661313,
        0.5481671573183453,
    ]
    assert json.loads((output / "provenance.json").read_text())["stage"] == "prepared"
    assert open_pipeline(output).subject_indices == ("4001",)


def test_raw_gaze_geometry_is_persisted_after_build_without_changing_fingerprint(
    tmp_path,
) -> None:
    """Raw geometry is pending until a successful build and is not a signal step."""
    input_dir = tmp_path / "raw"
    _write_raw_subject(input_dir, "4001", 2)

    configured = init_pipeline(input_dir)
    configured.load_eeg("*_raw.fif")
    configured.make_epochs(event_id={"stim": 1}, time_window=(0, 0.1))
    assert configured.configure_gaze(
        viewing_distance_cm=80,
        screen_width_cm=53.2,
        screen_width_px=1920,
    ) is configured
    configured_root = tmp_path / "configured"
    assert not configured_root.exists()
    configured_dataset = configured.build_epochs(
        configured_root,
        task="task",
        progress=False,
    )

    plain = init_pipeline(input_dir)
    plain.load_eeg("*_raw.fif")
    plain.make_epochs(event_id={"stim": 1}, time_window=(0, 0.1))
    plain_dataset = plain.build_epochs(tmp_path / "plain", task="task", progress=False)

    provenance = json.loads((configured_root / "provenance.json").read_text())
    assert provenance["gaze_geometry"] == {
        "viewing_distance_cm": 80.0,
        "screen_width_cm": 53.2,
        "screen_width_px": 1920,
    }
    assert configured_dataset.manifest.loc[0, "pipeline_fingerprint"] == (
        plain_dataset.manifest.loc[0, "pipeline_fingerprint"]
    )


def test_prepared_gaze_geometry_is_idempotent_overwritable_and_inherited(
    tmp_path,
    monkeypatch,
) -> None:
    """Reopened prepared roots persist one geometry across later subject writes."""
    import mveeg.prep.dataset as dataset_module

    output = tmp_path / "prepared"
    first = (
        init_external(subject_index="4001", data=np.zeros((2, 1, 10)))
        .make_epochs(sampling_rate=100, ch_names=["Cz"])
        .configure_gaze(
            viewing_distance_cm=80,
            screen_width_cm=53.2,
            screen_width_px=1920,
        )
        .build_epochs(output, task="task")
    )
    fingerprint_before = first.manifest.loc[0, "pipeline_fingerprint"]
    reopened = open_pipeline(output)

    with monkeypatch.context() as patch:
        writes = []
        original_write = dataset_module._write_json_atomic

        def track_write(value, path):
            writes.append(path)
            original_write(value, path)

        patch.setattr(dataset_module, "_write_json_atomic", track_write)
        assert reopened.configure_gaze(
            viewing_distance_cm=80.0,
            screen_width_cm=53.2,
            screen_width_px=1920,
        ) is reopened
        assert writes == []
        reopened.configure_gaze(
            viewing_distance_cm=90,
            screen_width_cm=60,
            screen_width_px=2560,
        )
        assert writes == [output / "provenance.json"]

    second = (
        init_external(subject_index="4002", data=np.zeros((2, 1, 10)))
        .make_epochs(sampling_rate=100, ch_names=["Cz"])
        .build_epochs(output, task="task")
    )
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["gaze_geometry"] == {
        "viewing_distance_cm": 90.0,
        "screen_width_cm": 60.0,
        "screen_width_px": 2560,
    }
    assert second.subject_indices == ("4001", "4002")
    assert set(second.manifest["pipeline_fingerprint"]) == {fingerprint_before}


def test_preprocessed_dataset_cannot_configure_gaze(tmp_path) -> None:
    """Only prepared roots own the persisted acquisition geometry."""
    epochs = assign_identity(_epochs(2), "4001")
    pipeline_spec = {"kind": "test_preprocessed"}
    builder = DatasetBuilder(
        tmp_path / "preprocessed",
        task="task",
        stage="preprocessed",
        pipeline_fingerprint=fingerprint(pipeline_spec),
        pipeline_spec=pipeline_spec,
        recompute="never",
        subject_indices=["4001"],
        complete_subject_set=True,
    )
    builder.write_subject("4001", epochs, input_fingerprint="input")
    preprocessed = builder.finish()

    with pytest.raises(ValueError, match="only available for prepared"):
        preprocessed.configure_gaze(
            viewing_distance_cm=80,
            screen_width_cm=53.2,
            screen_width_px=1920,
        )


def test_dataset_load_epochs_validates_events_metadata_mirror(tmp_path) -> None:
    output = tmp_path / "prepared"
    dataset = (
        init_external(subject_index="4001", data=_epochs(2))
        .merge_metadata(pd.DataFrame({"condition": ["a", "b"]}))
        .build_epochs(output, task="task")
    )
    events_path = dataset.path_for_subject("4001", "events")
    events = pd.read_csv(events_path, sep="\t")
    events.loc[1, "condition"] = "changed"
    events.to_csv(events_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="same base metadata values"):
        dataset.load_epochs("4001")


def test_external_dataset_rejects_partial_global_rebuild(tmp_path) -> None:
    """A changed global pipeline cannot silently mix old and new subjects."""
    output = tmp_path / "prepared"
    for subject in ["4001", "4002"]:
        (
            init_external(subject_index=subject, data=np.zeros((2, 1, 10)))
            .make_epochs(sampling_rate=100, ch_names=["Cz"])
            .build_epochs(output, task="task")
        )
    changed = init_external(subject_index="4001", data=np.zeros((2, 1, 10))).make_epochs(
        sampling_rate=100, ch_names=["Pz"]
    )
    with pytest.raises(RuntimeError, match="rebuilding every existing subject"):
        changed.build_epochs(output, task="task", recompute="changed")


def test_recompute_never_reuses_and_changed_updates_subject(tmp_path) -> None:
    """Input-only changes obey the explicit never/changed policy."""
    output = tmp_path / "prepared"
    first = init_external(subject_index="4001", data=np.zeros((2, 1, 10))).make_epochs(
        sampling_rate=100, ch_names=["Cz"]
    )
    first.build_epochs(output, task="task")
    changed = init_external(subject_index="4001", data=np.ones((2, 1, 10))).make_epochs(
        sampling_rate=100, ch_names=["Cz"]
    )
    with pytest.warns(UserWarning, match="Input changed"):
        reused = changed.build_epochs(output, task="task", recompute="never")
    assert reused.load_epochs("4001").get_data(copy=True).sum() == 0
    updated = changed.build_epochs(output, task="task", recompute="changed")
    assert updated.load_epochs("4001").get_data(copy=True).sum() == 20


def test_reuse_requires_all_manifest_files(tmp_path) -> None:
    """Missing events or signal-sidecar files cannot be reported as reused."""
    output = tmp_path / "prepared"
    pipeline = init_external(subject_index="4001", data=np.zeros((2, 1, 10))).make_epochs(
        sampling_rate=100, ch_names=["Cz"]
    )
    dataset = pipeline.build_epochs(output, task="task")
    dataset.path_for_subject("4001", "events").unlink()
    with pytest.raises(FileNotFoundError, match="events_path"):
        pipeline.build_epochs(output, task="task", recompute="never")


def test_brainvision_fingerprint_includes_signal_and_marker_files(tmp_path) -> None:
    """BrainVision headers bring their signal and marker companions into provenance."""
    header = tmp_path / "recording.vhdr"
    signal = tmp_path / "recording.eeg"
    marker = tmp_path / "recording.vmrk"
    for path in (header, signal, marker):
        path.write_bytes(b"test")
    assert _source_dependencies(header) == [header, signal, marker]


def test_raw_pipeline_prefilters_then_aligns_then_selects(tmp_path) -> None:
    """Raw behavior filtering occurs before strict alignment and selection after."""
    input_dir = tmp_path / "raw"
    subject_dir = input_dir / "sub4001"
    subject_dir.mkdir(parents=True)
    info = mne.create_info(["Cz", "Pz"], sfreq=100, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((2, 600)), info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(onset=[1.0, 3.0], duration=[0.0, 0.0], description=["stim", "stim"])
    )
    raw.save(subject_dir / "sub4001_raw.fif", overwrite=True, verbose="ERROR")
    pd.DataFrame(
        {
            "trial_type": ["exp", "pra", "discard"],
            "rejection": ["no", "no", "yes"],
            "label": ["A", "B", "A"],
        }
    ).to_csv(subject_dir / "sub4001_beh.csv", index=False)

    pipeline = init_pipeline(input_dir)
    pipeline.load_eeg("*_raw.fif")
    pipeline.load_behavior(
        "*_beh.csv",
        include={"trial_type": ["exp", "pra"], "rejection": "no"},
    )
    pipeline.make_epochs(event_id={"stim": 1}, time_window=(-0.1, 0.2))
    pipeline.align_behavior()
    pipeline.select_epochs(include={"trial_type": "exp"})
    dataset = pipeline.build_epochs(tmp_path / "prepared", task="task")

    saved = dataset.load_epochs("4001")
    assert len(saved) == 1
    assert saved.metadata.loc[0, "trial_type"] == "exp"
    assert dataset.run_summary.to_dict("records") == [
        {"subject_index": "4001", "status": "written"}
    ]


def test_raw_build_progress_tracks_selected_and_reused(tmp_path, monkeypatch) -> None:
    """The one dataset bar covers every selected subject, including reuse."""
    input_dir = tmp_path / "raw"
    _write_raw_subject(input_dir, "4001", 2)
    _write_raw_subject(input_dir, "4002", 2)
    output = tmp_path / "prepared"
    calls = []

    def capture_tqdm(iterable, **kwargs):
        items = list(iterable)
        calls.append((len(items), kwargs))
        return items

    monkeypatch.setattr("mveeg.prep.pipeline.tqdm", capture_tqdm)
    pipeline = init_pipeline(input_dir)
    pipeline.load_eeg("*_raw.fif")
    pipeline.make_epochs(event_id={"stim": 1}, time_window=(0, 0.1))

    written = pipeline.build_epochs(output, task="task", progress=False)
    assert calls == [
        (2, {"desc": "Building epochs", "unit": "subject", "disable": True})
    ]
    assert written.run_summary["status"].tolist() == ["written", "written"]

    calls.clear()
    reused = pipeline.build_epochs(output, task="task", progress=True)
    assert calls == [
        (2, {"desc": "Building epochs", "unit": "subject", "disable": False})
    ]
    assert reused.run_summary["status"].tolist() == ["reused", "reused"]


def test_filter_eeg_silences_raw_loading() -> None:
    """Package-owned loading does not emit MNE's uninformative read log."""
    calls = []

    class RawStub:
        def copy(self):
            return self

        def load_data(self, *, verbose=None):
            calls.append(verbose)
            return self

        def filter(self, **kwargs):
            return self

    raw = RawStub()
    assert filter_eeg(raw, h_freq=40) is raw
    assert calls == ["ERROR"]


def test_existing_dataset_is_unchanged_when_global_rebuild_fails(tmp_path) -> None:
    """A later-subject failure discards all earlier staged global changes."""
    input_dir = tmp_path / "raw"
    _write_raw_subject(input_dir, "4001", 2)
    _write_raw_subject(input_dir, "4002", 3)
    output = tmp_path / "prepared"

    initial = init_pipeline(input_dir)
    initial.load_eeg("*_raw.fif")
    initial.make_epochs(event_id={"stim": 1}, time_window=(0, 0.1))
    initial.build_epochs(output, task="task")
    old_provenance = (output / "provenance.json").read_text()

    def change_then_fail(epochs):
        if len(epochs) == 3:
            raise RuntimeError("synthetic second-subject failure")
        changed = epochs.copy()
        changed._data += 1
        return changed

    changed = init_pipeline(input_dir)
    changed.load_eeg("*_raw.fif")
    changed.make_epochs(event_id={"stim": 1}, time_window=(0, 0.1))
    changed.add_epoch_step(change_then_fail, name="failing_change", version="1")
    with pytest.raises(RuntimeError, match="second-subject failure"):
        changed.build_epochs(output, task="task", recompute="changed")

    reopened = open_pipeline(output)
    assert (output / "provenance.json").read_text() == old_provenance
    assert reopened.load_epochs("4001").get_data(copy=True).sum() == 0
    assert not list(tmp_path.glob(".prepared.mveeg-stage-*"))
    with pytest.raises(RuntimeError, match="outside the current cohort"):
        initial.build_epochs(
            output,
            task="task",
            exclude_subjects=["4002"],
            recompute="never",
        )


def test_raw_pipeline_requires_real_phase_dependencies(tmp_path) -> None:
    """Optional modalities stay optional while epoch construction remains required."""
    pipeline = init_pipeline(tmp_path)
    with pytest.raises(RuntimeError, match="load_eeg"):
        pipeline.make_epochs(event_id={"stim": 1}, time_window=(0, 1))
    pipeline.load_eeg("*.fif")
    with pytest.raises(RuntimeError, match="make_epochs"):
        pipeline.select_epochs(include={"condition": "A"})
    pipeline.make_epochs(event_id={"stim": 1}, time_window=(0, 1))
    pipeline.load_behavior("*.csv")
    pipeline.align_behavior()
