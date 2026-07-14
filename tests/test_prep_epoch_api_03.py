"""Focused tests for the 0.3 raw epoch-construction contract."""

from __future__ import annotations

import inspect

import mne
import numpy as np
import pytest

from mveeg.prep import steps
from mveeg.prep.pipeline import RawPipeline


EVENT_ID = {
    "start": 1,
    "target": 2,
    "end": 3,
    "trial_a": 10,
    "trial_b": 20,
}
TRIAL_SEQUENCES = {
    10: (1, 10, 2, 3),
    20: ((1, 20, 2, 3), (1, 20, 3, 2)),
}
EVENTS = np.asarray(
    [
        [100, 0, 1],
        [110, 0, 10],
        [120, 0, 2],
        [130, 0, 3],
        [300, 0, 1],
        [310, 0, 20],
        [320, 0, 3],
        [330, 0, 2],
    ],
    dtype=int,
)


def _raw() -> mne.io.RawArray:
    info = mne.create_info(["Cz"], sfreq=100, ch_types="eeg")
    return mne.io.RawArray(np.zeros((1, 500)), info, verbose="ERROR")


def _sequence_epochs(time_zero=None, **kwargs):
    return steps.make_epochs(
        _raw(),
        event_id=EVENT_ID,
        time_window=(-0.04, 0.04),
        trial_sequences=TRIAL_SEQUENCES,
        time_zero=time_zero,
        events=EVENTS,
        baseline=None,
        **kwargs,
    )


def test_sequence_time_zero_supports_default_shared_and_per_trial_targets():
    default = _sequence_epochs()
    shared = _sequence_epochs(2)
    mapped = _sequence_epochs({10: 2, 20: 3})

    assert default.events[:, 0].tolist() == [110, 310]
    assert shared.events[:, 0].tolist() == [120, 330]
    assert mapped.events[:, 0].tolist() == [120, 320]
    assert default.metadata.loc[0, "trial_a"] == pytest.approx(0)
    assert default.metadata.loc[0, "target"] == pytest.approx(0.1)
    assert shared.metadata.loc[0, "trial_a"] == pytest.approx(-0.1)
    assert shared.metadata.loc[0, "target"] == pytest.approx(0)


@pytest.mark.parametrize(
    ("time_zero", "message"),
    [
        ({10: 2}, "exactly"),
        ({10: 2, 20: 2, 30: 2}, "exactly"),
    ],
)
def test_sequence_time_zero_mapping_must_match_trial_keys(time_zero, message):
    with pytest.raises(ValueError, match=message):
        _sequence_epochs(time_zero)


@pytest.mark.parametrize(
    "sequences",
    [
        {10: ((1, 10, 2, 3), (1, 10, 3))},
        {10: ((1, 10, 2, 3), (1, 10, 2, 2, 3))},
    ],
)
def test_target_event_must_occur_once_in_every_alternative(sequences):
    with pytest.raises(ValueError, match="exactly once in every alternative"):
        steps.make_epochs(
            _raw(),
            event_id=EVENT_ID,
            time_window=(-0.04, 0.04),
            trial_sequences=sequences,
            time_zero=2,
            events=EVENTS,
        )


def test_direct_event_mode_rejects_time_zero():
    with pytest.raises(ValueError, match="only valid when trial_sequences"):
        steps.make_epochs(
            _raw(),
            event_id={"start": 1},
            time_window=(-0.04, 0.04),
            time_zero=1,
            events=EVENTS,
        )


@pytest.mark.parametrize(
    "time_window",
    [(np.nan, 1), (-1, np.inf), (-np.inf, 1)],
)
def test_time_window_requires_finite_endpoints(time_window):
    with pytest.raises(ValueError, match="finite"):
        steps.make_epochs(
            _raw(),
            event_id={"start": 1},
            time_window=time_window,
            events=EVENTS,
        )


@pytest.mark.parametrize(
    "time_window",
    [(0, 0), (1, 0), (0,), (0, 1, 2), (True, 1), "01", ("0", "1")],
)
def test_time_window_requires_two_increasing_values(time_window):
    with pytest.raises(ValueError, match="exactly two|start earlier"):
        steps.make_epochs(
            _raw(),
            event_id={"start": 1},
            time_window=time_window,
            events=EVENTS,
        )


@pytest.mark.parametrize(
    "sampling_rate", [0, -1, np.nan, np.inf, True, np.bool_(True), "50"]
)
def test_sampling_rate_requires_a_finite_positive_number(sampling_rate):
    with pytest.raises(ValueError, match="finite positive"):
        steps.make_epochs(
            _raw(),
            event_id={"start": 1},
            time_window=(-0.04, 0.04),
            sampling_rate=sampling_rate,
            events=EVENTS,
        )


def test_sampling_rate_none_preserves_the_native_rate():
    epochs = steps.make_epochs(
        _raw(),
        event_id={"start": 1},
        time_window=(-0.04, 0.04),
        events=EVENTS,
    )

    assert epochs.info["sfreq"] == pytest.approx(100)


def test_sampling_rate_resamples_epochs_and_old_public_names_are_absent():
    epochs = _sequence_epochs(sampling_rate=50)

    assert epochs.info["sfreq"] == pytest.approx(50)
    for function in (steps.make_epochs, RawPipeline.make_epochs):
        parameters = inspect.signature(function).parameters
        assert "time_window" in parameters
        assert "sampling_rate" in parameters
        assert {"tmin", "tmax", "sfreq", "timelock_index"}.isdisjoint(parameters)
    assert list(inspect.signature(RawPipeline.sync_eyelink).parameters) == ["self"]


def test_raw_pipeline_resolves_time_zero_and_sync_inherits_epoch_config(
    tmp_path,
    monkeypatch,
):
    pipeline = RawPipeline(tmp_path)
    (
        pipeline.load_eeg()
        .load_eyelink()
        .make_epochs(
            event_id=EVENT_ID,
            time_window=(-0.04, 0.04),
            trial_sequences=TRIAL_SEQUENCES,
            time_zero=2,
            baseline=None,
            sampling_rate=50,
            events=EVENTS,
        )
        .sync_eyelink()
    )

    make_operation, sync_operation = pipeline._operations
    assert make_operation.params["time_zero"] == {10: 2, 20: 2}
    assert sync_operation.params == {}

    raw = _raw()
    monkeypatch.setattr(pipeline, "_load_eeg", lambda subject_dir: raw.copy())
    monkeypatch.setattr(pipeline, "_load_eye", lambda subject_dir: raw.copy())
    captured = {}

    def fake_sync(epochs, eye_raw, **kwargs):
        captured.update(kwargs)
        return epochs

    monkeypatch.setattr(steps, "sync_eyelink", fake_sync)
    epochs = pipeline._process_subject(tmp_path)

    assert captured == {
        "event_id": EVENT_ID,
        "time_window": (float(epochs.tmin), float(epochs.tmax)),
        "trial_sequences": TRIAL_SEQUENCES,
        "time_zero": {10: 2, 20: 2},
        "baseline": None,
        "sampling_rate": 50.0,
    }


def test_raw_pipeline_validates_and_resolves_default_time_zero_at_registration(tmp_path):
    pipeline = RawPipeline(tmp_path).load_eeg()
    pipeline.make_epochs(
        event_id=EVENT_ID,
        time_window=(-0.04, 0.04),
        trial_sequences=TRIAL_SEQUENCES,
    )

    make_spec = pipeline._pipeline_spec()["steps"][0]
    assert make_spec["time_zero"] == {10: 10, 20: 20}

    for invalid_window in ((np.nan, 1), (-1, np.inf)):
        candidate = RawPipeline(tmp_path).load_eeg()
        with pytest.raises(ValueError, match="finite"):
            candidate.make_epochs(
                event_id={"start": 1},
                time_window=invalid_window,
            )

    for invalid_rate in (0, np.inf, True):
        candidate = RawPipeline(tmp_path).load_eeg()
        with pytest.raises(ValueError, match="finite positive"):
            candidate.make_epochs(
                event_id={"start": 1},
                time_window=(-0.04, 0.04),
                sampling_rate=invalid_rate,
            )
