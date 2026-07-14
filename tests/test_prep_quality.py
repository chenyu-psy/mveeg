"""Tests for the 0.3 signal-quality stage boundaries."""

from __future__ import annotations

from types import SimpleNamespace
import warnings

import mne
import numpy as np
import pytest

from mveeg.prep.gaze import _degrees_to_pixels, normalize_gaze_geometry
from mveeg.prep.quality import (
    EligibilityResult,
    _hf_window_starts,
    apply_autoreject,
    check_eligibility,
    label_artifact_rules,
    load_quality_state,
    save_quality_state,
)


def _eligibility_state(
    epochs: mne.Epochs,
    eligible: np.ndarray | None = None,
) -> EligibilityResult:
    """Build a minimal, valid quality state for post-AutoReject rule tests."""

    if eligible is None:
        eligible = np.ones(len(epochs), dtype=bool)
    return EligibilityResult(
        eligible=np.asarray(eligible, dtype=bool),
        rule_masks={},
        channel_reasons=np.full(
            (len(epochs), len(epochs.ch_names)), "", dtype="<U256"
        ),
    )


def _hf_epochs(data: np.ndarray, *, sfreq: float = 100) -> mne.Epochs:
    """Create EEG epochs whose filtered values can be controlled in tests."""

    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError("data must have epoch, channel, and sample dimensions")
    info = mne.create_info(
        [f"E{i}" for i in range(data.shape[1])],
        sfreq=sfreq,
        ch_types="eeg",
    )
    events = np.column_stack(
        (
            np.arange(data.shape[0]) * (data.shape[2] + 1),
            np.zeros(data.shape[0], dtype=int),
            np.ones(data.shape[0], dtype=int),
        )
    )
    return mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"condition": 1},
        tmin=0,
        verbose="ERROR",
    )


def _hf_epochs_from_log_power(
    log_power: np.ndarray,
    *,
    n_times: int = 100,
    sfreq: float = 100,
) -> mne.Epochs:
    """Create full-epoch signals with an exact log-mean-square metric."""

    log_power = np.asarray(log_power, dtype=float)
    if log_power.ndim != 2:
        raise ValueError("log_power must have epoch and channel dimensions")
    alternating = np.where(np.arange(n_times) % 2, -1.0, 1.0)
    amplitudes = np.exp(log_power / 2)
    return _hf_epochs(amplitudes[:, :, np.newaxis] * alternating, sfreq=sfreq)


def _hf_config(**updates) -> dict[str, object]:
    config: dict[str, object] = {
        "band": (20.0, 40.0),
        "window_duration": 0.1,
        "z_threshold": 3.0,
        "min_noisy_fraction": 0.5,
        "bad_channels": 1,
    }
    config.update(updates)
    return config


def _quiet_review_config() -> dict[str, object]:
    return {
        "eeg": {
            "p2p": 1e6,
            "step": 1e6,
            "absolute_value": 1e6,
            "bad_channels": 1,
        }
    }


@pytest.fixture
def no_op_epoch_filter(monkeypatch):
    """Treat synthetic samples as filtered data and record filter reuse."""

    calls: list[tuple[float | None, float | None]] = []

    def passthrough(data, sfreq, l_freq, h_freq, *args, **kwargs):
        assert data.flags.c_contiguous
        assert data.flags.owndata
        calls.append((l_freq, h_freq))
        return data

    monkeypatch.setattr(mne.filter, "filter_data", passthrough)
    return calls


def _label_rules(
    epochs: mne.Epochs,
    *,
    eligibility: EligibilityResult | None = None,
    bad_epochs: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    reject_config: dict[str, object] | None = None,
    review_config: dict[str, object] | None = None,
    ignore_channels=(),
):
    """Call the public rule stage with an explicit AutoReject state."""

    n_eeg = sum(channel_type == "eeg" for channel_type in epochs.get_channel_types())
    if eligibility is None:
        eligibility = _eligibility_state(epochs)
    if bad_epochs is None:
        bad_epochs = np.zeros(len(epochs), dtype=bool)
    if labels is None:
        labels = np.full((len(epochs), n_eeg), -1, dtype=np.int8)
    return label_artifact_rules(
        epochs,
        eligibility,
        bad_epochs,
        autoreject_labels=labels,
        reject_config={} if reject_config is None else reject_config,
        review_config=(
            _quiet_review_config() if review_config is None else review_config
        ),
        ignore_channels=ignore_channels,
    )


def _epochs() -> mne.Epochs:
    rng = np.random.default_rng(4)
    data = rng.normal(scale=2e-6, size=(6, 6, 101))
    data[2, :5, 50] = 300e-6
    info = mne.create_info([f"E{i}" for i in range(6)], sfreq=100, ch_types="eeg")
    events = np.column_stack((np.arange(6) * 200, np.zeros(6, dtype=int), np.ones(6, dtype=int)))
    return mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"condition": 1},
        tmin=-0.2,
        verbose="ERROR",
    )


def _eligibility_config() -> dict[str, object]:
    return {
        "time_window": (-0.2, 0.8),
        "eeg": {
            "p2p": 150e-6,
            "step": 90e-6,
            "absolute_value": 150e-6,
            "bad_channels": 5,
        },
    }


def _gaze_geometry() -> dict[str, float | int]:
    return normalize_gaze_geometry(
        viewing_distance_cm=80,
        screen_width_cm=53.2,
        screen_width_px=1920,
    )


def _gaze_epochs() -> mne.Epochs:
    rng = np.random.default_rng(18)
    channel_names = [
        "Cz",
        "xpos_left",
        "xpos_right",
        "ypos_left",
        "ypos_right",
    ]
    channel_types = ["eeg", "eyegaze", "eyegaze", "eyegaze", "eyegaze"]
    data = rng.normal(scale=2e-6, size=(3, 5, 101))
    data[:, 1:, :] = 0
    data[1, 1:3, :] = 200
    info = mne.create_info(channel_names, sfreq=100, ch_types=channel_types)
    events = np.column_stack(
        (np.arange(3) * 200, np.zeros(3, dtype=int), np.ones(3, dtype=int))
    )
    return mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"condition": 1},
        tmin=-0.2,
        verbose="ERROR",
    )


def test_eligibility_marks_extreme_trial_without_dropping_it():
    epochs = _epochs()

    result = check_eligibility(epochs, _eligibility_config())

    assert result.eligible.tolist() == [True, True, False, True, True, True]
    assert result.channel_reasons.shape == (6, 6)
    assert "large_absolute_value" in result.channel_reasons[2, 0]
    assert len(epochs) == 6


def test_eligibility_rejects_label_only_ignore_channels_option():
    config = _eligibility_config()
    config["ignore_channels"] = ["E0"]

    with pytest.raises(ValueError, match="belongs to label_artifacts"):
        check_eligibility(_epochs(), config)


def test_disabled_autoreject_preserves_complete_epoch_order():
    epochs = _epochs()
    eligibility = check_eligibility(epochs, _eligibility_config())

    result = apply_autoreject(epochs, eligibility.eligible, {"enabled": False})

    np.testing.assert_allclose(result.epochs.get_data(), epochs.get_data())
    assert result.bad_epochs.tolist() == [False] * len(epochs)
    assert result.labels.shape == (len(epochs), 6)


def test_quality_state_round_trip_uses_no_object_arrays(tmp_path):
    epochs = _epochs()
    eligibility = check_eligibility(epochs, _eligibility_config())
    autoreject = apply_autoreject(epochs, eligibility.eligible, None)

    save_quality_state(tmp_path / "quality.npz", eligibility, autoreject)
    loaded_eligibility, loaded_autoreject = load_quality_state(tmp_path / "quality.npz")

    np.testing.assert_array_equal(loaded_eligibility.eligible, eligibility.eligible)
    np.testing.assert_array_equal(
        loaded_eligibility.channel_reasons,
        eligibility.channel_reasons,
    )
    np.testing.assert_array_equal(loaded_autoreject["bad_epochs"], autoreject.bad_epochs)
    assert loaded_autoreject["schema_version"] == 2
    with np.load(tmp_path / "quality.npz", allow_pickle=False) as saved:
        assert all(saved[key].dtype.kind != "O" for key in saved.files)


def test_autoreject_accepts_scalar_candidates_and_shuffled_cv(monkeypatch, tmp_path):
    """One synthetic fit covers config normalization and diagnostic persistence."""

    captured = {}

    class FakeAutoReject:
        """Return deterministic AutoReject-shaped state without fitting data."""

        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.picks = tuple(kwargs["picks"])
            self.consensus = kwargs.get("consensus", np.linspace(0, 1, 11))
            self.n_interpolate = kwargs["n_interpolate"]
            self.random_state = kwargs["random_state"]
            self.n_jobs = kwargs["n_jobs"]

        def fit_transform(self, epochs, *, return_log):
            assert return_log
            self.consensus_ = {"eeg": float(self.consensus[0])}
            self.n_interpolate_ = {"eeg": int(self.n_interpolate[0])}
            self.threshes_ = {channel: 100e-6 for channel in self.picks}
            self.loss_ = {
                "eeg": np.zeros(
                    (len(self.consensus), len(self.n_interpolate), 5), dtype=float
                )
            }
            labels = np.full((len(epochs), len(epochs.ch_names)), np.nan)
            for channel in self.picks:
                labels[:, epochs.ch_names.index(channel)] = 0
            log = SimpleNamespace(
                bad_epochs=np.zeros(len(epochs), dtype=bool),
                labels=labels,
            )
            return epochs.copy(), log

    monkeypatch.setattr("mveeg.prep.quality.AutoReject", FakeAutoReject)
    epochs = _epochs()
    eligibility = check_eligibility(epochs, _eligibility_config())
    result = apply_autoreject(
        epochs,
        eligibility.eligible,
        {
            "consensus": 0.2,
            "n_interpolate": 3,
            "cv": {"n_splits": 5, "shuffle": True, "random_state": 7},
            "random_state": 0,
            "exclude_channels": ["E0"],
        },
    )

    assert captured["consensus"] == [0.2]
    assert captured["n_interpolate"] == [3]
    assert captured["cv"].get_n_splits() == 5
    assert captured["cv"].shuffle
    assert captured["cv"].random_state == 7
    assert captured["picks"] == ["E1", "E2", "E3", "E4", "E5"]
    assert result.eeg_channels == ("E0", "E1", "E2", "E3", "E4", "E5")
    assert result.diagnostics["autoreject_channels"] == (
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
    )
    assert result.diagnostics["excluded_channels"] == ("E0",)
    assert result.diagnostics["info_bad_channels"] == ()
    assert (result.labels[eligibility.eligible, 0] == -1).all()
    assert result.diagnostics["consensus_"] == 0.2
    assert result.diagnostics["n_interpolate_"] == 3
    assert result.diagnostics["config_hash"]

    save_quality_state(tmp_path / "quality-v2.npz", eligibility, result)
    _, loaded = load_quality_state(tmp_path / "quality-v2.npz")
    assert loaded["consensus_"] == 0.2
    assert loaded["n_interpolate_"] == 3
    assert loaded["cv_config"] == {
        "kind": "KFold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": 7,
    }
    assert loaded["autoreject_channels"].tolist() == ["E1", "E2", "E3", "E4", "E5"]
    assert loaded["excluded_channels"].tolist() == ["E0"]
    assert loaded["info_bad_channels"].size == 0
    assert loaded["thresholds"] == {
        channel: 100e-6 for channel in result.diagnostics["autoreject_channels"]
    }
    assert loaded["versions"]["autoreject"]
    assert loaded["input_summary"]["n_eligible"] == int(eligibility.eligible.sum())
    assert loaded["input_summary"]["n_eeg_channels"] == 6
    assert loaded["input_summary"]["n_autoreject_channels"] == 5

    apply_autoreject(
        epochs,
        eligibility.eligible,
        {
            "consensus": [0.2, 0.3],
            "n_interpolate": [2, 3],
            "cv": 5,
        },
    )
    assert captured["consensus"] == [0.2, 0.3]
    assert captured["n_interpolate"] == [2, 3]
    assert captured["cv"] == 5
    assert captured["picks"] == list(result.eeg_channels)


def test_autoreject_exclude_channels_validates_configuration():
    """Invalid exclusions fail before an expensive AutoReject fit starts."""

    epochs = _epochs()
    eligible = np.ones(len(epochs), dtype=bool)

    with pytest.raises(ValueError, match="use autoreject.exclude_channels"):
        apply_autoreject(epochs, eligible, {"ignore_channels": ["E0"]})
    with pytest.raises(TypeError, match="sequence of channel names"):
        apply_autoreject(epochs, eligible, {"exclude_channels": "E0"})
    with pytest.raises(ValueError, match="duplicate"):
        apply_autoreject(epochs, eligible, {"exclude_channels": ["E0", "E0"]})
    with pytest.raises(ValueError, match="unknown channel"):
        apply_autoreject(epochs, eligible, {"exclude_channels": ["missing"]})
    with pytest.raises(ValueError, match="leaves no EEG channels"):
        apply_autoreject(
            epochs,
            eligible,
            {"exclude_channels": list(epochs.ch_names)},
        )

    epochs_with_eog = _epochs()
    epochs_with_eog.set_channel_types({"E5": "eog"})
    with pytest.raises(ValueError, match="only supports EEG channels"):
        apply_autoreject(
            epochs_with_eog,
            eligible,
            {"exclude_channels": ["E5"]},
        )


def test_autoreject_missing_consensus_warns_and_uses_default_grid(monkeypatch):
    """Legacy configs remain valid but make their implicit grid visible."""

    class FakeAutoReject:
        """Expose the default consensus grid used by AutoReject."""

        def __init__(self, **kwargs):
            assert "consensus" not in kwargs
            self.consensus = np.linspace(0, 1, 11)
            self.n_interpolate = kwargs["n_interpolate"]
            self.random_state = kwargs["random_state"]
            self.n_jobs = kwargs["n_jobs"]

        def fit_transform(self, epochs, *, return_log):
            self.consensus_ = {"eeg": 0.1}
            self.n_interpolate_ = {"eeg": 1}
            self.threshes_ = {channel: 100e-6 for channel in epochs.ch_names}
            self.loss_ = {"eeg": np.zeros((11, 1, 2))}
            return epochs.copy(), SimpleNamespace(
                bad_epochs=np.zeros(len(epochs), dtype=bool),
                labels=np.zeros((len(epochs), len(epochs.ch_names))),
            )

    monkeypatch.setattr("mveeg.prep.quality.AutoReject", FakeAutoReject)
    epochs = _epochs()
    with pytest.warns(UserWarning, match="consensus is not set"):
        result = apply_autoreject(
            epochs,
            np.ones(len(epochs), dtype=bool),
            {"n_interpolate": 1, "cv": 2},
        )

    np.testing.assert_allclose(result.diagnostics["consensus_grid"], np.linspace(0, 1, 11))


def test_legacy_quality_state_loads_with_empty_v2_diagnostics(tmp_path):
    """Schema-v1 NPZ files remain readable without fabricated diagnostics."""

    path = tmp_path / "legacy-quality.npz"
    np.savez_compressed(
        path,
        eligible=np.array([True, False]),
        eligibility_channel_reasons=np.full((2, 1), "", dtype="<U1"),
        autoreject_bad_epochs=np.array([False, True]),
        autoreject_interpolated_channels=np.array([0, 0]),
        autoreject_labels=np.array([[0], [1]], dtype=np.int8),
        autoreject_eeg_channels=np.array(["Cz"]),
    )

    _, loaded = load_quality_state(path)

    assert loaded["schema_version"] == 1
    assert loaded["consensus_"] is None
    assert loaded["thresholds"] == {}
    assert loaded["cv_config"] == {}
    assert loaded["autoreject_channels"].size == 0
    assert loaded["excluded_channels"].size == 0
    assert loaded["info_bad_channels"].size == 0


def test_actual_autoreject_keeps_full_order_and_stores_only_eeg_labels():
    rng = np.random.default_rng(8)
    channel_names = ["Fz", "Cz", "Pz", "Oz", "EOG"]
    info = mne.create_info(channel_names, 100, ["eeg"] * 4 + ["eog"])
    info.set_montage(
        mne.channels.make_standard_montage("standard_1020"),
        on_missing="ignore",
    )
    info["bads"] = ["Oz"]
    data = rng.normal(scale=2e-6, size=(20, 5, 80))
    events = np.column_stack(
        (np.arange(20) * 100, np.zeros(20, dtype=int), np.ones(20, dtype=int))
    )
    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"condition": 1},
        tmin=-0.2,
        verbose="ERROR",
    )
    eligible = np.ones(len(epochs), dtype=bool)
    eligible[3] = False

    result = apply_autoreject(
        epochs,
        eligible,
        {
            "enabled": True,
            "consensus": 1.0,
            "n_interpolate": 1,
            "cv": {"n_splits": 2, "shuffle": True, "random_state": 0},
            "n_jobs": 1,
            "random_state": 0,
            "verbose": False,
            "exclude_channels": ["Pz"],
        },
    )

    assert len(result.epochs) == len(epochs)
    assert result.eeg_channels == ("Fz", "Cz", "Pz", "Oz")
    assert result.labels.shape == (len(epochs), 4)
    assert result.labels[3].tolist() == [-1, -1, -1, -1]
    assert result.diagnostics["autoreject_channels"] == ("Fz", "Cz")
    assert result.diagnostics["excluded_channels"] == ("Pz",)
    assert result.diagnostics["info_bad_channels"] == ("Oz",)
    assert (result.labels[eligible, 2] == -1).all()
    assert (result.labels[eligible, 3] == -1).all()
    assert np.isin(result.labels[eligible, :2], [0, 1, 2]).all()
    np.testing.assert_allclose(
        result.epochs.get_data()[:, 2:4],
        epochs.get_data()[:, 2:4],
    )
    np.testing.assert_array_equal(result.epochs.events, epochs.events)


def test_ignore_channels_excludes_review_contribution_but_keeps_reason():
    epochs = _epochs()
    epochs._data[0, 0, 50] = 130e-6
    eligibility = check_eligibility(epochs, _eligibility_config())
    review = {
        "time_window": (-0.2, 0.8),
        "eeg": {
            "p2p": 100e-6,
            "step": 1,
            "absolute_value": 100e-6,
            "bad_channels": 1,
        },
    }

    result = label_artifact_rules(
        epochs,
        eligibility,
        np.zeros(len(epochs), dtype=bool),
        autoreject_labels=np.full((len(epochs), 6), -1, dtype=np.int8),
        reject_config={},
        review_config=review,
        ignore_channels=["E0"],
    )

    assert not result.epoch_review[0]
    assert "moderate_absolute_value" in result.review_reasons[0, 0]


def test_hf_noise_requires_sustained_windows_and_bad_channel_boundary(
    no_op_epoch_filter,
):
    rng = np.random.default_rng(31)
    # Match the 3.25-second CL41 decision window: a 100-ms burst should not be
    # equivalent to high-frequency activity sustained through most of a trial.
    data = rng.normal(scale=1e-6, size=(12, 3, 325))
    carrier = np.where(np.arange(325) % 2, -2e-4, 2e-4)
    data[10, :2, 145:155] += carrier[145:155]
    data[11, :2, 100:300] += carrier[100:300]
    epochs = _hf_epochs(data)
    rule = _hf_config(bad_channels=2)

    result = _label_rules(epochs, reject_config={"hf_noise": rule})

    assert "high_frequency_noise" not in result.rejected_reasons[10, 0]
    assert "high_frequency_noise" in result.rejected_reasons[11, 0]
    assert "high_frequency_noise" in result.rejected_reasons[11, 1]
    assert result.epoch_rejected[[10, 11]].tolist() == [False, True]

    unreachable = _label_rules(
        epochs,
        reject_config={"hf_noise": _hf_config(bad_channels=3)},
    )
    assert not unreachable.epoch_rejected[11]
    assert "high_frequency_noise" in unreachable.rejected_reasons[11, 0]


def test_hf_windows_use_half_overlap_and_include_the_final_complete_window():
    assert _hf_window_starts(n_times=11, window=4).tolist() == [0, 2, 4, 6, 7]


def test_hf_measurement_filter_never_changes_saved_epoch_signal(monkeypatch):
    rng = np.random.default_rng(38)
    epochs = _hf_epochs(rng.normal(scale=1e-6, size=(6, 2, 100)))
    original = epochs.get_data(copy=True)

    def overwrite_measurement_copy(data, *args, **kwargs):
        data[:] = 0
        return data

    monkeypatch.setattr(mne.filter, "filter_data", overwrite_measurement_copy)
    with pytest.raises(ValueError, match="scale"):
        _label_rules(epochs, reject_config={"hf_noise": _hf_config()})

    np.testing.assert_array_equal(epochs.get_data(copy=True), original)


def test_hf_z_threshold_is_inclusive(no_op_epoch_filter):
    log_power = np.asarray([[-24.0], [-22.0], [-23.0 + 3 * 1.4826]])
    epochs = _hf_epochs_from_log_power(log_power)
    labels = np.asarray([[0], [0], [2]], dtype=np.int8)
    data = epochs.get_data(copy=True)
    data = data[:, :, np.ones(data.shape[2], dtype=bool)]
    powers = np.stack([np.nanmean(np.square(data), axis=2)], axis=2)
    metric = np.log(np.maximum(powers, np.finfo(float).tiny))[:, 0, 0]
    center = float(np.median(metric[:2]))
    scale = 1.4826 * float(np.median(np.abs(metric[:2] - center)))
    boundary = float((metric[2] - center) / scale)
    base = _hf_config(
        window_duration=1.0,
        z_threshold=boundary,
        min_noisy_fraction=1.0,
    )

    included = _label_rules(
        epochs,
        labels=labels,
        reject_config={"hf_noise": base},
    )
    excluded = _label_rules(
        epochs,
        labels=labels,
        reject_config={
            "hf_noise": {
                **base,
                "z_threshold": np.nextafter(boundary, np.inf),
            }
        },
    )

    assert "high_frequency_noise" in included.rejected_reasons[2, 0]
    assert "high_frequency_noise" not in excluded.rejected_reasons[2, 0]


def test_hf_noisy_window_fraction_is_inclusive(no_op_epoch_filter):
    data = np.empty((3, 1, 6), dtype=float)
    alternating = np.where(np.arange(6) % 2, -1.0, 1.0)
    data[0, 0] = np.exp(-24 / 2) * alternating
    data[1, 0] = np.exp(-22 / 2) * alternating
    data[2, 0] = 0
    data[2, 0, :2] = np.sqrt(2 * np.exp(-17)) * alternating[:2]
    data[2, 0, 4:] = np.sqrt(2 * np.exp(-23)) * alternating[4:]
    epochs = _hf_epochs(data)
    labels = np.asarray([[0], [0], [2]], dtype=np.int8)
    rule = _hf_config(
        window_duration=0.04,
        z_threshold=3.0,
        min_noisy_fraction=0.5,
    )

    included = _label_rules(
        epochs,
        labels=labels,
        reject_config={"hf_noise": rule},
    )
    excluded = _label_rules(
        epochs,
        labels=labels,
        reject_config={
            "hf_noise": {
                **rule,
                "min_noisy_fraction": np.nextafter(0.5, 1.0),
            }
        },
    )

    assert "high_frequency_noise" in included.rejected_reasons[2, 0]
    assert "high_frequency_noise" not in excluded.rejected_reasons[2, 0]


@pytest.mark.parametrize(
    ("excluded_by", "target_is_noisy"),
    [
        ("eligibility", True),
        ("bad_epoch", True),
        ("channel_bad", True),
        ("interpolation", True),
        ("disabled_autoreject", False),
    ],
)
def test_hf_reference_exclusions(
    no_op_epoch_filter,
    excluded_by,
    target_is_noisy,
):
    epochs = _hf_epochs_from_log_power(
        -24 + np.asarray([[0.0], [2.0], [20.0], [12.0]])
    )
    eligible = np.ones(len(epochs), dtype=bool)
    bad_epochs = np.zeros(len(epochs), dtype=bool)
    labels = np.zeros((len(epochs), 1), dtype=np.int8)
    if excluded_by == "eligibility":
        eligible[2] = False
    elif excluded_by == "bad_epoch":
        bad_epochs[2] = True
    elif excluded_by == "channel_bad":
        labels[2, 0] = 1
    elif excluded_by == "interpolation":
        labels[2, 0] = 2
    else:
        labels.fill(-1)

    result = _label_rules(
        epochs,
        eligibility=_eligibility_state(epochs, eligible),
        bad_epochs=bad_epochs,
        labels=labels,
        reject_config={
            "hf_noise": _hf_config(
                window_duration=1.0,
                min_noisy_fraction=1.0,
            )
        },
    )

    assert (
        "high_frequency_noise" in result.rejected_reasons[3, 0]
    ) is target_is_noisy


def test_hf_info_bads_are_not_scored_but_ignored_channels_keep_reason(
    no_op_epoch_filter,
):
    epochs = _hf_epochs_from_log_power(
        np.asarray([[-24.0, -24.0], [-22.0, -22.0], [-12.0, -12.0]])
    )
    epochs.info["bads"] = ["E1"]
    result = _label_rules(
        epochs,
        labels=np.zeros((len(epochs), 2), dtype=np.int8),
        reject_config={
            "hf_noise": _hf_config(
                window_duration=1.0,
                min_noisy_fraction=1.0,
            )
        },
        ignore_channels=["E0"],
    )

    assert "high_frequency_noise" in result.rejected_reasons[2, 0]
    assert result.rejected_reasons[2, 1] == ""
    assert not result.epoch_rejected[2]


def test_hf_requires_two_reference_epochs_per_channel(no_op_epoch_filter):
    epochs = _hf_epochs_from_log_power(np.asarray([[-24.0], [-22.0], [-12.0]]))
    eligibility = _eligibility_state(
        epochs,
        np.asarray([True, False, False]),
    )

    with pytest.raises(ValueError, match="reference"):
        _label_rules(
            epochs,
            eligibility=eligibility,
            reject_config={
                "hf_noise": _hf_config(
                    window_duration=1.0,
                    min_noisy_fraction=1.0,
                )
            },
        )


def test_hf_rejects_zero_scaled_mad(no_op_epoch_filter):
    epochs = _hf_epochs_from_log_power(np.full((3, 1), -24.0))

    with pytest.raises(ValueError, match="MAD|scale"):
        _label_rules(
            epochs,
            reject_config={
                "hf_noise": _hf_config(
                    window_duration=1.0,
                    min_noisy_fraction=1.0,
                )
            },
        )


def test_hf_filter_is_reused_for_equal_bands(no_op_epoch_filter):
    rng = np.random.default_rng(32)
    epochs = _hf_epochs(rng.normal(scale=1e-6, size=(8, 2, 100)))
    hard = _hf_config(z_threshold=100)
    review = {**_quiet_review_config(), "hf_noise": _hf_config(z_threshold=100)}

    _label_rules(
        epochs,
        reject_config={"hf_noise": hard},
        review_config=review,
    )
    assert no_op_epoch_filter == [(20.0, 40.0)]

    no_op_epoch_filter.clear()
    review["hf_noise"] = _hf_config(band=(15.0, 35.0), z_threshold=100)
    _label_rules(
        epochs,
        reject_config={"hf_noise": hard},
        review_config=review,
    )
    assert no_op_epoch_filter == [(20.0, 40.0), (15.0, 35.0)]


def test_hf_stage_windows_reasons_and_status_precedence(no_op_epoch_filter):
    rng = np.random.default_rng(33)
    data = rng.normal(scale=1e-6, size=(10, 1, 100))
    data[9, 0, 50:] += np.where(np.arange(50) % 2, -2e-4, 2e-4)
    epochs = _hf_epochs(data)
    rule = _hf_config(min_noisy_fraction=0.5)
    review = {
        **_quiet_review_config(),
        "time_window": (0.5, 0.99),
        "hf_noise": rule,
    }

    review_only = _label_rules(
        epochs,
        reject_config={"time_window": (0.0, 0.49), "hf_noise": rule},
        review_config=review,
    )
    assert review_only.rejected_reasons[9, 0] == ""
    assert "moderate_high_frequency_noise" in review_only.review_reasons[9, 0]
    assert not review_only.epoch_rejected[9]
    assert review_only.epoch_review[9]

    rejected = _label_rules(
        epochs,
        reject_config={"time_window": (0.5, 0.99), "hf_noise": rule},
        review_config=review,
    )
    assert "high_frequency_noise" in rejected.rejected_reasons[9, 0]
    assert rejected.review_reasons[9, 0] == ""
    assert rejected.epoch_rejected[9]
    assert not rejected.epoch_review[9]


@pytest.mark.parametrize(
    "missing",
    [
        "band",
        "window_duration",
        "z_threshold",
        "min_noisy_fraction",
        "bad_channels",
    ],
)
def test_hf_config_requires_every_new_key(no_op_epoch_filter, missing):
    epochs = _hf_epochs(np.random.default_rng(34).normal(size=(4, 1, 100)))
    rule = _hf_config()
    rule.pop(missing)

    with pytest.raises(ValueError, match=missing):
        _label_rules(epochs, reject_config={"hf_noise": rule})


def test_hf_config_rejects_legacy_mad_multiplier(no_op_epoch_filter):
    epochs = _hf_epochs(np.random.default_rng(35).normal(size=(4, 1, 100)))
    rule = {**_hf_config(), "mad_multiplier": 5}

    with pytest.raises(ValueError, match="mad_multiplier"):
        _label_rules(epochs, reject_config={"hf_noise": rule})


def test_hf_config_must_be_a_mapping(no_op_epoch_filter):
    epochs = _hf_epochs(np.random.default_rng(35).normal(size=(4, 1, 100)))

    with pytest.raises(TypeError, match="mapping"):
        _label_rules(epochs, reject_config={"hf_noise": []})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("band", ()),
        ("band", (20.0,)),
        ("band", (0.0, 40.0)),
        ("band", (20.0, 20.0)),
        ("band", (40.0, 20.0)),
        ("band", (20.0, 50.0)),
        ("band", (np.nan, 40.0)),
        ("window_duration", 0),
        ("window_duration", -0.1),
        ("window_duration", 1.01),
        ("z_threshold", 0),
        ("z_threshold", np.inf),
        ("min_noisy_fraction", 0),
        ("min_noisy_fraction", 1.01),
        ("bad_channels", 0),
        ("bad_channels", 1.0),
        ("bad_channels", True),
    ],
)
def test_hf_config_rejects_invalid_values(no_op_epoch_filter, key, value):
    epochs = _hf_epochs(np.random.default_rng(36).normal(size=(4, 2, 100)))

    with pytest.raises(ValueError):
        _label_rules(
            epochs,
            reject_config={"hf_noise": _hf_config(**{key: value})},
        )


def test_hf_bad_channels_may_be_intentionally_unreachable(no_op_epoch_filter):
    epochs = _hf_epochs(np.random.default_rng(37).normal(size=(4, 1, 100)))

    result = _label_rules(
        epochs,
        reject_config={"hf_noise": _hf_config(bad_channels=99)},
    )

    assert not result.epoch_rejected.any()


def test_gaze_geometry_and_degree_conversion_are_strict_and_explicit():
    geometry = _gaze_geometry()

    assert geometry == {
        "viewing_distance_cm": 80.0,
        "screen_width_cm": 53.2,
        "screen_width_px": 1920,
    }
    assert [_degrees_to_pixels(value, geometry) for value in (1.25, 1.0, 0.75)] == [
        63,
        50,
        38,
    ]
    with pytest.raises(TypeError, match="unexpected keyword"):
        normalize_gaze_geometry(
            distance_mm=800,
            width_mm=532,
            resolution_x=1920,
        )
    with pytest.raises(ValueError, match="positive integer"):
        normalize_gaze_geometry(
            viewing_distance_cm=80,
            screen_width_cm=53.2,
            screen_width_px=1920.5,
        )


def test_eligibility_gaze_uses_independent_rules_and_provenance_geometry():
    epochs = _gaze_epochs()
    config = _eligibility_config()
    config["gaze"] = {"deviation_deg": 1.25, "shift_deg": 0.75}

    result = check_eligibility(epochs, config, gaze_geometry=_gaze_geometry())

    assert result.eligible.tolist() == [True, False, True]
    assert "large_gaze_deviation" in result.channel_reasons[1, 1]
    with pytest.raises(ValueError, match="requires gaze_geometry"):
        check_eligibility(epochs, config)
    with pytest.raises(ValueError, match="only supports"):
        check_eligibility(
            epochs,
            {**_eligibility_config(), "gaze": {"deviation": 1.25, "shift": 0.75}},
            gaze_geometry=_gaze_geometry(),
        )
    with pytest.raises(ValueError, match="comes from dataset provenance"):
        check_eligibility(
            epochs,
            {**config, "screen": {"distance_mm": 800}},
            gaze_geometry=_gaze_geometry(),
        )


def test_gaze_missing_is_explicit_binocular_and_stage_specific():
    epochs = _gaze_epochs()
    epochs._data[:, 1:, :] = 0
    epochs._data[0, [1, 3], :] = np.nan
    epochs._data[1, 1:5, :11] = np.nan
    epochs._data[2, 1:5, :10] = np.nan
    threshold = 10 / len(epochs.times)

    baseline = check_eligibility(epochs, _eligibility_config())

    assert baseline.eligible.tolist() == [True, True, True]
    assert not baseline.rule_masks["dropout"][:, 1:].any()

    hard_config = {
        **_eligibility_config(),
        "gaze": {"max_missing_fraction": threshold},
    }
    hard = check_eligibility(epochs, hard_config)

    assert hard.eligible.tolist() == [True, False, True]
    assert hard.channel_reasons[1, 1:5].tolist() == ["gaze_missing"] * 4

    review = label_artifact_rules(
        epochs,
        baseline,
        np.zeros(len(epochs), dtype=bool),
        autoreject_labels=np.full((len(epochs), 1), -1, dtype=np.int8),
        reject_config={},
        review_config={
            **_quiet_review_config(),
            "gaze": {"max_missing_fraction": threshold},
        },
    )

    assert review.epoch_review.tolist() == [False, True, False]
    assert review.review_reasons[1, 1:5].tolist() == ["gaze_missing"] * 4


@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, True])
def test_gaze_missing_fraction_is_validated(value):
    with pytest.raises(ValueError, match="between 0 and 1"):
        check_eligibility(
            _gaze_epochs(),
            {
                **_eligibility_config(),
                "gaze": {"max_missing_fraction": value},
            },
        )


def test_gaze_missing_requires_a_complete_eye_pair():
    epochs = _gaze_epochs().drop_channels(["ypos_left", "ypos_right"])

    with pytest.raises(ValueError, match="complete xpos/ypos pairs"):
        check_eligibility(
            epochs,
            {
                **_eligibility_config(),
                "gaze": {"max_missing_fraction": 0.1},
            },
        )

    with pytest.raises(ValueError, match="at least one gaze rule"):
        check_eligibility(
            _gaze_epochs(),
            {**_eligibility_config(), "gaze": {}},
        )


def test_all_nan_gaze_reductions_do_not_warn():
    epochs = _gaze_epochs()
    epochs._data[0, 1:, :] = np.nan
    gaze_rule = {"deviation_deg": 1.0, "shift_deg": 0.75}

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        eligibility = check_eligibility(
            epochs,
            {**_eligibility_config(), "gaze": gaze_rule},
            gaze_geometry=_gaze_geometry(),
        )
        result = label_artifact_rules(
            epochs,
            eligibility,
            np.zeros(len(epochs), dtype=bool),
            autoreject_labels=np.full((len(epochs), 1), -1, dtype=np.int8),
            reject_config={},
            review_config={**_quiet_review_config(), "gaze": gaze_rule},
            gaze_geometry=_gaze_geometry(),
        )

    assert eligibility.eligible[0]
    assert not result.epoch_review[0]


def test_gaze_rules_require_eyegaze_channels_and_reject_stage_is_unsupported():
    gaze_rule = {"deviation_deg": 1.0, "shift_deg": 0.75}
    with pytest.raises(ValueError, match="at least one eyegaze channel"):
        check_eligibility(
            _epochs(),
            {**_eligibility_config(), "gaze": gaze_rule},
            gaze_geometry=_gaze_geometry(),
        )

    gaze_epochs = _gaze_epochs()
    eligibility = check_eligibility(gaze_epochs, _eligibility_config())
    with pytest.raises(ValueError, match="reject.gaze is unsupported"):
        label_artifact_rules(
            gaze_epochs,
            eligibility,
            np.zeros(len(gaze_epochs), dtype=bool),
            autoreject_labels=np.full(
                (len(gaze_epochs), 1), -1, dtype=np.int8
            ),
            reject_config={"gaze": gaze_rule},
            review_config={
                "eeg": {
                    "p2p": 1,
                    "step": 1,
                    "absolute_value": 1,
                    "bad_channels": 1,
                }
            },
            gaze_geometry=_gaze_geometry(),
        )


def test_review_gaze_rule_labels_deviation_after_autoreject():
    epochs = _gaze_epochs()
    eligibility = check_eligibility(epochs, _eligibility_config())

    result = label_artifact_rules(
        epochs,
        eligibility,
        np.zeros(len(epochs), dtype=bool),
        autoreject_labels=np.full((len(epochs), 1), -1, dtype=np.int8),
        reject_config={},
        review_config={
            "time_window": (-0.2, 0.8),
            "eeg": {
                "p2p": 1,
                "step": 1,
                "absolute_value": 1,
                "bad_channels": 1,
            },
            "gaze": {"deviation_deg": 1.0, "shift_deg": 0.75},
        },
        gaze_geometry=_gaze_geometry(),
    )

    assert result.epoch_review.tolist() == [False, True, False]
    assert "moderate_gaze_deviation" in result.review_reasons[1, 1]
