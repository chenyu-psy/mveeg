"""Signal-quality steps used by the 0.3 preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping, Sequence
import warnings

import autoreject as autoreject_package
import mne
import numpy as np
from autoreject import AutoReject
from sklearn.model_selection import KFold

from .gaze import _degrees_to_pixels, normalize_gaze_geometry


@dataclass(frozen=True)
class EligibilityResult:
    """Eligibility decisions and their channel-level reasons."""

    eligible: np.ndarray
    rule_masks: dict[str, np.ndarray]
    channel_reasons: np.ndarray


@dataclass(frozen=True)
class AutorejectResult:
    """AutoReject output expanded back to the complete epoch order."""

    epochs: mne.Epochs
    bad_epochs: np.ndarray
    interpolated_channels: np.ndarray
    labels: np.ndarray
    eeg_channels: tuple[str, ...]
    diagnostics: Mapping[str, object] | None = None


@dataclass(frozen=True)
class ArtifactRuleResult:
    """Automatic reject/review masks produced after signal preprocessing."""

    rejected_reasons: np.ndarray
    review_reasons: np.ndarray
    epoch_rejected: np.ndarray
    epoch_review: np.ndarray


_ELIGIBILITY_CODES = {
    "dropout": "dropout",
    "flatline": "flatline",
    "gaze_missing": "gaze_missing",
    "large_absolute_value": "large_absolute_value",
    "large_gaze_deviation": "large_gaze_deviation",
    "large_gaze_shift": "large_gaze_shift",
    "large_step": "large_step",
    "large_peak_to_peak": "large_peak_to_peak",
}

_QUALITY_SCHEMA_VERSION = 2

_AUTOREJECT_CONFIG_KEYS = {
    "consensus",
    "cv",
    "enabled",
    "exclude_channels",
    "n_interpolate",
    "n_jobs",
    "random_state",
    "verbose",
}


def _candidate_values(
    value: object,
    *,
    name: str,
    integer: bool,
) -> list[float] | list[int]:
    """Normalize one AutoReject scalar or candidate sequence."""

    if isinstance(value, (str, bytes, bytearray, bool, np.bool_)):
        raise TypeError(f"autoreject.{name} must be a number or a sequence of numbers.")
    try:
        values = np.atleast_1d(value).tolist()
    except Exception as error:
        raise TypeError(
            f"autoreject.{name} must be a number or a sequence of numbers."
        ) from error
    if not values:
        raise ValueError(f"autoreject.{name} cannot be empty.")

    normalized: list[float] | list[int]
    if integer:
        normalized = []
        for item in values:
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral):
                raise TypeError(f"autoreject.{name} values must be integers.")
            item = int(item)
            if item < 0:
                raise ValueError(f"autoreject.{name} values must be nonnegative.")
            normalized.append(item)
        return normalized

    normalized = []
    for item in values:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
            raise TypeError(f"autoreject.{name} values must be numbers.")
        item = float(item)
        if not np.isfinite(item) or not 0 <= item <= 1:
            raise ValueError(f"autoreject.{name} values must be between 0 and 1.")
        normalized.append(item)
    return normalized


def _cv_config(value: object) -> tuple[int | KFold, dict[str, object]]:
    """Normalize an integer or mapping into an AutoReject CV specification."""

    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        n_splits = int(value)
        if n_splits < 2:
            raise ValueError("autoreject.cv must use at least two folds.")
        return n_splits, {
            "kind": "KFold",
            "n_splits": n_splits,
            "shuffle": False,
            "random_state": None,
        }
    if not isinstance(value, Mapping):
        raise TypeError("autoreject.cv must be an integer or a mapping.")

    extra = sorted(set(value).difference({"n_splits", "shuffle", "random_state"}))
    if extra:
        raise ValueError(f"autoreject.cv contains unsupported keys: {extra}.")
    n_splits = value.get("n_splits", 10)
    if isinstance(n_splits, (bool, np.bool_)) or not isinstance(n_splits, Integral):
        raise TypeError("autoreject.cv.n_splits must be an integer.")
    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError("autoreject.cv.n_splits must be at least two.")
    shuffle = value.get("shuffle", False)
    if not isinstance(shuffle, (bool, np.bool_)):
        raise TypeError("autoreject.cv.shuffle must be boolean.")
    shuffle = bool(shuffle)
    random_state = value.get("random_state")
    if isinstance(random_state, (bool, np.bool_)) or (
        random_state is not None and not isinstance(random_state, Integral)
    ):
        raise TypeError("autoreject.cv.random_state must be an integer or None.")
    random_state = None if random_state is None else int(random_state)
    if not shuffle and random_state is not None:
        raise ValueError("autoreject.cv.random_state requires shuffle=True.")
    resolved = {
        "kind": "KFold",
        "n_splits": n_splits,
        "shuffle": shuffle,
        "random_state": random_state,
    }
    return (
        KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state),
        resolved,
    )


def _autoreject_excluded_channels(
    config: Mapping[str, object],
    epochs: mne.Epochs,
    eeg_channels: Sequence[str],
) -> tuple[str, ...]:
    """Validate and return EEG channels excluded from AutoReject."""

    extra = sorted(set(config).difference(_AUTOREJECT_CONFIG_KEYS))
    if "ignore_channels" in extra:
        raise ValueError(
            "autoreject.ignore_channels is unsupported; use "
            "autoreject.exclude_channels instead."
        )
    if extra:
        raise ValueError(f"autoreject contains unsupported keys: {extra}.")

    value = config.get("exclude_channels", ())
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("autoreject.exclude_channels must be a sequence of channel names.")
    excluded = tuple(value)
    if any(not isinstance(channel, str) for channel in excluded):
        raise TypeError("autoreject.exclude_channels values must be channel names.")
    if len(set(excluded)) != len(excluded):
        raise ValueError("autoreject.exclude_channels cannot contain duplicates.")
    unknown = sorted(set(excluded).difference(epochs.ch_names))
    if unknown:
        raise ValueError(f"autoreject.exclude_channels contains unknown channels: {unknown}.")
    non_eeg = sorted(set(excluded).difference(eeg_channels))
    if non_eeg:
        raise ValueError(
            "autoreject.exclude_channels only supports EEG channels; "
            f"received: {non_eeg}."
        )
    return excluded


def _installed_version(package: str) -> str:
    """Return one installed package version without importing mveeg recursively."""

    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _input_summary(
    epochs: mne.Epochs,
    eligible: np.ndarray,
    eeg_channels: Sequence[str],
) -> dict[str, object]:
    """Build a compact identity summary for one AutoReject input."""

    return {
        "n_epochs": len(epochs),
        "n_eligible": int(np.sum(eligible)),
        "n_eeg_channels": len(eeg_channels),
        "sampling_rate": float(epochs.info["sfreq"]),
        "events_hash": sha256(
            np.ascontiguousarray(epochs.events).tobytes()
        ).hexdigest(),
        "eligible_mask_hash": sha256(
            np.ascontiguousarray(eligible).tobytes()
        ).hexdigest(),
    }


def _config_hash(config: Mapping[str, object]) -> str:
    """Hash a resolved, JSON-safe AutoReject configuration."""

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _base_autoreject_diagnostics(
    epochs: mne.Epochs,
    eligible: np.ndarray,
    eeg_channels: Sequence[str],
    autoreject_channels: Sequence[str] = (),
    excluded_channels: Sequence[str] = (),
    info_bad_channels: Sequence[str] = (),
) -> dict[str, object]:
    """Create diagnostics shared by enabled and disabled AutoReject runs."""

    input_summary = _input_summary(epochs, eligible, eeg_channels)
    input_summary["n_autoreject_channels"] = len(autoreject_channels)

    return {
        "consensus_": None,
        "n_interpolate_": None,
        "consensus_grid": np.asarray([], dtype=float),
        "n_interpolate_grid": np.asarray([], dtype=int),
        "threshold_values": np.asarray([], dtype=float),
        "loss": np.empty((0, 0, 0), dtype=float),
        "cv_config": {},
        "config_hash": "",
        "autoreject_channels": tuple(autoreject_channels),
        "excluded_channels": tuple(excluded_channels),
        "info_bad_channels": tuple(info_bad_channels),
        "versions": {
            "mveeg": _installed_version("mveeg"),
            "mne": str(mne.__version__),
            "autoreject": str(autoreject_package.__version__),
        },
        "input_summary": input_summary,
    }


def _window_data(
    epochs: mne.Epochs,
    time_window: tuple[float | None, float | None] | None,
) -> np.ndarray:
    data = epochs.get_data(copy=True)
    if time_window is None:
        return data
    tmin, tmax = time_window
    tmin = epochs.times[0] if tmin is None else tmin
    tmax = epochs.times[-1] if tmax is None else tmax
    keep = (epochs.times >= tmin) & (epochs.times <= tmax)
    if not np.any(keep):
        raise ValueError(f"time_window {time_window!r} does not overlap the epochs.")
    return data[:, :, keep]


def _match_both_eyes(epochs: mne.Epochs, mask: np.ndarray) -> np.ndarray:
    output = mask.copy()
    names = np.asarray(epochs.ch_names, dtype=object)
    required = {"xpos_left", "xpos_right", "ypos_left", "ypos_right"}
    if not required.issubset(names):
        return output
    for prefix in ("xpos", "ypos"):
        picks = np.flatnonzero([str(name).startswith(prefix) for name in names])
        output[:, picks] = output[:, picks].all(axis=1, keepdims=True)
    return output


def _channel_type_mask(epochs: mne.Epochs, channel_type: str) -> np.ndarray:
    return np.asarray(epochs.get_channel_types(), dtype=object) == channel_type


def _window_starts(n_times: int, window: int, step: int) -> np.ndarray:
    if window >= n_times:
        return np.asarray([0], dtype=int)
    starts = np.arange(0, n_times - window + 1, max(step, 1), dtype=int)
    return starts if len(starts) else np.asarray([0], dtype=int)


def _milliseconds_to_samples(epochs: mne.Epochs, duration_ms: float) -> int:
    return max(1, int(np.floor(duration_ms * epochs.info["sfreq"] / 1000)))


def _dropout_mask(epochs: mne.Epochs, time_window) -> np.ndarray:
    """Mark EEG channels containing at least one missing sample."""

    data = _window_data(epochs, time_window)
    mask = np.zeros(data.shape[:2], dtype=bool)
    eeg = _channel_type_mask(epochs, "eeg")
    if np.any(eeg):
        mask[:, eeg] = np.any(np.isnan(data[:, eeg]), axis=2)
    return mask


def _gaze_missing_mask(
    epochs: mne.Epochs,
    threshold: float,
    *,
    time_window,
    context: str,
) -> np.ndarray:
    """Mark trials whose binocular gaze-missing fraction exceeds a threshold."""

    names = {name: index for index, name in enumerate(epochs.ch_names)}
    eye_pairs: list[tuple[int, int]] = []
    incomplete: list[str] = []
    for eye in ("left", "right"):
        pair = (f"xpos_{eye}", f"ypos_{eye}")
        present = [name in names for name in pair]
        if all(present):
            eye_pairs.append((names[pair[0]], names[pair[1]]))
        elif any(present):
            incomplete.append(eye)
    if incomplete:
        raise ValueError(
            f"{context} requires complete xpos/ypos pairs; "
            f"incomplete eyes={incomplete}."
        )
    if not eye_pairs:
        raise ValueError(
            f"{context} requires xpos/ypos channels for at least one eye."
        )

    data = _window_data(epochs, time_window)
    eye_valid = np.stack(
        [np.isfinite(data[:, pair, :]).all(axis=1) for pair in eye_pairs],
        axis=1,
    )
    missing_fraction = np.mean(~eye_valid.any(axis=1), axis=1)
    flagged = missing_fraction > threshold
    mask = np.zeros(data.shape[:2], dtype=bool)
    picks = [pick for pair in eye_pairs for pick in pair]
    mask[:, picks] = flagged[:, np.newaxis]
    return mask


def _mean_ignoring_nan(values: np.ndarray) -> np.ndarray:
    """Average over samples without warning when a complete slice is NaN."""

    valid = ~np.isnan(values)
    count = valid.sum(axis=2)
    total = np.sum(np.where(valid, values, 0), axis=2)
    return np.divide(
        total,
        count,
        out=np.full(total.shape, np.nan, dtype=float),
        where=count > 0,
    )


def _flatline_mask(
    epochs: mne.Epochs,
    *,
    time_window,
    duration_ms: float = 200,
) -> np.ndarray:
    data = _window_data(epochs, time_window)
    duration = _milliseconds_to_samples(epochs, duration_ms)
    mask = np.zeros(data.shape[:2], dtype=bool)
    eeg = _channel_type_mask(epochs, "eeg")
    if not np.any(eeg):
        return mask
    changes = np.diff(data[:, eeg, :], axis=2) != 0

    def has_run(values: np.ndarray) -> bool:
        boundaries = np.flatnonzero(np.concatenate(([True], values, [True])))
        return bool(np.any(np.diff(boundaries) >= duration))

    mask[:, eeg] = np.apply_along_axis(has_run, 2, changes)
    return mask


def _absolute_mask(
    epochs: mne.Epochs,
    thresholds: Mapping[str, float],
    *,
    time_window,
) -> np.ndarray:
    """Mark channels whose absolute value exceeds a type-specific threshold."""

    data = _window_data(epochs, time_window)
    mask = np.zeros(data.shape[:2], dtype=bool)
    for channel_type, threshold in thresholds.items():
        picks = _channel_type_mask(epochs, channel_type)
        if np.any(picks):
            values = np.abs(data[:, picks])
            maximum = np.max(np.where(np.isnan(values), -np.inf, values), axis=2)
            mask[:, picks] = maximum > threshold
    return _match_both_eyes(epochs, mask)


def _step_mask(
    epochs: mne.Epochs,
    thresholds: Mapping[str, float],
    *,
    time_window,
    window_ms: float = 250,
    step_ms: float = 20,
) -> np.ndarray:
    """Mark channels with a large mean shift across adjacent half-windows."""

    data = _window_data(epochs, time_window)
    window = _milliseconds_to_samples(epochs, window_ms)
    step = _milliseconds_to_samples(epochs, step_ms)
    mask = np.zeros(data.shape[:2], dtype=bool)
    for channel_type, threshold in thresholds.items():
        picks = _channel_type_mask(epochs, channel_type)
        if not np.any(picks):
            continue
        for start in _window_starts(data.shape[2], window, step):
            stop = min(start + window, data.shape[2])
            middle = start + (stop - start) // 2
            first = _mean_ignoring_nan(data[:, picks, start:middle])
            second = _mean_ignoring_nan(data[:, picks, middle:stop])
            mask[:, picks] |= np.abs(first - second) > threshold
    return _match_both_eyes(epochs, mask)


def _peak_to_peak_mask(
    epochs: mne.Epochs,
    thresholds: Mapping[str, float],
    *,
    time_window,
    window_ms: float = 200,
    step_ms: float = 100,
) -> np.ndarray:
    """Mark channels with excessive peak-to-peak amplitude in a window."""

    data = _window_data(epochs, time_window)
    window = _milliseconds_to_samples(epochs, window_ms)
    step = _milliseconds_to_samples(epochs, step_ms)
    mask = np.zeros(data.shape[:2], dtype=bool)
    for channel_type, threshold in thresholds.items():
        picks = _channel_type_mask(epochs, channel_type)
        if not np.any(picks):
            continue
        for start in _window_starts(data.shape[2], window, step):
            values = data[:, picks, start : min(start + window, data.shape[2])]
            maximum = np.max(np.where(np.isnan(values), -np.inf, values), axis=2)
            minimum = np.min(np.where(np.isnan(values), np.inf, values), axis=2)
            mask[:, picks] |= maximum - minimum > threshold
    return _match_both_eyes(epochs, mask)


def _linear_mask(
    epochs: mne.Epochs,
    *,
    time_window,
    min_slope: float = 75e-6,
    min_r2: float = 0.3,
) -> np.ndarray:
    data = _window_data(epochs, time_window)
    mask = np.zeros(data.shape[:2], dtype=bool)
    eeg = _channel_type_mask(epochs, "eeg")
    if not np.any(eeg):
        return mask
    values = data[:, eeg]
    # Preserve the existing QC rule's sample-index fit and positive-slope test.
    x = np.arange(values.shape[2], dtype=float)
    design = np.column_stack((x, np.ones_like(x)))
    for trial in range(values.shape[0]):
        slopes, _, _, _ = np.linalg.lstsq(design, values[trial].T, rcond=None)
        fitted = design @ slopes
        residual = values[trial].T - fitted
        ss_res = np.sum(residual**2, axis=0)
        centered = values[trial] - values[trial].mean(axis=1, keepdims=True)
        ss_total = np.sum(centered**2, axis=1)
        ratio = np.divide(ss_res, ss_total, out=np.ones_like(ss_res), where=ss_total > 0)
        r2 = 1 - ratio
        mask[trial, eeg] = (slopes[0] > min_slope) & (r2 > min_r2)
    return mask


_HF_CONFIG_KEYS = {
    "band",
    "window_duration",
    "z_threshold",
    "min_noisy_fraction",
    "bad_channels",
}


def _validate_high_frequency_config(
    epochs: mne.Epochs,
    config: Mapping[str, object],
    *,
    time_window,
) -> tuple[float, float, int]:
    if not isinstance(config, Mapping):
        raise TypeError("hf_noise configuration must be a mapping.")
    missing = sorted(_HF_CONFIG_KEYS.difference(config))
    unknown = sorted(set(config).difference(_HF_CONFIG_KEYS))
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys {missing}")
        if unknown:
            details.append(f"unknown keys {unknown}")
        raise ValueError("hf_noise configuration has " + " and ".join(details) + ".")

    band = config["band"]
    if (
        not isinstance(band, Sequence)
        or isinstance(band, (str, bytes))
        or len(band) != 2
        or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in band)
    ):
        raise ValueError("hf_noise.band must contain two finite frequencies.")
    low, high = (float(value) for value in band)
    nyquist = float(epochs.info["sfreq"]) / 2
    if not np.isfinite([low, high]).all() or not 0 < low < high < nyquist:
        raise ValueError(
            f"hf_noise.band must satisfy 0 < low < high < Nyquist ({nyquist:g} Hz)."
        )

    duration = config["window_duration"]
    if isinstance(duration, (bool, np.bool_)) or not isinstance(duration, Real):
        raise ValueError("hf_noise.window_duration must be a finite positive number of seconds.")
    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("hf_noise.window_duration must be a finite positive number of seconds.")

    z_threshold = config["z_threshold"]
    if isinstance(z_threshold, (bool, np.bool_)) or not isinstance(z_threshold, Real):
        raise ValueError("hf_noise.z_threshold must be a finite positive number.")
    z_threshold = float(z_threshold)
    if not np.isfinite(z_threshold) or z_threshold <= 0:
        raise ValueError("hf_noise.z_threshold must be a finite positive number.")

    fraction = config["min_noisy_fraction"]
    if isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, Real):
        raise ValueError("hf_noise.min_noisy_fraction must be in (0, 1].")
    fraction = float(fraction)
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("hf_noise.min_noisy_fraction must be in (0, 1].")

    bad_channels = config["bad_channels"]
    if isinstance(bad_channels, (bool, np.bool_)) or not isinstance(bad_channels, Integral):
        raise ValueError("hf_noise.bad_channels must be a positive integer.")
    if int(bad_channels) <= 0:
        raise ValueError("hf_noise.bad_channels must be a positive integer.")

    keep = _time_window_mask(epochs, time_window)
    window = int(round(duration * float(epochs.info["sfreq"])))
    if window < 1 or window > int(keep.sum()):
        raise ValueError(
            "hf_noise.window_duration must fit within the selected QC time_window."
        )
    return low, high, window


def _time_window_mask(epochs: mne.Epochs, time_window) -> np.ndarray:
    if time_window is None:
        return np.ones(len(epochs.times), dtype=bool)
    tmin, tmax = time_window
    tmin = epochs.times[0] if tmin is None else tmin
    tmax = epochs.times[-1] if tmax is None else tmax
    keep = (epochs.times >= tmin) & (epochs.times <= tmax)
    if not np.any(keep):
        raise ValueError(f"time_window {time_window!r} does not overlap the epochs.")
    return keep


def _hf_window_starts(n_times: int, window: int) -> np.ndarray:
    step = max(1, window // 2)
    starts = np.arange(0, n_times - window + 1, step, dtype=int)
    final = n_times - window
    if starts[-1] != final:
        starts = np.append(starts, final)
    return starts


def _high_frequency_zscores(
    epochs: mne.Epochs,
    eligibility: EligibilityResult,
    autoreject_bad_epochs: np.ndarray,
    autoreject_labels: np.ndarray,
    config: Mapping[str, object],
    *,
    time_window,
) -> tuple[np.ndarray, np.ndarray]:
    low, high, window = _validate_high_frequency_config(
        epochs, config, time_window=time_window
    )
    all_eeg_picks = np.flatnonzero(_channel_type_mask(epochs, "eeg"))
    scored_picks = np.asarray(
        [pick for pick in all_eeg_picks if epochs.ch_names[pick] not in epochs.info["bads"]],
        dtype=int,
    )
    if not len(scored_picks):
        return np.empty((len(epochs), 0, 0)), scored_picks

    autoreject_labels = np.asarray(autoreject_labels)
    expected = (len(epochs), len(all_eeg_picks))
    if autoreject_labels.shape != expected:
        raise ValueError(
            "autoreject_labels must have shape "
            f"{expected}, found {autoreject_labels.shape}."
        )
    eeg_columns = {pick: column for column, pick in enumerate(all_eeg_picks)}
    scored_labels = autoreject_labels[:, [eeg_columns[pick] for pick in scored_picks]]

    data = np.take(epochs.get_data(copy=False), scored_picks, axis=1)
    mne.filter.filter_data(
        data,
        sfreq=float(epochs.info["sfreq"]),
        l_freq=low,
        h_freq=high,
        copy=False,
        verbose="ERROR",
    )
    data = data[:, :, _time_window_mask(epochs, time_window)]
    starts = _hf_window_starts(data.shape[2], window)
    powers = np.stack(
        [np.nanmean(np.square(data[:, :, start : start + window]), axis=2) for start in starts],
        axis=2,
    )
    log_power = np.log(np.maximum(powers, np.finfo(float).tiny))

    reference_epochs = np.asarray(eligibility.eligible, dtype=bool) & ~autoreject_bad_epochs
    active_autoreject = bool(np.any(autoreject_labels >= 0))
    zscores = np.full(log_power.shape, np.nan, dtype=float)
    for column, pick in enumerate(scored_picks):
        reference = reference_epochs.copy()
        if active_autoreject:
            reference &= scored_labels[:, column] == 0
        if int(reference.sum()) < 2:
            raise ValueError(
                "hf_noise requires at least two reference epochs for channel "
                f"{epochs.ch_names[pick]!r}."
            )
        values = log_power[reference, column].reshape(-1)
        values = values[np.isfinite(values)]
        center = float(np.median(values))
        scale = 1.4826 * float(np.median(np.abs(values - center)))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                "hf_noise could not estimate a positive robust scale for channel "
                f"{epochs.ch_names[pick]!r}."
            )
        zscores[:, column] = (log_power[:, column] - center) / scale
    return zscores, scored_picks


def _high_frequency_mask_from_zscores(
    epochs: mne.Epochs,
    zscores: np.ndarray,
    scored_picks: np.ndarray,
    config: Mapping[str, object],
) -> np.ndarray:
    mask = np.zeros((len(epochs), len(epochs.ch_names)), dtype=bool)
    if not len(scored_picks):
        return mask
    noisy_fraction = np.mean(zscores >= float(config["z_threshold"]), axis=2)
    mask[:, scored_picks] = noisy_fraction >= float(config["min_noisy_fraction"])
    return mask


def _append_reasons(reason_matrix: np.ndarray, masks: Mapping[str, np.ndarray]) -> np.ndarray:
    output = reason_matrix.astype("<U256", copy=True)
    for code, mask in masks.items():
        current = output[mask]
        output[mask] = np.where(current == "", code, np.char.add(np.char.add(current, ";"), code))
    return output


def _aggregate_channel_mask(
    epochs: mne.Epochs,
    masks: Sequence[np.ndarray],
    *,
    bad_channels: int,
    ignore_channels: Sequence[str] = (),
) -> np.ndarray:
    combined = np.logical_or.reduce(masks)
    eeg = _channel_type_mask(epochs, "eeg")
    if ignore_channels:
        eeg &= ~np.isin(np.asarray(epochs.ch_names, dtype=object), list(ignore_channels))
    return combined[:, eeg].sum(axis=1) >= bad_channels


def check_eligibility(
    epochs: mne.Epochs,
    config: Mapping[str, object],
    *,
    gaze_geometry: Mapping[str, object] | None = None,
) -> EligibilityResult:
    """Mark structurally invalid and extreme EEG/gaze trials before AutoReject."""

    if "ignore_channels" in config:
        raise ValueError(
            "ignore_channels belongs to label_artifacts(), not eligibility."
        )
    _reject_embedded_gaze_geometry(config, "eligibility")
    time_window = config.get("time_window")
    eeg = config["eeg"]
    gaze_thresholds: dict[str, float | int] = {}
    if "gaze" in config:
        _require_eyegaze_channels(epochs, "eligibility.gaze")
        gaze_thresholds = _normalize_gaze_rule(
            config["gaze"], gaze_geometry, context="eligibility.gaze"
        )
    masks = {
        "dropout": _dropout_mask(epochs, time_window),
        "flatline": _flatline_mask(epochs, time_window=time_window),
        "large_absolute_value": _absolute_mask(
            epochs, {"eeg": float(eeg["absolute_value"])}, time_window=time_window
        ),
        "large_step": _step_mask(epochs, {"eeg": float(eeg["step"])}, time_window=time_window),
        "large_peak_to_peak": _peak_to_peak_mask(
            epochs, {"eeg": float(eeg["p2p"])}, time_window=time_window
        ),
    }
    if "deviation_px" in gaze_thresholds:
        masks["large_gaze_deviation"] = _absolute_mask(
            epochs,
            {"eyegaze": gaze_thresholds["deviation_px"]},
            time_window=time_window,
        )
    if "shift_px" in gaze_thresholds:
        masks["large_gaze_shift"] = _step_mask(
            epochs,
            {"eyegaze": gaze_thresholds["shift_px"]},
            time_window=time_window,
            window_ms=80,
            step_ms=10,
        )
    if "max_missing_fraction" in gaze_thresholds:
        masks["gaze_missing"] = _gaze_missing_mask(
            epochs,
            float(gaze_thresholds["max_missing_fraction"]),
            time_window=time_window,
            context="eligibility.gaze.max_missing_fraction",
        )

    structural = masks["dropout"].any(axis=1) | masks["flatline"].any(axis=1)
    eeg_bad = _aggregate_channel_mask(
        epochs,
        [masks["large_absolute_value"], masks["large_step"], masks["large_peak_to_peak"]],
        bad_channels=int(eeg["bad_channels"]),
    )
    gaze_bad = np.zeros(len(epochs), dtype=bool)
    for code in ("large_gaze_deviation", "large_gaze_shift", "gaze_missing"):
        if code in masks:
            gaze_bad |= masks[code].any(axis=1)
    rejected = structural | eeg_bad | gaze_bad
    reasons = _append_reasons(np.full((len(epochs), len(epochs.ch_names)), "", dtype="<U256"), masks)
    return EligibilityResult(eligible=~rejected, rule_masks=masks, channel_reasons=reasons)


def apply_autoreject(
    epochs: mne.Epochs,
    eligible: np.ndarray,
    config: Mapping[str, object] | None,
) -> AutorejectResult:
    """Fit AutoReject on eligible trials and restore results to full order.

    Scalar ``consensus`` and ``n_interpolate`` values are treated as
    single-candidate lists. ``cv`` accepts either an integer or a mapping with
    ``n_splits``, ``shuffle``, and ``random_state``. EEG channel names in
    ``exclude_channels`` remain in the returned epochs but do not participate
    in threshold learning, cross-validation, voting, or interpolation.
    """

    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != (len(epochs),):
        raise ValueError("eligible must contain one boolean value per epoch.")
    eeg_picks = np.flatnonzero(_channel_type_mask(epochs, "eeg"))
    eeg_channels = tuple(epochs.ch_names[pick] for pick in eeg_picks)
    excluded_channels = (
        ()
        if config is None
        else _autoreject_excluded_channels(config, epochs, eeg_channels)
    )
    info_bad_channels = tuple(
        channel for channel in eeg_channels if channel in epochs.info["bads"]
    )
    unavailable = set(excluded_channels) | set(info_bad_channels)
    autoreject_picks = [
        pick for pick in eeg_picks if epochs.ch_names[pick] not in unavailable
    ]
    autoreject_channels = tuple(epochs.ch_names[pick] for pick in autoreject_picks)
    labels = np.full((len(epochs), len(eeg_channels)), -1, dtype=np.int8)
    bad = np.zeros(len(epochs), dtype=bool)
    interpolated = np.zeros(len(epochs), dtype=int)
    output = epochs.copy().load_data()
    diagnostics = _base_autoreject_diagnostics(
        epochs,
        eligible,
        eeg_channels,
        autoreject_channels,
        excluded_channels,
        info_bad_channels,
    )
    if (
        config is None
        or not config.get("enabled", True)
        or not np.any(eligible)
        or not eeg_channels
    ):
        return AutorejectResult(
            output, bad, interpolated, labels, eeg_channels, diagnostics
        )
    if not autoreject_channels:
        raise ValueError("autoreject.exclude_channels leaves no EEG channels to fit.")

    selected = np.flatnonzero(eligible)
    n_interpolate = _candidate_values(
        config.get("n_interpolate", [1, 2]),
        name="n_interpolate",
        integer=True,
    )
    cv, resolved_cv = _cv_config(config.get("cv", 10))
    random_state = config.get("random_state", 0)
    if isinstance(random_state, (bool, np.bool_)) or (
        random_state is not None and not isinstance(random_state, Integral)
    ):
        raise TypeError("autoreject.random_state must be an integer or None.")
    random_state = None if random_state is None else int(random_state)
    model_kwargs = {
        "n_interpolate": n_interpolate,
        "cv": cv,
        "picks": list(autoreject_channels),
        "random_state": random_state,
        "n_jobs": int(config.get("n_jobs", 1)),
        "verbose": config.get("verbose", False),
    }
    if "consensus" in config:
        model_kwargs["consensus"] = _candidate_values(
            config["consensus"], name="consensus", integer=False
        )
    else:
        warnings.warn(
            "autoreject.consensus is not set; using AutoReject's default "
            "candidate grid. Set consensus explicitly for reproducible "
            "study-level behavior.",
            UserWarning,
            stacklevel=2,
        )
    model = AutoReject(**model_kwargs)
    repaired, log = model.fit_transform(epochs[selected], return_log=True)
    raw_bad = np.asarray(log.bad_epochs, dtype=bool)
    if raw_bad.shape != (len(selected),):
        raise RuntimeError(
            "AutoReject returned an unexpected bad-epoch vector with shape "
            f"{raw_bad.shape}; expected ({len(selected)},)."
        )
    bad[selected] = raw_bad
    raw_labels = np.asarray(log.labels, dtype=float).copy()
    if raw_labels.shape == (len(selected), len(epochs.ch_names)):
        raw_labels = raw_labels[:, eeg_picks]
    elif raw_labels.shape == (len(selected), len(autoreject_channels)):
        expanded = np.full((len(selected), len(eeg_channels)), np.nan, dtype=float)
        positions = [eeg_channels.index(channel) for channel in autoreject_channels]
        expanded[:, positions] = raw_labels
        raw_labels = expanded
    elif raw_labels.shape != (len(selected), len(eeg_channels)):
        raise RuntimeError(
            "AutoReject returned an unexpected channel-label matrix with shape "
            f"{raw_labels.shape}; expected ({len(selected)}, {len(epochs.ch_names)}) "
            f"or ({len(selected)}, {len(eeg_channels)}) or "
            f"({len(selected)}, {len(autoreject_channels)})."
        )
    unavailable_positions = [
        eeg_channels.index(channel)
        for channel in (*excluded_channels, *info_bad_channels)
    ]
    raw_labels[:, unavailable_positions] = np.nan
    finite_labels = raw_labels[~np.isnan(raw_labels)]
    if not np.isin(finite_labels, [0, 1, 2]).all():
        raise RuntimeError("AutoReject returned channel labels outside 0, 1, and 2.")
    # RejectLog uses NaN for EEG channels already listed in info["bads"].
    raw_labels = np.nan_to_num(raw_labels, nan=-1).astype(np.int8)
    labels[selected] = raw_labels
    interpolated[selected] = np.sum(raw_labels == 2, axis=1)
    good_selected = selected[~raw_bad]
    repaired_data = repaired.get_data(copy=True)
    if repaired_data.shape[0] != len(good_selected):
        raise RuntimeError(
            "AutoReject returned an unexpected number of retained epochs: "
            f"{repaired_data.shape[0]}; expected {len(good_selected)}."
        )
    if len(good_selected):
        output._data[good_selected] = repaired_data
    consensus_grid = np.asarray(model.consensus, dtype=float)
    n_interpolate_grid = np.asarray(model.n_interpolate, dtype=int)
    resolved_config = {
        "enabled": True,
        "consensus": consensus_grid.tolist(),
        "n_interpolate": n_interpolate_grid.tolist(),
        "cv": resolved_cv,
        "random_state": model.random_state,
        "n_jobs": model.n_jobs,
        "exclude_channels": list(excluded_channels),
    }
    diagnostics.update(
        {
            "consensus_": float(model.consensus_["eeg"]),
            "n_interpolate_": int(model.n_interpolate_["eeg"]),
            "consensus_grid": consensus_grid,
            "n_interpolate_grid": n_interpolate_grid,
            "threshold_values": np.asarray(
                [
                    model.threshes_.get(channel, np.nan)
                    for channel in autoreject_channels
                ],
                dtype=float,
            ),
            "loss": np.asarray(model.loss_["eeg"], dtype=float),
            "cv_config": resolved_cv,
            "config_hash": _config_hash(resolved_config),
        }
    )
    return AutorejectResult(
        output, bad, interpolated, labels, eeg_channels, diagnostics
    )


def label_artifact_rules(
    epochs: mne.Epochs,
    eligibility: EligibilityResult,
    autoreject_bad_epochs: np.ndarray,
    *,
    autoreject_labels: np.ndarray,
    reject_config: Mapping[str, object],
    review_config: Mapping[str, object],
    ignore_channels: Sequence[str] = (),
    gaze_geometry: Mapping[str, object] | None = None,
) -> ArtifactRuleResult:
    """Run post-AutoReject reject/review rules without dropping epochs."""

    unknown_ignored = sorted(set(ignore_channels).difference(epochs.ch_names))
    if unknown_ignored:
        raise ValueError(f"ignore_channels contains unknown channels: {unknown_ignored}.")
    _reject_embedded_gaze_geometry(reject_config, "reject")
    _reject_embedded_gaze_geometry(review_config, "review")
    if "gaze" in reject_config:
        raise ValueError(
            "reject.gaze is unsupported; hard gaze rules belong in eligibility.gaze."
        )
    gaze_thresholds: dict[str, float | int] = {}
    if "gaze" in review_config:
        _require_eyegaze_channels(epochs, "review.gaze")
        gaze_thresholds = _normalize_gaze_rule(
            review_config["gaze"], gaze_geometry, context="review.gaze"
        )
    _validate_eligibility_result(epochs, eligibility)
    autoreject_bad_epochs = np.asarray(autoreject_bad_epochs, dtype=bool)
    if autoreject_bad_epochs.shape != (len(epochs),):
        raise ValueError("autoreject_bad_epochs must contain one boolean value per epoch.")

    reject_time_window = reject_config.get("time_window")
    review_time_window = review_config.get("time_window", reject_time_window)
    hf_cache: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] = {}

    def high_frequency_mask(config, time_window):
        low, high, window = _validate_high_frequency_config(
            epochs, config, time_window=time_window
        )
        normalized_window = None if time_window is None else tuple(time_window)
        key = (low, high, window, normalized_window)
        if key not in hf_cache:
            hf_cache[key] = _high_frequency_zscores(
                epochs,
                eligibility,
                autoreject_bad_epochs,
                autoreject_labels,
                config,
                time_window=time_window,
            )
        return _high_frequency_mask_from_zscores(epochs, *hf_cache[key], config)

    hard_masks = dict(eligibility.rule_masks)
    hard_hf = reject_config.get("hf_noise")
    if hard_hf is not None:
        hard_masks["high_frequency_noise"] = high_frequency_mask(
            hard_hf, reject_time_window
        )

    epoch_rejected = ~eligibility.eligible | autoreject_bad_epochs
    if hard_hf is not None:
        epoch_rejected |= _aggregate_channel_mask(
            epochs,
            [hard_masks["high_frequency_noise"]],
            bad_channels=int(hard_hf["bad_channels"]),
            ignore_channels=ignore_channels,
        )

    candidate = ~epoch_rejected
    review_masks: dict[str, np.ndarray] = {}
    eeg = review_config["eeg"]
    review_masks["moderate_absolute_value"] = _absolute_mask(
        epochs, {"eeg": float(eeg["absolute_value"])}, time_window=review_time_window
    )
    review_masks["moderate_step"] = _step_mask(
        epochs, {"eeg": float(eeg["step"])}, time_window=review_time_window
    )
    review_masks["moderate_peak_to_peak"] = _peak_to_peak_mask(
        epochs, {"eeg": float(eeg["p2p"])}, time_window=review_time_window
    )
    review_masks["linear_drift"] = _linear_mask(epochs, time_window=review_time_window)
    if "deviation_px" in gaze_thresholds:
        review_masks["moderate_gaze_deviation"] = _absolute_mask(
            epochs,
            {"eyegaze": gaze_thresholds["deviation_px"]},
            time_window=review_time_window,
        )
    if "shift_px" in gaze_thresholds:
        review_masks["moderate_gaze_shift"] = _step_mask(
            epochs,
            {"eyegaze": gaze_thresholds["shift_px"]},
            time_window=review_time_window,
            window_ms=80,
            step_ms=10,
        )
    if "max_missing_fraction" in gaze_thresholds:
        review_masks["gaze_missing"] = _gaze_missing_mask(
            epochs,
            float(gaze_thresholds["max_missing_fraction"]),
            time_window=review_time_window,
            context="review.gaze.max_missing_fraction",
        )
    review_hf = review_config.get("hf_noise")
    if review_hf is not None:
        review_masks["moderate_high_frequency_noise"] = high_frequency_mask(
            review_hf, review_time_window
        )

    for mask in review_masks.values():
        mask[~candidate] = False
    epoch_review = _aggregate_channel_mask(
        epochs,
        [
            review_masks["moderate_absolute_value"],
            review_masks["moderate_step"],
            review_masks["moderate_peak_to_peak"],
            review_masks["linear_drift"],
        ],
        bad_channels=int(eeg["bad_channels"]),
        ignore_channels=ignore_channels,
    )
    for code in ("moderate_gaze_deviation", "moderate_gaze_shift", "gaze_missing"):
        if code in review_masks:
            channel_mask = review_masks[code].copy()
            if ignore_channels:
                channel_mask[:, np.isin(epochs.ch_names, list(ignore_channels))] = False
            epoch_review |= channel_mask.any(axis=1)
    if review_hf is not None:
        epoch_review |= _aggregate_channel_mask(
            epochs,
            [review_masks["moderate_high_frequency_noise"]],
            bad_channels=int(review_hf["bad_channels"]),
            ignore_channels=ignore_channels,
        )
    epoch_review &= candidate

    rejected_reasons = _append_reasons(
        np.full((len(epochs), len(epochs.ch_names)), "", dtype="<U256"), hard_masks
    )
    review_reasons = _append_reasons(
        np.full((len(epochs), len(epochs.ch_names)), "", dtype="<U256"), review_masks
    )
    return ArtifactRuleResult(rejected_reasons, review_reasons, epoch_rejected, epoch_review)


def save_quality_state(
    path: str | Path,
    eligibility: EligibilityResult,
    autoreject: AutorejectResult,
) -> None:
    """Save schema-v2 quality state without pickle-backed arrays."""

    diagnostics = dict(autoreject.diagnostics or {})
    consensus = diagnostics.get("consensus_")
    n_interpolate = diagnostics.get("n_interpolate_")
    provenance = {
        "config_hash": str(diagnostics.get("config_hash", "")),
        "versions": diagnostics.get("versions", {}),
        "input_summary": diagnostics.get("input_summary", {}),
    }

    payload: dict[str, np.ndarray] = {
        "quality_schema_version": np.asarray(_QUALITY_SCHEMA_VERSION, dtype=np.int16),
        "eligible": eligibility.eligible,
        "eligibility_channel_reasons": eligibility.channel_reasons,
        "autoreject_bad_epochs": autoreject.bad_epochs,
        "autoreject_interpolated_channels": autoreject.interpolated_channels,
        "autoreject_labels": autoreject.labels,
        "autoreject_eeg_channels": np.asarray(autoreject.eeg_channels, dtype=str),
        "autoreject_model_channels": np.asarray(
            diagnostics.get("autoreject_channels", ()), dtype=str
        ),
        "autoreject_excluded_channels": np.asarray(
            diagnostics.get("excluded_channels", ()), dtype=str
        ),
        "autoreject_info_bad_channels": np.asarray(
            diagnostics.get("info_bad_channels", ()), dtype=str
        ),
        "autoreject_consensus": np.asarray(
            np.nan if consensus is None else consensus, dtype=float
        ),
        "autoreject_n_interpolate": np.asarray(
            -1 if n_interpolate is None else n_interpolate, dtype=int
        ),
        "autoreject_consensus_grid": np.asarray(
            diagnostics.get("consensus_grid", []), dtype=float
        ),
        "autoreject_n_interpolate_grid": np.asarray(
            diagnostics.get("n_interpolate_grid", []), dtype=int
        ),
        "autoreject_threshold_values": np.asarray(
            diagnostics.get("threshold_values", []), dtype=float
        ),
        "autoreject_loss": np.asarray(diagnostics.get("loss", []), dtype=float),
        "autoreject_cv_json": np.asarray(
            json.dumps(
                diagnostics.get("cv_config", {}),
                sort_keys=True,
                separators=(",", ":"),
            ),
            dtype=str,
        ),
        "autoreject_provenance_json": np.asarray(
            json.dumps(provenance, sort_keys=True, separators=(",", ":")),
            dtype=str,
        ),
    }
    for code, mask in eligibility.rule_masks.items():
        payload[f"eligibility__{code}"] = mask
    np.savez_compressed(Path(path), **payload)


def load_quality_state(path: str | Path) -> tuple[EligibilityResult, dict[str, object]]:
    """Read current or legacy quality state without enabling pickle."""

    with np.load(Path(path), allow_pickle=False) as saved:
        files = set(saved.files)
        masks = {
            key.removeprefix("eligibility__"): saved[key].astype(bool)
            for key in saved.files
            if key.startswith("eligibility__")
        }
        eligibility = EligibilityResult(
            eligible=saved["eligible"].astype(bool),
            rule_masks=masks,
            channel_reasons=saved["eligibility_channel_reasons"].astype("<U256"),
        )
        schema_version = (
            int(saved["quality_schema_version"].item())
            if "quality_schema_version" in files
            else 1
        )
        consensus = (
            float(saved["autoreject_consensus"].item())
            if "autoreject_consensus" in files
            else np.nan
        )
        n_interpolate = (
            int(saved["autoreject_n_interpolate"].item())
            if "autoreject_n_interpolate" in files
            else -1
        )
        threshold_values = (
            saved["autoreject_threshold_values"].astype(float)
            if "autoreject_threshold_values" in files
            else np.asarray([], dtype=float)
        )
        cv_config = (
            json.loads(str(saved["autoreject_cv_json"].item()))
            if "autoreject_cv_json" in files
            else {}
        )
        provenance = (
            json.loads(str(saved["autoreject_provenance_json"].item()))
            if "autoreject_provenance_json" in files
            else {}
        )
        eeg_channels = saved["autoreject_eeg_channels"].astype(str)
        autoreject_channels = (
            saved["autoreject_model_channels"].astype(str)
            if "autoreject_model_channels" in files
            else np.asarray([], dtype=str)
        )
        threshold_channels = (
            autoreject_channels
            if len(autoreject_channels) == len(threshold_values)
            else eeg_channels
        )
        autoreject: dict[str, object] = {
            "bad_epochs": saved["autoreject_bad_epochs"].astype(bool),
            "interpolated_channels": saved["autoreject_interpolated_channels"].astype(int),
            "labels": saved["autoreject_labels"].astype(np.int8),
            "eeg_channels": eeg_channels,
            "autoreject_channels": autoreject_channels,
            "excluded_channels": (
                saved["autoreject_excluded_channels"].astype(str)
                if "autoreject_excluded_channels" in files
                else np.asarray([], dtype=str)
            ),
            "info_bad_channels": (
                saved["autoreject_info_bad_channels"].astype(str)
                if "autoreject_info_bad_channels" in files
                else np.asarray([], dtype=str)
            ),
            "schema_version": schema_version,
            "consensus_": consensus if np.isfinite(consensus) else None,
            "n_interpolate_": n_interpolate if n_interpolate >= 0 else None,
            "consensus_grid": (
                saved["autoreject_consensus_grid"].astype(float)
                if "autoreject_consensus_grid" in files
                else np.asarray([], dtype=float)
            ),
            "n_interpolate_grid": (
                saved["autoreject_n_interpolate_grid"].astype(int)
                if "autoreject_n_interpolate_grid" in files
                else np.asarray([], dtype=int)
            ),
            "threshold_values": threshold_values,
            "thresholds": (
                dict(zip(threshold_channels.tolist(), threshold_values.tolist()))
                if len(threshold_values) == len(threshold_channels)
                else {}
            ),
            "loss": (
                saved["autoreject_loss"].astype(float)
                if "autoreject_loss" in files
                else np.asarray([], dtype=float)
            ),
            "cv_config": cv_config,
            "config_hash": provenance.get("config_hash", ""),
            "versions": provenance.get("versions", {}),
            "input_summary": provenance.get("input_summary", {}),
        }
    return eligibility, autoreject


def _validate_eligibility_result(
    epochs: mne.Epochs,
    eligibility: EligibilityResult,
) -> None:
    """Reject stale or corrupt quality state before artifact labeling."""

    expected_channels = (len(epochs), len(epochs.ch_names))
    eligible = np.asarray(eligibility.eligible)
    if eligible.shape != (len(epochs),) or eligible.dtype.kind != "b":
        raise ValueError("Eligibility state must contain one boolean value per epoch.")
    reasons = np.asarray(eligibility.channel_reasons)
    if reasons.shape != expected_channels:
        raise ValueError(
            "Eligibility channel reasons have shape "
            f"{reasons.shape}; expected {expected_channels}."
        )
    for code, mask in eligibility.rule_masks.items():
        if np.asarray(mask).shape != expected_channels:
            raise ValueError(
                f"Eligibility rule {code!r} has shape {np.asarray(mask).shape}; "
                f"expected {expected_channels}."
            )


_GAZE_RULE_KEYS = {"deviation_deg", "shift_deg", "max_missing_fraction"}
_GAZE_DEGREE_KEYS = {"deviation_deg", "shift_deg"}


def _validate_gaze_rule(rule: object, *, context: str) -> dict[str, object]:
    """Validate the shape and missing-fraction value of one gaze rule."""

    if not isinstance(rule, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    if not rule:
        raise ValueError(f"{context} must contain at least one gaze rule.")
    extra = sorted(set(rule).difference(_GAZE_RULE_KEYS))
    if extra:
        raise ValueError(
            f"{context} only supports {sorted(_GAZE_RULE_KEYS)}; extra={extra}."
        )
    if "max_missing_fraction" in rule:
        value = rule["max_missing_fraction"]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise ValueError(
                f"{context}.max_missing_fraction must be a finite number "
                "between 0 and 1."
            )
    return dict(rule)


def _gaze_rule_requires_geometry(rule: object, *, context: str) -> bool:
    """Return whether a validated gaze rule contains degree thresholds."""

    return bool(_GAZE_DEGREE_KEYS.intersection(_validate_gaze_rule(rule, context=context)))


def _normalize_gaze_rule(
    rule: object,
    gaze_geometry: Mapping[str, object] | None,
    *,
    context: str,
) -> dict[str, float | int]:
    """Validate one gaze rule and convert configured degree thresholds."""

    normalized = _validate_gaze_rule(rule, context=context)
    output: dict[str, float | int] = {}
    if _GAZE_DEGREE_KEYS.intersection(normalized):
        geometry = _normalize_gaze_geometry_mapping(gaze_geometry, context=context)
        if "deviation_deg" in normalized:
            output["deviation_px"] = _degrees_to_pixels(
                normalized["deviation_deg"], geometry
            )
        if "shift_deg" in normalized:
            output["shift_px"] = _degrees_to_pixels(
                normalized["shift_deg"], geometry
            )
    if "max_missing_fraction" in normalized:
        output["max_missing_fraction"] = float(normalized["max_missing_fraction"])
    return output


def _normalize_gaze_geometry_mapping(
    gaze_geometry: Mapping[str, object] | None,
    *,
    context: str,
) -> dict[str, float | int]:
    """Normalize provenance geometry with a context-rich contract error."""

    if gaze_geometry is None:
        raise ValueError(f"{context} requires gaze_geometry in dataset provenance.")
    if not isinstance(gaze_geometry, Mapping):
        raise TypeError("gaze_geometry in dataset provenance must be a mapping.")
    try:
        return normalize_gaze_geometry(**dict(gaze_geometry))
    except TypeError as error:
        raise ValueError(
            "gaze_geometry must contain exactly viewing_distance_cm, "
            "screen_width_cm, and screen_width_px."
        ) from error


def _reject_embedded_gaze_geometry(
    config: Mapping[str, object],
    context: str,
) -> None:
    """Keep geometry in dataset provenance rather than rule configuration."""

    embedded = sorted(set(config).intersection({"screen", "gaze_geometry"}))
    if embedded:
        raise ValueError(
            f"{context} cannot define {embedded}; gaze geometry comes from dataset provenance."
        )


def _require_eyegaze_channels(epochs: mne.Epochs, context: str) -> None:
    """Require at least one MNE eyegaze channel for a configured gaze rule."""

    if not np.any(_channel_type_mask(epochs, "eyegaze")):
        raise ValueError(f"{context} requires at least one eyegaze channel.")
