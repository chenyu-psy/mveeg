"""Signal-quality steps used by the 0.3 preprocessing pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import mne
import numpy as np

from ..gaze import _degrees_to_pixels, normalize_gaze_geometry


@dataclass(frozen=True)
class EligibilityResult:
    """Eligibility decisions and their channel-level reasons."""

    eligible: np.ndarray
    rule_masks: dict[str, np.ndarray]
    channel_reasons: np.ndarray


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
            f"{context} requires complete xpos/ypos pairs; incomplete eyes={incomplete}."
        )
    if not eye_pairs:
        raise ValueError(f"{context} requires xpos/ypos channels for at least one eye.")

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


def _append_reasons(
    reason_matrix: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> np.ndarray:
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


_HF_CONFIG_KEYS = {
    "band",
    "window_duration",
    "z_threshold",
    "min_noisy_fraction",
    "bad_channels",
}


def check_eligibility(
    epochs: mne.Epochs,
    config: Mapping[str, object],
    *,
    gaze_geometry: Mapping[str, object] | None = None,
) -> EligibilityResult:
    """Mark structurally invalid and extreme EEG/gaze trials before AutoReject."""

    if "ignore_channels" in config:
        raise ValueError("ignore_channels belongs to label_artifacts(), not eligibility.")
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
    reasons = _append_reasons(
        np.full((len(epochs), len(epochs.ch_names)), "", dtype="<U256"), masks
    )
    return EligibilityResult(eligible=~rejected, rule_masks=masks, channel_reasons=reasons)


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
        raise ValueError(f"{context} only supports {sorted(_GAZE_RULE_KEYS)}; extra={extra}.")
    if "max_missing_fraction" in rule:
        value = rule["max_missing_fraction"]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise ValueError(
                f"{context}.max_missing_fraction must be a finite number between 0 and 1."
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
            output["deviation_px"] = _degrees_to_pixels(normalized["deviation_deg"], geometry)
        if "shift_deg" in normalized:
            output["shift_px"] = _degrees_to_pixels(normalized["shift_deg"], geometry)
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
