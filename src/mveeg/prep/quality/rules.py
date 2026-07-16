"""Signal-quality steps used by the 0.3 preprocessing pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import mne
import numpy as np

from .eligibility import (
    EligibilityResult,
    _absolute_mask,
    _aggregate_channel_mask,
    _append_reasons,
    _channel_type_mask,
    _gaze_missing_mask,
    _linear_mask,
    _normalize_gaze_rule,
    _peak_to_peak_mask,
    _reject_embedded_gaze_geometry,
    _require_eyegaze_channels,
    _step_mask,
)
from .state import validate_eligibility_result


@dataclass(frozen=True)
class ArtifactRuleResult:
    """Automatic reject/review masks produced after signal preprocessing."""

    rejected_reasons: np.ndarray
    review_reasons: np.ndarray
    epoch_rejected: np.ndarray
    epoch_review: np.ndarray


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
        raise ValueError(f"hf_noise.band must satisfy 0 < low < high < Nyquist ({nyquist:g} Hz).")

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
        raise ValueError("hf_noise.window_duration must fit within the selected QC time_window.")
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
    low, high, window = _validate_high_frequency_config(epochs, config, time_window=time_window)
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
            f"autoreject_labels must have shape {expected}, found {autoreject_labels.shape}."
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
    zscores = np.full(log_power.shape, np.nan, dtype=float)
    for column, pick in enumerate(scored_picks):
        reference = reference_epochs.copy()
        channel_labels = scored_labels[:, column]
        if np.any(channel_labels >= 0):
            reference &= channel_labels == 0
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
        raise ValueError("reject.gaze is unsupported; hard gaze rules belong in eligibility.gaze.")
    gaze_thresholds: dict[str, float | int] = {}
    if "gaze" in review_config:
        _require_eyegaze_channels(epochs, "review.gaze")
        gaze_thresholds = _normalize_gaze_rule(
            review_config["gaze"], gaze_geometry, context="review.gaze"
        )
    validate_eligibility_result(epochs, eligibility)
    autoreject_bad_epochs = np.asarray(autoreject_bad_epochs, dtype=bool)
    if autoreject_bad_epochs.shape != (len(epochs),):
        raise ValueError("autoreject_bad_epochs must contain one boolean value per epoch.")

    reject_time_window = reject_config.get("time_window")
    review_time_window = review_config.get("time_window", reject_time_window)
    hf_cache: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] = {}

    def high_frequency_mask(config, time_window):
        low, high, window = _validate_high_frequency_config(epochs, config, time_window=time_window)
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
        hard_masks["high_frequency_noise"] = high_frequency_mask(hard_hf, reject_time_window)

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
