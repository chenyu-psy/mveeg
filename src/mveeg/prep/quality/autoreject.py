"""Signal-quality steps used by the 0.3 preprocessing pipeline."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real

import autoreject as autoreject_package
import mne
import numpy as np
from autoreject import AutoReject
from sklearn.model_selection import KFold

from .eligibility import _channel_type_mask


@dataclass(frozen=True)
class AutorejectResult:
    """AutoReject output expanded back to the complete epoch order."""

    epochs: mne.Epochs
    bad_epochs: np.ndarray
    interpolated_channels: np.ndarray
    labels: np.ndarray
    eeg_channels: tuple[str, ...]
    diagnostics: Mapping[str, object] | None = None


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
        raise TypeError(f"autoreject.{name} must be a number or a sequence of numbers.") from error
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
            "autoreject.ignore_channels is unsupported; use autoreject.exclude_channels instead."
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
            f"autoreject.exclude_channels only supports EEG channels; received: {non_eeg}."
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
        "events_hash": sha256(np.ascontiguousarray(epochs.events).tobytes()).hexdigest(),
        "eligible_mask_hash": sha256(np.ascontiguousarray(eligible).tobytes()).hexdigest(),
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
        () if config is None else _autoreject_excluded_channels(config, epochs, eeg_channels)
    )
    info_bad_channels = tuple(channel for channel in eeg_channels if channel in epochs.info["bads"])
    unavailable = set(excluded_channels) | set(info_bad_channels)
    autoreject_picks = [pick for pick in eeg_picks if epochs.ch_names[pick] not in unavailable]
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
        return AutorejectResult(output, bad, interpolated, labels, eeg_channels, diagnostics)
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
        eeg_channels.index(channel) for channel in (*excluded_channels, *info_bad_channels)
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
        output.get_data(copy=False)[good_selected] = repaired_data
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
                [model.threshes_.get(channel, np.nan) for channel in autoreject_channels],
                dtype=float,
            ),
            "loss": np.asarray(model.loss_["eeg"], dtype=float),
            "cv_config": resolved_cv,
            "config_hash": _config_hash(resolved_config),
        }
    )
    return AutorejectResult(output, bad, interpolated, labels, eeg_channels, diagnostics)
