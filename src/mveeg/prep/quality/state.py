"""Current quality-sidecar serialization and validation."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np

from .autoreject import AutorejectResult
from .eligibility import EligibilityResult

QUALITY_SCHEMA_VERSION = 3


def save_quality_state(
    path: str | Path,
    eligibility: EligibilityResult,
    autoreject: AutorejectResult,
) -> None:
    """Save current schema quality state without pickle-backed arrays."""

    diagnostics = dict(autoreject.diagnostics or {})
    consensus = diagnostics.get("consensus_")
    n_interpolate = diagnostics.get("n_interpolate_")
    provenance = {
        "config_hash": str(diagnostics.get("config_hash", "")),
        "versions": diagnostics.get("versions", {}),
        "input_summary": diagnostics.get("input_summary", {}),
    }
    payload: dict[str, np.ndarray] = {
        "quality_schema_version": np.asarray(QUALITY_SCHEMA_VERSION, dtype=np.int16),
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
        "autoreject_consensus": np.asarray(np.nan if consensus is None else consensus, dtype=float),
        "autoreject_n_interpolate": np.asarray(
            -1 if n_interpolate is None else n_interpolate, dtype=int
        ),
        "autoreject_consensus_grid": np.asarray(diagnostics.get("consensus_grid", []), dtype=float),
        "autoreject_n_interpolate_grid": np.asarray(
            diagnostics.get("n_interpolate_grid", []), dtype=int
        ),
        "autoreject_threshold_values": np.asarray(
            diagnostics.get("threshold_values", []), dtype=float
        ),
        "autoreject_loss": np.asarray(diagnostics.get("loss", []), dtype=float),
        "autoreject_cv_json": np.asarray(
            json.dumps(diagnostics.get("cv_config", {}), sort_keys=True, separators=(",", ":")),
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
    """Read only the current quality schema; older sidecars must be regenerated."""

    with np.load(Path(path), allow_pickle=False) as saved:
        schema_version = int(saved["quality_schema_version"].item())
        if schema_version != QUALITY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported quality schema {schema_version}; regenerate schema "
                f"{QUALITY_SCHEMA_VERSION} sidecars."
            )
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
        consensus = float(saved["autoreject_consensus"].item())
        n_interpolate = int(saved["autoreject_n_interpolate"].item())
        threshold_values = saved["autoreject_threshold_values"].astype(float)
        eeg_channels = saved["autoreject_eeg_channels"].astype(str)
        model_channels = saved["autoreject_model_channels"].astype(str)
        threshold_channels = (
            model_channels if len(model_channels) == len(threshold_values) else eeg_channels
        )
        cv_config = json.loads(str(saved["autoreject_cv_json"].item()))
        provenance = json.loads(str(saved["autoreject_provenance_json"].item()))
        autoreject: dict[str, object] = {
            "bad_epochs": saved["autoreject_bad_epochs"].astype(bool),
            "interpolated_channels": saved["autoreject_interpolated_channels"].astype(int),
            "labels": saved["autoreject_labels"].astype(np.int8),
            "eeg_channels": eeg_channels,
            "autoreject_channels": model_channels,
            "excluded_channels": saved["autoreject_excluded_channels"].astype(str),
            "info_bad_channels": saved["autoreject_info_bad_channels"].astype(str),
            "schema_version": schema_version,
            "consensus_": consensus if np.isfinite(consensus) else None,
            "n_interpolate_": n_interpolate if n_interpolate >= 0 else None,
            "consensus_grid": saved["autoreject_consensus_grid"].astype(float),
            "n_interpolate_grid": saved["autoreject_n_interpolate_grid"].astype(int),
            "threshold_values": threshold_values,
            "thresholds": (
                dict(zip(threshold_channels.tolist(), threshold_values.tolist(), strict=True))
                if len(threshold_values) == len(threshold_channels)
                else {}
            ),
            "loss": saved["autoreject_loss"].astype(float),
            "cv_config": cv_config,
            "config_hash": provenance.get("config_hash", ""),
            "versions": provenance.get("versions", {}),
            "input_summary": provenance.get("input_summary", {}),
        }
    return eligibility, autoreject


def validate_eligibility_result(
    epochs: mne.BaseEpochs,
    eligibility: EligibilityResult,
) -> None:
    """Reject stale or corrupt eligibility state before artifact labeling."""

    expected_channels = (len(epochs), len(epochs.ch_names))
    eligible = np.asarray(eligibility.eligible)
    if eligible.shape != (len(epochs),) or eligible.dtype.kind != "b":
        raise ValueError("Eligibility state must contain one boolean value per epoch.")
    reasons = np.asarray(eligibility.channel_reasons)
    if reasons.shape != expected_channels:
        raise ValueError(
            f"Eligibility channel reasons have shape {reasons.shape}; expected {expected_channels}."
        )
    for code, mask in eligibility.rule_masks.items():
        if np.asarray(mask).shape != expected_channels:
            raise ValueError(
                f"Eligibility rule {code!r} has shape {np.asarray(mask).shape}; "
                f"expected {expected_channels}."
            )
