"""Quality-control helpers for trial- and channel-level EEG preprocessing."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject

if TYPE_CHECKING:
    from .core import Preprocess


def _format_log_section(title: str, lines: list[str]) -> str:
    """Format one plain-text log section with a short title and indented rows."""
    formatted_lines = [f"{title}:"]
    formatted_lines.extend([f"  {line}" for line in lines])
    return "\n".join(formatted_lines)


def _format_rule_counts(rule_counts: list[tuple[str, str]]) -> list[str]:
    """Align rule summaries so counts are easy to scan in plain-text logs."""
    if len(rule_counts) == 0:
        return []

    label_width = max(len(label) for label, _ in rule_counts)
    return [f"{label:<{label_width}}  {summary}" for label, summary in rule_counts]


def append_reason(reason_matrix: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    """Append a short quality control label to selected trial-channel cells."""
    reason_matrix[mask] = reason_matrix[mask] + f"{label} "
    return reason_matrix


def summarize_trial_mask(mask: np.ndarray) -> str:
    """Format a trial-level mask as count and percent for logs."""
    n_trials = len(mask)
    n_flagged = int(mask.sum())
    pct = round(n_flagged / n_trials * 100, 1) if n_trials else 0.0
    return f"{n_flagged} ({pct}%)"


def join_trial_reasons(rule_masks: dict[str, np.ndarray]) -> np.ndarray:
    """Collapse per-channel rule masks into one reason string per trial."""
    first_mask = next(iter(rule_masks.values()))
    trial_reasons = []
    for trial_ix in range(first_mask.shape[0]):
        labels = [label for label, mask in rule_masks.items() if mask[trial_ix].any()]
        trial_reasons.append("; ".join(labels))
    return np.array(trial_reasons, dtype=object)


def expand_trial_mask(mask: np.ndarray, keep_mask: np.ndarray, n_trials: int) -> np.ndarray:
    """Expand a subset trial-by-channel mask back to the full trial length.

    Parameters
    ----------
    mask : np.ndarray
        Trial-by-channel mask defined on the kept subset of trials.
    keep_mask : np.ndarray
        Boolean mask showing which original trials were kept in the subset.
    n_trials : int
        Number of trials in the original epochs object.

    Returns
    -------
    np.ndarray
        Trial-by-channel mask aligned to the original trial order.
    """
    full_mask = np.zeros((n_trials, mask.shape[1]), dtype=bool)
    full_mask[keep_mask] = mask
    return full_mask


def _masked_rms(data: np.ndarray | np.ma.MaskedArray, *, axis: int) -> np.ndarray:
    """Compute RMS while safely handling masked values from dynamic windows.

    Parameters
    ----------
    data : np.ndarray | np.ma.MaskedArray
        Input array with values along a time axis.
    axis : int
        Axis used when collapsing values into an RMS metric.

    Returns
    -------
    np.ndarray
        RMS values with masked entries treated as missing values.
    """
    if np.ma.isMaskedArray(data):
        filled = data.filled(np.nan)
        return np.sqrt(np.nanmean(np.square(filled), axis=axis))
    return np.sqrt(np.mean(np.square(data), axis=axis))


def _median_plus_mad_threshold(values: np.ndarray, mad_multiplier: float) -> np.ndarray:
    """Build per-channel thresholds from median plus MAD.

    Parameters
    ----------
    values : np.ndarray
        Trial-by-channel metric values for one subject.
    mad_multiplier : float
        Scale factor applied to MAD before adding it to the median.

    Returns
    -------
    np.ndarray
        One threshold per channel.
    """
    channel_median = np.nanmedian(values, axis=0)
    channel_mad = np.nanmedian(np.abs(values - channel_median), axis=0)
    thresholds = channel_median + mad_multiplier * channel_mad
    return np.where(np.isnan(thresholds), np.inf, thresholds)


def run_hf_noise_qc_rule(
    pre: "Preprocess",
    epochs: mne.Epochs,
    *,
    hf_cfg: dict,
) -> np.ndarray:
    """Flag trial-channel cells with sustained high-frequency EEG noise.

    Parameters
    ----------
    pre : Preprocess
        Configured preprocessing object that defines the rejection window.
    epochs : mne.Epochs
        Epoched data used for one QC stage.
    hf_cfg : dict
        High-frequency QC settings. Expected keys are ``band``,
        ``metric``, ``threshold_mode``, and ``mad_multiplier``.

    Returns
    -------
    np.ndarray
        Boolean trial-by-channel mask in the original channel order.
    """
    if hf_cfg.get("metric", "rms") != "rms":
        raise ValueError("hf_noise metric must be 'rms'.")
    if hf_cfg.get("threshold_mode", "median_plus_mad") != "median_plus_mad":
        raise ValueError("hf_noise threshold_mode must be 'median_plus_mad'.")
    band_low, band_high = hf_cfg["band"]
    filtered_epochs = epochs.copy().load_data()
    filtered_epochs.filter(
        l_freq=band_low,
        h_freq=band_high,
        picks="eeg",
        verbose="ERROR",
    )

    chan_types = np.array(filtered_epochs.info.get_channel_types())
    eeg_mask = chan_types == "eeg"
    hf_mask = np.zeros((len(filtered_epochs), len(filtered_epochs.ch_names)), dtype=bool)
    if not np.any(eeg_mask):
        return hf_mask

    eeg_window_data = pre._get_data_from_rej_period(filtered_epochs)[:, eeg_mask, :]
    rms_values = _masked_rms(eeg_window_data, axis=2)
    thresholds = _median_plus_mad_threshold(rms_values, hf_cfg["mad_multiplier"])
    hf_mask[:, eeg_mask] = rms_values > thresholds[np.newaxis, :]
    return hf_mask


def count_flagged_eeg_channels(epochs: mne.Epochs, mask: np.ndarray) -> np.ndarray:
    """Count flagged EEG channels per trial, excluding eye-tracking channels.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data that define the channel types for ``mask``.
    mask : np.ndarray
        Boolean trial-by-channel mask produced by one or more QC rules.

    Returns
    -------
    np.ndarray
        One count per trial showing how many EEG channels were flagged.

    Notes
    -----
    The QC scripts use ``bad_channels`` to summarize EEG sensor quality.
    Eye-tracking channels are handled by their own gaze rules and should not
    inflate the EEG bad-channel threshold.
    """
    chan_types = np.array(epochs.info.get_channel_types())
    eeg_mask = chan_types == "eeg"
    if not np.any(eeg_mask):
        return np.zeros(mask.shape[0], dtype=int)
    return mask[:, eeg_mask].sum(axis=1)


def replace_clean_epochs(
    epochs: mne.Epochs,
    cleaned_epochs: mne.Epochs,
    keep_mask: np.ndarray,
    autoreject_bad_epoch: np.ndarray,
) -> mne.Epochs:
    """Insert cleaned autoreject data back into the original trial order.

    Parameters
    ----------
    epochs : mne.Epochs
        Original epochs before autoreject.
    cleaned_epochs : mne.Epochs
        Output epochs returned by autoreject after local interpolation.
    keep_mask : np.ndarray
        Boolean mask showing which original trials were sent to autoreject.
    autoreject_bad_epoch : np.ndarray
        Boolean mask showing which original trials autoreject still marked as
        bad and therefore dropped from the cleaned output.

    Returns
    -------
    mne.Epochs
        Copy of ``epochs`` with cleaned data inserted for the trials that
        survived autoreject.
    """
    cleaned_full = epochs.copy().load_data()
    good_keep_mask = keep_mask & ~autoreject_bad_epoch
    if np.any(good_keep_mask):
        cleaned_full._data[good_keep_mask] = cleaned_epochs.get_data()
    return cleaned_full


def split_autoreject_output(
    reject_log,
    *,
    keep_mask: np.ndarray,
    n_trials: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map autoreject bad-epoch and interpolation counts to full trial order.

    Parameters
    ----------
    reject_log : object
        Log returned by autoreject.
    keep_mask : np.ndarray
        Boolean mask showing which original trials were sent to autoreject.
    n_trials : int
        Number of trials in the original epochs object.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Full-length bad-epoch mask and full-length interpolation counts.
    """
    autoreject_bad_epoch = np.zeros(n_trials, dtype=bool)
    autoreject_interp_channels = np.zeros(n_trials, dtype=int)
    raw_bad_epoch = np.asarray(reject_log.bad_epochs, dtype=bool)
    raw_interp_channels = np.sum(np.asarray(reject_log.labels) == 2, axis=1)

    autoreject_bad_epoch[keep_mask] = raw_bad_epoch
    autoreject_interp_channels[keep_mask] = raw_interp_channels
    return autoreject_bad_epoch, autoreject_interp_channels


def run_autoreject_local(
    epochs: mne.Epochs,
    *,
    n_interpolate: list[int] | tuple[int, ...],
    cv: int,
    random_state: int,
    n_jobs: int = 1,
    autoreject_verbose: bool = True,
    progress_callback=None,
) -> tuple[mne.Epochs, object]:
    """Repair small numbers of bad EEG channels within epochs using autoreject.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs before local interpolation.
    n_interpolate : list[int] | tuple[int, ...]
        Candidate counts of EEG channels that autoreject may interpolate in
        one epoch.
    cv : int
        Cross-validation folds used by autoreject while tuning thresholds.
    random_state : int
        Random seed for reproducible autoreject fitting.
    n_jobs : int, optional
        Number of workers used inside autoreject.
    autoreject_verbose : bool, optional
        Whether to show autoreject's built-in verbose/progress output.
    progress_callback : callable | None, optional
        Optional callback that receives short status labels before each major
        autoreject step.

    Returns
    -------
    tuple[mne.Epochs, object]
        Cleaned epochs and the autoreject log describing rejected and
        interpolated channels.
    """
    autoreject_model = AutoReject(
        n_interpolate=list(n_interpolate),
        cv=cv,
        picks="eeg",
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=autoreject_verbose,
    )
    if progress_callback is not None:
        progress_callback("fit autoreject")
    autoreject_model.fit(epochs)

    if progress_callback is not None:
        progress_callback("apply interpolation")
    cleaned_epochs, reject_log = autoreject_model.transform(epochs, return_log=True)
    return cleaned_epochs, reject_log


def build_trial_qc_table(
    pre: "Preprocess",
    epochs: mne.Epochs,
    hard_rule_masks: dict[str, np.ndarray],
    soft_rule_masks: dict[str, np.ndarray],
    *,
    hard_bad_channel_limit: int,
    soft_bad_channel_limit: int,
    hard_hf_bad_channel_limit: int | None = None,
    soft_hf_bad_channel_limit: int | None = None,
    forced_hard_trial: np.ndarray | None = None,
    autoreject_bad_epoch: np.ndarray | None = None,
    autoreject_interp_channels: np.ndarray | None = None,
) -> pd.DataFrame:
    """Create trial-level quality control categories and readable summaries.

    Parameters
    ----------
    pre : Preprocess
        Configured preprocessing object that provides event labels.
    epochs : mne.Epochs
        Epoched data after any optional local interpolation.
    hard_rule_masks : dict[str, np.ndarray]
        Channel-level hard-rejection masks.
    soft_rule_masks : dict[str, np.ndarray]
        Channel-level soft-review masks.
    hard_bad_channel_limit : int
        Reject a trial when hard-flagged channels are at least this limit.
    soft_bad_channel_limit : int
        Mark a trial as unclear when soft-flagged channels are at least this
        limit.
    hard_hf_bad_channel_limit : int | None, optional
        Reject a trial when hard high-frequency-noise channels are at least
        this limit. If ``None``, the HF rule does not trigger trial rejection.
    soft_hf_bad_channel_limit : int | None, optional
        Mark a trial as unclear when soft high-frequency-noise channels are at
        least this limit. If ``None``, the HF rule does not trigger review.
    forced_hard_trial : np.ndarray | None, optional
        Optional precomputed hard-rejection trial mask. This is useful when
        loading saved hard-reject checkpoints.
    autoreject_bad_epoch : np.ndarray | None, optional
        Full-length mask showing which trials autoreject still judged as bad.
    autoreject_interp_channels : np.ndarray | None, optional
        Full-length count of locally interpolated channels per trial.

    Returns
    -------
    pd.DataFrame
        Trial-level QC summary table.
    """
    hard_mask = np.logical_or.reduce(list(hard_rule_masks.values()))
    soft_mask = np.logical_or.reduce(list(soft_rule_masks.values()))

    hard_bad_channel_counts = count_flagged_eeg_channels(epochs, hard_mask)
    soft_bad_channel_counts = count_flagged_eeg_channels(epochs, soft_mask)
    hard_hf_mask = hard_rule_masks.get("HARD_HF_NOISE", np.zeros_like(hard_mask))
    soft_hf_mask = soft_rule_masks.get("SOFT_HF_NOISE", np.zeros_like(soft_mask))
    hard_hf_counts = count_flagged_eeg_channels(epochs, hard_hf_mask)
    soft_hf_counts = count_flagged_eeg_channels(epochs, soft_hf_mask)

    hard_trial = hard_bad_channel_counts >= hard_bad_channel_limit
    soft_trial = soft_bad_channel_counts >= soft_bad_channel_limit
    hard_hf_trial = np.zeros(len(epochs), dtype=bool)
    soft_hf_trial = np.zeros(len(epochs), dtype=bool)
    if hard_hf_bad_channel_limit is not None:
        hard_hf_trial = hard_hf_counts >= hard_hf_bad_channel_limit
    if soft_hf_bad_channel_limit is not None:
        soft_hf_trial = soft_hf_counts >= soft_hf_bad_channel_limit
    if autoreject_bad_epoch is None:
        autoreject_bad_epoch = np.zeros(len(epochs), dtype=bool)
    if autoreject_interp_channels is None:
        autoreject_interp_channels = np.zeros(len(epochs), dtype=int)

    hard_trial |= autoreject_bad_epoch | hard_hf_trial
    if forced_hard_trial is not None:
        hard_trial |= forced_hard_trial
    soft_trial |= soft_hf_trial

    trial_category = np.full(len(epochs), "accepted", dtype=object)
    trial_category[soft_trial] = "unclear"
    trial_category[hard_trial] = "rejected"
    trial_category_codes = {"accepted": 1, "unclear": 2, "rejected": 3}

    return pd.DataFrame(
        {
            "trial_index": np.arange(len(epochs)),
            "trial_type": [pre.event_dict_inv[event_code] for event_code in epochs.events[:, 2]],
            "trial_qc_category": trial_category,
            "trial_qc_code": [trial_category_codes[label] for label in trial_category],
            "hard_flagged_channels": hard_bad_channel_counts,
            "soft_flagged_channels": soft_bad_channel_counts,
            "hf_noise_hard_flag": hard_hf_trial,
            "hf_noise_review_flag": soft_hf_trial,
            "hard_reasons": join_trial_reasons(hard_rule_masks),
            "soft_reasons": join_trial_reasons(soft_rule_masks),
            "autoreject_bad_epoch": autoreject_bad_epoch,
            "autoreject_interp_channels": autoreject_interp_channels,
        }
    )


def build_reject_qc_table(
    pre: "Preprocess",
    epochs: mne.Epochs,
    hard_rule_masks: dict[str, np.ndarray],
    *,
    hard_bad_channel_limit: int,
    hard_hf_bad_channel_limit: int | None = None,
) -> pd.DataFrame:
    """Create a readable trial table for the hard-rejection QC stage.

    Parameters
    ----------
    pre : Preprocess
        Configured preprocessing object that provides event labels.
    epochs : mne.Epochs
        Epoched data before autoreject and soft-review QC.
    hard_rule_masks : dict[str, np.ndarray]
        Channel-level hard-rejection masks.
    hard_bad_channel_limit : int
        Reject a trial when hard-flagged channels are at least this limit.
    hard_hf_bad_channel_limit : int | None, optional
        Reject a trial when hard high-frequency-noise channels are at least
        this limit.

    Returns
    -------
    pd.DataFrame
        Trial-level summary of the hard-rejection step.
    """
    hard_mask = np.logical_or.reduce(list(hard_rule_masks.values()))
    hard_bad_channel_counts = count_flagged_eeg_channels(epochs, hard_mask)
    hard_trial = hard_bad_channel_counts >= hard_bad_channel_limit
    hard_hf_mask = hard_rule_masks.get("HARD_HF_NOISE", np.zeros_like(hard_mask))
    hard_hf_counts = count_flagged_eeg_channels(epochs, hard_hf_mask)
    hard_hf_flag = np.zeros(len(epochs), dtype=bool)
    if hard_hf_bad_channel_limit is not None:
        hard_hf_flag = hard_hf_counts >= hard_hf_bad_channel_limit
    hard_trial |= hard_hf_flag
    return pd.DataFrame(
        {
            "trial_index": np.arange(len(epochs)),
            "trial_type": [pre.event_dict_inv[event_code] for event_code in epochs.events[:, 2]],
            "hard_rejected": hard_trial,
            "hard_qc_code": hard_trial.astype(int) + 1,
            "hard_flagged_channels": hard_bad_channel_counts,
            "hf_noise_hard_flag": hard_hf_flag,
            "hard_reasons": join_trial_reasons(hard_rule_masks),
        }
    )


def attach_trial_qc_to_metadata(epochs: mne.Epochs, trial_qc: pd.DataFrame) -> mne.Epochs:
    """Add trial-level quality control columns to the epochs metadata."""
    metadata = epochs.metadata.copy() if epochs.metadata is not None else pd.DataFrame(index=np.arange(len(epochs)))
    for col in trial_qc.columns:
        metadata[col] = trial_qc[col].to_numpy()
    epochs.metadata = metadata
    return epochs


def run_subject_artifact_qc(
    pre: "Preprocess",
    epochs: mne.Epochs,
    *,
    reject_qc: dict,
    review_qc: dict,
    autoreject_cfg: dict | None = None,
    progress_callback=None,
) -> dict[str, object]:
    """Run artifact checks and build trial-level QC outputs for one subject.

    Parameters
    ----------
    pre : Preprocess
        Configured preprocessing object that provides artifact-check helpers.
    epochs : mne.Epochs
        Epoched data for one subject before trial-level QC labels are attached.
    reject_qc : dict
        Hard-rejection thresholds used before autoreject. Expected keys are
        ``reject_qc["eeg"]`` and ``reject_qc["gaze"]``. Optionally include
        ``reject_qc["hf_noise"]`` for high-frequency noise checks.
    review_qc : dict
        Soft-review thresholds used after autoreject. Expected keys are
        ``review_qc["eeg"]`` and ``review_qc["gaze"]``. Optionally include
        ``review_qc["hf_noise"]`` for high-frequency noise checks.
    autoreject_cfg : dict | None, optional
        Optional local autoreject settings. When enabled, EEG channels are
        locally interpolated before the rule-based QC summary is computed.
    progress_callback : callable | None, optional
        Optional callback that receives short status labels during the QC
        pipeline so scripts can show a more informative progress bar.

    Returns
    -------
    dict[str, object]
        Updated epochs, channel-level artifact labels, rule masks, and the
        trial-level QC table.
    """
    reject_result = run_subject_reject_qc(
        pre,
        epochs,
        reject_qc=reject_qc,
        progress_callback=progress_callback,
    )
    autoreject_result = run_subject_autoreject(
        pre,
        epochs,
        reject_result=reject_result,
        autoreject_cfg=autoreject_cfg,
        progress_callback=progress_callback,
    )
    soft_qc_result = run_subject_soft_review_qc(
        pre,
        autoreject_result["epochs_for_soft_qc"],
        reject_result=reject_result,
        review_qc=review_qc,
        hard_bad_channel_limit=reject_result["hard_bad_channel_limit"],
        autoreject_bad_epoch=autoreject_result["autoreject_bad_epoch"],
        autoreject_interp_channels=autoreject_result["autoreject_interp_channels"],
        progress_callback=progress_callback,
    )
    soft_qc_result["autoreject_log"] = autoreject_result["autoreject_log"]
    return soft_qc_result


def run_subject_reject_qc(
    pre: "Preprocess",
    epochs: mne.Epochs,
    *,
    reject_qc: dict,
    progress_callback=None,
) -> dict[str, object]:
    """Run the hard-rejection QC stage before any autoreject repair.

    Parameters
    ----------
    pre : Preprocess
        Configured preprocessing object that provides artifact-check helpers.
    epochs : mne.Epochs
        Epoched data for one subject before any trial-level QC decisions.
    reject_qc : dict
        Hard-rejection thresholds used before autoreject. Expected keys are
        ``reject_qc["eeg"]`` and ``reject_qc["gaze"]``. Optionally include
        ``reject_qc["hf_noise"]`` for high-frequency noise checks.
    progress_callback : callable | None, optional
        Optional callback that receives short status labels during QC.

    Returns
    -------
    dict[str, object]
        Hard-rule masks, flagged-channel counts, and the trial-level hard
        rejection mask.

    Notes
    -----
    Rules are run in a cost-aware order. After each step, trials that already
    meet the hard-rejection threshold are removed from later (more expensive)
    checks to reduce runtime on large datasets.
    """
    reject_eeg = reject_qc["eeg"]
    reject_gaze = reject_qc["gaze"]
    reject_hf = reject_qc.get("hf_noise")
    n_trials = len(epochs)
    empty_mask = np.zeros((n_trials, len(epochs.ch_names)), dtype=bool)
    hard_rule_masks = {
        "DROP": empty_mask.copy(),
        "FLAT": empty_mask.copy(),
        "HARD_ABS": empty_mask.copy(),
        "HARD_GAZE_DEVIATION": empty_mask.copy(),
        "HARD_GAZE_SHIFT": empty_mask.copy(),
        "HARD_STEP": empty_mask.copy(),
        "HARD_P2P": empty_mask.copy(),
        "HARD_HF_NOISE": empty_mask.copy(),
    }

    hard_bad_channel_counts = np.zeros(n_trials, dtype=int)
    hard_trial = np.zeros(n_trials, dtype=bool)

    def update_hard_trial_from_masks() -> None:
        """Update trial-level hard decisions from the current rule masks."""
        nonlocal hard_bad_channel_counts, hard_trial
        combined_mask = np.logical_or.reduce(list(hard_rule_masks.values()))
        hard_bad_channel_counts = count_flagged_eeg_channels(epochs, combined_mask)
        hard_trial = hard_bad_channel_counts >= reject_eeg["bad_channels"]

    def run_rule_for_remaining(
        *,
        step_label: str,
        rule_key: str,
        rule_fn,
    ) -> None:
        """Run one hard rule only on trials not yet hard rejected."""
        candidate_mask = ~hard_trial
        if not np.any(candidate_mask):
            return
        if progress_callback is not None:
            progress_callback(step_label)
        candidate_epochs = epochs[candidate_mask]
        candidate_rule_mask = rule_fn(candidate_epochs)
        hard_rule_masks[rule_key] = expand_trial_mask(candidate_rule_mask, candidate_mask, n_trials)
        update_hard_trial_from_masks()

    run_rule_for_remaining(
        step_label="dropout",
        rule_key="DROP",
        rule_fn=lambda ep: pre.artreject_nan(ep),
    )
    run_rule_for_remaining(
        step_label="flatline",
        rule_key="FLAT",
        rule_fn=lambda ep: pre.artreject_flatline(ep, rejection_criteria={"eeg": 0, "eyegaze": 0}, flatline_duration=200),
    )
    run_rule_for_remaining(
        step_label="hard abs",
        rule_key="HARD_ABS",
        rule_fn=lambda ep: pre.artreject_value(ep, rejection_criteria={"eeg": reject_eeg["absolute_value"]}),
    )
    run_rule_for_remaining(
        step_label="gaze deviation",
        rule_key="HARD_GAZE_DEVIATION",
        rule_fn=lambda ep: pre.artreject_value(ep, rejection_criteria={"eyegaze": pre.deg2pix(reject_gaze["deviation"])}),
    )
    run_rule_for_remaining(
        step_label="gaze shift",
        rule_key="HARD_GAZE_SHIFT",
        rule_fn=lambda ep: pre.artreject_step(
            ep,
            rejection_criteria={"eyegaze": pre.deg2pix(reject_gaze["shift"])},
            win=80,
            win_step=10,
        ),
    )
    run_rule_for_remaining(
        step_label="hard step",
        rule_key="HARD_STEP",
        rule_fn=lambda ep: pre.artreject_step(ep, rejection_criteria={"eeg": reject_eeg["step"]}, win=250, win_step=20),
    )
    run_rule_for_remaining(
        step_label="hard p2p",
        rule_key="HARD_P2P",
        rule_fn=lambda ep: pre.artreject_slidingP2P(ep, rejection_criteria={"eeg": reject_eeg["p2p"]}, win=200, win_step=100),
    )

    if reject_hf is not None:
        run_rule_for_remaining(
            step_label="hard hf_noise",
            rule_key="HARD_HF_NOISE",
            rule_fn=lambda ep: run_hf_noise_qc_rule(pre, ep, hf_cfg=reject_hf),
        )

    hard_mask = np.logical_or.reduce(list(hard_rule_masks.values()))
    hard_hf_counts = count_flagged_eeg_channels(epochs, hard_rule_masks["HARD_HF_NOISE"])
    if reject_hf is not None:
        hard_trial |= hard_hf_counts >= reject_hf["bad_channels"]
    return {
        "hard_rule_masks": hard_rule_masks,
        "hard_mask": hard_mask,
        "hard_bad_channel_counts": hard_bad_channel_counts,
        "hard_trial": hard_trial,
        "hard_bad_channel_limit": reject_eeg["bad_channels"],
        "hard_hf_bad_channel_limit": reject_hf.get("bad_channels") if reject_hf is not None else None,
    }


def run_subject_autoreject(
    pre: "Preprocess",
    epochs: mne.Epochs,
    *,
    reject_result: dict[str, object],
    autoreject_cfg: dict | None = None,
    progress_callback=None,
) -> dict[str, object]:
    """Run only autoreject and return cleaned epochs plus trial-level flags.

    Parameters
    ----------
    pre : Preprocess
        Configured preprocessing object used by lower-level QC helpers.
    epochs : mne.Epochs
        Epoched data for one subject before soft-review QC labels are attached.
    reject_result : dict[str, object]
        Output from ``run_subject_reject_qc``.
    autoreject_cfg : dict | None, optional
        Optional local autoreject settings. When disabled, this function
        returns the input epochs and all-false autoreject flags. Use
        ``autoreject_cfg["verbose"]`` to control autoreject's own progress
        output.
    progress_callback : callable | None, optional
        Optional callback that receives short status labels during autoreject.

    Returns
    -------
    dict[str, object]
        Cleaned epochs plus full-length autoreject masks and interpolation
        counts aligned to the original trial order.
    """
    _ = pre
    hard_trial = reject_result["hard_trial"]
    autoreject_log = None
    autoreject_bad_epoch = np.zeros(len(epochs), dtype=bool)
    autoreject_interp_channels = np.zeros(len(epochs), dtype=int)
    epochs_for_soft_qc = epochs.copy()

    autoreject_keep_mask = ~hard_trial
    if autoreject_cfg is not None and autoreject_cfg.get("enabled", False) and np.any(autoreject_keep_mask):
        epochs_to_repair = epochs[autoreject_keep_mask]
        cleaned_epochs, autoreject_log = run_autoreject_local(
            epochs_to_repair,
            n_interpolate=autoreject_cfg.get("n_interpolate", [1, 2]),
            cv=autoreject_cfg.get("cv", 10),
            random_state=autoreject_cfg.get("random_state"),
            n_jobs=autoreject_cfg.get("n_jobs", 1),
            autoreject_verbose=autoreject_cfg.get("verbose", True),
            progress_callback=progress_callback,
        )
        autoreject_bad_epoch, autoreject_interp_channels = split_autoreject_output(
            autoreject_log,
            keep_mask=autoreject_keep_mask,
            n_trials=len(epochs),
        )
        epochs_for_soft_qc = replace_clean_epochs(
            epochs,
            cleaned_epochs,
            autoreject_keep_mask,
            autoreject_bad_epoch,
        )
    return {
        "epochs_for_soft_qc": epochs_for_soft_qc,
        "autoreject_log": autoreject_log,
        "autoreject_bad_epoch": autoreject_bad_epoch,
        "autoreject_interp_channels": autoreject_interp_channels,
        "autoreject_keep_mask": autoreject_keep_mask,
    }


def run_subject_soft_review_qc(
    pre: "Preprocess",
    epochs: mne.Epochs,
    *,
    reject_result: dict[str, object],
    review_qc: dict,
    hard_bad_channel_limit: int,
    autoreject_bad_epoch: np.ndarray | None = None,
    autoreject_interp_channels: np.ndarray | None = None,
    progress_callback=None,
) -> dict[str, object]:
    """Run soft-review QC and final trial labeling using fixed autoreject flags.

    Parameters
    ----------
    pre : Preprocess
        Configured preprocessing object that provides artifact-check helpers.
    epochs : mne.Epochs
        Epoched data after optional local autoreject interpolation.
    reject_result : dict[str, object]
        Output from ``run_subject_reject_qc``.
    review_qc : dict
        Soft-review thresholds used after autoreject. Expected keys are
        ``review_qc["eeg"]`` and ``review_qc["gaze"]``. Optionally include
        ``review_qc["hf_noise"]`` for high-frequency noise checks.
    hard_bad_channel_limit : int
        Hard-rejection trial threshold from the configured reject QC settings.
    autoreject_bad_epoch : np.ndarray | None, optional
        Full-length mask showing which trials autoreject still judged as bad.
    autoreject_interp_channels : np.ndarray | None, optional
        Full-length count of locally interpolated channels per trial.
    progress_callback : callable | None, optional
        Optional callback that receives short status labels during soft-review.

    Returns
    -------
    dict[str, object]
        Updated epochs, channel-level artifact labels, rule masks, and the
        trial-level QC table.
    """
    hard_rule_masks = reject_result["hard_rule_masks"]
    hard_p2p = hard_rule_masks["HARD_P2P"]
    hard_steps = hard_rule_masks["HARD_STEP"]
    hard_eeg_absolute = hard_rule_masks["HARD_ABS"]
    hard_hf_noise = hard_rule_masks.get("HARD_HF_NOISE", np.zeros_like(hard_p2p))
    hard_gaze_deviation = hard_rule_masks["HARD_GAZE_DEVIATION"]
    hard_gaze_shift = hard_rule_masks["HARD_GAZE_SHIFT"]
    flatline = hard_rule_masks["FLAT"]
    hard_trial = reject_result["hard_trial"]

    review_eeg = review_qc["eeg"]
    review_gaze = review_qc["gaze"]
    review_hf = review_qc.get("hf_noise")

    if autoreject_bad_epoch is None:
        autoreject_bad_epoch = np.zeros(len(epochs), dtype=bool)
    if autoreject_interp_channels is None:
        autoreject_interp_channels = np.zeros(len(epochs), dtype=int)

    soft_candidate_mask = ~(hard_trial | autoreject_bad_epoch)
    soft_rule_masks = {
        "SOFT_P2P": np.zeros_like(hard_p2p),
        "SOFT_STEP": np.zeros_like(hard_steps),
        "SOFT_ABS": np.zeros_like(hard_eeg_absolute),
        "SOFT_HF_NOISE": np.zeros_like(hard_hf_noise),
        "SOFT_GAZE_DEVIATION": np.zeros_like(hard_gaze_deviation),
        "SOFT_GAZE_SHIFT": np.zeros_like(hard_gaze_shift),
        "SOFT_LIN": np.zeros_like(hard_p2p),
    }

    if np.any(soft_candidate_mask):
        if progress_callback is not None:
            progress_callback("run soft QC")
        soft_epochs = epochs[soft_candidate_mask]

        soft_p2p = pre.artreject_slidingP2P(
            soft_epochs, rejection_criteria={"eeg": review_eeg["p2p"]}, win=200, win_step=100
        )
        soft_steps = pre.artreject_step(
            soft_epochs, rejection_criteria={"eeg": review_eeg["step"]}, win=250, win_step=20
        )
        soft_absolute_value = pre.artreject_value(
            soft_epochs,
            rejection_criteria={"eeg": review_eeg["absolute_value"]},
        )
        soft_gaze_deviation = pre.artreject_value(
            soft_epochs,
            rejection_criteria={"eyegaze": pre.deg2pix(review_gaze["deviation"])},
        )
        soft_gaze_shift = pre.artreject_step(
            soft_epochs,
            rejection_criteria={"eyegaze": pre.deg2pix(review_gaze["shift"])},
            win=80,
            win_step=10,
        )
        linear_fit = pre.artreject_linear(soft_epochs)
        soft_hf_noise = np.zeros_like(soft_p2p)
        if review_hf is not None:
            soft_hf_noise = run_hf_noise_qc_rule(pre, soft_epochs, hf_cfg=review_hf)

        soft_rule_masks = {
            "SOFT_P2P": expand_trial_mask(soft_p2p & ~hard_p2p[soft_candidate_mask], soft_candidate_mask, len(epochs)),
            "SOFT_STEP": expand_trial_mask(soft_steps & ~hard_steps[soft_candidate_mask], soft_candidate_mask, len(epochs)),
            "SOFT_ABS": expand_trial_mask(
                soft_absolute_value & ~hard_eeg_absolute[soft_candidate_mask],
                soft_candidate_mask,
                len(epochs),
            ),
            "SOFT_HF_NOISE": expand_trial_mask(
                soft_hf_noise & ~hard_hf_noise[soft_candidate_mask],
                soft_candidate_mask,
                len(epochs),
            ),
            "SOFT_GAZE_DEVIATION": expand_trial_mask(
                soft_gaze_deviation & ~hard_gaze_deviation[soft_candidate_mask],
                soft_candidate_mask,
                len(epochs),
            ),
            "SOFT_GAZE_SHIFT": expand_trial_mask(
                soft_gaze_shift & ~hard_gaze_shift[soft_candidate_mask],
                soft_candidate_mask,
                len(epochs),
            ),
            "SOFT_LIN": expand_trial_mask(linear_fit, soft_candidate_mask, len(epochs)),
        }

    artifact_labels = np.char.array(np.full(flatline.shape, "", dtype="<U160"))
    for label, mask in hard_rule_masks.items():
        artifact_labels = append_reason(artifact_labels, mask, label)
    for label, mask in soft_rule_masks.items():
        artifact_labels = append_reason(artifact_labels, mask, label)

    trial_qc_table = build_trial_qc_table(
        pre,
        epochs,
        hard_rule_masks,
        soft_rule_masks,
        hard_bad_channel_limit=hard_bad_channel_limit,
        soft_bad_channel_limit=review_eeg["bad_channels"],
        hard_hf_bad_channel_limit=reject_result.get("hard_hf_bad_channel_limit"),
        soft_hf_bad_channel_limit=review_hf.get("bad_channels") if review_hf is not None else None,
        forced_hard_trial=reject_result.get("hard_trial"),
        autoreject_bad_epoch=autoreject_bad_epoch,
        autoreject_interp_channels=autoreject_interp_channels,
    )
    epochs = attach_trial_qc_to_metadata(epochs, trial_qc_table)

    if progress_callback is not None:
        progress_callback("attach QC labels")
    return {
        "epochs": epochs,
        "artifact_labels": artifact_labels,
        "trial_qc": trial_qc_table,
        "hard_rule_masks": hard_rule_masks,
        "soft_rule_masks": soft_rule_masks,
    }


def log_subject_qc_summary(qc_result: dict[str, object], reject_qc: dict, review_qc: dict) -> None:
    """Print a readable summary of one subject's QC results.

    Parameters
    ----------
    qc_result : dict[str, object]
        Output from ``run_subject_artifact_qc``.
    reject_qc : dict
        Hard-rejection QC thresholds used for this subject.
    review_qc : dict
        Soft-review QC thresholds used for this subject.

    Returns
    -------
    None
        The function prints QC summaries to stdout.
    """
    trial_qc = qc_result["trial_qc"]
    hard_rule_masks = qc_result["hard_rule_masks"]
    soft_rule_masks = qc_result["soft_rule_masks"]
    epochs = qc_result["epochs"]
    artifact_labels = qc_result["artifact_labels"]
    reject_limit = reject_qc["eeg"]["bad_channels"]
    review_limit = review_qc["eeg"]["bad_channels"]

    rejected_trial_mask = trial_qc["trial_qc_category"].eq("rejected").to_numpy()
    review_trial_mask = trial_qc["trial_qc_category"].eq("unclear").to_numpy()
    accepted_trial_mask = trial_qc["trial_qc_category"].eq("accepted").to_numpy()
    hard_hf_noise = hard_rule_masks.get("HARD_HF_NOISE", np.zeros_like(hard_rule_masks["HARD_P2P"]))
    soft_hf_noise = soft_rule_masks.get("SOFT_HF_NOISE", np.zeros_like(soft_rule_masks["SOFT_P2P"]))

    hard_rule_lines = _format_rule_counts(
        [
            ("Large EEG peak-to-peak", summarize_trial_mask(hard_rule_masks["HARD_P2P"].any(1))),
            ("Large EEG steps", summarize_trial_mask(hard_rule_masks["HARD_STEP"].any(1))),
            ("Large EEG amplitude", summarize_trial_mask(hard_rule_masks["HARD_ABS"].any(1))),
            ("HF noise bursts", summarize_trial_mask(hard_hf_noise.any(1))),
            ("Large gaze deviation", summarize_trial_mask(hard_rule_masks["HARD_GAZE_DEVIATION"].any(1))),
            ("Large gaze shifts", summarize_trial_mask(hard_rule_masks["HARD_GAZE_SHIFT"].any(1))),
            ("Flatline", summarize_trial_mask(hard_rule_masks["FLAT"].any(1))),
            ("Dropout", summarize_trial_mask(hard_rule_masks["DROP"].any(1))),
        ]
    )
    soft_rule_lines = _format_rule_counts(
        [
            ("Mild EEG peak-to-peak", summarize_trial_mask(soft_rule_masks["SOFT_P2P"].any(1))),
            ("Mild EEG steps", summarize_trial_mask(soft_rule_masks["SOFT_STEP"].any(1))),
            ("Mild EEG amplitude", summarize_trial_mask(soft_rule_masks["SOFT_ABS"].any(1))),
            ("HF noise check", summarize_trial_mask(soft_hf_noise.any(1))),
            ("Mild gaze deviation", summarize_trial_mask(soft_rule_masks["SOFT_GAZE_DEVIATION"].any(1))),
            ("Small gaze shifts", summarize_trial_mask(soft_rule_masks["SOFT_GAZE_SHIFT"].any(1))),
            ("Linear drift", summarize_trial_mask(soft_rule_masks["SOFT_LIN"].any(1))),
        ]
    )
    top_channel_counts = (artifact_labels != "").sum(0)
    top_channel_lines = [
        f"{epochs.ch_names[i]:<10} {int(top_channel_counts[i])}"
        for i in np.argsort(top_channel_counts)[::-1][0:5]
    ]

    print(
        "\n".join(
            [
                _format_log_section(
                    "QC overview",
                    [
                        f"Rejected trials   {summarize_trial_mask(rejected_trial_mask)}",
                        f"Unclear trials    {summarize_trial_mask(review_trial_mask)}",
                        f"Accepted trials   {summarize_trial_mask(accepted_trial_mask)}",
                    ],
                ),
                "",
                _format_log_section(
                    "Trial decision thresholds",
                    [
                        (
                            "Reject if hard-flagged channels >= "
                            f"{reject_limit}"
                        ),
                        (
                            "Mark unclear if soft-flagged channels >= "
                            f"{review_limit}"
                        ),
                    ],
                ),
                "",
                _format_log_section("Hard rejection rules", hard_rule_lines),
                "",
                _format_log_section("Soft review rules", soft_rule_lines),
                "",
                _format_log_section("Most-flagged channels", top_channel_lines),
            ]
        )
    )


def _count_reasons(reason_values: pd.Series) -> Counter:
    """Count rejection labels from a semicolon-separated reason column."""
    counts: Counter = Counter()
    for value in reason_values.dropna().astype(str):
        for label in value.split(";"):
            clean_label = label.strip()
            if clean_label != "":
                counts[clean_label] += 1
    return counts


def _format_top_reasons(reason_counts: Counter, *, top_n: int) -> str:
    """Format the most common rejection reasons as ``label=count`` entries."""
    if len(reason_counts) == 0:
        return "none"
    top_items = reason_counts.most_common(top_n)
    return "; ".join(f"{label}={count}" for label, count in top_items)


def summarize_trial_rejection(
    data_dir: str | Path,
    *,
    stage: str = "auto",
    hard_stage: str = "prepared",
    top_n_reasons: int = 3,
) -> None:
    """Summarize current saved trial rejection status at any preprocessing step.

    Parameters
    ----------
    data_dir : str | Path
        Preprocessed data root that contains the ``derivatives`` folder.
    stage : str, optional
        Which saved checkpoint stage to summarize. Use ``"hard"`` for
        ``*_hard_qc.tsv``, ``"final"`` for ``*_trial_qc.tsv``, or ``"auto"``
        to use final when available and hard otherwise. ``"auto"`` lets one
        call work after step 1/3, 2/3, or 3/3.
    hard_stage : str, optional
        Intermediate stage label used in hard-reject filenames.
    top_n_reasons : int, optional
        Number of most common rejection reasons to display per subject.

    Returns
    -------
    None
        Prints a compact multi-line summary per subject.
    """
    if stage not in {"auto", "hard", "final"}:
        raise ValueError("stage must be one of: auto, hard, final")
    if top_n_reasons < 1:
        raise ValueError("top_n_reasons must be >= 1")

    derivatives_dir = Path(data_dir) / "derivatives"
    subject_dirs = sorted(path for path in derivatives_dir.glob("sub-*") if path.is_dir())
    if len(subject_dirs) == 0:
        print("No subject folders found in derivatives yet.")
        return

    def print_subject_block(
        *,
        subject_label: str,
        stage_label: str,
        accepted: int,
        rejected: int,
        unclear: int,
        hard_hf: int,
        review_hf: int,
        top_reasons: str,
    ) -> None:
        """Print one readable subject-level rejection summary block."""
        print(f"{subject_label} [{stage_label}]")
        print(f"  trials : accepted={accepted}, rejected={rejected}, unclear={unclear}")
        print(f"  hf     : hard={hard_hf}, review={review_hf}")
        print(f"  reasons: {top_reasons}")
        print()

    print("Saved trial rejection summary:")
    for subject_dir in subject_dirs:
        eeg_dir = subject_dir / "eeg"
        final_paths = sorted(eeg_dir.glob("*_trial_qc.tsv"))
        hard_paths = sorted(eeg_dir.glob(f"*_{hard_stage}_hard_qc.tsv"))

        use_final = stage == "final" or (stage == "auto" and len(final_paths) > 0)
        use_hard = stage == "hard" or (stage == "auto" and len(final_paths) == 0)

        if use_final:
            if len(final_paths) == 0:
                print(f"{subject_dir.name}: final checkpoint not found")
                continue
            trial_qc = pd.read_csv(final_paths[0], sep="\t")
            summary_col = "final_qc_category" if "final_qc_category" in trial_qc.columns else "trial_qc_category"
            counts = trial_qc[summary_col].value_counts()
            accepted = int(counts.get("accepted", 0))
            rejected = int(counts.get("rejected", 0))
            unclear = int(counts.get("unclear", 0))
            hard_hf = int(trial_qc["hf_noise_hard_flag"].sum()) if "hf_noise_hard_flag" in trial_qc.columns else 0
            review_hf = int(trial_qc["hf_noise_review_flag"].sum()) if "hf_noise_review_flag" in trial_qc.columns else 0
            reject_rows = trial_qc[trial_qc[summary_col].eq("rejected")]
            reason_col = "hard_reasons" if "hard_reasons" in reject_rows.columns else None
            reject_reason_counts = _count_reasons(reject_rows[reason_col]) if reason_col is not None else Counter()
            top_reasons = _format_top_reasons(reject_reason_counts, top_n=top_n_reasons)
            print_subject_block(
                subject_label=subject_dir.name,
                stage_label="final",
                accepted=accepted,
                rejected=rejected,
                unclear=unclear,
                hard_hf=hard_hf,
                review_hf=review_hf,
                top_reasons=top_reasons,
            )
            continue

        if use_hard:
            if len(hard_paths) == 0:
                print(f"{subject_dir.name}: hard checkpoint not found")
                continue
            hard_qc = pd.read_csv(hard_paths[0], sep="\t")
            if "hard_rejected" not in hard_qc.columns:
                print(f"{subject_dir.name}: missing 'hard_rejected' column")
                continue
            rejected_mask = hard_qc["hard_rejected"].astype(bool)
            n_total = int(len(hard_qc))
            rejected = int(rejected_mask.sum())
            accepted = n_total - rejected
            hard_hf = int(hard_qc["hf_noise_hard_flag"].sum()) if "hf_noise_hard_flag" in hard_qc.columns else 0
            reject_reason_counts = (
                _count_reasons(hard_qc.loc[rejected_mask, "hard_reasons"])
                if "hard_reasons" in hard_qc.columns
                else Counter()
            )
            top_reasons = _format_top_reasons(reject_reason_counts, top_n=top_n_reasons)
            print_subject_block(
                subject_label=subject_dir.name,
                stage_label="hard",
                accepted=accepted,
                rejected=rejected,
                unclear=0,
                hard_hf=hard_hf,
                review_hf=0,
                top_reasons=top_reasons,
            )

    print()
