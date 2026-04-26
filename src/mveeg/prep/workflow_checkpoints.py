"""Checkpoint and trial-state helpers for preprocessing workflows."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd


TRIAL_STATE_COLUMNS = [
    "trial_index",
    "trial_type",
    "hard_reject",
    "hard_reasons",
    "hard_flagged_channels",
    "hf_noise_hard_flag",
    "autoreject_reject",
    "autoreject_interp_channels",
    "soft_reject",
    "final_keep",
    "final_qc_category",
    "state_stage",
]


def load_trial_state(flow, subject_number: str) -> pd.DataFrame | None:
    """Load the saved unified trial-state table when it exists."""
    trial_state_path = flow.trial_state_path(subject_number)
    if not trial_state_path.exists():
        return None
    return pd.read_csv(trial_state_path, sep="\t", keep_default_na=False)


def write_trial_state(flow, subject_number: str, trial_state: pd.DataFrame) -> Path:
    """Write one subject's unified trial-state table in a stable layout."""
    trial_state_path = flow.trial_state_path(subject_number)
    trial_state_path.parent.mkdir(parents=True, exist_ok=True)

    trial_state = trial_state.copy()
    for column in TRIAL_STATE_COLUMNS:
        if column not in trial_state.columns:
            if column.endswith("_reject") or column in {"hf_noise_hard_flag", "final_keep"}:
                trial_state[column] = False
            elif column.endswith("_channels") or column.endswith("_index"):
                trial_state[column] = 0
            else:
                trial_state[column] = ""

    trial_state = trial_state.loc[:, TRIAL_STATE_COLUMNS].sort_values("trial_index").reset_index(drop=True)
    trial_state.to_csv(trial_state_path, sep="\t", index=False)
    return trial_state_path


def save_reject_trial_state(flow, subject_number: str, reject_table: pd.DataFrame) -> Path:
    """Save unified trial state after the hard-reject stage."""
    trial_state = load_trial_state(flow, subject_number)
    if trial_state is None:
        trial_state = reject_table.loc[:, ["trial_index", "trial_type"]].copy()
    else:
        trial_state = trial_state.copy()

    trial_state["hard_reject"] = reject_table["hard_rejected"].to_numpy(dtype=bool)
    trial_state["hard_reasons"] = reject_table["hard_reasons"].astype(str).to_numpy()
    trial_state["hard_flagged_channels"] = reject_table["hard_flagged_channels"].to_numpy(dtype=int)
    trial_state["hf_noise_hard_flag"] = reject_table["hf_noise_hard_flag"].to_numpy(dtype=bool)
    if "autoreject_reject" not in trial_state.columns:
        trial_state["autoreject_reject"] = False
    if "soft_reject" not in trial_state.columns:
        trial_state["soft_reject"] = False
    trial_state["final_keep"] = ~trial_state["hard_reject"].to_numpy(dtype=bool)
    trial_state["final_qc_category"] = np.where(
        trial_state["final_keep"].to_numpy(dtype=bool),
        "accepted",
        "rejected",
    )
    trial_state["state_stage"] = "hard"
    return write_trial_state(flow, subject_number, trial_state)


def save_autoreject_trial_state(
    flow,
    subject_number: str,
    *,
    autoreject_bad_epoch: np.ndarray,
    autoreject_interp_channels: np.ndarray,
) -> Path:
    """Save unified trial state after autoreject decisions are known."""
    trial_state = load_trial_state(flow, subject_number)
    if trial_state is None:
        trial_state = pd.DataFrame({"trial_index": np.arange(len(autoreject_bad_epoch), dtype=int)})
        trial_state["trial_type"] = ""

    trial_state = trial_state.copy()
    if "hard_reject" not in trial_state.columns:
        trial_state["hard_reject"] = False
    if "soft_reject" not in trial_state.columns:
        trial_state["soft_reject"] = False
    trial_state["autoreject_reject"] = np.asarray(autoreject_bad_epoch, dtype=bool)
    trial_state["autoreject_interp_channels"] = np.asarray(autoreject_interp_channels, dtype=int)
    trial_state["final_keep"] = ~(
        trial_state["hard_reject"].to_numpy(dtype=bool)
        | trial_state["autoreject_reject"].to_numpy(dtype=bool)
        | trial_state["soft_reject"].to_numpy(dtype=bool)
    )
    trial_state["final_qc_category"] = np.where(
        trial_state["final_keep"].to_numpy(dtype=bool),
        "accepted",
        "rejected",
    )
    trial_state["state_stage"] = "autoreject"
    return write_trial_state(flow, subject_number, trial_state)


def save_final_trial_state(flow, subject_number: str, trial_qc: pd.DataFrame) -> Path:
    """Save unified trial state after the final QC table is produced."""
    trial_state = load_trial_state(flow, subject_number)
    if trial_state is None:
        trial_state = trial_qc.loc[:, ["trial_index", "trial_type"]].copy()
    else:
        trial_state = trial_state.copy()

    final_qc = (
        trial_qc["final_qc_category"].astype(str)
        if "final_qc_category" in trial_qc.columns
        else trial_qc["trial_qc_category"].astype(str)
    )
    if "hard_reject" not in trial_state.columns:
        trial_state["hard_reject"] = trial_qc["trial_qc_category"].astype(str).eq("rejected").to_numpy()
    if "autoreject_reject" not in trial_state.columns and "autoreject_bad_epoch" in trial_qc.columns:
        trial_state["autoreject_reject"] = trial_qc["autoreject_bad_epoch"].to_numpy(dtype=bool)
    elif "autoreject_reject" not in trial_state.columns:
        trial_state["autoreject_reject"] = False

    trial_state["soft_reject"] = trial_qc["trial_qc_category"].astype(str).eq("unclear").to_numpy(dtype=bool)
    trial_state["final_keep"] = final_qc.eq("accepted").to_numpy(dtype=bool)
    trial_state["final_qc_category"] = final_qc.to_numpy(dtype=object)
    if "hard_reasons" in trial_qc.columns:
        trial_state["hard_reasons"] = trial_qc["hard_reasons"].astype(str).to_numpy()
    if "hard_flagged_channels" in trial_qc.columns:
        trial_state["hard_flagged_channels"] = trial_qc["hard_flagged_channels"].to_numpy(dtype=int)
    if "hf_noise_hard_flag" in trial_qc.columns:
        trial_state["hf_noise_hard_flag"] = trial_qc["hf_noise_hard_flag"].to_numpy(dtype=bool)
    if "autoreject_interp_channels" in trial_qc.columns:
        trial_state["autoreject_interp_channels"] = trial_qc["autoreject_interp_channels"].to_numpy(dtype=int)
    trial_state["state_stage"] = "final"
    return write_trial_state(flow, subject_number, trial_state)


def save_subject_reject_checkpoint(
    flow,
    subject_number: str,
    epochs: mne.Epochs,
    reject_result: dict[str, object],
    *,
    stage: str = "prepared",
) -> tuple[Path, Path]:
    """Save hard-reject QC outputs so later steps can resume from them."""
    from . import qc as preprocess_qc

    pre = flow._require_preprocessor()
    reject_qc, _, _ = flow._require_qc_config()
    table_path = flow.reject_qc_table_path(subject_number, stage=stage)
    masks_path = flow.reject_qc_masks_path(subject_number, stage=stage)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    hard_rule_masks = reject_result["hard_rule_masks"]
    reject_table = preprocess_qc.build_reject_qc_table(
        pre,
        epochs,
        hard_rule_masks,
        hard_bad_channel_limit=reject_qc["eeg"]["bad_channels"],
        hard_hf_bad_channel_limit=reject_qc.get("hf_noise", {}).get("bad_channels"),
    )
    reject_table.to_csv(table_path, sep="\t", index=False)
    np.savez_compressed(
        masks_path,
        hard_trial=np.asarray(reject_result["hard_trial"], dtype=bool),
        hard_bad_channel_counts=np.asarray(reject_result["hard_bad_channel_counts"], dtype=int),
        **{label: np.asarray(mask, dtype=bool) for label, mask in hard_rule_masks.items()},
    )
    save_reject_trial_state(flow, subject_number, reject_table)
    return table_path, masks_path


def has_subject_reject_checkpoint(flow, subject_number: str, *, stage: str = "prepared") -> bool:
    """Return whether both hard-reject checkpoint files already exist."""
    table_path = flow.reject_qc_table_path(subject_number, stage=stage)
    masks_path = flow.reject_qc_masks_path(subject_number, stage=stage)
    return table_path.exists() and masks_path.exists()


def save_subject_autoreject_checkpoint(
    flow,
    subject_number: str,
    autoreject_result: dict[str, object],
    *,
    source_stage: str = "prepared",
    cleaned_stage: str = "prepared_autoreject",
    overwrite: bool = True,
) -> tuple[Path, Path]:
    """Save autoreject masks plus cleaned epochs for later soft-review QC."""
    cleaned_epochs_path = flow.final_epochs_path(subject_number)
    cleaned_epochs_path.parent.mkdir(parents=True, exist_ok=True)
    autoreject_result["epochs_for_soft_qc"].save(
        cleaned_epochs_path,
        overwrite=overwrite,
        verbose="ERROR",
    )
    masks_path = flow.autoreject_masks_path(subject_number, stage=source_stage)
    masks_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        masks_path,
        autoreject_bad_epoch=np.asarray(autoreject_result["autoreject_bad_epoch"], dtype=bool),
        autoreject_interp_channels=np.asarray(autoreject_result["autoreject_interp_channels"], dtype=int),
    )
    save_autoreject_trial_state(
        flow,
        subject_number,
        autoreject_bad_epoch=np.asarray(autoreject_result["autoreject_bad_epoch"], dtype=bool),
        autoreject_interp_channels=np.asarray(autoreject_result["autoreject_interp_channels"], dtype=int),
    )
    return cleaned_epochs_path, masks_path


def load_checkpoint(
    flow,
    subject_number: str,
    *,
    kind: str,
    stage: str = "prepared",
    cleaned_stage: str = "prepared_autoreject",
) -> dict[str, object]:
    """Load one checkpoint for one subject."""
    if kind == "reject":
        reject_qc, _, _ = flow._require_qc_config()
        masks_path = flow.reject_qc_masks_path(subject_number, stage=stage)
        with np.load(masks_path) as saved:
            hard_rule_masks = {
                key: saved[key].astype(bool)
                for key in saved.files
                if key not in {"hard_trial", "hard_bad_channel_counts"}
            }
            hard_mask = np.logical_or.reduce(list(hard_rule_masks.values()))
            return {
                "hard_rule_masks": hard_rule_masks,
                "hard_mask": hard_mask,
                "hard_bad_channel_counts": saved["hard_bad_channel_counts"].astype(int),
                "hard_trial": saved["hard_trial"].astype(bool),
                "hard_bad_channel_limit": reject_qc["eeg"]["bad_channels"],
                "hard_hf_bad_channel_limit": reject_qc.get("hf_noise", {}).get("bad_channels"),
            }

    if kind == "autoreject":
        epochs_for_soft_qc = mne.read_epochs(
            flow.final_epochs_path(subject_number),
            preload=True,
            verbose="ERROR",
        )
        masks_path = flow.autoreject_masks_path(subject_number, stage=stage)
        with np.load(masks_path) as saved:
            return {
                "epochs_for_soft_qc": epochs_for_soft_qc,
                "autoreject_bad_epoch": saved["autoreject_bad_epoch"].astype(bool),
                "autoreject_interp_channels": saved["autoreject_interp_channels"].astype(int),
            }

    raise ValueError(f"Unsupported checkpoint kind '{kind}'. Use reject or autoreject.")


def has_subject_autoreject_checkpoint(
    flow,
    subject_number: str,
    *,
    source_stage: str = "prepared",
    cleaned_stage: str = "prepared_autoreject",
) -> bool:
    """Return whether autoreject masks and cleaned epochs both exist."""
    masks_path = flow.autoreject_masks_path(subject_number, stage=source_stage)
    epochs_path = flow.final_epochs_path(subject_number)
    return masks_path.exists() and epochs_path.exists()
