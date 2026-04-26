"""Input/output helpers for the EEG decoding workflow."""

from __future__ import annotations

from pathlib import Path
import json

import mne
import numpy as np
import pandas as pd

from .._shared.io_filters import (
    load_subject_data_with_filters,
    load_subject_info_with_channel_drop,
    load_subject_metadata_table,
)
from ..io.bids import derivative_file_path, get_subject_ids_from_derivatives
from .config import DecodingConfig


def get_subject_ids(
    data_dir: str | Path,
    subject_prefix: str = "sub",
    derivative_dirname: str = "derivatives",
    datatype: str = "eeg",
) -> list[str]:
    """Return subject IDs available in the derivatives folder."""

    return get_subject_ids_from_derivatives(
        data_dir,
        subject_prefix=subject_prefix,
        derivative_dirname=derivative_dirname,
        datatype=datatype,
    )


def load_subject_metadata(subject_id: str, cfg: DecodingConfig) -> pd.DataFrame:
    """Load the metadata table saved for one subject's epochs."""

    return load_subject_metadata_table(subject_id, cfg)


def load_subject_decoding_data(
    subject_id: str,
    cfg: DecodingConfig,
    return_metadata: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load one subject and apply the configured trial filters."""

    return load_subject_data_with_filters(
        subject_id=subject_id,
        cfg=cfg,
        return_metadata=return_metadata,
    )


def load_subject_info(subject_id: str, cfg: DecodingConfig) -> mne.Info:
    """Load epochs info for topography plotting after matching channel drops."""

    return load_subject_info_with_channel_drop(subject_id, cfg)


def subject_result_paths(output_dir: str | Path, subject_id: str) -> dict[str, Path]:
    """Return the saved file paths for one subject's decoding outputs.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level results folder.
    subject_id : str
        Subject identifier without the `sub-` prefix.

    Returns
    -------
    dict[str, Path]
        Standard file paths for one subject's saved outputs.
    """

    output_dir = Path(output_dir)
    return {
        "decoding": output_dir / f"sub-{subject_id}_decoding.npz",
        "hyperplane": output_dir / f"sub-{subject_id}_hyperplane.npz",
        "trial_summary": output_dir / f"sub-{subject_id}_trials.json",
    }


def generalization_subject_result_paths(output_dir: str | Path, subject_id: str) -> dict[str, Path]:
    """Return the saved file paths for one subject's generalization outputs.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level results folder.
    subject_id : str
        Subject identifier without the `sub-` prefix.

    Returns
    -------
    dict[str, Path]
        Standard file paths for one subject's saved generalization outputs.
    """

    output_dir = Path(output_dir)
    return {
        "generalization": output_dir / f"sub-{subject_id}_generalization.npz",
        "trial_summary": output_dir / f"sub-{subject_id}_trials.json",
    }


def list_saved_subject_ids(output_dir: str | Path) -> list[str]:
    """Return decoding subject IDs that have a complete saved cache set.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level results folder.

    Returns
    -------
    list[str]
        Sorted subject IDs whose decoding cache files are all present.
    """

    output_dir = Path(output_dir)
    subject_ids = []
    for trial_summary_path in sorted(output_dir.glob("sub-*_trials.json")):
        subject_id = trial_summary_path.stem.replace("sub-", "").replace("_trials", "")
        if subject_result_exists(output_dir, subject_id):
            subject_ids.append(subject_id)
    return subject_ids


def list_saved_generalization_subject_ids(output_dir: str | Path) -> list[str]:
    """Return generalization subject IDs that have a complete saved cache set.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level results folder.

    Returns
    -------
    list[str]
        Sorted subject IDs whose generalization cache files are all present.
    """

    output_dir = Path(output_dir)
    subject_ids = []
    for trial_summary_path in sorted(output_dir.glob("sub-*_trials.json")):
        subject_id = trial_summary_path.stem.replace("sub-", "").replace("_trials", "")
        if generalization_subject_result_exists(output_dir, subject_id):
            subject_ids.append(subject_id)
    return subject_ids


def save_subject_results(
    output_dir: str | Path,
    subject_id: str,
    times_ms: np.ndarray,
    ch_names: list[str],
    decoding_result: dict[str, np.ndarray | list[str] | int],
    hyperplane_result: dict[str, object],
    trial_summary_row: dict[str, object],
) -> None:
    """Save all subject-level outputs used to rebuild decoding summaries.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level results folder.
    subject_id : str
        Subject identifier without the `sub-` prefix.
    times_ms : np.ndarray
        Decoding time points in milliseconds.
    ch_names : list[str]
        Channel names kept for decoding.
    decoding_result : dict[str, np.ndarray | list[str] | int]
        Subject decoding output returned by the workflow.
    hyperplane_result : dict[str, object]
        Subject hyperplane output returned by the workflow.
    trial_summary_row : dict[str, object]
        Trial summary row for one subject.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = subject_result_paths(output_dir, subject_id)
    labels = np.asarray(decoding_result["label_order"], dtype=object)
    np.savez(
        paths["decoding"],
        times_ms=np.asarray(times_ms),
        accuracy=np.asarray(decoding_result["accuracy"]),
        perm_accuracy=np.asarray(decoding_result["perm_accuracy"]),
        confusion_matrix=np.asarray(decoding_result["confusion_matrix"]),
        channel_weights=np.asarray(decoding_result["channel_weights"]),
        channel_patterns=np.asarray(decoding_result["channel_patterns"]),
        repeat_cv=decoding_result["accuracy_by_repeat"]["cv_repeat"].to_numpy(dtype=int),
        repeat_data_type=decoding_result["accuracy_by_repeat"]["data_type"].to_numpy(dtype=object),
        repeat_perm_id=decoding_result["accuracy_by_repeat"]["perm_id"].to_numpy(dtype=int),
        repeat_time_ix=decoding_result["accuracy_by_repeat"]["time_ix"].to_numpy(dtype=int),
        repeat_n_correct=decoding_result["accuracy_by_repeat"]["n_correct"].to_numpy(dtype=int),
        repeat_n_test_trials=decoding_result["accuracy_by_repeat"]["n_test_trials"].to_numpy(dtype=int),
        repeat_accuracy=decoding_result["accuracy_by_repeat"]["accuracy"].to_numpy(dtype=float),
        repeat_balanced_accuracy=decoding_result["accuracy_by_repeat"]["balanced_accuracy"].to_numpy(dtype=float),
        repeat_chance_level=decoding_result["accuracy_by_repeat"]["chance_level"].to_numpy(dtype=float),
        label_order=labels,
        ch_names=np.asarray(ch_names, dtype=object),
        n_input_trials=np.asarray(decoding_result["n_input_trials"]),
        n_training_trials=np.asarray(decoding_result["n_training_trials"]),
        n_binned_trials=np.asarray(decoding_result["n_binned_trials"]),
    )

    trial_distance = hyperplane_result["trial_distance"]
    np.savez(
        paths["hyperplane"],
        trial_id=trial_distance["trial_id"].to_numpy(dtype=int),
        condition=trial_distance["condition"].to_numpy(dtype=object),
        distance=np.stack(trial_distance["distance"].to_numpy()),
        group_order=np.asarray(hyperplane_result["group_order"], dtype=object),
        label_order=np.asarray(hyperplane_result["label_order"], dtype=object),
    )
    with open(paths["trial_summary"], "w", encoding="utf-8") as f:
        json.dump(trial_summary_row, f, indent=2)


def load_saved_subject_results(
    output_dir: str | Path,
    subject_id: str,
) -> dict[str, object]:
    """Load all saved outputs for one subject.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level results folder.
    subject_id : str
        Subject identifier without the `sub-` prefix.

    Returns
    -------
    dict[str, object]
        Subject-level outputs in the same structure used by the workflow.
    """

    paths = subject_result_paths(output_dir, subject_id)

    with np.load(paths["decoding"], allow_pickle=True) as saved:
        accuracy_by_repeat = pd.DataFrame(
            {
                "cv_repeat": saved["repeat_cv"].astype(int),
                "data_type": saved["repeat_data_type"].astype(object),
                "perm_id": saved["repeat_perm_id"].astype(int),
                "time_ix": saved["repeat_time_ix"].astype(int),
                "n_correct": saved["repeat_n_correct"].astype(int),
                "n_test_trials": saved["repeat_n_test_trials"].astype(int),
                "accuracy": saved["repeat_accuracy"].astype(float),
                "balanced_accuracy": saved["repeat_balanced_accuracy"].astype(float),
                "chance_level": saved["repeat_chance_level"].astype(float),
            }
        )
        result = {
            "times_ms": saved["times_ms"],
            "accuracy": saved["accuracy"],
            "perm_accuracy": saved["perm_accuracy"],
            "confusion_matrix": saved["confusion_matrix"],
            "channel_weights": saved["channel_weights"],
            "channel_patterns": saved["channel_patterns"],
            "accuracy_by_repeat": accuracy_by_repeat,
            "label_order": saved["label_order"].tolist(),
            "ch_names": saved["ch_names"].tolist(),
            "n_input_trials": int(saved["n_input_trials"]),
            "n_training_trials": int(saved["n_training_trials"]),
            "n_binned_trials": int(saved["n_binned_trials"]),
        }

    with np.load(paths["hyperplane"], allow_pickle=True) as saved:
        trial_distance = pd.DataFrame(
            {
                "trial_id": saved["trial_id"].astype(int),
                "condition": saved["condition"].astype(object),
                "distance": [row for row in saved["distance"]],
            }
        )
        hyperplane = {
            "trial_distance": trial_distance,
            "group_order": saved["group_order"].tolist(),
            "label_order": saved["label_order"].tolist(),
        }

    with open(paths["trial_summary"], "r", encoding="utf-8") as f:
        trial_summary_row = json.load(f)

    return {
        "result": result,
        "hyperplane": hyperplane,
        "trial_summary_row": trial_summary_row,
        "window_times_ms": result["times_ms"],
        "window_masks": None,
        "ch_names": result["ch_names"],
    }


def save_generalization_subject_results(
    output_dir: str | Path,
    subject_id: str,
    times_ms: np.ndarray,
    generalization_result: dict[str, np.ndarray | pd.DataFrame | list[str] | int],
    trial_summary_row: dict[str, object],
) -> None:
    """Save one subject's generalization outputs for later reuse.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level results folder.
    subject_id : str
        Subject identifier without the `sub-` prefix.
    times_ms : np.ndarray
        Generalization time-window centers in milliseconds.
    generalization_result : dict[str, np.ndarray | pd.DataFrame | list[str] | int]
        Subject generalization output returned by the workflow.
    trial_summary_row : dict[str, object]
        Trial summary row for one subject.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = generalization_subject_result_paths(output_dir, subject_id)
    labels = np.asarray(generalization_result["label_order"], dtype=object)
    accuracy_by_repeat = generalization_result["accuracy_by_repeat"]
    np.savez(
        paths["generalization"],
        times_ms=np.asarray(times_ms),
        accuracy=np.asarray(generalization_result["accuracy"]),
        perm_accuracy=np.asarray(generalization_result["perm_accuracy"]),
        repeat_cv=accuracy_by_repeat["cv_repeat"].to_numpy(dtype=int),
        repeat_data_type=accuracy_by_repeat["data_type"].to_numpy(dtype=object),
        repeat_perm_id=accuracy_by_repeat["perm_id"].to_numpy(dtype=int),
        repeat_train_time_ix=accuracy_by_repeat["train_time_ix"].to_numpy(dtype=int),
        repeat_test_time_ix=accuracy_by_repeat["test_time_ix"].to_numpy(dtype=int),
        repeat_n_correct=accuracy_by_repeat["n_correct"].to_numpy(dtype=int),
        repeat_n_test_trials=accuracy_by_repeat["n_test_trials"].to_numpy(dtype=int),
        repeat_accuracy=accuracy_by_repeat["accuracy"].to_numpy(dtype=float),
        repeat_balanced_accuracy=accuracy_by_repeat["balanced_accuracy"].to_numpy(dtype=float),
        repeat_chance_level=accuracy_by_repeat["chance_level"].to_numpy(dtype=float),
        label_order=labels,
        n_input_trials=np.asarray(generalization_result["n_input_trials"]),
        n_training_trials=np.asarray(generalization_result["n_training_trials"]),
        n_binned_trials=np.asarray(generalization_result["n_binned_trials"]),
    )
    with open(paths["trial_summary"], "w", encoding="utf-8") as f:
        json.dump(trial_summary_row, f, indent=2)


def load_saved_generalization_subject_results(
    output_dir: str | Path,
    subject_id: str,
) -> dict[str, object]:
    """Load all saved generalization outputs for one subject.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level results folder.
    subject_id : str
        Subject identifier without the `sub-` prefix.

    Returns
    -------
    dict[str, object]
        Subject-level outputs in the same structure used by the workflow.
    """

    paths = generalization_subject_result_paths(output_dir, subject_id)

    with np.load(paths["generalization"], allow_pickle=True) as saved:
        accuracy_by_repeat = pd.DataFrame(
            {
                "cv_repeat": saved["repeat_cv"].astype(int),
                "data_type": saved["repeat_data_type"].astype(object),
                "perm_id": saved["repeat_perm_id"].astype(int),
                "train_time_ix": saved["repeat_train_time_ix"].astype(int),
                "test_time_ix": saved["repeat_test_time_ix"].astype(int),
                "n_correct": saved["repeat_n_correct"].astype(int),
                "n_test_trials": saved["repeat_n_test_trials"].astype(int),
                "accuracy": saved["repeat_accuracy"].astype(float),
                "balanced_accuracy": saved["repeat_balanced_accuracy"].astype(float),
                "chance_level": saved["repeat_chance_level"].astype(float),
            }
        )
        result = {
            "times_ms": saved["times_ms"],
            "accuracy": saved["accuracy"],
            "perm_accuracy": saved["perm_accuracy"],
            "accuracy_by_repeat": accuracy_by_repeat,
            "label_order": saved["label_order"].tolist(),
            "n_input_trials": int(saved["n_input_trials"]),
            "n_training_trials": int(saved["n_training_trials"]),
            "n_binned_trials": int(saved["n_binned_trials"]),
        }

    with open(paths["trial_summary"], "r", encoding="utf-8") as f:
        trial_summary_row = json.load(f)

    return {
        "result": result,
        "trial_summary_row": trial_summary_row,
        "window_times_ms": result["times_ms"],
    }


def subject_result_exists(output_dir: str | Path, subject_id: str) -> bool:
    """Return whether all saved outputs exist for one subject.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level results folder.
    subject_id : str
        Subject identifier without the `sub-` prefix.

    Returns
    -------
    bool
        True when decoding, hyperplane, and trial summary files all exist.
    """

    paths = subject_result_paths(output_dir, subject_id)
    return all(path.exists() for path in paths.values())


def generalization_subject_result_exists(output_dir: str | Path, subject_id: str) -> bool:
    """Return whether all saved generalization outputs exist for one subject.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level results folder.
    subject_id : str
        Subject identifier without the `sub-` prefix.

    Returns
    -------
    bool
        True when the saved generalization and trial-summary files both exist.
    """

    paths = generalization_subject_result_paths(output_dir, subject_id)
    return all(path.exists() for path in paths.values())


def epochs_path_for_subject(subject_id: str, cfg: DecodingConfig) -> Path:
    """Return the saved epochs path for one subject."""

    return derivative_file_path(
        cfg.dataset.data_dir,
        subject_id,
        cfg.dataset.experiment_name,
        "epo",
        ".fif",
        subject_prefix=cfg.dataset.subject_prefix,
        derivative_dirname=cfg.dataset.derivative_dirname,
        datatype=cfg.dataset.derivative_datatype,
        derivative_label=cfg.dataset.derivative_label,
    )


def events_path_for_subject(subject_id: str, cfg: DecodingConfig) -> Path:
    """Return the events sidecar path for one subject."""

    return derivative_file_path(
        cfg.dataset.data_dir,
        subject_id,
        cfg.dataset.experiment_name,
        "events",
        ".tsv",
        subject_prefix=cfg.dataset.subject_prefix,
        derivative_dirname=cfg.dataset.derivative_dirname,
        datatype=cfg.dataset.derivative_datatype,
        derivative_label=cfg.dataset.derivative_label,
    )


def load_events_table(subject_id: str, cfg: DecodingConfig) -> pd.DataFrame:
    """Load the saved events table for one subject."""

    return pd.read_csv(events_path_for_subject(subject_id, cfg), sep="\t")


def trial_qc_path_for_subject(subject_id: str, cfg: DecodingConfig) -> Path:
    """Return the trial-QC sidecar path for one subject."""

    return derivative_file_path(
        cfg.dataset.data_dir,
        subject_id,
        cfg.dataset.experiment_name,
        "trial_qc",
        ".tsv",
        subject_prefix=cfg.dataset.subject_prefix,
        derivative_dirname=cfg.dataset.derivative_dirname,
        datatype=cfg.dataset.derivative_datatype,
        derivative_label=cfg.dataset.derivative_label,
    )


def load_trial_qc_table(subject_id: str, cfg: DecodingConfig) -> pd.DataFrame | None:
    """Load the optional trial-QC table for one subject."""

    trial_qc_path = trial_qc_path_for_subject(subject_id, cfg)
    if not trial_qc_path.exists():
        return None
    return pd.read_csv(trial_qc_path, sep="\t")
