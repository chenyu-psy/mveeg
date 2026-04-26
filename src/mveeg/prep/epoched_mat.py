"""Convert already-epoched MATLAB EEG arrays into project derivatives.

This module is for source datasets that already store EEG as MATLAB arrays,
typically with one file for EEG data, one for condition labels, and one for a
time axis. It does not download or extract archives; those steps are clearer
as script-level commands because they are source-specific.
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

from ..io.bids import build_subject_label, derivative_file_path


def normalize_selected_subjects(selected_subjects: list[str]) -> set[str]:
    """Return selected subject IDs without common BIDS prefixes.

    Parameters
    ----------
    selected_subjects : list[str]
        Subject IDs such as ``"1"``, ``"sub1"``, or ``"sub-1"``.

    Returns
    -------
    set[str]
        Normalized subject IDs. An empty input returns an empty set, which
        callers use to mean "include all subjects".
    """

    normalized = set()
    for subject in selected_subjects:
        subject_id = str(subject).replace("sub-", "", 1).replace("sub", "", 1)
        normalized.add(subject_id)
    return normalized


def list_subject_ids(source_dir: Path, selected_subjects: list[str]) -> list[str]:
    """Return subject IDs with `*_xdata.mat` files.

    Parameters
    ----------
    source_dir : Path
        Folder containing files named like ``1_xdata.mat``.
    selected_subjects : list[str]
        Optional subject filter. Leave empty to include all source subjects.

    Returns
    -------
    list[str]
        Sorted subject IDs without the ``sub-`` prefix.

    Raises
    ------
    FileNotFoundError
        If `source_dir` does not exist.
    """

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_dir}")

    available_ids = [
        path.name.replace("_xdata.mat", "")
        for path in source_dir.glob("*_xdata.mat")
    ]
    available_ids = sorted(available_ids, key=lambda subject_id: int(subject_id))

    selected_set = normalize_selected_subjects(selected_subjects)
    if len(selected_set) == 0:
        return available_ids
    return [subject_id for subject_id in available_ids if subject_id in selected_set]


def load_subject_arrays(source_dir: Path, subject_id: str) -> dict[str, np.ndarray]:
    """Load epoched MATLAB arrays for one subject.

    Parameters
    ----------
    source_dir : Path
        Folder containing ``*_xdata.mat``, ``*_ydata.mat``, and ``*_info.mat``.
    subject_id : str
        Source subject number as a string.

    Returns
    -------
    dict[str, np.ndarray]
        EEG data, condition labels, time axis, and source id.

    Raises
    ------
    FileNotFoundError
        If any required source file is missing.
    """

    paths = {
        "xdata": source_dir / f"{subject_id}_xdata.mat",
        "ydata": source_dir / f"{subject_id}_ydata.mat",
        "info": source_dir / f"{subject_id}_info.mat",
    }
    missing_paths = [path for path in paths.values() if not path.exists()]
    if len(missing_paths) > 0:
        missing_str = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing source files for subject {subject_id}: {missing_str}")

    xdata = loadmat(paths["xdata"])["xdata"]
    ydata = loadmat(paths["ydata"])["ydata"].reshape(-1).astype(int)
    info_mat = loadmat(paths["info"])
    times_ms = info_mat["times"].reshape(-1).astype(float)
    source_unique_id = int(info_mat["unique_id"].reshape(-1)[0])

    return {
        "xdata": xdata,
        "ydata": ydata,
        "times_ms": times_ms,
        "source_unique_id": np.array([source_unique_id], dtype=int),
    }


def build_condition_metadata(
    subject_id: str,
    ydata: np.ndarray,
    condition_map: dict[int, dict[str, object]],
) -> pd.DataFrame:
    """Build trial metadata from numeric condition codes.

    Parameters
    ----------
    subject_id : str
        Source subject number as a string.
    ydata : np.ndarray
        One condition code per trial.
    condition_map : dict[int, dict[str, object]]
        Mapping from numeric ydata codes to metadata fields. Each entry must
        include a ``"condition"`` value.

    Returns
    -------
    pd.DataFrame
        Trial-level metadata aligned with the EEG epochs.

    Raises
    ------
    ValueError
        If a condition code is not listed in `condition_map`.
    """

    unknown_codes = sorted(set(np.unique(ydata)) - set(condition_map))
    if len(unknown_codes) > 0:
        raise ValueError(f"Found unmapped condition codes for subject {subject_id}: {unknown_codes}")

    rows = []
    for trial_ix, code in enumerate(ydata, start=1):
        condition_info = dict(condition_map[int(code)])
        row = {
            "subject_id": subject_id,
            "trial_id": trial_ix,
            "label": condition_info["condition"],
            "condition": condition_info["condition"],
            "ydata_code": int(code),
            "final_qc_category": "accepted",
        }
        for key, value in condition_info.items():
            if key != "condition":
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_epochs_array(
    source_data: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    event_id: dict[str, int],
    channel_prefix: str = "E",
) -> mne.EpochsArray:
    """Convert one subject's source arrays into MNE epochs.

    Parameters
    ----------
    source_data : dict[str, np.ndarray]
        Loaded ``xdata``, ``ydata``, and ``times_ms`` arrays.
    metadata : pd.DataFrame
        Trial metadata aligned with the EEG data.
    event_id : dict[str, int]
        Mapping from condition labels to event codes.
    channel_prefix : str, optional
        Prefix used when the source files do not provide channel labels.

    Returns
    -------
    mne.EpochsArray
        Epoched EEG data ready for the project's loaders.

    Notes
    -----
    Some external MATLAB exports omit electrode labels or montage positions.
    This function preserves channel order by assigning labels such as ``E01``.
    """

    xdata = np.asarray(source_data["xdata"], dtype=float)
    ydata = np.asarray(source_data["ydata"], dtype=int)
    times_ms = np.asarray(source_data["times_ms"], dtype=float)

    data = np.transpose(xdata, (2, 0, 1))
    if data.shape[0] != len(metadata):
        raise ValueError(
            "The EEG trial count did not match metadata rows. "
            f"Found {data.shape[0]} trials and {len(metadata)} rows."
        )
    if data.shape[2] != len(times_ms):
        raise ValueError(
            "The EEG time axis did not match the info times. "
            f"Found {data.shape[2]} samples and {len(times_ms)} time points."
        )

    sample_step_ms = float(np.median(np.diff(times_ms)))
    sfreq = 1000.0 / sample_step_ms
    tmin = float(times_ms[0]) / 1000.0
    ch_names = [f"{channel_prefix}{ch_ix:02d}" for ch_ix in range(1, data.shape[1] + 1)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))

    events = np.column_stack(
        [
            np.arange(len(ydata), dtype=int),
            np.zeros(len(ydata), dtype=int),
            ydata.astype(int),
        ]
    )

    return mne.EpochsArray(
        data=data,
        info=info,
        events=events,
        event_id=event_id,
        tmin=tmin,
        metadata=metadata,
        verbose="ERROR",
    )


def build_derivative_paths(
    data_dir: Path,
    subject_id: str,
    experiment_name: str,
    subject_prefix: str = "sub",
    derivative_dirname: str = "derivatives",
    datatype: str = "eeg",
    derivative_label: str = "preprocessed",
) -> dict[str, Path]:
    """Build standardized derivative paths for one subject.

    Parameters
    ----------
    data_dir : Path
        Root preprocessed-data folder.
    subject_id : str
        Subject identifier without the ``sub-`` prefix.
    experiment_name : str
        Experiment name used in derivative filenames.
    subject_prefix : str, optional
        BIDS subject prefix.
    derivative_dirname : str, optional
        Derivative folder name under `data_dir`.
    datatype : str, optional
        BIDS datatype folder.
    derivative_label : str, optional
        Label written after ``desc-``.

    Returns
    -------
    dict[str, Path]
        Paths for epochs, events, trial QC, trial state, and sidecar files.
    """

    def one_path(suffix: str, extension: str) -> Path:
        """Build one derivative path."""

        return derivative_file_path(
            data_dir=data_dir,
            subject_id=subject_id,
            experiment_name=experiment_name,
            suffix=suffix,
            extension=extension,
            subject_prefix=subject_prefix,
            derivative_dirname=derivative_dirname,
            datatype=datatype,
            derivative_label=derivative_label,
        )

    return {
        "epochs": one_path("epo", ".fif"),
        "events": one_path("events", ".tsv"),
        "trial_qc": one_path("trial_qc", ".tsv"),
        "trial_state": one_path("trial_state", ".tsv"),
        "sidecar": one_path("eeg", ".json"),
    }


def save_standard_derivatives(
    data_dir: Path,
    subject_id: str,
    experiment_name: str,
    epochs: mne.EpochsArray,
    metadata: pd.DataFrame,
    source_label: str,
    overwrite: bool,
) -> dict[str, Path]:
    """Save one subject in the project's standard derivative format.

    Parameters
    ----------
    data_dir : Path
        Root preprocessed-data folder.
    subject_id : str
        Subject identifier without the ``sub-`` prefix.
    experiment_name : str
        Experiment name used in derivative filenames.
    epochs : mne.EpochsArray
        Converted EEG epochs.
    metadata : pd.DataFrame
        Trial metadata aligned with `epochs`.
    source_label : str
        Human-readable source description written to the EEG sidecar.
    overwrite : bool
        Whether to replace existing derivative files.

    Returns
    -------
    dict[str, Path]
        Paths to the saved derivative files.
    """

    output_paths = build_derivative_paths(data_dir, subject_id, experiment_name)
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    if output_paths["epochs"].exists() and not overwrite:
        raise FileExistsError(
            f"Derivative files already exist for subject {subject_id}. "
            "Set overwrite=True to replace them."
        )

    event_cols = [
        col for col in ["trial_id", "label", "condition", "feature", "load", "ydata_code"]
        if col in metadata.columns
    ]
    events_df = metadata.loc[:, event_cols].copy()
    events_df.insert(0, "onset", epochs.events[:, 0] / float(epochs.info["sfreq"]))
    events_df.insert(1, "duration", 0.0)
    events_df = events_df.rename(columns={"label": "trial_type", "ydata_code": "value"})
    events_df.to_csv(output_paths["events"], sep="\t", index=False)

    qc_cols = [
        col for col in ["trial_id", "label", "condition", "ydata_code", "final_qc_category"]
        if col in metadata.columns
    ]
    metadata.loc[:, qc_cols].to_csv(output_paths["trial_qc"], sep="\t", index=False)

    trial_state_df = pd.DataFrame(
        {
            "trial_index": np.arange(len(metadata), dtype=int),
            "trial_type": metadata["label"].astype(str).to_numpy(),
            "hard_reject": np.zeros(len(metadata), dtype=bool),
            "hard_reasons": np.full(len(metadata), "", dtype=object),
            "hard_flagged_channels": np.zeros(len(metadata), dtype=int),
            "hf_noise_hard_flag": np.zeros(len(metadata), dtype=bool),
            "autoreject_reject": np.zeros(len(metadata), dtype=bool),
            "autoreject_interp_channels": np.zeros(len(metadata), dtype=int),
            "soft_reject": np.zeros(len(metadata), dtype=bool),
            "final_keep": metadata["final_qc_category"].eq("accepted").to_numpy(),
            "final_qc_category": metadata["final_qc_category"].astype(str).to_numpy(),
            "state_stage": np.full(len(metadata), "source_preprocessed", dtype=object),
        }
    )
    trial_state_df.to_csv(output_paths["trial_state"], sep="\t", index=False)

    epochs.save(output_paths["epochs"], overwrite=overwrite, verbose="ERROR")

    sidecar = {
        "TaskName": experiment_name,
        "RecordingType": "epoched",
        "SamplingFrequency": float(epochs.info["sfreq"]),
        "EpochLength": float(epochs.times[-1] - epochs.times[0]),
        "Source": source_label,
        "Description": (
            "Converted from epoched MATLAB arrays. Generated channel names "
            "preserve source order but should not be used as montage labels."
        ),
    }
    with open(output_paths["sidecar"], "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    return output_paths


def write_dataset_summary(
    data_dir: Path,
    subject_ids: list[str],
    experiment_name: str,
    generated_by: str,
) -> None:
    """Write simple dataset-level files for converted derivatives.

    Parameters
    ----------
    data_dir : Path
        Root preprocessed-data folder for this experiment.
    subject_ids : list[str]
        Subject IDs included in the conversion.
    experiment_name : str
        Experiment name written into the dataset metadata.
    generated_by : str
        Script or script path that generated the derivative files.

    Side Effects
    ------------
    Writes ``dataset_description.json``, ``participants.tsv``, and ``README``.
    """

    dataset_description = {
        "Name": experiment_name,
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": generated_by,
                "Description": "Convert epoched MATLAB arrays into MNE epochs.",
            }
        ],
    }
    with open(data_dir / "dataset_description.json", "w", encoding="utf-8") as f:
        json.dump(dataset_description, f, indent=2)

    participants_df = pd.DataFrame(
        {
            "participant_id": [build_subject_label(subject_id) for subject_id in subject_ids],
            "source_subject_id": subject_ids,
        }
    )
    participants_df.to_csv(data_dir / "participants.tsv", sep="\t", index=False)

    readme_text = (
        f"This folder stores encoding-ready derivatives for {experiment_name}.\n"
        f"Run {generated_by} to rebuild the files.\n"
    )
    (data_dir / "README").write_text(readme_text, encoding="utf-8")


def convert_subject(
    source_dir: Path,
    data_dir: Path,
    experiment_name: str,
    subject_id: str,
    condition_map: dict[int, dict[str, object]],
    source_label: str,
    overwrite: bool,
) -> dict[str, object]:
    """Convert one subject from MATLAB arrays to standard derivatives.

    Parameters
    ----------
    source_dir : Path
        Extracted source folder.
    data_dir : Path
        Root preprocessed-data folder.
    experiment_name : str
        Experiment name used in derivative filenames.
    subject_id : str
        Subject to convert.
    condition_map : dict[int, dict[str, object]]
        Mapping from numeric ydata codes to metadata fields.
    source_label : str
        Human-readable source label for the EEG sidecar.
    overwrite : bool
        Whether to replace existing derivative files.

    Returns
    -------
    dict[str, object]
        Conversion summary for this subject.
    """

    event_id = {value["condition"]: code for code, value in condition_map.items()}
    source_data = load_subject_arrays(source_dir, subject_id)
    metadata = build_condition_metadata(subject_id, source_data["ydata"], condition_map)
    epochs = build_epochs_array(source_data, metadata, event_id)
    output_paths = save_standard_derivatives(
        data_dir=data_dir,
        subject_id=subject_id,
        experiment_name=experiment_name,
        epochs=epochs,
        metadata=metadata,
        source_label=source_label,
        overwrite=overwrite,
    )

    return {
        "subject_id": subject_id,
        "status": "converted",
        "n_epochs": len(epochs),
        "n_channels": len(epochs.ch_names),
        "sfreq": float(epochs.info["sfreq"]),
        "tmin_s": float(epochs.times[0]),
        "tmax_s": float(epochs.times[-1]),
        "epochs_path": str(output_paths["epochs"]),
    }
