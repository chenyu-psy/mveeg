"""Workflow helpers that sit on top of the core EEG preprocessing engine.

This module keeps script code short and readable by storing preprocessing
setup in one explicit workflow object. Use ``create_flow()`` to build the
workflow, configure it in stages, then run subject-level preprocessing with
its methods.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TextIO
from typing import TYPE_CHECKING

import mne
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..io.bids import derivative_file_path
from . import core as preprocess_core

if TYPE_CHECKING:
    from .core import Preprocess


def get_subject_selection(
    *,
    root_dir: str | Path,
    selected_subjects: list[str] | None = None,
    condition_replacements: dict[str, dict[str, str]] | None = None,
) -> tuple[list[Path], dict[str, Path]]:
    """Return the subjects that should be processed in one preprocessing run.

    Parameters
    ----------
    root_dir : str | Path
        Folder that contains one raw-data folder per subject.
    selected_subjects : list[str] | None, optional
        Optional whitelist of raw subject folder names to process.
    condition_replacements : dict[str, dict[str, str]] | None, optional
        Optional subject assembly plan. Subjects that appear only as donor
        sources are excluded from the main processing loop, because they are
        loaded on demand when assembling another subject.

    Returns
    -------
    tuple[list[Path], dict[str, Path]]
        Selected raw subject folders and a name-to-path lookup table.
    """
    subject_dir_map = {
        path.name: path
        for path in sorted(path for path in Path(root_dir).iterdir() if path.is_dir())
    }

    if selected_subjects:
        subject_dirs = [subject_dir_map[str(subject)] for subject in selected_subjects]
    else:
        subject_dirs = list(subject_dir_map.values())

    replacement_map = condition_replacements or {}
    target_subjects = set(replacement_map.keys())
    replacement_donors = {
        donor_subject
        for subject_replacements in replacement_map.values()
        for donor_subject in subject_replacements.values()
    }
    donor_only_subjects = replacement_donors - target_subjects
    subject_dirs = [subject_dir for subject_dir in subject_dirs if subject_dir.name not in donor_only_subjects]
    return subject_dirs, subject_dir_map


def create_flow() -> "PreprocessWorkflow":
    """Create an empty preprocessing workflow.

    Returns
    -------
    PreprocessWorkflow
        Workflow object ready for stepwise configuration.
    """
    return PreprocessWorkflow()


STATUS_COLUMNS = [
    "subject_number",
    "intermediate_saved",
    "hard_qc_saved",
    "autoreject_done",
    "final_saved",
    "last_completed_step",
    "status",
    "error_message",
    "updated_at",
]

HARD_REJECT_STEP_BASE = [
    "dropout",
    "flatline",
    "hard abs",
    "gaze deviation",
    "gaze shift",
    "hard step",
    "hard p2p",
]

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


class PreprocessWorkflow:
    """Store preprocessing setup in a few explicit workflow stages.

    Notes
    -----
    Build the workflow in a few small steps:

    1. Create the workflow with ``create_flow()``.
    2. Call ``configure_io(...)`` with IO fields such as ``data_dir`` and
       ``root_dir``.
    3. Call ``configure_subject_selection(...)`` to define which subjects
       belong to the current run.
    4. Call ``configure_behavior(...)`` to store behavior-alignment rules.
    5. Call ``configure_preprocessor(...)`` and ``configure_qc(...)`` before
       subject-level preprocessing starts.

    This keeps the script structure close to the research workflow instead
    of hiding many unrelated settings inside one large constructor call.
    """

    def __init__(self) -> None:
        """Create an empty workflow to be configured in small steps."""
        self.io_config: preprocess_core.IOConfig | None = None
        self.pre: Preprocess | None = None
        self.overwrite_all = False
        self.behavior_suffix = "_beh.csv"
        self.reref_channels: tuple[str, ...] | None = None
        self.pre_filter_rules: dict | None = None
        self.post_filter_rules: dict | None = None
        self.manual_trial_exclusions: dict[str, dict[str, list[int]]] | None = None
        self.reject_qc: dict | None = None
        self.review_qc: dict | None = None
        self.autoreject_cfg: dict | None = None
        self.subject_dirs: list[Path] = []
        self.subject_dir_map: dict[str, Path] = {}

    def configure_io(
        self,
        *,
        data_dir: str | Path,
        root_dir: str | Path,
        experiment_name: str | None = None,
        subject_prefix: str = "sub",
        derivative_dirname: str = "derivatives",
        derivative_label: str = "preprocessed",
        raw_datatype: str = "eeg",
        derivative_datatype: str = "eeg",
        drop_channels: list | tuple | None = None,
    ) -> None:
        """Store the shared IO configuration for this workflow.

        Parameters
        ----------
        data_dir : str | Path
            BIDS-organized output directory written by this workflow.
        root_dir : str | Path
            Folder that contains one raw-data folder per subject.
        experiment_name : str | None, optional
            Task label used in BIDS-style filenames.
        subject_prefix : str, optional
            Prefix used in BIDS-style subject labels.
        derivative_dirname : str, optional
            Name of the derivatives folder under ``data_dir``.
        derivative_label : str, optional
            Label used in derivative EEG filenames.
        raw_datatype : str, optional
            BIDS datatype name for imported raw files.
        derivative_datatype : str, optional
            BIDS datatype name for saved derivative files.
        drop_channels : list | tuple | None, optional
            Channels removed before saving the preprocessed data.

        Returns
        -------
        None
            The IO configuration is stored on ``self.io_config``.
        """
        self.io_config = preprocess_core.IOConfig(
            data_dir=data_dir,
            root_dir=root_dir,
            experiment_name=experiment_name,
            subject_prefix=subject_prefix,
            derivative_dirname=derivative_dirname,
            derivative_label=derivative_label,
            raw_datatype=raw_datatype,
            derivative_datatype=derivative_datatype,
            drop_channels=drop_channels,
        )
        Path(self.io_config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.io_config.root_dir).mkdir(parents=True, exist_ok=True)

    @property
    def data_dir(self) -> Path:
        """Return the BIDS-organized output directory for this workflow."""
        return Path(self._require_io_config().data_dir)

    @property
    def root_dir(self) -> Path:
        """Return the raw-data directory for this workflow."""
        return Path(self._require_io_config().root_dir)

    @property
    def experiment_name(self) -> str | None:
        """Return the experiment name used in BIDS-style outputs."""
        return self._require_io_config().experiment_name

    @property
    def subject_prefix(self) -> str:
        """Return the subject-prefix label used in BIDS-style outputs."""
        return self._require_io_config().subject_prefix

    @property
    def derivative_datatype(self) -> str:
        """Return the datatype folder used for derivative outputs."""
        return self._require_io_config().derivative_datatype

    def _empty_status_table(self) -> pd.DataFrame:
        """Return an empty preprocessing status table with standard columns."""
        return pd.DataFrame(columns=STATUS_COLUMNS)

    def _status_table_subjects(self) -> list[str]:
        """Return selected subject labels in the current workflow order."""
        return [subject_path.name for subject_path in self.subject_dirs]

    def load_status_table(self, status_path: str | Path) -> pd.DataFrame:
        """Load a preprocessing status table, creating defaults when missing.

        Parameters
        ----------
        status_path : str | Path
            TSV file used to track run progress across subjects.

        Returns
        -------
        pd.DataFrame
            Status table with one row per selected subject.
        """
        status_path = Path(status_path)
        if status_path.exists():
            status_table = pd.read_csv(status_path, sep="\t")
        else:
            status_table = self._empty_status_table()

        subject_numbers = self._status_table_subjects()
        if len(status_table) == 0:
            status_table = pd.DataFrame({"subject_number": subject_numbers})

        missing_subjects = [subject for subject in subject_numbers if subject not in set(status_table["subject_number"])]
        if missing_subjects:
            status_table = pd.concat(
                [status_table, pd.DataFrame({"subject_number": missing_subjects})],
                ignore_index=True,
            )

        status_table = status_table[status_table["subject_number"].isin(subject_numbers)].copy()
        status_table = status_table.set_index("subject_number").reindex(subject_numbers).reset_index()

        for col in STATUS_COLUMNS:
            if col not in status_table.columns:
                status_table[col] = pd.NA

        bool_cols = ["intermediate_saved", "hard_qc_saved", "autoreject_done", "final_saved"]
        for col in bool_cols:
            status_table[col] = status_table[col].fillna(False).astype(bool)

        text_defaults = {
            "last_completed_step": "",
            "status": "pending",
            "error_message": "",
            "updated_at": "",
        }
        for col, default_value in text_defaults.items():
            status_table[col] = status_table[col].fillna(default_value).astype(str)

        return status_table[STATUS_COLUMNS]

    def save_status_table(self, status_path: str | Path, status_table: pd.DataFrame) -> Path:
        """Write the preprocessing status table to disk.

        Parameters
        ----------
        status_path : str | Path
            TSV file used to track run progress across subjects.
        status_table : pd.DataFrame
            Status table to write.

        Returns
        -------
        pathlib.Path
            Path of the saved status table.
        """
        status_path = Path(status_path)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_table.to_csv(status_path, sep="\t", index=False)
        return status_path

    def update_subject_status(
        self,
        status_path: str | Path,
        subject_number: str,
        *,
        intermediate_saved: bool | None = None,
        hard_qc_saved: bool | None = None,
        autoreject_done: bool | None = None,
        final_saved: bool | None = None,
        last_completed_step: str | None = None,
        status: str | None = None,
        error_message: str | None = None,
    ) -> pd.DataFrame:
        """Update one subject row in the preprocessing status table.

        Parameters
        ----------
        status_path : str | Path
            TSV file used to track run progress across subjects.
        subject_number : str
            Subject label used in raw-data folders and saved outputs.
        intermediate_saved, hard_qc_saved, autoreject_done, final_saved : bool | None, optional
            Optional step-completion flags to update for this subject.
        last_completed_step : str | None, optional
            Short label describing the latest completed step.
        status : str | None, optional
            Overall run status such as ``"pending"``, ``"running"``,
            ``"completed"``, or ``"error"``.
        error_message : str | None, optional
            Error text stored when a subject fails.

        Returns
        -------
        pd.DataFrame
            Updated full status table.
        """
        status_table = self.load_status_table(status_path)
        row_ix = status_table.index[status_table["subject_number"].eq(subject_number)][0]
        updates = {
            "intermediate_saved": intermediate_saved,
            "hard_qc_saved": hard_qc_saved,
            "autoreject_done": autoreject_done,
            "final_saved": final_saved,
            "last_completed_step": last_completed_step,
            "status": status,
            "error_message": error_message,
        }
        for col, value in updates.items():
            if value is not None:
                status_table.at[row_ix, col] = value
        status_table.at[row_ix, "updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_status_table(status_path, status_table)
        return status_table

    def should_overwrite(self, overwrite_step: bool, overwrite_all: bool | None = None) -> bool:
        """Resolve one step's overwrite decision from global and local flags.

        Parameters
        ----------
        overwrite_step : bool
            Step-specific overwrite switch.
        overwrite_all : bool | None, optional
            Optional global overwrite switch. If ``None``, use the value stored
            in this workflow during ``configure_subject_selection(...)``.

        Returns
        -------
        bool
            Final overwrite decision for the step.
        """
        if overwrite_all is None:
            overwrite_all = self.overwrite_all
        return bool(overwrite_all or overwrite_step)

    def should_run_step(
        self,
        *,
        subject_number: str,
        step: str,
        overwrite_step: bool,
        overwrite_all: bool | None = None,
    ) -> bool:
        """Return whether one preprocessing step should run for one subject.

        Parameters
        ----------
        subject_number : str
            Subject label used in raw-data folders and derivative outputs.
        step : str
            Step name. Supported values are ``"hard_reject"``,
            ``"autoreject"``, and ``"review"``.
        overwrite_step : bool
            Step-level overwrite switch.
        overwrite_all : bool | None, optional
            Optional global overwrite switch. If ``None``, use the value stored
            in this workflow during ``configure_subject_selection(...)``.

        Returns
        -------
        bool
            ``True`` when the step should run.
        """
        if step == "hard_reject":
            already_done = self.has_subject_reject_checkpoint(subject_number)
        elif step == "autoreject":
            already_done = self.has_subject_autoreject_checkpoint(subject_number)
        elif step == "review":
            already_done = self.has_saved_final_outputs(subject_number)
        else:
            raise ValueError(f"Unsupported step '{step}'. Use hard_reject, autoreject, or review.")
        return self.should_overwrite(overwrite_step, overwrite_all=overwrite_all) or (not already_done)

    def get_subject_dirs_pending_final(
        self,
        *,
        stage: str = "prepared",
        overwrite: bool = False,
    ) -> list[Path]:
        """Return subject directories that should run QC and final saving.

        Parameters
        ----------
        stage : str, optional
            Intermediate stage label that must exist before QC can run.
        overwrite : bool, optional
            If ``True``, include every subject with saved intermediate epochs,
            even when final outputs already exist. This is useful when the
            run-level overwrite flag should force QC and autoreject to run
            again from the saved intermediate file.

        Returns
        -------
        list[pathlib.Path]
            Subject directories with saved intermediate epochs and, unless
            ``overwrite`` is ``True``, without existing final saved outputs.
        """
        return [
            subject_path
            for subject_path in self.subject_dirs
            if self.has_saved_intermediate_epochs(subject_path.name, stage=stage)
            and (overwrite or not self.has_saved_final_outputs(subject_path.name))
        ]

    def _require_io_config(self) -> preprocess_core.IOConfig:
        """Return the configured IO settings or raise a clear error.

        Returns
        -------
        preprocess_core.IOConfig
            Configured IO settings stored on this workflow.

        Raises
        ------
        ValueError
            If ``configure_io`` has not been called yet.
        """
        if self.io_config is None:
            raise ValueError("PreprocessWorkflow requires configure_io(...) before later setup steps.")
        return self.io_config

    def configure_preprocessor(
        self,
        *,
        trial_window: tuple[float, float] | list[float],
        srate: int | None = None,
        baseline_time: tuple[float, float] | None = None,
        rejection_time: tuple[float, float] | list[float] | None = None,
        reject_between_codes: tuple | None = None,
        filter_freqs: tuple[None | int | float, None | int | float] = preprocess_core.FILTER_FREQS,
        reref_channels: list[str] | tuple[str, ...] | None = None,
        event_dict: dict,
        event_code_dict: dict,
        timelock_ix: int | dict | None = None,
        event_names: dict | None = None,
    ) -> None:
        """Build the preprocessing engine from plain epoch and event settings.

        Parameters
        ----------
        trial_window : tuple[float, float] | list[float]
            Epoch start and end time, in seconds, relative to the time-lock
            event.
        srate : int | None, optional
            Recording sampling rate in Hz.
        baseline_time : tuple[float, float] | None, optional
            Time window used for baseline correction.
        rejection_time : tuple[float, float] | list[float] | None, optional
            Time window used for artifact rejection.
        reject_between_codes : tuple | None, optional
            Optional event-code window used for artifact rejection instead of a
            direct time window.
        filter_freqs : tuple[None | int | float, None | int | float], optional
            Low- and high-pass filter settings applied before epoching.
        reref_channels : list[str] | tuple[str, ...] | None, optional
            Reference channel names used before filtering. Leave ``None`` to
            keep the recording reference unchanged.
        event_dict : dict
            Mapping from event labels to integer event codes.
        event_code_dict : dict
            Mapping from each trial code to the expected event-code sequence.
        timelock_ix : int | dict | None, optional
            Shared or per-condition index of the time-lock event.
        event_names : dict | None, optional
            Full mapping of visible event names used in saved outputs.

        Returns
        -------
        None
            The configured preprocessor is stored on ``self.pre``.
        """
        io_config = self._require_io_config()
        if reref_channels is None:
            self.reref_channels = None
        else:
            self.reref_channels = tuple(reref_channels)
        trial_start, trial_end = trial_window
        epoch_config = preprocess_core.EpochConfig(
            trial_start=trial_start,
            trial_end=trial_end,
            srate=srate,
            baseline_time=baseline_time,
            rejection_time=rejection_time,
            reject_between_codes=reject_between_codes,
            filter_freqs=filter_freqs,
        )
        event_config = preprocess_core.EventConfig(
            event_dict=event_dict,
            event_code_dict=event_code_dict,
            timelock_ix=timelock_ix,
            event_names=event_names,
        )
        self.pre = preprocess_core.Preprocess(
            io_config=io_config,
            epoch_config=epoch_config,
            event_config=event_config,
        )

    def configure_subject_selection(
        self,
        *,
        overwrite: bool = False,
        selected_subjects: list[str] | None = None,
        condition_replacements: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """Select which raw subject folders belong to this preprocessing run.

        Parameters
        ----------
        overwrite : bool, optional
            Global overwrite preference used by step-level run decisions.
            This flag does not change ``self.subject_dirs`` directly.
        selected_subjects : list[str] | None, optional
            Optional whitelist of raw subject folder names to process.
        condition_replacements : dict[str, dict[str, str]] | None, optional
            Optional subject assembly plan. Donor-only subjects are excluded
            from the main processing loop because they are loaded on demand.

        Returns
        -------
        None
            The selected subject folders are stored on ``self.subject_dirs``
            and ``self.subject_dir_map``.
        """
        self.overwrite_all = bool(overwrite)
        io_config = self._require_io_config()
        self.subject_dirs, self.subject_dir_map = get_subject_selection(
            root_dir=io_config.root_dir,
            selected_subjects=selected_subjects,
            condition_replacements=condition_replacements,
        )

    def configure_behavior(
        self,
        *,
        behavior_suffix: str = "_beh.csv",
        pre_filter_rules: dict | None = None,
        post_filter_rules: dict | None = None,
        manual_trial_exclusions: dict[str, dict[str, list[int]]] | None = None,
    ) -> None:
        """Store behavior-alignment and manual-fix settings for this run.

        Parameters
        ----------
        behavior_suffix : str, optional
            Required filename ending for each subject's behavior CSV.
        pre_filter_rules : dict | None, optional
            Behavior filters applied before EEG-behavior alignment.
        post_filter_rules : dict | None, optional
            Rules used after alignment to mark which epochs should be kept.
        manual_trial_exclusions : dict[str, dict[str, list[int]]] | None, optional
            Subject-specific EEG and eye-tracker trial drops before alignment.

        Returns
        -------
        None
            The behavior and manual-fix settings are stored on the workflow.
        """
        self.behavior_suffix = behavior_suffix
        self.pre_filter_rules = pre_filter_rules
        self.post_filter_rules = post_filter_rules
        self.manual_trial_exclusions = manual_trial_exclusions

    def configure_qc(
        self,
        *,
        reject_qc: dict,
        review_qc: dict,
        autoreject_cfg: dict | None = None,
    ) -> None:
        """Store quality-control settings for the current preprocessing run.

        Parameters
        ----------
        reject_qc : dict
            Hard-rejection thresholds used before autoreject. Expected format:
            ``{"gaze": {"deviation": ... , "shift": ...}, "eeg": {"p2p": ...,
            "step": ..., "absolute_value": ..., "bad_channels": ...},
            "hf_noise": {"band": (..., ...), "metric": "rms",
            "threshold_mode": "median_plus_mad", "mad_multiplier": ...,
            "bad_channels": ...}}``.
        review_qc : dict
            Soft-review thresholds used after autoreject with the same nested
            structure as ``reject_qc``.
        autoreject_cfg : dict | None, optional
            Optional local autoreject settings used before rule-based QC.

        Returns
        -------
        None
            The QC settings are stored on the workflow object.
        """
        self.reject_qc = reject_qc
        self.review_qc = review_qc
        self.autoreject_cfg = autoreject_cfg

    def _manual_trial_exclusions(self) -> dict[str, dict[str, list[int]]]:
        """Return manual trial exclusions with empty EEG and eye defaults.

        Returns
        -------
        dict[str, dict[str, list[int]]]
            Manual exclusions keyed by stream type and subject label.
        """
        if self.manual_trial_exclusions is None:
            return {"eeg": {}, "eye": {}}
        return self.manual_trial_exclusions

    def _require_preprocessor(self) -> "Preprocess":
        """Return the configured preprocessor or raise a clear error.

        Returns
        -------
        Preprocess
            Configured preprocessing engine stored on this workflow.

        Raises
        ------
        ValueError
            If ``configure_preprocessor`` has not been called yet.
        """
        if self.pre is None:
            raise ValueError("PreprocessWorkflow requires configure_preprocessor(...) before subject processing.")
        return self.pre

    def _require_qc_config(self) -> tuple[dict, dict, dict | None]:
        """Return QC settings or raise a clear error.

        Returns
        -------
        tuple[dict, dict, dict | None]
            Reject-stage QC settings, review-stage QC settings, and optional
            autoreject settings.

        Raises
        ------
        ValueError
            If ``configure_qc`` has not been called yet.
        """
        if self.reject_qc is None or self.review_qc is None:
            raise ValueError("PreprocessWorkflow requires configure_qc(...) before trial-level QC.")
        return self.reject_qc, self.review_qc, self.autoreject_cfg

    def get_reject_qc_steps(self) -> list[str]:
        """Return hard-reject progress labels in the workflow execution order.

        Returns
        -------
        list[str]
            Ordered list of hard-reject step labels used by progress
            bars. Includes ``"hard hf_noise"`` only when HF QC is configured.
        """
        reject_qc, _, _ = self._require_qc_config()
        steps = list(HARD_REJECT_STEP_BASE)
        if reject_qc.get("hf_noise") is not None:
            steps.append("hard hf_noise")
        return steps

    def load_subject_streams(
        self,
        subject_number: str,
    ) -> tuple[mne.io.BaseRaw, np.ndarray, mne.io.BaseRaw | None, np.ndarray | None, bool, pd.DataFrame]:
        """Load one subject's EEG, optional eye data, and behavior table.

        Parameters
        ----------
        subject_number : str
            Subject label used in the raw-data folder names.
        Returns
        -------
        tuple[mne.io.BaseRaw, np.ndarray, mne.io.BaseRaw | None, np.ndarray | None, bool, pd.DataFrame]
            Imported EEG stream and events, optional eye stream and events,
            whether eye data were found, and the filtered behavior table.
        """
        pre = self._require_preprocessor()
        eeg, eeg_events = pre.import_eeg(subject_number, overwrite=True)
        eye = None
        eye_events = None
        has_eye_data = True
        try:
            eye, eye_events = pre.import_eyetracker(subject_number, overwrite=True)
        except FileNotFoundError as err:
            has_eye_data = False
            print(f"Skipping eyetracking import for subject {subject_number}: {err}")
        pre.import_behavior(subject_number, suffix=self.behavior_suffix)

        behavior_data = pre.load_behavior_table(subject_number, suffix=self.behavior_suffix)
        if self.pre_filter_rules is not None:
            if "trial_types" in self.pre_filter_rules:
                behavior_data = behavior_data[behavior_data["trial_type"].isin(self.pre_filter_rules["trial_types"])]
            if "rejection" in self.pre_filter_rules:
                behavior_data = behavior_data[behavior_data["rejection"].eq(self.pre_filter_rules["rejection"])]
        behavior_data = behavior_data.reset_index(drop=True)
        return eeg, eeg_events, eye, eye_events, has_eye_data, behavior_data

    def build_subject_epochs(
        self,
        subject_number: str,
        eeg: mne.io.BaseRaw,
        eeg_events: np.ndarray,
        eye: mne.io.BaseRaw | None,
        eye_events: np.ndarray | None,
        *,
        has_eye_data: bool,
    ) -> mne.Epochs:
        """Build epochs from imported recording streams for one subject.

        Parameters
        ----------
        subject_number : str
            Subject label used to look up manual exclusions.
        eeg : mne.io.BaseRaw
            Imported EEG recording.
        eeg_events : np.ndarray
            EEG event table for epoch creation.
        eye : mne.io.BaseRaw | None
            Imported eye-tracking stream when available.
        eye_events : np.ndarray | None
            Eye-tracking event table when available.
        has_eye_data : bool
            Whether this subject has a usable eye-tracking stream.

        Returns
        -------
        mne.Epochs
            Epoched and aligned subject data.
        """
        pre = self._require_preprocessor()
        manual_trial_exclusions = self._manual_trial_exclusions()

        eeg.load_data()
        if self.reref_channels is not None:
            reref_index = mne.pick_channels(eeg.ch_names, list(self.reref_channels))
            if len(reref_index) == 0:
                raise ValueError(f"Could not find rereference channels: {self.reref_channels}")
            eeg.apply_function(
                pre.rereference_to_average,
                picks=["eeg"],
                reref_values=np.squeeze(eeg.get_data()[reref_index]),
            )
        eeg.filter(*pre.filter_freqs, n_jobs=-1, verbose="ERROR")
        if has_eye_data:
            return pre.make_and_sync_epochs(
                eeg,
                eeg_events,
                eye,
                eye_events,
                eeg_trials_drop=manual_trial_exclusions["eeg"].get(subject_number, []),
                eye_trials_drop=manual_trial_exclusions["eye"].get(subject_number, []),
            )
        return pre.make_eeg_epochs(
            eeg,
            eeg_events,
            eeg_trials_drop=manual_trial_exclusions["eeg"].get(subject_number, []),
        )

    def keep_aligned_experimental_trials(
        self,
        subject_number: str,
        epochs: mne.Epochs,
        behavior_data: pd.DataFrame,
    ) -> mne.Epochs:
        """Align epochs to behavior rows and keep requested trial types.

        Parameters
        ----------
        subject_number : str
            Subject label used in the saved outputs.
        epochs : mne.Epochs
            Epoched data before the final behavior-based keep filter.
        behavior_data : pd.DataFrame
            Behavior rows that were already filtered before alignment.

        Returns
        -------
        mne.Epochs
            Aligned epochs after the requested trial-type filter.
        """
        pre = self._require_preprocessor()

        def mark_epochs_to_keep(matched_behavior_data: pd.DataFrame) -> pd.DataFrame:
            """Mark aligned rows that should remain in the saved epochs.

            Parameters
            ----------
            matched_behavior_data : pd.DataFrame
                Behavior rows that matched the saved epochs one-to-one.

            Returns
            -------
            pd.DataFrame
                Matched behavior table with a boolean ``keep_epoch`` column.
            """
            matched_behavior_data = matched_behavior_data.copy()
            if self.post_filter_rules is not None and "keep_trial_type" in self.post_filter_rules:
                keep_epoch = matched_behavior_data["trial_type"].eq(self.post_filter_rules["keep_trial_type"])
            else:
                keep_epoch = pd.Series(True, index=matched_behavior_data.index)
            matched_behavior_data["keep_epoch"] = keep_epoch
            return matched_behavior_data

        return pre.exclude_practice_trials(
            subject_number,
            epochs,
            suffix=self.behavior_suffix,
            behavior=behavior_data,
            matched_behavior_filter=mark_epochs_to_keep,
        )

    def prepare_subject_epochs(
        self,
        subject_number: str,
    ) -> mne.Epochs:
        """Import, epoch, and behavior-align one raw recording.

        Parameters
        ----------
        subject_number : str
            Subject label used throughout preprocessing.
        Returns
        -------
        mne.Epochs
            Subject epochs after behavior alignment and trial-type filtering.
        """
        eeg, eeg_events, eye, eye_events, has_eye_data, behavior_data = self.load_subject_streams(subject_number)
        epochs = self.build_subject_epochs(
            subject_number,
            eeg,
            eeg_events,
            eye,
            eye_events,
            has_eye_data=has_eye_data,
        )
        return self.keep_aligned_experimental_trials(
            subject_number,
            epochs,
            behavior_data,
        )

    def run_subject_qc(self, epochs: mne.Epochs) -> dict[str, object]:
        """Run the stored QC pipeline on one subject's epochs.

        Parameters
        ----------
        epochs : mne.Epochs
            Subject epochs after alignment and any subject assembly step.

        Returns
        -------
        dict[str, object]
            QC output from ``mveeg.prep.qc.run_subject_artifact_qc``.
        """
        from . import qc as preprocess_qc

        pre = self._require_preprocessor()
        reject_qc, review_qc, autoreject_cfg = self._require_qc_config()
        return preprocess_qc.run_subject_artifact_qc(
            pre,
            epochs,
            reject_qc=reject_qc,
            review_qc=review_qc,
            autoreject_cfg=autoreject_cfg,
        )

    def run_subject_reject_qc(self, epochs: mne.Epochs, progress_callback=None) -> dict[str, object]:
        """Run the hard-rejection QC stage on one subject's epochs.

        Parameters
        ----------
        epochs : mne.Epochs
            Subject epochs after alignment and any subject assembly step.
        progress_callback : callable | None, optional
            Optional callback that receives short status labels during the hard
            reject stage.

        Returns
        -------
        dict[str, object]
            Output from ``mveeg.prep.qc.run_subject_reject_qc``.
        """
        from . import qc as preprocess_qc

        pre = self._require_preprocessor()
        reject_qc, _, _ = self._require_qc_config()
        return preprocess_qc.run_subject_reject_qc(
            pre,
            epochs,
            reject_qc=reject_qc,
            progress_callback=progress_callback,
        )

    def run_subject_autoreject(
        self,
        epochs: mne.Epochs,
        reject_result: dict[str, object],
        progress_callback=None,
    ) -> dict[str, object]:
        """Run only autoreject for one subject and return aligned trial flags.

        Parameters
        ----------
        epochs : mne.Epochs
            Subject epochs after alignment and any subject assembly step.
        reject_result : dict[str, object]
            Output from ``run_subject_reject_qc`` for the same epochs.
        progress_callback : callable | None, optional
            Optional callback that receives short status labels during
            autoreject fitting and interpolation.

        Returns
        -------
        dict[str, object]
            Output from ``mveeg.prep.qc.run_subject_autoreject``.
        """
        from . import qc as preprocess_qc

        pre = self._require_preprocessor()
        _, _, autoreject_cfg = self._require_qc_config()
        return preprocess_qc.run_subject_autoreject(
            pre,
            epochs,
            reject_result=reject_result,
            autoreject_cfg=autoreject_cfg,
            progress_callback=progress_callback,
        )

    def run_subject_soft_review_qc(
        self,
        epochs: mne.Epochs,
        reject_result: dict[str, object],
        *,
        autoreject_bad_epoch: np.ndarray | None = None,
        autoreject_interp_channels: np.ndarray | None = None,
        progress_callback=None,
    ) -> dict[str, object]:
        """Run only soft-review QC using supplied autoreject trial decisions.

        Parameters
        ----------
        epochs : mne.Epochs
            Subject epochs after optional local autoreject interpolation.
        reject_result : dict[str, object]
            Output from ``run_subject_reject_qc`` for the same epochs.
        autoreject_bad_epoch : np.ndarray | None, optional
            Full-length mask showing which trials autoreject still judged as
            bad.
        autoreject_interp_channels : np.ndarray | None, optional
            Full-length count of locally interpolated channels per trial.
        progress_callback : callable | None, optional
            Optional callback that receives short status labels during soft
            review and metadata attachment.

        Returns
        -------
        dict[str, object]
            Output from ``mveeg.prep.qc.run_subject_soft_review_qc``.
        """
        from . import qc as preprocess_qc

        pre = self._require_preprocessor()
        reject_qc, review_qc, _ = self._require_qc_config()
        return preprocess_qc.run_subject_soft_review_qc(
            pre,
            epochs,
            reject_result=reject_result,
            review_qc=review_qc,
            hard_bad_channel_limit=reject_qc["eeg"]["bad_channels"],
            autoreject_bad_epoch=autoreject_bad_epoch,
            autoreject_interp_channels=autoreject_interp_channels,
            progress_callback=progress_callback,
        )

    def summarize_trial_rejection(
        self,
        *,
        stage: str = "auto",
        hard_stage: str = "prepared",
        top_n_reasons: int = 3,
    ) -> None:
        """Print a trial-rejection summary using this workflow's data directory.

        Parameters
        ----------
        stage : str, optional
            Summary source stage. Use ``"hard"``, ``"final"``, or ``"auto"``.
        hard_stage : str, optional
            Intermediate stage label used in hard-reject checkpoint filenames.
        top_n_reasons : int, optional
            Number of most frequent reject reasons printed per subject.

        Returns
        -------
        None
            Prints one summary line per subject.
        """
        from . import qc as preprocess_qc

        preprocess_qc.summarize_trial_rejection(
            self.data_dir,
            stage=stage,
            hard_stage=hard_stage,
            top_n_reasons=top_n_reasons,
        )

    def log_subject_qc_summary(self, subject_number: str) -> None:
        """Print detailed QC summary for one saved subject on demand.

        Parameters
        ----------
        subject_number : str
            Subject label, for example ``"sub2001"``.

        Returns
        -------
        None
            Prints a detailed QC breakdown for the requested subject.
        """
        from . import qc as preprocess_qc

        if not self.has_subject_reject_checkpoint(subject_number):
            print(f"{subject_number}: reject checkpoint not found.")
            return
        if not self.has_subject_autoreject_checkpoint(subject_number):
            print(f"{subject_number}: autoreject checkpoint not found.")
            return

        reject_result = self.load_checkpoint(subject_number, kind="reject")
        autoreject_checkpoint = self.load_checkpoint(subject_number, kind="autoreject")
        review_result = self.run_subject_soft_review_qc(
            autoreject_checkpoint["epochs_for_soft_qc"],
            reject_result,
            autoreject_bad_epoch=autoreject_checkpoint["autoreject_bad_epoch"],
            autoreject_interp_channels=autoreject_checkpoint["autoreject_interp_channels"],
        )
        preprocess_qc.log_subject_qc_summary(review_result, self.reject_qc, self.review_qc)

    def save_subject_data(
        self,
        subject_number: str,
        epochs: mne.Epochs,
        artifact_labels: np.ndarray,
        trial_qc: pd.DataFrame,
    ) -> None:
        """Save one subject's epochs and QC outputs using the stored preprocessor.

        Parameters
        ----------
        subject_number : str
            Subject label used in the saved derivative filenames.
        epochs : mne.Epochs
            Final saved epochs for this subject.
        artifact_labels : np.ndarray
            Channel-wise artifact labels saved alongside the epochs.
        trial_qc : pd.DataFrame
            Trial-level QC table saved alongside the epochs.

        Returns
        -------
        None
            The subject files are written to the workflow's derivative folder.
        """
        pre = self._require_preprocessor()
        pre.save_all_data(subject_number, epochs, artifact_labels, trial_qc)
        self._save_final_trial_state(subject_number, trial_qc)

    def intermediate_epochs_path(self, subject_number: str, *, stage: str = "prepared") -> Path:
        """Return the derivative path for one saved intermediate epochs file.

        Parameters
        ----------
        subject_number : str
            Subject label used in raw-data folders and derivative outputs.
        stage : str, optional
            Short label describing which preprocessing stage the saved epochs
            represent.

        Returns
        -------
        pathlib.Path
            Path where the intermediate epochs file should be written.
        """
        if stage == "prepared":
            return self.final_epochs_path(subject_number)

        io_config = self._require_io_config()
        return derivative_file_path(
            io_config.data_dir,
            subject_number,
            io_config.experiment_name,
            suffix=f"{stage}_epo",
            extension=".fif",
            subject_prefix=io_config.subject_prefix,
            derivative_dirname=io_config.derivative_dirname,
            datatype=io_config.derivative_datatype,
            derivative_label=io_config.derivative_label,
        )

    def final_epochs_path(self, subject_number: str) -> Path:
        """Return the saved final epochs path for one subject."""
        io_config = self._require_io_config()
        return derivative_file_path(
            io_config.data_dir,
            subject_number,
            io_config.experiment_name,
            suffix="epo",
            extension=".fif",
            subject_prefix=io_config.subject_prefix,
            derivative_dirname=io_config.derivative_dirname,
            datatype=io_config.derivative_datatype,
            derivative_label=io_config.derivative_label,
        )

    def trial_qc_path(self, subject_number: str) -> Path:
        """Return the saved trial-level QC table path for one subject."""
        io_config = self._require_io_config()
        return derivative_file_path(
            io_config.data_dir,
            subject_number,
            io_config.experiment_name,
            suffix="trial_qc",
            extension=".tsv",
            subject_prefix=io_config.subject_prefix,
            derivative_dirname=io_config.derivative_dirname,
            datatype=io_config.derivative_datatype,
            derivative_label=io_config.derivative_label,
        )

    def trial_state_path(self, subject_number: str) -> Path:
        """Return the unified trial-state table path for one subject."""
        io_config = self._require_io_config()
        return derivative_file_path(
            io_config.data_dir,
            subject_number,
            io_config.experiment_name,
            suffix="trial_state",
            extension=".tsv",
            subject_prefix=io_config.subject_prefix,
            derivative_dirname=io_config.derivative_dirname,
            datatype=io_config.derivative_datatype,
            derivative_label=io_config.derivative_label,
        )

    def artifact_labels_path(self, subject_number: str) -> Path:
        """Return the saved artifact-label table path for one subject."""
        io_config = self._require_io_config()
        return derivative_file_path(
            io_config.data_dir,
            subject_number,
            io_config.experiment_name,
            suffix="artifacts",
            extension=".tsv",
            subject_prefix=io_config.subject_prefix,
            derivative_dirname=io_config.derivative_dirname,
            datatype=io_config.derivative_datatype,
            derivative_label=io_config.derivative_label,
        )

    def has_saved_intermediate_epochs(self, subject_number: str, *, stage: str = "prepared") -> bool:
        """Return whether the saved intermediate epochs file exists."""
        return self.intermediate_epochs_path(subject_number, stage=stage).exists()

    def has_saved_final_outputs(self, subject_number: str) -> bool:
        """Return whether the final saved outputs for one subject exist."""
        return (
            self.final_epochs_path(subject_number).exists()
            and self.trial_qc_path(subject_number).exists()
            and self.artifact_labels_path(subject_number).exists()
        )

    def save_intermediate_epochs(
        self,
        subject_number: str,
        epochs: mne.Epochs,
        *,
        stage: str = "prepared",
        overwrite: bool = True,
    ) -> Path:
        """Save one intermediate epochs object for a later preprocessing step.

        Parameters
        ----------
        subject_number : str
            Subject label used in raw-data folders and derivative outputs.
        epochs : mne.Epochs
            Epochs to save as an intermediate checkpoint.
        stage : str, optional
            Short label describing which preprocessing stage this file stores.
        overwrite : bool, optional
            Whether an existing intermediate file may be replaced.

        Returns
        -------
        pathlib.Path
            Path of the saved intermediate epochs file.
        """
        epochs_path = self.intermediate_epochs_path(subject_number, stage=stage)
        epochs_path.parent.mkdir(parents=True, exist_ok=True)
        epochs.save(epochs_path, overwrite=overwrite, verbose="ERROR")
        return epochs_path

    def load_trial_state(self, subject_number: str) -> pd.DataFrame | None:
        """Load the saved unified trial-state table when it exists."""
        trial_state_path = self.trial_state_path(subject_number)
        if not trial_state_path.exists():
            return None
        return pd.read_csv(trial_state_path, sep="\t", keep_default_na=False)

    def _write_trial_state(self, subject_number: str, trial_state: pd.DataFrame) -> Path:
        """Write one subject's unified trial-state table in a stable layout."""
        trial_state_path = self.trial_state_path(subject_number)
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

    def _save_reject_trial_state(self, subject_number: str, reject_table: pd.DataFrame) -> Path:
        """Save unified trial state after the hard-reject stage."""
        trial_state = self.load_trial_state(subject_number)
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
        return self._write_trial_state(subject_number, trial_state)

    def _save_autoreject_trial_state(
        self,
        subject_number: str,
        *,
        autoreject_bad_epoch: np.ndarray,
        autoreject_interp_channels: np.ndarray,
    ) -> Path:
        """Save unified trial state after autoreject decisions are known."""
        trial_state = self.load_trial_state(subject_number)
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
        return self._write_trial_state(subject_number, trial_state)

    def _save_final_trial_state(self, subject_number: str, trial_qc: pd.DataFrame) -> Path:
        """Save unified trial state after the final QC table is produced."""
        trial_state = self.load_trial_state(subject_number)
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
        return self._write_trial_state(subject_number, trial_state)

    def reject_qc_table_path(self, subject_number: str, *, stage: str = "prepared") -> Path:
        """Return the derivative path for one saved hard-reject QC table.

        Parameters
        ----------
        subject_number : str
            Subject label used in raw-data folders and derivative outputs.
        stage : str, optional
            Short label describing which intermediate epochs version was used.

        Returns
        -------
        pathlib.Path
            Path where the hard-reject QC table should be written.
        """
        io_config = self._require_io_config()
        return derivative_file_path(
            io_config.data_dir,
            subject_number,
            io_config.experiment_name,
            suffix=f"{stage}_hard_qc",
            extension=".tsv",
            subject_prefix=io_config.subject_prefix,
            derivative_dirname=io_config.derivative_dirname,
            datatype=io_config.derivative_datatype,
            derivative_label=io_config.derivative_label,
        )

    def reject_qc_masks_path(self, subject_number: str, *, stage: str = "prepared") -> Path:
        """Return the derivative path for saved hard-reject masks.

        Parameters
        ----------
        subject_number : str
            Subject label used in raw-data folders and derivative outputs.
        stage : str, optional
            Short label describing which intermediate epochs version was used.

        Returns
        -------
        pathlib.Path
            Path where the hard-reject mask checkpoint should be written.
        """
        io_config = self._require_io_config()
        return derivative_file_path(
            io_config.data_dir,
            subject_number,
            io_config.experiment_name,
            suffix=f"{stage}_hard_qc_masks",
            extension=".npz",
            subject_prefix=io_config.subject_prefix,
            derivative_dirname=io_config.derivative_dirname,
            datatype=io_config.derivative_datatype,
            derivative_label=io_config.derivative_label,
        )

    def autoreject_masks_path(self, subject_number: str, *, stage: str = "prepared") -> Path:
        """Return the derivative path for saved autoreject trial masks.

        Parameters
        ----------
        subject_number : str
            Subject label used in raw-data folders and derivative outputs.
        stage : str, optional
            Short label describing which intermediate epochs version was used.

        Returns
        -------
        pathlib.Path
            Path where the autoreject checkpoint masks should be written.
        """
        io_config = self._require_io_config()
        return derivative_file_path(
            io_config.data_dir,
            subject_number,
            io_config.experiment_name,
            suffix=f"{stage}_autoreject_masks",
            extension=".npz",
            subject_prefix=io_config.subject_prefix,
            derivative_dirname=io_config.derivative_dirname,
            datatype=io_config.derivative_datatype,
            derivative_label=io_config.derivative_label,
        )

    def load_intermediate_epochs(
        self,
        subject_number: str,
        *,
        stage: str = "prepared",
        preload: bool = True,
    ) -> mne.Epochs:
        """Load one previously saved intermediate epochs file.

        Parameters
        ----------
        subject_number : str
            Subject label used in raw-data folders and derivative outputs.
        stage : str, optional
            Short label describing which preprocessing stage should be loaded.
        preload : bool, optional
            Whether to read the epochs data into memory immediately.

        Returns
        -------
        mne.Epochs
            Intermediate epochs loaded from disk.
        """
        epochs_path = self.intermediate_epochs_path(subject_number, stage=stage)
        return mne.read_epochs(epochs_path, preload=preload, verbose="ERROR")

    def save_subject_reject_checkpoint(
        self,
        subject_number: str,
        epochs: mne.Epochs,
        reject_result: dict[str, object],
        *,
        stage: str = "prepared",
    ) -> tuple[Path, Path]:
        """Save hard-reject QC outputs so later steps can resume from them.

        Parameters
        ----------
        subject_number : str
            Subject label used in raw-data folders and derivative outputs.
        epochs : mne.Epochs
            Intermediate epochs used to compute the hard-reject stage.
        reject_result : dict[str, object]
            Output from ``run_subject_reject_qc`` for these epochs.
        stage : str, optional
            Short label describing which intermediate epochs version was used.

        Returns
        -------
        tuple[pathlib.Path, pathlib.Path]
            Saved hard-reject trial table path and hard-reject mask checkpoint
            path.
        """
        from . import qc as preprocess_qc

        pre = self._require_preprocessor()
        reject_qc, _, _ = self._require_qc_config()
        table_path = self.reject_qc_table_path(subject_number, stage=stage)
        masks_path = self.reject_qc_masks_path(subject_number, stage=stage)
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
        self._save_reject_trial_state(subject_number, reject_table)
        return table_path, masks_path

    def has_subject_reject_checkpoint(
        self,
        subject_number: str,
        *,
        stage: str = "prepared",
    ) -> bool:
        """Return whether both hard-reject checkpoint files already exist."""
        table_path = self.reject_qc_table_path(subject_number, stage=stage)
        masks_path = self.reject_qc_masks_path(subject_number, stage=stage)
        return table_path.exists() and masks_path.exists()

    def save_subject_autoreject_checkpoint(
        self,
        subject_number: str,
        autoreject_result: dict[str, object],
        *,
        source_stage: str = "prepared",
        cleaned_stage: str = "prepared_autoreject",
        overwrite: bool = True,
    ) -> tuple[Path, Path]:
        """Save autoreject masks plus cleaned epochs for later soft-review QC.

        Parameters
        ----------
        subject_number : str
            Subject label used in raw-data folders and derivative outputs.
        autoreject_result : dict[str, object]
            Output from ``run_subject_autoreject`` for this subject.
        source_stage : str, optional
            Stage label of the epochs that were given to autoreject.
        cleaned_stage : str, optional
            Stage label used when saving autoreject-cleaned intermediate epochs.
        overwrite : bool, optional
            Whether existing saved checkpoint files may be replaced.

        Returns
        -------
        tuple[pathlib.Path, pathlib.Path]
            Saved cleaned-epochs path and autoreject-mask checkpoint path.
        """
        cleaned_epochs_path = self.final_epochs_path(subject_number)
        cleaned_epochs_path.parent.mkdir(parents=True, exist_ok=True)
        autoreject_result["epochs_for_soft_qc"].save(
            cleaned_epochs_path,
            overwrite=overwrite,
            verbose="ERROR",
        )
        masks_path = self.autoreject_masks_path(subject_number, stage=source_stage)
        masks_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            masks_path,
            autoreject_bad_epoch=np.asarray(autoreject_result["autoreject_bad_epoch"], dtype=bool),
            autoreject_interp_channels=np.asarray(autoreject_result["autoreject_interp_channels"], dtype=int),
        )
        self._save_autoreject_trial_state(
            subject_number,
            autoreject_bad_epoch=np.asarray(autoreject_result["autoreject_bad_epoch"], dtype=bool),
            autoreject_interp_channels=np.asarray(autoreject_result["autoreject_interp_channels"], dtype=int),
        )
        return cleaned_epochs_path, masks_path

    def load_checkpoint(
        self,
        subject_number: str,
        *,
        kind: str,
        stage: str = "prepared",
        cleaned_stage: str = "prepared_autoreject",
    ) -> dict[str, object]:
        """Load one checkpoint for one subject.

        Parameters
        ----------
        subject_number : str
            Subject label used in raw-data folders and derivative outputs.
        kind : str
            Checkpoint type. Supported values are ``"reject"`` and
            ``"autoreject"``.
        stage : str, optional
            Stage label for checkpoint masks. For ``kind="reject"``, this is
            the hard-reject stage. For ``kind="autoreject"``, this is the
            source stage used when autoreject was run.
        cleaned_stage : str, optional
            Stage label of cleaned intermediate epochs used for
            ``kind="autoreject"``.

        Returns
        -------
        dict[str, object]
            Checkpoint content dictionary for the requested type.
        """
        if kind == "reject":
            reject_qc, _, _ = self._require_qc_config()
            masks_path = self.reject_qc_masks_path(subject_number, stage=stage)
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
                self.final_epochs_path(subject_number),
                preload=True,
                verbose="ERROR",
            )
            masks_path = self.autoreject_masks_path(subject_number, stage=stage)
            with np.load(masks_path) as saved:
                return {
                    "epochs_for_soft_qc": epochs_for_soft_qc,
                    "autoreject_bad_epoch": saved["autoreject_bad_epoch"].astype(bool),
                    "autoreject_interp_channels": saved["autoreject_interp_channels"].astype(int),
                }

        raise ValueError(f"Unsupported checkpoint kind '{kind}'. Use reject or autoreject.")

    def has_subject_autoreject_checkpoint(
        self,
        subject_number: str,
        *,
        source_stage: str = "prepared",
        cleaned_stage: str = "prepared_autoreject",
    ) -> bool:
        """Return whether autoreject masks and cleaned epochs both exist."""
        masks_path = self.autoreject_masks_path(subject_number, stage=source_stage)
        epochs_path = self.final_epochs_path(subject_number)
        return masks_path.exists() and epochs_path.exists()

    def build_subject_intermediate_epochs(
        self,
        subject_number: str,
        subject_assembly_plan: dict[str, str],
        *,
        stage: str = "prepared",
        overwrite: bool = True,
    ) -> Path:
        """Prepare one subject and save intermediate epochs before trial QC.

        Parameters
        ----------
        subject_number : str
            Subject label used in raw-data folders and saved outputs.
        subject_assembly_plan : dict[str, str]
            Optional donor-subject mapping used when one saved subject should be
            assembled from multiple recordings.
        stage : str, optional
            Short label describing which preprocessing stage this file stores.
        overwrite : bool, optional
            Whether an existing intermediate file may be replaced.

        Returns
        -------
        pathlib.Path
            Path of the saved intermediate epochs file.
        """
        epochs = self.prepare_subject_epochs(subject_number)
        if len(subject_assembly_plan) > 0:
            epochs = self.assemble_subject_epochs(
                subject_number=subject_number,
                base_epochs=epochs,
                assembly_plan=subject_assembly_plan,
            )
        epochs = reset_epoch_event_samples(epochs)
        return self.save_intermediate_epochs(subject_number, epochs, stage=stage, overwrite=overwrite)

    def build_all_intermediate_epochs(
        self,
        *,
        subject_assembly_plans: dict[str, dict[str, str]] | None = None,
        stage: str = "prepared",
        overwrite: bool = True,
    ) -> None:
        """Build and save intermediate epochs for all selected subjects.

        Parameters
        ----------
        subject_assembly_plans : dict[str, dict[str, str]] | None, optional
            Optional mapping from subject label to donor-subject assembly plan.
        stage : str, optional
            Short label describing which preprocessing stage this file stores.
        overwrite : bool, optional
            Step-level overwrite switch. The final overwrite decision is
            resolved inside the workflow using the stored global overwrite
            setting.

        Returns
        -------
        None
            Intermediate epochs are written to disk for each selected subject.
        """
        overwrite = self.should_overwrite(overwrite)
        subject_dirs = self.subject_dirs
        if not overwrite:
            subject_dirs = [
                subject_dir
                for subject_dir in subject_dirs
                if not self.has_saved_intermediate_epochs(subject_dir.name, stage=stage)
            ]
        total_subjects = len(subject_dirs)
        if total_subjects == 0:
            print("No subjects selected for preprocessing. Check overwrite settings and selected_subjects.")
            return

        subject_assembly_plans = {} if subject_assembly_plans is None else subject_assembly_plans
        subject_bar = tqdm(
            subject_dirs,
            total=total_subjects,
            desc="Build intermediate epochs",
            unit="subject",
        )

        for subject_path in subject_bar:
            subject_number = subject_path.name
            subject_assembly_plan = subject_assembly_plans.get(subject_number, {})
            subject_bar.set_postfix_str(subject_number)
            self.build_subject_intermediate_epochs(
                subject_number,
                subject_assembly_plan,
                stage=stage,
                overwrite=overwrite,
            )
        subject_bar.close()

    def finish_subject_from_intermediate(
        self,
        subject_number: str,
        *,
        stage: str = "prepared",
        reuse_saved_reject: bool = True,
        reject_progress_callback=None,
    ) -> dict[str, object]:
        """Load intermediate epochs, run QC, and save final subject outputs.

        Parameters
        ----------
        subject_number : str
            Subject label used in raw-data folders and saved outputs.
        stage : str, optional
            Intermediate stage label used when the epochs were saved.
        reuse_saved_reject : bool, optional
            If ``True``, reuse a saved hard-reject checkpoint when it already
            exists instead of recomputing the hard-reject stage.
        reject_progress_callback : callable | None, optional
            Optional callback that receives short status labels during the hard
            reject stage when a saved checkpoint is not reused.

        Returns
        -------
        dict[str, object]
            Final QC output including cleaned epochs, artifact labels, and the
            saved trial-level QC table.
        """
        from . import qc as preprocess_qc

        epochs = self.load_intermediate_epochs(subject_number, stage=stage)
        if reuse_saved_reject and self.has_subject_reject_checkpoint(subject_number, stage=stage):
            reject_result = self.load_checkpoint(subject_number, kind="reject", stage=stage)
        else:
            reject_result = self.run_subject_reject_qc(
                epochs,
                progress_callback=reject_progress_callback,
            )
            self.save_subject_reject_checkpoint(
                subject_number,
                epochs,
                reject_result,
                stage=stage,
            )
        autoreject_result = self.run_subject_autoreject(epochs, reject_result)
        qc_result = self.run_subject_soft_review_qc(
            autoreject_result["epochs_for_soft_qc"],
            reject_result,
            autoreject_bad_epoch=autoreject_result["autoreject_bad_epoch"],
            autoreject_interp_channels=autoreject_result["autoreject_interp_channels"],
        )
        epochs = qc_result["epochs"]
        artifact_labels = qc_result["artifact_labels"]
        trial_qc = qc_result["trial_qc"]
        preprocess_qc.log_subject_qc_summary(qc_result, self.reject_qc, self.review_qc)
        self.save_subject_data(subject_number, epochs, artifact_labels, trial_qc)
        return qc_result

    def assemble_subject_epochs(
        self,
        *,
        subject_number: str,
        base_epochs: mne.Epochs,
        assembly_plan: dict[str, str],
        condition_column: str = "condition",
    ) -> mne.Epochs:
        """Assemble one saved subject from one or more source recordings.

        Parameters
        ----------
        subject_number : str
            Subject label for the saved output subject.
        base_epochs : mne.Epochs
            Standard preprocessed epochs from the subject's main recording.
        assembly_plan : dict[str, str]
            Mapping from condition value to source subject label, for example
            ``{"irr": "sub10021"}``.
        condition_column : str, optional
            Metadata column that defines which condition values should be
            assembled across source subjects.
        Returns
        -------
        mne.Epochs
            Concatenated epochs assembled from the requested source recordings.

        Raises
        ------
        ValueError
            If no subject-directory map is available for donor recordings.
        """
        if not self.subject_dir_map:
            raise ValueError("assemble_subject_epochs requires subject_dir_map when donor recordings are used.")
        if base_epochs.metadata is None or condition_column not in base_epochs.metadata.columns:
            raise ValueError(f"Base epochs metadata must contain '{condition_column}' for subject assembly.")

        epoch_parts = []
        condition_values = [
            str(value)
            for value in base_epochs.metadata[condition_column].dropna().drop_duplicates().tolist()
        ]
        keep_map = {condition_value: subject_number for condition_value in condition_values}
        keep_map.update(assembly_plan)

        print(f"Assembly plan for {subject_number}: {keep_map}")
        for condition_value, source_subject in keep_map.items():
            if source_subject == subject_number:
                source_epochs = base_epochs.copy()
            else:
                source_epochs = self.prepare_subject_epochs(source_subject)

            source_epochs = keep_epochs_by_metadata_value(source_epochs, condition_column, condition_value)
            epoch_parts.append(source_epochs)
            print(
                f"Using {len(source_epochs)} {condition_value} trials from {source_subject} "
                f"for saved subject {subject_number}."
            )

        return mne.concatenate_epochs(epoch_parts, add_offset=False)


def keep_epochs_by_metadata_value(epochs: mne.Epochs, column: str, value: str) -> mne.Epochs:
    """Keep epochs whose metadata column matches one requested value."""
    if epochs.metadata is None or column not in epochs.metadata.columns:
        raise ValueError(f"Aligned epochs metadata must contain a '{column}' column.")

    keep_trials = epochs.metadata[column].eq(value).to_numpy()
    if not np.any(keep_trials):
        raise ValueError(f"No rows matched {column} == {value!r}.")
    epochs_kept = epochs.copy()
    drop_ix = np.flatnonzero(~keep_trials).tolist()
    if len(drop_ix) > 0:
        epochs_kept.drop(drop_ix, verbose="ERROR")
    return epochs_kept


def reset_epoch_event_samples(epochs: mne.Epochs) -> mne.Epochs:
    """Rewrite epoch event sample numbers so they are strictly increasing."""
    epochs_reset = epochs.copy()
    epochs_reset.events[:, 0] = np.arange(len(epochs_reset), dtype=int)
    return epochs_reset


def write_run_log_header(
    output_file: TextIO,
    *,
    total_subjects: int,
    log_path: str | Path,
    started_at: datetime | None = None,
) -> None:
    """Write a short preprocessing-run header to the log file.

    Parameters
    ----------
    output_file : TextIO
        Open text stream used for the preprocessing log.
    total_subjects : int
        Number of subjects selected for the current run.
    log_path : str | Path
        Location of the log file on disk.
    started_at : datetime | None, optional
        Run start time. If omitted, the current time is used.

    Returns
    -------
    None
        The function writes a formatted run header to ``output_file``.
    """
    started_at = datetime.now() if started_at is None else started_at
    output_file.write("\n" + "=" * 72 + "\n")
    output_file.write("PREPROCESSING RUN\n")
    output_file.write("=" * 72 + "\n")
    output_file.write(f"Started: {started_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
    output_file.write(f"Subjects in this run: {total_subjects}\n")
    output_file.write(f"Log file: {Path(log_path).resolve()}\n\n")


def write_subject_log_header(
    output_file: TextIO,
    *,
    subject_ix: int,
    total_subjects: int,
    subject_number: str,
) -> None:
    """Write one subject header to the preprocessing log.

    Parameters
    ----------
    output_file : TextIO
        Open text stream used for the preprocessing log.
    subject_ix : int
        One-based index of the current subject in this run.
    total_subjects : int
        Total number of subjects selected for the run.
    subject_number : str
        Subject label shown in the log.

    Returns
    -------
    None
        The function writes a formatted subject header to ``output_file``.
    """
    output_file.write("\n" + "-" * 72 + "\n")
    output_file.write(f"Subject {subject_ix}/{total_subjects}: {subject_number}\n")
    output_file.write("-" * 72 + "\n")


@contextmanager
def redirect_output_to_file(output_file):
    """Temporarily redirect stdout and stderr to an open text stream."""
    import sys

    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = output_file
    sys.stderr = sys.stdout
    try:
        yield None
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr
