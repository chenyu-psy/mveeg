"""Workflow helpers that sit on top of the core EEG preprocessing engine.

This module keeps script code short and readable by storing preprocessing
setup in one explicit workflow object. Use ``create_flow()`` to build the
workflow, configure it in stages, then run subject-level preprocessing with
its methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mne
import numpy as np
import pandas as pd

from . import core as preprocess_core
from .epochs import keep_epochs_by_metadata_value, reset_epoch_event_samples
from .logging import (
    redirect_output_to_file,
    write_run_log_header,
    write_subject_log_header,
)
from .subject_selection import get_subject_selection
from . import workflow_paths
from . import workflow_status
from . import workflow_checkpoints
from .workflow_status import STATUS_COLUMNS
from . import workflow_subjects

if TYPE_CHECKING:
    from .core import Preprocess


def create_flow() -> "PreprocessWorkflow":
    """Create an empty preprocessing workflow.

    Returns
    -------
    PreprocessWorkflow
        Workflow object ready for stepwise configuration.
    """
    return PreprocessWorkflow()


HARD_REJECT_STEP_BASE = [
    "dropout",
    "flatline",
    "hard abs",
    "gaze deviation",
    "gaze shift",
    "hard step",
    "hard p2p",
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
    4. Optionally call ``configure_behavior(...)`` when behavior rows should
       be aligned to EEG epochs.
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
        self.behavior_name_pattern: str | None = None
        self.behavior_suffix: str | None = None
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
        return workflow_status.empty_status_table()

    def _status_table_subjects(self) -> list[str]:
        """Return selected subject labels in the current workflow order."""
        return workflow_status.status_table_subjects(self.subject_dirs)

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
        return workflow_status.load_status_table(status_path, self.subject_dirs)

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
        return workflow_status.save_status_table(status_path, status_table)

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
        return workflow_status.update_subject_status(
            status_path=status_path,
            subject_dirs=self.subject_dirs,
            subject_number=subject_number,
            intermediate_saved=intermediate_saved,
            hard_qc_saved=hard_qc_saved,
            autoreject_done=autoreject_done,
            final_saved=final_saved,
            last_completed_step=last_completed_step,
            status=status,
            error_message=error_message,
        )

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
        return workflow_status.should_overwrite(overwrite_step, overwrite_all)

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
        name_pattern: str | None = None,
        behavior_suffix: str | None = None,
        pre_filter_rules: dict | None = None,
        post_filter_rules: dict | None = None,
        manual_trial_exclusions: dict[str, dict[str, list[int]]] | None = None,
    ) -> None:
        """Store behavior-alignment and manual-fix settings for this run.

        Parameters
        ----------
        name_pattern : str | None, optional
            Glob pattern used inside each subject folder to find the behavior
            CSV, for example ``"*_beh.csv"``. When ``None``, this workflow does
            not load or align behavior data.
        behavior_suffix : str | None, optional
            Compatibility argument for older scripts. ``"_beh.csv"`` is mapped
            to ``name_pattern="*_beh.csv"``.
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
        if name_pattern is None and behavior_suffix is not None:
            name_pattern = f"*{behavior_suffix}"
        self.behavior_name_pattern = name_pattern
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
    ) -> tuple[mne.io.BaseRaw, np.ndarray, mne.io.BaseRaw | None, np.ndarray | None, bool, pd.DataFrame | None]:
        """Load one subject's EEG, optional eye data, and behavior table."""
        return workflow_subjects.load_subject_streams(self, subject_number)

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
        """Build epochs from imported recording streams for one subject."""
        return workflow_subjects.build_subject_epochs(
            self,
            subject_number,
            eeg,
            eeg_events,
            eye,
            eye_events,
            has_eye_data=has_eye_data,
        )

    def keep_aligned_experimental_trials(
        self,
        subject_number: str,
        epochs: mne.Epochs,
        behavior_data: pd.DataFrame | None,
    ) -> mne.Epochs:
        """Align epochs to behavior rows when behavior data is configured."""
        return workflow_subjects.keep_aligned_experimental_trials(
            self,
            subject_number,
            epochs,
            behavior_data,
        )

    def prepare_subject_epochs(
        self,
        subject_number: str,
    ) -> mne.Epochs:
        """Import, epoch, and behavior-align one raw recording."""
        return workflow_subjects.prepare_subject_epochs(self, subject_number)

    def run_subject_qc(self, epochs: mne.Epochs) -> dict[str, object]:
        """Run the stored QC pipeline on one subject's epochs."""
        return workflow_subjects.run_subject_qc(self, epochs)

    def run_subject_reject_qc(self, epochs: mne.Epochs, progress_callback=None) -> dict[str, object]:
        """Run the hard-rejection QC stage on one subject's epochs."""
        return workflow_subjects.run_subject_reject_qc(self, epochs, progress_callback=progress_callback)

    def run_subject_autoreject(
        self,
        epochs: mne.Epochs,
        reject_result: dict[str, object],
        progress_callback=None,
    ) -> dict[str, object]:
        """Run only autoreject for one subject and return aligned trial flags."""
        return workflow_subjects.run_subject_autoreject(
            self,
            epochs,
            reject_result,
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
        """Run only soft-review QC using supplied autoreject trial decisions."""
        return workflow_subjects.run_subject_soft_review_qc(
            self,
            epochs,
            reject_result,
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
        """Print a trial-rejection summary using this workflow's data directory."""
        workflow_subjects.summarize_trial_rejection(
            self,
            stage=stage,
            hard_stage=hard_stage,
            top_n_reasons=top_n_reasons,
        )

    def log_subject_qc_summary(self, subject_number: str) -> None:
        """Print detailed QC summary for one saved subject on demand."""
        workflow_subjects.log_subject_qc_summary(self, subject_number)

    def save_subject_data(
        self,
        subject_number: str,
        epochs: mne.Epochs,
        artifact_labels: np.ndarray,
        trial_qc: pd.DataFrame,
    ) -> None:
        """Save one subject's epochs and QC outputs using the stored preprocessor."""
        workflow_subjects.save_subject_data(self, subject_number, epochs, artifact_labels, trial_qc)

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
        return workflow_paths.intermediate_epochs_path(
            self._require_io_config(),
            subject_number,
            stage=stage,
        )

    def final_epochs_path(self, subject_number: str) -> Path:
        """Return the saved final epochs path for one subject."""
        return workflow_paths.final_epochs_path(self._require_io_config(), subject_number)

    def trial_qc_path(self, subject_number: str) -> Path:
        """Return the saved trial-level QC table path for one subject."""
        return workflow_paths.trial_qc_path(self._require_io_config(), subject_number)

    def trial_state_path(self, subject_number: str) -> Path:
        """Return the unified trial-state table path for one subject."""
        return workflow_paths.trial_state_path(self._require_io_config(), subject_number)

    def artifact_labels_path(self, subject_number: str) -> Path:
        """Return the saved artifact-label table path for one subject."""
        return workflow_paths.artifact_labels_path(self._require_io_config(), subject_number)

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
        return workflow_checkpoints.load_trial_state(self, subject_number)

    def _write_trial_state(self, subject_number: str, trial_state: pd.DataFrame) -> Path:
        """Write one subject's unified trial-state table in a stable layout."""
        return workflow_checkpoints.write_trial_state(self, subject_number, trial_state)

    def _save_reject_trial_state(self, subject_number: str, reject_table: pd.DataFrame) -> Path:
        """Save unified trial state after the hard-reject stage."""
        return workflow_checkpoints.save_reject_trial_state(self, subject_number, reject_table)

    def _save_autoreject_trial_state(
        self,
        subject_number: str,
        *,
        autoreject_bad_epoch: np.ndarray,
        autoreject_interp_channels: np.ndarray,
    ) -> Path:
        """Save unified trial state after autoreject decisions are known."""
        return workflow_checkpoints.save_autoreject_trial_state(
            self,
            subject_number,
            autoreject_bad_epoch=autoreject_bad_epoch,
            autoreject_interp_channels=autoreject_interp_channels,
        )

    def _save_final_trial_state(self, subject_number: str, trial_qc: pd.DataFrame) -> Path:
        """Save unified trial state after the final QC table is produced."""
        return workflow_checkpoints.save_final_trial_state(self, subject_number, trial_qc)

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
        return workflow_paths.reject_qc_table_path(
            self._require_io_config(),
            subject_number,
            stage=stage,
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
        return workflow_paths.reject_qc_masks_path(
            self._require_io_config(),
            subject_number,
            stage=stage,
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
        return workflow_paths.autoreject_masks_path(
            self._require_io_config(),
            subject_number,
            stage=stage,
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
        return workflow_checkpoints.save_subject_reject_checkpoint(
            self,
            subject_number,
            epochs,
            reject_result,
            stage=stage,
        )

    def has_subject_reject_checkpoint(
        self,
        subject_number: str,
        *,
        stage: str = "prepared",
    ) -> bool:
        """Return whether both hard-reject checkpoint files already exist."""
        return workflow_checkpoints.has_subject_reject_checkpoint(
            self,
            subject_number,
            stage=stage,
        )

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
        return workflow_checkpoints.save_subject_autoreject_checkpoint(
            self,
            subject_number,
            autoreject_result,
            source_stage=source_stage,
            cleaned_stage=cleaned_stage,
            overwrite=overwrite,
        )

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
        return workflow_checkpoints.load_checkpoint(
            self,
            subject_number,
            kind=kind,
            stage=stage,
            cleaned_stage=cleaned_stage,
        )

    def has_subject_autoreject_checkpoint(
        self,
        subject_number: str,
        *,
        source_stage: str = "prepared",
        cleaned_stage: str = "prepared_autoreject",
    ) -> bool:
        """Return whether autoreject masks and cleaned epochs both exist."""
        return workflow_checkpoints.has_subject_autoreject_checkpoint(
            self,
            subject_number,
            source_stage=source_stage,
            cleaned_stage=cleaned_stage,
        )

    def build_subject_intermediate_epochs(
        self,
        subject_number: str,
        subject_assembly_plan: dict[str, str],
        *,
        stage: str = "prepared",
        overwrite: bool = True,
    ) -> Path:
        """Prepare one subject and save intermediate epochs before trial QC."""
        return workflow_subjects.build_subject_intermediate_epochs(
            self,
            subject_number,
            subject_assembly_plan,
            stage=stage,
            overwrite=overwrite,
        )

    def build_all_intermediate_epochs(
        self,
        *,
        subject_assembly_plans: dict[str, dict[str, str]] | None = None,
        stage: str = "prepared",
        overwrite: bool = True,
    ) -> None:
        """Build and save intermediate epochs for all selected subjects."""
        workflow_subjects.build_all_intermediate_epochs(
            self,
            subject_assembly_plans=subject_assembly_plans,
            stage=stage,
            overwrite=overwrite,
        )

    def finish_subject_from_intermediate(
        self,
        subject_number: str,
        *,
        stage: str = "prepared",
        reuse_saved_reject: bool = True,
        reject_progress_callback=None,
    ) -> dict[str, object]:
        """Load intermediate epochs, run QC, and save final subject outputs."""
        return workflow_subjects.finish_subject_from_intermediate(
            self,
            subject_number,
            stage=stage,
            reuse_saved_reject=reuse_saved_reject,
            reject_progress_callback=reject_progress_callback,
        )

    def assemble_subject_epochs(
        self,
        *,
        subject_number: str,
        base_epochs: mne.Epochs,
        assembly_plan: dict[str, str],
        condition_column: str = "condition",
    ) -> mne.Epochs:
        """Assemble one saved subject from one or more source recordings."""
        return workflow_subjects.assemble_subject_epochs(
            self,
            subject_number=subject_number,
            base_epochs=base_epochs,
            assembly_plan=assembly_plan,
            condition_column=condition_column,
        )
