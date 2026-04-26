"""Derivative path helpers for preprocessing workflow checkpoints."""

from __future__ import annotations

from pathlib import Path

from ..io.bids import derivative_file_path


def intermediate_epochs_path(io_config, subject_number: str, *, stage: str = "prepared") -> Path:
    """Return the derivative path for one saved intermediate epochs file."""
    if stage == "prepared":
        return final_epochs_path(io_config, subject_number)
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


def final_epochs_path(io_config, subject_number: str) -> Path:
    """Return the saved final epochs path for one subject."""
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


def trial_qc_path(io_config, subject_number: str) -> Path:
    """Return the saved trial-level QC table path for one subject."""
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


def trial_state_path(io_config, subject_number: str) -> Path:
    """Return the unified trial-state table path for one subject."""
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


def artifact_labels_path(io_config, subject_number: str) -> Path:
    """Return the saved artifact-label table path for one subject."""
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


def reject_qc_table_path(io_config, subject_number: str, *, stage: str = "prepared") -> Path:
    """Return the derivative path for one saved hard-reject QC table."""
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


def reject_qc_masks_path(io_config, subject_number: str, *, stage: str = "prepared") -> Path:
    """Return the derivative path for saved hard-reject masks."""
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


def autoreject_masks_path(io_config, subject_number: str, *, stage: str = "prepared") -> Path:
    """Return the derivative path for saved autoreject trial masks."""
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
