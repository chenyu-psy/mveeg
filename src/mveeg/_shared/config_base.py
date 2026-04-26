"""Base configuration objects shared by analysis workflows.

This module holds only dataset and trial-filter fields that are common to both
encoding and decoding pipelines. Task-specific modeling parameters should stay
in the task modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataPathsConfig:
    """Dataset paths and derivative naming shared across analysis workflows.

    Parameters
    ----------
    data_dir : str | Path
        Root folder containing the subject derivatives.
    experiment_name : str
        Experiment tag used in derivative filenames.
    subject_prefix : str
        Prefix used in subject folder labels.
    derivative_dirname : str
        Name of the derivatives folder under ``data_dir``.
    derivative_datatype : str
        BIDS datatype folder inside derivatives.
    derivative_label : str
        ``desc-`` tag used in derivative filenames.
    """

    data_dir: str | Path
    experiment_name: str
    subject_prefix: str = "sub"
    derivative_dirname: str = "derivatives"
    derivative_datatype: str = "eeg"
    derivative_label: str = "preprocessed"


@dataclass
class TrialFilterRulesConfig:
    """Trial-level filtering rules shared across analysis workflows.

    Parameters
    ----------
    qc_col : str | None
        Metadata column used as the main quality-control label. Set ``None`` to
        skip QC-based filtering.
    keep_qc : tuple[str, ...]
        Allowed QC labels in ``qc_col``.
    exclude_metadata : dict[str, tuple | str] | None
        Additional per-column exclusion rules. ``"notna"`` means "exclude
        non-missing rows". A tuple means "exclude these values".
    """

    qc_col: str | None = "final_qc_category"
    keep_qc: tuple[str, ...] = ("accepted",)
    exclude_metadata: dict[str, tuple | str] | None = None


@dataclass
class ConditionGroupsConfig:
    """Condition grouping shared across analysis workflows.

    Parameters
    ----------
    train_cond : dict[str, list[str]] | list[str]
        Mapping from analysis labels to raw condition values, or an explicit
        list of raw condition values.
    test_cond : dict[str, list[str]] | list[str]
        Condition groups kept after filtering.
    cond_col : str
        Metadata column that stores raw condition values.
    """

    train_cond: dict[str, list[str]] | list[str]
    test_cond: dict[str, list[str]] | list[str]
    cond_col: str


@dataclass
class EpochProcessingConfig:
    """Epoch-level settings shared across analysis workflows.

    Parameters
    ----------
    crop_time : tuple[float, float] | None
        Time interval kept from each epoch in seconds.
    drop_channel_types : tuple[str, ...]
        Channel types removed before analysis.
    drop_channels : tuple[str, ...]
        Additional channel names removed before analysis.
    """

    crop_time: tuple[float, float] | None = None
    drop_channel_types: tuple[str, ...] = ("eog", "eyegaze", "pupil", "misc")
    drop_channels: tuple[str, ...] = ()


@dataclass
class SubjectLoadConfig:
    """Grouped shared config for loading and filtering subject data.

    Parameters
    ----------
    dataset : DataPathsConfig
        Dataset and derivative path settings.
    conditions : ConditionGroupsConfig
        Condition grouping definitions.
    filters : TrialFilterRulesConfig
        Trial-level filtering rules.
    epoch : EpochProcessingConfig
        Epoch cropping and channel-drop settings.
    """

    dataset: DataPathsConfig
    conditions: ConditionGroupsConfig
    filters: TrialFilterRulesConfig
    epoch: EpochProcessingConfig
