"""Shared subject loading and trial-filtering utilities.

These helpers centralize metadata handling and trial filtering so decoding and
encoding modules apply exactly the same inclusion rules.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from ..io.bids import derivative_file_path


def epochs_path_for_subject(subject_id: str, dataset_cfg) -> Path:
    """Return the saved epochs path for one subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier without the ``sub-`` prefix.
    dataset_cfg : object
        Config-like object with dataset fields used by ``derivative_file_path``.

    Returns
    -------
    Path
        Path to the subject epochs FIF file.
    """

    return derivative_file_path(
        dataset_cfg.data_dir,
        subject_id,
        dataset_cfg.experiment_name,
        "epo",
        ".fif",
        subject_prefix=dataset_cfg.subject_prefix,
        derivative_dirname=dataset_cfg.derivative_dirname,
        datatype=dataset_cfg.derivative_datatype,
        derivative_label=dataset_cfg.derivative_label,
    )


def events_path_for_subject(subject_id: str, dataset_cfg) -> Path:
    """Return the events sidecar path for one subject."""

    return derivative_file_path(
        dataset_cfg.data_dir,
        subject_id,
        dataset_cfg.experiment_name,
        "events",
        ".tsv",
        subject_prefix=dataset_cfg.subject_prefix,
        derivative_dirname=dataset_cfg.derivative_dirname,
        datatype=dataset_cfg.derivative_datatype,
        derivative_label=dataset_cfg.derivative_label,
    )


def trial_qc_path_for_subject(subject_id: str, dataset_cfg) -> Path:
    """Return the trial-QC sidecar path for one subject."""

    return derivative_file_path(
        dataset_cfg.data_dir,
        subject_id,
        dataset_cfg.experiment_name,
        "trial_qc",
        ".tsv",
        subject_prefix=dataset_cfg.subject_prefix,
        derivative_dirname=dataset_cfg.derivative_dirname,
        datatype=dataset_cfg.derivative_datatype,
        derivative_label=dataset_cfg.derivative_label,
    )


def load_events_table(subject_id: str, dataset_cfg) -> pd.DataFrame:
    """Load the saved events table for one subject."""

    return pd.read_csv(events_path_for_subject(subject_id, dataset_cfg), sep="\t")


def load_trial_qc_table(subject_id: str, dataset_cfg) -> pd.DataFrame | None:
    """Load the optional trial-QC table for one subject.

    Returns
    -------
    pd.DataFrame | None
        Trial-QC table when present, otherwise ``None``.
    """

    trial_qc_path = trial_qc_path_for_subject(subject_id, dataset_cfg)
    if not trial_qc_path.exists():
        return None
    return pd.read_csv(trial_qc_path, sep="\t")


def channels_to_drop_by_rule(epochs: mne.Epochs, decode_cfg) -> list[str]:
    """Return channels removed according to decode settings.

    Parameters
    ----------
    epochs : mne.Epochs
        Subject epochs before channel removal.
    decode_cfg : object
        Config-like object with ``drop_channel_types`` and ``drop_channels``.

    Returns
    -------
    list[str]
        Sorted channel names that should be dropped.
    """

    selected = set()
    chan_types = epochs.get_channel_types()
    for ch_name, ch_type in zip(epochs.ch_names, chan_types):
        if ch_type in decode_cfg.drop_channel_types:
            selected.add(ch_name)
    for ch_name in decode_cfg.drop_channels:
        if ch_name in epochs.ch_names:
            selected.add(ch_name)
    return sorted(selected)


def load_subject_metadata_table(subject_id: str, cfg) -> pd.DataFrame:
    """Load metadata for one subject and attach QC labels when needed.

    Parameters
    ----------
    subject_id : str
        Subject identifier without the ``sub-`` prefix.
    cfg : object
        Config-like object with ``dataset`` and ``filters`` fields.

    Returns
    -------
    pd.DataFrame
        Trial metadata table aligned with epoch rows.
    """

    epochs_path = epochs_path_for_subject(subject_id, cfg.dataset)
    epochs = mne.read_epochs(epochs_path, preload=False, verbose="ERROR")

    metadata = epochs.metadata.copy() if epochs.metadata is not None else None
    if metadata is None or "label" not in metadata.columns:
        metadata = load_events_table(subject_id, cfg.dataset)

    if cfg.filters.qc_col is not None and cfg.filters.qc_col not in metadata.columns:
        trial_qc = load_trial_qc_table(subject_id, cfg.dataset)
        if trial_qc is not None and cfg.filters.qc_col in trial_qc.columns:
            metadata = metadata.copy()
            metadata[cfg.filters.qc_col] = trial_qc[cfg.filters.qc_col].to_numpy()

    if len(metadata) != len(epochs):
        raise ValueError(
            "Metadata rows did not match the number of epochs for subject "
            f"{subject_id}. Found {len(metadata)} metadata rows and {len(epochs)} epochs."
        )

    return metadata


def apply_exclusion_rule(column: pd.Series, rule: tuple | str) -> np.ndarray:
    """Apply one metadata exclusion rule and return a keep mask."""

    keep_mask = np.ones(len(column), dtype=bool)

    if rule == "notna":
        excluded_values = ()
        exclude_non_missing = True
    else:
        excluded_values = tuple(rule)
        exclude_non_missing = False

    if len(excluded_values) > 0:
        keep_mask &= ~column.isin(excluded_values).to_numpy()

    if exclude_non_missing:
        keep_mask &= column.isna().to_numpy()

    return keep_mask


def apply_trial_filters(metadata: pd.DataFrame, cfg) -> np.ndarray:
    """Build a keep mask for one metadata table using configured rules.

    Parameters
    ----------
    metadata : pd.DataFrame
        Trial metadata table.
    cfg : object
        Config-like object with ``conditions`` and ``filters`` fields.

    Returns
    -------
    np.ndarray
        Boolean mask of rows kept for modeling.
    """

    keep_mask = np.ones(len(metadata), dtype=bool)

    if cfg.conditions.cond_col not in metadata.columns:
        raise ValueError(
            f"Could not find condition column '{cfg.conditions.cond_col}'."
        )

    test_cond = cfg.conditions.test_cond
    if isinstance(test_cond, dict):
        test_group_values = []
        for group_values in test_cond.values():
            for value in group_values:
                if value not in test_group_values:
                    test_group_values.append(value)
    else:
        test_group_values = list(test_cond)

    if len(test_group_values) > 0:
        keep_mask &= metadata[cfg.conditions.cond_col].isin(test_group_values).to_numpy()

    if cfg.filters.qc_col is not None and cfg.filters.qc_col in metadata.columns:
        keep_mask &= metadata[cfg.filters.qc_col].isin(cfg.filters.keep_qc).to_numpy()

    exclude_metadata = {} if cfg.filters.exclude_metadata is None else cfg.filters.exclude_metadata
    for column, rule in exclude_metadata.items():
        if column in metadata.columns:
            keep_mask &= apply_exclusion_rule(metadata[column], rule)

    return keep_mask


def load_subject_data_with_filters(
    subject_id: str,
    cfg,
    *,
    return_metadata: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load one subject and apply shared trial filters.

    Parameters
    ----------
    subject_id : str
        Subject identifier without the ``sub-`` prefix.
    cfg : object
        Config-like object with dataset/filters/decode/conditions fields and a
        ``label_for_metadata_row`` method.
    return_metadata : bool, optional
        Whether to include the filtered metadata in the returned tuple.

    Returns
    -------
    tuple
        ``(data, labels, times, ch_names)`` or
        ``(data, labels, times, ch_names, filtered_metadata)``.
    """

    epochs_path = epochs_path_for_subject(subject_id, cfg.dataset)
    epochs = mne.read_epochs(epochs_path, preload=True, verbose="ERROR")

    epoch_cfg = cfg.decode if hasattr(cfg, "decode") else cfg.epoch
    chans_to_drop = channels_to_drop_by_rule(epochs, epoch_cfg)
    if len(chans_to_drop) > 0:
        epochs.drop_channels(chans_to_drop)

    if epoch_cfg.crop_time is not None:
        epochs.crop(tmin=epoch_cfg.crop_time[0], tmax=epoch_cfg.crop_time[1], include_tmax=True)

    metadata = load_subject_metadata_table(subject_id, cfg)

    if "label" not in metadata.columns:
        raise ValueError(
            f"Could not find required metadata column 'label' for subject {subject_id}."
        )

    keep_mask = apply_trial_filters(metadata, cfg)
    if not np.any(keep_mask):
        raise ValueError(f"No usable trials remain for subject {subject_id} after filtering.")

    filtered_epochs = epochs[np.flatnonzero(keep_mask)]
    filtered_metadata = metadata.loc[keep_mask].reset_index(drop=True)

    data = filtered_epochs.get_data(copy=True)
    if hasattr(cfg, "label_for_metadata_row"):
        labels = cfg.label_for_metadata_row(filtered_metadata)
        train_label_order = cfg.train_label_order()
    else:
        source_values = filtered_metadata[cfg.conditions.cond_col].to_numpy(dtype=object)
        if isinstance(cfg.conditions.train_cond, dict):
            labels = np.empty(len(filtered_metadata), dtype=object)
            labels[:] = ""
            for label, group_values in cfg.conditions.train_cond.items():
                row_mask = np.isin(source_values, group_values)
                labels[row_mask] = label
            train_label_order = list(cfg.conditions.train_cond)
        else:
            labels = source_values
            train_label_order = list(cfg.conditions.train_cond)

    missing_labels = sorted(
        set(label for label in np.unique(labels) if label != "") - set(train_label_order)
    )
    if len(missing_labels) > 0:
        raise ValueError(
            f"Found training labels not listed in cfg.conditions.train_cond for subject {subject_id}: {missing_labels}"
        )

    times = filtered_epochs.times.copy()
    ch_names = filtered_epochs.ch_names.copy()
    if return_metadata:
        return data, labels, times, ch_names, filtered_metadata
    return data, labels, times, ch_names


def load_subject_info_with_channel_drop(subject_id: str, cfg) -> mne.Info:
    """Load epochs info after applying channel-drop rules.

    Parameters
    ----------
    subject_id : str
        Subject identifier without the ``sub-`` prefix.
    cfg : object
        Config-like object with dataset and decode fields.

    Returns
    -------
    mne.Info
        Channel info aligned with downstream analysis data.
    """

    epochs = mne.read_epochs(epochs_path_for_subject(subject_id, cfg.dataset), preload=True, verbose="ERROR")
    chans_to_drop = channels_to_drop_by_rule(epochs, cfg.decode)
    if len(chans_to_drop) > 0:
        epochs.drop_channels(chans_to_drop)
    return epochs.info.copy()
