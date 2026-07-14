"""Small shared rules for channel and trial selection."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd


def channels_to_drop_by_rule(epochs: mne.Epochs, epoch_config) -> list[str]:
    """Return channels selected by configured names or MNE channel types."""

    selected = {
        name
        for name, kind in zip(epochs.ch_names, epochs.get_channel_types())
        if kind in epoch_config.drop_channel_types
    }
    selected.update(
        name for name in epoch_config.drop_channels if name in epochs.ch_names
    )
    return sorted(selected)


def apply_exclusion_rule(column: pd.Series, rule: tuple | str) -> np.ndarray:
    """Apply one metadata exclusion rule and return a keep mask."""

    if rule == "notna":
        return column.isna().to_numpy()
    return ~column.isin(tuple(rule)).to_numpy()


def apply_trial_filters(metadata: pd.DataFrame, config) -> np.ndarray:
    """Build a keep mask from condition, artifact, and metadata rules."""

    condition_column = config.conditions.cond_col
    if condition_column not in metadata.columns:
        raise ValueError(f"Could not find condition column {condition_column!r}.")

    test_condition = config.conditions.test_cond
    if isinstance(test_condition, dict):
        values = list(
            dict.fromkeys(
                value
                for group_values in test_condition.values()
                for value in group_values
            )
        )
    else:
        values = list(test_condition)

    keep = np.ones(len(metadata), dtype=bool)
    if values:
        keep &= metadata[condition_column].isin(values).to_numpy()

    qc_column = config.filters.qc_col
    if qc_column is not None and qc_column in metadata.columns:
        keep &= metadata[qc_column].isin(config.filters.keep_qc).to_numpy()

    for column, rule in (config.filters.exclude_metadata or {}).items():
        if column in metadata.columns:
            keep &= apply_exclusion_rule(metadata[column], rule)
    return keep
