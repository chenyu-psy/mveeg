"""Shared analysis trial-selection mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def drop_redundant_subject_alias(metadata: pd.DataFrame) -> pd.DataFrame:
    """Drop ``subject`` only when it exactly repeats canonical identity."""

    if "subject" not in metadata.columns:
        return metadata
    subject = metadata["subject"].astype("string").str.strip()
    subject_index = metadata["subject_index"].astype("string").str.strip()
    same = subject.eq(subject_index).all()
    if not same:
        subject_number = pd.to_numeric(subject, errors="coerce")
        index_number = pd.to_numeric(subject_index, errors="coerce")
        same = subject_number.notna().all() and index_number.notna().all()
        same = same and np.array_equal(subject_number, index_number)
    return metadata.drop(columns="subject") if same else metadata


def base_trial_mask(
    metadata: pd.DataFrame,
    *,
    qc: str | None,
    keep: list[object] | tuple[object, ...],
    exclude: dict[str, object],
) -> np.ndarray:
    """Apply shared QC and explicit metadata exclusion rules."""

    mask = np.ones(len(metadata), dtype=bool)
    if qc is not None:
        if qc not in metadata.columns:
            raise ValueError(f"QC column {qc!r} is missing; label artifacts or set qc=None.")
        mask &= metadata[qc].isin(tuple(keep)).to_numpy()
    for column, rule in exclude.items():
        if column not in metadata.columns:
            raise ValueError(f"Trial exclusion column {column!r} is missing.")
        mask &= apply_exclusion_rule(metadata[column], rule)
    return mask


def map_values(values: np.ndarray, mapping: dict[str, list[object]]) -> np.ndarray:
    """Map raw values to one named group while retaining unmapped rows as null."""

    labels = np.full(len(values), None, dtype=object)
    for label, raw_values in mapping.items():
        labels[np.isin(values, raw_values)] = label
    return labels


def apply_exclusion_rule(column: pd.Series, rule: tuple | str) -> np.ndarray:
    """Apply one metadata exclusion rule and return a keep mask."""

    if rule == "notna":
        return column.isna().to_numpy()
    return ~column.isin(tuple(rule)).to_numpy()
