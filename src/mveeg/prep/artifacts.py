"""Canonical artifact sidecars for prepared EEG epochs.

Automatic labels and manual decisions share one keyed table, but remain
separate fields so relabeling cannot silently overwrite reviewed decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re

import numpy as np
import pandas as pd


ARTIFACT_STATUSES = ("accepted", "review", "rejected")
KEY_COLUMNS = ("subject_index", "epoch_index")
ARTIFACT_COLUMNS = (
    *KEY_COLUMNS,
    "initial_status",
    "final_status",
    "epoch_reasons",
    "reviewed",
)
ARTIFACT_SUMMARY_COLUMNS = (
    *KEY_COLUMNS,
    "initial_status",
    "final_status",
    "epoch_reasons",
)

_REASON_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _reason_codes(value: object) -> list[str]:
    """Return validated snake-case reason codes from one table cell."""
    if value is None or value is pd.NA:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        values = value.split(";")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
    else:
        values = [value]

    codes = []
    for raw_code in values:
        if raw_code is None or raw_code is pd.NA:
            continue
        code = str(raw_code).strip()
        if not code:
            continue
        if _REASON_PATTERN.fullmatch(code) is None:
            raise ValueError(
                f'Artifact reason "{code}" must be a stable snake_case code.'
            )
        codes.append(code)
    return codes


def _join_reasons(*values: object) -> object:
    """Join reason values deterministically, returning missing for no reason."""
    codes = sorted({code for value in values for code in _reason_codes(value)})
    return ";".join(codes) if codes else pd.NA


def _empty_reason_matrix(n_epochs: int, n_channels: int) -> np.ndarray:
    """Create an empty object matrix for normalized artifact reasons."""
    matrix = np.empty((n_epochs, n_channels), dtype=object)
    matrix[:] = pd.NA
    return matrix


def _coerce_reason_matrix(
    reasons: pd.DataFrame | np.ndarray | Mapping[str, np.ndarray] | None,
    *,
    channel_names: Sequence[str],
    n_epochs: int,
    name: str,
) -> np.ndarray:
    """Normalize a reason matrix or reason-to-mask mapping."""
    n_channels = len(channel_names)
    if reasons is None:
        return _empty_reason_matrix(n_epochs, n_channels)

    if isinstance(reasons, Mapping):
        normalized = _empty_reason_matrix(n_epochs, n_channels)
        for code, raw_mask in reasons.items():
            validated_code = _reason_codes(code)
            if len(validated_code) != 1:
                raise ValueError(f"{name} mapping keys must contain one reason code.")
            mask = np.asarray(raw_mask)
            if mask.shape != (n_epochs, n_channels):
                raise ValueError(
                    f"{name}[{code!r}] has shape {mask.shape}; expected "
                    f"({n_epochs}, {n_channels})."
                )
            if mask.dtype.kind != "b":
                raise TypeError(f"{name}[{code!r}] must be a boolean mask.")
            for row, column in zip(*np.where(mask)):
                normalized[row, column] = _join_reasons(
                    normalized[row, column], validated_code[0]
                )
        return normalized

    if isinstance(reasons, pd.DataFrame):
        missing = [channel for channel in channel_names if channel not in reasons.columns]
        extra = [channel for channel in reasons.columns if channel not in channel_names]
        if missing or extra:
            raise ValueError(
                f"{name} columns must match channel_names; missing={missing}, extra={extra}."
            )
        raw_matrix = reasons.loc[:, list(channel_names)].to_numpy(dtype=object)
    else:
        raw_matrix = np.asarray(reasons, dtype=object)

    if raw_matrix.shape != (n_epochs, n_channels):
        raise ValueError(
            f"{name} has shape {raw_matrix.shape}; expected ({n_epochs}, {n_channels})."
        )
    return np.array(
        [[_join_reasons(value) for value in row] for row in raw_matrix],
        dtype=object,
    )


def _coerce_epoch_reasons(
    reasons: Sequence[object] | Mapping[str, np.ndarray] | None,
    *,
    n_epochs: int,
) -> np.ndarray:
    """Normalize explicit epoch-level reason codes."""
    if reasons is None:
        result = np.empty(n_epochs, dtype=object)
        result[:] = pd.NA
        return result
    if isinstance(reasons, Mapping):
        result = np.empty(n_epochs, dtype=object)
        result[:] = pd.NA
        for code, raw_mask in reasons.items():
            validated_code = _reason_codes(code)
            mask = np.asarray(raw_mask)
            if len(validated_code) != 1 or mask.shape != (n_epochs,) or mask.dtype.kind != "b":
                raise ValueError(
                    f"epoch_reasons[{code!r}] must be one snake_case code "
                    f"mapped to a boolean vector of length {n_epochs}."
                )
            for row in np.flatnonzero(mask):
                result[row] = _join_reasons(result[row], validated_code[0])
        return result
    if len(reasons) != n_epochs:
        raise ValueError(f"epoch_reasons has length {len(reasons)}; expected {n_epochs}.")
    return np.array([_join_reasons(value) for value in reasons], dtype=object)


def _coerce_epoch_mask(
    mask: Sequence[bool] | np.ndarray | None,
    *,
    n_epochs: int,
    name: str,
) -> np.ndarray:
    """Normalize an optional explicit epoch-level status mask."""
    if mask is None:
        return np.zeros(n_epochs, dtype=bool)
    array = np.asarray(mask)
    if array.shape != (n_epochs,) or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean vector with length {n_epochs}.")
    return array.astype(bool, copy=True)


def _coerce_reviewed(series: pd.Series) -> pd.Series:
    """Normalize a reviewed column without accepting arbitrary truthy values."""
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
    }
    normalized = series.map(mapping)
    if normalized.isna().any():
        raise ValueError("reviewed must contain only true or false values.")
    return normalized.astype(bool)


def artifact_channel_columns(table: pd.DataFrame) -> list[str]:
    """Return channel-reason columns in their stored order."""
    return [str(column) for column in table.columns if str(column).startswith("channel_")]


def validate_artifact_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a canonical copy of one subject's artifact table.

    Parameters
    ----------
    table : pandas.DataFrame
        Artifact sidecar with one row per epoch.

    Returns
    -------
    pandas.DataFrame
        Canonical copy ordered as fixed fields followed by channel fields.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError("Artifact data must be a pandas DataFrame.")
    if table.empty:
        raise ValueError("Artifact data must contain at least one epoch.")
    if table.columns.duplicated().any():
        raise ValueError("Artifact columns must be unique.")

    missing = [column for column in ARTIFACT_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"Artifact data are missing required columns: {missing}.")
    extra = [
        column
        for column in table.columns
        if column not in ARTIFACT_COLUMNS and not str(column).startswith("channel_")
    ]
    if extra:
        raise ValueError(f"Artifact data contain unsupported columns: {extra}.")

    canonical = table.copy()
    if canonical["subject_index"].isna().any():
        raise ValueError("subject_index cannot be missing.")
    canonical["subject_index"] = canonical["subject_index"].astype(str)
    if canonical["subject_index"].str.strip().eq("").any():
        raise ValueError("subject_index cannot be empty.")
    if canonical["subject_index"].nunique() != 1:
        raise ValueError("Each artifact sidecar must contain exactly one subject.")

    numeric_epoch = pd.to_numeric(canonical["epoch_index"], errors="coerce")
    if numeric_epoch.isna().any() or (~np.isfinite(numeric_epoch)).any():
        raise ValueError("epoch_index must contain finite integers.")
    if (numeric_epoch < 0).any() or not np.equal(numeric_epoch, np.floor(numeric_epoch)).all():
        raise ValueError("epoch_index must contain non-negative integers.")
    canonical["epoch_index"] = numeric_epoch.astype(int)
    if canonical.loc[:, list(KEY_COLUMNS)].duplicated().any():
        raise ValueError("subject_index + epoch_index must uniquely identify each epoch.")

    for column in ("initial_status", "final_status"):
        invalid = ~canonical[column].isin(ARTIFACT_STATUSES)
        if invalid.any():
            values = sorted(canonical.loc[invalid, column].astype(str).unique())
            raise ValueError(f"{column} contains invalid statuses: {values}.")
        canonical[column] = canonical[column].astype(str)

    canonical["reviewed"] = _coerce_reviewed(canonical["reviewed"])
    changed_without_review = (
        ~canonical["reviewed"]
        & canonical["final_status"].ne(canonical["initial_status"])
    )
    if changed_without_review.any():
        raise ValueError("Unreviewed epochs must have final_status == initial_status.")

    reason_columns = ["epoch_reasons", *artifact_channel_columns(canonical)]
    for column in reason_columns:
        canonical[column] = pd.Series(
            [_join_reasons(value) for value in canonical[column]],
            index=canonical.index,
            dtype="string",
        )
    ordered = [*ARTIFACT_COLUMNS, *artifact_channel_columns(canonical)]
    return canonical.loc[:, ordered].reset_index(drop=True)


def build_artifact_table(
    subject_index: str | int,
    epoch_index: Sequence[int],
    channel_names: Sequence[str],
    *,
    rejected_reasons: pd.DataFrame | np.ndarray | Mapping[str, np.ndarray] | None = None,
    review_reasons: pd.DataFrame | np.ndarray | Mapping[str, np.ndarray] | None = None,
    epoch_rejected: Sequence[bool] | np.ndarray | None = None,
    epoch_review: Sequence[bool] | np.ndarray | None = None,
    epoch_reasons: Sequence[object] | Mapping[str, np.ndarray] | None = None,
    ignore_channels: Sequence[str] = (),
    previous: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build or relabel one subject's canonical artifact sidecar.

    Ignored channels remain labeled in channel columns but do not contribute
    to trial status or ``epoch_reasons``. When supplied, each explicit epoch
    mask is authoritative for that severity; otherwise the status is derived
    from any nonignored channel. Rejection takes precedence over review.

    Parameters
    ----------
    subject_index : str or int
        Stable subject identifier.
    epoch_index : sequence of int
        Stable epoch identifiers in output order.
    channel_names : sequence of str
        Channel names corresponding to reason-matrix columns.
    rejected_reasons, review_reasons : DataFrame, ndarray, mapping, or None
        Reason cells, or reason-code mappings to boolean epoch-by-channel masks.
    epoch_rejected, epoch_review : sequence of bool or None
        Authoritative trial decisions after configured channel-count and other
        scientific rules have been aggregated.
    epoch_reasons : sequence, mapping, or None
        Additional trial-level reason codes.
    ignore_channels : sequence of str
        Channels excluded only from trial aggregation.
    previous : pandas.DataFrame or None
        Prior sidecar for the exact same keys. Reviewed final decisions are
        preserved; unreviewed decisions follow the new automatic status.

    Returns
    -------
    pandas.DataFrame
        Canonical artifact sidecar.
    """
    epoch_index = list(epoch_index)
    channel_names = [str(channel) for channel in channel_names]
    n_epochs = len(epoch_index)
    if subject_index is None or subject_index is pd.NA:
        raise ValueError("subject_index cannot be missing.")
    if isinstance(subject_index, float) and np.isnan(subject_index):
        raise ValueError("subject_index cannot be missing.")
    if not str(subject_index).strip():
        raise ValueError("subject_index cannot be empty.")
    if n_epochs == 0:
        raise ValueError("At least one epoch is required.")
    if len(set(channel_names)) != len(channel_names):
        raise ValueError("channel_names must be unique.")
    if any(not channel.strip() for channel in channel_names):
        raise ValueError("channel_names cannot contain empty names.")

    ignored = {str(channel) for channel in ignore_channels}
    unknown_ignored = sorted(ignored.difference(channel_names))
    if unknown_ignored:
        raise ValueError(f"ignore_channels contains unknown channels: {unknown_ignored}.")

    rejected_matrix = _coerce_reason_matrix(
        rejected_reasons,
        channel_names=channel_names,
        n_epochs=n_epochs,
        name="rejected_reasons",
    )
    review_matrix = _coerce_reason_matrix(
        review_reasons,
        channel_names=channel_names,
        n_epochs=n_epochs,
        name="review_reasons",
    )
    combined_matrix = np.array(
        [
            [
                _join_reasons(rejected_matrix[row, column], review_matrix[row, column])
                for column in range(len(channel_names))
            ]
            for row in range(n_epochs)
        ],
        dtype=object,
    )

    included_columns = np.array(
        [channel not in ignored for channel in channel_names], dtype=bool
    )
    if included_columns.any():
        channel_rejected = pd.notna(rejected_matrix[:, included_columns]).any(axis=1)
        channel_review = pd.notna(review_matrix[:, included_columns]).any(axis=1)
    else:
        channel_rejected = np.zeros(n_epochs, dtype=bool)
        channel_review = np.zeros(n_epochs, dtype=bool)

    rejected = (
        channel_rejected
        if epoch_rejected is None
        else _coerce_epoch_mask(
            epoch_rejected, n_epochs=n_epochs, name="epoch_rejected"
        )
    )
    review = (
        channel_review
        if epoch_review is None
        else _coerce_epoch_mask(epoch_review, n_epochs=n_epochs, name="epoch_review")
    )
    initial_status = np.full(n_epochs, "accepted", dtype=object)
    initial_status[review] = "review"
    initial_status[rejected] = "rejected"

    explicit_reasons = _coerce_epoch_reasons(epoch_reasons, n_epochs=n_epochs)
    aggregated_reasons = []
    for row in range(n_epochs):
        included_reasons = [
            combined_matrix[row, column]
            for column in np.flatnonzero(included_columns)
        ]
        aggregated_reasons.append(_join_reasons(explicit_reasons[row], *included_reasons))

    table = pd.DataFrame(
        {
            "subject_index": str(subject_index),
            "epoch_index": epoch_index,
            "initial_status": initial_status,
            "final_status": initial_status.copy(),
            "epoch_reasons": aggregated_reasons,
            "reviewed": False,
        }
    )
    for column, channel in enumerate(channel_names):
        table[f"channel_{channel}"] = combined_matrix[:, column]
    table = validate_artifact_table(table)
    if previous is None:
        return table

    previous = validate_artifact_table(previous)
    new_keys = pd.MultiIndex.from_frame(table.loc[:, list(KEY_COLUMNS)])
    previous_keys = pd.MultiIndex.from_frame(previous.loc[:, list(KEY_COLUMNS)])
    missing = new_keys.difference(previous_keys).tolist()
    extra = previous_keys.difference(new_keys).tolist()
    if missing or extra:
        raise ValueError(
            "Previous artifact keys must match exactly; "
            f"missing={missing}, extra={extra}."
        )
    previous_by_key = previous.set_index(list(KEY_COLUMNS)).reindex(new_keys)
    reviewed = previous_by_key["reviewed"].to_numpy(dtype=bool)
    table["reviewed"] = reviewed
    table.loc[reviewed, "final_status"] = previous_by_key.loc[
        reviewed, "final_status"
    ].to_numpy()
    return validate_artifact_table(table)


def project_artifact_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Return keyed epoch statuses and reasons for downstream metadata joins."""
    canonical = validate_artifact_table(table)
    return canonical.loc[:, list(ARTIFACT_SUMMARY_COLUMNS)].copy()


def read_artifact_table(path: str | Path) -> pd.DataFrame:
    """Read and validate one artifact TSV sidecar."""
    table = pd.read_csv(
        path,
        sep="\t",
        dtype={"subject_index": "string"},
        keep_default_na=False,
    )
    return validate_artifact_table(table)


def write_artifact_table(table: pd.DataFrame, path: str | Path) -> Path:
    """Validate and atomically write one artifact TSV sidecar."""
    canonical = validate_artifact_table(table)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    canonical.to_csv(temporary_path, sep="\t", index=False, na_rep="")
    temporary_path.replace(path)
    return path
