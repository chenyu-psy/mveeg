"""Trial metadata identity, transformation, and sidecar merge contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from numbers import Number

import numpy as np
import pandas as pd

IDENTITY_COLUMNS = ("subject_index", "epoch_index")
ARTIFACT_METADATA_COLUMNS = ("initial_status", "final_status", "epoch_reasons")
_METADATA_FLOAT_ATOL = 5e-4


def assign_metadata_variables(
    metadata: pd.DataFrame,
    variables: Mapping[str, Callable[[pd.DataFrame], object]],
) -> pd.DataFrame:
    """Evaluate ordered scalar or trial-aligned metadata variables."""

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame.")
    validate_metadata_variables(variables)
    validate_identity(metadata, context="metadata")

    output = metadata.copy().reset_index(drop=True)
    identity = output.loc[:, IDENTITY_COLUMNS].copy()
    for name, function in variables.items():
        value = function(output.copy())
        if isinstance(value, pd.DataFrame):
            raise TypeError(f"Metadata variable {name!r} must return one column, not a DataFrame.")
        if not np.isscalar(value):
            try:
                value_length = len(value)
            except TypeError as error:
                raise TypeError(
                    f"Metadata variable {name!r} must return a scalar or one column."
                ) from error
            if value_length != len(output):
                raise ValueError(
                    f"Metadata variable {name!r} returned {value_length} values for "
                    f"{len(output)} trials."
                )
        output[name] = value

    if not output.loc[:, IDENTITY_COLUMNS].equals(identity):
        raise ValueError(
            "transform_metadata must preserve subject_index and epoch_index values and order."
        )
    return output


def validate_metadata_variables(
    variables: Mapping[str, Callable[[pd.DataFrame], object]],
) -> None:
    """Validate the common public ``transform_metadata`` registration contract."""

    if not isinstance(variables, Mapping):
        raise TypeError("variables must be a mapping from column names to callables.")
    if not variables:
        raise ValueError("transform_metadata requires at least one named variable.")
    for name, function in variables.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Metadata variable names must be non-empty strings.")
        if name in IDENTITY_COLUMNS:
            raise ValueError("transform_metadata cannot replace subject_index or epoch_index.")
        if not callable(function):
            raise TypeError(f"Metadata variable {name!r} must be defined by a callable.")


def normalize_identity(metadata: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Return metadata with canonical string/integer identity dtypes."""

    output = metadata.copy()
    missing = sorted(set(IDENTITY_COLUMNS).difference(output.columns))
    if missing:
        raise ValueError(f"{context} is missing identity columns: {missing}")
    if output["subject_index"].isna().any():
        raise ValueError(f"{context} subject_index cannot contain missing values.")
    output["subject_index"] = output["subject_index"].astype(str)
    if output["subject_index"].str.strip().eq("").any():
        raise ValueError(f"{context} subject_index cannot contain empty values.")
    epoch_index = pd.to_numeric(output["epoch_index"], errors="raise")
    if epoch_index.isna().any() or not np.equal(epoch_index, np.floor(epoch_index)).all():
        raise ValueError(f"{context} epoch_index must contain integers without missing values.")
    output["epoch_index"] = epoch_index.astype(np.int64)
    validate_identity(output, context=context)
    return output.reset_index(drop=True)


def validate_identity(metadata: pd.DataFrame, *, context: str) -> None:
    """Require non-missing, unique canonical trial keys."""

    missing = sorted(set(IDENTITY_COLUMNS).difference(metadata.columns))
    if missing:
        raise ValueError(f"{context} is missing identity columns: {missing}")
    if metadata.loc[:, IDENTITY_COLUMNS].isna().any(axis=None):
        raise ValueError(f"{context} identity columns cannot contain missing values.")
    if metadata.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{context} must have unique subject_index + epoch_index keys.")


def validate_metadata_mirror(
    epochs_metadata: pd.DataFrame,
    events_metadata: pd.DataFrame,
) -> None:
    """Require FIF metadata and events.tsv to contain the same base table."""

    left = normalize_identity(epochs_metadata, context="epochs metadata")
    right = normalize_identity(events_metadata, context="events.tsv")
    if len(left) != len(right):
        raise ValueError("Epochs metadata and events.tsv must contain the same number of rows.")
    if list(left.columns) != list(right.columns):
        raise ValueError(
            "Epochs metadata and events.tsv must contain the same columns in the same order."
        )
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=_METADATA_FLOAT_ATOL,
            check_categorical=False,
        )
    except AssertionError as error:
        detail = _first_mismatch(left, right)
        raise ValueError(
            "Epochs metadata and events.tsv must contain the same values in the same trial "
            f"order. {detail}"
        ) from error


def merge_artifact_metadata(metadata: pd.DataFrame, artifacts: pd.DataFrame) -> pd.DataFrame:
    """Key-merge the three analysis-facing artifact fields into metadata."""

    base = normalize_identity(metadata, context="events.tsv")
    sidecar = normalize_identity(artifacts, context="artifacts.tsv")
    missing = sorted(set(ARTIFACT_METADATA_COLUMNS).difference(sidecar.columns))
    if missing:
        raise ValueError(f"artifacts.tsv is missing required summary columns: {missing}")
    collisions = sorted(set(ARTIFACT_METADATA_COLUMNS).intersection(base.columns))
    if collisions:
        raise ValueError(f"Artifact fields are duplicated in base metadata: {collisions}")
    merged = base.merge(
        sidecar.loc[:, [*IDENTITY_COLUMNS, *ARTIFACT_METADATA_COLUMNS]],
        on=list(IDENTITY_COLUMNS),
        how="outer",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        mismatches = merged.loc[merged["_merge"].ne("both"), [*IDENTITY_COLUMNS, "_merge"]].to_dict(
            orient="records"
        )
        raise ValueError(
            "events.tsv and artifacts.tsv must contain exactly the same trial keys. "
            f"Mismatches: {mismatches[:10]}"
        )
    merged = merged.drop(columns="_merge")
    if (
        not merged.loc[:, IDENTITY_COLUMNS]
        .reset_index(drop=True)
        .equals(base.loc[:, IDENTITY_COLUMNS].reset_index(drop=True))
    ):
        raise ValueError("Artifact merge changed trial order.")
    return merged


def _first_mismatch(left: pd.DataFrame, right: pd.DataFrame) -> str:
    for column in left.columns:
        for position, (left_value, right_value) in enumerate(
            zip(left[column], right[column], strict=True)
        ):
            left_missing = pd.isna(left_value)
            right_missing = pd.isna(right_value)
            if left_missing or right_missing:
                matches = bool(left_missing and right_missing)
            elif isinstance(left_value, Number) and isinstance(right_value, Number):
                matches = bool(
                    np.isclose(left_value, right_value, rtol=0, atol=_METADATA_FLOAT_ATOL)
                )
            else:
                matches = bool(left_value == right_value)
            if not matches:
                epoch_index = int(right.iloc[position]["epoch_index"])
                return (
                    f"First mismatch at row {position} (epoch_index={epoch_index}), "
                    f"column {column!r}: FIF={left_value!r}, events.tsv={right_value!r}."
                )
    return ""
