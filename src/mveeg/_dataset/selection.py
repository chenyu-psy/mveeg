"""Dataset-level subject exclusions, independent of trial quality and models."""

from __future__ import annotations

import csv
import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .manifest import normalize_subject_index

SELECTION_COLUMNS = ["subject_index", "reason"]


def validate_subject_selection(
    selection: pd.DataFrame,
    subjects: Sequence[str],
    *,
    drop_unknown: bool = False,
) -> pd.DataFrame:
    """Validate exclusions and normalize identity before reading or writing.

    Unknown subjects are an error when writing, where they usually mean a typo.
    When reading (``drop_unknown=True``) they are dropped with a warning, because
    excluding a subject that is no longer in the dataset is harmless.
    """

    selection["subject_index"] = selection["subject_index"].map(normalize_subject_index)
    if selection["subject_index"].duplicated().any():
        raise ValueError("Subject selection contains duplicate normalized subject_index values.")
    unknown = sorted(set(selection["subject_index"]).difference(subjects))
    if unknown and drop_unknown:
        warnings.warn(
            f"subject_selection.tsv excludes subjects not in the dataset; ignoring {unknown}.",
            stacklevel=3,
        )
        selection = selection.loc[~selection["subject_index"].isin(unknown)].reset_index(drop=True)
    elif unknown:
        raise ValueError(f"Subject selection contains unknown subjects: {unknown}.")
    if not selection["reason"].map(lambda value: isinstance(value, str)).all():
        raise TypeError("Subject exclusion reasons must be strings.")
    return selection


def read_subject_selection(root: Path, subjects: Sequence[str]) -> pd.DataFrame:
    """Read the complete exclusion list; an absent or empty file excludes nobody."""

    path = root / "subject_selection.tsv"
    if not path.exists():
        return pd.DataFrame(columns=SELECTION_COLUMNS)
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = [row for row in csv.reader(stream, delimiter="\t", strict=True) if row]
    except csv.Error as error:
        raise ValueError(f"Invalid subject_selection.tsv: {error}") from error
    if not rows:
        return pd.DataFrame(columns=SELECTION_COLUMNS)
    if rows[0] != SELECTION_COLUMNS or any(len(row) != 2 for row in rows[1:]):
        raise ValueError(
            "subject_selection.tsv must contain only subject_index and reason columns."
        )
    return validate_subject_selection(
        pd.DataFrame(rows[1:], columns=rows[0]), subjects, drop_unknown=True
    )
