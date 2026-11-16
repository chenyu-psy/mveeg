"""Shared handling of researcher-supplied value lists."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def is_value_list(value: object) -> bool:
    """Return whether ``value`` is an ordered, non-string collection of values.

    Lists, tuples, ranges, 1-D NumPy arrays, and pandas Series/Index/arrays
    (for example ``df["load"].unique()``) qualify. Strings, mappings, and sets
    do not; sets have no stable order for provenance.
    """

    if isinstance(value, (str, bytes, bytearray, Mapping, set, frozenset)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim == 1
    return isinstance(value, (Sequence, pd.Series, pd.Index, pd.api.extensions.ExtensionArray))


def value_list(value: object) -> list[object]:
    """Return a list of plain Python values from a value list."""

    return [item.item() if isinstance(item, np.generic) else item for item in value]
