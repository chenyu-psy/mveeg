"""Trial-metadata helpers for encoding regression workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MetadataAssignment:
    """Ordered pandas-style metadata assignment."""

    assignments: dict[str, object]

    def __call__(self, metadata: pd.DataFrame) -> pd.DataFrame:
        output = metadata.copy()
        for name, value in self.assignments.items():
            output[name] = value(output) if callable(value) else value
        return output


def assign_metadata(**assignments: object) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return an ordered metadata assignment callable.

    Each keyword is assigned in order, matching ``DataFrame.assign`` behavior
    but allowing later columns to reference earlier columns.
    """

    return MetadataAssignment(dict(assignments))
