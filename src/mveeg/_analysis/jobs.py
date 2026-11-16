"""Parallel worker count resolution shared by model pipelines."""

from __future__ import annotations

from numbers import Integral

from joblib import effective_n_jobs


def resolve_n_jobs(n_jobs: int) -> int:
    """Return the worker count for a joblib-style ``n_jobs`` value.

    Positive values are used as given. Negative values follow joblib, so
    ``-1`` uses every CPU and ``-2`` all but one.
    """

    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer; use -1 for all CPUs.")
    return int(effective_n_jobs(int(n_jobs)))
