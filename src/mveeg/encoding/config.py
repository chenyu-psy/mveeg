"""Configuration objects for the encoding workflow."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EncodingConfig:
    """Configuration for building and validating encoding designs.

    Parameters
    ----------
    add_intercept : bool
        Whether to prepend an intercept column to the trial-level design.
    validation_mode : str
        Validation strictness mode. One of ``"estimable"``,
        ``"estimable_independent"``, or ``"strict"``.
    tolerance : float
        Numerical tolerance used for rank and linear-dependency checks.
    """

    add_intercept: bool = True
    validation_mode: str = "estimable_independent"
    tolerance: float = 1e-10
