"""Model-fitting helpers for encoding analyses."""

from __future__ import annotations

import numpy as np


def fit_time_resolved_multivariate_ols(
    data: np.ndarray,
    design_matrix: np.ndarray,
    design_names: list[str],
) -> dict[str, np.ndarray | list[str]]:
    """Fit a time-resolved multivariate OLS model.

    Parameters
    ----------
    data : np.ndarray
        EEG data with shape ``(n_trials, n_channels, n_times)``.
    design_matrix : np.ndarray
        Trial-level design matrix with shape ``(n_trials, n_predictors)``.
    design_names : list[str]
        Names for design-matrix columns.

    Returns
    -------
    dict[str, np.ndarray | list[str]]
        Dictionary with beta maps shaped ``(n_predictors, n_channels, n_times)``
        and aligned predictor names.

    Notes
    -----
    This function is intentionally simple for transparency: each time point is
    fit with a separate least-squares solve.
    """

    if data.ndim != 3:
        raise ValueError("data must have shape (n_trials, n_channels, n_times).")
    if design_matrix.ndim != 2:
        raise ValueError("design_matrix must be a 2D matrix.")
    if data.shape[0] != design_matrix.shape[0]:
        raise ValueError("Number of trials in data and design_matrix must match.")
    if design_matrix.shape[1] != len(design_names):
        raise ValueError("design_names length must match design_matrix columns.")

    n_trials, n_channels, n_times = data.shape
    n_predictors = design_matrix.shape[1]

    betas = np.empty((n_predictors, n_channels, n_times), dtype=float)
    for time_ix in range(n_times):
        y_t = data[:, :, time_ix]
        beta_t, *_ = np.linalg.lstsq(design_matrix, y_t, rcond=None)
        betas[:, :, time_ix] = beta_t

    return {
        "predictor_names": list(design_names),
        "betas": betas,
    }
