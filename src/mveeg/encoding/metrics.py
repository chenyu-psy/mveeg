"""Numerical estimators for the regularized component encoding model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class CovarianceEstimate:
    covariance: np.ndarray
    precision: np.ndarray
    method: str
    n_train_trials: int
    n_channels: int
    rank: int
    condition_number: float
    log_determinant: float
    shrinkage: float
    status: str


def fit_ridge(
    data: np.ndarray,
    design: np.ndarray,
    penalties: np.ndarray,
) -> np.ndarray:
    """Fit all channels and time bins under one component/condition ridge solve."""

    data = np.asarray(data, dtype=float)
    design = np.asarray(design, dtype=float)
    penalties = np.asarray(penalties, dtype=float)
    if data.ndim != 3:
        raise ValueError("data must have shape (trials, channels, times).")
    if design.ndim != 2 or design.shape[0] != data.shape[0]:
        raise ValueError("design rows must match data trials.")
    if penalties.shape != (design.shape[1],):
        raise ValueError("penalties must contain one value per design column.")
    if not np.all(np.isfinite(penalties)) or np.any(penalties < 0):
        raise ValueError("penalties must be finite non-negative values.")

    system = design.T @ design + np.diag(penalties)
    response = data.reshape(data.shape[0], -1)
    betas = np.linalg.solve(system, design.T @ response)
    return betas.reshape(design.shape[1], data.shape[1], data.shape[2])


def estimate_channel_covariance(
    residuals: np.ndarray,
    *,
    method: str = "shrinkage",
    variance_floor: float = 1e-12,
) -> CovarianceEstimate:
    """Estimate raw-scale channel covariance from one time bin's residuals."""

    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2:
        raise ValueError("residuals must have shape (trials, channels).")
    if residuals.shape[0] < 2 or residuals.shape[1] < 1:
        raise ValueError("At least two residual trials and one channel are required.")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("residuals must contain only finite values.")

    method = str(method).lower()
    n_trials, n_channels = residuals.shape
    centered = residuals - residuals.mean(axis=0, keepdims=True)
    if method == "identity":
        covariance = np.eye(n_channels)
        precision = np.eye(n_channels)
        shrinkage = np.nan
    elif method == "diagonal":
        variances = np.maximum(np.mean(centered**2, axis=0), variance_floor)
        covariance = np.diag(variances)
        precision = np.diag(1.0 / variances)
        shrinkage = np.nan
    elif method == "shrinkage":
        scales = np.sqrt(np.mean(centered**2, axis=0))
        scales = np.maximum(scales, np.sqrt(variance_floor))
        standardized = centered / scales
        estimator = LedoitWolf(assume_centered=True).fit(standardized)
        covariance = scales[:, None] * estimator.covariance_ * scales[None, :]
        inverse_scales = 1.0 / scales
        precision = (
            inverse_scales[:, None]
            * estimator.precision_
            * inverse_scales[None, :]
        )
        shrinkage = float(estimator.shrinkage_)
    else:
        raise ValueError("Internal covariance method must be shrinkage, diagonal, or identity.")

    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0 or not np.isfinite(log_determinant):
        raise ValueError("Estimated covariance must be finite and positive definite.")
    if not np.all(np.isfinite(precision)):
        raise ValueError("Estimated precision must be finite.")
    rank = int(np.linalg.matrix_rank(covariance))
    condition_number = float(np.linalg.cond(covariance))
    return CovarianceEstimate(
        covariance=np.asarray(covariance, dtype=float),
        precision=np.asarray(precision, dtype=float),
        method=method,
        n_train_trials=int(n_trials),
        n_channels=int(n_channels),
        rank=rank,
        condition_number=condition_number,
        log_determinant=float(log_determinant),
        shrinkage=shrinkage,
        status="warning" if condition_number > 1e8 else "ok",
    )


def compute_pattern_expression(
    data: np.ndarray,
    beta_patterns: np.ndarray,
    precision_matrices: np.ndarray,
    *,
    denominator_tol: float = 1e-12,
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    """Compute covariance-whitened signed component expression."""

    data = np.asarray(data, dtype=float)
    beta_patterns = np.asarray(beta_patterns, dtype=float)
    precision_matrices = np.asarray(precision_matrices, dtype=float)
    if data.ndim != 3:
        raise ValueError("data must have shape (trials, channels, times).")
    if beta_patterns.ndim != 3:
        raise ValueError("beta_patterns must have shape (components, channels, times).")
    n_trials, n_channels, n_times = data.shape
    if beta_patterns.shape[1:] != (n_channels, n_times):
        raise ValueError("beta_patterns must match data channels and times.")
    if precision_matrices.shape != (n_times, n_channels, n_channels):
        raise ValueError("precision_matrices have the wrong shape.")

    expression = np.full((n_trials, beta_patterns.shape[0], n_times), np.nan)
    warnings: list[dict[str, int | str]] = []
    for time_index in range(n_times):
        precision = precision_matrices[time_index]
        for component_index, beta in enumerate(beta_patterns[:, :, time_index]):
            denominator_squared = float(beta.T @ precision @ beta)
            if denominator_squared <= denominator_tol or not np.isfinite(
                denominator_squared
            ):
                warnings.append(
                    {
                        "component_index": component_index,
                        "time_index": time_index,
                        "status": "small_expression_denominator",
                    }
                )
                continue
            expression[:, component_index, time_index] = (
                data[:, :, time_index] @ precision @ beta
            ) / np.sqrt(denominator_squared)
    return expression, warnings
