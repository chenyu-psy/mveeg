"""Numerical metrics for EEG encoding analyses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass
class CovarianceEstimate:
    """Channel covariance and precision estimated from training residuals.

    Parameters
    ----------
    covariance : np.ndarray
        Channel-by-channel residual covariance matrix.
    precision : np.ndarray
        Channel-by-channel precision matrix used for whitening metrics.
    method : str
        Covariance method used to estimate the matrices.
    n_train_trials : int
        Number of training residual rows used for estimation.
    n_channels : int
        Number of EEG channels.
    rank : int
        Matrix rank of the covariance estimate.
    condition_number : float
        Numerical condition number of the covariance estimate.
    log_determinant : float
        Stable log determinant of the covariance estimate.
    shrinkage_value : float
        Ledoit-Wolf shrinkage value for ``"shrinkage"`` mode, otherwise
        ``NaN``.
    status : str
        Human-readable status string for diagnostics.
    """

    covariance: np.ndarray
    precision: np.ndarray
    method: str
    n_train_trials: int
    n_channels: int
    rank: int
    condition_number: float
    log_determinant: float
    shrinkage_value: float
    status: str


def estimate_channel_covariance(
    residuals: np.ndarray,
    *,
    method: str = "shrinkage",
    variance_floor: float = 1e-12,
) -> CovarianceEstimate:
    """Estimate channel covariance and precision from training residuals.

    Parameters
    ----------
    residuals : np.ndarray
        Training residuals with shape ``(n_trials, n_channels)``.
    method : str
        One of ``"identity"``, ``"diagonal"``, ``"sample"``, or
        ``"shrinkage"``.
    variance_floor : float
        Minimum diagonal variance used by diagonal covariance mode.

    Returns
    -------
    CovarianceEstimate
        Covariance, precision, and numerical diagnostics.

    Raises
    ------
    ValueError
        If inputs are invalid, the covariance mode is unknown, or the selected
        covariance estimate is numerically undefined.
    """

    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2:
        raise ValueError("residuals must have shape (n_trials, n_channels).")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("residuals must contain only finite values.")

    n_trials, n_channels = residuals.shape
    if n_trials < 2:
        raise ValueError("At least two training residuals are needed.")
    if n_channels < 1:
        raise ValueError("residuals must include at least one channel.")

    method = str(method).lower()
    if method == "identity":
        covariance = np.eye(n_channels, dtype=float)
        precision = np.eye(n_channels, dtype=float)
        shrinkage_value = np.nan
    elif method == "diagonal":
        variances = np.var(residuals, axis=0, ddof=1)
        variances = np.where(variances > variance_floor, variances, variance_floor)
        covariance = np.diag(variances)
        precision = np.diag(1.0 / variances)
        shrinkage_value = np.nan
    elif method == "sample":
        covariance = np.cov(residuals, rowvar=False, ddof=1)
        covariance = np.atleast_2d(covariance).astype(float)
        rank = int(np.linalg.matrix_rank(covariance))
        if rank < n_channels:
            raise ValueError(
                "sample covariance is rank deficient; use shrinkage, diagonal, "
                "or identity covariance instead."
            )
        precision = np.linalg.solve(covariance, np.eye(n_channels, dtype=float))
        shrinkage_value = np.nan
    elif method == "shrinkage":
        estimator = LedoitWolf().fit(residuals)
        covariance = estimator.covariance_.astype(float)
        precision = estimator.precision_.astype(float)
        shrinkage_value = float(estimator.shrinkage_)
    else:
        raise ValueError(
            "covariance must be one of {'identity', 'diagonal', 'sample', 'shrinkage'}."
        )

    if covariance.shape != (n_channels, n_channels):
        raise ValueError("Estimated covariance has the wrong shape.")
    if precision.shape != (n_channels, n_channels):
        raise ValueError("Estimated precision has the wrong shape.")
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(precision)):
        raise ValueError("Estimated covariance and precision must be finite.")

    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0 or not np.isfinite(logdet):
        raise ValueError("Estimated covariance must have a finite positive determinant.")

    rank = int(np.linalg.matrix_rank(covariance))
    condition_number = float(np.linalg.cond(covariance))
    status = "ok"
    if condition_number > 1e8:
        status = "high_condition_number"

    return CovarianceEstimate(
        covariance=covariance,
        precision=precision,
        method=method,
        n_train_trials=int(n_trials),
        n_channels=int(n_channels),
        rank=rank,
        condition_number=condition_number,
        log_determinant=float(logdet),
        shrinkage_value=shrinkage_value,
        status=status,
    )


def compute_pattern_expression(
    *,
    test_data: np.ndarray,
    beta_patterns: np.ndarray,
    precision_matrices: np.ndarray,
    denominator_tol: float = 1e-12,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Compute covariance-whitened signed pattern expression.

    Parameters
    ----------
    test_data : np.ndarray
        Held-out EEG data with shape ``(n_trials, n_channels, n_times)``.
    beta_patterns : np.ndarray
        Predictor beta patterns with shape ``(n_effects, n_channels, n_times)``.
    precision_matrices : np.ndarray
        Precision matrices with shape ``(n_times, n_channels, n_channels)``.
    denominator_tol : float
        Denominator values at or below this tolerance produce ``NaN``.

    Returns
    -------
    tuple[np.ndarray, list[dict[str, object]]]
        Expression array with shape ``(n_trials, n_effects, n_times)`` and
        warning rows for too-small denominators.
    """

    test_data = np.asarray(test_data, dtype=float)
    beta_patterns = np.asarray(beta_patterns, dtype=float)
    precision_matrices = np.asarray(precision_matrices, dtype=float)
    if test_data.ndim != 3:
        raise ValueError("test_data must have shape (n_trials, n_channels, n_times).")
    if beta_patterns.ndim != 3:
        raise ValueError("beta_patterns must have shape (n_effects, n_channels, n_times).")

    n_trials, n_channels, n_times = test_data.shape
    n_effects = beta_patterns.shape[0]
    if beta_patterns.shape[1:] != (n_channels, n_times):
        raise ValueError("beta_patterns must match test_data channel/time dimensions.")
    if precision_matrices.shape != (n_times, n_channels, n_channels):
        raise ValueError("precision_matrices must have shape (n_times, n_channels, n_channels).")

    expression = np.full((n_trials, n_effects, n_times), np.nan, dtype=float)
    warnings = []
    for time_ix in range(n_times):
        precision = precision_matrices[time_ix, :, :]
        y_t = test_data[:, :, time_ix]
        for effect_ix in range(n_effects):
            beta = beta_patterns[effect_ix, :, time_ix]
            denom_sq = float(beta.T @ precision @ beta)
            if denom_sq <= denominator_tol or not np.isfinite(denom_sq):
                warnings.append(
                    {
                        "effect_index": int(effect_ix),
                        "time_index": int(time_ix),
                        "status": "small_expression_denominator",
                    }
                )
                continue
            expression[:, effect_ix, time_ix] = (y_t @ precision @ beta) / np.sqrt(
                denom_sq
            )

    return expression, warnings


def compute_prediction_metrics(
    *,
    test_data: np.ndarray,
    prediction: np.ndarray,
    train_mean: np.ndarray,
    precision: np.ndarray,
    log_determinant: float,
) -> dict[str, float]:
    """Compute held-out multichannel prediction metrics for one time bin.

    Parameters
    ----------
    test_data : np.ndarray
        Held-out EEG data with shape ``(n_trials, n_channels)``.
    prediction : np.ndarray
        Model prediction with shape ``(n_trials, n_channels)``.
    train_mean : np.ndarray
        Training-set mean EEG vector with shape ``(n_channels,)`` used as the
        baseline prediction.
    precision : np.ndarray
        Channel precision matrix.
    log_determinant : float
        Log determinant of ``covariance``.

    Returns
    -------
    dict[str, float]
        Ordinary and whitened held-out prediction metrics.
    """

    test_data = np.asarray(test_data, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    train_mean = np.asarray(train_mean, dtype=float)
    if test_data.shape != prediction.shape:
        raise ValueError("test_data and prediction must have the same shape.")
    if train_mean.shape != (test_data.shape[1],):
        raise ValueError("train_mean must have shape (n_channels,).")
    if not np.isfinite(log_determinant):
        raise ValueError("log_determinant must be finite.")

    error = test_data - prediction
    baseline_error = test_data - train_mean.reshape(1, -1)
    sse_model = float(np.sum(error ** 2))
    sse_baseline = float(np.sum(baseline_error ** 2))
    cv_r2 = np.nan if sse_baseline == 0 else 1.0 - sse_model / sse_baseline

    model_quad = np.einsum("ij,jk,ik->i", error, precision, error)
    baseline_quad = np.einsum("ij,jk,ik->i", baseline_error, precision, baseline_error)
    wsse_model = float(np.sum(model_quad))
    wsse_baseline = float(np.sum(baseline_quad))
    wmse = float(np.mean(model_quad))
    wcv_r2 = np.nan if wsse_baseline == 0 else 1.0 - wsse_model / wsse_baseline

    n_trials, n_channels = test_data.shape
    log2pi = np.log(2.0 * np.pi)
    loglik_total = float(
        -0.5 * (n_trials * (n_channels * log2pi + log_determinant) + wsse_model)
    )
    return {
        "cv_r2": float(cv_r2),
        "wmse": wmse,
        "wcv_r2": float(wcv_r2),
        "heldout_loglik_total": loglik_total,
        "heldout_loglik_mean": loglik_total / float(n_trials),
    }
