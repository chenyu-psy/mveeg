"""Classifier construction and interpretable linear patterns."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

CLASSIFIERS = {"logistic_regression", "lda", "linear_svm"}


def classifier_settings(name: str, parameters: dict[str, object]) -> dict[str, object]:
    """Validate and resolve one built-in classifier specification."""

    if name not in CLASSIFIERS:
        raise ValueError(f"classifier must be one of {sorted(CLASSIFIERS)}.")
    resolved = dict(parameters)
    if name == "logistic_regression":
        resolved.setdefault("solver", "lbfgs")
        resolved.setdefault("max_iter", 1000)
    elif name == "linear_svm":
        if resolved.get("kernel", "linear") != "linear":
            raise ValueError("linear_svm only accepts kernel='linear'.")
        resolved["kernel"] = "linear"
        resolved.setdefault("probability", False)
    return resolved


def make_classifier(name: str, parameters: dict[str, object]):
    """Construct one fresh built-in classifier."""

    settings = classifier_settings(name, parameters)
    if name == "logistic_regression":
        return LogisticRegression(**settings)
    if name == "lda":
        return LinearDiscriminantAnalysis(**settings)
    return SVC(**settings)


def native_evidence(model, data: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return classifier-native decision values and their per-trial shape."""

    values = np.asarray(model.decision_function(data), dtype=float)
    shape = [] if values.ndim == 1 else list(values.shape[1:])
    return values, shape


def haufe_patterns(
    raw_training: np.ndarray,
    scaled_training: np.ndarray,
    model,
) -> np.ndarray:
    """Return one Haufe activation pattern per native linear filter.

    The returned array has shape ``(n_components, n_channels)``. Patterns are
    expressed in the original sensor scale even though the estimator is fitted
    in standardized feature space.
    """

    filters = np.asarray(model.coef_, dtype=float)
    if filters.ndim == 1:
        filters = filters[np.newaxis, :]
    if filters.ndim != 2 or filters.shape[1] != raw_training.shape[1]:
        raise ValueError("Classifier coefficients do not match the EEG channel axis.")
    if len(raw_training) < 2:
        raise ValueError("At least two training pseudotrials are required for Haufe patterns.")

    scores = scaled_training @ filters.T
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]
    joined = np.column_stack([raw_training, scores])
    covariance = np.cov(joined, rowvar=False, ddof=1)
    n_channels = raw_training.shape[1]
    cov_xs = covariance[:n_channels, n_channels:]
    cov_s = np.atleast_2d(covariance[n_channels:, n_channels:])
    patterns = cov_xs @ np.linalg.pinv(cov_s)

    score_variance = np.var(scores, axis=0, ddof=1)
    patterns[:, score_variance <= np.finfo(float).eps] = np.nan
    return patterns.T


def pattern_components(model, classifier: str) -> list[list[str]]:
    """Describe each native coefficient row without changing its geometry."""

    classes = [str(value) for value in model.classes_]
    rows = np.atleast_2d(np.asarray(model.coef_)).shape[0]
    if rows == 1:
        return [classes]
    if classifier == "linear_svm":
        pairs = [list(pair) for pair in combinations(classes, 2)]
        if len(pairs) != rows:
            raise ValueError("linear_svm coefficient rows do not match its class pairs.")
        return pairs
    if rows != len(classes):
        raise ValueError("Classifier coefficient rows do not match fitted classes.")
    return [[label] for label in classes]
