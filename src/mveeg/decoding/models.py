"""Model-building helpers for the EEG decoding workflow."""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import DecodingConfig


def build_classifier(cfg: DecodingConfig):
    """Return the classifier used for decoding."""

    if cfg.model.classifier is not None:
        return cfg.model.classifier

    return build_classifier_from_spec(cfg.model.classifier_spec)


def build_classifier_from_spec(classifier_spec: dict) -> object:
    """Build a classifier from a structured script-facing specification."""

    backend = classifier_spec["backend"]
    model_name = classifier_spec["model_name"]
    model_params = dict(classifier_spec["model_params"])

    if backend != "sklearn":
        raise ValueError(
            f"Unsupported classifier backend: {backend}. "
            "Add a builder for this backend before using it in an analysis script."
        )

    if model_name == "logistic_regression":
        if "solver" not in model_params:
            model_params["solver"] = "lbfgs"
        if "max_iter" not in model_params:
            model_params["max_iter"] = 1000
        return LogisticRegression(**model_params)

    if model_name == "lda":
        return LinearDiscriminantAnalysis(**model_params)

    if model_name == "svm_linear":
        if "kernel" not in model_params:
            model_params["kernel"] = "linear"
        if "probability" not in model_params:
            model_params["probability"] = False
        return SVC(**model_params)

    raise ValueError(
        f"Unsupported sklearn classifier model_name: {model_name}. "
        "Add it to build_classifier_from_spec before using it in an analysis script."
    )


def binary_pattern_sign(model, label_order: list[str]) -> float:
    """Return a sign that aligns weights with the requested label order."""

    if len(label_order) != 2 or not hasattr(model, "classes_"):
        return 1.0

    model_classes = list(model.classes_)
    if model_classes == label_order:
        return 1.0
    if model_classes == label_order[::-1]:
        return -1.0
    return 1.0


def get_binary_weights(model, scaler: StandardScaler) -> np.ndarray:
    """Extract classifier weights for binary decoding models."""

    coef = np.asarray(model.coef_, dtype=float)
    if coef.ndim == 2:
        coef = coef[0]
    return coef / scaler.scale_


def compute_haufe_pattern(
    X_train: np.ndarray,
    X_train_scaled: np.ndarray,
    model,
) -> np.ndarray:
    """Compute Haufe patterns from one fitted binary classifier."""

    weights = np.asarray(model.coef_, dtype=float)
    if weights.ndim == 2:
        weights = weights[0]

    cov_x = np.cov(X_train, rowvar=False)
    decision = X_train_scaled @ weights
    cov_s = np.var(decision, ddof=1)

    if cov_s == 0:
        return np.zeros(X_train.shape[1], dtype=float)
    return (cov_x @ weights) / cov_s
