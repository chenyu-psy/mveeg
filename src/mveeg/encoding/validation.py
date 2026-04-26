"""Validation helpers for condition encoding designs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Structured validation output for an encoding design matrix.

    Parameters
    ----------
    is_valid : bool
        Whether the design passed the requested validation mode.
    mode : str
        Validation mode used for the checks.
    rank : int
        Matrix rank of the tested design matrix.
    n_cols : int
        Number of columns in the tested design matrix.
    condition_number : float
        Matrix condition number.
    aliased_columns : list[str]
        Columns found to be exact linear combinations of others.
    vif_table : pd.DataFrame
        Variance inflation factors for non-intercept columns.
    messages : list[str]
        Human-readable diagnostics explaining the result.
    """

    is_valid: bool
    mode: str
    rank: int
    n_cols: int
    condition_number: float
    aliased_columns: list[str]
    vif_table: pd.DataFrame
    messages: list[str]


def _column_r2_on_others(X: np.ndarray, col_ix: int, tol: float) -> float:
    """Return R^2 when predicting one column from the remaining columns."""

    y = X[:, col_ix]
    X_other = np.delete(X, col_ix, axis=1)

    if X_other.shape[1] == 0:
        return 0.0

    beta, *_ = np.linalg.lstsq(X_other, y, rcond=None)
    fitted = X_other @ beta
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    if ss_tot <= tol:
        return 1.0
    return 1.0 - ss_res / ss_tot


def _compute_vif_table(X: np.ndarray, col_names: list[str], tol: float) -> pd.DataFrame:
    """Compute VIF values for non-intercept columns."""

    rows = []
    for col_ix, col_name in enumerate(col_names):
        if col_name == "intercept":
            continue
        r2 = _column_r2_on_others(X, col_ix, tol=tol)
        if r2 >= 1.0 - tol:
            vif = np.inf
        else:
            vif = 1.0 / (1.0 - r2)
        rows.append({"variable": col_name, "r2": float(r2), "vif": float(vif)})
    return pd.DataFrame(rows)


def _strict_coverage_checks(condition_encoding: pd.DataFrame, variable_cols: list[str]) -> list[str]:
    """Run strict coverage checks on the condition-level encoding table."""

    messages = []
    for var in variable_cols:
        n_unique = condition_encoding[var].nunique(dropna=False)
        if n_unique < 2:
            messages.append(
                f"Variable '{var}' has fewer than two unique condition-level values."
            )
            continue

        other_cols = [col for col in variable_cols if col != var]
        if len(other_cols) == 0:
            continue

        has_context_contrast = False
        grouped = condition_encoding.groupby(other_cols, dropna=False)
        for _, group_df in grouped:
            if group_df[var].nunique(dropna=False) >= 2:
                has_context_contrast = True
                break

        if not has_context_contrast:
            messages.append(
                f"Variable '{var}' has no within-context contrast when other variables are fixed."
            )

    return messages


def validate_encoding(
    X: np.ndarray,
    var_names: list[str],
    *,
    mode: str = "estimable_independent",
    tol: float = 1e-10,
    condition_encoding: pd.DataFrame | None = None,
) -> ValidationResult:
    """Validate whether an encoding design matrix supports target inferences.

    Parameters
    ----------
    X : np.ndarray
        Trial-level design matrix.
    var_names : list[str]
        Column names aligned with ``X``.
    mode : str
        Validation strictness mode:
        ``"estimable"``, ``"estimable_independent"``, or ``"strict"``.
    tol : float
        Numerical tolerance for rank and dependency checks.
    condition_encoding : pd.DataFrame | None
        Optional condition-level encoding table used by strict coverage checks.

    Returns
    -------
    ValidationResult
        Structured diagnostics for design validity.

    Examples
    --------
    >>> X = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=float)
    >>> result = validate_encoding(X, ["intercept", "feature_a", "feature_b"])
    >>> result.is_valid
    True
    """

    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")
    if X.shape[1] != len(var_names):
        raise ValueError("len(var_names) must match the number of columns in X.")
    if mode not in {"estimable", "estimable_independent", "strict"}:
        raise ValueError(
            "mode must be one of {'estimable', 'estimable_independent', 'strict'}."
        )

    rank = int(np.linalg.matrix_rank(X, tol=tol))
    n_cols = int(X.shape[1])
    condition_number = float(np.linalg.cond(X))

    messages = []
    aliased_columns = []

    is_estimable = rank == n_cols
    if not is_estimable:
        messages.append(
            f"Design matrix is rank-deficient (rank={rank}, columns={n_cols})."
        )

    vif_table = _compute_vif_table(X, var_names, tol=tol)

    if mode in {"estimable_independent", "strict"}:
        for col_ix, col_name in enumerate(var_names):
            if col_name == "intercept":
                continue
            r2 = _column_r2_on_others(X, col_ix, tol=tol)
            if r2 >= 1.0 - tol:
                aliased_columns.append(col_name)

        if len(aliased_columns) > 0:
            messages.append(
                "These variables are exact linear combinations of others: "
                f"{sorted(aliased_columns)}"
            )

    if mode == "strict":
        if condition_encoding is None:
            messages.append(
                "Strict mode requires condition_encoding for coverage checks."
            )
        else:
            variable_cols = [col for col in condition_encoding.columns if col != "condition"]
            strict_messages = _strict_coverage_checks(condition_encoding, variable_cols)
            messages.extend(strict_messages)

    if np.any(np.isinf(vif_table.get("vif", pd.Series([], dtype=float)))):
        messages.append("At least one variable has infinite VIF.")

    if condition_number > 1e8:
        messages.append(
            f"Condition number is high ({condition_number:.2e}); estimates may be numerically unstable."
        )

    is_valid = len(messages) == 0

    if mode == "estimable":
        is_valid = is_estimable
        if is_estimable and len(messages) == 0:
            messages.append("Design is estimable.")
    elif mode == "estimable_independent":
        is_valid = is_estimable and len(aliased_columns) == 0
        if is_valid and len(messages) == 0:
            messages.append("Design is estimable and each variable has unique information.")
    else:
        strict_failures = [m for m in messages if "Strict" in m or "within-context" in m or "fewer than" in m]
        is_valid = is_estimable and len(aliased_columns) == 0 and len(strict_failures) == 0
        if is_valid and len(messages) == 0:
            messages.append("Design passed strict estimability and coverage checks.")

    return ValidationResult(
        is_valid=is_valid,
        mode=mode,
        rank=rank,
        n_cols=n_cols,
        condition_number=condition_number,
        aliased_columns=sorted(aliased_columns),
        vif_table=vif_table,
        messages=messages,
    )
