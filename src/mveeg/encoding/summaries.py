"""Summary-table builders for encoding model outputs.

This module reshapes already-computed training and testing outputs into
analysis-ready tables for downstream modeling in Python or R.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_pattern_strength(pattern: np.ndarray) -> np.ndarray:
    """Return the L2 pattern strength across channels for each time point.

    Parameters
    ----------
    pattern : np.ndarray
        Beta pattern with shape ``(n_channels, n_times)``.

    Returns
    -------
    np.ndarray
        Pattern-strength time series with shape ``(n_times,)``.
    """

    if pattern.ndim != 2:
        raise ValueError("pattern must have shape (n_channels, n_times).")

    return np.sqrt(np.sum(np.square(pattern.astype(float)), axis=0))


def build_training_pattern_strength_table(
    *,
    subject: str,
    fold_id: int,
    effect: str,
    times_s: np.ndarray | list[float],
    pattern_strength: np.ndarray,
    data_type: str,
    null_draw: int,
    n_null_repeats: int,
) -> pd.DataFrame:
    """Build a long training table for one fold/effect/data-type combination.

    Parameters
    ----------
    subject : str
        Subject identifier.
    fold_id : int
        Cross-validation fold identifier.
    effect : str
        Effect label written to the long table.
    times_s : np.ndarray | list[float]
        Time axis in seconds.
    pattern_strength : np.ndarray
        One-dimensional pattern-strength series aligned with ``times_s``.
    data_type : str
        One of ``"pattern"`` or ``"null"``.
    null_draw : int
        Null-draw index. Use ``0`` for the observed pattern rows.
    n_null_repeats : int
        Number of null draws requested for this run.

    Returns
    -------
    pd.DataFrame
        Long table with columns ``subject``, ``fold``, ``effect``, ``time_ms``,
        ``data_type``, ``pattern_strength``, ``null_draw``, and
        ``n_null_repeats``.
    """

    times_s = np.asarray(times_s, dtype=float)
    pattern_strength = np.asarray(pattern_strength, dtype=float)

    if pattern_strength.ndim != 1:
        raise ValueError("pattern_strength must be one-dimensional.")
    if len(pattern_strength) != len(times_s):
        raise ValueError("pattern_strength length must match times_s length.")
    if data_type not in {"pattern", "null"}:
        raise ValueError("data_type must be 'pattern' or 'null'.")

    return pd.DataFrame(
        {
            "subject": np.repeat(str(subject), len(times_s)),
            "fold": np.repeat(int(fold_id), len(times_s)),
            "effect": np.repeat(str(effect), len(times_s)),
            "time_ms": times_s * 1000.0,
            "data_type": np.repeat(str(data_type), len(times_s)),
            "pattern_strength": pattern_strength,
            "null_draw": np.repeat(int(null_draw), len(times_s)),
            "n_null_repeats": np.repeat(int(n_null_repeats), len(times_s)),
        }
    )


def build_testing_coefficient_tables(
    *,
    subject: str,
    fold_id: int,
    condition_labels: np.ndarray | list[str],
    trial_index: np.ndarray | list[int],
    times_s: np.ndarray | list[float],
    coef_by_name: dict[str, np.ndarray],
    coef_by_predictor: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build wide and long testing tables for held-out effect coefficients.

    Parameters
    ----------
    subject : str
        Subject identifier.
    fold_id : int
        Cross-validation fold identifier.
    condition_labels : np.ndarray | list[str]
        Condition label for each held-out trial.
    trial_index : np.ndarray | list[int]
        Trial index for each held-out trial.
    times_s : np.ndarray | list[float]
        Time axis in seconds.
    coef_by_name : dict[str, np.ndarray]
        Trial-by-time coefficient matrix for each reconstructed basis name.
        This may include ``"intercept"`` when the design formula has one.
    coef_by_predictor : dict[str, np.ndarray]
        Trial-by-time coefficient matrix for each modeled predictor.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Wide table with coefficient columns and long table with ``effect`` and
        ``coefficient`` columns.
    """

    condition_labels = np.asarray(condition_labels, dtype=object)
    trial_index = np.asarray(trial_index, dtype=int)
    times_s = np.asarray(times_s, dtype=float)

    matrices = {
        f"coef_{coef_name}": np.asarray(coef_values, dtype=float)
        for coef_name, coef_values in coef_by_name.items()
    }
    for name, matrix in matrices.items():
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be 2D.")

    first_matrix = next(iter(matrices.values()))
    n_trials, n_times = first_matrix.shape
    for name, matrix in matrices.items():
        if matrix.shape != (n_trials, n_times):
            raise ValueError(f"{name} must have shape {(n_trials, n_times)}.")
    if len(condition_labels) != n_trials:
        raise ValueError("condition_labels length must match coefficient rows.")
    if len(trial_index) != n_trials:
        raise ValueError("trial_index length must match coefficient rows.")
    if len(times_s) != n_times:
        raise ValueError("times_s length must match coefficient columns.")

    wide_data = {
        "subject": np.repeat(str(subject), n_trials * n_times),
        "fold": np.repeat(int(fold_id), n_trials * n_times),
        "condition": np.repeat(condition_labels.astype(str), n_times),
        "trial_index": np.repeat(trial_index.astype(int), n_times),
        "time_ms": np.tile(times_s * 1000.0, n_trials),
    }
    for coef_name, coef_values in coef_by_name.items():
        wide_data[f"coef_{coef_name}"] = np.asarray(coef_values, dtype=float).reshape(
            -1
        )
    wide_df = pd.DataFrame(wide_data)

    long_rows = []
    base_cols = ["subject", "fold", "condition", "trial_index", "time_ms"]
    for predictor_name in coef_by_predictor:
        coef_col = f"coef_{predictor_name}"
        long_rows.append(
            wide_df.loc[:, base_cols].assign(
                effect=predictor_name,
                coefficient=wide_df[coef_col].to_numpy(dtype=float),
            )
        )
    long_df = pd.concat(long_rows, ignore_index=True)

    return wide_df, long_df


def build_condition_average_coefficient_table(
    trial_table: pd.DataFrame,
) -> pd.DataFrame:
    """Average testing coefficients by subject, effect, condition, and time.

    Parameters
    ----------
    trial_table : pd.DataFrame
        Long testing table from ``build_testing_coefficient_tables``.

    Returns
    -------
    pd.DataFrame
        Condition-averaged table with columns ``subject``, ``effect``,
        ``condition``, ``time_ms``, and ``mean_coefficient``.
    """

    required_cols = {
        "subject",
        "effect",
        "condition",
        "time_ms",
        "coefficient",
    }
    missing_cols = sorted(required_cols.difference(trial_table.columns))
    if len(missing_cols) > 0:
        raise ValueError(f"trial_table is missing required columns: {missing_cols}")

    return (
        trial_table.groupby(["subject", "effect", "condition", "time_ms"], as_index=False)
        .agg(mean_coefficient=("coefficient", "mean"))
        .sort_values(["subject", "effect", "condition", "time_ms"])
        .reset_index(drop=True)
    )


def _validate_expression_inputs(
    condition_labels: np.ndarray,
    times: np.ndarray,
    expression_by_effect: dict[str, np.ndarray],
    trial_index: np.ndarray,
) -> dict[str, np.ndarray]:
    """Validate effect-level expression matrices before building output tables.

    Parameters
    ----------
    condition_labels : np.ndarray
        Condition labels with shape ``(n_trials,)``.
    times : np.ndarray
        Time axis with shape ``(n_times,)``.
    expression_by_effect : dict[str, np.ndarray]
        Pattern-expression matrix for each effect. Each matrix must have shape
        ``(n_trials, n_times)``.
    trial_index : np.ndarray
        Trial index values with shape ``(n_trials,)``.

    Returns
    -------
    dict[str, np.ndarray]
        Validated float matrices keyed by effect name.

    Raises
    ------
    ValueError
        If any input shape is incompatible.
    """

    if len(expression_by_effect) == 0:
        raise ValueError("expression_by_effect must include at least one effect.")

    validated = {}
    expected_shape = None
    for effect_name, expression in expression_by_effect.items():
        effect_label = str(effect_name).strip()
        if effect_label == "":
            raise ValueError("Effect names in expression_by_effect must not be empty.")

        matrix = np.asarray(expression, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"Expression for effect '{effect_label}' must be 2D.")
        if expected_shape is None:
            expected_shape = matrix.shape
        elif matrix.shape != expected_shape:
            raise ValueError(
                "All expression matrices must have identical shapes. "
                f"Expected {expected_shape}, got {matrix.shape} for '{effect_label}'."
            )
        validated[effect_label] = matrix

    n_trials, n_times = expected_shape

    if len(condition_labels) != n_trials:
        raise ValueError(
            "condition_labels length must match expression trial count. "
            f"Got {len(condition_labels)} labels and {n_trials} trials."
        )

    if len(times) != n_times:
        raise ValueError(
            "times length must match expression time count. "
            f"Got {len(times)} times and {n_times} time points."
        )

    if len(trial_index) != n_trials:
        raise ValueError(
            "trial_index length must match expression trial count. "
            f"Got {len(trial_index)} indices and {n_trials} trials."
        )

    return validated


def build_trial_pattern_expression_table(
    *,
    subject: str,
    condition_labels: np.ndarray | list[str],
    times: np.ndarray | list[float],
    expression_by_effect: dict[str, np.ndarray],
    trial_index: np.ndarray | list[int] | None = None,
) -> pd.DataFrame:
    """Build a trial-level pattern-expression table for any named effects.

    Parameters
    ----------
    subject : str
        Subject identifier written to every output row.
    condition_labels : np.ndarray | list[str]
        Condition label for each trial.
    times : np.ndarray | list[float]
        Time axis aligned with the expression matrices.
    expression_by_effect : dict[str, np.ndarray]
        Pattern-expression matrix for each effect. Each matrix must have shape
        ``(n_trials, n_times)``.
    trial_index : np.ndarray | list[int] | None
        Optional original trial index values. If omitted, zero-based row indices
        are used.

    Returns
    -------
    pd.DataFrame
        Long table with one row per subject, trial, time point, and effect.
    """

    condition_labels = np.asarray(condition_labels, dtype=object)
    times = np.asarray(times, dtype=float)

    if trial_index is None:
        trial_index = np.arange(len(condition_labels), dtype=int)
    else:
        trial_index = np.asarray(trial_index, dtype=int)

    expression_by_effect = _validate_expression_inputs(
        condition_labels=condition_labels,
        times=times,
        expression_by_effect=expression_by_effect,
        trial_index=trial_index,
    )

    rows = []
    for effect_name, expression in expression_by_effect.items():
        for trial_ix in range(expression.shape[0]):
            for time_ix, time_value in enumerate(times):
                rows.append(
                    {
                        "subject": str(subject),
                        "condition": str(condition_labels[trial_ix]),
                        "trial_index": int(trial_index[trial_ix]),
                        "time": float(time_value),
                        "effect": effect_name,
                        "pattern_expression": float(expression[trial_ix, time_ix]),
                    }
                )

    return pd.DataFrame(rows)


def build_condition_average_pattern_expression_table(
    trial_table: pd.DataFrame,
) -> pd.DataFrame:
    """Average pattern expression by subject, condition, effect, and time.

    Parameters
    ----------
    trial_table : pd.DataFrame
        Trial-level table from ``build_trial_pattern_expression_table``.

    Returns
    -------
    pd.DataFrame
        Condition-averaged table with one row per subject, condition, effect,
        and time point.
    """

    required_cols = {
        "subject",
        "condition",
        "effect",
        "time",
        "pattern_expression",
    }
    missing_cols = sorted(required_cols.difference(trial_table.columns))
    if len(missing_cols) > 0:
        raise ValueError(
            "trial_table is missing required columns: "
            f"{missing_cols}"
        )

    return (
        trial_table.groupby(["subject", "condition", "effect", "time"], as_index=False)
        .agg(mean_pattern_expression=("pattern_expression", "mean"))
        .sort_values(["subject", "condition", "effect", "time"])
        .reset_index(drop=True)
    )
