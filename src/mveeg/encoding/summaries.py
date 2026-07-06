"""Summary-table builders for encoding model outputs.

This module reshapes already-computed training and testing outputs into
analysis-ready tables for downstream modeling in Python or R.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def build_pattern_expression_trial_table(
    *,
    subject: str,
    fold_id: int,
    condition_labels: np.ndarray | list[str],
    trial_index: np.ndarray | list[int],
    times_s: np.ndarray | list[float],
    effect_names: list[str],
    expression: np.ndarray,
    covariance_method: str,
) -> pd.DataFrame:
    """Build the primary held-out trial-level pattern-expression table.

    Parameters
    ----------
    subject : str
        Subject identifier written to every output row.
    fold_id : int
        Cross-validation fold number.
    condition_labels : np.ndarray | list[str]
        Condition label for each held-out trial.
    trial_index : np.ndarray | list[int]
        Original trial index for each held-out trial.
    times_s : np.ndarray | list[float]
        Time values in seconds.
    effect_names : list[str]
        Names for modeled non-intercept effects.
    expression : np.ndarray
        Expression values with shape ``(n_trials, n_effects, n_times)``.
    covariance_method : str
        Covariance method used to compute expression.

    Returns
    -------
    pd.DataFrame
        Long table with ``expression`` values in milliseconds.
    """

    condition_labels = np.asarray(condition_labels, dtype=object)
    trial_index = np.asarray(trial_index, dtype=int)
    times_s = np.asarray(times_s, dtype=float)
    expression = np.asarray(expression, dtype=float)
    if expression.ndim != 3:
        raise ValueError("expression must have shape (n_trials, n_effects, n_times).")

    n_trials, n_effects, n_times = expression.shape
    if len(condition_labels) != n_trials:
        raise ValueError("condition_labels length must match expression trials.")
    if len(trial_index) != n_trials:
        raise ValueError("trial_index length must match expression trials.")
    if len(effect_names) != n_effects:
        raise ValueError("effect_names length must match expression effects.")
    if len(times_s) != n_times:
        raise ValueError("times_s length must match expression times.")

    rows = []
    for effect_ix, effect_name in enumerate(effect_names):
        rows.append(
            pd.DataFrame(
                {
                    "subject": np.repeat(str(subject), n_trials * n_times),
                    "fold": np.repeat(int(fold_id), n_trials * n_times),
                    "trial_index": np.repeat(trial_index, n_times),
                    "condition": np.repeat(condition_labels.astype(str), n_times),
                    "effect": np.repeat(str(effect_name), n_trials * n_times),
                    "time_ms": np.tile(times_s * 1000.0, n_trials),
                    "expression": expression[:, effect_ix, :].reshape(-1),
                    "covariance_method": np.repeat(
                        str(covariance_method), n_trials * n_times
                    ),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def build_condition_pattern_expression_table(
    trial_table: pd.DataFrame,
) -> pd.DataFrame:
    """Average held-out expression by subject, condition, effect, and time.

    Parameters
    ----------
    trial_table : pd.DataFrame
        Trial-level table from ``build_pattern_expression_trial_table``.

    Returns
    -------
    pd.DataFrame
        Condition-level expression summary with mean, SD, SE, and trial count.
    """

    required_cols = {
        "subject",
        "condition",
        "effect",
        "time_ms",
        "expression",
        "covariance_method",
    }
    missing_cols = sorted(required_cols.difference(trial_table.columns))
    if len(missing_cols) > 0:
        raise ValueError(f"trial_table is missing required columns: {missing_cols}")

    summary = (
        trial_table.groupby(
            ["subject", "condition", "effect", "time_ms", "covariance_method"],
            as_index=False,
        )
        .agg(
            expression_mean=("expression", "mean"),
            expression_sd=("expression", "std"),
            n_trials=("trial_index", "nunique"),
        )
        .sort_values(["subject", "condition", "effect", "time_ms"])
        .reset_index(drop=True)
    )
    summary["expression_se"] = summary["expression_sd"] / np.sqrt(
        summary["n_trials"].astype(float)
    )
    return summary.loc[
        :,
        [
            "subject",
            "condition",
            "effect",
            "time_ms",
            "expression_mean",
            "expression_sd",
            "expression_se",
            "n_trials",
            "covariance_method",
        ],
    ]


def build_effect_slope_table(
    *,
    subject: str,
    trial_expression_df: pd.DataFrame,
    trial_design: pd.DataFrame,
    effect_names: list[str],
    times_s: np.ndarray | list[float],
    covariance_method: str,
) -> pd.DataFrame:
    """Fit subject-level expression slopes for each effect and time.

    Parameters
    ----------
    subject : str
        Subject identifier.
    trial_expression_df : pd.DataFrame
        Held-out expression table containing all folds for one subject.
    trial_design : pd.DataFrame
        Trial-level design table with one row per original trial and columns
        matching ``effect_names``.
    effect_names : list[str]
        Non-intercept effects to summarize.
    times_s : np.ndarray | list[float]
        Time values in seconds.
    covariance_method : str
        Covariance method used for expression values.

    Returns
    -------
    pd.DataFrame
        Subject-level slope table.
    """

    times_ms = np.asarray(times_s, dtype=float) * 1000.0
    rows = []
    for effect_name in effect_names:
        if effect_name not in trial_design.columns:
            raise ValueError(f"trial_design is missing effect column '{effect_name}'.")
        x_by_trial = trial_design[effect_name].to_numpy(dtype=float)
        for time_ms in times_ms:
            rows_df = trial_expression_df.loc[
                (trial_expression_df["effect"] == effect_name)
                & (trial_expression_df["time_ms"] == time_ms)
            ]
            y = rows_df["expression"].to_numpy(dtype=float)
            trial_ix = rows_df["trial_index"].to_numpy(dtype=int)
            x = x_by_trial[trial_ix]
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 2 or np.nanstd(x[valid]) == 0:
                slope = np.nan
                intercept = np.nan
                n_trials = int(valid.sum())
            else:
                design = np.column_stack([np.ones(valid.sum(), dtype=float), x[valid]])
                coef, *_ = np.linalg.lstsq(design, y[valid], rcond=None)
                intercept = float(coef[0])
                slope = float(coef[1])
                n_trials = int(valid.sum())
            rows.append(
                {
                    "subject": str(subject),
                    "effect": str(effect_name),
                    "time_ms": float(time_ms),
                    "slope": slope,
                    "intercept": intercept,
                    "n_trials": n_trials,
                    "n_folds": int(rows_df["fold"].nunique()),
                    "covariance_method": str(covariance_method),
                }
            )

    return pd.DataFrame(rows)
