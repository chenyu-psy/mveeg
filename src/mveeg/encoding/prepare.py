"""Design-construction helpers for encoding analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_condition_encoding_from_df(
    df: pd.DataFrame,
    *,
    condition_col: str,
    variable_cols: list[str],
    condition_order: list[str] | None = None,
) -> pd.DataFrame:
    """Build a condition-level encoding table from trial-level metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level table containing a condition column and variable columns.
    condition_col : str
        Column defining task conditions.
    variable_cols : list[str]
        Variable columns used as encoding predictors.
    condition_order : list[str] | None
        Optional explicit condition order for output rows.

    Returns
    -------
    pd.DataFrame
        Condition-level encoding table with one row per condition.

    Raises
    ------
    ValueError
        If a condition maps to inconsistent variable values.

    Examples
    --------
    >>> tbl = pd.DataFrame({"cond": ["A", "A", "B"], "feature_a": [0, 0, 1]})
    >>> build_condition_encoding_from_df(tbl, condition_col="cond", variable_cols=["feature_a"])
      condition  feature_a
    0         A          0
    1         B          1
    """

    missing_cols = [col for col in [condition_col, *variable_cols] if col not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Missing required columns: {missing_cols}")

    grouped = df.loc[:, [condition_col, *variable_cols]].copy()
    duplicates = grouped.groupby(condition_col, as_index=False).nunique(dropna=False)
    inconsistent_cols = []
    for col in variable_cols:
        bad_conditions = duplicates.loc[duplicates[col] > 1, condition_col].tolist()
        if len(bad_conditions) > 0:
            inconsistent_cols.append((col, bad_conditions))

    if len(inconsistent_cols) > 0:
        issues = "; ".join(
            [f"{col} inconsistent in {conds}" for col, conds in inconsistent_cols]
        )
        raise ValueError(
            "Each condition must map to one unique value per variable. "
            f"Found inconsistencies: {issues}"
        )

    condition_df = grouped.drop_duplicates(subset=[condition_col]).copy()
    if condition_order is None:
        condition_order = condition_df[condition_col].tolist()

    missing_order = [cond for cond in condition_order if cond not in condition_df[condition_col].values]
    if len(missing_order) > 0:
        raise ValueError(f"condition_order includes unknown conditions: {missing_order}")

    condition_df[condition_col] = pd.Categorical(condition_df[condition_col], condition_order, ordered=True)
    condition_df = condition_df.sort_values(condition_col).reset_index(drop=True)
    return condition_df.rename(columns={condition_col: "condition"})


def build_trial_encoding(
    condition_encoding: pd.DataFrame,
    trial_conditions: np.ndarray | list[str],
    *,
    add_intercept: bool = True,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Expand condition-level encoding to a trial-level design matrix.

    Parameters
    ----------
    condition_encoding : pd.DataFrame
        Condition-level encoding table. Must include a ``condition`` column and
        one or more numeric variable columns.
    trial_conditions : np.ndarray | list[str]
        Condition label for each trial.
    add_intercept : bool
        Whether to prepend an intercept column of ones.

    Returns
    -------
    tuple[np.ndarray, list[str], pd.DataFrame]
        Trial-level design matrix, column names, and trial-level table.

    Raises
    ------
    ValueError
        If trials contain unknown conditions.
    """

    if "condition" not in condition_encoding.columns:
        raise ValueError("condition_encoding must include a 'condition' column.")

    variable_cols = [col for col in condition_encoding.columns if col != "condition"]
    if len(variable_cols) == 0:
        raise ValueError("condition_encoding must include at least one variable column.")

    trial_table = pd.DataFrame({"condition": np.asarray(trial_conditions, dtype=object)})
    merged = trial_table.merge(condition_encoding, on="condition", how="left", validate="many_to_one")

    if merged[variable_cols].isna().any().any():
        unknown_conditions = sorted(
            set(merged.loc[merged[variable_cols].isna().any(axis=1), "condition"].astype(str).tolist())
        )
        raise ValueError(
            "Found trial conditions missing from condition_encoding: "
            f"{unknown_conditions}"
        )

    X = merged[variable_cols].to_numpy(dtype=float)
    col_names = list(variable_cols)

    if add_intercept:
        X = np.column_stack([np.ones(len(merged), dtype=float), X])
        col_names = ["intercept", *col_names]

    return X, col_names, merged
