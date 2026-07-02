"""Design-validation helpers for encoding workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EncodingConfig
from .prepare import build_condition_encoding_from_df, build_trial_encoding
from .validation import ValidationResult, validate_encoding


def run_encoding_design_check(
    *,
    trial_conditions: np.ndarray | list[str],
    condition_encoding: pd.DataFrame | None = None,
    trial_df: pd.DataFrame | None = None,
    condition_col: str | None = None,
    variable_cols: list[str] | None = None,
    condition_order: list[str] | None = None,
    cfg: EncodingConfig | None = None,
) -> dict[str, object]:
    """Build and validate an encoding design from matrix or DataFrame input."""

    if cfg is None:
        cfg = EncodingConfig()

    if condition_encoding is None:
        if trial_df is None or condition_col is None or variable_cols is None:
            raise ValueError(
                "Provide either condition_encoding directly, or trial_df with "
                "condition_col and variable_cols."
            )
        condition_encoding = build_condition_encoding_from_df(
            trial_df,
            condition_col=condition_col,
            variable_cols=variable_cols,
            condition_order=condition_order,
        )

    X, design_names, trial_encoding = build_trial_encoding(
        condition_encoding=condition_encoding,
        trial_conditions=trial_conditions,
        add_intercept=cfg.add_intercept,
    )

    validation: ValidationResult = validate_encoding(
        X,
        design_names,
        mode=cfg.validation_mode,
        tol=cfg.tolerance,
        condition_encoding=condition_encoding,
    )

    return {
        "condition_encoding": condition_encoding,
        "trial_encoding": trial_encoding,
        "design_matrix": X,
        "design_names": design_names,
        "validation": validation,
    }



def _add_interaction_column(
    condition_encoding: pd.DataFrame,
    term: str,
    *,
    allowed_predictors: set[str],
) -> pd.Series:
    """Return one condition-level interaction column from existing predictors."""

    factors = [factor.strip() for factor in term.split(":")]
    if len(factors) < 2 or any(factor == "" for factor in factors):
        raise ValueError(f"Invalid interaction term '{term}'.")
    unknown = [factor for factor in factors if factor not in allowed_predictors]
    if len(unknown) > 0:
        raise ValueError(
            f"Unknown predictor(s) {unknown} in interaction term '{term}'. "
            f"Allowed predictors: {sorted(allowed_predictors)}"
        )

    values = np.ones(len(condition_encoding), dtype=float)
    for factor in factors:
        values = values * condition_encoding[factor].to_numpy(dtype=float)
    return pd.Series(values, index=condition_encoding.index, name=term)


def build_formula_condition_encoding(
    condition_encoding: pd.DataFrame,
    glm_formula: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build a formula-selected condition table with simple interactions.

    Parameters
    ----------
    condition_encoding : pd.DataFrame
        Condition-level design table with one ``condition`` column and numeric
        predictor columns.
    glm_formula : str
        Formula using additive terms, ``a:b`` interactions, and ``a * b``
        shorthand. Complex formula functions are intentionally unsupported.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, object]]
        Condition table restricted to selected/generated columns and the parsed
        formula metadata.
    """

    if "condition" not in condition_encoding.columns:
        raise ValueError("condition_encoding must include a 'condition' column.")

    parsed = validate_glm_formula(
        glm_formula,
        allowed_predictors=set(condition_encoding.columns).difference({"condition"}),
    )
    output = condition_encoding.loc[:, ["condition"]].copy()
    for predictor in parsed["predictors"]:
        if predictor in condition_encoding.columns:
            output[predictor] = condition_encoding[predictor].to_numpy(dtype=float)
        else:
            output[predictor] = _add_interaction_column(
                condition_encoding,
                predictor,
                allowed_predictors=set(condition_encoding.columns).difference({"condition"}),
            )

    return output, parsed


def validate_glm_formula(
    glm_formula: str,
    *,
    allowed_predictors: set[str],
) -> dict[str, object]:
    """Validate and parse a small R-style GLM formula."""

    formula_text = str(glm_formula).strip()
    if not formula_text.startswith("~"):
        raise ValueError(
            f"glm_formula must start with '~'. Got: '{glm_formula}'"
        )

    rhs = formula_text[1:].strip()
    if rhs == "":
        raise ValueError("glm_formula must include right-hand-side terms.")

    raw_terms = [term.strip() for term in rhs.split("+")]
    raw_terms = [term for term in raw_terms if term != ""]
    if len(raw_terms) == 0:
        raise ValueError("glm_formula has no valid terms after parsing.")

    add_intercept = True
    predictors: list[str] = []
    interactions: list[str] = []

    def add_predictor(name: str) -> None:
        """Add one predictor name while preserving first-seen order."""

        if name not in predictors:
            predictors.append(name)

    def add_interaction(term: str) -> None:
        """Add one interaction term after validating its factor names."""

        factors = [factor.strip() for factor in term.split(":")]
        if len(factors) < 2 or any(factor == "" for factor in factors):
            raise ValueError(f"Invalid interaction term '{term}'.")
        unknown = [factor for factor in factors if factor not in allowed_predictors]
        if len(unknown) > 0:
            raise ValueError(
                f"Unknown predictor(s) {unknown} in interaction term '{term}'. "
                f"Allowed predictors: {sorted(allowed_predictors)}"
            )
        label = ":".join(factors)
        add_predictor(label)
        if label not in interactions:
            interactions.append(label)

    for term in raw_terms:
        if term in {"1"}:
            add_intercept = True
            continue
        if term in {"0", "-1"}:
            add_intercept = False
            continue
        if any(token in term for token in ["^", "(", ")"]):
            raise ValueError(
                "glm_formula supports additive terms and simple interactions "
                "(e.g., '~ 1 + feature_a * feature_b'). "
                f"Unsupported term: '{term}'. "
            )
        if "*" in term:
            factors = [factor.strip() for factor in term.split("*")]
            if len(factors) != 2 or any(factor == "" for factor in factors):
                raise ValueError(
                    "Only two-way '*' interactions are supported. "
                    f"Unsupported term: '{term}'."
                )
            unknown = [factor for factor in factors if factor not in allowed_predictors]
            if len(unknown) > 0:
                raise ValueError(
                    f"Unknown predictor(s) {unknown} in glm_formula '{glm_formula}'. "
                    f"Allowed predictors: {sorted(allowed_predictors)}"
                )
            for factor in factors:
                add_predictor(factor)
            add_interaction(":".join(factors))
            continue
        if ":" in term:
            add_interaction(term)
            continue
        if term not in allowed_predictors:
            raise ValueError(
                f"Unknown predictor '{term}' in glm_formula '{glm_formula}'. "
                f"Allowed predictors: {sorted(allowed_predictors)}"
            )
        add_predictor(term)

    if len(predictors) == 0:
        raise ValueError(
            "glm_formula must include at least one predictor besides intercept."
        )

    return {
        "add_intercept": add_intercept,
        "predictors": predictors,
        "interactions": interactions,
    }

