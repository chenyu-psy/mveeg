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



def validate_glm_formula(
    glm_formula: str,
    *,
    allowed_predictors: set[str],
) -> dict[str, object]:
    """Validate and parse a simple R-style GLM formula."""

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

    for term in raw_terms:
        if term in {"1"}:
            add_intercept = True
            continue
        if term in {"0", "-1"}:
            add_intercept = False
            continue
        if any(token in term for token in [":", "*", "^", "(", ")"]):
            raise ValueError(
                "glm_formula supports additive main effects only (e.g., "
                "'~ 1 + feature_a + feature_b'). "
                f"Unsupported term: '{term}'. "
                "R-style interaction terms such as 'feature_a * feature_b' "
                "expand to main effects plus an interaction and add an extra effect."
            )
        if term not in allowed_predictors:
            raise ValueError(
                f"Unknown predictor '{term}' in glm_formula '{glm_formula}'. "
                f"Allowed predictors: {sorted(allowed_predictors)}"
            )
        if term not in predictors:
            predictors.append(term)

    if len(predictors) == 0:
        raise ValueError(
            "glm_formula must include at least one predictor besides intercept."
        )

    return {
        "add_intercept": add_intercept,
        "predictors": predictors,
    }


