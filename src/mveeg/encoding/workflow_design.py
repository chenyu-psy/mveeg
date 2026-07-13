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


def build_formula_metadata_design(
    metadata: pd.DataFrame,
    formula: str,
    *,
    fit_indices: np.ndarray | list[int] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, object]]:
    """Build a formula-selected trial-level design from metadata."""

    if len(metadata) == 0:
        raise ValueError("metadata must include at least one trial row.")
    fit_indices = (
        np.arange(len(metadata), dtype=int)
        if fit_indices is None
        else np.asarray(fit_indices, dtype=int)
    )
    if len(fit_indices) == 0:
        raise ValueError("fit_indices must include at least one row.")

    parsed = validate_glm_formula(
        formula,
        allowed_predictors=set(metadata.columns),
    )
    columns = []
    names = []
    term_types = []
    trial_design = metadata.copy()
    if parsed["add_intercept"]:
        columns.append(np.ones(len(metadata), dtype=float))
        names.append("intercept")
        term_types.append("intercept")

    for predictor in parsed["predictors"]:
        if ":" in predictor:
            values = _metadata_interaction_column(metadata, predictor)
            trial_design[predictor] = values
        else:
            if predictor not in metadata.columns:
                raise ValueError(f"metadata is missing formula column '{predictor}'.")
            values = metadata[predictor].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"fixed term '{predictor}' contains non-finite values.")
        columns.append(values.astype(float))
        names.append(predictor)
        term_types.append("fixed")

    random_terms = []
    for random in parsed["random_terms"]:
        variable = random["variable"]
        if variable not in metadata.columns:
            raise ValueError(f"metadata is missing random term column '{variable}'.")
        values = metadata[variable].astype(str).to_numpy()
        train_levels = sorted(_ordered_unique(values[fit_indices]))
        random_names = []
        for level in train_levels:
            safe_level = _safe_level_name(level)
            name = f"random_{variable}_{safe_level}"
            columns.append((values == level).astype(float))
            names.append(name)
            term_types.append("random")
            random_names.append(name)
        random_terms.append({"variable": variable, "levels": train_levels, "columns": random_names})

    design = np.column_stack(columns).astype(float) if columns else np.empty((len(metadata), 0), dtype=float)
    parsed = {
        **parsed,
        "term_types": term_types,
        "random_terms": random_terms,
    }
    return design, names, trial_design, parsed


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
    if "~" in formula_text:
        lhs, rhs = formula_text.split("~", maxsplit=1)
        _ = lhs.strip()
    else:
        rhs = formula_text

    rhs = rhs.strip()
    if rhs == "":
        raise ValueError("formula must include right-hand-side terms.")

    raw_terms = _split_formula_terms(rhs)
    raw_terms = [term for term in raw_terms if term != ""]
    if len(raw_terms) == 0:
        raise ValueError("formula has no valid terms after parsing.")

    add_intercept = True
    predictors: list[str] = []
    interactions: list[str] = []
    random_terms: list[dict[str, str]] = []

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
        random_term = _parse_random_intercept(term)
        if random_term is not None:
            random_terms.append(random_term)
            continue
        if any(token in term for token in ["^", "(", ")", "|"]):
            raise ValueError(
                "formula supports additive terms, simple interactions, and "
                "random intercept terms "
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
                    f"Unknown predictor(s) {unknown} in formula '{glm_formula}'. "
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
                f"Unknown predictor '{term}' in formula '{glm_formula}'. "
                f"Allowed predictors: {sorted(allowed_predictors)}"
            )
        add_predictor(term)

    if len(predictors) == 0 and len(random_terms) == 0:
        raise ValueError(
            "formula must include at least one term besides intercept."
        )

    return {
        "add_intercept": add_intercept,
        "predictors": predictors,
        "interactions": interactions,
        "random_terms": random_terms,
    }


def _split_formula_terms(rhs: str) -> list[str]:
    terms = []
    depth = 0
    start = 0
    for ix, char in enumerate(rhs):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced formula parentheses in '{rhs}'.")
        elif char == "+" and depth == 0:
            terms.append(rhs[start:ix].strip())
            start = ix + 1
    if depth != 0:
        raise ValueError(f"Unbalanced formula parentheses in '{rhs}'.")
    terms.append(rhs[start:].strip())
    return terms


def _parse_random_intercept(term: str) -> dict[str, str] | None:
    text = term.strip()
    if not (text.startswith("(") and text.endswith(")") and "|" in text):
        return None
    left, right = [part.strip() for part in text[1:-1].split("|", maxsplit=1)]
    if left != "1" or right == "":
        raise ValueError("Only random intercept terms like '(1 | column)' are supported.")
    return {"variable": right}


def _metadata_interaction_column(metadata: pd.DataFrame, term: str) -> np.ndarray:
    factors = [factor.strip() for factor in term.split(":")]
    values = np.ones(len(metadata), dtype=float)
    for factor in factors:
        if factor not in metadata.columns:
            raise ValueError(f"metadata is missing interaction factor '{factor}'.")
        values = values * metadata[factor].to_numpy(dtype=float)
    return values


def _ordered_unique(values: np.ndarray) -> list[str]:
    return list(dict.fromkeys(np.asarray(values, dtype=object).astype(str).tolist()))


def _safe_level_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(value)).strip("_") or "level"
