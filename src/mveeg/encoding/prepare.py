"""Trial roles and component/condition design construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_DROP_TYPES = ("eog", "eyegaze", "pupil", "misc")
RESERVED_TRIAL_COLUMNS = {"subject", "trial", "training_group", "expression_group"}


@dataclass(frozen=True)
class EncodingDesign:
    """One subject's complete component-plus-condition design."""

    matrix: np.ndarray
    predictors: pd.DataFrame
    components: tuple[str, ...]
    component_indices: tuple[int, ...]
    interactions: tuple[str, ...]


def validate_groups(
    conditions: Mapping[str, Sequence[object]],
    expression: Mapping[str, Sequence[object]] | None,
) -> tuple[dict[str, list[object]], dict[str, list[object]]]:
    """Validate independent training and expression mappings."""

    condition_map = _validate_mapping("conditions", conditions)
    expression_map = (
        {label: list(values) for label, values in condition_map.items()}
        if expression is None
        else _validate_mapping("expression", expression)
    )
    expression_values = [value for values in expression_map.values() for value in values]
    missing = [
        value
        for values in condition_map.values()
        for value in values
        if value not in expression_values
    ]
    if missing:
        raise ValueError(
            "expression must include every conditions value; "
            f"missing {missing}."
        )
    return condition_map, expression_map


def select_trials(
    metadata: pd.DataFrame,
    *,
    target: str,
    conditions: dict[str, list[object]],
    expression: dict[str, list[object]],
    qc: str | None,
    keep: Sequence[object],
    exclude: Mapping[str, Sequence[object] | str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Apply QC and return training/expression labels plus source row indices."""

    metadata = _drop_redundant_subject(metadata)
    if target not in metadata.columns:
        raise ValueError(f"target column {target!r} is missing after transform_metadata.")
    collisions = sorted(RESERVED_TRIAL_COLUMNS.intersection(metadata.columns))
    if collisions:
        raise ValueError(f"Metadata uses reserved encoding columns: {collisions}.")

    mask = np.ones(len(metadata), dtype=bool)
    if qc is not None:
        if qc not in metadata.columns:
            raise ValueError(f"QC column {qc!r} is missing; label artifacts or set qc=None.")
        mask &= metadata[qc].isin(tuple(keep)).to_numpy()
    for column, rule in exclude.items():
        if column not in metadata.columns:
            raise ValueError(f"Trial exclusion column {column!r} is missing.")
        if rule == "notna":
            mask &= metadata[column].isna().to_numpy()
        else:
            mask &= ~metadata[column].isin(tuple(rule)).to_numpy()

    values = metadata[target].to_numpy(dtype=object)
    training_labels = _map_values(values, conditions)
    expression_labels = _map_values(values, expression)
    mask &= pd.notna(expression_labels)
    if not np.any(mask):
        raise ValueError("No trials remain after selection and encoding mappings.")

    rows = np.flatnonzero(mask)
    selected = metadata.loc[mask].reset_index(drop=True).copy()
    return selected, training_labels[mask], expression_labels[mask], rows


def build_design(
    metadata: pd.DataFrame,
    *,
    formula: str,
    training_labels: np.ndarray,
    condition_order: Sequence[str],
    penalty: Mapping[str, float],
) -> EncodingDesign:
    """Build the numeric component design and complete condition basis."""

    parsed = parse_formula(formula, allowed_predictors=set(metadata.columns))
    columns = [np.ones(len(metadata), dtype=float)]
    names = ["intercept"]
    terms = ["intercept"]
    roles = ["intercept"]
    penalties = [0.0]

    for predictor in parsed["predictors"]:
        factors = predictor.split(":")
        values = np.ones(len(metadata), dtype=float)
        try:
            for factor in factors:
                values *= metadata[factor].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"formula component {predictor!r} must contain numeric values."
            ) from error
        if not np.all(np.isfinite(values)):
            raise ValueError(f"formula component {predictor!r} contains non-finite values.")
        columns.append(values)
        names.append(predictor)
        terms.append(predictor)
        roles.append("component")
        penalties.append(float(penalty["component"]))

    component_count = len(columns)
    training_labels = np.asarray(training_labels, dtype=object)
    for condition in condition_order:
        name = f"condition[{condition}]"
        if name in names:
            raise ValueError(f"Generated condition predictor collides with {name!r}.")
        columns.append((training_labels == condition).astype(float))
        names.append(name)
        terms.append("condition")
        roles.append("condition")
        penalties.append(float(penalty["condition"]))

    matrix = np.column_stack(columns)

    predictors = pd.DataFrame(
        {
            "predictor": names,
            "term": terms,
            "role": roles,
            "penalty": penalties,
        }
    )
    return EncodingDesign(
        matrix=matrix,
        predictors=predictors,
        components=tuple(parsed["predictors"]),
        component_indices=tuple(range(1, component_count)),
        interactions=tuple(parsed["interactions"]),
    )


def parse_formula(formula: str, *, allowed_predictors: set[str]) -> dict[str, list[str]]:
    """Parse the deliberately small RHS-only component formula language."""

    text = str(formula).strip()
    if "~" in text:
        raise ValueError("formula must contain only right-hand-side component terms.")
    if text == "":
        raise ValueError("formula must include at least one component.")

    predictors: list[str] = []
    interactions: list[str] = []

    def add(name: str) -> None:
        if name not in predictors:
            predictors.append(name)

    for term in (part.strip() for part in text.split("+")):
        if term == "" or term == "1":
            continue
        if term in {"0", "-1"}:
            raise ValueError("The encoding model always includes an intercept.")
        if any(token in term for token in ("(", ")", "|", "^")):
            raise ValueError(f"Unsupported formula term: {term!r}.")
        if "*" in term:
            factors = [factor.strip() for factor in term.split("*")]
            if len(factors) != 2 or any(factor == "" for factor in factors):
                raise ValueError("Only two-way '*' interactions are supported.")
            _require_factors(factors, allowed_predictors, formula)
            for factor in factors:
                add(factor)
            interaction = ":".join(factors)
            add(interaction)
            if interaction not in interactions:
                interactions.append(interaction)
            continue
        factors = [factor.strip() for factor in term.split(":")]
        _require_factors(factors, allowed_predictors, formula)
        name = ":".join(factors)
        add(name)
        if len(factors) > 1 and name not in interactions:
            interactions.append(name)

    if not predictors:
        raise ValueError("formula must include at least one component besides intercept.")
    if "intercept" in predictors:
        raise ValueError("'intercept' is reserved by the encoding model.")
    return {"predictors": predictors, "interactions": interactions}


def _require_factors(
    factors: Sequence[str],
    allowed_predictors: set[str],
    formula: str,
) -> None:
    if any(factor == "" for factor in factors):
        raise ValueError(f"Invalid interaction in formula {formula!r}.")
    unknown = [factor for factor in factors if factor not in allowed_predictors]
    if unknown:
        raise ValueError(
            f"Unknown component(s) {unknown} in formula {formula!r}; "
            f"available columns are {sorted(allowed_predictors)}."
        )


def _validate_mapping(
    name: str,
    mapping: Mapping[str, Sequence[object]],
) -> dict[str, list[object]]:
    if not isinstance(mapping, Mapping) or len(mapping) < 1:
        raise ValueError(f"{name} must contain at least one named group.")
    output: dict[str, list[object]] = {}
    seen: list[object] = []
    for label, raw_values in mapping.items():
        if not isinstance(label, str) or label.strip() == "":
            raise ValueError(f"{name} labels must be non-empty strings.")
        if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
            raise TypeError(f"{name}[{label!r}] must be a non-string sequence.")
        values = list(raw_values)
        if not values:
            raise ValueError(f"{name}[{label!r}] cannot be empty.")
        overlap = [
            value
            for index, value in enumerate(values)
            if value in seen or value in values[:index]
        ]
        if overlap:
            raise ValueError(f"{name} values must map to one group; duplicated {overlap}.")
        output[label] = values
        seen.extend(values)
    return output


def _map_values(values: np.ndarray, mapping: dict[str, list[object]]) -> np.ndarray:
    labels = np.full(len(values), None, dtype=object)
    for label, raw_values in mapping.items():
        labels[np.isin(values, raw_values)] = label
    return labels


def _drop_redundant_subject(metadata: pd.DataFrame) -> pd.DataFrame:
    if "subject" not in metadata.columns:
        return metadata
    subject = metadata["subject"].astype("string").str.strip()
    subject_index = metadata["subject_index"].astype("string").str.strip()
    same = subject.eq(subject_index).all()
    if not same:
        subject_number = pd.to_numeric(subject, errors="coerce")
        index_number = pd.to_numeric(subject_index, errors="coerce")
        same = subject_number.notna().all() and index_number.notna().all()
        same = same and np.array_equal(subject_number, index_number)
    return metadata.drop(columns="subject") if same else metadata
