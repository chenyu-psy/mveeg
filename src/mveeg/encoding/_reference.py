"""Observed-code references for held-out component expression."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from ._metrics import compute_pattern_expression
from ._prepare import EncodingDesign


@dataclass(frozen=True)
class ExpressionReference:
    component: str
    component_index: int
    value: float
    rows: np.ndarray


def normalize_expression_reference(reference):
    """Validate sparse component-to-numeric-code settings without guessing labels."""
    if reference is None:
        return {}
    values = dict(reference)
    for component, value in values.items():
        if not isinstance(component, str) or not component.strip():
            raise ValueError("expression_reference keys must be nonempty component names.")
        if not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError("expression_reference values must be finite numeric component codes.")
    return {component: float(value) for component, value in values.items()}


def build_expression_references(
    design: EncodingDesign, reference: dict[str, float]
) -> tuple[ExpressionReference, ...]:
    """Fix observed matching-code membership independently of condition labels."""
    unknown = set(reference) - set(design.components)
    if unknown:
        raise ValueError(f"expression_reference names unknown components: {sorted(unknown)}.")
    if not reference:
        return ()
    plans = []
    for component, value in reference.items():
        index = design.components.index(component)
        rows = np.flatnonzero(design.matrix[:, design.component_indices[index]] == value)
        if not len(rows):
            raise ValueError(
                f"Component {component!r} has no expression trials with exact code {value}. "
                "Reference codes must be observed; round/define transformed codes explicitly "
                "if floating arithmetic does not produce the intended exact value."
            )
        plans.append(ExpressionReference(component, index, value, rows))
    return tuple(plans)


def compute_expression_references(
    *,
    subject,
    fold,
    plans,
    data,
    train_mean,
    train_rows,
    test_rows,
    raw_expression,
    beta_patterns,
    precision,
    trial_ids,
    times,
):
    """Pool equally weighted matching observations held out from this fit.

    Never-trained expression-only reference trials may supply auxiliary projections
    under every fold model. Public once-only scores and training splits stay intact.
    """
    lookup = np.full(len(data), -1, dtype=int)
    lookup[test_rows] = np.arange(len(test_rows))
    offsets = np.empty((len(plans), len(times)))
    reference_rows, member_rows = [], []
    for reference_index, plan in enumerate(plans):
        members = plan.rows[~np.isin(plan.rows, train_rows)]
        if not len(members):
            raise ValueError(
                f"Component {plan.component!r}, code {plan.value}, fold {fold}: "
                "no matching reference trials are held out from this model. "
                "Use a reference represented in every chosen fold, fewer folds, or "
                "an explicit CV seed with valid label coverage; no cross-fold borrowing "
                "or training-trial reuse is performed."
            )
        weights = np.full(len(members), 1 / len(members))
        positions = lookup[members]
        ordinary = positions >= 0
        values = np.empty((len(members), len(times)))
        values[ordinary] = raw_expression[positions[ordinary], plan.component_index, :]
        if np.any(~ordinary):
            extra, _ = compute_pattern_expression(
                data[members[~ordinary]] - train_mean,
                beta_patterns[[plan.component_index]],
                precision,
            )
            values[~ordinary] = extra[:, 0, :]
        if not np.isfinite(values).all():
            raise ValueError(
                f"Component {plan.component!r}, code {plan.value}, fold {fold}: "
                "reference expression is nonfinite (for example, a small filter denominator). "
                "An undefined or near-null pattern cannot define a reference."
            )
        offsets[reference_index] = values.mean(axis=0)
        for ti, time in enumerate(times):
            reference_rows.append(
                {
                    "subject_index": subject,
                    "fold": fold,
                    "component": plan.component,
                    "time": int(time),
                    "reference_value": plan.value,
                    "reference_expression": offsets[reference_index, ti],
                    "n_trials": len(members),
                }
            )
            for i, row in enumerate(members):
                member_rows.append(
                    {
                        "subject_index": subject,
                        "epoch_index": int(trial_ids[row]),
                        "fold": fold,
                        "component": plan.component,
                        "time": int(time),
                        "reference_value": plan.value,
                        "weight": weights[i],
                        "raw_expression": values[i, ti],
                        "auxiliary": bool(not ordinary[i]),
                    }
                )
    return offsets, pd.DataFrame(reference_rows), pd.DataFrame(member_rows)
