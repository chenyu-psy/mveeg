"""Automatic zero-code references for held-out component expression."""

from __future__ import annotations

from dataclasses import dataclass

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


def zero_code_references(
    design: EncodingDesign, training_labels: np.ndarray
) -> tuple[ExpressionReference, ...]:
    """Reference every component whose condition design contains code 0.

    A component is referenced when its code is constant within each training
    condition and at least one training condition has code exactly 0. All
    training conditions must be present for every subject, so the referenced
    components are the same for every subject. Trial-varying components keep
    the default training-mean zero. Every trial with code 0, including
    expression-only trials, belongs to the reference pool.
    """
    labels = np.asarray(training_labels, dtype=object)
    training = pd.notna(labels)
    plans = []
    for index, component in enumerate(design.components):
        codes = design.matrix[:, design.component_indices[index]]
        training_codes, training_groups = codes[training], labels[training]
        constant = all(
            np.all(training_codes[training_groups == group] == first)
            for group in pd.unique(training_groups)
            for first in [training_codes[training_groups == group][0]]
        )
        if not constant or not np.any(training_codes == 0):
            continue
        plans.append(ExpressionReference(component, index, 0.0, np.flatnonzero(codes == 0)))
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
