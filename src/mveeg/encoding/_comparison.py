"""Independent-partition signal energy in fixed theoretical condition subspaces."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedKFold

from ._analysis import _subject_seed
from ._prepare import parse_formula


def condition_weights(conditions, weights):
    """Normalize one positive condition measure, shared across all candidates."""
    if weights is not None and set(weights) != set(conditions):
        raise ValueError("weights must name every training condition exactly once.")
    values = np.array([1.0 if weights is None else weights[c] for c in conditions], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Condition weights must be finite and positive.")
    return dict(
        zip(conditions, (values / values.max()) / (values / values.max()).sum(), strict=True)
    )


def _basis(matrix):
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = np.finfo(float).eps * max(matrix.shape) * (singular[0] if singular.size else 0)
    return u[:, singular > tolerance]


def _design_basis(matrix):
    scales = np.linalg.norm(matrix, axis=0)
    return _basis(matrix[:, scales > 0] / scales[scales > 0])


def _condition_design(metadata, labels, conditions, formula):
    terms = (
        []
        if formula is None or formula.strip() == "1"
        else parse_formula(formula, allowed_predictors=set(metadata.columns))["predictors"]
    )
    columns = [np.ones(len(metadata))]
    for term in terms:
        values = np.ones(len(metadata))
        for factor in term.split(":"):
            values *= metadata[factor].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Comparison component {term!r} must be finite and numeric.")
        columns.append(values)
    matrix = np.column_stack(columns)
    design = []
    for condition in conditions:
        rows = matrix[labels == condition]
        if not len(rows):
            raise ValueError(f"Missing training condition {condition!r}.")
        if not np.all(rows == rows[0]):
            raise ValueError(
                "Comparison requires condition-constant codes; "
                f"{condition!r} varies within condition."
            )
        design.append(rows[0])
    return np.array(design), ["intercept", *terms]


def build_projectors(metadata, labels, conditions, models, nuisance, weights):
    """Build joint projectors after the same weighted nuisance removal."""
    root_weight = np.sqrt([weights[c] for c in conditions])
    nuisance_design, _ = _condition_design(metadata, labels, conditions, nuisance)
    nuisance_basis = _design_basis(root_weight[:, None] * nuisance_design)
    transform = (np.eye(len(conditions)) - nuisance_basis @ nuisance_basis.T) * root_weight
    projectors, designs = {}, []
    for model, formula in models.items():
        design, terms = _condition_design(metadata, labels, conditions, formula)
        # Drop numerical remnants of nuisance columns before determining model rank.
        residualized = transform @ design
        scale = np.linalg.norm(root_weight[:, None] * design, axis=0)
        residualized[:, np.linalg.norm(residualized, axis=0) <= 1e-12 * scale] = 0
        basis = _design_basis(residualized)
        projectors[model] = basis @ basis.T
        for c, condition in enumerate(conditions):
            for j, term in enumerate(terms):
                designs.append(
                    {
                        "model": model,
                        "condition": condition,
                        "term": term,
                        "value": design[c, j],
                        "weight": weights[condition],
                    }
                )
    return transform, projectors, pd.DataFrame(designs)


def partition_trials(labels, *, groups, folds, seed):
    """Use supplied independent groups or explicitly assumed exchangeable trial folds."""
    if groups is None:
        if folds < 3:
            raise ValueError("Comparison requires at least three independent partitions.")
        counts = pd.Series(labels).value_counts()
        if (counts < folds).any():
            raise ValueError(
                f"Every condition needs at least {folds} trials; counts={counts.to_dict()}."
            )
        assignments = np.empty(len(labels), dtype=object)
        split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for index, (_, rows) in enumerate(split.split(np.zeros(len(labels)), labels), start=1):
            assignments[rows] = str(index)
    else:
        if pd.isna(groups).any():
            raise ValueError("Independent group labels cannot be missing.")
        assignments = np.asarray(groups).astype(str)
    partitions = list(pd.unique(assignments))
    if len(partitions) < 3:
        raise ValueError("Comparison requires at least three independent groups.")
    for partition in partitions:
        missing = set(labels) - set(labels[assignments == partition])
        if missing:
            raise ValueError(
                f"Group {partition!r} is missing conditions {sorted(missing)}; "
                "condition/run confounding or incomplete coverage cannot be imputed."
            )
    return assignments, partitions


def noise_precision(residuals, noise_df=None):
    """Shrink full retained-channel covariance without deleting sample-null directions.

    The sample rank is a diagnostic, not an estimate of physical channel support.
    No evaluation EEG enters either the covariance or its regularization.
    """
    centered = residuals - residuals.mean(axis=0, keepdims=True)
    rank = _basis(centered.T).shape[1]
    if not rank:
        raise ValueError("No estimable channel noise directions in the remaining groups.")
    scales = np.sqrt(np.mean(centered**2, axis=0))
    if np.any(scales == 0):
        raise ValueError(
            "A retained channel has no complement noise variance; drop flat channels "
            "before estimating comparison precision."
        )
    # Shrink correlations, preserving channel variance differences instead of
    # pulling high- and low-variance EEG sensors toward a common variance.
    estimate = LedoitWolf(assume_centered=True).fit(centered / scales)
    if estimate.shrinkage_ == 0 and rank < centered.shape[1]:
        raise ValueError(
            "The complement covariance is rank deficient without positive shrinkage; "
            "its unsampled channel variance cannot be estimated."
        )
    degrees = len(centered) - 1 if noise_df is None else noise_df
    precision = estimate.precision_ / scales[:, None] / scales[None, :]
    precision *= degrees / len(centered)
    covariance = scales[:, None] * estimate.covariance_ * scales[None, :]
    return (
        precision,
        rank,
        float(estimate.shrinkage_),
        float(np.linalg.cond(covariance)),
    )


def compare_subject(
    *,
    subject,
    data,
    metadata,
    labels,
    trial_ids,
    times,
    conditions,
    models: Mapping[str, str],
    nuisance,
    weights,
    groups,
    folds,
    seed,
):
    """Estimate paired subspace energies; pair products are not inferential replicates."""
    labels = np.asarray(labels, dtype=object)
    if data.ndim != 3 or data.shape[0] != len(metadata) or not np.isfinite(data).all():
        raise ValueError(
            "Comparison EEG must be finite (trials, channels, times), aligned to metadata."
        )
    transform, projectors, designs = build_projectors(
        metadata, labels, conditions, models, nuisance, weights
    )
    assignments, partitions = partition_trials(
        labels, groups=groups, folds=folds, seed=_subject_seed(seed, subject)
    )
    means = {
        group: np.stack(
            [data[(assignments == group) & (labels == c)].mean(axis=0) for c in conditions]
        )
        for group in partitions
    }
    pairs, diagnostics = [], []
    for first, second in combinations(partitions, 2):
        train = (assignments != first) & (assignments != second)
        residuals = data[train].copy()
        for c in conditions:
            mask = labels[train] == c
            residuals[mask] -= residuals[mask].mean(axis=0, keepdims=True)
        degrees = int(train.sum()) - len(conditions)
        if degrees < 2:
            raise ValueError(
                f"Groups excluding {first!r}, {second!r} leave fewer than two "
                "within-condition noise degrees of freedom; more repetitions/groups are needed."
            )
        for ti, time in enumerate(times):
            precision, rank, shrinkage, condition_number = noise_precision(
                residuals[:, :, ti], degrees
            )
            z_first = transform @ means[first][:, :, ti]
            z_second = transform @ means[second][:, :, ti]
            normalized_second = z_second @ precision
            for model, projector in projectors.items():
                strength = np.sum((projector @ z_first) * normalized_second)
                pairs.append(
                    {
                        "group_a": first,
                        "group_b": second,
                        "time": int(time),
                        "model": model,
                        "strength": float(strength),
                    }
                )
            diagnostics.append(
                {
                    "group_a": first,
                    "group_b": second,
                    "time": int(time),
                    "n_noise_trials": int(train.sum()),
                    "noise_df": degrees,
                    "channel_rank": rank,
                    "n_channels": data.shape[1],
                    "shrinkage": shrinkage,
                    "condition_number": condition_number,
                }
            )
    diagnostics = pd.DataFrame(diagnostics)
    if diagnostics.noise_df.min() < data.shape[1]:
        warnings.warn(
            f"Subject {subject}: complement noise df range "
            f"{diagnostics.noise_df.min()}..{diagnostics.noise_df.max()} is below "
            f"{data.shape[1]} retained channels for at least one pair, so noise "
            "information may be sample-limited. Full-channel shrinkage fills "
            "sample-null directions; population precision and model "
            "rankings may remain uncertain. Inspect covariance_diagnostics.",
            RuntimeWarning,
            stacklevel=2,
        )
    pairs = pd.DataFrame(pairs)
    strengths = pairs.groupby(["time", "model"], sort=False)["strength"].mean().reset_index()
    strengths["model_rank"] = strengths["model"].map(
        {name: int(round(np.trace(p))) for name, p in projectors.items()}
    )
    comparisons = []
    for first, second in combinations(models, 2):
        p1, p2 = projectors[first], projectors[second]
        nested = np.allclose(p2 @ p1, p1, atol=1e-10, rtol=1e-10)
        same = np.allclose(p1, p2, atol=1e-10, rtol=1e-10)
        for time in times:
            values = strengths[strengths["time"] == time].set_index("model")["strength"]
            comparisons.append(
                {
                    "time": int(time),
                    "model_1": first,
                    "model_2": second,
                    "delta_strength": values[second] - values[first],
                    "nested": bool(nested),
                    "same_span": bool(same),
                }
            )
    tables = {
        "trial_groups": pd.DataFrame(
            {"epoch_index": trial_ids, "condition": labels, "group_label": assignments}
        ),
        "model_designs": designs,
        "pair_strengths": pairs,
        "strengths": strengths,
        "comparisons": pd.DataFrame(comparisons),
        "covariance_diagnostics": diagnostics,
    }
    for frame in tables.values():
        frame.insert(0, "subject_index", subject)
    return tables
