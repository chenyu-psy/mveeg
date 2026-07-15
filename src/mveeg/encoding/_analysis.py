"""Per-subject fitting for the regularized component encoding model."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from ._metrics import compute_pattern_expression, estimate_channel_covariance, fit_ridge
from ._prepare import EncodingDesign, build_design


@dataclass
class SubjectEncoding:
    tables: dict[str, pd.DataFrame]
    predictors: pd.DataFrame


def encode_subject(
    *,
    subject: str,
    data: np.ndarray,
    metadata: pd.DataFrame,
    training_labels: np.ndarray,
    expression_labels: np.ndarray,
    trial_ids: np.ndarray,
    times: np.ndarray,
    condition_order: list[str],
    formula: str,
    penalty: dict[str, float],
    folds: int,
    seed: int,
    n_jobs: int = 1,
    progress: bool = True,
) -> SubjectEncoding:
    """Fit one subject and return public result rows."""

    data = np.asarray(data, dtype=float)
    training_labels = np.asarray(training_labels, dtype=object)
    expression_labels = np.asarray(expression_labels, dtype=object)
    trial_ids = np.asarray(trial_ids, dtype=int)
    times = np.asarray(times, dtype=int)
    if data.ndim != 3 or data.shape[0] != len(metadata):
        raise ValueError("data and metadata must contain the same trials.")
    if len(training_labels) != len(metadata) or len(expression_labels) != len(metadata):
        raise ValueError("Trial role labels must align with metadata.")
    if len(trial_ids) != len(metadata) or data.shape[2] != len(times):
        raise ValueError("Trial IDs or time bins do not align with data.")

    design = build_design(
        metadata,
        formula=formula,
        training_labels=training_labels,
        condition_order=condition_order,
        penalty=penalty,
    )
    training = pd.notna(training_labels)
    counts = pd.Series(training_labels[training]).value_counts()
    missing = [condition for condition in condition_order if condition not in counts]
    too_small = {label: int(count) for label, count in counts.items() if count < folds}
    if missing or too_small:
        raise ValueError(
            f"Every training condition needs at least {folds} trials; "
            f"missing={missing}, counts_below_folds={too_small}."
        )

    subject_seed = _subject_seed(seed, subject)
    split = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=subject_seed,
    )
    training_rows = np.flatnonzero(training)
    expression_only = np.flatnonzero(~training)
    rng = np.random.default_rng(subject_seed)
    expression_assignments = np.array_split(rng.permutation(expression_only), folds)
    fold_specs = []
    for fold_index, (train_local, test_local) in enumerate(
        split.split(training_rows, training_labels[training]),
        start=1,
    ):
        fold_specs.append(
            (
                fold_index,
                training_rows[train_local],
                np.concatenate([training_rows[test_local], expression_assignments[fold_index - 1]]),
            )
        )

    generator = Parallel(
        n_jobs=min(n_jobs, folds),
        return_as="generator",
    )(
        delayed(_fit_fold)(
            subject=subject,
            fold=fold,
            train_rows=train_rows,
            test_rows=test_rows,
            data=data,
            metadata=metadata,
            design=design,
            expression_labels=expression_labels,
            trial_ids=trial_ids,
            times=times,
        )
        for fold, train_rows, test_rows in fold_specs
    )
    fold_results = list(
        tqdm(
            generator,
            total=folds,
            desc=f"Encoding sub-{subject}",
            unit="fold",
            disable=not progress,
        )
    )

    full_rows = np.flatnonzero(training)
    full_beta = fit_ridge(
        data[full_rows],
        design.matrix[full_rows],
        design.predictors["penalty"].to_numpy(dtype=float),
    )
    coefficients = _coefficient_table(
        subject=subject,
        beta=full_beta,
        predictor_names=design.predictors["predictor"].tolist(),
        channel_count=data.shape[1],
        times=times,
    )
    diagnostics = _design_diagnostics(
        subject=subject,
        fold=0,
        rows=full_rows,
        metadata=metadata,
        design=design,
    )
    for result in fold_results:
        diagnostics.extend(result["design_diagnostics"])

    return SubjectEncoding(
        tables={
            "coefficients": coefficients,
            "pattern_expression": pd.concat(
                [result["pattern_expression"] for result in fold_results],
                ignore_index=True,
            )
            .sort_values(["subject_index", "epoch_index", "component", "time"])
            .reset_index(drop=True),
            "design_diagnostics": pd.DataFrame(diagnostics),
            "covariance_diagnostics": pd.concat(
                [result["covariance_diagnostics"] for result in fold_results],
                ignore_index=True,
            ),
        },
        predictors=design.predictors,
    )


def _fit_fold(
    *,
    subject: str,
    fold: int,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    data: np.ndarray,
    metadata: pd.DataFrame,
    design: EncodingDesign,
    expression_labels: np.ndarray,
    trial_ids: np.ndarray,
    times: np.ndarray,
) -> dict[str, object]:
    train_design = design.matrix[train_rows]
    beta = fit_ridge(
        data[train_rows],
        train_design,
        design.predictors["penalty"].to_numpy(dtype=float),
    )
    train_mean = data[train_rows].mean(axis=0, keepdims=True)
    precision = np.empty((data.shape[2], data.shape[1], data.shape[1]))
    covariance_rows = []
    for time_index, time in enumerate(times):
        residuals = data[train_rows, :, time_index] - train_design @ beta[:, :, time_index]
        estimate = estimate_channel_covariance(residuals)
        precision[time_index] = estimate.precision
        covariance_rows.append(
            {
                "subject_index": subject,
                "fold": fold,
                "time": int(time),
                "n_train_trials": estimate.n_train_trials,
                "n_channels": estimate.n_channels,
                "rank": estimate.rank,
                "condition_number": estimate.condition_number,
                "log_determinant": estimate.log_determinant,
                "shrinkage": estimate.shrinkage,
                "status": estimate.status,
            }
        )

    expression, warnings = compute_pattern_expression(
        data[test_rows] - train_mean,
        beta[np.asarray(design.component_indices)],
        precision,
    )
    diagnostics = _design_diagnostics(
        subject=subject,
        fold=fold,
        rows=train_rows,
        metadata=metadata,
        design=design,
    )
    diagnostics.extend(
        {
            "subject_index": subject,
            "fold": fold,
            "diagnostic": warning["status"],
            "predictor": design.components[int(warning["component_index"])],
            "value": float(warning["time_index"]),
            "threshold": np.nan,
            "status": "warning",
            "message": "expression denominator was too small",
        }
        for warning in warnings
    )
    return {
        "pattern_expression": _expression_table(
            subject=subject,
            fold=fold,
            rows=test_rows,
            trial_ids=trial_ids,
            expression_labels=expression_labels,
            components=list(design.components),
            times=times,
            expression=expression,
        ),
        "design_diagnostics": diagnostics,
        "covariance_diagnostics": pd.DataFrame(covariance_rows),
    }


def _expression_table(
    *,
    subject: str,
    fold: int,
    rows: np.ndarray,
    trial_ids: np.ndarray,
    expression_labels: np.ndarray,
    components: list[str],
    times: np.ndarray,
    expression: np.ndarray,
) -> pd.DataFrame:
    tables = []
    for component_index, component in enumerate(components):
        tables.append(
            pd.DataFrame(
                {
                    "subject_index": np.repeat(subject, len(rows) * len(times)),
                    "epoch_index": np.repeat(trial_ids[rows], len(times)),
                    "time": np.tile(times, len(rows)),
                    "component": np.repeat(component, len(rows) * len(times)),
                    "expression_group": np.repeat(expression_labels[rows], len(times)),
                    "expression": expression[:, component_index, :].reshape(-1),
                    "fold": np.repeat(fold, len(rows) * len(times)),
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def _coefficient_table(
    *,
    subject: str,
    beta: np.ndarray,
    predictor_names: list[str],
    channel_count: int,
    times: np.ndarray,
) -> pd.DataFrame:
    tables = []
    for predictor_index, predictor in enumerate(predictor_names):
        tables.append(
            pd.DataFrame(
                {
                    "subject_index": np.repeat(subject, channel_count * len(times)),
                    "time": np.tile(times, channel_count),
                    "channel_index": np.repeat(np.arange(channel_count), len(times)),
                    "predictor": np.repeat(predictor, channel_count * len(times)),
                    "beta": beta[predictor_index].reshape(-1),
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def _design_diagnostics(
    *,
    subject: str,
    fold: int,
    rows: np.ndarray,
    metadata: pd.DataFrame,
    design: EncodingDesign,
) -> list[dict[str, object]]:
    matrix = design.matrix[rows]
    component_columns = 1 + len(design.components)
    diagnostics = []
    for name, values in (
        ("component", matrix[:, :component_columns]),
        ("model", matrix),
    ):
        rank = int(np.linalg.matrix_rank(values))
        columns = int(values.shape[1])
        diagnostics.extend(
            [
                {
                    "subject_index": subject,
                    "fold": fold,
                    "diagnostic": f"{name}_rank",
                    "predictor": None,
                    "value": float(rank),
                    "threshold": float(columns),
                    "status": "ok" if rank == columns else "regularized",
                    "message": "" if rank == columns else "rank deficiency handled by ridge",
                },
                {
                    "subject_index": subject,
                    "fold": fold,
                    "diagnostic": f"{name}_condition_number",
                    "predictor": None,
                    "value": float(np.linalg.cond(values)),
                    "threshold": 1e8,
                    "status": "warning" if np.linalg.cond(values) > 1e8 else "ok",
                    "message": "",
                },
            ]
        )
    for interaction in design.interactions:
        factors = interaction.split(":")
        for values, count in metadata.iloc[rows].groupby(factors, dropna=False).size().items():
            values = values if isinstance(values, tuple) else (values,)
            diagnostics.append(
                {
                    "subject_index": subject,
                    "fold": fold,
                    "diagnostic": "interaction_cell_n_trials",
                    "predictor": interaction,
                    "value": float(count),
                    "threshold": 20.0,
                    "status": "warning" if count < 20 else "ok",
                    "message": ",".join(
                        f"{factor}={value}" for factor, value in zip(factors, values)
                    ),
                }
            )
    return diagnostics


def _subject_seed(seed: int, subject: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{subject}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "little")
