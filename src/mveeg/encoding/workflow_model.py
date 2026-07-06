"""Lower-level model-fitting workflow helpers for encoding analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .._shared.io_filters import load_subject_data_with_filters
from .._shared.time_windows import average_time_windows, build_time_windows
from .._shared.workflow_subjects import process_subjects
from .config import EncodingConfig
from .io import README_FILENAME, save_encoding_model_result
from .metrics import (
    compute_pattern_expression,
    compute_prediction_metrics,
    estimate_channel_covariance,
)
from .run import fit_time_resolved_multivariate_ols
from .summaries import (
    build_condition_pattern_expression_table,
    build_effect_slope_table,
    build_pattern_expression_trial_table,
)
from .workflow_design import build_formula_condition_encoding, run_encoding_design_check
from .workflow_outputs import build_encoding_topography_outputs, export_encoding_model_outputs


MODEL_OUTPUT_FILES = {
    "pattern_expression_trial": "pattern_expression_trial.csv",
    "condition_pattern_expression": "condition_pattern_expression.csv",
    "effect_slope": "effect_slope.csv",
    "design_diagnostics": "design_diagnostics.csv",
    "covariance_diagnostics": "covariance_diagnostics.csv",
    "subject_summary": "subject_summary.csv",
    "run_summary": "run_summary.csv",
    "skipped_subjects": "skipped_subjects.csv",
    "topography_values": "topography/topography_values.csv",
    "topography_coords": "topography/topography_coords.csv",
    "readme": README_FILENAME,
}

MODEL_COMPARISON_OUTPUT_FILES = {
    "model_comparison": "model_comparison.csv",
    "design_diagnostics": "design_diagnostics.csv",
    "covariance_diagnostics": "covariance_diagnostics.csv",
    "run_summary": "run_summary.csv",
    "skipped_subjects": "skipped_subjects.csv",
    "readme": README_FILENAME,
}


def _standardize_trials_by_train_stats(
    *,
    train_data: np.ndarray,
    test_data: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score EEG data using fold-specific train statistics.

    Parameters
    ----------
    train_data : np.ndarray
        Training EEG array with shape ``(n_trials, n_channels, n_times)``.
    test_data : np.ndarray
        Held-out EEG array with shape ``(n_trials, n_channels, n_times)``.
    tol : float
        Minimum standard deviation required before division.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Standardized training and held-out arrays.
    """

    if train_data.ndim != 3 or test_data.ndim != 3:
        raise ValueError("train_data and test_data must be 3D arrays.")
    if train_data.shape[1:] != test_data.shape[1:]:
        raise ValueError(
            "train_data and test_data must have matching channel/time dimensions."
        )

    train_mean = np.mean(train_data, axis=0, keepdims=True)
    train_std = np.std(train_data, axis=0, keepdims=True)
    safe_std = np.where(train_std > tol, train_std, 1.0)
    return (train_data - train_mean) / safe_std, (test_data - train_mean) / safe_std


def _load_subject_arrays(
    *,
    subject_id: str,
    loader_cfg,
    source_condition_col: str,
    source_to_condition: dict[str, str],
    time_window_ms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Load one subject's EEG, time axis, channels, and mapped conditions."""

    data, _, times_s, ch_names, metadata = load_subject_data_with_filters(
        subject_id=subject_id,
        cfg=loader_cfg,
        return_metadata=True,
    )
    window_centers_ms, window_masks = build_time_windows(
        np.asarray(times_s, dtype=float),
        int(time_window_ms),
    )
    data = average_time_windows(data, window_masks)
    times_s = window_centers_ms.astype(float) / 1000.0

    if source_condition_col not in metadata.columns:
        raise ValueError(
            f"Metadata is missing required condition column '{source_condition_col}'."
        )
    observed_source_labels = set(
        metadata[source_condition_col].dropna().astype(str).unique()
    )
    unknown_source_labels = sorted(observed_source_labels.difference(source_to_condition))
    if len(unknown_source_labels) > 0:
        raise ValueError(
            f"Found labels in '{source_condition_col}' missing from source_to_condition: "
            f"{unknown_source_labels}"
        )

    metadata = metadata.copy()
    metadata["condition"] = metadata[source_condition_col].astype(str).map(
        source_to_condition
    )
    condition_values = metadata["condition"].to_numpy(dtype=object).astype(str)
    return data, condition_values, times_s, list(ch_names), metadata


def _build_subject_design(
    *,
    condition_values: np.ndarray,
    condition_encoding: pd.DataFrame,
    glm_formula: str,
    design_cfg: EncodingConfig,
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, object], object]:
    """Build and validate one subject's formula-selected design."""

    formula_condition_encoding, parsed_formula = build_formula_condition_encoding(
        condition_encoding,
        glm_formula,
    )
    effective_design_cfg = EncodingConfig(
        add_intercept=bool(parsed_formula["add_intercept"]),
        validation_mode=design_cfg.validation_mode,
        tolerance=design_cfg.tolerance,
    )
    design_output = run_encoding_design_check(
        trial_conditions=condition_values,
        condition_encoding=formula_condition_encoding,
        cfg=effective_design_cfg,
    )
    validation = design_output["validation"]
    if validation.rank < validation.n_cols:
        raise ValueError(
            f"Design matrix is rank deficient (rank={validation.rank}, "
            f"columns={validation.n_cols})."
        )
    if not validation.is_valid:
        raise ValueError("; ".join(validation.messages))

    return (
        design_output["design_matrix"],
        list(design_output["design_names"]),
        design_output["trial_encoding"],
        parsed_formula,
        validation,
    )


def _condition_cv(
    *,
    condition_values: np.ndarray,
    cv_n_splits: int,
    cv_shuffle: bool,
    cv_random_state: int,
) -> StratifiedKFold:
    """Return a stratified K-fold splitter after checking condition counts."""

    condition_counts = pd.Series(condition_values).value_counts()
    insufficient_conditions = condition_counts[condition_counts < cv_n_splits]
    if len(insufficient_conditions) > 0:
        raise ValueError(
            "Not enough trials per condition for stratified CV. "
            f"Need at least {cv_n_splits} trials each, found: "
            f"{insufficient_conditions.to_dict()}"
        )

    return StratifiedKFold(
        n_splits=cv_n_splits,
        shuffle=cv_shuffle,
        random_state=cv_random_state if cv_shuffle else None,
    )


def _design_diagnostic_rows(
    *,
    subject: str,
    model: str,
    validation,
    design_matrix: np.ndarray,
    design_names: list[str],
) -> list[dict[str, object]]:
    """Build design-level diagnostic rows for one subject/model."""

    rows = [
        {
            "subject": str(subject),
            "model": str(model),
            "fold": np.nan,
            "diagnostic": "rank_X",
            "effect": "",
            "value": float(validation.rank),
            "threshold": float(validation.n_cols),
            "status": "ok",
            "message": "",
        },
        {
            "subject": str(subject),
            "model": str(model),
            "fold": np.nan,
            "diagnostic": "condition_number_X",
            "effect": "",
            "value": float(validation.condition_number),
            "threshold": 30.0,
            "status": "warning" if validation.condition_number > 30 else "ok",
            "message": "high design condition number"
            if validation.condition_number > 30
            else "",
        },
    ]
    for col_ix, col_name in enumerate(design_names):
        if col_name == "intercept":
            continue
        col_sd = float(np.std(design_matrix[:, col_ix]))
        rows.append(
            {
                "subject": str(subject),
                "model": str(model),
                "fold": np.nan,
                "diagnostic": "predictor_sd",
                "effect": str(col_name),
                "value": col_sd,
                "threshold": 0.0,
                "status": "warning" if col_sd == 0 else "ok",
                "message": "zero-variance predictor" if col_sd == 0 else "",
            }
        )
    return rows


def _interaction_diagnostic_rows(
    *,
    subject: str,
    model: str,
    fold_id: int | None,
    trial_encoding: pd.DataFrame,
    trial_indices: np.ndarray,
    interactions: list[str],
    min_trials_per_cell: int,
) -> list[dict[str, object]]:
    """Build interaction-cell trial count diagnostics."""

    rows = []
    if len(interactions) == 0:
        return rows
    trial_df = trial_encoding.iloc[trial_indices, :].copy()
    for interaction in interactions:
        factors = interaction.split(":")
        cell_counts = (
            trial_df.groupby(factors, dropna=False)
            .size()
            .reset_index(name="n_trials")
        )
        for _, row in cell_counts.iterrows():
            cell_label = ",".join(f"{factor}={row[factor]}" for factor in factors)
            n_trials = int(row["n_trials"])
            rows.append(
                {
                    "subject": str(subject),
                    "model": str(model),
                    "fold": np.nan if fold_id is None else int(fold_id),
                    "diagnostic": "interaction_cell_n_trials",
                    "effect": str(interaction),
                    "value": float(n_trials),
                    "threshold": float(min_trials_per_cell),
                    "status": "warning" if n_trials < min_trials_per_cell else "ok",
                    "message": cell_label,
                }
            )
    return rows


def _covariance_rows(
    *,
    subject: str,
    model: str,
    fold_id: int,
    times_s: np.ndarray,
    covariance_method: str,
    train_data: np.ndarray,
    train_design: np.ndarray,
    betas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Estimate covariance per time bin and return matrices plus diagnostics."""

    n_times = train_data.shape[2]
    n_channels = train_data.shape[1]
    precision_matrices = np.empty((n_times, n_channels, n_channels), dtype=float)
    covariance_matrices = np.empty((n_times, n_channels, n_channels), dtype=float)
    log_determinants = np.empty(n_times, dtype=float)
    rows = []

    for time_ix in range(n_times):
        prediction = train_design @ betas[:, :, time_ix]
        residuals = train_data[:, :, time_ix] - prediction
        estimate = estimate_channel_covariance(
            residuals,
            method=covariance_method,
        )
        precision_matrices[time_ix, :, :] = estimate.precision
        covariance_matrices[time_ix, :, :] = estimate.covariance
        log_determinants[time_ix] = estimate.log_determinant
        rows.append(
            {
                "subject": str(subject),
                "model": str(model),
                "fold": int(fold_id),
                "time_ms": float(times_s[time_ix] * 1000.0),
                "covariance_method": estimate.method,
                "n_train_trials": estimate.n_train_trials,
                "n_channels": estimate.n_channels,
                "rank": estimate.rank,
                "condition_number": estimate.condition_number,
                "log_determinant": estimate.log_determinant,
                "shrinkage_value": estimate.shrinkage_value,
                "status": estimate.status,
            }
        )

    return precision_matrices, covariance_matrices, log_determinants, rows


def _fit_one_encoding_subject(
    *,
    subject_id: str,
    subject_results_dir: Path | None,
    loader_cfg,
    source_condition_col: str,
    source_to_condition: dict[str, str],
    train_condition_labels: tuple[str, ...] | None,
    condition_encoding: pd.DataFrame,
    design_cfg: EncodingConfig,
    glm_formula: str,
    covariance: str,
    cv_n_splits: int,
    cv_shuffle: bool,
    cv_random_state: int,
    time_window_ms: int,
    standardize_data: bool,
    progress_bar,
) -> dict[str, object]:
    """Fit one subject's cross-validated pattern-expression model."""

    progress_bar.set_postfix_str("loading input")
    data, condition_values, times_s, ch_names, _metadata = _load_subject_arrays(
        subject_id=subject_id,
        loader_cfg=loader_cfg,
        source_condition_col=source_condition_col,
        source_to_condition=source_to_condition,
        time_window_ms=time_window_ms,
    )
    progress_bar.set_postfix_str("preparing")
    design_matrix, design_names, trial_encoding, parsed_formula, validation = (
        _build_subject_design(
            condition_values=condition_values,
            condition_encoding=condition_encoding,
            glm_formula=glm_formula,
            design_cfg=design_cfg,
        )
    )

    if train_condition_labels is None:
        train_condition_mask = np.ones(len(condition_values), dtype=bool)
    else:
        missing_train_labels = sorted(
            set(train_condition_labels).difference(condition_values)
        )
        if len(missing_train_labels) > 0:
            raise ValueError(
                "train_condition_labels includes labels missing from loaded trials: "
                f"{missing_train_labels}"
            )
        train_condition_mask = np.isin(condition_values, train_condition_labels)

    cv = _condition_cv(
        condition_values=condition_values,
        cv_n_splits=cv_n_splits,
        cv_shuffle=cv_shuffle,
        cv_random_state=cv_random_state,
    )
    effect_names = [name for name in design_names if name != "intercept"]
    effect_indices = [design_names.index(name) for name in effect_names]

    raw_beta_patterns = np.empty(
        (cv_n_splits, len(design_names), data.shape[1], data.shape[2]),
        dtype=float,
    )
    trial_expression_tables = []
    covariance_rows = []
    design_rows = _design_diagnostic_rows(
        subject=subject_id,
        model="encoding",
        validation=validation,
        design_matrix=design_matrix,
        design_names=design_names,
    )
    design_rows.extend(
        _interaction_diagnostic_rows(
            subject=subject_id,
            model="encoding",
            fold_id=None,
            trial_encoding=trial_encoding,
            trial_indices=np.arange(len(trial_encoding)),
            interactions=parsed_formula["interactions"],
            min_trials_per_cell=20,
        )
    )

    for fold_ix, (train_idx, test_idx) in enumerate(
        cv.split(np.zeros(len(condition_values)), condition_values),
        start=0,
    ):
        fold_id = fold_ix + 1
        progress_bar.set_postfix_str(f"fold {fold_id}/{cv_n_splits}")
        if len(test_idx) == 0:
            raise ValueError("Each cross-validation fold must include held-out trials.")
        train_model_idx = train_idx[train_condition_mask[train_idx]]
        if len(train_model_idx) <= len(design_names):
            raise ValueError(
                "Too few training trials for the fitted encoding model. "
                f"Need more than {len(design_names)}, got {len(train_model_idx)}."
            )

        train_data = data[train_model_idx, :, :]
        test_data = data[test_idx, :, :]
        if standardize_data:
            train_data, test_data = _standardize_trials_by_train_stats(
                train_data=train_data,
                test_data=test_data,
            )

        train_design = design_matrix[train_model_idx, :]
        fit_result = fit_time_resolved_multivariate_ols(
            data=train_data,
            design_matrix=train_design,
            design_names=design_names,
        )
        betas = fit_result["betas"].astype(float)
        raw_beta_patterns[fold_ix, :, :, :] = betas
        precision_matrices, _covariances, _logdets, fold_cov_rows = _covariance_rows(
            subject=subject_id,
            model="encoding",
            fold_id=fold_id,
            times_s=times_s,
            covariance_method=covariance,
            train_data=train_data,
            train_design=train_design,
            betas=betas,
        )
        covariance_rows.extend(fold_cov_rows)

        expression, denominator_warnings = compute_pattern_expression(
            test_data=test_data,
            beta_patterns=betas[effect_indices, :, :],
            precision_matrices=precision_matrices,
        )
        for warning in denominator_warnings:
            design_rows.append(
                {
                    "subject": str(subject_id),
                    "model": "encoding",
                    "fold": int(fold_id),
                    "diagnostic": warning["status"],
                    "effect": effect_names[int(warning["effect_index"])],
                    "value": float(warning["time_index"]),
                    "threshold": np.nan,
                    "status": "warning",
                    "message": "expression denominator too small",
                }
            )

        trial_expression_tables.append(
            build_pattern_expression_trial_table(
                subject=subject_id,
                fold_id=fold_id,
                condition_labels=condition_values[test_idx],
                trial_index=test_idx,
                times_s=times_s,
                effect_names=effect_names,
                expression=expression,
                covariance_method=covariance,
            )
        )
        design_rows.extend(
            _interaction_diagnostic_rows(
                subject=subject_id,
                model="encoding",
                fold_id=fold_id,
                trial_encoding=trial_encoding,
                trial_indices=test_idx,
                interactions=parsed_formula["interactions"],
                min_trials_per_cell=5,
            )
        )
        progress_bar.update(1)

    pattern_expression_trial_df = pd.concat(trial_expression_tables, ignore_index=True)
    condition_pattern_expression_df = build_condition_pattern_expression_table(
        pattern_expression_trial_df
    )
    effect_slope_df = build_effect_slope_table(
        subject=subject_id,
        trial_expression_df=pattern_expression_trial_df,
        trial_design=trial_encoding,
        effect_names=effect_names,
        times_s=times_s,
        covariance_method=covariance,
    )

    payload = {
        "subject": np.asarray(subject_id, dtype=object),
        "times_s": np.asarray(times_s, dtype=float),
        "ch_names": np.asarray(ch_names, dtype=object),
        "n_trials": np.asarray(data.shape[0], dtype=int),
        "n_channels": np.asarray(data.shape[1], dtype=int),
        "n_times": np.asarray(data.shape[2], dtype=int),
        "n_folds": np.asarray(cv_n_splits, dtype=int),
        "time_window_ms": np.asarray(int(time_window_ms), dtype=int),
        "condition_levels": np.asarray(sorted(np.unique(condition_values).tolist()), dtype=object),
        "standardize_data": np.asarray(bool(standardize_data), dtype=bool),
        "covariance_method": np.asarray(str(covariance), dtype=object),
        "predictor_names": np.asarray(design_names, dtype=object),
        "raw_beta_patterns": raw_beta_patterns,
    }
    if subject_results_dir is not None:
        save_encoding_model_result(
            output_dir=subject_results_dir,
            subject_id=subject_id,
            payload=payload,
        )

    return {
        "payload": payload,
        "pattern_expression_trial_df": pattern_expression_trial_df,
        "condition_pattern_expression_df": condition_pattern_expression_df,
        "effect_slope_df": effect_slope_df,
        "design_diagnostics_df": pd.DataFrame(design_rows),
        "covariance_diagnostics_df": pd.DataFrame(covariance_rows),
    }


def run_encoding_workflow(
    *,
    subject_ids: list[str],
    subject_results_dir: str | Path | None = None,
    loader_cfg,
    condition_encoding: pd.DataFrame,
    design_cfg: EncodingConfig,
    glm_formula: str,
    source_to_condition: dict[str, str] | None = None,
    train_condition_labels: list[str] | tuple[str, ...] | None = None,
    overwrite: bool = False,
    cv_n_splits: int = 5,
    cv_shuffle: bool = True,
    cv_random_state: int = 42,
    time_window_ms: int = 50,
    standardize_data: bool = True,
    covariance: str = "shrinkage",
    results_dir: str | Path | None = None,
    run_name: str = "encoding_model",
    config_payload: dict[str, object] | None = None,
    topography: dict[str, object] | None = None,
    log_path: str | Path | None = None,
) -> dict[str, object]:
    """Run cross-validated pattern-expression encoding workflow.

    Parameters
    ----------
    subject_ids : list[str]
        Subject identifiers to process.
    subject_results_dir : str | Path | None
        Optional folder for subject-level beta-pattern payloads.
    loader_cfg : object
        Subject-loading config.
    condition_encoding : pd.DataFrame
        Condition-level design table.
    design_cfg : EncodingConfig
        Design validation settings.
    glm_formula : str
        Formula selecting condition-level predictors and simple interactions.
    source_to_condition : dict[str, str] | None
        Mapping from raw metadata labels to analysis condition labels.
    train_condition_labels : list[str] | tuple[str, ...] | None
        Optional analysis condition labels used for model fitting.
    overwrite : bool
        Present for workflow consistency. Subject beta payloads are rebuilt in
        this no-compatibility version.
    cv_n_splits, cv_shuffle, cv_random_state : int | bool
        Cross-validation settings.
    time_window_ms : int
        Time-bin width in milliseconds.
    standardize_data : bool
        Whether to standardize EEG with training-fold statistics.
    covariance : str
        Covariance method for expression metrics.
    results_dir : str | Path | None
        Optional group-level export directory.
    run_name : str
        Name recorded in the run summary.
    config_payload : dict[str, object] | None
        Optional JSON-serializable configuration.
    topography : dict[str, object] | None
        Optional topography export settings.
    log_path : str | Path | None
        Optional detailed log path.

    Returns
    -------
    dict[str, object]
        Encoding output tables and subject beta payloads.
    """

    _ = overwrite
    if cv_n_splits < 2:
        raise ValueError("cv_n_splits must be at least 2.")
    if time_window_ms < 1:
        raise ValueError("time_window_ms must be at least 1 millisecond.")
    if source_to_condition is None:
        raise ValueError(
            "Provide source_to_condition for mapping raw labels to analysis conditions."
        )

    if subject_results_dir is not None:
        subject_results_dir = Path(subject_results_dir)
        subject_results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(results_dir) if results_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    source_condition_col = loader_cfg.conditions.cond_col
    source_to_condition = {
        str(key): str(value) for key, value in source_to_condition.items()
    }
    if train_condition_labels is not None:
        train_condition_labels = tuple(str(label) for label in train_condition_labels)

    subject_results = {}

    def process_one_subject(subject_id: str, progress_bar):
        result = _fit_one_encoding_subject(
            subject_id=subject_id,
            subject_results_dir=subject_results_dir,
            loader_cfg=loader_cfg,
            source_condition_col=source_condition_col,
            source_to_condition=source_to_condition,
            train_condition_labels=train_condition_labels,
            condition_encoding=condition_encoding,
            design_cfg=design_cfg,
            glm_formula=glm_formula,
            covariance=covariance,
            cv_n_splits=cv_n_splits,
            cv_shuffle=cv_shuffle,
            cv_random_state=cv_random_state,
            time_window_ms=time_window_ms,
            standardize_data=standardize_data,
            progress_bar=progress_bar,
        )
        subject_results[str(subject_id)] = result
        return result, False

    skipped_subjects_df = process_subjects(
        subject_ids=[str(subject_id) for subject_id in subject_ids],
        progress_total=cv_n_splits,
        log_label="Encoding run",
        log_path=log_path,
        experiment_name=loader_cfg.dataset.experiment_name,
        process_one_subject=process_one_subject,
    )

    if len(subject_results) == 0:
        skipped_summary = (
            skipped_subjects_df
            if len(skipped_subjects_df) > 0
            else pd.DataFrame(columns=["subject", "reason"])
        )
        raise RuntimeError(
            "No subjects were successfully processed for encoding.\n"
            f"Failure summary:\n{skipped_summary.to_string(index=False)}"
        )

    subject_payloads = {
        subject_id: result["payload"] for subject_id, result in subject_results.items()
    }
    subject_summary_df = pd.DataFrame(
        [
            {
                "subject": str(payload["subject"].item()),
                "n_trials": int(payload["n_trials"].item()),
                "n_channels": int(payload["n_channels"].item()),
                "n_times": int(payload["n_times"].item()),
                "n_folds": int(payload["n_folds"].item()),
                "condition_levels": ",".join(
                    payload["condition_levels"].astype(str).tolist()
                ),
                "covariance_method": str(payload["covariance_method"].item()),
            }
            for payload in subject_payloads.values()
        ]
    ).sort_values("subject").reset_index(drop=True)
    run_summary_df = pd.DataFrame(
        {
            "name": [run_name],
            "n_subjects_requested": [len(subject_ids)],
            "n_subjects_completed": [len(subject_summary_df)],
            "n_subjects_skipped": [len(skipped_subjects_df)],
            "time_window_ms": [int(time_window_ms)],
            "standardize_data": [bool(standardize_data)],
            "covariance_method": [str(covariance)],
        }
    )

    pattern_expression_trial_df = pd.concat(
        [result["pattern_expression_trial_df"] for result in subject_results.values()],
        ignore_index=True,
    ).sort_values(["effect", "condition", "subject", "fold", "trial_index", "time_ms"]).reset_index(drop=True)
    condition_pattern_expression_df = pd.concat(
        [result["condition_pattern_expression_df"] for result in subject_results.values()],
        ignore_index=True,
    ).sort_values(["effect", "condition", "subject", "time_ms"]).reset_index(drop=True)
    effect_slope_df = pd.concat(
        [result["effect_slope_df"] for result in subject_results.values()],
        ignore_index=True,
    ).sort_values(["effect", "subject", "time_ms"]).reset_index(drop=True)
    design_diagnostics_df = pd.concat(
        [result["design_diagnostics_df"] for result in subject_results.values()],
        ignore_index=True,
    )
    covariance_diagnostics_df = pd.concat(
        [result["covariance_diagnostics_df"] for result in subject_results.values()],
        ignore_index=True,
    )

    if output_dir is None:
        topography_outputs = build_encoding_topography_outputs(
            topography=topography,
            subject_payloads=subject_payloads,
            loader_cfg=loader_cfg,
        )
    else:
        topography_outputs = export_encoding_model_outputs(
            output_dir=output_dir,
            output_files=MODEL_OUTPUT_FILES,
            subject_summary_df=subject_summary_df,
            skipped_subjects_df=skipped_subjects_df,
            run_summary_df=run_summary_df,
            pattern_expression_trial_df=pattern_expression_trial_df,
            condition_pattern_expression_df=condition_pattern_expression_df,
            effect_slope_df=effect_slope_df,
            design_diagnostics_df=design_diagnostics_df,
            covariance_diagnostics_df=covariance_diagnostics_df,
            config_payload=config_payload,
            topography=topography,
            subject_payloads=subject_payloads,
            loader_cfg=loader_cfg,
        )

    return {
        "subject_summary_df": subject_summary_df,
        "skipped_subjects_df": skipped_subjects_df,
        "run_summary_df": run_summary_df,
        "pattern_expression_trial_df": pattern_expression_trial_df,
        "condition_pattern_expression_df": condition_pattern_expression_df,
        "effect_slope_df": effect_slope_df,
        "design_diagnostics_df": design_diagnostics_df,
        "covariance_diagnostics_df": covariance_diagnostics_df,
        "subject_payloads": subject_payloads,
        **topography_outputs,
    }


def _fit_model_comparison_subject(
    *,
    subject_id: str,
    loader_cfg,
    source_condition_col: str,
    source_to_condition: dict[str, str],
    condition_encoding: pd.DataFrame,
    design_cfg: EncodingConfig,
    models: dict[str, str | list[str]],
    reference_model: str,
    covariance: str,
    cv_n_splits: int,
    cv_shuffle: bool,
    cv_random_state: int,
    time_window_ms: int,
    standardize_data: bool,
    progress_bar,
) -> dict[str, pd.DataFrame]:
    """Compare one subject's candidate encoding models by held-out prediction."""

    progress_bar.set_postfix_str("loading input")
    data, condition_values, times_s, _ch_names, _metadata = _load_subject_arrays(
        subject_id=subject_id,
        loader_cfg=loader_cfg,
        source_condition_col=source_condition_col,
        source_to_condition=source_to_condition,
        time_window_ms=time_window_ms,
    )
    cv = _condition_cv(
        condition_values=condition_values,
        cv_n_splits=cv_n_splits,
        cv_shuffle=cv_shuffle,
        cv_random_state=cv_random_state,
    )

    model_specs = {
        str(model_name): _model_spec_to_formula(model_spec)
        for model_name, model_spec in models.items()
    }
    model_designs = {}
    design_rows = []
    for model_name, formula in model_specs.items():
        design_matrix, design_names, trial_encoding, parsed_formula, validation = (
            _build_subject_design(
                condition_values=condition_values,
                condition_encoding=condition_encoding,
                glm_formula=formula,
                design_cfg=design_cfg,
            )
        )
        model_designs[model_name] = {
            "formula": formula,
            "design_matrix": design_matrix,
            "design_names": design_names,
            "trial_encoding": trial_encoding,
            "parsed_formula": parsed_formula,
            "validation": validation,
        }
        design_rows.extend(
            _design_diagnostic_rows(
                subject=subject_id,
                model=model_name,
                validation=validation,
                design_matrix=design_matrix,
                design_names=design_names,
            )
        )
        design_rows.extend(
            _interaction_diagnostic_rows(
                subject=subject_id,
                model=model_name,
                fold_id=None,
                trial_encoding=trial_encoding,
                trial_indices=np.arange(len(trial_encoding)),
                interactions=parsed_formula["interactions"],
                min_trials_per_cell=20,
            )
        )

    metric_rows = []
    covariance_rows = []
    for fold_ix, (train_idx, test_idx) in enumerate(
        cv.split(np.zeros(len(condition_values)), condition_values),
        start=0,
    ):
        fold_id = fold_ix + 1
        if len(test_idx) == 0:
            raise ValueError("Each cross-validation fold must include held-out trials.")
        base_train_data = data[train_idx, :, :]
        base_test_data = data[test_idx, :, :]
        if standardize_data:
            base_train_data, base_test_data = _standardize_trials_by_train_stats(
                train_data=base_train_data,
                test_data=base_test_data,
            )

        fold_metrics = []
        for model_name, model_info in model_designs.items():
            progress_bar.set_postfix_str(
                f"fold {fold_id}/{cv_n_splits}, {model_name}"
            )
            design_matrix = model_info["design_matrix"]
            design_names = model_info["design_names"]
            if len(train_idx) <= len(design_names):
                raise ValueError(
                    "Too few training trials for the fitted encoding model. "
                    f"Need more than {len(design_names)}, got {len(train_idx)}."
                )
            train_design = design_matrix[train_idx, :]
            test_design = design_matrix[test_idx, :]
            fit_result = fit_time_resolved_multivariate_ols(
                data=base_train_data,
                design_matrix=train_design,
                design_names=design_names,
            )
            betas = fit_result["betas"].astype(float)
            (
                precision_matrices,
                _covariance_matrices,
                log_determinants,
                fold_cov_rows,
            ) = _covariance_rows(
                subject=subject_id,
                model=model_name,
                fold_id=fold_id,
                times_s=times_s,
                covariance_method=covariance,
                train_data=base_train_data,
                train_design=train_design,
                betas=betas,
            )
            covariance_rows.extend(fold_cov_rows)

            model_info_fold = []
            for time_ix, time_value in enumerate(times_s):
                prediction = test_design @ betas[:, :, time_ix]
                metrics = compute_prediction_metrics(
                    test_data=base_test_data[:, :, time_ix],
                    prediction=prediction,
                    train_mean=base_train_data[:, :, time_ix].mean(axis=0),
                    precision=precision_matrices[time_ix, :, :],
                    log_determinant=log_determinants[time_ix],
                )
                model_info_fold.append(
                    {
                        "subject": str(subject_id),
                        "model": str(model_name),
                        "reference_model": str(reference_model),
                        "fold": int(fold_id),
                        "time_ms": float(time_value * 1000.0),
                        **metrics,
                        "n_train_trials": int(len(train_idx)),
                        "n_test_trials": int(len(test_idx)),
                        "n_predictors": int(len(design_names)),
                        "rank_X": int(model_info["validation"].rank),
                        "condition_number_X": float(
                            model_info["validation"].condition_number
                        ),
                        "covariance_method": str(covariance),
                    }
                )
            fold_metrics.extend(model_info_fold)

            design_rows.extend(
                _interaction_diagnostic_rows(
                    subject=subject_id,
                    model=model_name,
                    fold_id=fold_id,
                    trial_encoding=model_info["trial_encoding"],
                    trial_indices=test_idx,
                    interactions=model_info["parsed_formula"]["interactions"],
                    min_trials_per_cell=5,
                )
            )

        metric_rows.extend(_add_model_deltas(fold_metrics, reference_model))
        progress_bar.update(1)

    model_comparison_df = (
        pd.DataFrame(metric_rows)
        .groupby(["subject", "model", "reference_model", "time_ms"], as_index=False)
        .agg(
            cv_r2=("cv_r2", "mean"),
            wmse=("wmse", "mean"),
            wcv_r2=("wcv_r2", "mean"),
            heldout_loglik_total=("heldout_loglik_total", "sum"),
            heldout_loglik_mean=("heldout_loglik_mean", "mean"),
            delta_cv_r2=("delta_cv_r2", "mean"),
            delta_wmse=("delta_wmse", "mean"),
            delta_wcv_r2=("delta_wcv_r2", "mean"),
            delta_loglik=("delta_loglik", "sum"),
            n_train_trials=("n_train_trials", "sum"),
            n_test_trials=("n_test_trials", "sum"),
            n_predictors=("n_predictors", "first"),
            rank_X=("rank_X", "first"),
            condition_number_X=("condition_number_X", "first"),
            covariance_method=("covariance_method", "first"),
        )
        .sort_values(["model", "subject", "time_ms"])
        .reset_index(drop=True)
    )
    return {
        "model_comparison_df": model_comparison_df,
        "design_diagnostics_df": pd.DataFrame(design_rows),
        "covariance_diagnostics_df": pd.DataFrame(covariance_rows),
    }


def _model_spec_to_formula(model_spec: str | list[str]) -> str:
    """Normalize a model-comparison spec to a formula string."""

    if isinstance(model_spec, str):
        spec = model_spec.strip()
        if spec.startswith("~"):
            return spec
        return "~ 1 + " + spec
    if len(model_spec) == 0:
        raise ValueError("Each model must include at least one predictor.")
    return "~ 1 + " + " + ".join(str(term) for term in model_spec)


def _add_model_deltas(
    rows: list[dict[str, object]],
    reference_model: str,
) -> list[dict[str, object]]:
    """Add per-fold metric deltas relative to the reference model."""

    df = pd.DataFrame(rows)
    ref_df = df.loc[df["model"] == reference_model]
    if len(ref_df) == 0:
        raise ValueError(f"reference_model '{reference_model}' was not evaluated.")
    ref_cols = [
        "subject",
        "fold",
        "time_ms",
        "cv_r2",
        "wmse",
        "wcv_r2",
        "heldout_loglik_total",
    ]
    ref_df = ref_df.loc[:, ref_cols].rename(
        columns={
            "cv_r2": "ref_cv_r2",
            "wmse": "ref_wmse",
            "wcv_r2": "ref_wcv_r2",
            "heldout_loglik_total": "ref_loglik",
        }
    )
    out = df.merge(ref_df, on=["subject", "fold", "time_ms"], how="left")
    out["delta_cv_r2"] = out["cv_r2"] - out["ref_cv_r2"]
    out["delta_wmse"] = out["wmse"] - out["ref_wmse"]
    out["delta_wcv_r2"] = out["wcv_r2"] - out["ref_wcv_r2"]
    out["delta_loglik"] = out["heldout_loglik_total"] - out["ref_loglik"]
    return out.drop(
        columns=["ref_cv_r2", "ref_wmse", "ref_wcv_r2", "ref_loglik"]
    ).to_dict(orient="records")


def compare_encoding_models_workflow(
    *,
    subject_ids: list[str],
    loader_cfg,
    condition_encoding: pd.DataFrame,
    design_cfg: EncodingConfig,
    models: dict[str, str | list[str]],
    source_to_condition: dict[str, str] | None = None,
    reference_model: str | None = None,
    overwrite: bool = False,
    cv_n_splits: int = 5,
    cv_shuffle: bool = True,
    cv_random_state: int = 42,
    time_window_ms: int = 50,
    standardize_data: bool = True,
    covariance: str = "shrinkage",
    results_dir: str | Path | None = None,
    run_name: str = "encoding_model_comparison",
    config_payload: dict[str, object] | None = None,
    log_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Compare candidate condition-level encoding models by held-out EEG prediction."""

    _ = overwrite
    if len(models) == 0:
        raise ValueError("models must include at least one model.")
    if reference_model is None:
        reference_model = next(iter(models))
    if reference_model not in models:
        raise ValueError("reference_model must be one of the provided models.")
    if source_to_condition is None:
        raise ValueError(
            "Provide source_to_condition for mapping raw labels to analysis conditions."
        )

    output_dir = Path(results_dir) if results_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    source_condition_col = loader_cfg.conditions.cond_col
    source_to_condition = {
        str(key): str(value) for key, value in source_to_condition.items()
    }
    subject_results = {}

    def process_one_subject(subject_id: str, progress_bar):
        result = _fit_model_comparison_subject(
            subject_id=subject_id,
            loader_cfg=loader_cfg,
            source_condition_col=source_condition_col,
            source_to_condition=source_to_condition,
            condition_encoding=condition_encoding,
            design_cfg=design_cfg,
            models=models,
            reference_model=str(reference_model),
            covariance=covariance,
            cv_n_splits=cv_n_splits,
            cv_shuffle=cv_shuffle,
            cv_random_state=cv_random_state,
            time_window_ms=time_window_ms,
            standardize_data=standardize_data,
            progress_bar=progress_bar,
        )
        subject_results[str(subject_id)] = result
        return result, False

    skipped_subjects_df = process_subjects(
        subject_ids=[str(subject_id) for subject_id in subject_ids],
        progress_total=cv_n_splits,
        log_label="Encoding model comparison run",
        log_path=log_path,
        experiment_name=loader_cfg.dataset.experiment_name,
        process_one_subject=process_one_subject,
    )
    if len(subject_results) == 0:
        skipped_summary = (
            skipped_subjects_df
            if len(skipped_subjects_df) > 0
            else pd.DataFrame(columns=["subject", "reason"])
        )
        raise RuntimeError(
            "No subjects were successfully processed for model comparison.\n"
            f"Failure summary:\n{skipped_summary.to_string(index=False)}"
        )

    model_comparison_df = pd.concat(
        [result["model_comparison_df"] for result in subject_results.values()],
        ignore_index=True,
    ).sort_values(["model", "subject", "time_ms"]).reset_index(drop=True)
    design_diagnostics_df = pd.concat(
        [result["design_diagnostics_df"] for result in subject_results.values()],
        ignore_index=True,
    )
    covariance_diagnostics_df = pd.concat(
        [result["covariance_diagnostics_df"] for result in subject_results.values()],
        ignore_index=True,
    )
    run_summary_df = pd.DataFrame(
        {
            "name": [run_name],
            "n_subjects_requested": [len(subject_ids)],
            "n_subjects_completed": [model_comparison_df["subject"].nunique()],
            "n_subjects_skipped": [len(skipped_subjects_df)],
            "time_window_ms": [int(time_window_ms)],
            "standardize_data": [bool(standardize_data)],
            "covariance_method": [str(covariance)],
            "reference_model": [str(reference_model)],
        }
    )

    if output_dir is not None:
        model_comparison_df.to_csv(
            output_dir / MODEL_COMPARISON_OUTPUT_FILES["model_comparison"],
            index=False,
        )
        design_diagnostics_df.to_csv(
            output_dir / MODEL_COMPARISON_OUTPUT_FILES["design_diagnostics"],
            index=False,
        )
        covariance_diagnostics_df.to_csv(
            output_dir / MODEL_COMPARISON_OUTPUT_FILES["covariance_diagnostics"],
            index=False,
        )
        run_summary_df.to_csv(
            output_dir / MODEL_COMPARISON_OUTPUT_FILES["run_summary"],
            index=False,
        )
        if len(skipped_subjects_df) > 0:
            skipped_subjects_df.to_csv(
                output_dir / MODEL_COMPARISON_OUTPUT_FILES["skipped_subjects"],
                index=False,
            )
        if config_payload is not None:
            import json

            with open(output_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config_payload, f, indent=2)

    return {
        "model_comparison_df": model_comparison_df,
        "design_diagnostics_df": design_diagnostics_df,
        "covariance_diagnostics_df": covariance_diagnostics_df,
        "run_summary_df": run_summary_df,
        "skipped_subjects_df": skipped_subjects_df,
    }


run_encoding = run_encoding_workflow
compare_encoding_models = compare_encoding_models_workflow
