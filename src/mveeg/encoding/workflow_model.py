"""Model-fitting workflow helpers for encoding analyses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .._shared.io_filters import load_subject_data_with_filters
from .._shared.time_windows import average_time_windows, build_time_windows
from .config import EncodingConfig
from .io import (
    README_FILENAME,
    load_encoding_model_result,
    save_encoding_model_result,
    write_pattern_expression_readme,
)
from .prepare import build_trial_encoding
from .run import fit_time_resolved_multivariate_ols
from .summaries import (
    build_condition_average_coefficient_table,
    build_testing_coefficient_tables,
    build_training_pattern_strength_table,
    compute_pattern_strength,
)
from .workflow_design import run_encoding_design_check, validate_glm_formula


MODEL_OUTPUT_FILES = {
    "training_pattern_strength": "training_pattern_strength.csv",
    "testing_effect_coefficients": "testing_effect_coefficients.csv",
    "testing_effect_coefficients_wide": "testing_effect_coefficients_wide.csv",
    "condition_average_coefficients": "condition_average_coefficients.csv",
    "subject_summary": "subject_summary.csv",
    "run_summary": "run_summary.csv",
    "skipped_subjects": "skipped_subjects.csv",
    "readme": README_FILENAME,
}

TEST_METHOD_NAME = "coefficient_reconstruction"
TEST_METHOD_VERSION = "v2_condition_shuffled_null_2026-04-17"


def _build_model_summary_row(
    *,
    subject: str,
    n_trials: int,
    n_channels: int,
    n_times: int,
    n_folds: int,
    condition_levels: list[str],
    training_pattern_strength_df: pd.DataFrame,
    testing_coefficient_df: pd.DataFrame,
) -> dict[str, object]:
    """Build one subject-level summary row for the main encoding workflow."""

    row = {
        "subject": str(subject),
        "n_trials": int(n_trials),
        "n_channels": int(n_channels),
        "n_times": int(n_times),
        "n_folds": int(n_folds),
        "condition_levels": ",".join(condition_levels),
    }

    for effect_name, effect_df in training_pattern_strength_df.groupby("effect"):
        pattern_rows = effect_df.loc[effect_df["data_type"] == "pattern"]
        row[f"mean_pattern_strength__{effect_name}"] = float(
            pattern_rows["pattern_strength"].mean()
        )

    for effect_name, effect_df in testing_coefficient_df.groupby("effect"):
        row[f"mean_coefficient__{effect_name}"] = float(
            effect_df["coefficient"].mean()
        )

    return row



def _standardize_trials_by_train_stats(
    *,
    train_data: np.ndarray,
    test_data: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score EEG data using fold-specific train statistics."""

    if train_data.ndim != 3 or test_data.ndim != 3:
        raise ValueError("train_data and test_data must be 3D arrays.")
    if train_data.shape[1:] != test_data.shape[1:]:
        raise ValueError(
            "train_data and test_data must have matching channel/time dimensions."
        )

    train_mean = np.mean(train_data, axis=0, keepdims=True)
    train_std = np.std(train_data, axis=0, keepdims=True)
    safe_std = np.where(train_std > tol, train_std, 1.0)

    train_z = (train_data - train_mean) / safe_std
    test_z = (test_data - train_mean) / safe_std
    return train_z, test_z



def _fit_condition_shuffled_effect_patterns(
    *,
    train_data: np.ndarray,
    train_conditions: np.ndarray,
    condition_encoding: pd.DataFrame,
    add_intercept: bool,
    design_names: list[str],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Return shuffled full-model beta patterns for all modeled effects.

    The training-set condition labels are permuted first, then the full design
    matrix is rebuilt from the shuffled condition sequence. This makes the null
    reflect a world where condition information is absent, rather than only
    perturbing one target predictor column at a time.

    Parameters
    ----------
    train_data : np.ndarray
        Training EEG data with shape ``(n_trials, n_channels, n_times)``.
    train_conditions : np.ndarray
        Condition label for each training trial.
    condition_encoding : pd.DataFrame
        Condition-to-predictor lookup table used to rebuild the design matrix.
    add_intercept : bool
        Whether the rebuilt design should include an intercept column.
    design_names : list[str]
        Expected design column names from the observed-data fit.
    rng : np.random.Generator
        Random number generator used for label shuffling.

    Returns
    -------
    dict[str, np.ndarray]
        Beta patterns keyed by predictor name for the shuffled full-model fit.
    """

    shuffled_conditions = rng.permutation(np.asarray(train_conditions, dtype=object))
    shuffled_design, shuffled_names, _ = build_trial_encoding(
        condition_encoding=condition_encoding,
        trial_conditions=shuffled_conditions,
        add_intercept=add_intercept,
    )
    if list(shuffled_names) != list(design_names):
        raise ValueError(
            "Shuffled design names do not match the observed design. "
            f"Expected {design_names}, got {shuffled_names}."
        )

    fit_result = fit_time_resolved_multivariate_ols(
        data=train_data,
        design_matrix=shuffled_design,
        design_names=shuffled_names,
    )
    fit_predictor_to_ix = {
        name: ix for ix, name in enumerate(fit_result["predictor_names"])
    }
    return {
        predictor_name: fit_result["betas"][predictor_ix, :, :].astype(float)
        for predictor_name, predictor_ix in fit_predictor_to_ix.items()
        if predictor_name != "intercept"
    }



def _reconstruct_trial_coefficients(
    *,
    trial_pattern: np.ndarray,
    basis_patterns: list[np.ndarray],
    tol: float = 1e-12,
) -> np.ndarray:
    """Reconstruct basis coefficients for one trial and time.

    Parameters
    ----------
    trial_pattern : np.ndarray
        Held-out EEG vector with shape ``(n_channels,)`` for one time bin.
    basis_patterns : list[np.ndarray]
        Training beta patterns for the same time bin. The list should start
        with the intercept pattern, followed by the modeled predictors in the
        same order used by the design matrix.
    tol : float, optional
        Small ridge term used when the basis is nearly singular.

    Returns
    -------
    np.ndarray
        Reconstructed coefficients in the same order as ``basis_patterns``.
    """

    basis = np.column_stack([pattern.astype(float) for pattern in basis_patterns])
    target = trial_pattern.astype(float)
    gram = basis.T @ basis
    if np.linalg.cond(gram) > 1.0 / tol:
        gram = gram + np.eye(gram.shape[0], dtype=float) * tol
    coef = np.linalg.solve(gram, basis.T @ target)
    return coef.astype(float)



def _reconstruct_time_resolved_coefficients(
    *,
    test_data: np.ndarray,
    fold_patterns: dict[str, np.ndarray],
    predictor_names: list[str],
) -> dict[str, np.ndarray]:
    """Reconstruct trial-level coefficients across all held-out times.

    Parameters
    ----------
    test_data : np.ndarray
        Held-out EEG matrix with shape ``(n_trials, n_channels, n_times)``.
    fold_patterns : dict[str, np.ndarray]
        Training beta patterns for the intercept and each modeled predictor.
    predictor_names : list[str]
        Design names in coefficient order. Must include ``"intercept"`` first
        when the model contains an intercept.

    Returns
    -------
    dict[str, np.ndarray]
        Trial-by-time coefficient matrix for each name in ``predictor_names``.
    """

    n_trials, n_channels, n_times = test_data.shape
    for name in predictor_names:
        if name not in fold_patterns:
            raise ValueError(f"Missing beta pattern for predictor '{name}'.")
        pattern = fold_patterns[name]
        if pattern.shape != (n_channels, n_times):
            raise ValueError(f"{name} must have shape (n_channels, n_times).")

    coef_by_predictor = {
        name: np.empty((n_trials, n_times), dtype=float) for name in predictor_names
    }

    for trial_ix in range(n_trials):
        for time_ix in range(n_times):
            basis_patterns = [
                fold_patterns[name][:, time_ix] for name in predictor_names
            ]
            trial_coefs = _reconstruct_trial_coefficients(
                trial_pattern=test_data[trial_ix, :, time_ix],
                basis_patterns=basis_patterns,
            )
            for coef_ix, name in enumerate(predictor_names):
                coef_by_predictor[name][trial_ix, time_ix] = trial_coefs[coef_ix]

    return coef_by_predictor



def _saved_result_matches_current_settings(
    saved: dict[str, np.ndarray],
    *,
    standardize_data: bool,
    time_window_ms: int,
    n_null_repeats: int,
    source_condition_col: str,
    source_to_condition: dict[str, str],
    train_condition_labels: tuple[str, ...] | None,
) -> bool:
    """Return whether one cached subject result matches current run settings.

    Parameters
    ----------
    saved : dict[str, np.ndarray]
        Subject-level NPZ payload loaded from a previous encoding run.
    standardize_data : bool
        Whether the current run standardizes held-out data using training-set
        statistics.
    time_window_ms : int
        Width of the current time bins in milliseconds.
    n_null_repeats : int
        Number of condition-shuffled null draws requested for the current run.
    source_condition_col : str
        Metadata column used to read the raw condition labels.
    source_to_condition : dict[str, str]
        Mapping from raw metadata labels to analysis condition labels.
    train_condition_labels : tuple[str, ...] | None
        Analysis condition labels allowed in training folds. ``None`` means all
        loaded conditions are used for training.

    Returns
    -------
    bool
        ``True`` when the saved result was created with matching run settings.
        Older cache files that do not record the condition source are treated
        as non-matching so they are recomputed.
    """

    required_keys = {
        "subject",
        "times_s",
        "ch_names",
        "n_trials",
        "n_channels",
        "n_times",
        "n_folds",
        "time_window_ms",
        "condition_levels",
        "standardize_data",
        "test_method_name",
        "test_method_version",
        "predictor_names",
        "raw_beta_patterns",
        "pattern_strength_pattern",
        "pattern_strength_null_draws",
        "n_null_repeats",
        "coef_predictor_names",
        "coef_values",
        "coef_fold",
        "coef_condition",
        "coef_trial_index",
        "coef_time_ms",
        "source_condition_col",
        "source_condition_keys",
        "source_condition_values",
        "train_condition_labels",
    }
    if not required_keys.issubset(saved.keys()):
        return False

    saved_source_map = dict(
        zip(
            saved["source_condition_keys"].astype(str).tolist(),
            saved["source_condition_values"].astype(str).tolist(),
            strict=True,
        )
    )
    expected_source_map = {
        str(key): str(value) for key, value in source_to_condition.items()
    }
    saved_train_labels = tuple(saved["train_condition_labels"].astype(str).tolist())
    expected_train_labels = (
        tuple() if train_condition_labels is None else tuple(train_condition_labels)
    )
    return (
        bool(saved["standardize_data"].item()) == bool(standardize_data)
        and int(saved["time_window_ms"].item()) == int(time_window_ms)
        and int(saved["n_null_repeats"].item()) == int(n_null_repeats)
        and str(saved["test_method_name"].item()) == TEST_METHOD_NAME
        and str(saved["test_method_version"].item()) == TEST_METHOD_VERSION
        and str(saved["source_condition_col"].item()) == str(source_condition_col)
        and saved_source_map == expected_source_map
        and saved_train_labels == expected_train_labels
    )



def _build_training_df_from_saved(saved: dict[str, np.ndarray]) -> pd.DataFrame:
    """Rebuild the training long table from one saved NPZ payload."""

    subject = str(saved["subject"].item())
    times_s = saved["times_s"].astype(float)
    predictor_names = saved["predictor_names"].astype(str).tolist()
    raw_effect_names = [name for name in predictor_names if name != "intercept"]
    observed_strength = saved["pattern_strength_pattern"].astype(float)
    null_draws = saved["pattern_strength_null_draws"].astype(float)
    n_null_repeats = int(saved["n_null_repeats"].item())

    rows = []
    for fold_ix in range(observed_strength.shape[0]):
        fold_id = fold_ix + 1
        for effect_ix, raw_effect_name in enumerate(raw_effect_names):
            rows.append(
                build_training_pattern_strength_table(
                    subject=subject,
                    fold_id=fold_id,
                    effect=raw_effect_name,
                    times_s=times_s,
                    pattern_strength=observed_strength[fold_ix, effect_ix, :],
                    data_type="pattern",
                    null_draw=0,
                    n_null_repeats=n_null_repeats,
                )
            )
            for null_ix in range(null_draws.shape[2]):
                rows.append(
                    build_training_pattern_strength_table(
                        subject=subject,
                        fold_id=fold_id,
                        effect=raw_effect_name,
                        times_s=times_s,
                        pattern_strength=null_draws[fold_ix, effect_ix, null_ix, :],
                        data_type="null",
                        null_draw=null_ix + 1,
                        n_null_repeats=n_null_repeats,
                    )
                )

    return pd.concat(rows, ignore_index=True)



def _build_testing_dfs_from_saved(
    saved: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild testing long and wide tables from one saved NPZ payload."""

    subject = str(saved["subject"].item())
    coef_predictor_names = saved["coef_predictor_names"].astype(str).tolist()
    coef_values = saved["coef_values"].astype(float)
    wide_data = {
        "subject": np.repeat(subject, len(saved["coef_fold"])),
        "fold": saved["coef_fold"].astype(int),
        "condition": saved["coef_condition"].astype(str),
        "trial_index": saved["coef_trial_index"].astype(int),
        "time_ms": saved["coef_time_ms"].astype(float),
    }
    for predictor_ix, predictor_name in enumerate(coef_predictor_names):
        wide_data[f"coef_{predictor_name}"] = coef_values[:, predictor_ix]
    coef_wide_df = pd.DataFrame(wide_data)

    long_rows = []
    base_cols = ["subject", "fold", "condition", "trial_index", "time_ms"]
    for predictor_name in coef_predictor_names:
        if predictor_name == "intercept":
            continue
        long_rows.append(
            coef_wide_df.loc[:, base_cols].assign(
                effect=predictor_name,
                coefficient=coef_wide_df[f"coef_{predictor_name}"].to_numpy(dtype=float),
            )
        )
    coef_long_df = pd.concat(long_rows, ignore_index=True)
    return coef_long_df, coef_wide_df



def run_encoding(
    *,
    subject_ids: list[str],
    subject_results_dir: str | Path,
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
    n_null_repeats: int = 20,
    results_dir: str | Path | None = None,
    run_name: str = "encoding_model",
    config_payload: dict[str, object] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run fold-wise training and testing exports for the encoding model.

    Parameters
    ----------
    subject_ids : list[str]
        Subject identifiers to process.
    subject_results_dir : str | Path
        Folder used for per-subject cache files.
    loader_cfg : object
        Subject-loading config. ``loader_cfg.conditions.cond_col`` defines the
        metadata column containing raw condition labels.
    condition_encoding : pd.DataFrame
        Condition-level design table with one ``condition`` column and one
        column per modeled predictor.
    design_cfg : EncodingConfig
        Validation settings for the encoding design matrix.
    glm_formula : str
        Additive R-style formula used to select predictors from
        ``condition_encoding``.
    source_to_condition : dict[str, str] | None
        Mapping from raw labels in ``loader_cfg.conditions.cond_col`` to the
        analysis condition labels in ``condition_encoding``.
    train_condition_labels : list[str] | tuple[str, ...] | None
        Optional analysis condition labels used for model fitting inside each
        fold. Testing still uses all loaded held-out conditions.

    Returns
    -------
    dict[str, pd.DataFrame]
        Subject summaries and long/wide encoding output tables.
    """

    if cv_n_splits < 2:
        raise ValueError("cv_n_splits must be at least 2.")
    if time_window_ms < 1:
        raise ValueError("time_window_ms must be at least 1 millisecond.")
    if n_null_repeats < 1:
        raise ValueError("n_null_repeats must be at least 1.")
    if source_to_condition is None:
        raise ValueError(
            "Provide source_to_condition for mapping raw labels to analysis conditions."
        )

    subject_results_dir = Path(subject_results_dir)
    subject_results_dir.mkdir(parents=True, exist_ok=True)
    source_condition_col = loader_cfg.conditions.cond_col
    source_to_condition = {
        str(key): str(value) for key, value in source_to_condition.items()
    }
    if train_condition_labels is not None:
        train_condition_labels = tuple(str(label) for label in train_condition_labels)
    parsed_formula = validate_glm_formula(
        glm_formula,
        allowed_predictors=set(condition_encoding.columns).difference({"condition"}),
    )
    formula_predictors = parsed_formula["predictors"]
    formula_add_intercept = bool(parsed_formula["add_intercept"])
    formula_condition_encoding = condition_encoding.loc[
        :, ["condition", *formula_predictors]
    ].copy()
    effective_design_cfg = EncodingConfig(
        add_intercept=formula_add_intercept,
        validation_mode=design_cfg.validation_mode,
        tolerance=design_cfg.tolerance,
    )
    output_dir = Path(results_dir) if results_dir is not None else subject_results_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_summary_rows = []
    training_tables = []
    testing_tables = []
    testing_wide_tables = []
    skipped_rows = []
    for subject_id in subject_ids:
        try:
            if not overwrite:
                try:
                    saved = load_encoding_model_result(subject_results_dir, subject_id)
                except FileNotFoundError:
                    saved = None
                if saved is not None and _saved_result_matches_current_settings(
                    saved,
                    standardize_data=standardize_data,
                    time_window_ms=time_window_ms,
                    n_null_repeats=n_null_repeats,
                    source_condition_col=source_condition_col,
                    source_to_condition=source_to_condition,
                    train_condition_labels=train_condition_labels,
                ):
                    training_pattern_strength_df = _build_training_df_from_saved(saved)
                    testing_coefficient_df, testing_coefficient_wide_df = _build_testing_dfs_from_saved(saved)
                    condition_levels = saved["condition_levels"].astype(str).tolist()
                    subject_summary_rows.append(
                        _build_model_summary_row(
                            subject=str(saved["subject"].item()),
                            n_trials=int(saved["n_trials"].item()),
                            n_channels=int(saved["n_channels"].item()),
                            n_times=int(saved["n_times"].item()),
                            n_folds=int(saved["n_folds"].item()),
                            condition_levels=condition_levels,
                            training_pattern_strength_df=training_pattern_strength_df,
                            testing_coefficient_df=testing_coefficient_df,
                        )
                    )
                    training_tables.append(training_pattern_strength_df)
                    testing_tables.append(testing_coefficient_df)
                    testing_wide_tables.append(testing_coefficient_wide_df)
                    print(f"sub-{subject_id}: reused")
                    continue

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

            metadata["condition"] = (
                metadata[source_condition_col].astype(str).map(source_to_condition)
            )
            design_output = run_encoding_design_check(
                trial_conditions=metadata["condition"].to_numpy(dtype=object),
                condition_encoding=formula_condition_encoding,
                cfg=effective_design_cfg,
            )

            validation = design_output["validation"]
            if not validation.is_valid:
                raise ValueError("; ".join(validation.messages))

            design_matrix = design_output["design_matrix"]
            design_names = design_output["design_names"]
            predictor_names = list(design_names)

            condition_values = metadata["condition"].to_numpy(dtype=object).astype(str)
            condition_counts = pd.Series(condition_values).value_counts()
            insufficient_conditions = condition_counts[condition_counts < cv_n_splits]
            if len(insufficient_conditions) > 0:
                raise ValueError(
                    "Not enough trials per condition for stratified CV. "
                    f"Need at least {cv_n_splits} trials each, found: "
                    f"{insufficient_conditions.to_dict()}"
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

            cv = StratifiedKFold(
                n_splits=cv_n_splits,
                shuffle=cv_shuffle,
                random_state=cv_random_state if cv_shuffle else None,
            )
            rng = np.random.default_rng(cv_random_state)
            training_rows = []
            testing_rows = []
            testing_wide_rows = []
            n_predictors = len(predictor_names)
            effect_predictors = list(formula_predictors)
            n_effects = len(effect_predictors)
            raw_beta_patterns = np.empty(
                (cv_n_splits, n_predictors, data.shape[1], data.shape[2]),
                dtype=float,
            )
            pattern_strength_pattern = np.empty(
                (cv_n_splits, n_effects, data.shape[2]),
                dtype=float,
            )
            pattern_strength_null_draws = np.empty(
                (cv_n_splits, n_effects, n_null_repeats, data.shape[2]),
                dtype=float,
            )

            for fold_ix, (train_idx, test_idx) in enumerate(
                cv.split(np.zeros(len(condition_values)), condition_values),
                start=0,
            ):
                fold_id = fold_ix + 1
                train_model_idx = train_idx[train_condition_mask[train_idx]]
                if len(train_model_idx) == 0:
                    raise ValueError(
                        "No training trials remain after applying train_condition_labels."
                    )

                train_data = data[train_model_idx, :, :]
                test_data = data[test_idx, :, :]
                train_conditions = condition_values[train_model_idx]
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
                fold_predictor_to_ix = {
                    name: ix for ix, name in enumerate(fit_result["predictor_names"])
                }
                fold_patterns = {
                    predictor_name: fit_result["betas"][predictor_ix, :, :].astype(float)
                    for predictor_name, predictor_ix in fold_predictor_to_ix.items()
                }
                for predictor_ix, predictor_name in enumerate(predictor_names):
                    raw_beta_patterns[fold_ix, predictor_ix, :, :] = fold_patterns[
                        predictor_name
                    ]

                for effect_ix, effect_name in enumerate(effect_predictors):
                    observed_strength = compute_pattern_strength(
                        fold_patterns[effect_name]
                    )
                    pattern_strength_pattern[fold_ix, effect_ix, :] = observed_strength
                    training_rows.append(
                        build_training_pattern_strength_table(
                            subject=subject_id,
                            fold_id=fold_id,
                            effect=effect_name,
                            times_s=np.asarray(times_s, dtype=float),
                            pattern_strength=observed_strength,
                            data_type="pattern",
                            null_draw=0,
                            n_null_repeats=n_null_repeats,
                        )
                    )

                for null_ix in range(n_null_repeats):
                    shuffled_patterns = _fit_condition_shuffled_effect_patterns(
                        train_data=train_data,
                        train_conditions=train_conditions,
                        condition_encoding=formula_condition_encoding,
                        add_intercept=formula_add_intercept,
                        design_names=design_names,
                        rng=rng,
                    )
                    for effect_ix, effect_name in enumerate(effect_predictors):
                        null_strength = compute_pattern_strength(
                            shuffled_patterns[effect_name]
                        )
                        pattern_strength_null_draws[
                            fold_ix, effect_ix, null_ix, :
                        ] = null_strength
                        training_rows.append(
                            build_training_pattern_strength_table(
                                subject=subject_id,
                                fold_id=fold_id,
                                effect=effect_name,
                                times_s=np.asarray(times_s, dtype=float),
                                pattern_strength=null_strength,
                                data_type="null",
                                null_draw=null_ix + 1,
                                n_null_repeats=n_null_repeats,
                            )
                        )

                test_conditions = condition_values[test_idx]
                test_trial_index = np.asarray(test_idx, dtype=int)
                coef_by_name = _reconstruct_time_resolved_coefficients(
                    test_data=test_data,
                    fold_patterns=fold_patterns,
                    predictor_names=predictor_names,
                )
                coefficient_wide_df, coefficient_long_df = build_testing_coefficient_tables(
                    subject=subject_id,
                    fold_id=fold_id,
                    condition_labels=test_conditions,
                    trial_index=test_trial_index,
                    times_s=np.asarray(times_s, dtype=float),
                    coef_by_name=coef_by_name,
                    coef_by_predictor={
                        predictor_name: coef_by_name[predictor_name]
                        for predictor_name in effect_predictors
                    },
                )
                testing_wide_rows.append(coefficient_wide_df)
                testing_rows.append(coefficient_long_df)

            training_pattern_strength_df = pd.concat(training_rows, ignore_index=True)
            training_pattern_strength_df = training_pattern_strength_df.sort_values(
                ["effect", "data_type", "fold", "null_draw", "time_ms"]
            ).reset_index(drop=True)
            testing_coefficient_wide_df = pd.concat(testing_wide_rows, ignore_index=True)
            testing_coefficient_wide_df = testing_coefficient_wide_df.sort_values(
                ["condition", "fold", "trial_index", "time_ms"]
            ).reset_index(drop=True)
            testing_coefficient_df = pd.concat(testing_rows, ignore_index=True)
            testing_coefficient_df = testing_coefficient_df.sort_values(
                ["effect", "condition", "fold", "trial_index", "time_ms"]
            ).reset_index(drop=True)
            condition_coefficient_df = build_condition_average_coefficient_table(
                testing_coefficient_df
            )
            condition_levels = sorted(np.unique(condition_values).tolist())
            coef_values = np.column_stack(
                [
                    testing_coefficient_wide_df[f"coef_{predictor_name}"].to_numpy(
                        dtype=float
                    )
                    for predictor_name in predictor_names
                ]
            )
            save_encoding_model_result(
                output_dir=subject_results_dir,
                subject_id=subject_id,
                payload={
                    "subject": np.asarray(subject_id, dtype=object),
                    "times_s": np.asarray(times_s, dtype=float),
                    "ch_names": np.asarray(ch_names, dtype=object),
                    "n_trials": np.asarray(data.shape[0], dtype=int),
                    "n_channels": np.asarray(data.shape[1], dtype=int),
                    "n_times": np.asarray(data.shape[2], dtype=int),
                    "n_folds": np.asarray(cv_n_splits, dtype=int),
                    "time_window_ms": np.asarray(int(time_window_ms), dtype=int),
                    "condition_levels": np.asarray(condition_levels, dtype=object),
                    "standardize_data": np.asarray(bool(standardize_data), dtype=bool),
                    "test_method_name": np.asarray(TEST_METHOD_NAME, dtype=object),
                    "test_method_version": np.asarray(TEST_METHOD_VERSION, dtype=object),
                    "source_condition_col": np.asarray(source_condition_col, dtype=object),
                    "source_condition_keys": np.asarray(
                        sorted(source_to_condition.keys()),
                        dtype=object,
                    ),
                    "source_condition_values": np.asarray(
                        [
                            source_to_condition[key]
                            for key in sorted(source_to_condition.keys())
                        ],
                        dtype=object,
                    ),
                    "train_condition_labels": np.asarray(
                        [] if train_condition_labels is None else train_condition_labels,
                        dtype=object,
                    ),
                    "predictor_names": np.asarray(predictor_names, dtype=object),
                    "raw_beta_patterns": raw_beta_patterns,
                    "pattern_strength_pattern": pattern_strength_pattern,
                    "pattern_strength_null_draws": pattern_strength_null_draws,
                    "n_null_repeats": np.asarray(int(n_null_repeats), dtype=int),
                    "coef_predictor_names": np.asarray(predictor_names, dtype=object),
                    "coef_values": coef_values,
                    "coef_fold": testing_coefficient_wide_df["fold"].to_numpy(dtype=int),
                    "coef_condition": testing_coefficient_wide_df[
                        "condition"
                    ].to_numpy(dtype=object),
                    "coef_trial_index": testing_coefficient_wide_df[
                        "trial_index"
                    ].to_numpy(dtype=int),
                    "coef_time_ms": testing_coefficient_wide_df["time_ms"].to_numpy(
                        dtype=float
                    ),
                },
            )

            subject_summary_rows.append(
                _build_model_summary_row(
                    subject=subject_id,
                    n_trials=data.shape[0],
                    n_channels=data.shape[1],
                    n_times=data.shape[2],
                    n_folds=cv_n_splits,
                    condition_levels=condition_levels,
                    training_pattern_strength_df=training_pattern_strength_df,
                    testing_coefficient_df=testing_coefficient_df,
                )
            )
            training_tables.append(training_pattern_strength_df)
            testing_tables.append(testing_coefficient_df)
            testing_wide_tables.append(testing_coefficient_wide_df)
            print(f"sub-{subject_id}: done")
        except Exception as err:
            skipped_rows.append({"subject": str(subject_id), "reason": str(err)})
            print(f"sub-{subject_id}: skipped ({err})")

    if len(training_tables) == 0:
        raise RuntimeError(
            "No subjects were successfully processed for training pattern strength."
        )
    if len(testing_tables) == 0:
        raise RuntimeError(
            "No subjects were successfully processed for testing coefficients."
        )

    subject_summary_df = pd.DataFrame(subject_summary_rows).sort_values("subject").reset_index(drop=True)
    skipped_subjects_df = pd.DataFrame(skipped_rows)
    run_summary_df = pd.DataFrame(
        {
            "name": [run_name],
            "n_subjects_requested": [len(subject_ids)],
            "n_subjects_completed": [len(subject_summary_df)],
            "n_subjects_skipped": [len(skipped_subjects_df)],
            "test_method_name": [TEST_METHOD_NAME],
            "test_method_version": [TEST_METHOD_VERSION],
            "time_window_ms": [int(time_window_ms)],
            "standardize_data": [bool(standardize_data)],
            "n_null_repeats": [int(n_null_repeats)],
        }
    )
    training_pattern_strength_df = pd.concat(training_tables, ignore_index=True)
    training_pattern_strength_df = training_pattern_strength_df.sort_values(
        ["effect", "data_type", "subject", "fold", "null_draw", "time_ms"]
    ).reset_index(drop=True)
    testing_coefficient_df = pd.concat(testing_tables, ignore_index=True)
    testing_coefficient_df = testing_coefficient_df.sort_values(
        ["effect", "condition", "subject", "fold", "trial_index", "time_ms"]
    ).reset_index(drop=True)
    testing_coefficient_wide_df = pd.concat(testing_wide_tables, ignore_index=True)
    testing_coefficient_wide_df = testing_coefficient_wide_df.sort_values(
        ["condition", "subject", "fold", "trial_index", "time_ms"]
    ).reset_index(drop=True)
    condition_coefficient_df = build_condition_average_coefficient_table(
        testing_coefficient_df
    )

    subject_summary_df.to_csv(output_dir / MODEL_OUTPUT_FILES["subject_summary"], index=False)
    if len(skipped_subjects_df) > 0:
        skipped_subjects_df.to_csv(output_dir / MODEL_OUTPUT_FILES["skipped_subjects"], index=False)
    run_summary_df.to_csv(output_dir / MODEL_OUTPUT_FILES["run_summary"], index=False)
    training_pattern_strength_df.to_csv(
        output_dir / MODEL_OUTPUT_FILES["training_pattern_strength"],
        index=False,
    )
    testing_coefficient_df.to_csv(
        output_dir / MODEL_OUTPUT_FILES["testing_effect_coefficients"],
        index=False,
    )
    testing_coefficient_wide_df.to_csv(
        output_dir / MODEL_OUTPUT_FILES["testing_effect_coefficients_wide"],
        index=False,
    )
    condition_coefficient_df.to_csv(
        output_dir / MODEL_OUTPUT_FILES["condition_average_coefficients"],
        index=False,
    )
    write_pattern_expression_readme(output_dir)
    if config_payload is not None:
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)

    return {
        "subject_summary_df": subject_summary_df,
        "skipped_subjects_df": skipped_subjects_df,
        "run_summary_df": run_summary_df,
        "training_pattern_strength_df": training_pattern_strength_df,
        "testing_coefficient_df": testing_coefficient_df,
        "testing_coefficient_wide_df": testing_coefficient_wide_df,
        "condition_coefficient_df": condition_coefficient_df,
    }
