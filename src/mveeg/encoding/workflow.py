"""Public workflow facade for encoding analyses.

These helpers keep encoding scripts focused on analysis decisions while smaller
workflow modules handle paths, design checks, pattern exports, and model runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .._shared.topography import build_topography_coord_table
from ..io.results import (
    init_result_store,
    read_internal_table,
    read_result_subject_status,
    replace_result_tables,
    replace_subject_table_rows,
    resolve_result_file,
    result_config_hash,
    result_skipped_table,
    run_incremental_result_store,
    update_result_subject_status,
)
from .._shared import (
    ConditionGroupsConfig,
    DataPathsConfig,
    EpochProcessingConfig,
    SubjectLoadConfig,
    TrialFilterRulesConfig,
)
from .config import EncodingConfig
from .workflow_design import run_encoding_design_check, validate_glm_formula
from .workflow_model import (
    MODEL_OUTPUT_FILES,
    TEST_METHOD_NAME,
    TEST_METHOD_VERSION,
    run_encoding_workflow,
)
from .io import load_subject_info
from .summaries import build_condition_average_coefficient_table
from .workflow_paths import infer_experiment_settings, prepare_encoding_paths
from .workflow_pattern_expression import (
    LEGACY_OUTPUT_FILES,
    export_encoding_outputs,
    run_pattern_expression_workflow,
)


__all__ = [
    "LEGACY_OUTPUT_FILES",
    "MODEL_OUTPUT_FILES",
    "TEST_METHOD_NAME",
    "TEST_METHOD_VERSION",
    "export_encoding_outputs",
    "infer_experiment_settings",
    "prepare_encoding_paths",
    "run_encoding",
    "run_encoding_workflow",
    "run_encoding_design_check",
    "run_pattern_expression",
    "run_pattern_expression_workflow",
    "validate_glm_formula",
]


_ENCODING_STORE_TABLES = {
    "subject_summary": "_mveeg_subject_summary",
    "training_pattern_strength": "_mveeg_subject_training_pattern_strength",
    "testing_effect_coefficients": "_mveeg_subject_testing_effect_coefficients",
    "testing_effect_coefficients_wide": "_mveeg_subject_testing_effect_coefficients_wide",
    "beta_patterns": "_mveeg_subject_beta_patterns",
}

_PATTERN_STORE_TABLES = {
    "trial_expression": "_mveeg_subject_trial_expression",
    "condition_expression": "_mveeg_subject_condition_expression",
    "trials": "_mveeg_subject_trials",
}


def run_encoding(
    *,
    data_dir: str | Path,
    subject_ids: list[str],
    trial_filters: dict[str, object],
    encoding_params: dict[str, object],
    condition_encoding: pd.DataFrame,
    condition_label_map: dict[str, list[str]],
    glm_formula: str,
    overwrite: bool,
    name: str,
    file: str | Path | None = None,
    train_condition_labels: list[str] | tuple[str, ...] | None = None,
    topography: dict[str, object] | None = None,
    experiment_name: str | None = None,
    cond_col: str = "label",
) -> dict[str, pd.DataFrame]:
    """Run one encoding analysis from grouped script-level settings.

    Parameters
    ----------
    data_dir : str | Path
        Preprocessed data folder used for the current experiment.
    subject_ids : list[str]
        Subject IDs requested by the caller.
    trial_filters : dict[str, object]
        Trial inclusion and exclusion settings.
    encoding_params : dict[str, object]
        Time-window, channel-drop, cross-validation, and null settings.
    condition_encoding : pd.DataFrame
        Condition-level design table with one ``condition`` column and one
        column per modeled predictor.
    condition_label_map : dict[str, list[str]]
        Mapping from analysis condition labels to raw metadata labels.
    glm_formula : str
        Additive R-style formula used to select predictors from
        ``condition_encoding``.
    overwrite : bool
        Whether to recompute when ``file`` already exists.
    name : str
        Analysis name used for display and output folders.
    file : str | Path | None
        Optional single-file DuckDB cache for the returned result tables. Paths
        without a suffix receive ``.duckdb``. Existing files are loaded unless
        ``overwrite=True``.
    train_condition_labels : list[str] | tuple[str, ...] | None
        Optional analysis condition labels used for model fitting.
    topography : dict[str, object] | None
        Optional encoding topography export settings. Use ``time_window_ms``
        as ``(start_ms, end_ms)`` for one window, or ``time_windows_ms`` as a
        named window dictionary for multi-window export.
    experiment_name : str | None
        Experiment name used to locate derivative files. If ``None``, the final
        folder name from ``data_dir`` is used.
    cond_col : str
        Metadata column used to read raw condition labels.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Encoding result tables keyed by stable public names.
    """

    experiment_name, _ = infer_experiment_settings(
        data_dir=data_dir,
        experiment_name=experiment_name,
        results_subdir=None,
    )

    source_to_condition = {
        source: label
        for label, sources in condition_label_map.items()
        for source in sources
    }
    validation_mode = encoding_params.get("validation_mode", "estimable_independent")
    tolerance = encoding_params.get("tolerance", 1e-10)

    loader_cfg = SubjectLoadConfig(
        dataset=DataPathsConfig(
            data_dir=data_dir,
            experiment_name=experiment_name,
        ),
        conditions=ConditionGroupsConfig(
            train_cond=condition_label_map,
            test_cond=condition_label_map,
            cond_col=cond_col,
        ),
        filters=TrialFilterRulesConfig(
            qc_col=trial_filters["qc_col"],
            keep_qc=tuple(trial_filters["keep_qc"]),
            exclude_metadata=trial_filters["exclude_metadata"],
        ),
        epoch=EpochProcessingConfig(
            crop_time=encoding_params["crop_time"],
            drop_channel_types=encoding_params["drop_channel_types"],
            drop_channels=encoding_params["drop_channels"],
        ),
    )
    design_cfg = EncodingConfig(
        add_intercept=True,
        validation_mode=validation_mode,
        tolerance=tolerance,
    )
    cv_n_splits = encoding_params.get("cv_n_splits", encoding_params.get("n_splits", 5))
    cv_shuffle = encoding_params.get("cv_shuffle", encoding_params.get("shuffle", True))
    cv_random_state = encoding_params.get(
        "cv_random_state",
        encoding_params.get("random_state", 42),
    )

    result_file = resolve_result_file(file)
    if result_file is not None:
        return _run_encoding_store(
            result_file=result_file,
            subject_ids=subject_ids,
            loader_cfg=loader_cfg,
            condition_encoding=condition_encoding,
            design_cfg=design_cfg,
            glm_formula=glm_formula,
            source_to_condition=source_to_condition,
            train_condition_labels=train_condition_labels,
            overwrite=overwrite,
            cv_n_splits=int(cv_n_splits),
            cv_shuffle=bool(cv_shuffle),
            cv_random_state=int(cv_random_state),
            time_window_ms=int(encoding_params["time_window_ms"]),
            standardize_data=bool(encoding_params.get("standardize_data", True)),
            n_null_repeats=int(encoding_params["n_null_repeats"]),
            run_name=name,
            topography=topography,
            config_payload={
                "analysis_type": "encoding",
                "data_dir": str(data_dir),
                "experiment_name": experiment_name,
                "trial_filters": trial_filters,
                "encoding_params": encoding_params,
                "condition_encoding": condition_encoding.to_dict(orient="list"),
                "condition_label_map": condition_label_map,
                "glm_formula": glm_formula,
                "train_condition_labels": train_condition_labels,
                "topography": topography,
                "cond_col": cond_col,
            },
        )

    print(f"Running {name}")
    run_output = run_encoding_workflow(
        subject_ids=subject_ids,
        subject_results_dir=None,
        loader_cfg=loader_cfg,
        condition_encoding=condition_encoding,
        design_cfg=design_cfg,
        glm_formula=glm_formula,
        source_to_condition=source_to_condition,
        train_condition_labels=train_condition_labels,
        overwrite=overwrite,
        cv_n_splits=int(cv_n_splits),
        cv_shuffle=bool(cv_shuffle),
        cv_random_state=int(cv_random_state),
        time_window_ms=int(encoding_params["time_window_ms"]),
        standardize_data=bool(encoding_params.get("standardize_data", True)),
        n_null_repeats=int(encoding_params["n_null_repeats"]),
        results_dir=None,
        run_name=name,
        config_payload=None,
        topography=topography,
        log_path=None,
    )

    tables = _public_table_names(run_output)
    return tables


def _run_encoding_store(
    *,
    result_file: Path,
    subject_ids: list[str],
    loader_cfg,
    condition_encoding: pd.DataFrame,
    design_cfg: EncodingConfig,
    glm_formula: str,
    source_to_condition: dict[str, str],
    train_condition_labels: list[str] | tuple[str, ...] | None,
    overwrite: bool,
    cv_n_splits: int,
    cv_shuffle: bool,
    cv_random_state: int,
    time_window_ms: int,
    standardize_data: bool,
    n_null_repeats: int,
    run_name: str,
    topography: dict[str, object] | None,
    config_payload: dict[str, object],
) -> dict[str, pd.DataFrame]:
    init_result_store(
        result_file,
        analysis_type="encoding",
        config_hash=result_config_hash(config_payload),
        run_name=run_name,
    )
    return run_incremental_result_store(
        result_file,
        subject_ids=subject_ids,
        overwrite=overwrite,
        process_subjects=lambda subjects_to_run: _process_encoding_store_subjects(
            result_file=result_file,
            subject_ids=subjects_to_run,
            loader_cfg=loader_cfg,
            condition_encoding=condition_encoding,
            design_cfg=design_cfg,
            glm_formula=glm_formula,
            source_to_condition=source_to_condition,
            train_condition_labels=train_condition_labels,
            cv_n_splits=cv_n_splits,
            cv_shuffle=cv_shuffle,
            cv_random_state=cv_random_state,
            time_window_ms=time_window_ms,
            standardize_data=standardize_data,
            n_null_repeats=n_null_repeats,
            run_name=run_name,
        ),
        finalize=lambda: _finalize_encoding_store(
            result_file=result_file,
            loader_cfg=loader_cfg,
            run_name=run_name,
            topography=topography,
            time_window_ms=time_window_ms,
            standardize_data=standardize_data,
            n_null_repeats=n_null_repeats,
        ),
    )


def _process_encoding_store_subjects(
    *,
    result_file: Path,
    subject_ids: list[str],
    loader_cfg,
    condition_encoding: pd.DataFrame,
    design_cfg: EncodingConfig,
    glm_formula: str,
    source_to_condition: dict[str, str],
    train_condition_labels: list[str] | tuple[str, ...] | None,
    cv_n_splits: int,
    cv_shuffle: bool,
    cv_random_state: int,
    time_window_ms: int,
    standardize_data: bool,
    n_null_repeats: int,
    run_name: str,
) -> None:
    run_output = run_encoding_workflow(
        subject_ids=subject_ids,
        subject_results_dir=None,
        loader_cfg=loader_cfg,
        condition_encoding=condition_encoding,
        design_cfg=design_cfg,
        glm_formula=glm_formula,
        source_to_condition=source_to_condition,
        train_condition_labels=train_condition_labels,
        overwrite=True,
        cv_n_splits=cv_n_splits,
        cv_shuffle=cv_shuffle,
        cv_random_state=cv_random_state,
        time_window_ms=time_window_ms,
        standardize_data=standardize_data,
        n_null_repeats=n_null_repeats,
        results_dir=None,
        run_name=run_name,
        config_payload=None,
        topography=None,
        log_path=None,
    )
    completed_subjects = run_output["subject_summary_df"]["subject"].astype(str).tolist()
    replace_subject_table_rows(
        result_file,
        subject_ids=subject_ids,
        tables={
            _ENCODING_STORE_TABLES["subject_summary"]: run_output["subject_summary_df"],
            _ENCODING_STORE_TABLES["training_pattern_strength"]: run_output["training_pattern_strength_df"],
            _ENCODING_STORE_TABLES["testing_effect_coefficients"]: run_output["testing_coefficient_df"],
            _ENCODING_STORE_TABLES["testing_effect_coefficients_wide"]: run_output["testing_coefficient_wide_df"],
            _ENCODING_STORE_TABLES["beta_patterns"]: _encoding_beta_patterns_to_long(run_output["subject_payloads"]),
        },
    )
    update_result_subject_status(
        result_file,
        completed_subject_ids=completed_subjects,
        skipped=run_output["skipped_subjects_df"],
    )


def _finalize_encoding_store(
    *,
    result_file: Path,
    loader_cfg,
    run_name: str,
    topography: dict[str, object] | None,
    time_window_ms: int,
    standardize_data: bool,
    n_null_repeats: int,
) -> dict[str, pd.DataFrame]:
    subject_summary_df = read_internal_table(result_file, _ENCODING_STORE_TABLES["subject_summary"])
    training_pattern_strength_df = read_internal_table(result_file, _ENCODING_STORE_TABLES["training_pattern_strength"])
    testing_coefficient_df = read_internal_table(result_file, _ENCODING_STORE_TABLES["testing_effect_coefficients"])
    testing_coefficient_wide_df = read_internal_table(result_file, _ENCODING_STORE_TABLES["testing_effect_coefficients_wide"])
    beta_patterns_df = read_internal_table(result_file, _ENCODING_STORE_TABLES["beta_patterns"])
    skipped_subjects_df = result_skipped_table(result_file)

    if len(subject_summary_df) == 0:
        raise RuntimeError("No completed subjects were available in the encoding result store.")

    subject_summary_df = subject_summary_df.sort_values("subject").reset_index(drop=True)
    training_pattern_strength_df = _sort_if_columns(
        training_pattern_strength_df,
        ["effect", "data_type", "subject", "fold", "null_draw", "time_ms"],
    )
    testing_coefficient_df = _sort_if_columns(
        testing_coefficient_df,
        ["effect", "condition", "subject", "fold", "trial_index", "time_ms"],
    )
    testing_coefficient_wide_df = _sort_if_columns(
        testing_coefficient_wide_df,
        ["condition", "subject", "fold", "trial_index", "time_ms"],
    )
    condition_coefficient_df = build_condition_average_coefficient_table(testing_coefficient_df)
    statuses = read_result_subject_status(result_file)
    run_summary_df = pd.DataFrame(
        {
            "name": [run_name],
            "n_subjects_requested": [len(statuses)],
            "n_subjects_completed": [len(subject_summary_df)],
            "n_subjects_skipped": [len(skipped_subjects_df)],
            "test_method_name": [TEST_METHOD_NAME],
            "test_method_version": [TEST_METHOD_VERSION],
            "time_window_ms": [int(time_window_ms)],
            "standardize_data": [bool(standardize_data)],
            "n_null_repeats": [int(n_null_repeats)],
        }
    )
    tables = {
        "subject_summary": subject_summary_df,
        "skipped": skipped_subjects_df,
        "run_summary": run_summary_df,
        "training_pattern_strength": training_pattern_strength_df,
        "testing_effect_coefficients": testing_coefficient_df,
        "testing_effect_coefficients_wide": testing_coefficient_wide_df,
        "condition_average_coefficients": condition_coefficient_df,
    }
    if topography is not None:
        tables.update(
            _encoding_topography_tables_from_long(
                beta_patterns_df=beta_patterns_df,
                topography=topography,
                loader_cfg=loader_cfg,
                first_subject=str(subject_summary_df["subject"].iloc[0]),
            )
        )
    replace_result_tables(result_file, tables)
    return tables


def _encoding_beta_patterns_to_long(
    subject_payloads: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Convert subject beta-pattern arrays into a DuckDB-friendly long table."""

    rows = []
    for payload in subject_payloads.values():
        subject = str(payload["subject"].item())
        beta_patterns = payload["raw_beta_patterns"].astype(float)
        predictor_names = payload["predictor_names"].astype(str).tolist()
        ch_names = payload["ch_names"].astype(str).tolist()
        times_ms = payload["times_s"].astype(float) * 1000.0
        n_channels = len(ch_names)
        n_times = len(times_ms)
        for fold_ix in range(beta_patterns.shape[0]):
            for effect_ix, effect in enumerate(predictor_names):
                values = beta_patterns[fold_ix, effect_ix, :, :]
                rows.append(
                    pd.DataFrame(
                        {
                            "subject": np.repeat(subject, n_channels * n_times),
                            "fold": np.repeat(fold_ix + 1, n_channels * n_times),
                            "effect": np.repeat(effect, n_channels * n_times),
                            "channel": np.repeat(ch_names, n_times),
                            "time_ms": np.tile(times_ms, n_channels),
                            "beta": values.reshape(-1),
                        }
                    )
                )
    if len(rows) == 0:
        return pd.DataFrame(columns=["subject", "fold", "effect", "channel", "time_ms", "beta"])
    return pd.concat(rows, ignore_index=True)


def _encoding_topography_tables_from_long(
    *,
    beta_patterns_df: pd.DataFrame,
    topography: dict[str, object],
    loader_cfg,
    first_subject: str,
) -> dict[str, pd.DataFrame]:
    """Build encoding topography public tables from stored beta-pattern rows."""

    windows_ms, include_window_name = _encoding_topography_windows(topography)
    value_tables = []
    for window_name, (start_ms, end_ms) in windows_ms.items():
        window_df = beta_patterns_df.loc[
            beta_patterns_df["time_ms"].between(start_ms, end_ms, inclusive="both")
        ]
        if len(window_df) == 0:
            raise ValueError(
                "No encoding beta pattern time bins were found between "
                f"{start_ms} ms and {end_ms} ms."
            )
        subject_df = (
            window_df.groupby(["subject", "effect", "channel"], as_index=False)["beta"]
            .mean()
            .rename(columns={"beta": "subject_value"})
        )
        n_subject_df = (
            subject_df.groupby("effect", as_index=False)["subject"]
            .nunique()
            .rename(columns={"subject": "n_subjects"})
        )
        group_df = (
            subject_df.groupby(["effect", "channel"], as_index=False)["subject_value"]
            .mean()
            .rename(columns={"subject_value": "raw_value"})
            .merge(n_subject_df, on="effect", how="left")
        )
        group_df = _add_effect_z_scores(group_df)
        if include_window_name:
            group_df["window_name"] = window_name
        group_df["window_start_ms"] = int(start_ms)
        group_df["window_end_ms"] = int(end_ms)
        output_cols = [
            "channel",
            "effect",
            "raw_value",
            "z_value",
            "window_start_ms",
            "window_end_ms",
            "n_subjects",
        ]
        if include_window_name:
            output_cols = [
                "channel",
                "effect",
                "window_name",
                "raw_value",
                "z_value",
                "window_start_ms",
                "window_end_ms",
                "n_subjects",
            ]
        value_tables.append(group_df.loc[:, output_cols])

    channels = beta_patterns_df["channel"].drop_duplicates().astype(str).tolist()
    return {
        "topography_values": pd.concat(value_tables, ignore_index=True),
        "topography_coords": build_topography_coord_table(
            info=load_subject_info(first_subject, loader_cfg),
            channels=channels,
        ),
    }


def _encoding_topography_windows(
    topography: dict[str, object],
) -> tuple[dict[str, tuple[int, int]], bool]:
    has_single_window = "time_window_ms" in topography
    has_named_windows = "time_windows_ms" in topography
    if has_single_window == has_named_windows:
        raise ValueError(
            "topography must include exactly one of 'time_window_ms' "
            "or 'time_windows_ms'."
        )
    if has_single_window:
        start_ms, end_ms = topography["time_window_ms"]
        return {"topography": (int(start_ms), int(end_ms))}, False
    return {
        str(window_name): (int(window_ms[0]), int(window_ms[1]))
        for window_name, window_ms in topography["time_windows_ms"].items()
    }, True


def _add_effect_z_scores(group_df: pd.DataFrame) -> pd.DataFrame:
    z_rows = []
    for _effect, effect_df in group_df.groupby("effect", sort=False):
        values = effect_df["raw_value"].to_numpy(dtype=float)
        value_std = values.std(ddof=0)
        if value_std == 0:
            z_values = np.zeros(len(values), dtype=float)
        else:
            z_values = (values - values.mean()) / value_std
        z_rows.append(effect_df.assign(z_value=z_values))
    return pd.concat(z_rows, ignore_index=True)


def _sort_if_columns(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if len(table) == 0 or not set(columns).issubset(table.columns):
        return table
    return table.sort_values(columns).reset_index(drop=True)


def run_pattern_expression(
    *,
    subject_ids: list[str],
    subject_inputs: dict[str, dict[str, object]],
    overwrite: bool,
    name: str,
    file: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run a pattern-expression analysis from script settings.

    Parameters
    ----------
    subject_ids : list[str]
        Subject IDs requested by the caller.
    subject_inputs : dict[str, dict[str, object]]
        Per-subject arrays used to build trial- and condition-level expression
        tables.
    overwrite : bool
        Whether to recompute when ``file`` already exists.
    name : str
        Analysis name recorded in the run-summary table.
    file : str | Path | None
        Optional single-file DuckDB cache for the returned result tables.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Pattern-expression result tables keyed by stable public names.
    """

    result_file = resolve_result_file(file)
    if result_file is not None:
        return _run_pattern_expression_store(
            result_file=result_file,
            subject_ids=subject_ids,
            subject_inputs=subject_inputs,
            overwrite=overwrite,
            name=name,
        )

    print(f"Running {name}")
    run_output = run_pattern_expression_workflow(
        subject_ids=subject_ids,
        subject_inputs=subject_inputs,
        subject_results_dir=None,
        overwrite=True,
        log_path=None,
    )

    run_summary_df = pd.DataFrame(
        {
            "name": [name],
            "n_subjects_requested": [len(subject_ids)],
            "n_subjects_completed": [len(run_output["trial_summary_df"])],
            "n_subjects_skipped": [len(run_output["skipped_subjects_df"])],
        }
    )
    tables = {
        "trial_expression": run_output["trial_expression_df"],
        "condition_expression": run_output["condition_expression_df"],
        "trials": run_output["trial_summary_df"],
        "skipped": run_output["skipped_subjects_df"],
        "run_summary": run_summary_df,
    }
    return tables


def _run_pattern_expression_store(
    *,
    result_file: Path,
    subject_ids: list[str],
    subject_inputs: dict[str, dict[str, object]],
    overwrite: bool,
    name: str,
) -> dict[str, pd.DataFrame]:
    init_result_store(
        result_file,
        analysis_type="pattern_expression",
        config_hash=result_config_hash({"analysis_type": "pattern_expression"}),
        run_name=name,
    )
    return run_incremental_result_store(
        result_file,
        subject_ids=subject_ids,
        overwrite=overwrite,
        process_subjects=lambda subjects_to_run: _process_pattern_expression_store_subjects(
            result_file=result_file,
            subject_ids=subjects_to_run,
            subject_inputs=subject_inputs,
        ),
        finalize=lambda: _finalize_pattern_expression_store(
            result_file=result_file,
            name=name,
        ),
    )


def _process_pattern_expression_store_subjects(
    *,
    result_file: Path,
    subject_ids: list[str],
    subject_inputs: dict[str, dict[str, object]],
) -> None:
    run_output = run_pattern_expression_workflow(
        subject_ids=subject_ids,
        subject_inputs=subject_inputs,
        subject_results_dir=None,
        overwrite=True,
        log_path=None,
    )
    completed_subjects = run_output["trial_summary_df"]["subject"].astype(str).tolist()
    replace_subject_table_rows(
        result_file,
        subject_ids=subject_ids,
        tables={
            _PATTERN_STORE_TABLES["trial_expression"]: run_output["trial_expression_df"],
            _PATTERN_STORE_TABLES["condition_expression"]: run_output["condition_expression_df"],
            _PATTERN_STORE_TABLES["trials"]: run_output["trial_summary_df"],
        },
    )
    update_result_subject_status(
        result_file,
        completed_subject_ids=completed_subjects,
        skipped=run_output["skipped_subjects_df"],
    )


def _finalize_pattern_expression_store(
    *,
    result_file: Path,
    name: str,
) -> dict[str, pd.DataFrame]:
    trial_expression_df = read_internal_table(result_file, _PATTERN_STORE_TABLES["trial_expression"])
    condition_expression_df = read_internal_table(result_file, _PATTERN_STORE_TABLES["condition_expression"])
    trial_summary_df = read_internal_table(result_file, _PATTERN_STORE_TABLES["trials"])
    skipped_subjects_df = result_skipped_table(result_file)
    if len(trial_summary_df) == 0:
        raise RuntimeError("No completed subjects were available in the pattern-expression result store.")
    trial_expression_df = _sort_if_columns(
        trial_expression_df,
        ["subject", "condition", "effect", "trial_index", "time"],
    )
    condition_expression_df = _sort_if_columns(
        condition_expression_df,
        ["subject", "condition", "effect", "time"],
    )
    trial_summary_df = trial_summary_df.sort_values("subject").reset_index(drop=True)
    statuses = read_result_subject_status(result_file)
    run_summary_df = pd.DataFrame(
        {
            "name": [name],
            "n_subjects_requested": [len(statuses)],
            "n_subjects_completed": [len(trial_summary_df)],
            "n_subjects_skipped": [len(skipped_subjects_df)],
        }
    )
    tables = {
        "trial_expression": trial_expression_df,
        "condition_expression": condition_expression_df,
        "trials": trial_summary_df,
        "skipped": skipped_subjects_df,
        "run_summary": run_summary_df,
    }
    replace_result_tables(result_file, tables)
    return tables


def _public_table_names(run_output: dict[str, object]) -> dict[str, pd.DataFrame]:
    """Return public table names for encoding workflow outputs."""

    name_map = {
        "subject_summary_df": "subject_summary",
        "skipped_subjects_df": "skipped",
        "run_summary_df": "run_summary",
        "training_pattern_strength_df": "training_pattern_strength",
        "testing_coefficient_df": "testing_effect_coefficients",
        "testing_coefficient_wide_df": "testing_effect_coefficients_wide",
        "condition_coefficient_df": "condition_average_coefficients",
        "topography_values_df": "topography_values",
        "topography_coords_df": "topography_coords",
    }
    return {
        public_name: table
        for internal_name, public_name in name_map.items()
        if isinstance((table := run_output.get(internal_name)), pd.DataFrame)
    }
