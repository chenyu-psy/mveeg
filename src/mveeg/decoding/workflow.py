"""Public workflow facade for decoding analyses.

These helpers keep scripts focused on research decisions while delegating path,
subject-loop, and output-export details to smaller workflow modules.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .._shared.metadata import metadata_transform_spec
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
from .config import (
    ConditionConfig,
    DatasetConfig,
    DecodeParamConfig,
    DecodingConfig,
    ModelConfig,
    TrialFilterConfig,
)
from .io import get_subject_ids as discover_subject_ids
from .workflow_outputs import (
    CORE_OUTPUT_FILES,
    build_decoding_result_tables,
    build_generalization_accuracy_table,
    export_decoding_outputs,
)
from .workflow_paths import (
    infer_experiment_settings,
    prepare_decoding_paths,
    save_decoding_config,
)
from .workflow_subjects import (
    run_decoding_workflow,
    run_generalization_workflow,
)


__all__ = [
    "CORE_OUTPUT_FILES",
    "build_generalization_accuracy_table",
    "export_decoding_outputs",
    "infer_experiment_settings",
    "prepare_decoding_paths",
    "run_decoding",
    "run_decoding_workflow",
    "run_generalization_decoding",
    "run_generalization_workflow",
    "save_decoding_config",
]


_DECODING_STORE_TABLES = {
    "accuracy": "_mveeg_subject_accuracy",
    "hyperplane": "_mveeg_subject_hyperplane",
    "trials": "_mveeg_subject_trials",
    "pattern": "_mveeg_subject_pattern",
}

_GENERALIZATION_STORE_TABLES = {
    "accuracy": "_mveeg_subject_accuracy",
    "trials": "_mveeg_subject_trials",
}


def run_decoding(
    *,
    data_dir: str | Path,
    subject_ids: list[str],
    trial_filters: dict[str, object],
    decoding_params: dict[str, object],
    classifier: dict[str, object],
    overwrite: bool,
    name: str,
    train_conditions: dict[str, list[str]],
    test_conditions: dict[str, list[str]],
    topo_windows_ms: dict[str, tuple[int, int]],
    file: str | Path | None = None,
    experiment_name: str | None = None,
    cond_col: str = "label",
    metadata_transform=None,
    metadata_transform_name: str | None = None,
    metadata_transform_version: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Run one decoding analysis from script-level settings.

    Parameters
    ----------
    data_dir : str | Path
        Preprocessed data folder used for the current experiment.
    subject_ids : list[str]
        Subject IDs requested by the caller.
    trial_filters : dict[str, object]
        Shared trial filters used for this run.
    decoding_params : dict[str, object]
        Shared decoding parameters used for this run.
    classifier : dict[str, object]
        Classifier specification used for this run.
    overwrite : bool
        Whether to recompute when ``file`` already exists.
    name : str
        Analysis name used for display and output folders.
    train_conditions : dict[str, list[str]]
        Training labels and the task conditions they include.
    test_conditions : dict[str, list[str]]
        Output groups kept for testing and summaries.
    topo_windows_ms : dict[str, tuple[int, int]]
        Named time windows exported as channel-value summaries for R
        topography plotting.
    file : str | Path | None
        Optional single-file DuckDB analysis store. Paths without a suffix
        receive ``.duckdb``. Existing stores reuse completed subjects and only
        process requested subjects that are missing, pending, skipped, or
        explicitly overwritten.
    experiment_name : str | None
        Experiment name used to locate the derivatives files. If ``None``, the
        final folder name from ``data_dir`` is used.
    cond_col : str
        Metadata column used to read condition labels.
    metadata_transform : callable | None
        Row-preserving metadata transform applied after keyed artifact merge
        and before trial filtering and label construction.
    metadata_transform_name, metadata_transform_version : str | None
        Stable transform identity included in result fingerprints. Both are
        required when ``metadata_transform`` is provided.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Decoding result tables keyed by stable public names.
    """

    transform_spec = metadata_transform_spec(
        metadata_transform,
        name=metadata_transform_name,
        version=metadata_transform_version,
    )
    experiment_name, _ = infer_experiment_settings(
        data_dir=data_dir,
        experiment_name=experiment_name,
        results_subdir=None,
    )
    cfg = _build_decoding_config(
        data_dir=data_dir,
        experiment_name=experiment_name,
        trial_filters=trial_filters,
        decoding_params=decoding_params,
        classifier=classifier,
        train_conditions=train_conditions,
        test_conditions=test_conditions,
        cond_col=cond_col,
    )

    result_file = resolve_result_file(file)
    if result_file is not None:
        return _run_decoding_store(
            result_file=result_file,
            subject_ids=subject_ids,
            cfg=cfg,
            classifier=classifier,
            overwrite=overwrite,
            name=name,
            topo_windows_ms=topo_windows_ms,
            metadata_transform=metadata_transform,
            transform_spec=transform_spec,
        )

    print(f"Running {name}")
    available_subject_ids = discover_subject_ids(data_dir)

    run_output = run_decoding_workflow(
        subject_ids=subject_ids,
        available_subject_ids=available_subject_ids,
        cfg=cfg,
        subject_results_dir=None,
        overwrite=True,
        metadata_transform=metadata_transform,
        log_path=None,
    )

    run_summary_df = pd.DataFrame(
        {
            "name": [name],
            "classifier_backend": [classifier["backend"]],
            "classifier_model": [classifier["model_name"]],
            "n_subjects_requested": [len(subject_ids)],
            "n_subjects_completed": [len(run_output["trial_summary_df"])],
            "n_subjects_skipped": [len(run_output["skipped_subjects_df"])],
        }
    )
    tables = build_decoding_result_tables(
        run_output=run_output,
        cfg=cfg,
        topo_windows_ms=topo_windows_ms,
        run_summary_df=run_summary_df,
    )

    return tables


def run_generalization_decoding(
    *,
    data_dir: str | Path,
    subject_ids: list[str],
    trial_filters: dict[str, object],
    decoding_params: dict[str, object],
    classifier: dict[str, object],
    overwrite: bool,
    name: str,
    train_conditions: dict[str, list[str]],
    test_conditions: dict[str, list[str]],
    file: str | Path | None = None,
    experiment_name: str | None = None,
    cond_col: str = "label",
    metadata_transform=None,
    metadata_transform_name: str | None = None,
    metadata_transform_version: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Run one all-window generalization analysis from script settings.

    Parameters
    ----------
    data_dir : str | Path
        Preprocessed data folder used for the current experiment.
    subject_ids : list[str]
        Subject IDs requested by the caller.
    trial_filters : dict[str, object]
        Shared trial filters used for this run.
    decoding_params : dict[str, object]
        Shared decoding parameters used for this run.
    classifier : dict[str, object]
        Classifier specification used for this run.
    overwrite : bool
        Whether to recompute when ``file`` already exists.
    name : str
        Analysis name used for display and output folders.
    train_conditions : dict[str, list[str]]
        Training labels and the task conditions they include.
    test_conditions : dict[str, list[str]]
        Output groups kept for testing and summaries.
    file : str | Path | None
        Optional single-file DuckDB analysis store. Paths without a suffix
        receive ``.duckdb``. Existing stores reuse completed subjects and only
        process requested subjects that are missing, pending, skipped, or
        explicitly overwritten.
    experiment_name : str | None
        Experiment name used to locate the derivatives files. If ``None``, the
        final folder name from ``data_dir`` is used.
    cond_col : str
        Metadata column used to read condition labels.
    metadata_transform : callable | None
        Row-preserving metadata transform applied after keyed artifact merge
        and before trial filtering and label construction.
    metadata_transform_name, metadata_transform_version : str | None
        Stable transform identity included in result fingerprints. Both are
        required when ``metadata_transform`` is provided.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Generalization result tables keyed by stable public names.
    """

    transform_spec = metadata_transform_spec(
        metadata_transform,
        name=metadata_transform_name,
        version=metadata_transform_version,
    )
    experiment_name, _ = infer_experiment_settings(
        data_dir=data_dir,
        experiment_name=experiment_name,
        results_subdir=None,
    )
    cfg = _build_decoding_config(
        data_dir=data_dir,
        experiment_name=experiment_name,
        trial_filters=trial_filters,
        decoding_params=decoding_params,
        classifier=classifier,
        train_conditions=train_conditions,
        test_conditions=test_conditions,
        cond_col=cond_col,
    )

    result_file = resolve_result_file(file)
    if result_file is not None:
        return _run_generalization_store(
            result_file=result_file,
            subject_ids=subject_ids,
            cfg=cfg,
            classifier=classifier,
            overwrite=overwrite,
            name=name,
            metadata_transform=metadata_transform,
            transform_spec=transform_spec,
        )

    print(f"Running {name}")
    available_subject_ids = discover_subject_ids(data_dir)

    run_output = run_generalization_workflow(
        subject_ids=subject_ids,
        available_subject_ids=available_subject_ids,
        cfg=cfg,
        subject_results_dir=None,
        overwrite=True,
        metadata_transform=metadata_transform,
        log_path=None,
    )

    run_summary_df = pd.DataFrame(
        {
            "name": [name],
            "classifier_backend": [classifier["backend"]],
            "classifier_model": [classifier["model_name"]],
            "n_subjects_requested": [len(subject_ids)],
            "n_subjects_completed": [len(run_output["trial_summary_df"])],
            "n_subjects_skipped": [len(run_output["skipped_subjects_df"])],
            "generalization_mode": ["all_time_windows"],
            "n_time_windows": [len(run_output["window_times_ms"])],
        }
    )
    tables = {
        "accuracy": run_output["accuracy_df"],
        "trials": run_output["trial_summary_df"],
        "skipped": run_output["skipped_subjects_df"],
        "run_summary": run_summary_df,
    }

    return tables


def _build_decoding_config(
    *,
    data_dir: str | Path,
    experiment_name: str,
    trial_filters: dict[str, object],
    decoding_params: dict[str, object],
    classifier: dict[str, object],
    train_conditions: dict[str, list[str]],
    test_conditions: dict[str, list[str]],
    cond_col: str,
) -> DecodingConfig:
    """Build the shared decoding config used by both public facades."""

    return DecodingConfig(
        dataset=DatasetConfig(
            data_dir=data_dir,
            experiment_name=experiment_name,
        ),
        conditions=ConditionConfig(
            train_cond=train_conditions,
            test_cond=test_conditions,
            cond_col=cond_col,
        ),
        filters=TrialFilterConfig(
            qc_col=trial_filters["qc_col"],
            keep_qc=tuple(trial_filters["keep_qc"]),
            exclude_metadata=trial_filters["exclude_metadata"],
        ),
        decode=DecodeParamConfig(
            crop_time=decoding_params["crop_time"],
            time_window_ms=decoding_params["time_window_ms"],
            trial_bin_size=decoding_params["trial_bin_size"],
            n_splits=decoding_params["n_splits"],
            n_repeats=decoding_params["n_repeats"],
            n_jobs=decoding_params["n_jobs"],
            drop_channel_types=decoding_params["drop_channel_types"],
            drop_channels=decoding_params["drop_channels"],
        ),
        model=ModelConfig(
            classifier_spec=classifier,
        ),
    )


def _process_decoding_store_subjects(
    *,
    result_file: Path,
    subject_ids: list[str],
    cfg: DecodingConfig,
    metadata_transform,
) -> None:
    """Process requested same-time decoding subjects into the result store."""

    available_subject_ids = discover_subject_ids(cfg.dataset.data_dir)
    run_output = run_decoding_workflow(
        subject_ids=subject_ids,
        available_subject_ids=available_subject_ids,
        cfg=cfg,
        subject_results_dir=None,
        overwrite=True,
        metadata_transform=metadata_transform,
        log_path=None,
    )
    completed_subjects = run_output["trial_summary_df"]["subject"].astype(str).tolist()
    replace_subject_table_rows(
        result_file,
        subject_ids=subject_ids,
        tables={
            _DECODING_STORE_TABLES["accuracy"]: run_output["accuracy_df"],
            _DECODING_STORE_TABLES["hyperplane"]: run_output["hyperplane_df"],
            _DECODING_STORE_TABLES["trials"]: run_output["trial_summary_df"],
            _DECODING_STORE_TABLES["pattern"]: run_output["pattern_df"],
        },
    )
    update_result_subject_status(
        result_file,
        completed_subject_ids=completed_subjects,
        skipped=run_output["skipped_subjects_df"],
    )


def _run_decoding_store(
    *,
    result_file: Path,
    subject_ids: list[str],
    cfg: DecodingConfig,
    classifier: dict[str, object],
    overwrite: bool,
    name: str,
    topo_windows_ms: dict[str, tuple[int, int]],
    metadata_transform,
    transform_spec: dict[str, str] | None,
) -> dict[str, pd.DataFrame]:
    config_hash = result_config_hash(
        {
            "analysis_type": "decoding",
            "config": cfg.to_dict(),
            "topo_windows_ms": topo_windows_ms,
            "metadata_transform": transform_spec,
        }
    )
    init_result_store(
        result_file,
        analysis_type="decoding",
        config_hash=config_hash,
        run_name=name,
    )
    return run_incremental_result_store(
        result_file,
        subject_ids=subject_ids,
        overwrite=overwrite,
        process_subjects=lambda subjects_to_run: _process_decoding_store_subjects(
            result_file=result_file,
            subject_ids=subjects_to_run,
            cfg=cfg,
            metadata_transform=metadata_transform,
        ),
        finalize=lambda: _finalize_decoding_store(
            result_file=result_file,
            cfg=cfg,
            classifier=classifier,
            name=name,
            topo_windows_ms=topo_windows_ms,
        ),
    )


def _finalize_decoding_store(
    *,
    result_file: Path,
    cfg: DecodingConfig,
    classifier: dict[str, object],
    name: str,
    topo_windows_ms: dict[str, tuple[int, int]],
) -> dict[str, pd.DataFrame]:
    accuracy_df = read_internal_table(result_file, _DECODING_STORE_TABLES["accuracy"])
    hyperplane_df = read_internal_table(result_file, _DECODING_STORE_TABLES["hyperplane"])
    trial_summary_df = read_internal_table(result_file, _DECODING_STORE_TABLES["trials"])
    pattern_df = read_internal_table(result_file, _DECODING_STORE_TABLES["pattern"])
    skipped_subjects_df = result_skipped_table(result_file)

    if len(trial_summary_df) == 0:
        raise RuntimeError("No completed subjects were available in the decoding result store.")

    topography_tables = build_decoding_result_tables(
        run_output={
            "trial_summary_df": trial_summary_df,
            "skipped_subjects_df": skipped_subjects_df,
            "accuracy_df": accuracy_df,
            "hyperplane_df": hyperplane_df,
            "pattern_df": pattern_df,
            "reference_ch_names": pd.unique(pattern_df["channel"]).tolist(),
            "topography_subject_id": str(trial_summary_df["subject"].iloc[0]),
        },
        cfg=cfg,
        topo_windows_ms=topo_windows_ms,
    )
    statuses = read_result_subject_status(result_file)
    run_summary_df = pd.DataFrame(
        {
            "name": [name],
            "classifier_backend": [classifier["backend"]],
            "classifier_model": [classifier["model_name"]],
            "n_subjects_requested": [len(statuses)],
            "n_subjects_completed": [len(trial_summary_df)],
            "n_subjects_skipped": [len(skipped_subjects_df)],
        }
    )
    tables = {
        "accuracy": accuracy_df,
        "hyperplane": hyperplane_df,
        "topography_values": topography_tables["topography_values"],
        "topography_coords": topography_tables["topography_coords"],
        "trials": trial_summary_df,
        "skipped": skipped_subjects_df,
        "run_summary": run_summary_df,
    }
    replace_result_tables(result_file, tables)
    return tables


def _run_generalization_store(
    *,
    result_file: Path,
    subject_ids: list[str],
    cfg: DecodingConfig,
    classifier: dict[str, object],
    overwrite: bool,
    name: str,
    metadata_transform,
    transform_spec: dict[str, str] | None,
) -> dict[str, pd.DataFrame]:
    config_hash = result_config_hash(
        {
            "analysis_type": "generalization_decoding",
            "config": cfg.to_dict(),
            "metadata_transform": transform_spec,
        }
    )
    init_result_store(
        result_file,
        analysis_type="generalization_decoding",
        config_hash=config_hash,
        run_name=name,
    )
    return run_incremental_result_store(
        result_file,
        subject_ids=subject_ids,
        overwrite=overwrite,
        process_subjects=lambda subjects_to_run: _process_generalization_store_subjects(
            result_file=result_file,
            subject_ids=subjects_to_run,
            cfg=cfg,
            metadata_transform=metadata_transform,
        ),
        finalize=lambda: _finalize_generalization_store(
            result_file=result_file,
            classifier=classifier,
            name=name,
        ),
    )


def _process_generalization_store_subjects(
    *,
    result_file: Path,
    subject_ids: list[str],
    cfg: DecodingConfig,
    metadata_transform,
) -> None:
    """Process requested generalization subjects into the result store."""

    available_subject_ids = discover_subject_ids(cfg.dataset.data_dir)
    run_output = run_generalization_workflow(
        subject_ids=subject_ids,
        available_subject_ids=available_subject_ids,
        cfg=cfg,
        subject_results_dir=None,
        overwrite=True,
        metadata_transform=metadata_transform,
        log_path=None,
    )
    completed_subjects = run_output["trial_summary_df"]["subject"].astype(str).tolist()
    replace_subject_table_rows(
        result_file,
        subject_ids=subject_ids,
        tables={
            _GENERALIZATION_STORE_TABLES["accuracy"]: run_output["accuracy_df"],
            _GENERALIZATION_STORE_TABLES["trials"]: run_output["trial_summary_df"],
        },
    )
    update_result_subject_status(
        result_file,
        completed_subject_ids=completed_subjects,
        skipped=run_output["skipped_subjects_df"],
    )


def _finalize_generalization_store(
    *,
    result_file: Path,
    classifier: dict[str, object],
    name: str,
) -> dict[str, pd.DataFrame]:
    accuracy_df = read_internal_table(result_file, _GENERALIZATION_STORE_TABLES["accuracy"])
    trial_summary_df = read_internal_table(result_file, _GENERALIZATION_STORE_TABLES["trials"])
    skipped_subjects_df = result_skipped_table(result_file)
    if len(trial_summary_df) == 0:
        raise RuntimeError("No completed subjects were available in the generalization result store.")
    statuses = read_result_subject_status(result_file)
    run_summary_df = pd.DataFrame(
        {
            "name": [name],
            "classifier_backend": [classifier["backend"]],
            "classifier_model": [classifier["model_name"]],
            "n_subjects_requested": [len(statuses)],
            "n_subjects_completed": [len(trial_summary_df)],
            "n_subjects_skipped": [len(skipped_subjects_df)],
            "generalization_mode": ["all_time_windows"],
            "n_time_windows": [accuracy_df["train_time_ms"].nunique()],
        }
    )
    tables = {
        "accuracy": accuracy_df,
        "trials": trial_summary_df,
        "skipped": skipped_subjects_df,
        "run_summary": run_summary_df,
    }
    replace_result_tables(result_file, tables)
    return tables
