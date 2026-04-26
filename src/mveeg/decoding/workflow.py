"""Public workflow facade for decoding analyses.

These helpers keep scripts focused on research decisions while delegating path,
subject-loop, and output-export details to smaller workflow modules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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


def run_decoding(
    *,
    base_dir: str | Path,
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
    experiment_name: str | None = None,
    results_subdir: str | None = None,
    cond_col: str = "label",
) -> dict[str, object]:
    """Run one decoding analysis from script-level settings.

    Parameters
    ----------
    base_dir : str | Path
        Project root used to create the decoding output folders.
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
        Whether to rerun subjects that already have saved subject-level outputs.
    name : str
        Analysis name used for display and output folders.
    train_conditions : dict[str, list[str]]
        Training labels and the task conditions they include.
    test_conditions : dict[str, list[str]]
        Output groups kept for testing and summaries.
    topo_windows_ms : dict[str, tuple[int, int]]
        Named time windows exported for topography summaries.
    experiment_name : str | None
        Experiment name used to locate the derivatives files. If ``None``, the
        final folder name from ``data_dir`` is used.
    results_subdir : str | None
        Experiment-specific folder written below `results`. If ``None``, the
        final folder name from ``data_dir`` is used.
    cond_col : str
        Metadata column used to read condition labels.

    Returns
    -------
    dict[str, object]
        Paths, saved configuration, decoding outputs, and a short summary
        table for the current analysis.
    """

    experiment_name, results_subdir = infer_experiment_settings(
        data_dir=data_dir,
        experiment_name=experiment_name,
        results_subdir=results_subdir,
    )

    run_paths = prepare_decoding_paths(base_dir, name, results_subdir=results_subdir)

    cfg = DecodingConfig(
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

    config_to_save = save_decoding_config(run_paths["results_dir"], cfg)

    print(f"Running {name}")
    print(f"Detailed log file: {run_paths['log_path']}")
    available_subject_ids = discover_subject_ids(data_dir)

    run_output = run_decoding_workflow(
        subject_ids=subject_ids,
        available_subject_ids=available_subject_ids,
        cfg=cfg,
        subject_results_dir=run_paths["subject_results_dir"],
        overwrite=overwrite,
        log_path=run_paths["log_path"],
    )

    export_decoding_outputs(
        run_output=run_output,
        cfg=cfg,
        results_dir=run_paths["results_dir"],
        figures_dir=run_paths["figures_dir"],
        topo_windows_ms=topo_windows_ms,
    )

    summary_df = pd.DataFrame(
        {
            "name": [name],
            "classifier_backend": [classifier["backend"]],
            "classifier_model": [classifier["model_name"]],
            "n_subjects_requested": [len(subject_ids)],
            "n_subjects_completed": [len(run_output["trial_summary_df"])],
            "n_subjects_skipped": [len(run_output["skipped_subjects_df"])],
        }
    )

    return {
        "paths": run_paths,
        "config": config_to_save,
        "run_output": run_output,
        "summary_df": summary_df,
    }


def run_generalization_decoding(
    *,
    base_dir: str | Path,
    data_dir: str | Path,
    subject_ids: list[str],
    trial_filters: dict[str, object],
    decoding_params: dict[str, object],
    classifier: dict[str, object],
    overwrite: bool,
    name: str,
    train_conditions: dict[str, list[str]],
    test_conditions: dict[str, list[str]],
    experiment_name: str | None = None,
    results_subdir: str | None = None,
    cond_col: str = "label",
) -> dict[str, object]:
    """Run one all-window generalization analysis from script settings.

    Parameters
    ----------
    base_dir : str | Path
        Project root used to create the decoding output folders.
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
        Whether to rerun subjects even when the group-level CSV already exists.
    name : str
        Analysis name used for display and output folders.
    train_conditions : dict[str, list[str]]
        Training labels and the task conditions they include.
    test_conditions : dict[str, list[str]]
        Output groups kept for testing and summaries.
    experiment_name : str | None
        Experiment name used to locate the derivatives files. If ``None``, the
        final folder name from ``data_dir`` is used.
    results_subdir : str | None
        Experiment-specific folder written below `results`. If ``None``, the
        final folder name from ``data_dir`` is used.
    cond_col : str
        Metadata column used to read condition labels.

    Returns
    -------
    dict[str, object]
        Paths, saved configuration, decoding outputs, and a short summary
        table for the current analysis.
    """

    experiment_name, results_subdir = infer_experiment_settings(
        data_dir=data_dir,
        experiment_name=experiment_name,
        results_subdir=results_subdir,
    )

    run_paths = prepare_decoding_paths(base_dir, name, results_subdir=results_subdir)

    cfg = DecodingConfig(
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

    config_to_save = save_decoding_config(run_paths["results_dir"], cfg)
    config_to_save["generalization"] = {
        "mode": "all_time_windows",
    }
    with open(run_paths["results_dir"] / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2)

    print(f"Running {name}")
    print(f"Detailed log file: {run_paths['log_path']}")
    available_subject_ids = discover_subject_ids(data_dir)

    run_output = run_generalization_workflow(
        subject_ids=subject_ids,
        available_subject_ids=available_subject_ids,
        cfg=cfg,
        subject_results_dir=run_paths["subject_results_dir"],
        overwrite=overwrite,
        log_path=run_paths["log_path"],
    )

    trial_path = run_paths["results_dir"] / CORE_OUTPUT_FILES["trial_summary"]
    result_path = run_paths["results_dir"] / CORE_OUTPUT_FILES["generalization_accuracy_cv"]
    skipped_path = run_paths["results_dir"] / CORE_OUTPUT_FILES["skipped_subjects"]
    run_output["trial_summary_df"].to_csv(trial_path, index=False)
    run_output["accuracy_df"].to_csv(result_path, index=False)
    if len(run_output["skipped_subjects_df"]) > 0:
        run_output["skipped_subjects_df"].to_csv(skipped_path, index=False)
    elif skipped_path.exists():
        skipped_path.unlink()

    summary_df = pd.DataFrame(
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

    return {
        "paths": run_paths,
        "config": config_to_save,
        "run_output": run_output,
        "summary_df": summary_df,
    }
