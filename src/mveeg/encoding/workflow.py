"""Public workflow facade for encoding analyses.

These helpers keep encoding scripts focused on analysis decisions while smaller
workflow modules handle paths, design checks, pattern exports, and model runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def run_encoding(
    *,
    base_dir: str | Path,
    data_dir: str | Path,
    subject_ids: list[str],
    trial_filters: dict[str, object],
    encoding_params: dict[str, object],
    condition_encoding: pd.DataFrame,
    condition_label_map: dict[str, list[str]],
    glm_formula: str,
    overwrite: bool,
    name: str,
    train_condition_labels: list[str] | tuple[str, ...] | None = None,
    topography: dict[str, object] | None = None,
    experiment_name: str | None = None,
    results_subdir: str | None = None,
    cond_col: str = "label",
) -> dict[str, object]:
    """Run one encoding analysis from grouped script-level settings.

    Parameters
    ----------
    base_dir : str | Path
        Project root used to create the encoding output folder.
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
        Whether to rerun subjects that already have saved subject-level outputs.
    name : str
        Analysis name used for display and output folders.
    train_condition_labels : list[str] | tuple[str, ...] | None
        Optional analysis condition labels used for model fitting.
    topography : dict[str, object] | None
        Optional encoding topography export settings. Use ``time_window_ms``
        as ``(start_ms, end_ms)`` for one window, or ``time_windows_ms`` as a
        named window dictionary for multi-window export.
    experiment_name : str | None
        Experiment name used to locate derivative files. If ``None``, the final
        folder name from ``data_dir`` is used.
    results_subdir : str | None
        Folder name written below ``results``. If ``None``, the final folder
        name from ``data_dir`` is used.
    cond_col : str
        Metadata column used to read raw condition labels.

    Returns
    -------
    dict[str, object]
        Paths, saved configuration, encoding outputs, and a short summary table
        for the current analysis.
    """

    experiment_name, results_subdir = infer_experiment_settings(
        data_dir=data_dir,
        experiment_name=experiment_name,
        results_subdir=results_subdir,
    )
    run_paths = prepare_encoding_paths(base_dir, name, results_subdir=results_subdir)

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

    config_to_save = {
        "trial_filters": trial_filters,
        "encoding_params": encoding_params,
        "condition_label_map": condition_label_map,
        "condition_encoding": condition_encoding.to_dict(orient="list"),
        "condition_column": cond_col,
        "glm_formula": glm_formula,
        "train_condition_labels": train_condition_labels,
        "topography": topography,
        "overwrite": overwrite,
        "run_name": name,
    }

    print(f"Running {name}")
    print(f"Detailed log file: {run_paths['log_path']}")
    run_output = run_encoding_workflow(
        subject_ids=subject_ids,
        subject_results_dir=run_paths["subject_results_dir"],
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
        results_dir=run_paths["results_dir"],
        run_name=name,
        config_payload=config_to_save,
        topography=topography,
        log_path=run_paths["log_path"],
    )

    summary_df = pd.DataFrame(
        {
            "name": [name],
            "n_subjects_requested": [len(subject_ids)],
            "n_subjects_completed": [len(run_output["subject_summary_df"])],
            "n_subjects_skipped": [len(run_output["skipped_subjects_df"])],
        }
    )

    return {
        "paths": run_paths,
        "config": config_to_save,
        "run_output": run_output,
        "summary_df": summary_df,
    }


def run_pattern_expression(
    *,
    base_dir: str | Path,
    subject_ids: list[str],
    subject_inputs: dict[str, dict[str, object]],
    overwrite: bool,
    name: str,
    results_subdir: str = "main",
) -> dict[str, object]:
    """Run a pattern-expression export analysis from script settings."""

    run_paths = prepare_encoding_paths(base_dir, name, results_subdir=results_subdir)

    print(f"Running {name}")
    print(f"Detailed log file: {run_paths['log_path']}")

    run_output = run_pattern_expression_workflow(
        subject_ids=subject_ids,
        subject_inputs=subject_inputs,
        subject_results_dir=run_paths["subject_results_dir"],
        overwrite=overwrite,
        log_path=run_paths["log_path"],
    )

    export_encoding_outputs(
        run_output=run_output,
        results_dir=run_paths["results_dir"],
    )

    summary_df = pd.DataFrame(
        {
            "name": [name],
            "n_subjects_requested": [len(subject_ids)],
            "n_subjects_completed": [len(run_output["trial_summary_df"])],
            "n_subjects_skipped": [len(run_output["skipped_subjects_df"])],
        }
    )

    return {
        "paths": run_paths,
        "run_output": run_output,
        "summary_df": summary_df,
    }
