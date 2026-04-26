"""Workflow helpers for decoding analyses.

These helpers keep scripts focused on research decisions while handling
the repeated subject loop, logging, and output export in one place.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .._shared.topography import save_window_topography_set
from .config import (
    ConditionConfig,
    DatasetConfig,
    DecodeParamConfig,
    DecodingConfig,
    ModelConfig,
    TrialFilterConfig,
)
from .io import (
    generalization_subject_result_exists,
    get_subject_ids as discover_subject_ids,
    list_saved_generalization_subject_ids,
    list_saved_subject_ids,
    load_saved_generalization_subject_results,
    load_saved_subject_results,
    load_subject_decoding_data,
    load_subject_info,
    load_subject_metadata,
    save_generalization_subject_results,
    save_subject_results,
    subject_result_exists,
)
from .prepare import average_time_windows, build_time_windows
from .run import run_subject_decoding, run_subject_generalization, run_subject_hyperplane
from .summaries import (
    build_accuracy_table,
    build_channel_contrib_table,
    build_hyperplane_table,
)


CORE_OUTPUT_FILES = {
    "trial_summary": "trials.csv",
    "accuracy_cv": "acc_cv.csv",
    "generalization_accuracy_cv": "acc_generalization_cv.csv",
    "hyperplane_subject": "dist_subject.csv",
    "skipped_subjects": "skipped.csv",
    "topography_manifest": "topo_manifest.csv",
}


def _normalize_subject_labels(subjects: list[str] | pd.Series) -> list[str]:
    """Return subject labels as plain strings without a ``sub-`` prefix.

    Parameters
    ----------
    subjects : list[str] | pd.Series
        Subject labels collected from the caller request or a saved summary
        table. Labels may appear as ``"2001"`` or ``"sub-2001"``.

    Returns
    -------
    list[str]
        Subject labels normalized to bare ID strings such as ``"2001"``.
    """

    normalized = []
    for subject in subjects:
        subject_str = str(subject).strip()
        if subject_str.startswith("sub-"):
            subject_str = subject_str[4:]
        normalized.append(subject_str)
    return normalized


def _ordered_unique_subject_ids(subject_ids: list[str]) -> list[str]:
    """Return normalized subject IDs with duplicates removed in input order.

    Parameters
    ----------
    subject_ids : list[str]
        Subject IDs from the caller request, available-data scan, or cache
        listing.

    Returns
    -------
    list[str]
        Normalized subject IDs with duplicates removed while preserving the
        original order.
    """

    ordered_subjects = []
    seen_subjects = set()
    for subject_id in _normalize_subject_labels(subject_ids):
        if subject_id in seen_subjects:
            continue
        ordered_subjects.append(subject_id)
        seen_subjects.add(subject_id)
    return ordered_subjects


def _build_subject_run_plan(
    requested_subject_ids: list[str],
    available_subject_ids: list[str],
    cached_subject_ids: list[str],
) -> dict[str, object]:
    """Return the processing and retention sets for one workflow run.

    Parameters
    ----------
    requested_subject_ids : list[str]
        Subject IDs requested by the caller after any manual filtering.
    available_subject_ids : list[str]
        Subject IDs currently discoverable from the data directory.
    cached_subject_ids : list[str]
        Subject IDs that already have a complete subject-level cache set for
        this run directory.

    Returns
    -------
    dict[str, object]
        Plan describing the processing subset and the seed subject set used for
        the final rebuilt outputs.
    """

    requested_subjects = _ordered_unique_subject_ids(requested_subject_ids)
    available_subjects = _ordered_unique_subject_ids(available_subject_ids)
    cached_subjects = _ordered_unique_subject_ids(cached_subject_ids)
    available_set = set(available_subjects)
    requested_subjects = [subject_id for subject_id in requested_subjects if subject_id in available_set]

    is_full_run = len(requested_subjects) == len(available_subjects) and set(requested_subjects) == available_set
    if is_full_run:
        keep_seed_subjects = available_subjects
    else:
        keep_seed_subjects = sorted(set(cached_subjects).union(requested_subjects))

    return {
        "requested_subjects": requested_subjects,
        "available_subjects": available_subjects,
        "cached_subjects": cached_subjects,
        "subjects_to_process": requested_subjects,
        "keep_seed_subjects": keep_seed_subjects,
        "is_full_run": is_full_run,
    }


def _process_subjects(
    *,
    subject_ids: list[str],
    cfg: DecodingConfig,
    progress_total: int,
    log_label: str,
    log_path: str | Path | None,
    process_one_subject,
) -> pd.DataFrame:
    """Process one requested subject list using the provided callback.

    Parameters
    ----------
    subject_ids : list[str]
        Subject IDs to process during the current run.
    cfg : DecodingConfig
        Decoding settings for the current analysis.
    progress_total : int
        Total progress-bar steps per subject.
    log_label : str
        Short label written to the detailed log header.
    log_path : str | Path | None
        Optional path for the detailed technical log.
    process_one_subject : callable
        Callback that accepts ``(subject_id, progress_bar)`` and returns
        ``(result_bundle, used_saved_result)``. The callback may raise to mark
        one subject as failed.

    Returns
    -------
    pd.DataFrame
        Table of subjects that failed during the current run.
    """

    skipped_subjects = []
    subject_bars = {}
    for bar_ix, subject_id in enumerate(subject_ids):
        subject_bars[subject_id] = tqdm(
            total=progress_total,
            desc=f"sub-{subject_id}",
            unit="step",
            position=bar_ix,
            leave=True,
        )

    log_file = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
        log_file.write(f"{log_label}: {cfg.dataset.experiment_name}\n")
        log_file.write(f"Subjects processed this run: {len(subject_ids)}\n\n")

    try:
        for subject_id in subject_ids:
            subject_bar = subject_bars[subject_id]
            try:
                if log_file is None:
                    _, used_saved_result = process_one_subject(subject_id, subject_bar)
                else:
                    with redirect_stdout(log_file), redirect_stderr(log_file):
                        _, used_saved_result = process_one_subject(subject_id, subject_bar)

                if used_saved_result:
                    subject_bar.set_postfix_str("reused")
                else:
                    subject_bar.set_postfix_str("done")
            except Exception as err:
                skipped_subjects.append({"subject": subject_id, "reason": str(err)})
                subject_bar.set_postfix_str("failed")
                print(f"sub-{subject_id} failed: {err}")
                if log_file is not None:
                    log_file.write(f"sub-{subject_id} failed: {err}\n")
    finally:
        for subject_bar in subject_bars.values():
            subject_bar.close()
        if log_file is not None:
            log_file.close()

    return pd.DataFrame(skipped_subjects)


def prepare_decoding_paths(
    base_dir: str | Path,
    run_name: str,
    results_subdir: str = "main",
) -> dict[str, Path]:
    """Create and return the standard output folders for one decoding run.

    Parameters
    ----------
    base_dir : str | Path
        Project root that contains the shared `results` folder.
    run_name : str
        Analysis name used for the decoding output folder.
    results_subdir : str, optional
        Folder written below `results`.

    Returns
    -------
    dict[str, Path]
        Standard result, figure, and log paths for one decoding run.
    """

    base_dir = Path(base_dir)
    results_dir = base_dir / "results" / results_subdir / "decoding" / run_name
    subject_results_dir = results_dir / "subject_level"
    figures_dir = results_dir / "figures"
    log_path = results_dir / "decoding.log"

    results_dir.mkdir(parents=True, exist_ok=True)
    subject_results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return {
        "results_dir": results_dir,
        "subject_results_dir": subject_results_dir,
        "figures_dir": figures_dir,
        "log_path": log_path,
    }


def save_decoding_config(results_dir: str | Path, cfg: DecodingConfig) -> dict[str, object]:
    """Save the decoding configuration as JSON for one run."""

    config_to_save = cfg.to_json_dict()

    results_dir = Path(results_dir)
    with open(results_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2)

    return config_to_save


def infer_experiment_settings(
    data_dir: str | Path,
    experiment_name: str | None,
    results_subdir: str | None,
) -> tuple[str, str]:
    """Fill in experiment-specific decoding settings from the data folder.

    Parameters
    ----------
    data_dir : str | Path
        Preprocessed data folder for the current experiment, such as
        ``data/preprocessed/exp2``.
    experiment_name : str | None
        Experiment label used to find derivative files. If ``None``, the final
        folder name from ``data_dir`` is used.
    results_subdir : str | None
        Folder name written below ``results``. If ``None``, the final folder
        name from ``data_dir`` is used.

    Returns
    -------
    tuple[str, str]
        Experiment label and results subdirectory used by the decoding helpers.

    Examples
    --------
    >>> infer_experiment_settings("data/preprocessed/exp2", None, None)
    ('exp2', 'exp2')
    """

    data_dir = Path(data_dir)
    inferred_name = data_dir.name

    if experiment_name is None:
        experiment_name = inferred_name
    if results_subdir is None:
        results_subdir = inferred_name

    return experiment_name, results_subdir


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


def run_decoding_workflow(
    subject_ids: list[str],
    available_subject_ids: list[str],
    cfg: DecodingConfig,
    subject_results_dir: str | Path,
    overwrite: bool = True,
    log_path: str | Path | None = None,
) -> dict[str, object]:
    """Run one decoding analysis across all requested subjects.

    Parameters
    ----------
    subject_ids : list[str]
        Subject IDs requested by the caller.
    cfg : DecodingConfig
        Decoding settings for the current analysis.
    subject_results_dir : str | Path
        Folder used for subject-level decoding files.
    overwrite : bool
        Whether to rerun subjects that already have saved subject-level outputs.
    log_path : str | Path | None
        Optional path for the detailed technical log.

    Returns
    -------
    dict[str, object]
        Subject-level outputs and group summary tables for one analysis.
    """

    run_plan = _build_subject_run_plan(
        requested_subject_ids=subject_ids,
        available_subject_ids=available_subject_ids,
        cached_subject_ids=list_saved_subject_ids(subject_results_dir),
    )
    print(
        f"Processing {len(run_plan['subjects_to_process'])} requested subjects; "
        f"final outputs will keep {len(run_plan['keep_seed_subjects'])} subjects before failure filtering."
    )

    shared_state = {
        "window_times_ms": None,
        "window_masks": None,
        "reference_ch_names": None,
    }

    def process_one_subject(subject_id: str, progress_bar) -> tuple[dict[str, object], bool]:
        used_saved_result = False
        if not overwrite and subject_result_exists(subject_results_dir, subject_id):
            result_bundle = load_saved_subject_results(subject_results_dir, subject_id)
            progress_bar.update(progress_bar.total)
            used_saved_result = True
        else:
            result_bundle = _run_single_subject(
                subject_id=subject_id,
                cfg=cfg,
                window_times_ms=shared_state["window_times_ms"],
                window_masks=shared_state["window_masks"],
                reference_ch_names=shared_state["reference_ch_names"],
                subject_results_dir=subject_results_dir,
                progress_bar=progress_bar,
            )

        if shared_state["window_times_ms"] is None:
            shared_state["window_times_ms"] = result_bundle["window_times_ms"]
            shared_state["window_masks"] = result_bundle["window_masks"]
        if shared_state["reference_ch_names"] is None:
            shared_state["reference_ch_names"] = result_bundle["ch_names"]
        return result_bundle, used_saved_result

    skipped_subjects_df = _process_subjects(
        subject_ids=run_plan["subjects_to_process"],
        cfg=cfg,
        progress_total=cfg.decode.n_repeats * 2,
        log_label="Decoding run",
        log_path=log_path,
        process_one_subject=process_one_subject,
    )

    failed_subjects = set(_normalize_subject_labels(skipped_subjects_df["subject"])) if len(skipped_subjects_df) > 0 else set()
    final_subject_ids = [
        subject_id
        for subject_id in run_plan["keep_seed_subjects"]
        if subject_id not in failed_subjects and subject_result_exists(subject_results_dir, subject_id)
    ]
    final_subject_bundles = {
        subject_id: load_saved_subject_results(subject_results_dir, subject_id)
        for subject_id in final_subject_ids
    }
    return _build_decoding_run_output(
        subject_bundles=final_subject_bundles,
        skipped_subjects_df=skipped_subjects_df,
    )


def run_generalization_workflow(
    subject_ids: list[str],
    available_subject_ids: list[str],
    cfg: DecodingConfig,
    subject_results_dir: str | Path,
    overwrite: bool = True,
    log_path: str | Path | None = None,
) -> dict[str, object]:
    """Run one all-window generalization analysis across subjects.

    Parameters
    ----------
    subject_ids : list[str]
        Subject IDs requested by the caller.
    cfg : DecodingConfig
        Decoding settings for the current analysis.
    log_path : str | Path | None
        Optional path for the detailed technical log.

    Returns
    -------
    dict[str, object]
        Group-level accuracy table, trial summary, and skipped-subject table.
    """

    run_plan = _build_subject_run_plan(
        requested_subject_ids=subject_ids,
        available_subject_ids=available_subject_ids,
        cached_subject_ids=list_saved_generalization_subject_ids(subject_results_dir),
    )
    print(
        f"Processing {len(run_plan['subjects_to_process'])} requested subjects; "
        f"final outputs will keep {len(run_plan['keep_seed_subjects'])} subjects before failure filtering."
    )

    def process_one_subject(subject_id: str, progress_bar) -> tuple[dict[str, object], bool]:
        used_saved_result = False
        if not overwrite and generalization_subject_result_exists(subject_results_dir, subject_id):
            result_bundle = load_saved_generalization_subject_results(subject_results_dir, subject_id)
            progress_bar.update(progress_bar.total)
            used_saved_result = True
        else:
            result_bundle = _run_single_subject_generalization(
                subject_id=subject_id,
                cfg=cfg,
                subject_results_dir=subject_results_dir,
                progress_bar=progress_bar,
            )
        return result_bundle, used_saved_result

    skipped_subjects_df = _process_subjects(
        subject_ids=run_plan["subjects_to_process"],
        cfg=cfg,
        progress_total=cfg.decode.n_repeats,
        log_label="Generalization run",
        log_path=log_path,
        process_one_subject=process_one_subject,
    )

    failed_subjects = set(_normalize_subject_labels(skipped_subjects_df["subject"])) if len(skipped_subjects_df) > 0 else set()
    final_subject_ids = [
        subject_id
        for subject_id in run_plan["keep_seed_subjects"]
        if subject_id not in failed_subjects and generalization_subject_result_exists(subject_results_dir, subject_id)
    ]
    final_subject_bundles = {
        subject_id: load_saved_generalization_subject_results(subject_results_dir, subject_id)
        for subject_id in final_subject_ids
    }
    return _build_generalization_run_output(
        subject_bundles=final_subject_bundles,
        skipped_subjects_df=skipped_subjects_df,
    )


def _build_decoding_run_output(
    subject_bundles: dict[str, dict[str, object]],
    skipped_subjects_df: pd.DataFrame,
) -> dict[str, object]:
    """Rebuild decoding run-level outputs from saved per-subject caches.

    Parameters
    ----------
    subject_bundles : dict[str, dict[str, object]]
        Saved subject bundles loaded from the subject-level cache.
    skipped_subjects_df : pd.DataFrame
        Subjects that failed during the current run.

    Returns
    -------
    dict[str, object]
        Rebuilt run-level decoding outputs.
    """

    if len(subject_bundles) == 0:
        skipped_summary = skipped_subjects_df if len(skipped_subjects_df) > 0 else pd.DataFrame(columns=["subject", "reason"])
        raise RuntimeError(
            "No subjects were available to rebuild the decoding outputs.\n"
            f"Failure summary:\n{skipped_summary.to_string(index=False)}"
        )

    subject_ids = list(subject_bundles)
    first_bundle = subject_bundles[subject_ids[0]]
    window_times_ms = first_bundle["window_times_ms"]
    reference_ch_names = first_bundle["ch_names"]

    subject_results = {subject_id: bundle["result"] for subject_id, bundle in subject_bundles.items()}
    hyperplane_results = {subject_id: bundle["hyperplane"] for subject_id, bundle in subject_bundles.items()}
    trial_summary_rows = [bundle["trial_summary_row"] for bundle in subject_bundles.values()]

    trial_summary_df = pd.DataFrame(trial_summary_rows).sort_values("subject").reset_index(drop=True)
    accuracy_df = build_accuracy_table(subject_results, window_times_ms)
    hyperplane_df = build_hyperplane_table(hyperplane_results, window_times_ms)
    pattern_df = build_channel_contrib_table(
        subject_results=subject_results,
        times_ms=window_times_ms,
        ch_names=reference_ch_names,
        value_key="channel_patterns",
    )

    return {
        "trial_summary_df": trial_summary_df,
        "skipped_subjects_df": skipped_subjects_df,
        "accuracy_df": accuracy_df,
        "hyperplane_df": hyperplane_df,
        "pattern_df": pattern_df,
        "window_times_ms": window_times_ms,
        "reference_ch_names": reference_ch_names,
        "topography_subject_id": subject_ids[0],
    }


def _build_generalization_run_output(
    subject_bundles: dict[str, dict[str, object]],
    skipped_subjects_df: pd.DataFrame,
) -> dict[str, object]:
    """Rebuild generalization run-level outputs from saved per-subject caches.

    Parameters
    ----------
    subject_bundles : dict[str, dict[str, object]]
        Saved subject bundles loaded from the subject-level cache.
    skipped_subjects_df : pd.DataFrame
        Subjects that failed during the current run.

    Returns
    -------
    dict[str, object]
        Rebuilt run-level generalization outputs.
    """

    if len(subject_bundles) == 0:
        skipped_summary = skipped_subjects_df if len(skipped_subjects_df) > 0 else pd.DataFrame(columns=["subject", "reason"])
        raise RuntimeError(
            "No subjects were available to rebuild the generalization outputs.\n"
            f"Failure summary:\n{skipped_summary.to_string(index=False)}"
        )

    subject_results = {subject_id: bundle["result"] for subject_id, bundle in subject_bundles.items()}
    trial_summary_rows = [bundle["trial_summary_row"] for bundle in subject_bundles.values()]
    first_bundle = next(iter(subject_bundles.values()))
    window_times_ms = first_bundle["window_times_ms"]

    trial_summary_df = pd.DataFrame(trial_summary_rows).sort_values("subject").reset_index(drop=True)
    accuracy_df = build_generalization_accuracy_table(
        subject_results=subject_results,
        window_times_ms=window_times_ms,
    )

    return {
        "trial_summary_df": trial_summary_df,
        "skipped_subjects_df": skipped_subjects_df,
        "accuracy_df": accuracy_df,
        "window_times_ms": window_times_ms,
    }


def export_decoding_outputs(
    run_output: dict[str, object],
    cfg: DecodingConfig,
    results_dir: str | Path,
    figures_dir: str | Path,
    topo_windows_ms: dict[str, tuple[int, int]],
) -> pd.DataFrame:
    """Save summary tables and topographies for one completed decoding run."""

    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)

    trial_summary_df = run_output["trial_summary_df"]
    skipped_subjects_df = run_output["skipped_subjects_df"]
    accuracy_df = run_output["accuracy_df"]
    hyperplane_df = run_output["hyperplane_df"]
    pattern_df = run_output["pattern_df"]
    topography_subject_id = run_output["topography_subject_id"]

    trial_summary_df.to_csv(results_dir / CORE_OUTPUT_FILES["trial_summary"], index=False)
    accuracy_df.to_csv(results_dir / CORE_OUTPUT_FILES["accuracy_cv"], index=False)
    hyperplane_df.to_csv(results_dir / CORE_OUTPUT_FILES["hyperplane_subject"], index=False)

    if len(skipped_subjects_df) > 0:
        skipped_subjects_df.to_csv(results_dir / CORE_OUTPUT_FILES["skipped_subjects"], index=False)
    else:
        skipped_path = results_dir / CORE_OUTPUT_FILES["skipped_subjects"]
        if skipped_path.exists():
            skipped_path.unlink()

    group_pattern_df = (
        pattern_df.groupby(["channel", "time_ms"], as_index=False)["value"]
        .mean()
        .sort_values(["channel", "time_ms"])
        .reset_index(drop=True)
    )
    topo_info = load_subject_info(topography_subject_id, cfg)
    topography_df = save_window_topography_set(
        channel_df=group_pattern_df,
        info=topo_info,
        output_dir=figures_dir,
        windows_ms=topo_windows_ms,
        value_col="value",
        filename_prefix="topo",
        title_prefix=None,
        colorbar_label="Z-scored pattern",
        zscore_within_window=True,
    )
    topography_df.to_csv(results_dir / CORE_OUTPUT_FILES["topography_manifest"], index=False)
    return topography_df


def build_generalization_accuracy_table(
    subject_results: dict[str, dict[str, object]],
    window_times_ms: np.ndarray,
) -> pd.DataFrame:
    """Convert subject-level generalization outputs into a long accuracy table.

    Parameters
    ----------
    subject_results : dict[str, dict[str, object]]
        Subject generalization outputs returned by the workflow.
    window_times_ms : np.ndarray
        Shared time-window centers in milliseconds for both axes.

    Returns
    -------
    pd.DataFrame
        One row per subject, train time, test time, repeat, and data type.
    """

    rows = []
    for subject_id, result in subject_results.items():
        repeat_df = result["accuracy_by_repeat"].copy()
        repeat_df["subject"] = subject_id
        repeat_df["train_time_ms"] = repeat_df["train_time_ix"].map(
            {time_ix: int(time_ms) for time_ix, time_ms in enumerate(window_times_ms)}
        )
        repeat_df["test_time_ms"] = repeat_df["test_time_ix"].map(
            {time_ix: int(time_ms) for time_ix, time_ms in enumerate(window_times_ms)}
        )
        rows.append(
            repeat_df.loc[
                :,
                [
                    "subject",
                    "train_time_ms",
                    "test_time_ms",
                    "cv_repeat",
                    "data_type",
                    "perm_id",
                    "accuracy",
                    "balanced_accuracy",
                    "n_correct",
                    "n_test_trials",
                    "chance_level",
                ],
            ]
        )

    return pd.concat(rows, ignore_index=True)


def _run_single_subject(
    subject_id: str,
    cfg: DecodingConfig,
    window_times_ms,
    window_masks,
    reference_ch_names,
    subject_results_dir: str | Path,
    progress_bar,
) -> dict[str, object]:
    """Run decoding for one subject and return all subject-level outputs."""

    metadata = load_subject_metadata(subject_id, cfg)
    trial_summary_row = {}

    data, labels, times_s, ch_names, metadata_filtered = load_subject_decoding_data(
        subject_id,
        cfg,
        return_metadata=True,
    )

    if reference_ch_names is not None and ch_names != reference_ch_names:
        raise ValueError(
            "Channel order differed across subjects. "
            f"sub-{subject_id} had channels {ch_names}, expected {reference_ch_names}."
        )

    if window_times_ms is None or window_masks is None:
        window_times_ms, window_masks = build_time_windows(times_s=times_s, window_ms=cfg.decode.time_window_ms)

    data_windowed = average_time_windows(data, window_masks)
    result = run_subject_decoding(
        data=data_windowed,
        labels=labels,
        cfg=cfg,
        progress_bar=progress_bar,
    )
    hyperplane = run_subject_hyperplane(
        data=data_windowed,
        labels=labels,
        group_labels=cfg.test_group_for_metadata_row(metadata_filtered),
        trial_ids=metadata_filtered.index.to_numpy(dtype=int),
        group_order=cfg.test_group_order(),
        cfg=cfg,
        progress_bar=progress_bar,
    )

    training_counts = pd.Series(labels[labels != ""]).value_counts()
    output_counts = pd.Series(cfg.test_group_for_metadata_row(metadata_filtered)).value_counts()

    trial_summary_row["subject"] = subject_id
    trial_summary_row["n_input_trials"] = result["n_input_trials"]
    trial_summary_row["n_training_trials"] = result["n_training_trials"]
    trial_summary_row["n_binned_trials"] = result["n_binned_trials"]

    for condition in cfg.train_label_order():
        trial_summary_row[f"n_train_{condition.replace('/', '_').lower()}"] = int(
            training_counts.get(condition, 0)
        )
    for group_name in cfg.test_group_order():
        trial_summary_row[f"n_test_{group_name.replace('/', '_').lower()}"] = int(
            output_counts.get(group_name, 0)
        )
    save_subject_results(
        output_dir=subject_results_dir,
        subject_id=subject_id,
        times_ms=window_times_ms,
        ch_names=ch_names,
        decoding_result=result,
        hyperplane_result=hyperplane,
        trial_summary_row=trial_summary_row,
    )

    return {
        "result": result,
        "hyperplane": hyperplane,
        "trial_summary_row": trial_summary_row,
        "window_times_ms": window_times_ms,
        "window_masks": window_masks,
        "ch_names": ch_names,
    }
def _run_single_subject_generalization(
    subject_id: str,
    cfg: DecodingConfig,
    subject_results_dir: str | Path,
    progress_bar,
) -> dict[str, object]:
    """Run generalization decoding for one subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier without the `sub-` prefix.
    cfg : DecodingConfig
        Decoding settings for the current analysis.
    subject_results_dir : str | Path
        Folder used for subject-level generalization cache files.
    progress_bar : object
        Tqdm progress bar updated after each repeat.

    Returns
    -------
    dict[str, object]
        Subject-level generalization result, shared time windows, and trial
        counts used for the summary table.
    """

    data, labels, times_s, _, metadata_filtered = load_subject_decoding_data(
        subject_id,
        cfg,
        return_metadata=True,
    )

    window_times_ms, window_masks = build_time_windows(times_s=times_s, window_ms=cfg.decode.time_window_ms)
    data_windowed = average_time_windows(data, window_masks)
    result = run_subject_generalization(
        data=data_windowed,
        labels=labels,
        cfg=cfg,
        progress_bar=progress_bar,
    )

    training_counts = pd.Series(labels[labels != ""]).value_counts()
    output_counts = pd.Series(cfg.test_group_for_metadata_row(metadata_filtered)).value_counts()

    trial_summary_row = {
        "subject": subject_id,
        "n_input_trials": result["n_input_trials"],
        "n_training_trials": result["n_training_trials"],
        "n_binned_trials": result["n_binned_trials"],
    }

    for condition in cfg.train_label_order():
        trial_summary_row[f"n_train_{condition.replace('/', '_').lower()}"] = int(
            training_counts.get(condition, 0)
        )
    for group_name in cfg.test_group_order():
        trial_summary_row[f"n_test_{group_name.replace('/', '_').lower()}"] = int(
            output_counts.get(group_name, 0)
        )

    save_generalization_subject_results(
        output_dir=subject_results_dir,
        subject_id=subject_id,
        times_ms=window_times_ms,
        generalization_result=result,
        trial_summary_row=trial_summary_row,
    )

    return {
        "result": result,
        "trial_summary_row": trial_summary_row,
        "window_times_ms": window_times_ms,
    }
