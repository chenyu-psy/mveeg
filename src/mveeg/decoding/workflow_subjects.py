"""Subject-loop helpers for decoding workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .._shared.workflow_subjects import process_subjects
from .config import DecodingConfig
from .io import (
    generalization_subject_result_exists,
    list_saved_generalization_subject_ids,
    list_saved_subject_ids,
    load_saved_generalization_subject_results,
    load_saved_subject_results,
    load_subject_decoding_data,
    save_generalization_subject_results,
    save_subject_results,
    subject_result_exists,
)
from .prepare import average_time_windows, build_time_windows
from .run import run_subject_decoding, run_subject_generalization, run_subject_hyperplane
from .workflow_outputs import _build_decoding_run_output, _build_generalization_run_output


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



def run_decoding_workflow(
    subject_ids: list[str],
    available_subject_ids: list[str],
    cfg: DecodingConfig,
    subject_results_dir: str | Path | None = None,
    overwrite: bool = True,
    metadata_transform=None,
    log_path: str | Path | None = None,
) -> dict[str, object]:
    """Run one decoding analysis across all requested subjects.

    Parameters
    ----------
    subject_ids : list[str]
        Subject IDs requested by the caller.
    cfg : DecodingConfig
        Decoding settings for the current analysis.
    subject_results_dir : str | Path | None
        Optional folder used for subject-level decoding cache files. When
        ``None``, subjects are processed in memory and no cache files are read
        or written.
    overwrite : bool
        Whether to rerun subjects that already have saved subject-level outputs.
    log_path : str | Path | None
        Optional path for the detailed technical log.

    Returns
    -------
    dict[str, object]
        Subject-level outputs and group summary tables for one analysis.
    """

    cached_subject_ids = [] if subject_results_dir is None else list_saved_subject_ids(subject_results_dir)
    run_plan = _build_subject_run_plan(
        requested_subject_ids=subject_ids,
        available_subject_ids=available_subject_ids,
        cached_subject_ids=cached_subject_ids,
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
    processed_bundles = {}

    def process_one_subject(subject_id: str, progress_bar) -> tuple[dict[str, object], bool]:
        used_saved_result = False
        if subject_results_dir is not None and not overwrite and subject_result_exists(subject_results_dir, subject_id):
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
                metadata_transform=metadata_transform,
                progress_bar=progress_bar,
            )
        processed_bundles[subject_id] = result_bundle

        if shared_state["window_times_ms"] is None:
            shared_state["window_times_ms"] = result_bundle["window_times_ms"]
            shared_state["window_masks"] = result_bundle["window_masks"]
        if shared_state["reference_ch_names"] is None:
            shared_state["reference_ch_names"] = result_bundle["ch_names"]
        return result_bundle, used_saved_result

    skipped_subjects_df = process_subjects(
        subject_ids=run_plan["subjects_to_process"],
        progress_total=cfg.decode.n_repeats * 2,
        log_label="Decoding run",
        log_path=log_path,
        experiment_name=cfg.dataset.experiment_name,
        process_one_subject=process_one_subject,
    )

    failed_subjects = set(_normalize_subject_labels(skipped_subjects_df["subject"])) if len(skipped_subjects_df) > 0 else set()
    if subject_results_dir is None:
        final_subject_bundles = {
            subject_id: bundle
            for subject_id, bundle in processed_bundles.items()
            if subject_id not in failed_subjects
        }
    else:
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
    subject_results_dir: str | Path | None = None,
    overwrite: bool = True,
    metadata_transform=None,
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

    cached_subject_ids = [] if subject_results_dir is None else list_saved_generalization_subject_ids(subject_results_dir)
    run_plan = _build_subject_run_plan(
        requested_subject_ids=subject_ids,
        available_subject_ids=available_subject_ids,
        cached_subject_ids=cached_subject_ids,
    )
    print(
        f"Processing {len(run_plan['subjects_to_process'])} requested subjects; "
        f"final outputs will keep {len(run_plan['keep_seed_subjects'])} subjects before failure filtering."
    )

    processed_bundles = {}

    def process_one_subject(subject_id: str, progress_bar) -> tuple[dict[str, object], bool]:
        used_saved_result = False
        if subject_results_dir is not None and not overwrite and generalization_subject_result_exists(subject_results_dir, subject_id):
            result_bundle = load_saved_generalization_subject_results(subject_results_dir, subject_id)
            progress_bar.update(progress_bar.total)
            used_saved_result = True
        else:
            result_bundle = _run_single_subject_generalization(
                subject_id=subject_id,
                cfg=cfg,
                subject_results_dir=subject_results_dir,
                metadata_transform=metadata_transform,
                progress_bar=progress_bar,
            )
        processed_bundles[subject_id] = result_bundle
        return result_bundle, used_saved_result

    skipped_subjects_df = process_subjects(
        subject_ids=run_plan["subjects_to_process"],
        progress_total=cfg.decode.n_repeats,
        log_label="Generalization run",
        log_path=log_path,
        experiment_name=cfg.dataset.experiment_name,
        process_one_subject=process_one_subject,
    )

    failed_subjects = set(_normalize_subject_labels(skipped_subjects_df["subject"])) if len(skipped_subjects_df) > 0 else set()
    if subject_results_dir is None:
        final_subject_bundles = {
            subject_id: bundle
            for subject_id, bundle in processed_bundles.items()
            if subject_id not in failed_subjects
        }
    else:
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



def _run_single_subject(
    subject_id: str,
    cfg: DecodingConfig,
    window_times_ms,
    window_masks,
    reference_ch_names,
    subject_results_dir: str | Path | None,
    metadata_transform,
    progress_bar,
) -> dict[str, object]:
    """Run decoding for one subject and return all subject-level outputs."""

    trial_summary_row = {}

    data, labels, times_s, ch_names, metadata_filtered = load_subject_decoding_data(
        subject_id,
        cfg,
        return_metadata=True,
        metadata_transform=metadata_transform,
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
        trial_ids=metadata_filtered["epoch_index"].to_numpy(dtype=int),
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
    if subject_results_dir is not None:
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
    subject_results_dir: str | Path | None,
    metadata_transform,
    progress_bar,
) -> dict[str, object]:
    """Run generalization decoding for one subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier without the `sub-` prefix.
    cfg : DecodingConfig
        Decoding settings for the current analysis.
    subject_results_dir : str | Path | None
        Optional folder used for subject-level generalization cache files.
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
        metadata_transform=metadata_transform,
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

    if subject_results_dir is not None:
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
