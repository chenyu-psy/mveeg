"""Output-table and export helpers for decoding workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .._shared.topography import save_window_topography_set
from .config import DecodingConfig
from .io import load_subject_info
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


