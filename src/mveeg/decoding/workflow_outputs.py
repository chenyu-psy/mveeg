"""Output-table and export helpers for decoding workflows."""

from __future__ import annotations

from pathlib import Path

import mne
from mne.channels.layout import _find_topomap_coords
import numpy as np
import pandas as pd

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
    "topography_values": "topography_values.csv",
    "topography_coords": "topography_coords.csv",
}

TOPOGRAPHY_EFFECT_LABEL = "Decoding pattern"


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
    topo_windows_ms: dict[str, tuple[int, int]],
) -> pd.DataFrame:
    """Save summary tables and R-ready topography data for one decoding run.

    Parameters
    ----------
    run_output : dict[str, object]
        Completed decoding outputs returned by ``run_decoding_workflow``.
    cfg : DecodingConfig
        Decoding configuration used to load the matched channel coordinates.
    results_dir : str | Path
        Folder where group-level decoding tables are written.
    topo_windows_ms : dict[str, tuple[int, int]]
        Named time windows exported as channel-value summaries for R plotting.

    Returns
    -------
    pd.DataFrame
        Channel-value table written to ``topography_values.csv``.
    """

    results_dir = Path(results_dir)

    trial_summary_df = run_output["trial_summary_df"]
    skipped_subjects_df = run_output["skipped_subjects_df"]
    accuracy_df = run_output["accuracy_df"]
    hyperplane_df = run_output["hyperplane_df"]
    pattern_df = run_output["pattern_df"]
    reference_ch_names = run_output["reference_ch_names"]
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

    topography_values_df = build_topography_value_table(
        pattern_df=pattern_df,
        windows_ms=topo_windows_ms,
    )
    topo_info = load_subject_info(topography_subject_id, cfg)
    topography_coords_df = build_topography_coord_table(
        info=topo_info,
        channels=reference_ch_names,
    )
    topography_values_df.to_csv(
        results_dir / CORE_OUTPUT_FILES["topography_values"],
        index=False,
    )
    topography_coords_df.to_csv(
        results_dir / CORE_OUTPUT_FILES["topography_coords"],
        index=False,
    )
    return topography_values_df


def build_topography_value_table(
    *,
    pattern_df: pd.DataFrame,
    windows_ms: dict[str, tuple[int, int]],
) -> pd.DataFrame:
    """Summarize decoding channel patterns for R topography plotting.

    Parameters
    ----------
    pattern_df : pd.DataFrame
        Long decoding pattern table with ``subject``, ``channel``,
        ``time_ms``, and ``value`` columns.
    windows_ms : dict[str, tuple[int, int]]
        Named time windows in milliseconds.

    Returns
    -------
    pd.DataFrame
        Table with one row per channel and requested time window. ``raw_value``
        is the group mean channel pattern, and ``z_value`` is z-scored across
        channels within the same window.
    """

    required_cols = {"subject", "channel", "time_ms", "value"}
    missing_cols = sorted(required_cols.difference(pattern_df.columns))
    if len(missing_cols) > 0:
        raise ValueError(f"pattern_df is missing required columns: {missing_cols}")

    rows = []
    for _window_name, (start_ms, end_ms) in windows_ms.items():
        window_mask = pattern_df["time_ms"].between(start_ms, end_ms, inclusive="both")
        window_df = pattern_df.loc[window_mask].copy()
        if len(window_df) == 0:
            raise ValueError(
                f"No decoding pattern values were found between {start_ms} ms and {end_ms} ms."
            )

        subject_channel_df = (
            window_df.groupby(["subject", "channel"], as_index=False)["value"]
            .mean()
            .rename(columns={"value": "subject_value"})
        )
        n_subjects = int(subject_channel_df["subject"].nunique())
        group_df = (
            subject_channel_df.groupby("channel", as_index=False)["subject_value"]
            .mean()
            .rename(columns={"subject_value": "raw_value"})
            .sort_values("channel")
            .reset_index(drop=True)
        )

        values = group_df["raw_value"].to_numpy(dtype=float)
        value_std = values.std(ddof=0)
        if value_std == 0:
            z_values = np.zeros(len(values), dtype=float)
        else:
            z_values = (values - values.mean()) / value_std

        group_df["effect"] = TOPOGRAPHY_EFFECT_LABEL
        group_df["z_value"] = z_values
        group_df["window_start_ms"] = int(start_ms)
        group_df["window_end_ms"] = int(end_ms)
        group_df["n_subjects"] = n_subjects
        rows.append(
            group_df.loc[
                :,
                [
                    "channel",
                    "effect",
                    "raw_value",
                    "z_value",
                    "window_start_ms",
                    "window_end_ms",
                    "n_subjects",
                ],
            ]
        )

    return pd.concat(rows, ignore_index=True)


def build_topography_coord_table(
    *,
    info: mne.Info,
    channels: list[str],
) -> pd.DataFrame:
    """Export MNE-projected electrode coordinates for R topography plotting.

    Parameters
    ----------
    info : mne.Info
        Channel metadata after applying the same channel-drop rules as the
        decoding data.
    channels : list[str]
        Decoding channel names that should appear in the output table.

    Returns
    -------
    pd.DataFrame
        Table with ``channel``, ``x``, and ``y`` columns. Coordinates are in
        millimeters, matching the WT13 R topography input files.
    """

    missing_channels = [ch_name for ch_name in channels if ch_name not in info["ch_names"]]
    if len(missing_channels) > 0:
        raise ValueError(
            "Topography coordinate export could not find these channels in the MNE info: "
            f"{missing_channels}"
        )

    picks = [info["ch_names"].index(ch_name) for ch_name in channels]
    try:
        coords = _find_topomap_coords(info, picks=picks)
    except (AttributeError, RuntimeError, ValueError) as err:
        raise ValueError(
            "Could not export topography coordinates. Make sure the saved epochs "
            "include electrode montage positions."
        ) from err

    coords = np.asarray(coords, dtype=float)
    if coords.shape != (len(channels), 2):
        raise ValueError(
            f"Expected topography coordinates to have shape {(len(channels), 2)}, "
            f"but found {coords.shape}."
        )
    if np.any(~np.isfinite(coords)):
        raise ValueError(
            "Topography coordinates contained missing or non-finite values. "
            "Make sure the saved epochs include electrode montage positions."
        )

    return pd.DataFrame(
        {
            "channel": channels,
            "x": coords[:, 0] * 1000.0,
            "y": coords[:, 1] * 1000.0,
        }
    )



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
