"""Summary-table helpers for the EEG decoding workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd

def build_accuracy_table(
    subject_results: dict[str, dict[str, np.ndarray | list[str] | int]],
    times_ms: np.ndarray,
) -> pd.DataFrame:
    """Convert repeat-level decoding outputs into a long accuracy table.

    Parameters
    ----------
    subject_results : dict[str, dict[str, np.ndarray | list[str] | int]]
        Subject decoding outputs returned by the workflow.
    times_ms : np.ndarray
        Decoding time points in milliseconds.

    Returns
    -------
    pd.DataFrame
        One row per subject, time bin, CV repeat, and data type.
    """

    rows = []
    for subject_id, result in subject_results.items():
        repeat_df = result["accuracy_by_repeat"].copy()
        repeat_df["subject"] = subject_id
        repeat_df["time_ms"] = repeat_df["time_ix"].map(
            {time_ix: int(time_ms) for time_ix, time_ms in enumerate(times_ms)}
        )
        rows.append(
            repeat_df.loc[
                :,
                [
                    "subject",
                    "time_ms",
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


def build_hyperplane_table(
    subject_results: dict[str, dict[str, object]],
    times_ms: np.ndarray,
) -> pd.DataFrame:
    """Convert subject hyperplane results into a long table.

    Parameters
    ----------
    subject_results : dict[str, dict[str, object]]
        Subject hyperplane outputs returned by the workflow.
    times_ms : np.ndarray
        Decoding time points in milliseconds.

    Returns
    -------
    pd.DataFrame
        One row per subject, trial, condition, and time bin.
    """

    rows = []
    for subject_id, result in subject_results.items():
        trial_distance = result["trial_distance"]
        for _, trial_row in trial_distance.iterrows():
            for time_ms, value in zip(times_ms, trial_row["distance"]):
                rows.append(
                    {
                        "subject": subject_id,
                        "trial_id": trial_row["trial_id"],
                        "condition": trial_row["condition"],
                        "time_ms": int(time_ms),
                        "distance": float(value),
                    }
                )
    return pd.DataFrame(rows)


def build_channel_contrib_table(
    subject_results: dict[str, dict[str, np.ndarray | list[str] | int]],
    times_ms: np.ndarray,
    ch_names: list[str],
    value_key: str = "channel_patterns",
) -> pd.DataFrame:
    """Convert channel-level decoding outputs into a long table."""

    rows = []
    n_channels = len(ch_names)
    n_times = len(times_ms)

    for subject_id, result in subject_results.items():
        if value_key not in result:
            raise KeyError(f"'{value_key}' was not found in the subject decoding results.")

        values = np.asarray(result[value_key], dtype=float)
        if values.shape != (n_channels, n_times):
            raise ValueError(
                f"Expected {value_key} to have shape {(n_channels, n_times)}, "
                f"but subject {subject_id} had shape {values.shape}."
            )

        for ch_ix, ch_name in enumerate(ch_names):
            for time_ix, time_ms in enumerate(times_ms):
                rows.append(
                    {
                        "subject": subject_id,
                        "channel": ch_name,
                        "time_ms": int(time_ms),
                        "value": float(values[ch_ix, time_ix]),
                    }
                )

    return pd.DataFrame(rows)
