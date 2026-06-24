"""Output-table and export helpers for encoding workflows."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .._shared.topography import build_topography_coord_table
from .io import load_subject_info, write_pattern_expression_readme


def _normalize_topography_windows(
    *,
    time_window_ms: tuple[int, int] | None = None,
    time_windows_ms: dict[str, tuple[int, int]] | None = None,
) -> tuple[dict[str, tuple[int, int]], bool]:
    """Return named topography windows and whether names should be exported.

    Parameters
    ----------
    time_window_ms : tuple[int, int] | None
        Backward-compatible single time window as ``(start_ms, end_ms)``.
    time_windows_ms : dict[str, tuple[int, int]] | None
        Named topography windows for multi-window export.

    Returns
    -------
    tuple[dict[str, tuple[int, int]], bool]
        Normalized windows and whether the output table should include a
        ``window_name`` column.
    """

    has_single_window = time_window_ms is not None
    has_named_windows = time_windows_ms is not None
    if has_single_window == has_named_windows:
        raise ValueError(
            "Provide exactly one topography window setting: "
            "'time_window_ms' for one window or 'time_windows_ms' for named windows."
        )

    if has_single_window:
        return {"topography": time_window_ms}, False

    if time_windows_ms is None:
        raise ValueError("topography time_windows_ms must be provided.")
    if len(time_windows_ms) == 0:
        raise ValueError("topography time_windows_ms must include at least one window.")

    normalized = {}
    for window_name, window_ms in time_windows_ms.items():
        window_label = str(window_name).strip()
        if window_label == "":
            raise ValueError("topography window names must not be empty.")
        normalized[window_label] = window_ms

    return normalized, True


def build_encoding_topography_value_table(
    *,
    subject_payloads: dict[str, dict[str, np.ndarray]],
    time_window_ms: tuple[int, int] | None = None,
    time_windows_ms: dict[str, tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Summarize encoding beta patterns for R topography plotting.

    Parameters
    ----------
    subject_payloads : dict[str, dict[str, np.ndarray]]
        Subject-level encoding model payloads keyed by subject ID. Each payload
        must contain ``raw_beta_patterns``, ``predictor_names``, ``ch_names``,
        and ``times_s``.
    time_window_ms : tuple[int, int] | None
        Backward-compatible inclusive time range for one topography window.
    time_windows_ms : dict[str, tuple[int, int]] | None
        Named inclusive time ranges for multi-window topography export.

    Returns
    -------
    pd.DataFrame
        Table with one row per predictor and channel. ``raw_value`` is the
        group mean beta value, and ``z_value`` is z-scored across channels
        within the same predictor map and time window.
    """

    if len(subject_payloads) == 0:
        raise ValueError("subject_payloads must include at least one subject.")

    first_subject = next(iter(subject_payloads))
    first_payload = subject_payloads[first_subject]
    predictor_names = first_payload["predictor_names"].astype(str).tolist()
    ch_names = first_payload["ch_names"].astype(str).tolist()
    windows_ms, include_window_name = _normalize_topography_windows(
        time_window_ms=time_window_ms,
        time_windows_ms=time_windows_ms,
    )

    window_tables = []
    for window_name, window_ms in windows_ms.items():
        start_ms, end_ms = int(window_ms[0]), int(window_ms[1])
        if start_ms > end_ms:
            raise ValueError(
                f"topography window '{window_name}' start must be <= end."
            )

        subject_rows = []
        for subject_id, payload in subject_payloads.items():
            subject_predictors = payload["predictor_names"].astype(str).tolist()
            subject_channels = payload["ch_names"].astype(str).tolist()
            if subject_predictors != predictor_names:
                raise ValueError(
                    "Encoding topography requires matching predictor order across subjects. "
                    f"Subject {subject_id} had {subject_predictors}, expected {predictor_names}."
                )
            if subject_channels != ch_names:
                raise ValueError(
                    "Encoding topography requires matching channel order across subjects. "
                    f"Subject {subject_id} did not match the first completed subject."
                )

            times_ms = payload["times_s"].astype(float) * 1000.0
            time_mask = (times_ms >= start_ms) & (times_ms <= end_ms)
            if not np.any(time_mask):
                raise ValueError(
                    "No encoding beta pattern time bins were found between "
                    f"{start_ms} ms and {end_ms} ms."
                )

            beta_patterns = payload["raw_beta_patterns"].astype(float)
            expected_shape = (len(predictor_names), len(ch_names), len(times_ms))
            if beta_patterns.shape[1:] != expected_shape:
                raise ValueError(
                    "raw_beta_patterns has incompatible predictor/channel/time dimensions. "
                    f"Expected trailing shape {expected_shape}, found {beta_patterns.shape[1:]}."
                )

            subject_values = beta_patterns[:, :, :, time_mask].mean(axis=(0, 3))
            for predictor_ix, predictor_name in enumerate(predictor_names):
                for channel_ix, channel in enumerate(ch_names):
                    subject_rows.append(
                        {
                            "subject": str(subject_id),
                            "channel": channel,
                            "effect": predictor_name,
                            "subject_value": float(
                                subject_values[predictor_ix, channel_ix]
                            ),
                        }
                    )

        subject_df = pd.DataFrame(subject_rows)
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

        channel_order = {
            channel: channel_ix for channel_ix, channel in enumerate(ch_names)
        }
        predictor_order = {
            predictor_name: predictor_ix
            for predictor_ix, predictor_name in enumerate(predictor_names)
        }
        group_df["effect_order"] = group_df["effect"].map(predictor_order)
        group_df["channel_order"] = group_df["channel"].map(channel_order)
        group_df = group_df.sort_values(["effect_order", "channel_order"]).reset_index(
            drop=True
        )

        z_rows = []
        for effect, effect_df in group_df.groupby("effect", sort=False):
            values = effect_df["raw_value"].to_numpy(dtype=float)
            value_std = values.std(ddof=0)
            if value_std == 0:
                z_values = np.zeros(len(values), dtype=float)
            else:
                z_values = (values - values.mean()) / value_std
            z_rows.append(effect_df.assign(z_value=z_values))

        value_df = pd.concat(z_rows, ignore_index=True)
        if include_window_name:
            value_df["window_name"] = window_name
        value_df["window_start_ms"] = start_ms
        value_df["window_end_ms"] = end_ms

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
        window_tables.append(value_df.loc[:, output_cols])

    return pd.concat(window_tables, ignore_index=True)


def export_encoding_model_outputs(
    *,
    output_dir: str | Path,
    output_files: dict[str, str],
    subject_summary_df: pd.DataFrame,
    skipped_subjects_df: pd.DataFrame,
    run_summary_df: pd.DataFrame,
    training_pattern_strength_df: pd.DataFrame,
    testing_coefficient_df: pd.DataFrame,
    testing_coefficient_wide_df: pd.DataFrame,
    condition_coefficient_df: pd.DataFrame,
    config_payload: dict[str, object] | None,
    topography: dict[str, object] | None,
    subject_payloads: dict[str, dict[str, np.ndarray]],
    loader_cfg,
) -> dict[str, pd.DataFrame | None]:
    """Write group-level encoding outputs and optional topography tables.

    Parameters
    ----------
    output_dir : str | Path
        Folder where group-level output tables are written.
    output_files : dict[str, str]
        Output filename registry for the encoding model workflow.
    subject_summary_df, skipped_subjects_df, run_summary_df : pd.DataFrame
        Run summary tables.
    training_pattern_strength_df, testing_coefficient_df : pd.DataFrame
        Long training and testing model outputs.
    testing_coefficient_wide_df, condition_coefficient_df : pd.DataFrame
        Wide testing and condition-averaged output tables.
    config_payload : dict[str, object] | None
        Optional JSON-serializable run configuration.
    topography : dict[str, object] | None
        Optional topography settings. Provide ``time_window_ms`` for one
        backward-compatible window or ``time_windows_ms`` for named windows.
    subject_payloads : dict[str, dict[str, np.ndarray]]
        Subject-level saved model payloads used for topography export.
    loader_cfg : object
        Encoding loader configuration used for channel-coordinate export.

    Returns
    -------
    dict[str, pd.DataFrame | None]
        Optional topography value and coordinate tables.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_summary_df.to_csv(output_dir / output_files["subject_summary"], index=False)
    if len(skipped_subjects_df) > 0:
        skipped_subjects_df.to_csv(output_dir / output_files["skipped_subjects"], index=False)
    else:
        skipped_path = output_dir / output_files["skipped_subjects"]
        if skipped_path.exists():
            skipped_path.unlink()
    run_summary_df.to_csv(output_dir / output_files["run_summary"], index=False)
    training_pattern_strength_df.to_csv(
        output_dir / output_files["training_pattern_strength"],
        index=False,
    )
    testing_coefficient_df.to_csv(
        output_dir / output_files["testing_effect_coefficients"],
        index=False,
    )
    testing_coefficient_wide_df.to_csv(
        output_dir / output_files["testing_effect_coefficients_wide"],
        index=False,
    )
    condition_coefficient_df.to_csv(
        output_dir / output_files["condition_average_coefficients"],
        index=False,
    )
    write_pattern_expression_readme(output_dir)
    if config_payload is not None:
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)

    topography_values_df = None
    topography_coords_df = None
    if topography is not None:
        has_single_window = "time_window_ms" in topography
        has_named_windows = "time_windows_ms" in topography
        if has_single_window == has_named_windows:
            raise ValueError(
                "topography must include exactly one of 'time_window_ms' "
                "or 'time_windows_ms'."
            )
        topography_values_path = output_dir / output_files["topography_values"]
        topography_coords_path = output_dir / output_files["topography_coords"]
        topography_values_path.parent.mkdir(parents=True, exist_ok=True)

        if has_single_window:
            topography_values_df = build_encoding_topography_value_table(
                subject_payloads=subject_payloads,
                time_window_ms=topography["time_window_ms"],
            )
        else:
            topography_values_df = build_encoding_topography_value_table(
                subject_payloads=subject_payloads,
                time_windows_ms=topography["time_windows_ms"],
            )
        first_subject = next(iter(subject_payloads))
        first_payload = subject_payloads[first_subject]
        topography_coords_df = build_topography_coord_table(
            info=load_subject_info(first_subject, loader_cfg),
            channels=first_payload["ch_names"].astype(str).tolist(),
        )
        topography_values_df.to_csv(topography_values_path, index=False)
        topography_coords_df.to_csv(topography_coords_path, index=False)

    return {
        "topography_values_df": topography_values_df,
        "topography_coords_df": topography_coords_df,
    }
