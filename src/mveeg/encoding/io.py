"""Input/output helpers for encoding model exports.

This module handles both the main encoding-model outputs and the
pattern-expression export helpers used by script workflows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .summaries import (
    build_condition_average_pattern_expression_table,
    build_trial_pattern_expression_table,
)


DEFAULT_OUTPUT_SUBDIR = Path("results") / "encoding_pattern_expression"
SUBJECT_RESULTS_DIRNAME = "subject_level"
TRIAL_OUTPUT_FILENAME = "expression_trial.csv"
CONDITION_OUTPUT_FILENAME = "expression_condition.csv"
README_FILENAME = "README.txt"
MODEL_RESULT_SUFFIX = "_encoding_model.npz"


def write_pattern_expression_readme(output_dir: str | Path) -> Path:
    """Write a short README describing the current encoding outputs.

    Parameters
    ----------
    output_dir : str | Path
        Folder where output files are saved.

    Returns
    -------
    Path
        Path to the written README file.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    readme_path = output_dir / README_FILENAME
    readme_text = (
        "Encoding model outputs\n"
        "\n"
        "Primary modeling tables:\n"
        "- training_pattern_strength.csv: subject/fold/effect/time rows for observed pattern and null draws.\n"
        "- testing_effect_coefficients.csv: subject/fold/effect/trial/time coefficient rows.\n"
        "- testing_effect_coefficients_wide.csv: subject/fold/trial/time rows with intercept and selected predictor coefficients.\n"
        "\n"
        "Auxiliary tables:\n"
        "- condition_average_coefficients.csv: condition means for visualization.\n"
        "- subject_summary.csv and run_summary.csv: quick run checks only.\n"
        "\n"
        "Interpretation notes:\n"
        "- Training pattern strength is the L2 norm of the raw beta pattern.\n"
        "- Training rows use data_type = pattern or data_type = null, where null comes from shuffled full-model condition labels.\n"
        "- Testing values are reconstructed held-out signed coefficients, not residualized projections.\n"
    )
    readme_path.write_text(readme_text, encoding="utf-8")
    return readme_path


def prepare_encoding_output_paths(
    output_dir: str | Path = DEFAULT_OUTPUT_SUBDIR,
) -> dict[str, Path]:
    """Create and return standard output folders for compatibility exports.

    Parameters
    ----------
    output_dir : str | Path
        Root output folder for encoding pattern-expression files.

    Returns
    -------
    dict[str, Path]
        Output paths for analysis-level and subject-level folders.
    """

    output_dir = Path(output_dir)
    subject_results_dir = output_dir / SUBJECT_RESULTS_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_results_dir.mkdir(parents=True, exist_ok=True)

    return {
        "results_dir": output_dir,
        "subject_results_dir": subject_results_dir,
    }


def subject_result_paths(
    output_dir: str | Path,
    subject_id: str,
) -> dict[str, Path]:
    """Return per-subject compatibility output file paths.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level output folder.
    subject_id : str
        Subject identifier without the ``sub-`` prefix.

    Returns
    -------
    dict[str, Path]
        Paths for trial-level and condition-level per-subject CSV files.
    """

    output_dir = Path(output_dir)
    subject_label = str(subject_id).strip()
    if subject_label.startswith("sub-"):
        subject_label = subject_label[4:]

    return {
        "trial_table": output_dir / f"sub-{subject_label}_expression_trial.csv",
        "condition_table": output_dir / f"sub-{subject_label}_expression_condition.csv",
    }


def subject_result_exists(output_dir: str | Path, subject_id: str) -> bool:
    """Return whether all compatibility per-subject output files exist.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level output folder.
    subject_id : str
        Subject identifier without the ``sub-`` prefix.

    Returns
    -------
    bool
        ``True`` when both per-subject output files exist.
    """

    paths = subject_result_paths(output_dir, subject_id)
    return all(path.exists() for path in paths.values())


def load_saved_subject_results(
    output_dir: str | Path,
    subject_id: str,
) -> dict[str, pd.DataFrame]:
    """Load saved compatibility per-subject encoding tables.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level output folder.
    subject_id : str
        Subject identifier without the ``sub-`` prefix.

    Returns
    -------
    dict[str, pd.DataFrame]
        Loaded trial-level and condition-level tables.
    """

    paths = subject_result_paths(output_dir, subject_id)
    return {
        "trial_table": pd.read_csv(paths["trial_table"]),
        "condition_table": pd.read_csv(paths["condition_table"]),
    }


def save_subject_results(
    output_dir: str | Path,
    subject_id: str,
    trial_table: pd.DataFrame,
    condition_table: pd.DataFrame,
) -> dict[str, Path]:
    """Save compatibility per-subject pattern-expression tables.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level output folder.
    subject_id : str
        Subject identifier without the ``sub-`` prefix.
    trial_table : pd.DataFrame
        Trial-level pattern-expression table.
    condition_table : pd.DataFrame
        Condition-averaged pattern-expression table.

    Returns
    -------
    dict[str, Path]
        Saved per-subject file paths.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = subject_result_paths(output_dir, subject_id)
    trial_table.to_csv(paths["trial_table"], index=False)
    condition_table.to_csv(paths["condition_table"], index=False)
    return paths


def encoding_model_result_path(output_dir: str | Path, subject_id: str) -> Path:
    """Return the subject-level NPZ path for one encoding-model run.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level results folder.
    subject_id : str
        Subject identifier without the ``sub-`` prefix.

    Returns
    -------
    Path
        Path to the subject-level NPZ cache.
    """

    subject_label = str(subject_id).strip()
    if subject_label.startswith("sub-"):
        subject_label = subject_label[4:]
    return Path(output_dir) / f"sub-{subject_label}{MODEL_RESULT_SUFFIX}"


def save_encoding_model_result(
    output_dir: str | Path,
    subject_id: str,
    payload: dict[str, np.ndarray],
) -> Path:
    """Save one subject's encoding-model arrays as an NPZ cache.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level results folder.
    subject_id : str
        Subject identifier without the ``sub-`` prefix.
    payload : dict[str, np.ndarray]
        Named arrays written into the NPZ file.

    Returns
    -------
    Path
        Path to the written NPZ file.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = encoding_model_result_path(output_dir, subject_id)
    np.savez(output_path, **payload)
    return output_path


def load_encoding_model_result(
    output_dir: str | Path,
    subject_id: str,
) -> dict[str, np.ndarray]:
    """Load one subject's encoding-model NPZ cache into memory.

    Parameters
    ----------
    output_dir : str | Path
        Subject-level results folder.
    subject_id : str
        Subject identifier without the ``sub-`` prefix.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary of arrays stored in the NPZ file.
    """

    output_path = encoding_model_result_path(output_dir, subject_id)
    with np.load(output_path, allow_pickle=True) as saved:
        return {key: saved[key] for key in saved.files}


def save_pattern_expression_tables(
    *,
    subject_id: str,
    trial_table: pd.DataFrame,
    condition_table: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_SUBDIR,
    overwrite: bool = True,
) -> dict[str, Path]:
    """Save trial-level and condition-level pattern-expression tables.

    Parameters
    ----------
    subject_id : str
        Subject identifier without the ``sub-`` prefix.
    trial_table : pd.DataFrame
        Trial-level pattern-expression table.
    condition_table : pd.DataFrame
        Condition-averaged pattern-expression table.
    output_dir : str | Path
        Output folder where files are written.
    overwrite : bool
        Whether to overwrite existing per-subject files. When ``False`` and
        per-subject files already exist, saved files are reused.

    Returns
    -------
    dict[str, Path]
        Paths for per-subject files and README.
    """

    paths = prepare_encoding_output_paths(output_dir)

    if not overwrite and subject_result_exists(paths["subject_results_dir"], subject_id):
        saved = load_saved_subject_results(paths["subject_results_dir"], subject_id)
        trial_table = saved["trial_table"]
        condition_table = saved["condition_table"]
        subject_paths = subject_result_paths(paths["subject_results_dir"], subject_id)
    else:
        subject_paths = save_subject_results(
            output_dir=paths["subject_results_dir"],
            subject_id=subject_id,
            trial_table=trial_table,
            condition_table=condition_table,
        )

    readme_path = write_pattern_expression_readme(paths["results_dir"])

    return {
        "subject_trial_table": subject_paths["trial_table"],
        "subject_condition_table": subject_paths["condition_table"],
        "readme": readme_path,
    }


def export_pattern_expression_results(
    *,
    subject_id: str,
    condition_labels,
    times,
    expression_by_effect: dict[str, np.ndarray],
    trial_index=None,
    output_dir: str | Path = DEFAULT_OUTPUT_SUBDIR,
    overwrite: bool = True,
) -> dict[str, object]:
    """Build and save generic effect-level pattern-expression outputs.

    Parameters
    ----------
    subject_id : str
        Subject identifier without the ``sub-`` prefix.
    condition_labels : array-like
        Condition label per trial.
    times : array-like
        Time axis aligned with expression matrices.
    expression_by_effect : dict[str, np.ndarray]
        Pattern-expression matrix for each effect. Each matrix must have shape
        ``(n_trials, n_times)``.
    trial_index : array-like | None
        Optional original trial index values.
    output_dir : str | Path
        Output folder for exported files.
    overwrite : bool
        Whether to overwrite existing per-subject saved outputs.

    Returns
    -------
    dict[str, object]
        Built tables and saved output paths.
    """

    subject_label = str(subject_id).strip()
    if subject_label.startswith("sub-"):
        subject_label = subject_label[4:]

    trial_table = build_trial_pattern_expression_table(
        subject=subject_label,
        condition_labels=condition_labels,
        times=times,
        expression_by_effect=expression_by_effect,
        trial_index=trial_index,
    )
    condition_table = build_condition_average_pattern_expression_table(trial_table)

    output_paths = save_pattern_expression_tables(
        subject_id=subject_label,
        trial_table=trial_table,
        condition_table=condition_table,
        output_dir=output_dir,
        overwrite=overwrite,
    )

    return {
        "trial_table": trial_table,
        "condition_table": condition_table,
        "output_paths": output_paths,
    }
