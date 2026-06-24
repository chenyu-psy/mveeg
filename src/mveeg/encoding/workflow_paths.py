"""Path helpers for encoding workflows."""

from __future__ import annotations

from pathlib import Path


def prepare_encoding_paths(
    base_dir: str | Path,
    run_name: str,
    results_subdir: str = "main",
) -> dict[str, Path]:
    """Create and return standard output folders for one encoding run."""

    base_dir = Path(base_dir)
    results_dir = base_dir / "results" / results_subdir / "encoding" / run_name
    subject_results_dir = results_dir / "subject_level"
    log_path = results_dir / "encoding.log"

    results_dir.mkdir(parents=True, exist_ok=True)
    subject_results_dir.mkdir(parents=True, exist_ok=True)

    return {
        "results_dir": results_dir,
        "subject_results_dir": subject_results_dir,
        "log_path": log_path,
    }


def infer_experiment_settings(
    data_dir: str | Path,
    experiment_name: str | None,
    results_subdir: str | None,
) -> tuple[str, str]:
    """Fill in experiment-specific encoding settings from the data folder.

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
        Experiment label and results subdirectory used by the encoding helpers.
    """

    data_dir = Path(data_dir)
    inferred_name = data_dir.name

    if experiment_name is None:
        experiment_name = inferred_name
    if results_subdir is None:
        results_subdir = inferred_name

    return experiment_name, results_subdir

