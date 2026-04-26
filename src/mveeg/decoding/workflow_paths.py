"""Path and config helpers for decoding workflows."""

from __future__ import annotations

import json
from pathlib import Path

from .config import DecodingConfig


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


