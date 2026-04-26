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


