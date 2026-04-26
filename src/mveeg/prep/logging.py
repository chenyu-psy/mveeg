"""Plain-text logging helpers for preprocessing workflows."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TextIO


def write_run_log_header(
    output_file: TextIO,
    *,
    total_subjects: int,
    log_path: str | Path,
    started_at: datetime | None = None,
) -> None:
    """Write a short preprocessing-run header to a log file.

    Parameters
    ----------
    output_file : TextIO
        Open text stream used for the preprocessing log.
    total_subjects : int
        Number of subjects selected for the current run.
    log_path : str | Path
        Location of the log file on disk.
    started_at : datetime | None, optional
        Run start time. If omitted, the current time is used.

    Returns
    -------
    None
        The function writes formatted text to ``output_file``.
    """
    started_at = datetime.now() if started_at is None else started_at
    output_file.write("\n" + "=" * 72 + "\n")
    output_file.write("PREPROCESSING RUN\n")
    output_file.write("=" * 72 + "\n")
    output_file.write(f"Started: {started_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
    output_file.write(f"Subjects in this run: {total_subjects}\n")
    output_file.write(f"Log file: {Path(log_path).resolve()}\n\n")


def write_subject_log_header(
    output_file: TextIO,
    *,
    subject_ix: int,
    total_subjects: int,
    subject_number: str,
) -> None:
    """Write one subject header to a preprocessing log.

    Parameters
    ----------
    output_file : TextIO
        Open text stream used for the preprocessing log.
    subject_ix : int
        One-based index of the current subject in this run.
    total_subjects : int
        Total number of subjects selected for the run.
    subject_number : str
        Subject label shown in the log.

    Returns
    -------
    None
        The function writes formatted text to ``output_file``.
    """
    output_file.write("\n" + "-" * 72 + "\n")
    output_file.write(f"Subject {subject_ix}/{total_subjects}: {subject_number}\n")
    output_file.write("-" * 72 + "\n")


@contextmanager
def redirect_output_to_file(output_file):
    """Temporarily redirect stdout and stderr to an open text stream."""
    import sys

    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = output_file
    sys.stderr = sys.stdout
    try:
        yield None
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr
