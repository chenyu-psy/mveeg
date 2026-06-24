"""Shared subject-loop helpers for workflow modules."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


def process_subjects(
    *,
    subject_ids: list[str],
    progress_total: int,
    log_label: str,
    log_path: str | Path | None,
    experiment_name: str,
    process_one_subject,
) -> pd.DataFrame:
    """Process requested subjects with one progress bar per subject.

    Parameters
    ----------
    subject_ids : list[str]
        Subject IDs to process during the current run.
    progress_total : int
        Total progress-bar steps per subject. Values below 1 are treated as 1.
    log_label : str
        Short label written to the detailed log header.
    log_path : str | Path | None
        Optional path for the detailed technical log.
    experiment_name : str
        Experiment label written in the detailed log header.
    process_one_subject : callable
        Callback that accepts ``(subject_id, progress_bar)`` and returns
        ``(result_bundle, used_saved_result)``. The callback may raise to mark
        one subject as failed.

    Returns
    -------
    pd.DataFrame
        Table of subjects that failed during the current run.
    """

    progress_total = max(int(progress_total), 1)
    skipped_subjects = []

    log_file = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
        log_file.write(f"{log_label}: {experiment_name}\n")
        log_file.write(f"Subjects processed this run: {len(subject_ids)}\n\n")

    try:
        for subject_id in subject_ids:
            with tqdm(
                total=progress_total,
                desc=f"sub-{subject_id}",
                unit="step",
                leave=True,
            ) as subject_bar:
                subject_bar.set_postfix_str("running")
                try:
                    if log_file is None:
                        _, used_saved_result = process_one_subject(subject_id, subject_bar)
                    else:
                        with redirect_stdout(log_file), redirect_stderr(log_file):
                            _, used_saved_result = process_one_subject(
                                subject_id,
                                subject_bar,
                            )

                    if used_saved_result:
                        subject_bar.set_postfix_str("reused")
                    else:
                        subject_bar.set_postfix_str("done")
                except Exception as err:
                    skipped_subjects.append({"subject": subject_id, "reason": str(err)})
                    subject_bar.set_postfix_str("failed")
                    print(f"sub-{subject_id} failed: {err}")
                    if log_file is not None:
                        log_file.write(f"sub-{subject_id} failed: {err}\n")
    finally:
        if log_file is not None:
            log_file.close()

    return pd.DataFrame(skipped_subjects)
