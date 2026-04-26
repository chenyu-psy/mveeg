"""Status-table helpers for resumable preprocessing workflows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


STATUS_COLUMNS = [
    "subject_number",
    "intermediate_saved",
    "hard_qc_saved",
    "autoreject_done",
    "final_saved",
    "last_completed_step",
    "status",
    "error_message",
    "updated_at",
]


def empty_status_table() -> pd.DataFrame:
    """Return an empty preprocessing status table with standard columns."""
    return pd.DataFrame(columns=STATUS_COLUMNS)


def status_table_subjects(subject_dirs: list[Path]) -> list[str]:
    """Return selected subject labels in workflow order."""
    return [subject_path.name for subject_path in subject_dirs]


def load_status_table(status_path: str | Path, subject_dirs: list[Path]) -> pd.DataFrame:
    """Load a preprocessing status table, creating defaults when missing.

    Parameters
    ----------
    status_path : str | Path
        TSV file used to track run progress across subjects.
    subject_dirs : list[Path]
        Selected subject folders for this run.

    Returns
    -------
    pd.DataFrame
        Status table with one row per selected subject.
    """
    status_path = Path(status_path)
    if status_path.exists():
        status_table = pd.read_csv(status_path, sep="\t")
    else:
        status_table = empty_status_table()

    subject_numbers = status_table_subjects(subject_dirs)
    if len(status_table) == 0:
        status_table = pd.DataFrame({"subject_number": subject_numbers})

    missing_subjects = [
        subject
        for subject in subject_numbers
        if subject not in set(status_table["subject_number"])
    ]
    if missing_subjects:
        status_table = pd.concat(
            [status_table, pd.DataFrame({"subject_number": missing_subjects})],
            ignore_index=True,
        )

    status_table = status_table[status_table["subject_number"].isin(subject_numbers)].copy()
    status_table = status_table.set_index("subject_number").reindex(subject_numbers).reset_index()

    for col in STATUS_COLUMNS:
        if col not in status_table.columns:
            status_table[col] = pd.NA

    bool_cols = ["intermediate_saved", "hard_qc_saved", "autoreject_done", "final_saved"]
    for col in bool_cols:
        status_table[col] = status_table[col].fillna(False).astype(bool)

    text_defaults = {
        "last_completed_step": "",
        "status": "pending",
        "error_message": "",
        "updated_at": "",
    }
    for col, default_value in text_defaults.items():
        status_table[col] = status_table[col].fillna(default_value).astype(str)

    return status_table[STATUS_COLUMNS]


def save_status_table(status_path: str | Path, status_table: pd.DataFrame) -> Path:
    """Write the preprocessing status table to disk."""
    status_path = Path(status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_table.to_csv(status_path, sep="\t", index=False)
    return status_path


def update_subject_status(
    *,
    status_path: str | Path,
    subject_dirs: list[Path],
    subject_number: str,
    intermediate_saved: bool | None = None,
    hard_qc_saved: bool | None = None,
    autoreject_done: bool | None = None,
    final_saved: bool | None = None,
    last_completed_step: str | None = None,
    status: str | None = None,
    error_message: str | None = None,
) -> pd.DataFrame:
    """Update one subject row in the preprocessing status table."""
    status_table = load_status_table(status_path, subject_dirs)
    row_ix = status_table.index[status_table["subject_number"].eq(subject_number)][0]
    updates = {
        "intermediate_saved": intermediate_saved,
        "hard_qc_saved": hard_qc_saved,
        "autoreject_done": autoreject_done,
        "final_saved": final_saved,
        "last_completed_step": last_completed_step,
        "status": status,
        "error_message": error_message,
    }
    for col, value in updates.items():
        if value is not None:
            status_table.at[row_ix, col] = value
    status_table.at[row_ix, "updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_status_table(status_path, status_table)
    return status_table


def should_overwrite(overwrite_step: bool, overwrite_all: bool) -> bool:
    """Resolve one step's overwrite decision from global and local flags."""
    return bool(overwrite_all or overwrite_step)
