"""Subject-selection helpers for preprocessing workflows."""

from __future__ import annotations

from pathlib import Path


def get_subject_selection(
    *,
    root_dir: str | Path,
    selected_subjects: list[str] | None = None,
    condition_replacements: dict[str, dict[str, str]] | None = None,
) -> tuple[list[Path], dict[str, Path]]:
    """Return subject folders selected for one preprocessing run.

    Parameters
    ----------
    root_dir : str | Path
        Folder that contains one raw-data folder per subject.
    selected_subjects : list[str] | None, optional
        Optional whitelist of raw subject folder names to process.
    condition_replacements : dict[str, dict[str, str]] | None, optional
        Optional subject assembly plan. Subjects that appear only as donor
        sources are excluded from the main processing loop because they are
        loaded on demand when assembling another subject.

    Returns
    -------
    tuple[list[Path], dict[str, Path]]
        Selected raw subject folders and a name-to-path lookup table.
    """
    subject_dir_map = {
        path.name: path
        for path in sorted(path for path in Path(root_dir).iterdir() if path.is_dir())
    }

    if selected_subjects:
        subject_dirs = [subject_dir_map[str(subject)] for subject in selected_subjects]
    else:
        subject_dirs = list(subject_dir_map.values())

    replacement_map = condition_replacements or {}
    target_subjects = set(replacement_map.keys())
    replacement_donors = {
        donor_subject
        for subject_replacements in replacement_map.values()
        for donor_subject in subject_replacements.values()
    }
    donor_only_subjects = replacement_donors - target_subjects
    subject_dirs = [
        subject_dir
        for subject_dir in subject_dirs
        if subject_dir.name not in donor_only_subjects
    ]
    return subject_dirs, subject_dir_map
