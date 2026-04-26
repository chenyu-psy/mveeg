"""Helpers for subject labels and derivative paths that follow MNE-BIDS naming.

These helpers keep the project's BIDS-style naming rules explicit and
consistent across preprocessing, visualization, and decoding code. The goal is
not to abstract away ``mne-bids``, but to avoid repeating the same filename
logic in multiple modules.
"""

from __future__ import annotations

from pathlib import Path


def normalize_subject_id(subject_label: str, subject_prefix: str = "sub") -> str:
    """Return a subject ID without the leading BIDS subject prefix.

    Parameters
    ----------
    subject_label : str
        Raw subject label such as ``"1001"``, ``"sub1001"``, or
        ``"sub-1001"``.
    subject_prefix : str, optional
        BIDS subject prefix used before the subject ID.

    Returns
    -------
    str
        Subject identifier without the leading prefix.
    """
    subject_str = str(subject_label)
    dashed_prefix = f"{subject_prefix}-"
    if subject_str.startswith(dashed_prefix):
        return subject_str.replace(dashed_prefix, "", 1)
    if subject_str.startswith(subject_prefix):
        return subject_str.replace(subject_prefix, "", 1)
    return subject_str


def build_subject_label(subject_id: str, subject_prefix: str = "sub") -> str:
    """Return a BIDS-style subject label.

    Parameters
    ----------
    subject_id : str
        Subject ID with or without the leading prefix.
    subject_prefix : str, optional
        BIDS subject prefix used before the normalized subject ID.

    Returns
    -------
    str
        Subject label such as ``"sub-1001"``.
    """
    normalized_id = normalize_subject_id(subject_id, subject_prefix=subject_prefix)
    return f"{subject_prefix}-{normalized_id}"


def build_task_stem(
    subject_id: str,
    experiment_name: str,
    subject_prefix: str = "sub",
) -> str:
    """Return the shared MNE-BIDS stem for raw task files.

    Parameters
    ----------
    subject_id : str
        Subject ID with or without the leading prefix.
    experiment_name : str
        Task or experiment label used in filenames.
    subject_prefix : str, optional
        BIDS subject prefix used before the normalized subject ID.

    Returns
    -------
    str
        Stem such as ``"sub-1001_task-experiment1"``.
    """
    return f"{build_subject_label(subject_id, subject_prefix=subject_prefix)}_task-{experiment_name}"


def build_derivative_stem(
    subject_id: str,
    experiment_name: str,
    subject_prefix: str = "sub",
    derivative_label: str = "preprocessed",
) -> str:
    """Return the shared MNE-BIDS-style stem for derivative files.

    Parameters
    ----------
    subject_id : str
        Subject ID with or without the leading prefix.
    experiment_name : str
        Task or experiment label used in filenames.
    subject_prefix : str, optional
        BIDS subject prefix used before the normalized subject ID.
    derivative_label : str, optional
        Label written after ``desc-`` in derivative filenames.

    Returns
    -------
    str
        Stem such as ``"sub-1001_experiment1_desc-preprocessed"``.
    """
    subject_label = build_subject_label(subject_id, subject_prefix=subject_prefix)
    return f"{subject_label}_{experiment_name}_desc-{derivative_label}"


def find_subject_dir(
    root_dir: str | Path,
    subject_label: str,
    subject_prefix: str = "sub",
) -> Path:
    """Return the raw-data folder for one subject.

    Parameters
    ----------
    root_dir : str | Path
        Parent folder that contains the raw subject folders.
    subject_label : str
        Subject ID with or without the leading prefix.
    subject_prefix : str, optional
        Subject prefix used in raw folder names.

    Returns
    -------
    Path
        Matching subject folder.

    Raises
    ------
    FileNotFoundError
        If no matching subject folder exists.
    """
    subject_str = str(subject_label)
    subject_id = normalize_subject_id(subject_str, subject_prefix=subject_prefix)
    candidates = [
        Path(root_dir) / subject_str,
        Path(root_dir) / f"{subject_prefix}{subject_id}",
        Path(root_dir) / build_subject_label(subject_id, subject_prefix=subject_prefix),
        Path(root_dir) / subject_id,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find a raw-data folder for subject {subject_label}")


def derivative_file_path(
    data_dir: str | Path,
    subject_id: str,
    experiment_name: str,
    suffix: str,
    extension: str,
    *,
    subject_prefix: str = "sub",
    derivative_dirname: str = "derivatives",
    datatype: str = "eeg",
    derivative_label: str = "preprocessed",
) -> Path:
    """Build one derivative file path using the project's MNE-BIDS-style stem.

    Parameters
    ----------
    data_dir : str | Path
        Dataset root that contains the derivative folder.
    subject_id : str
        Subject ID with or without the leading prefix.
    experiment_name : str
        Task or experiment label used in filenames.
    suffix : str
        File suffix such as ``"epo"`` or ``"trial_qc"``.
    extension : str
        Filename extension including the leading dot.
    subject_prefix : str, optional
        BIDS subject prefix used before the normalized subject ID.
    derivative_dirname : str, optional
        Name of the derivative folder below ``data_dir``.
    datatype : str, optional
        Datatype folder that contains the saved file.
    derivative_label : str, optional
        Label written after ``desc-`` in derivative filenames.

    Returns
    -------
    Path
        Full path to the requested derivative file.
    """
    subject_label = build_subject_label(subject_id, subject_prefix=subject_prefix)
    stem = build_derivative_stem(
        subject_id,
        experiment_name,
        subject_prefix=subject_prefix,
        derivative_label=derivative_label,
    )
    return Path(data_dir) / derivative_dirname / subject_label / datatype / f"{stem}_{suffix}{extension}"


def get_subject_ids_from_derivatives(
    data_dir: str | Path,
    *,
    subject_prefix: str = "sub",
    derivative_dirname: str = "derivatives",
    datatype: str = "eeg",
) -> list[str]:
    """Return subject IDs that have derivative data for one datatype.

    Parameters
    ----------
    data_dir : str | Path
        Dataset root that contains the derivative folder.
    subject_prefix : str, optional
        BIDS subject prefix used in derivative subject folders.
    derivative_dirname : str, optional
        Name of the derivative folder below ``data_dir``.
    datatype : str, optional
        Datatype folder to search for.

    Returns
    -------
    list[str]
        Sorted subject IDs without the leading subject prefix.
    """
    subject_dirs = sorted(Path(data_dir).joinpath(derivative_dirname).glob(f"{subject_prefix}-*/{datatype}"))
    return [normalize_subject_id(path.parent.name, subject_prefix=subject_prefix) for path in subject_dirs]
