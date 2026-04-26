"""I/O helpers for reusable EEG analysis workflows.

This package contains small helpers for reading, writing, and naming files in
ways that are shared across preprocessing, encoding, and decoding workflows.
"""

from .bids import (
    build_derivative_stem,
    build_subject_label,
    build_task_stem,
    derivative_file_path,
    find_subject_dir,
    get_subject_ids_from_derivatives,
    normalize_subject_id,
)

__all__ = [
    "normalize_subject_id",
    "build_subject_label",
    "build_task_stem",
    "build_derivative_stem",
    "find_subject_dir",
    "derivative_file_path",
    "get_subject_ids_from_derivatives",
]
