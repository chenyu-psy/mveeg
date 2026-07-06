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
from .results import (
    init_result_store,
    mark_result_subjects_pending,
    read_internal_table,
    read_result_subject_status,
    read_result_tables,
    replace_result_tables,
    replace_subject_table_rows,
    resolve_result_file,
    result_config_hash,
    result_skipped_table,
    result_store_subjects_to_run,
    run_incremental_result_store,
    update_result_subject_status,
    write_result_tables,
)

__all__ = [
    "normalize_subject_id",
    "build_subject_label",
    "build_task_stem",
    "build_derivative_stem",
    "find_subject_dir",
    "derivative_file_path",
    "get_subject_ids_from_derivatives",
    "init_result_store",
    "mark_result_subjects_pending",
    "read_internal_table",
    "read_result_subject_status",
    "resolve_result_file",
    "read_result_tables",
    "replace_result_tables",
    "replace_subject_table_rows",
    "result_config_hash",
    "result_skipped_table",
    "result_store_subjects_to_run",
    "run_incremental_result_store",
    "update_result_subject_status",
    "write_result_tables",
]
