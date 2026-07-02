"""Single-file result table storage for mveeg workflows."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Callable

import duckdb
import pandas as pd


_DUCKDB_SUFFIX = ".duckdb"
_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INTERNAL_PREFIX = "_mveeg_"
_STORE_SCHEMA_VERSION = 1


def resolve_result_file(file: str | Path | None) -> Path | None:
    """Return a normalized DuckDB result-file path.

    Parameters
    ----------
    file : str | Path | None
        Result file requested by the caller. ``None`` disables file-backed
        storage. Paths without a suffix receive the ``.duckdb`` suffix.

    Returns
    -------
    Path | None
        Normalized result path, or ``None`` when file storage is disabled.
    """

    if file is None:
        return None

    path = Path(file)
    if path.suffix == "":
        return path.with_suffix(_DUCKDB_SUFFIX)
    if path.suffix != _DUCKDB_SUFFIX:
        raise ValueError("mveeg result files must use the .duckdb suffix.")
    return path


def write_result_tables(file: str | Path, tables: dict[str, pd.DataFrame]) -> Path:
    """Write result tables to one DuckDB file.

    Parameters
    ----------
    file : str | Path
        Target ``.duckdb`` result file.
    tables : dict[str, pandas.DataFrame]
        Tables keyed by DuckDB table name.

    Returns
    -------
    Path
        Normalized path written by this call.
    """

    path = resolve_result_file(file)
    if path is None:
        raise ValueError("file must not be None when writing result tables.")
    _validate_table_names(tables)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp{_DUCKDB_SUFFIX}")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        with duckdb.connect(str(tmp_path)) as con:
            for table_name, table_df in tables.items():
                con.register("_mveeg_table", table_df)
                con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM _mveeg_table')
                con.unregister("_mveeg_table")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return path


def read_result_tables(
    file: str | Path,
    required_tables: list[str] | tuple[str, ...] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read result tables from a DuckDB result file.

    Parameters
    ----------
    file : str | Path
        Existing ``.duckdb`` result file.
    required_tables : list[str] | tuple[str, ...] | None
        Required table names. When ``None``, all user tables are returned.

    Returns
    -------
    dict[str, pandas.DataFrame]
        DataFrames keyed by table name.
    """

    path = resolve_result_file(file)
    if path is None:
        raise ValueError("file must not be None when reading result tables.")
    if not path.exists():
        raise FileNotFoundError(path)

    required = None if required_tables is None else list(required_tables)
    if required is not None:
        _validate_table_names({table_name: pd.DataFrame() for table_name in required})

    with duckdb.connect(str(path), read_only=True) as con:
        available = [
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        ]
        table_names = (
            sorted(table_name for table_name in available if not table_name.startswith(_INTERNAL_PREFIX))
            if required is None
            else required
        )
        missing = sorted(set(table_names).difference(available))
        if len(missing) > 0:
            raise ValueError(f"Result file is missing required tables: {missing}")
        return {
            table_name: con.execute(f'SELECT * FROM "{table_name}"').df()
            for table_name in table_names
        }


def result_config_hash(payload: dict[str, object]) -> str:
    """Return a stable hash for result-store compatibility checks."""

    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def init_result_store(
    file: str | Path,
    *,
    analysis_type: str,
    config_hash: str,
    run_name: str,
) -> Path:
    """Create or validate the DuckDB manifest for an analysis result store."""

    path = resolve_result_file(file)
    if path is None:
        raise ValueError("file must not be None when initializing a result store.")
    _validate_table_names({"_mveeg_run": pd.DataFrame(), "_mveeg_subjects": pd.DataFrame()})
    path.parent.mkdir(parents=True, exist_ok=True)
    now = _now_iso()

    with duckdb.connect(str(path)) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS _mveeg_run (
                analysis_type VARCHAR,
                schema_version INTEGER,
                config_hash VARCHAR,
                run_name VARCHAR,
                created_at VARCHAR,
                updated_at VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS _mveeg_subjects (
                subject VARCHAR,
                status VARCHAR,
                reason VARCHAR,
                updated_at VARCHAR
            )
            """
        )
        existing = con.execute("SELECT analysis_type, schema_version, config_hash FROM _mveeg_run").fetchall()
        if len(existing) == 0:
            con.execute(
                "INSERT INTO _mveeg_run VALUES (?, ?, ?, ?, ?, ?)",
                [analysis_type, _STORE_SCHEMA_VERSION, config_hash, run_name, now, now],
            )
        else:
            stored_type, schema_version, stored_hash = existing[0]
            if stored_type != analysis_type or int(schema_version) != _STORE_SCHEMA_VERSION or stored_hash != config_hash:
                raise ValueError(
                    "Existing result file was created with incompatible analysis settings. "
                    "Use a different file or overwrite the store."
                )
            con.execute("UPDATE _mveeg_run SET run_name = ?, updated_at = ?", [run_name, now])
    return path


def result_store_subjects_to_run(
    file: str | Path,
    subject_ids: list[str],
    *,
    overwrite: bool,
) -> list[str]:
    """Return requested subjects that should be processed for this run."""

    statuses = read_result_subject_status(file)
    requested = [str(subject_id) for subject_id in subject_ids]
    if overwrite:
        return requested
    return [
        subject_id
        for subject_id in requested
        if statuses.get(subject_id) != "complete"
    ]


def run_incremental_result_store(
    file: str | Path,
    *,
    subject_ids: list[str],
    overwrite: bool,
    process_subjects: Callable[[list[str]], None],
    finalize: Callable[[], dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """Run missing requested subjects, then rebuild public result tables."""

    subjects_to_run = result_store_subjects_to_run(
        file,
        subject_ids,
        overwrite=overwrite,
    )
    if len(subjects_to_run) > 0:
        mark_result_subjects_pending(file, subjects_to_run)
        process_subjects(subjects_to_run)
    return finalize()


def read_result_subject_status(file: str | Path) -> dict[str, str]:
    """Read per-subject store status."""

    path = resolve_result_file(file)
    if path is None or not path.exists():
        return {}
    with duckdb.connect(str(path), read_only=True) as con:
        if not _table_exists(con, "_mveeg_subjects"):
            return {}
        rows = con.execute("SELECT subject, status FROM _mveeg_subjects").fetchall()
    return {str(subject): str(status) for subject, status in rows}


def mark_result_subjects_pending(file: str | Path, subject_ids: list[str]) -> None:
    """Mark requested subjects as pending before processing."""

    if len(subject_ids) == 0:
        return
    now = _now_iso()
    with duckdb.connect(str(resolve_result_file(file))) as con:
        for subject_id in subject_ids:
            con.execute("DELETE FROM _mveeg_subjects WHERE subject = ?", [str(subject_id)])
            con.execute(
                "INSERT INTO _mveeg_subjects VALUES (?, ?, ?, ?)",
                [str(subject_id), "pending", None, now],
            )


def replace_subject_table_rows(
    file: str | Path,
    *,
    subject_ids: list[str],
    tables: dict[str, pd.DataFrame],
) -> None:
    """Replace rows for subjects in internal per-subject tables."""

    _validate_table_names(tables)
    subjects = [str(subject_id) for subject_id in subject_ids]
    with duckdb.connect(str(resolve_result_file(file))) as con:
        for table_name, table_df in tables.items():
            if len(subjects) > 0 and _table_exists(con, table_name):
                for subject_id in subjects:
                    con.execute(f'DELETE FROM "{table_name}" WHERE subject = ?', [subject_id])
            if len(table_df) > 0:
                _append_table(con, table_name, table_df)


def update_result_subject_status(
    file: str | Path,
    *,
    completed_subject_ids: list[str],
    skipped: pd.DataFrame,
) -> None:
    """Update complete and skipped statuses after a processing pass."""

    now = _now_iso()
    with duckdb.connect(str(resolve_result_file(file))) as con:
        for subject_id in completed_subject_ids:
            con.execute("DELETE FROM _mveeg_subjects WHERE subject = ?", [str(subject_id)])
            con.execute(
                "INSERT INTO _mveeg_subjects VALUES (?, ?, ?, ?)",
                [str(subject_id), "complete", None, now],
            )
        if len(skipped) > 0:
            for _, row in skipped.iterrows():
                subject_id = str(row["subject"])
                reason = str(row["reason"])
                con.execute("DELETE FROM _mveeg_subjects WHERE subject = ?", [subject_id])
                con.execute(
                    "INSERT INTO _mveeg_subjects VALUES (?, ?, ?, ?)",
                    [subject_id, "skipped", reason, now],
                )


def read_internal_table(file: str | Path, table_name: str) -> pd.DataFrame:
    """Read an internal DuckDB table, returning an empty table when absent."""

    _validate_table_names({table_name: pd.DataFrame()})
    path = resolve_result_file(file)
    if path is None or not path.exists():
        return pd.DataFrame()
    with duckdb.connect(str(path), read_only=True) as con:
        if not _table_exists(con, table_name):
            return pd.DataFrame()
        return con.execute(f'SELECT * FROM "{table_name}"').df()


def result_skipped_table(file: str | Path) -> pd.DataFrame:
    """Build the public skipped table from subject status."""

    path = resolve_result_file(file)
    if path is None or not path.exists():
        return pd.DataFrame(columns=["subject", "reason"])
    with duckdb.connect(str(path), read_only=True) as con:
        if not _table_exists(con, "_mveeg_subjects"):
            return pd.DataFrame(columns=["subject", "reason"])
        return con.execute(
            "SELECT subject, reason FROM _mveeg_subjects WHERE status = 'skipped' ORDER BY subject"
        ).df()


def replace_result_tables(file: str | Path, tables: dict[str, pd.DataFrame]) -> None:
    """Replace public result tables inside an existing DuckDB store."""

    _validate_table_names(tables)
    with duckdb.connect(str(resolve_result_file(file))) as con:
        for table_name, table_df in tables.items():
            con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            con.register("_mveeg_table", table_df)
            con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM _mveeg_table')
            con.unregister("_mveeg_table")
        if _table_exists(con, "_mveeg_run"):
            con.execute("UPDATE _mveeg_run SET updated_at = ?", [_now_iso()])


def _validate_table_names(tables: dict[str, pd.DataFrame]) -> None:
    """Reject table names that cannot be safely used as DuckDB identifiers."""

    bad_names = [table_name for table_name in tables if _TABLE_NAME_RE.match(table_name) is None]
    if len(bad_names) > 0:
        raise ValueError(f"Invalid result table names: {bad_names}")


def _append_table(con, table_name: str, table_df: pd.DataFrame) -> None:
    con.register("_mveeg_table", table_df)
    try:
        if _table_exists(con, table_name):
            con.execute(f'INSERT INTO "{table_name}" SELECT * FROM _mveeg_table')
        else:
            con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM _mveeg_table')
    finally:
        con.unregister("_mveeg_table")


def _table_exists(con, table_name: str) -> bool:
    return bool(
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
            [table_name],
        ).fetchone()[0]
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
