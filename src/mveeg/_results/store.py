"""Mechanical DuckDB operations shared by result writers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


def normalize_result_path(file: str | Path, *, analysis: str) -> Path:
    """Normalize a caller-provided DuckDB result path."""

    path = Path(file).expanduser()
    if path.suffix == "":
        path = path.with_suffix(".duckdb")
    if path.suffix != ".duckdb":
        raise ValueError(f"{analysis.capitalize()} results must use the .duckdb suffix.")
    return path.resolve()


def table_exists(connection: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Return whether a main-schema table exists."""

    return bool(
        connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name = ?",
            [table_name],
        ).fetchone()[0]
    )


def table_names(connection: duckdb.DuckDBPyConnection) -> list[str]:
    """Return all main-schema table names."""

    return [
        row[0]
        for row in connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    ]


def read_analysis_row(path: Path, *, analysis: str) -> dict[str, object] | None:
    """Read the single analysis row from an existing model store."""

    if not path.exists():
        return None
    with duckdb.connect(str(path), read_only=True) as connection:
        if not table_exists(connection, "analysis"):
            raise ValueError(f"Existing DuckDB file is not an mveeg {analysis} result.")
        columns = [row[1] for row in connection.execute("PRAGMA table_info('analysis')").fetchall()]
        values = connection.execute("SELECT * FROM analysis").fetchall()
        if len(values) != 1:
            raise ValueError(f"The {analysis} analysis table must contain exactly one row.")
    return dict(zip(columns, values[0], strict=True))


def create_subjects_table(connection: duckdb.DuckDBPyConnection) -> None:
    """Create the common subject-state table."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS subjects (
            subject_index VARCHAR,
            status VARCHAR,
            fingerprint VARCHAR,
            reason VARCHAR,
            updated VARCHAR
        )
        """
    )


def subject_state(path: Path) -> dict[str, tuple[str, str | None]]:
    """Return saved status and input fingerprint by subject index."""

    if not path.exists():
        return {}
    with duckdb.connect(str(path), read_only=True) as connection:
        if not table_exists(connection, "subjects"):
            return {}
        rows = connection.execute(
            "SELECT subject_index, status, fingerprint FROM subjects"
        ).fetchall()
    return {str(subject): (str(status), fingerprint) for subject, status, fingerprint in rows}


def mark_pending(
    path: Path,
    subject_index: str,
    fingerprint: str,
    *,
    subject_tables: tuple[str, ...],
) -> None:
    """Delete stale subject rows and mark a new transaction pending."""

    with duckdb.connect(str(path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            for table in subject_tables:
                if table_exists(connection, table):
                    connection.execute(
                        f'DELETE FROM "{table}" WHERE subject_index = ?', [subject_index]
                    )
            connection.execute("DELETE FROM subjects WHERE subject_index = ?", [subject_index])
            connection.execute(
                "INSERT INTO subjects VALUES (?, 'pending', ?, NULL, ?)",
                [subject_index, fingerprint, now()],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise


def mark_failed(path: Path, subject_index: str, fingerprint: str, reason: str) -> None:
    """Record a subject failure atomically."""

    with duckdb.connect(str(path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute("DELETE FROM subjects WHERE subject_index = ?", [subject_index])
            connection.execute(
                "INSERT INTO subjects VALUES (?, 'failed', ?, ?, ?)",
                [subject_index, fingerprint, reason, now()],
            )
            connection.execute("UPDATE analysis SET updated = ?", [now()])
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise


def mark_complete(
    connection: duckdb.DuckDBPyConnection,
    subject_index: str,
    fingerprint: str,
) -> None:
    """Mark a subject complete within its active write transaction."""

    connection.execute("DELETE FROM subjects WHERE subject_index = ?", [subject_index])
    connection.execute(
        "INSERT INTO subjects VALUES (?, 'complete', ?, NULL, ?)",
        [subject_index, fingerprint, now()],
    )
    connection.execute("UPDATE analysis SET updated = ?", [now()])


def completed_subjects(path: Path) -> int:
    """Return the number of completed subjects."""

    with duckdb.connect(str(path), read_only=True) as connection:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM subjects WHERE status = 'complete'"
            ).fetchone()[0]
        )


def insert_frame(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    """Create or append a frame while preserving exact column order."""

    if len(frame) == 0:
        return
    connection.register("_mveeg_rows", frame)
    try:
        if not table_exists(connection, table_name):
            connection.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM _mveeg_rows')
            return
        columns = [
            row[1] for row in connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        ]
        if columns != list(frame.columns):
            raise ValueError(
                f"{table_name} columns changed between subjects: "
                f"saved={columns}, new={list(frame.columns)}."
            )
        names = ", ".join(f'"{column}"' for column in columns)
        connection.execute(f'INSERT INTO "{table_name}" ({names}) SELECT {names} FROM _mveeg_rows')
    finally:
        connection.unregister("_mveeg_rows")


def require_equal_frame(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    expected: pd.DataFrame,
) -> None:
    """Require one support table to match earlier subjects exactly."""

    observed = connection.execute(f'SELECT * FROM "{table_name}"').df()
    if list(observed.columns) != list(expected.columns) or not observed.equals(
        expected.reset_index(drop=True)
    ):
        raise RuntimeError(f"{table_name} differs between subjects.")


def require_columns(
    frame: pd.DataFrame,
    expected: tuple[str, ...],
    *,
    table_name: str,
) -> None:
    """Require exact fixed-result columns and order."""

    if tuple(frame.columns) != expected:
        raise ValueError(
            f"{table_name} has invalid columns: expected={list(expected)}, "
            f"observed={list(frame.columns)}."
        )


def require_trial_prefix(frame: pd.DataFrame, prefix: tuple[str, ...]) -> None:
    """Require canonical identity and model columns before dynamic metadata."""

    observed = tuple(frame.columns[: len(prefix)])
    if observed != prefix:
        raise ValueError(f"trials must begin with {list(prefix)}, observed {list(observed)}.")


def now() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()
