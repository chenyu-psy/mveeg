"""Transactional DuckDB persistence for the public encoding contract."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import duckdb
import pandas as pd


SUBJECT_TABLES = (
    "trials",
    "coefficients",
    "pattern_expression",
    "design_diagnostics",
    "covariance_diagnostics",
)


def result_path(file: str | Path) -> Path:
    path = Path(file).expanduser()
    if path.suffix == "":
        path = path.with_suffix(".duckdb")
    if path.suffix != ".duckdb":
        raise ValueError("Encoding results must use the .duckdb suffix.")
    return path.resolve()


def read_analysis(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with duckdb.connect(str(path), read_only=True) as connection:
        if not _table_exists(connection, "analysis"):
            raise ValueError("Existing DuckDB file is not an mveeg encoding result.")
        columns = [row[1] for row in connection.execute("PRAGMA table_info('analysis')").fetchall()]
        values = connection.execute("SELECT * FROM analysis").fetchall()
        if len(values) != 1:
            raise ValueError("The encoding analysis table must contain exactly one row.")
    return dict(zip(columns, values[0]))


def initialize_store(
    path: Path,
    *,
    version: str,
    seed: int,
    config: dict[str, object],
    fingerprint: str,
    reset: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = _now()
    with duckdb.connect(str(path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            if reset:
                connection.execute("DROP VIEW IF EXISTS condition_pattern_expression")
                for table in _table_names(connection):
                    connection.execute(f'DROP TABLE "{table}"')
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis (
                    version VARCHAR,
                    seed BIGINT,
                    config JSON,
                    fingerprint VARCHAR,
                    created VARCHAR,
                    updated VARCHAR
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS subjects (
                    subject VARCHAR,
                    status VARCHAR,
                    fingerprint VARCHAR,
                    reason VARCHAR,
                    updated VARCHAR
                )
                """
            )
            if connection.execute("SELECT COUNT(*) FROM analysis").fetchone()[0] == 0:
                connection.execute(
                    "INSERT INTO analysis VALUES (?, ?, ?::JSON, ?, ?, ?)",
                    [
                        version,
                        seed,
                        json.dumps(config, sort_keys=True, default=str),
                        fingerprint,
                        now,
                        now,
                    ],
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise


def subject_state(path: Path) -> dict[str, tuple[str, str | None]]:
    if not path.exists():
        return {}
    with duckdb.connect(str(path), read_only=True) as connection:
        rows = connection.execute(
            "SELECT subject, status, fingerprint FROM subjects"
        ).fetchall()
    return {str(subject): (str(status), fingerprint) for subject, status, fingerprint in rows}


def mark_pending(path: Path, subject: str, fingerprint: str) -> None:
    with duckdb.connect(str(path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            for table in SUBJECT_TABLES:
                if _table_exists(connection, table):
                    connection.execute(f'DELETE FROM "{table}" WHERE subject = ?', [subject])
            connection.execute("DELETE FROM subjects WHERE subject = ?", [subject])
            connection.execute(
                "INSERT INTO subjects VALUES (?, 'pending', ?, NULL, ?)",
                [subject, fingerprint, _now()],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise


def mark_failed(path: Path, subject: str, fingerprint: str, reason: str) -> None:
    with duckdb.connect(str(path)) as connection:
        connection.execute("DELETE FROM subjects WHERE subject = ?", [subject])
        connection.execute(
            "INSERT INTO subjects VALUES (?, 'failed', ?, ?, ?)",
            [subject, fingerprint, reason, _now()],
        )
        connection.execute("UPDATE analysis SET updated = ?", [_now()])


def write_subject(
    path: Path,
    *,
    subject: str,
    fingerprint: str,
    trials: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    predictors: pd.DataFrame,
    channels: pd.DataFrame,
    time_bins: pd.DataFrame,
) -> None:
    with duckdb.connect(str(path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            _write_support(connection, predictors, channels, time_bins)
            _insert_frame(connection, "trials", trials)
            for name, table in tables.items():
                _insert_frame(connection, name, table)
            _create_condition_view(connection)
            connection.execute("DELETE FROM subjects WHERE subject = ?", [subject])
            connection.execute(
                "INSERT INTO subjects VALUES (?, 'complete', ?, NULL, ?)",
                [subject, fingerprint, _now()],
            )
            connection.execute("UPDATE analysis SET updated = ?", [_now()])
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise


def completed_subjects(path: Path) -> int:
    with duckdb.connect(str(path), read_only=True) as connection:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM subjects WHERE status = 'complete'"
            ).fetchone()[0]
        )


def _write_support(connection, predictors, channels, time_bins) -> None:
    if not _table_exists(connection, "predictors"):
        _insert_frame(connection, "predictors", predictors)
        _insert_frame(connection, "channels", channels)
        _insert_frame(connection, "time_bins", time_bins)
        return
    _require_equal_frame(connection, "predictors", predictors)
    _require_equal_frame(connection, "channels", channels)
    _require_equal_frame(connection, "time_bins", time_bins)


def _create_condition_view(connection) -> None:
    connection.execute(
        """
        CREATE OR REPLACE VIEW condition_pattern_expression AS
        SELECT
            subject,
            expression_group AS condition,
            component,
            time,
            AVG(expression) AS expression_mean,
            STDDEV_SAMP(expression) AS expression_sd,
            STDDEV_SAMP(expression) / SQRT(COUNT(DISTINCT trial)) AS expression_se,
            COUNT(DISTINCT trial) AS n_trials
        FROM pattern_expression
        GROUP BY subject, expression_group, component, time
        """
    )


def _insert_frame(connection, table_name: str, frame: pd.DataFrame) -> None:
    if len(frame) == 0:
        return
    connection.register("_mveeg_rows", frame)
    try:
        if not _table_exists(connection, table_name):
            connection.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM _mveeg_rows')
            return
        columns = [
            row[1]
            for row in connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        ]
        if set(columns) != set(frame.columns):
            raise ValueError(
                f"{table_name} columns changed between subjects: "
                f"saved={columns}, new={list(frame.columns)}."
            )
        names = ", ".join(f'"{column}"' for column in columns)
        connection.execute(
            f'INSERT INTO "{table_name}" ({names}) SELECT {names} FROM _mveeg_rows'
        )
    finally:
        connection.unregister("_mveeg_rows")


def _require_equal_frame(connection, table_name: str, expected: pd.DataFrame) -> None:
    observed = connection.execute(f'SELECT * FROM "{table_name}"').df()
    if list(observed.columns) != list(expected.columns) or not observed.equals(
        expected.reset_index(drop=True)
    ):
        raise RuntimeError(f"{table_name} differs between subjects.")


def _table_names(connection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    ]


def _table_exists(connection, table_name: str) -> bool:
    return bool(
        connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name = ?",
            [table_name],
        ).fetchone()[0]
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
