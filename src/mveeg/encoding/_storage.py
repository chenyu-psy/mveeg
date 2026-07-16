"""Encoding-specific DuckDB schema and subject write adapter."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from .._provenance import jsonable
from .._results.store import (
    completed_subjects,
    create_subjects_table,
    insert_frame,
    mark_complete,
    mark_failed,
    normalize_result_path,
    now,
    read_analysis_row,
    require_columns,
    require_equal_frame,
    require_trial_prefix,
    subject_state,
    table_exists,
    table_names,
)
from .._results.store import (
    mark_pending as _mark_pending,
)

RESULT_SCHEMA_VERSION = 1
SUBJECT_TABLES = (
    "trials",
    "coefficients",
    "pattern_expression",
    "design_diagnostics",
    "covariance_diagnostics",
)
ANALYSIS_COLUMNS = (
    "schema_version",
    "mveeg_version",
    "seed",
    "config",
    "fingerprint",
    "created",
    "updated",
)
RESULT_COLUMNS = {
    "coefficients": ("subject_index", "time", "channel", "predictor", "beta"),
    "pattern_expression": (
        "subject_index",
        "epoch_index",
        "time",
        "component",
        "expression_group",
        "expression",
        "fold",
    ),
    "design_diagnostics": (
        "subject_index",
        "fold",
        "diagnostic",
        "predictor",
        "value",
        "threshold",
        "status",
        "message",
    ),
    "covariance_diagnostics": (
        "subject_index",
        "fold",
        "time",
        "n_train_trials",
        "n_channels",
        "rank",
        "condition_number",
        "log_determinant",
        "shrinkage",
        "status",
    ),
}


def result_path(file: str | Path) -> Path:
    return normalize_result_path(file, analysis="encoding")


def read_analysis(path: Path) -> dict[str, object] | None:
    """Read and validate the current encoding analysis schema."""

    row = read_analysis_row(path, analysis="encoding")
    if row is None:
        return None
    if tuple(row) != ANALYSIS_COLUMNS or row.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("Unsupported encoding result schema; regenerate the result file.")
    return row


def initialize_store(
    path: Path,
    *,
    mveeg_version: str,
    seed: int,
    config: dict[str, object],
    fingerprint: str,
    reset: bool,
) -> None:
    """Create or reset encoding analysis and subject-state tables."""

    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = now()
    with duckdb.connect(str(path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            if reset:
                connection.execute("DROP VIEW IF EXISTS condition_pattern_expression")
                for table in table_names(connection):
                    connection.execute(f'DROP TABLE "{table}"')
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis (
                    schema_version INTEGER,
                    mveeg_version VARCHAR,
                    seed BIGINT,
                    config JSON,
                    fingerprint VARCHAR,
                    created VARCHAR,
                    updated VARCHAR
                )
                """
            )
            create_subjects_table(connection)
            if connection.execute("SELECT COUNT(*) FROM analysis").fetchone()[0] == 0:
                connection.execute(
                    "INSERT INTO analysis VALUES (?, ?, ?, ?::JSON, ?, ?, ?)",
                    [
                        RESULT_SCHEMA_VERSION,
                        mveeg_version,
                        seed,
                        json.dumps(jsonable(config), sort_keys=True),
                        fingerprint,
                        timestamp,
                        timestamp,
                    ],
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise


def mark_pending(path: Path, subject: str, fingerprint: str) -> None:
    _mark_pending(path, subject, fingerprint, subject_tables=SUBJECT_TABLES)


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
    """Write one complete encoding subject in a transaction."""

    require_trial_prefix(
        trials,
        ("subject_index", "epoch_index", "training_group", "expression_group"),
    )
    for name, table in tables.items():
        require_columns(table, RESULT_COLUMNS[name], table_name=name)
    with duckdb.connect(str(path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            _write_support(connection, predictors, channels, time_bins)
            insert_frame(connection, "trials", trials)
            for name, table in tables.items():
                insert_frame(connection, name, table)
            _create_condition_view(connection)
            mark_complete(connection, subject, fingerprint)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise


def _write_support(
    connection: duckdb.DuckDBPyConnection,
    predictors: pd.DataFrame,
    channels: pd.DataFrame,
    time_bins: pd.DataFrame,
) -> None:
    if not table_exists(connection, "predictors"):
        insert_frame(connection, "predictors", predictors)
        insert_frame(connection, "channels", channels)
        insert_frame(connection, "time_bins", time_bins)
        return
    require_equal_frame(connection, "predictors", predictors)
    require_equal_frame(connection, "channels", channels)
    require_equal_frame(connection, "time_bins", time_bins)


def _create_condition_view(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE OR REPLACE VIEW condition_pattern_expression AS
        SELECT
            subject_index,
            expression_group AS condition,
            component,
            time,
            AVG(expression) AS expression_mean,
            STDDEV_SAMP(expression) AS expression_sd,
            STDDEV_SAMP(expression) / SQRT(COUNT(DISTINCT epoch_index)) AS expression_se,
            COUNT(DISTINCT epoch_index) AS n_trials
        FROM pattern_expression
        GROUP BY subject_index, expression_group, component, time
        """
    )


__all__ = [
    "completed_subjects",
    "initialize_store",
    "mark_failed",
    "mark_pending",
    "read_analysis",
    "result_path",
    "subject_state",
    "write_subject",
]
