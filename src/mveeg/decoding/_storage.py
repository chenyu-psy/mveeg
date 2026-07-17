"""Decoding-specific DuckDB schema and subject write adapter."""

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

RESULT_SCHEMA_VERSION = 2
SUBJECT_TABLES = (
    "trials",
    "accuracy",
    "classifier_evidence",
    "confusion_matrix",
    "patterns",
    "generalization",
)
ANALYSIS_COLUMNS = (
    "schema_version",
    "mveeg_version",
    "output",
    "generalization",
    "seed",
    "config",
    "fingerprint",
    "created",
    "updated",
)
MEAN_COLUMNS = {
    "accuracy": ("subject_index", "time", "permutation", "accuracy", "n_correct", "n_trials"),
    "classifier_evidence": ("subject_index", "epoch_index", "time", "evidence", "n_models"),
    "confusion_matrix": ("subject_index", "time", "actual", "predicted", "count"),
    "patterns": ("subject_index", "time", "channel", "component", "pattern"),
    "generalization": (
        "subject_index",
        "condition",
        "train_time",
        "test_time",
        "permutation",
        "accuracy",
        "n_correct",
        "n_trials",
        "target_evidence",
    ),
}
ALL_COLUMNS = {
    "accuracy": (
        "subject_index",
        "repeat",
        "fold",
        "time",
        "permutation",
        "accuracy",
        "n_correct",
        "n_trials",
    ),
    "classifier_evidence": ("subject_index", "epoch_index", "repeat", "fold", "time", "evidence"),
    "confusion_matrix": ("subject_index", "repeat", "fold", "time", "actual", "predicted", "count"),
    "patterns": ("subject_index", "repeat", "fold", "time", "channel", "component", "pattern"),
    "generalization": (
        "subject_index",
        "condition",
        "repeat",
        "fold",
        "train_time",
        "test_time",
        "permutation",
        "accuracy",
        "n_correct",
        "n_trials",
        "target_evidence",
    ),
}


class _LegacyDecodingSchema(ValueError):
    """Known decoding schema that may be replaced by a full recomputation."""


def result_path(file: str | Path) -> Path:
    return normalize_result_path(file, analysis="decoding")


def read_analysis(path: Path) -> dict[str, object] | None:
    """Read and validate the current decoding analysis schema."""

    row = read_analysis_row(path, analysis="decoding")
    if row is None:
        return None
    if tuple(row) != ANALYSIS_COLUMNS:
        raise ValueError("Unsupported decoding result schema; regenerate the result file.")
    if row.get("schema_version") == 1:
        raise _LegacyDecodingSchema(
            "Decoding result schema 1 is incompatible with schema 2; "
            "use another file or recompute='all' to regenerate it."
        )
    if row.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("Unsupported decoding result schema; regenerate the result file.")
    return row


def initialize_store(
    path: Path,
    *,
    mveeg_version: str,
    output: str,
    generalization: dict[str, list[object]] | None,
    seed: int,
    config: dict[str, object],
    fingerprint: str,
    reset: bool,
) -> None:
    """Create or reset decoding analysis and subject-state tables."""

    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = now()
    with duckdb.connect(str(path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            if reset:
                for table in table_names(connection):
                    connection.execute(f'DROP TABLE "{table}"')
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis (
                    schema_version INTEGER,
                    mveeg_version VARCHAR,
                    output VARCHAR,
                    generalization JSON,
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
                    "INSERT INTO analysis VALUES (?, ?, ?, ?::JSON, ?, ?::JSON, ?, ?, ?)",
                    [
                        RESULT_SCHEMA_VERSION,
                        mveeg_version,
                        output,
                        None if generalization is None else json.dumps(jsonable(generalization)),
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
    classifier: dict[str, object],
    components: list[list[str]],
    channels: pd.DataFrame,
    time_bins: pd.DataFrame,
) -> None:
    """Write one complete decoding subject in a transaction."""

    require_trial_prefix(trials, ("subject_index", "epoch_index", "class", "evidence_group"))
    schemas = ALL_COLUMNS if "repeat" in tables["accuracy"].columns else MEAN_COLUMNS
    for name, table in tables.items():
        require_columns(table, schemas[name], table_name=name)
    with duckdb.connect(str(path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            _write_support(connection, classifier, components, channels, time_bins)
            insert_frame(connection, "trials", trials)
            for name, table in tables.items():
                insert_frame(connection, name, table)
            mark_complete(connection, subject, fingerprint)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise


def _write_support(
    connection: duckdb.DuckDBPyConnection,
    classifier: dict[str, object],
    components: list[list[str]],
    channels: pd.DataFrame,
    time_bins: pd.DataFrame,
) -> None:
    if not table_exists(connection, "classifier"):
        connection.execute(
            """
            CREATE TABLE classifier (
                name VARCHAR,
                parameters JSON,
                classes JSON,
                evidence_shape JSON
            )
            """
        )
        connection.execute(
            "INSERT INTO classifier VALUES (?, ?::JSON, ?::JSON, ?::JSON)",
            [
                classifier["name"],
                json.dumps(jsonable(classifier["parameters"]), sort_keys=True),
                json.dumps(jsonable(classifier["classes"])),
                json.dumps(jsonable(classifier["evidence_shape"])),
            ],
        )
        component_table = pd.DataFrame(
            {
                "component": range(len(components)),
                "classes": [json.dumps(values) for values in components],
            }
        )
        connection.register("_mveeg_rows", component_table)
        connection.execute(
            "CREATE TABLE pattern_components AS "
            "SELECT component, classes::JSON AS classes FROM _mveeg_rows"
        )
        connection.unregister("_mveeg_rows")
        insert_frame(connection, "channels", channels)
        insert_frame(connection, "time_bins", time_bins)
        return
    saved = connection.execute(
        "SELECT name, parameters::VARCHAR, classes::VARCHAR, evidence_shape::VARCHAR "
        "FROM classifier"
    ).fetchone()
    expected = (
        classifier["name"],
        json.dumps(jsonable(classifier["parameters"]), sort_keys=True, separators=(",", ":")),
        json.dumps(jsonable(classifier["classes"]), separators=(",", ":")),
        json.dumps(jsonable(classifier["evidence_shape"]), separators=(",", ":")),
    )
    normalized = (
        saved[0],
        *[
            json.dumps(json.loads(value), sort_keys=True, separators=(",", ":"))
            for value in saved[1:]
        ],
    )
    if normalized != expected:
        raise RuntimeError("Classifier geometry differs between subjects.")
    observed_components = connection.execute(
        "SELECT component, classes::VARCHAR FROM pattern_components ORDER BY component"
    ).fetchall()
    expected_components = [
        (index, json.dumps(values, separators=(",", ":")))
        for index, values in enumerate(components)
    ]
    normalized_components = [
        (index, json.dumps(json.loads(values), separators=(",", ":")))
        for index, values in observed_components
    ]
    if normalized_components != expected_components:
        raise RuntimeError("Pattern components differ between subjects.")
    require_equal_frame(connection, "channels", channels)
    require_equal_frame(connection, "time_bins", time_bins)


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
