"""Transactional DuckDB persistence for the public decoding contract."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import duckdb
import pandas as pd


SUBJECT_TABLES = (
    "trials",
    "accuracy",
    "classifier_evidence",
    "confusion_matrix",
    "patterns",
    "generalization",
)


def result_path(file: str | Path) -> Path:
    """Normalize the required caller-provided DuckDB path."""

    path = Path(file).expanduser()
    if path.suffix == "":
        path = path.with_suffix(".duckdb")
    if path.suffix != ".duckdb":
        raise ValueError("Decoding results must use the .duckdb suffix.")
    return path.resolve()


def read_analysis(path: Path) -> dict[str, object] | None:
    """Read the single analysis row when a decoding store already exists."""

    if not path.exists():
        return None
    with duckdb.connect(str(path), read_only=True) as connection:
        if not _table_exists(connection, "analysis"):
            raise ValueError("Existing DuckDB file is not an mveeg decoding result.")
        columns = [row[1] for row in connection.execute("PRAGMA table_info('analysis')").fetchall()]
        values = connection.execute("SELECT * FROM analysis").fetchall()
        if len(values) != 1:
            raise ValueError("The decoding analysis table must contain exactly one row.")
    return dict(zip(columns, values[0]))


def initialize_store(
    path: Path,
    *,
    version: str,
    output: str,
    generalization: dict[str, list[object]] | None,
    seed: int,
    config: dict[str, object],
    fingerprint: str,
    reset: bool,
) -> None:
    """Create or atomically reset the public analysis and subject tables."""

    path.parent.mkdir(parents=True, exist_ok=True)
    now = _now()
    with duckdb.connect(str(path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            if reset:
                for table in _table_names(connection):
                    connection.execute(f'DROP TABLE "{table}"')
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis (
                    version VARCHAR,
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
                    "INSERT INTO analysis VALUES (?, ?, ?::JSON, ?, ?::JSON, ?, ?, ?)",
                    [
                        version,
                        output,
                        (
                            None
                            if generalization is None
                            else json.dumps(generalization, sort_keys=True, default=str)
                        ),
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
    """Return saved status and input fingerprint by subject."""

    if not path.exists():
        return {}
    with duckdb.connect(str(path), read_only=True) as connection:
        if not _table_exists(connection, "subjects"):
            return {}
        rows = connection.execute(
            "SELECT subject, status, fingerprint FROM subjects"
        ).fetchall()
    return {str(subject): (str(status), fingerprint) for subject, status, fingerprint in rows}


def mark_pending(path: Path, subject: str, fingerprint: str) -> None:
    """Invalidate stale subject rows and mark the new transaction pending."""

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
    """Record a subject failure without hiding it from the public store."""

    with duckdb.connect(str(path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute("DELETE FROM subjects WHERE subject = ?", [subject])
            connection.execute(
                "INSERT INTO subjects VALUES (?, 'failed', ?, ?, ?)",
                [subject, fingerprint, reason, _now()],
            )
            connection.execute("UPDATE analysis SET updated = ?", [_now()])
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise


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
    """Write all rows for one completed subject in one transaction."""

    with duckdb.connect(str(path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            _write_support(connection, classifier, components, channels, time_bins)
            _insert_frame(connection, "trials", trials)
            for name, table in tables.items():
                _insert_frame(connection, name, table)
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
    """Return the number of completed subjects."""

    with duckdb.connect(str(path), read_only=True) as connection:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM subjects WHERE status = 'complete'"
            ).fetchone()[0]
        )


def _write_support(
    connection,
    classifier: dict[str, object],
    components: list[list[str]],
    channels: pd.DataFrame,
    time_bins: pd.DataFrame,
) -> None:
    if not _table_exists(connection, "classifier"):
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
                json.dumps(classifier["parameters"], sort_keys=True, default=str),
                json.dumps(classifier["classes"]),
                json.dumps(classifier["evidence_shape"]),
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
        _insert_frame(connection, "channels", channels)
        _insert_frame(connection, "time_bins", time_bins)
        return

    saved = connection.execute(
        "SELECT name, parameters::VARCHAR, classes::VARCHAR, evidence_shape::VARCHAR "
        "FROM classifier"
    ).fetchone()
    expected = (
        classifier["name"],
        json.dumps(classifier["parameters"], sort_keys=True, default=str, separators=(",", ":")),
        json.dumps(classifier["classes"], separators=(",", ":")),
        json.dumps(classifier["evidence_shape"], separators=(",", ":")),
    )
    normalized_saved = (saved[0], *[json.dumps(json.loads(value), sort_keys=True, separators=(",", ":")) for value in saved[1:]])
    if normalized_saved != expected:
        raise RuntimeError("Classifier geometry differs from the existing result subjects.")
    saved_components = connection.execute(
        "SELECT component, classes::VARCHAR FROM pattern_components ORDER BY component"
    ).fetchall()
    expected_components = [
        (component, json.dumps(values, separators=(",", ":")))
        for component, values in enumerate(components)
    ]
    normalized_components = [
        (component, json.dumps(json.loads(values), separators=(",", ":")))
        for component, values in saved_components
    ]
    if normalized_components != expected_components:
        raise RuntimeError("Pattern components differ between subjects.")
    _require_equal_frame(connection, "channels", channels)
    _require_equal_frame(connection, "time_bins", time_bins)


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
