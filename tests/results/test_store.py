"""Shared mechanical result-store contracts."""

import duckdb
import pandas as pd
import pytest

from mveeg._results.store import (
    create_subjects_table,
    delete_subject,
    insert_frame,
    mark_failed,
    mark_pending,
    require_columns,
    subject_state,
)


def test_subject_state_transitions_delete_stale_rows_atomically(tmp_path):
    path = tmp_path / "result.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE analysis (updated VARCHAR)")
        connection.execute("INSERT INTO analysis VALUES ('created')")
        create_subjects_table(connection)
        connection.execute("CREATE TABLE scores (subject_index VARCHAR, score DOUBLE)")
        connection.execute("INSERT INTO scores VALUES ('001', 1.0)")

    mark_pending(path, "001", "input-1", subject_tables=("scores",))
    assert subject_state(path) == {"001": ("pending", "input-1")}
    with duckdb.connect(str(path), read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 0

    mark_failed(path, "001", "input-1", "synthetic failure")
    assert subject_state(path) == {"001": ("failed", "input-1")}


def test_fixed_result_columns_require_exact_order():
    frame = pd.DataFrame({"subject_index": ["001"], "score": [1.0]})
    require_columns(frame, ("subject_index", "score"), table_name="scores")
    with pytest.raises(ValueError, match="invalid columns"):
        require_columns(frame, ("score", "subject_index"), table_name="scores")


def test_insert_frame_matches_existing_columns_by_name():
    connection = duckdb.connect(":memory:")
    try:
        insert_frame(
            connection,
            "scores",
            pd.DataFrame({"subject_index": ["001"], "score": [1.0]}),
        )
        insert_frame(
            connection,
            "scores",
            pd.DataFrame({"score": [2.0], "subject_index": ["002"]}),
        )

        rows = connection.execute(
            "SELECT subject_index, score FROM scores ORDER BY subject_index"
        ).fetchall()
        assert rows == [("001", 1.0), ("002", 2.0)]
    finally:
        connection.close()


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"subject_index": ["002"]}),
        pd.DataFrame(
            {
                "subject_index": ["002"],
                "score": [2.0],
                "condition": ["observed"],
            }
        ),
    ],
)
def test_insert_frame_rejects_different_columns(frame):
    connection = duckdb.connect(":memory:")
    try:
        insert_frame(
            connection,
            "scores",
            pd.DataFrame({"subject_index": ["001"], "score": [1.0]}),
        )

        with pytest.raises(ValueError, match="columns changed between subjects"):
            insert_frame(connection, "scores", frame)
    finally:
        connection.close()


@pytest.mark.parametrize("broken", [False, True])
def test_subject_deletion_is_atomic_and_keeps_other_subjects(tmp_path, broken):
    path = tmp_path / "result.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE analysis (updated VARCHAR)")
        connection.execute("INSERT INTO analysis VALUES ('created')")
        create_subjects_table(connection)
        connection.execute("INSERT INTO subjects VALUES ('001', 'complete', 'f', NULL, 'created')")
        connection.execute("INSERT INTO subjects VALUES ('002', 'complete', 'f', NULL, 'created')")
        connection.execute("CREATE TABLE scores (subject_index VARCHAR, score DOUBLE)")
        connection.execute("INSERT INTO scores VALUES ('001', 1.0), ('002', 2.0)")
        connection.execute("CREATE TABLE broken (score DOUBLE)")
    if broken:
        with pytest.raises(duckdb.BinderException):
            delete_subject(path, "001", subject_tables=("scores", "broken"))
    else:
        delete_subject(path, "001", subject_tables=("scores", "absent"))
    expected = ["001", "002"] if broken else ["002"]
    assert sorted(subject_state(path)) == expected
    with duckdb.connect(str(path), read_only=True) as connection:
        assert [
            row[0]
            for row in connection.execute(
                "SELECT subject_index FROM scores ORDER BY subject_index"
            ).fetchall()
        ] == expected
        if broken:
            assert connection.execute("SELECT updated FROM analysis").fetchone() == ("created",)
