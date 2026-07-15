"""Shared mechanical result-store contracts."""

import duckdb
import pandas as pd
import pytest

from mveeg._results.store import (
    create_subjects_table,
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
