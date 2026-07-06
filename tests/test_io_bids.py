"""Tests for path and result-file helpers exposed through ``mveeg.io``."""

from pathlib import Path

import pandas as pd

from mveeg.io import (
    build_derivative_stem,
    build_subject_label,
    derivative_file_path,
    normalize_subject_id,
)
from mveeg.io.results import (
    init_result_store,
    read_result_tables,
    replace_result_tables,
    resolve_result_file,
    result_config_hash,
    write_result_tables,
)


def test_subject_id_helpers_accept_common_labels():
    """Subject helpers should accept raw, compact, and BIDS-style labels."""
    assert normalize_subject_id("101") == "101"
    assert normalize_subject_id("sub101") == "101"
    assert normalize_subject_id("sub-101") == "101"
    assert build_subject_label("sub-101") == "sub-101"


def test_derivative_path_uses_configured_naming():
    """Derivative paths should keep dataset-specific names configurable."""
    path = derivative_file_path(
        "/data/project",
        "sub-101",
        "memory",
        "epo",
        ".fif",
        derivative_label="clean",
    )

    expected = Path(
        "/data/project/derivatives/sub-101/eeg/sub-101_memory_desc-clean_epo.fif"
    )
    assert path == expected
    assert build_derivative_stem("101", "memory") == "sub-101_memory_desc-preprocessed"


def test_result_file_defaults_to_duckdb_suffix(tmp_path):
    """Result files without a suffix should be stored as DuckDB databases."""

    assert resolve_result_file(tmp_path / "decode") == tmp_path / "decode.duckdb"


def test_result_file_rejects_other_suffixes(tmp_path):
    """mveeg result files intentionally support only DuckDB."""

    try:
        resolve_result_file(tmp_path / "decode.csv")
    except ValueError as err:
        assert ".duckdb" in str(err)
    else:
        raise AssertionError("Expected non-DuckDB suffix to raise ValueError.")


def test_write_and_read_result_tables(tmp_path):
    """Multiple DataFrames should round-trip through one DuckDB result file."""

    path = write_result_tables(
        tmp_path / "result",
        {
            "accuracy": pd.DataFrame({"subject": ["001"], "accuracy": [0.75]}),
            "run_summary": pd.DataFrame({"name": ["run_a"]}),
        },
    )

    tables = read_result_tables(path, required_tables=["accuracy", "run_summary"])

    assert path == tmp_path / "result.duckdb"
    assert tables["accuracy"]["accuracy"].tolist() == [0.75]
    assert tables["run_summary"]["name"].tolist() == ["run_a"]


def test_read_result_tables_hides_internal_tables(tmp_path):
    """Internal store tables should not appear in the public result dict."""

    path = write_result_tables(
        tmp_path / "result",
        {
            "accuracy": pd.DataFrame({"subject": ["001"], "accuracy": [0.75]}),
            "_mveeg_subjects": pd.DataFrame({"subject": ["001"], "status": ["complete"]}),
        },
    )

    tables = read_result_tables(path)

    assert sorted(tables) == ["accuracy"]


def test_read_result_tables_rejects_missing_required_table(tmp_path):
    """Incomplete DuckDB result files should fail clearly."""

    path = write_result_tables(tmp_path / "result", {"accuracy": pd.DataFrame({"x": [1]})})

    try:
        read_result_tables(path, required_tables=["accuracy", "trials"])
    except ValueError as err:
        assert "trials" in str(err)
    else:
        raise AssertionError("Expected missing table to raise ValueError.")


def test_result_store_rejects_config_mismatch(tmp_path):
    """A DuckDB result store should not accept incompatible run settings."""

    file = init_result_store(
        tmp_path / "decode",
        analysis_type="decoding",
        config_hash=result_config_hash({"a": 1}),
        run_name="decode",
    )
    replace_result_tables(file, {"accuracy": pd.DataFrame({"subject": ["001"]})})

    try:
        init_result_store(
            file,
            analysis_type="decoding",
            config_hash=result_config_hash({"a": 2}),
            run_name="decode",
        )
    except ValueError as err:
        assert "incompatible" in str(err)
    else:
        raise AssertionError("Expected config mismatch to raise ValueError.")
