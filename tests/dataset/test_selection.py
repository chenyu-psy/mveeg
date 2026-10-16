"""Subject exclusion sidecar read contracts."""

import pytest

from mveeg._dataset.selection import read_subject_selection


@pytest.mark.parametrize("contents", [None, "", "subject_index\treason\n"])
def test_missing_or_empty_selection_includes_everyone(tmp_path, contents):
    if contents is not None:
        (tmp_path / "subject_selection.tsv").write_text(contents)
    assert read_subject_selection(tmp_path, ("001",)).empty


@pytest.mark.parametrize(
    "contents",
    [
        "subject_index\tnote\n001\tbad\n",
        "subject_index\treason\n002\tbad\n",
        "subject_index\treason\n001\tbad\nsub-001\tbad\n",
        "subject_index\treason\n001\n",
        "subject_index\treason\n001\tbad\textra\n",
        'subject_index\treason\n001\t"unterminated',
    ],
)
def test_corrupt_selection_is_rejected(tmp_path, contents):
    (tmp_path / "subject_selection.tsv").write_text(contents)
    with pytest.raises(ValueError):
        read_subject_selection(tmp_path, ("001",))
