"""Researcher-supplied value-list contract."""

import numpy as np
import pandas as pd
import pytest

from mveeg._values import is_value_list, value_list
from mveeg.decoding._prepare import validate_groups
from mveeg.encoding._prepare import validate_groups as validate_encoding_groups


def test_value_lists_accept_arrays_and_pandas_but_not_strings_or_sets():
    loads = pd.Series([2, 4, 6, 2])
    for value in ([1, 2], (1, 2), range(2), np.array([1, 2]), loads.unique(), pd.Index([1, 2])):
        assert is_value_list(value)
    for value in ("ab", b"ab", {1, 2}, {"a": 1}, np.array(1), np.zeros((2, 2)), 3):
        assert not is_value_list(value)
    assert value_list(np.array([2, 4])) == [2, 4]
    assert all(type(item) is int for item in value_list(np.array([2, 4])))


def test_group_mappings_accept_unique_values():
    loads = pd.DataFrame({"load": [2, 4, 6, 2]})["load"]
    classes, evidence = validate_groups({"low": np.array([2]), "high": loads.unique()[1:]}, None)
    assert classes == {"low": [2], "high": [4, 6]}
    assert evidence == classes
    conditions, _ = validate_encoding_groups({"low": pd.Index([2])}, None)
    assert conditions == {"low": [2]}
    with pytest.raises(TypeError, match="list or array"):
        validate_groups({"low": "2", "high": [4]}, None)
