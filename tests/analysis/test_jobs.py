import numpy as np
import pytest
from joblib import effective_n_jobs

from mveeg._analysis.jobs import resolve_n_jobs


def test_resolve_n_jobs_follows_joblib_conventions():
    assert resolve_n_jobs(2) == 2
    assert resolve_n_jobs(np.int64(3)) == 3
    assert resolve_n_jobs(-1) == effective_n_jobs(-1)
    for invalid in (0, 1.5, True, "2"):
        with pytest.raises(ValueError, match="nonzero integer"):
            resolve_n_jobs(invalid)
