"""
Input validation utilities.

These helpers are intended to catch common data-quality problems early,
before expensive model fitting begins.
"""


def check_trial_count(n_trials: int, min_trials: int = 20) -> None:
    """Raise ValueError when the trial count is below the required minimum.

    Parameters
    ----------
    n_trials:
        Number of trials available for the analysis.
    min_trials:
        Minimum acceptable trial count.  Defaults to 20, which is a
        conservative lower bound for most multivariate analyses.

    Raises
    ------
    ValueError
        If ``n_trials < min_trials``.

    Examples
    --------
    >>> check_trial_count(50)          # passes silently
    >>> check_trial_count(10)          # raises ValueError
    """
    if n_trials < min_trials:
        raise ValueError(
            f"Too few trials: got {n_trials}, need at least {min_trials}."
        )
