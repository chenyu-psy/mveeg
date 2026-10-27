"""Paired metric stress test; synthetic only, fixed seed, no tuning by truth.

Condition-space x/z columns are orthogonal, so their projected vectors directly
yield the joint-projector energies. All metrics see the same complement noise
and independent evaluation means. Oracle targets are reported for diagnosis,
never used to choose shrinkage. Pair products are not inferential samples.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from mveeg.encoding._comparison import _basis, noise_precision


def _prototype_precision(residuals, degrees):
    basis = _basis(residuals.T)
    estimate = LedoitWolf(assume_centered=True).fit(residuals @ basis)
    return basis @ estimate.precision_ @ basis.T * degrees / len(residuals)


def run_benchmark(repeats=200):
    rng = np.random.default_rng(271033)
    channels = 32
    positions = np.arange(channels)
    correlated = 0.8 ** np.abs(positions[:, None] - positions)
    reference = np.eye(channels) - np.ones((channels, channels)) / channels
    cases = {
        "isotropic": np.eye(channels),
        "correlated": correlated,
        "spiked": np.diag([100.0] * 8 + [1.0] * 24),
        "reference_rank_loss": reference @ correlated @ reference,
    }
    codes = np.array([[-1.0, -1], [-1, 1], [1, -1], [1, 1]]) / 4
    evaluation_counts = np.array([4, 8, 5, 7])
    records = []
    for case, covariance in cases.items():
        eigenvalues, vectors = np.linalg.eigh(covariance)
        keep = eigenvalues > eigenvalues[-1] * 1e-12
        factor = vectors[:, keep] * np.sqrt(eigenvalues[keep])
        first = 5 * vectors[:, -1]
        second = vectors[:, np.flatnonzero(keep)[0]]
        oracle = (vectors[:, keep] / eigenvalues[keep]) @ vectors[:, keep].T
        oracle_a, oracle_b = first @ oracle @ first, second @ oracle @ second
        for observations in (12, 36, 132):
            labels = np.arange(observations) % 4
            degrees = observations - 4
            for repeat in range(repeats):
                residuals = rng.normal(size=(observations, keep.sum())) @ factor.T
                for condition in range(4):
                    residuals[labels == condition] -= residuals[labels == condition].mean(axis=0)
                full, rank, shrinkage, condition_number = noise_precision(residuals, degrees)
                prototype = _prototype_precision(residuals, degrees)
                raw = LedoitWolf(assume_centered=True).fit(residuals).precision_
                raw *= degrees / observations
                diagonal = np.diag(degrees / observations / np.mean(residuals**2, axis=0))
                errors = [
                    rng.normal(size=(4, keep.sum()))
                    @ factor.T
                    / np.sqrt(evaluation_counts)[:, None]
                    for _ in range(2)
                ]
                projected = [codes.T @ error for error in errors]
                for method, precision in (
                    ("prototype", prototype),
                    ("full_correlation_shrinkage", full),
                    ("full_raw_shrinkage", raw),
                    ("diagonal", diagonal),
                ):
                    a = (first + projected[0][0]) @ precision @ (first + projected[1][0])
                    b = (second + projected[0][1]) @ precision @ (second + projected[1][1])
                    null_b = projected[0][1] @ precision @ projected[1][1]
                    expected_a, expected_b = first @ precision @ first, second @ precision @ second
                    records.append(
                        {
                            "case": case,
                            "noise_df": degrees,
                            "method": method,
                            "repeat": repeat,
                            "sample_rank": rank,
                            "shrinkage": shrinkage,
                            "condition_number": condition_number,
                            "oracle_a": oracle_a,
                            "oracle_b": oracle_b,
                            "metric_a": expected_a,
                            "metric_b": expected_b,
                            "metric_relative_error": (
                                abs(expected_a - oracle_a) + abs(expected_b - oracle_b)
                            )
                            / (oracle_a + oracle_b),
                            "null_increment": null_b,
                            "nested_increment": b,
                            "positive_increment": b > 0,
                            "ranking_correct": bool((b > a) == (oracle_b > oracle_a)),
                        }
                    )
    return pd.DataFrame(records)


if __name__ == "__main__":
    results = run_benchmark()
    groups = results.groupby(["case", "noise_df", "method"])
    summary = groups[
        [
            "oracle_a",
            "oracle_b",
            "metric_a",
            "metric_b",
            "metric_relative_error",
            "null_increment",
            "nested_increment",
            "positive_increment",
            "ranking_correct",
            "sample_rank",
            "shrinkage",
        ]
    ].mean()
    summary["null_mc_se"] = groups.null_increment.std() / np.sqrt(groups.size())
    summary["metric_b_sd"] = groups.metric_b.std()
    print(summary.to_csv())
