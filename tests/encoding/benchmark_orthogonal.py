"""Paired synthetic comparison; run with the checkout's src on PYTHONPATH.

Fixed grid, no truth-based tuning. Orthogonal residual generation matches the
new estimand; overlapping residual generation explicitly violates a latent
component interpretation. Both methods are evaluated on the same independently
noisy condition means, with equal condition weights and raw predictor scale.
"""

import numpy as np
import pandas as pd

from mveeg.encoding._metrics import fit_orthogonal, fit_ridge


def run_benchmark(repeats=100):
    rng = np.random.default_rng(271032)
    x = np.linspace(1, 3, 6)
    z = x + 0.25 * np.array([0, 1, -1, 1, -1, 0])
    theory = np.column_stack([np.ones(6), x, z])
    weights = np.ones(6) / 6
    q, _ = np.linalg.qr(theory)
    projection = q @ q.T
    theta = rng.normal(scale=0.25, size=(3, 8))
    theta[0] = 0.1
    theta -= theta.mean(axis=1, keepdims=True)
    extra = rng.normal(scale=0.2, size=(6, 8))
    extra -= extra.mean(axis=1, keepdims=True)
    channel_covariance = 0.6 ** np.abs(np.arange(8)[:, None] - np.arange(8))
    channel_factor = np.linalg.cholesky(channel_covariance)
    records = []
    for residual_kind in ("orthogonal", "overlapping"):
        residual = (np.eye(6) - projection) @ extra if residual_kind == "orthogonal" else extra
        truth = theory @ theta + residual
        # Under misspecification this is a projection-defined target, not the
        # original generative component allocation, which is not identifiable.
        theory_target = projection @ truth
        for count_kind, counts in (("balanced", [6] * 6), ("uneven", [4, 8, 5, 9, 6, 4])):
            labels = np.repeat(np.arange(6), counts)
            indicators = np.eye(6)[labels]
            design = np.column_stack([theory[labels], indicators])
            for noise_scale in (0.2, 0.8, 1.6):
                for repeat in range(repeats):
                    noise = rng.normal(size=(len(labels), 8)) @ channel_factor.T
                    noise -= noise.mean(axis=1, keepdims=True)
                    data = (truth[labels] + noise_scale * noise)[:, :, None]
                    heldout_noise = rng.normal(size=(6, 8)) @ channel_factor.T / np.sqrt(20)
                    heldout_noise -= heldout_noise.mean(axis=1, keepdims=True)
                    heldout = truth + noise_scale * heldout_noise
                    for component_penalty in (0.1, 1.0, 10.0):
                        for residual_penalty in (0.01, 0.1, 1.0):
                            penalties = np.array(
                                [0, component_penalty, component_penalty] + [residual_penalty] * 6
                            )
                            old = fit_ridge(data, design, penalties)[:, :, 0]
                            new = fit_orthogonal(
                                data,
                                indicators,
                                theory,
                                weights,
                                component_penalty,
                                residual_penalty,
                            )[:, :, 0]
                            for method, beta in (("ridge", old), ("orthogonal", new)):
                                fitted_theory = theory @ beta[:3]
                                fitted = fitted_theory + beta[3:]
                                records.append(
                                    {
                                        "residual": residual_kind,
                                        "counts": count_kind,
                                        "noise": noise_scale,
                                        "repeat": repeat,
                                        "component_penalty": component_penalty,
                                        "condition_penalty": residual_penalty,
                                        "method": method,
                                        "theory_mse": np.mean((fitted_theory - theory_target) ** 2),
                                        "heldout_mse": np.mean((fitted - heldout) ** 2),
                                        "generative_beta_mse": np.mean((beta[:3] - theta) ** 2),
                                    }
                                )
    return pd.DataFrame(records)


if __name__ == "__main__":
    results = run_benchmark()
    print(
        results.groupby(
            ["residual", "counts", "noise", "component_penalty", "condition_penalty", "method"]
        )[["theory_mse", "heldout_mse", "generative_beta_mse"]]
        .mean()
        .to_csv()
    )
