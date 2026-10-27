"""Paired actual-CV probe of empirical observed-code references; synthetic EEG only.

Run with this checkout's src on PYTHONPATH. No truth-based parameter tuning.
Reference trials are pooled equally, including never-trained auxiliary evidence;
public scores stay once-only. Participant-level repetitions quantify variability,
not independent fold/trial inference.
"""

import numpy as np
import pandas as pd

from mveeg.encoding._analysis import encode_subject


def run_benchmark(repeats=200):
    rng = np.random.default_rng(271036)
    counts = [7, 11, 8, 5, 1, 3]
    labels = np.repeat(np.array(["A", "B", "C", "D", "P", "Q"], object), counts)
    x = np.repeat([0.0, 1.0, 2.0, 3.0, 0.0, 0.0], counts)
    z = np.repeat([0.0, 1.2, 1.8, 3.0, 0.0, 0.0], counts)
    training = np.array(
        [label if label in ["A", "B", "C", "D"] else None for label in labels], object
    )
    p1 = np.array([1.0, 0.5, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    p2 = np.array([0.8, 0.6, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    p1 -= p1.mean()
    p2 -= p2.mean()
    factor = np.linalg.cholesky(0.6 ** np.abs(np.arange(12)[:, None] - np.arange(12)))
    settings = dict(
        subject="001",
        metadata=pd.DataFrame({"x": x, "z": z}),
        training_labels=training,
        expression_labels=labels,
        trial_ids=np.arange(len(labels)),
        times=np.array([25]),
        condition_order=["A", "B", "C", "D"],
        formula="x+z",
        penalty={"component": 1.0, "condition": 0.1},
        folds=4,
        seed=7,
        progress=False,
        expression_reference={"x": 0},
    )
    records = []
    for signal in (0.0, 0.25, 1.0):
        for repeat in range(repeats):
            noise = rng.normal(size=(len(labels), 12)) @ factor.T
            noise -= noise.mean(axis=1, keepdims=True)
            data = (signal * (x[:, None] * p1 + z[:, None] * p2) + noise)[:, :, None] * 1e-6
            result = encode_subject(data=data, **settings)
            raw = result.tables["pattern_expression"].query("component == 'x'")
            relative = result.tables["referenced_pattern_expression"]
            raw_means = raw.groupby("expression_group").expression.mean()
            relative_means = relative.groupby("expression_group").expression.mean()
            preservation = 0.0
            for _, rows in relative.groupby(["fold", "time"]):
                preservation = max(
                    preservation,
                    float(np.max(np.abs(np.diff(rows.expression) - np.diff(rows.raw_expression)))),
                )
            audit = result.tables["expression_reference_trials"]
            offsets = result.tables["expression_references"].set_index(["fold", "time"])
            coordinate_error = 0.0
            for key, rows in audit.groupby(["fold", "time"]):
                offset = offsets.loc[key].reference_expression
                coordinate_error = max(
                    coordinate_error, abs(float(np.mean(rows.raw_expression - offset)))
                )
            records.append(
                {
                    "signal": signal,
                    "repeat": repeat,
                    "raw_A": raw_means.A,
                    "referenced_A": relative_means.A,
                    "raw_D_minus_A": raw_means.D - raw_means.A,
                    "referenced_D_minus_A": relative_means.D - relative_means.A,
                    "within_fold_difference_error": preservation,
                    "reference_coordinate_error": coordinate_error,
                }
            )
    return pd.DataFrame(records)


if __name__ == "__main__":
    results = run_benchmark()
    print(
        "seed=271036; counts=[7,11,8,5,1,3]; 12 correlated average-referenced channels; four folds"
    )
    for signal, rows in results.groupby("signal"):
        print("signal", signal, "repeats", len(rows))
        for name in ("raw_A", "referenced_A", "raw_D_minus_A", "referenced_D_minus_A"):
            print(
                name,
                "mean",
                rows[name].mean(),
                "SD",
                rows[name].std(ddof=0),
                "MC_SE",
                rows[name].std(ddof=0) / np.sqrt(len(rows)),
            )
        print("max within-fold difference error", rows.within_fold_difference_error.max())
        print("max reference-coordinate error", rows.reference_coordinate_error.max())
