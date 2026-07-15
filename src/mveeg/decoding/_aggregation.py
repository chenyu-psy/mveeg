"""Formatting of decoding repeat/fold rows for ``all`` and ``mean`` output."""

from __future__ import annotations

import numpy as np
import pandas as pd


def combine_all(
    repeat_results: dict[int, dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """Concatenate repeat-level fold rows in deterministic order."""

    ordered = [repeat_results[repeat] for repeat in sorted(repeat_results)]
    tables = {
        name: pd.concat([result[name] for result in ordered], ignore_index=True)
        for name in ordered[0]
    }
    return {
        "accuracy": tables["accuracy"][
            [
                "subject_index",
                "repeat",
                "fold",
                "time",
                "permutation",
                "accuracy",
                "n_correct",
                "n_trials",
            ]
        ],
        "classifier_evidence": tables["classifier_evidence"][
            [
                "subject_index",
                "epoch_index",
                "repeat",
                "fold",
                "time",
                "evidence",
            ]
        ],
        "confusion_matrix": tables["confusion_matrix"][
            [
                "subject_index",
                "repeat",
                "fold",
                "time",
                "actual",
                "predicted",
                "count",
            ]
        ],
        "patterns": tables["patterns"][
            [
                "subject_index",
                "repeat",
                "fold",
                "time",
                "channel_index",
                "component",
                "pattern",
            ]
        ],
        **(
            {
                "generalization": tables["generalization"]
                .sort_values(
                    [
                        "subject_index",
                        "condition",
                        "repeat",
                        "fold",
                        "train_time",
                        "test_time",
                        "permutation",
                    ],
                    kind="stable",
                )[
                    [
                        "subject_index",
                        "condition",
                        "repeat",
                        "fold",
                        "train_time",
                        "test_time",
                        "permutation",
                        "accuracy",
                        "n_correct",
                        "n_trials",
                    ]
                ]
                .reset_index(drop=True)
            }
            if "generalization" in tables
            else {}
        ),
    }


def summarize_repeat(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Collapse folds within one repeat before cross-repeat averaging."""

    confusion_keys = ["subject_index", "repeat", "time", "actual", "predicted"]
    pattern_keys = ["subject_index", "repeat", "time", "channel_index", "component"]
    summarized = {
        "accuracy": _summarize_repeat_accuracy(tables["accuracy"], ["time"]),
        "classifier_evidence": _summarize_repeat_evidence(tables["classifier_evidence"]),
        "confusion_matrix": (
            tables["confusion_matrix"]
            .groupby(confusion_keys, as_index=False, sort=False)["count"]
            .sum()
        ),
        "patterns": (
            tables["patterns"]
            .groupby(pattern_keys, as_index=False, sort=False)
            .agg(pattern_sum=("pattern", "sum"), n_models=("pattern", "count"))
        ),
    }
    if "generalization" in tables:
        summarized["generalization"] = _summarize_repeat_accuracy(
            tables["generalization"], ["condition", "train_time", "test_time"]
        )
    return summarized


def combine_mean(
    repeat_results: dict[int, dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """Combine repeat summaries into the public mean tables."""

    ordered = [repeat_results[repeat] for repeat in sorted(repeat_results)]
    tables = {
        name: pd.concat([result[name] for result in ordered], ignore_index=True)
        for name in ordered[0]
    }
    confusion_keys = ["subject_index", "time", "actual", "predicted"]
    pattern_keys = ["subject_index", "time", "channel_index", "component"]
    patterns = (
        tables["patterns"]
        .groupby(pattern_keys, as_index=False, sort=True)[["pattern_sum", "n_models"]]
        .sum()
    )
    patterns["pattern"] = patterns["pattern_sum"] / patterns["n_models"]
    patterns.loc[patterns["n_models"].eq(0), "pattern"] = np.nan
    formatted = {
        "accuracy": _combine_mean_accuracy(tables["accuracy"], ["time"]),
        "classifier_evidence": _combine_mean_evidence(tables["classifier_evidence"]),
        "confusion_matrix": (
            tables["confusion_matrix"]
            .groupby(confusion_keys, as_index=False, sort=True)["count"]
            .sum()
        ),
        "patterns": patterns[[*pattern_keys, "pattern"]],
    }
    if "generalization" in tables:
        formatted["generalization"] = _combine_mean_accuracy(
            tables["generalization"], ["condition", "train_time", "test_time"]
        )
    return formatted


def _summarize_repeat_accuracy(
    table: pd.DataFrame,
    time_columns: list[str],
) -> pd.DataFrame:
    keys = ["subject_index", "repeat", *time_columns, "permutation"]
    summarized = table.groupby(keys, as_index=False, sort=False)[["n_correct", "n_trials"]].sum()
    summarized["accuracy"] = summarized["n_correct"] / summarized["n_trials"]
    return summarized[[*keys, "accuracy", "n_correct", "n_trials"]]


def _summarize_repeat_evidence(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["subject_index", "repeat", "epoch_index", "time"]
    for key, group in table.groupby(keys, sort=False):
        values = [np.asarray(value, dtype=float) for value in group["evidence"]]
        total = np.sum(np.stack(values), axis=0)
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "evidence_sum": float(total) if total.ndim == 0 else total.tolist(),
                "n_models": len(values),
            }
        )
    return pd.DataFrame(rows, columns=[*keys, "evidence_sum", "n_models"])


def _combine_mean_accuracy(table: pd.DataFrame, time_columns: list[str]) -> pd.DataFrame:
    keys = ["subject_index", *time_columns, "permutation"]
    counts = table.groupby(keys, as_index=False, sort=True)[["n_correct", "n_trials"]].sum()
    counts["accuracy"] = counts["n_correct"] / counts["n_trials"]
    return counts[[*keys, "accuracy", "n_correct", "n_trials"]]


def _combine_mean_evidence(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["subject_index", "epoch_index", "time"]
    table = table.sort_values([*keys, "repeat"], kind="stable")
    for key, group in table.groupby(keys, sort=True):
        totals = [np.asarray(value, dtype=float) for value in group["evidence_sum"]]
        total = np.sum(np.stack(totals), axis=0)
        n_models = int(group["n_models"].sum())
        mean = total / n_models
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "evidence": float(mean) if mean.ndim == 0 else mean.tolist(),
                "n_models": n_models,
            }
        )
    return pd.DataFrame(rows, columns=[*keys, "evidence", "n_models"])
