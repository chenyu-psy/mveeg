"""Core decoding routines for subject-level EEG analyses."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .config import ConditionConfig, DatasetConfig, DecodeParamConfig, DecodingConfig, ModelConfig, TrialFilterConfig
from .models import binary_pattern_sign, build_classifier, compute_haufe_pattern, get_binary_weights
from .prepare import make_balanced_trial_bins, sample_balanced_indices, training_row_mask


def _worker_mp_context():
    """Return a multiprocessing context that is safe for the current platform.

    Returns
    -------
    multiprocessing.context.BaseContext
        ``spawn`` on macOS to avoid CoreFoundation fork warnings in script
        sessions, otherwise ``fork`` for Linux-style behavior.
    """

    if sys.platform == "darwin":
        return mp.get_context("spawn")
    return mp.get_context("fork")


def run_subject_decoding(
    data: np.ndarray,
    labels: np.ndarray,
    cfg: DecodingConfig,
    progress_bar=None,
) -> dict[str, np.ndarray | list[str] | int]:
    """Run repeated cross-validated decoding for one subject.

    Parameters
    ----------
    data : np.ndarray
        Window-averaged decoding data with shape
        ``(n_trials, n_channels, n_times)``.
    labels : np.ndarray
        Training labels for each trial.
    cfg : DecodingConfig
        Decoding settings for the current run.
    progress_bar : object | None
        Optional tqdm progress bar updated after each repeat.

    Returns
    -------
    dict[str, np.ndarray | list[str] | int]
        Subject-level decoding outputs, including one cross-validated accuracy
        summary per repeat.
    """

    label_order = cfg.train_label_order()
    train_mask = training_row_mask(labels, label_order)
    train_data = data[train_mask]
    train_labels = labels[train_mask]

    if len(train_labels) == 0:
        raise ValueError("No trials matched the configured training labels after filtering.")

    cfg_state = cfg_to_state(cfg)
    rng = np.random.default_rng(cfg.decode.random_state)
    repeat_seeds = rng.integers(0, 1_000_000_000, size=cfg.decode.n_repeats)

    if cfg.decode.n_jobs == 1:
        repeat_results = []
        for repeat_ix, repeat_seed in enumerate(repeat_seeds):
            repeat_results.append(
                run_decoding_repeat(
                    data=train_data,
                    labels=train_labels,
                    cfg=cfg,
                    label_order=label_order,
                    repeat_ix=repeat_ix,
                    repeat_seed=int(repeat_seed),
                )
            )
            if progress_bar is not None:
                progress_bar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=cfg.decode.n_jobs, mp_context=_worker_mp_context()) as pool:
            future_to_repeat = {
                pool.submit(
                    run_decoding_repeat,
                    train_data,
                    train_labels,
                    cfg_state,
                    label_order,
                    repeat_ix,
                    int(repeat_seed),
                ): repeat_ix
                for repeat_ix, repeat_seed in enumerate(repeat_seeds)
            }
            repeat_results = [None] * cfg.decode.n_repeats

            for future in as_completed(future_to_repeat):
                repeat_ix = future_to_repeat[future]
                repeat_results[repeat_ix] = future.result()
                if progress_bar is not None:
                    progress_bar.update(1)

    repeat_acc = [result[0] for result in repeat_results]
    repeat_acc_shuff = [result[1] for result in repeat_results]
    repeat_confusion = [result[2] for result in repeat_results]
    repeat_weights = [result[3] for result in repeat_results]
    repeat_patterns = [result[4] for result in repeat_results]
    repeat_accuracy_rows = [result[5] for result in repeat_results]
    n_binned_trials = repeat_results[0][6]

    return {
        "accuracy": np.mean(repeat_acc, axis=0),
        "perm_accuracy": np.mean(repeat_acc_shuff, axis=0),
        "confusion_matrix": np.mean(repeat_confusion, axis=0),
        "channel_weights": np.mean(repeat_weights, axis=0),
        "channel_patterns": np.mean(repeat_patterns, axis=0),
        "accuracy_by_repeat": pd.concat(repeat_accuracy_rows, ignore_index=True),
        "label_order": label_order,
        "n_input_trials": int(len(labels)),
        "n_training_trials": int(len(train_labels)),
        "n_binned_trials": int(n_binned_trials),
    }


def run_subject_generalization(
    data: np.ndarray,
    labels: np.ndarray,
    cfg: DecodingConfig,
    progress_bar=None,
) -> dict[str, np.ndarray | pd.DataFrame | int]:
    """Run all-window generalization decoding for one subject.

    Parameters
    ----------
    data : np.ndarray
        Window-averaged decoding data with shape
        ``(n_trials, n_channels, n_windows)``.
    labels : np.ndarray
        Training labels for each trial.
    cfg : DecodingConfig
        Decoding settings for the current run.
    progress_bar : object | None
        Optional tqdm progress bar updated after each repeat.

    Returns
    -------
    dict[str, np.ndarray | pd.DataFrame | int]
        Subject-level generalization outputs, including one accuracy matrix
        for every train-window by test-window combination.
    """

    label_order = cfg.train_label_order()
    train_mask = training_row_mask(labels, label_order)
    windowed_data = data[train_mask]
    train_labels = labels[train_mask]

    if len(train_labels) == 0:
        raise ValueError("No trials matched the configured training labels after filtering.")

    cfg_state = cfg_to_state(cfg)
    rng = np.random.default_rng(cfg.decode.random_state)
    repeat_seeds = rng.integers(0, 1_000_000_000, size=cfg.decode.n_repeats)

    if cfg.decode.n_jobs == 1:
        repeat_results = []
        for repeat_ix, repeat_seed in enumerate(repeat_seeds):
            repeat_results.append(
                run_generalization_repeat(
                    data=windowed_data,
                    labels=train_labels,
                    cfg=cfg,
                    label_order=label_order,
                    repeat_ix=repeat_ix,
                    repeat_seed=int(repeat_seed),
                )
            )
            if progress_bar is not None:
                progress_bar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=cfg.decode.n_jobs, mp_context=_worker_mp_context()) as pool:
            future_to_repeat = {
                pool.submit(
                    run_generalization_repeat,
                    windowed_data,
                    train_labels,
                    cfg_state,
                    label_order,
                    repeat_ix,
                    int(repeat_seed),
                ): repeat_ix
                for repeat_ix, repeat_seed in enumerate(repeat_seeds)
            }
            repeat_results = [None] * cfg.decode.n_repeats

            for future in as_completed(future_to_repeat):
                repeat_ix = future_to_repeat[future]
                repeat_results[repeat_ix] = future.result()
                if progress_bar is not None:
                    progress_bar.update(1)

    repeat_acc = [result[0] for result in repeat_results]
    repeat_acc_shuff = [result[1] for result in repeat_results]
    repeat_accuracy_rows = [result[2] for result in repeat_results]
    n_binned_trials = repeat_results[0][3]

    return {
        "accuracy": np.mean(repeat_acc, axis=0),
        "perm_accuracy": np.mean(repeat_acc_shuff, axis=0),
        "accuracy_by_repeat": pd.concat(repeat_accuracy_rows, ignore_index=True),
        "label_order": label_order,
        "n_input_trials": int(len(labels)),
        "n_training_trials": int(len(train_labels)),
        "n_binned_trials": int(n_binned_trials),
    }


def run_subject_hyperplane(
    data: np.ndarray,
    labels: np.ndarray,
    group_labels: np.ndarray,
    trial_ids: np.ndarray,
    group_order: list[str],
    cfg: DecodingConfig,
    progress_bar=None,
) -> dict[str, object]:
    """Run hyperplane distance across time for each original trial."""

    cfg_state = cfg_to_state(cfg)
    label_order = cfg.train_label_order()
    rng = np.random.default_rng(cfg.decode.random_state)
    repeat_seeds = rng.integers(0, 1_000_000_000, size=cfg.decode.n_repeats)

    if cfg.decode.n_jobs == 1:
        repeat_dist = []
        for repeat_ix, repeat_seed in enumerate(repeat_seeds):
            repeat_dist.append(
                run_hyperplane_repeat(
                    data=data,
                    labels=labels,
                    group_labels=group_labels,
                    trial_ids=trial_ids,
                    group_order=group_order,
                    label_order=label_order,
                    cfg=cfg,
                    repeat_ix=repeat_ix,
                    repeat_seed=int(repeat_seed),
                )
            )
            if progress_bar is not None:
                progress_bar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=cfg.decode.n_jobs, mp_context=_worker_mp_context()) as pool:
            future_to_repeat = {
                pool.submit(
                    run_hyperplane_repeat,
                    data,
                    labels,
                    group_labels,
                    trial_ids,
                    group_order,
                    label_order,
                    cfg_state,
                    repeat_ix,
                    int(repeat_seed),
                ): repeat_ix
                for repeat_ix, repeat_seed in enumerate(repeat_seeds)
            }
            repeat_dist = [None] * cfg.decode.n_repeats

            for future in as_completed(future_to_repeat):
                repeat_ix = future_to_repeat[future]
                repeat_dist[repeat_ix] = future.result()
                if progress_bar is not None:
                    progress_bar.update(1)

    trial_dist = average_hyperplane_repeats(repeat_dist)

    return {
        "trial_distance": trial_dist,
        "group_order": group_order,
        "label_order": cfg.train_label_order(),
    }


def cfg_to_state(cfg: DecodingConfig) -> dict:
    """Convert a decoding config into a plain dictionary for workers."""

    return cfg.to_dict()


def cfg_from_state(cfg_state: dict) -> DecodingConfig:
    """Rebuild a config from a saved state dictionary."""

    model_state = dict(cfg_state["model"])
    return DecodingConfig(
        dataset=DatasetConfig(**dict(cfg_state["dataset"])),
        conditions=ConditionConfig(**dict(cfg_state["conditions"])),
        filters=TrialFilterConfig(**dict(cfg_state["filters"])),
        decode=DecodeParamConfig(**dict(cfg_state["decode"])),
        model=ModelConfig(
            classifier_spec=model_state["classifier_spec"],
            classifier=model_state["classifier"],
        ),
    )


def run_decoding_repeat(
    data: np.ndarray,
    labels: np.ndarray,
    cfg: DecodingConfig | dict,
    label_order: list[str],
    repeat_ix: int,
    repeat_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int]:
    """Run one decoding repeat and return its summary outputs.

    Parameters
    ----------
    data : np.ndarray
        Trial-by-channel-by-time decoding data for the training pool.
    labels : np.ndarray
        Training labels for the current trial pool.
    cfg : DecodingConfig | dict
        Decoding settings or a plain saved state.
    label_order : list[str]
        Training label order used by the classifier.
    repeat_ix : int
        Repeat index used for bookkeeping.
    repeat_seed : int
        Random seed for the repeat-specific resampling.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int]
        Repeat-level decoding summaries, repeat-level accuracy rows, and the
        mean number of binned training trials.
    """

    if isinstance(cfg, dict):
        cfg = cfg_from_state(cfg)

    classifier = build_classifier(cfg)
    repeat_rng = np.random.default_rng(repeat_seed)
    balanced_idx = sample_balanced_indices(labels, repeat_rng)
    balanced_data = data[balanced_idx]
    balanced_labels = labels[balanced_idx]

    min_trial_count = np.unique(balanced_labels, return_counts=True)[1].min()
    if min_trial_count < cfg.decode.n_splits:
        raise ValueError(
            "Not enough balanced trials for the requested number of folds. "
            f"Minimum trials per condition was {min_trial_count}, but n_splits={cfg.decode.n_splits}."
        )

    splitter = StratifiedKFold(
        n_splits=cfg.decode.n_splits,
        shuffle=True,
        random_state=cfg.decode.random_state + repeat_ix,
    )

    fold_acc = []
    fold_acc_shuff = []
    fold_confusion = []
    fold_weights = []
    fold_patterns = []
    binned_trial_counts = []
    fold_correct_counts = []
    fold_perm_correct_counts = []
    fold_test_counts = []

    for _, (train_idx, test_idx) in enumerate(splitter.split(balanced_data, balanced_labels)):
        X_train_single = balanced_data[train_idx]
        X_test = balanced_data[test_idx]
        y_train = balanced_labels[train_idx]
        y_test = balanced_labels[test_idx]

        X_train, y_train_binned = make_balanced_trial_bins(
            data=X_train_single,
            labels=y_train,
            trial_bin_size=cfg.decode.trial_bin_size,
            rng=repeat_rng,
        )
        binned_trial_counts.append(len(y_train_binned))

        acc, acc_shuff, conf_mat, weights, patterns, correct_counts, perm_correct_counts, n_test = decode_one_fold(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train_binned,
            y_test=y_test,
            classifier=classifier,
            label_order=label_order,
            rng=repeat_rng,
        )
        fold_acc.append(acc)
        fold_acc_shuff.append(acc_shuff)
        fold_confusion.append(conf_mat)
        fold_weights.append(weights)
        fold_patterns.append(patterns)
        fold_correct_counts.append(correct_counts)
        fold_perm_correct_counts.append(perm_correct_counts)
        fold_test_counts.append(n_test)

    repeat_n_correct = np.sum(fold_correct_counts, axis=0)
    repeat_perm_n_correct = np.sum(fold_perm_correct_counts, axis=0)
    repeat_n_test = int(np.sum(fold_test_counts))
    chance_level = 1.0 / len(label_order)

    repeat_accuracy_rows = []
    for time_ix in range(len(repeat_n_correct)):
        repeat_accuracy_rows.append(
            {
                "cv_repeat": int(repeat_ix),
                "data_type": "real",
                "perm_id": 0,
                "time_ix": int(time_ix),
                "n_correct": int(repeat_n_correct[time_ix]),
                "n_test_trials": repeat_n_test,
                "accuracy": float(repeat_n_correct[time_ix] / repeat_n_test),
                "balanced_accuracy": float(repeat_n_correct[time_ix] / repeat_n_test),
                "chance_level": float(chance_level),
            }
        )
        repeat_accuracy_rows.append(
            {
                "cv_repeat": int(repeat_ix),
                "data_type": "perm",
                "perm_id": int(repeat_ix + 1),
                "time_ix": int(time_ix),
                "n_correct": int(repeat_perm_n_correct[time_ix]),
                "n_test_trials": repeat_n_test,
                "accuracy": float(repeat_perm_n_correct[time_ix] / repeat_n_test),
                "balanced_accuracy": float(repeat_perm_n_correct[time_ix] / repeat_n_test),
                "chance_level": float(chance_level),
            }
        )

    return (
        repeat_n_correct / repeat_n_test,
        repeat_perm_n_correct / repeat_n_test,
        np.mean(fold_confusion, axis=0),
        np.mean(fold_weights, axis=0),
        np.mean(fold_patterns, axis=0),
        pd.DataFrame(repeat_accuracy_rows),
        int(np.mean(binned_trial_counts)),
    )


def run_generalization_repeat(
    data: np.ndarray,
    labels: np.ndarray,
    cfg: DecodingConfig | dict,
    label_order: list[str],
    repeat_ix: int,
    repeat_seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    """Run one all-window generalization-decoding repeat.

    Parameters
    ----------
    data : np.ndarray
        Trial-by-channel-by-window decoding data for the current trial pool.
    labels : np.ndarray
        Training labels for the current trial pool.
    cfg : DecodingConfig | dict
        Decoding settings or a plain saved state.
    label_order : list[str]
        Training label order used by the classifier.
    repeat_ix : int
        Repeat index used for bookkeeping.
    repeat_seed : int
        Random seed for the repeat-specific resampling.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame, int]
        Repeat-level real and permutation accuracy matrices, the long repeat
        table, and the mean number of binned training trials.
    """

    if isinstance(cfg, dict):
        cfg = cfg_from_state(cfg)

    classifier = build_classifier(cfg)
    repeat_rng = np.random.default_rng(repeat_seed)
    balanced_idx = sample_balanced_indices(labels, repeat_rng)
    balanced_data = data[balanced_idx]
    balanced_labels = labels[balanced_idx]

    min_trial_count = np.unique(balanced_labels, return_counts=True)[1].min()
    if min_trial_count < cfg.decode.n_splits:
        raise ValueError(
            "Not enough balanced trials for the requested number of folds. "
            f"Minimum trials per condition was {min_trial_count}, but n_splits={cfg.decode.n_splits}."
        )

    splitter = StratifiedKFold(
        n_splits=cfg.decode.n_splits,
        shuffle=True,
        random_state=cfg.decode.random_state + repeat_ix,
    )

    n_windows = balanced_data.shape[2]
    fold_acc = []
    fold_acc_shuff = []
    binned_trial_counts = []
    fold_correct_counts = []
    fold_perm_correct_counts = []
    fold_test_counts = []

    for _, (train_idx, test_idx) in enumerate(splitter.split(balanced_data, balanced_labels)):
        X_train_single = balanced_data[train_idx]
        X_test = balanced_data[test_idx]
        y_train = balanced_labels[train_idx]
        y_test = balanced_labels[test_idx]

        X_train_binned, y_train_binned = make_balanced_trial_bins(
            data=X_train_single,
            labels=y_train,
            trial_bin_size=cfg.decode.trial_bin_size,
            rng=repeat_rng,
        )
        binned_trial_counts.append(len(y_train_binned))

        train_window_acc = np.empty((n_windows, n_windows), dtype=float)
        train_window_acc_shuff = np.empty((n_windows, n_windows), dtype=float)
        train_window_correct = np.empty((n_windows, n_windows), dtype=float)
        train_window_perm_correct = np.empty((n_windows, n_windows), dtype=float)

        for train_time_ix in range(n_windows):
            acc, acc_shuff, correct_counts, perm_correct_counts, n_test = decode_generalization_one_fold(
                X_train=X_train_binned[:, :, train_time_ix],
                X_test=X_test,
                y_train=y_train_binned,
                y_test=y_test,
                classifier=classifier,
                rng=repeat_rng,
            )
            train_window_acc[train_time_ix, :] = acc
            train_window_acc_shuff[train_time_ix, :] = acc_shuff
            train_window_correct[train_time_ix, :] = correct_counts
            train_window_perm_correct[train_time_ix, :] = perm_correct_counts

        fold_acc.append(train_window_acc)
        fold_acc_shuff.append(train_window_acc_shuff)
        fold_correct_counts.append(train_window_correct)
        fold_perm_correct_counts.append(train_window_perm_correct)
        fold_test_counts.append(n_test)

    repeat_n_correct = np.sum(fold_correct_counts, axis=0)
    repeat_perm_n_correct = np.sum(fold_perm_correct_counts, axis=0)
    repeat_n_test = int(np.sum(fold_test_counts))
    chance_level = 1.0 / len(label_order)

    repeat_accuracy_rows = []
    for train_time_ix in range(repeat_n_correct.shape[0]):
        for test_time_ix in range(repeat_n_correct.shape[1]):
            repeat_accuracy_rows.append(
                {
                    "cv_repeat": int(repeat_ix),
                    "data_type": "real",
                    "perm_id": 0,
                    "train_time_ix": int(train_time_ix),
                    "test_time_ix": int(test_time_ix),
                    "n_correct": int(repeat_n_correct[train_time_ix, test_time_ix]),
                    "n_test_trials": repeat_n_test,
                    "accuracy": float(repeat_n_correct[train_time_ix, test_time_ix] / repeat_n_test),
                    "balanced_accuracy": float(repeat_n_correct[train_time_ix, test_time_ix] / repeat_n_test),
                    "chance_level": float(chance_level),
                }
            )
            repeat_accuracy_rows.append(
                {
                    "cv_repeat": int(repeat_ix),
                    "data_type": "perm",
                    "perm_id": int(repeat_ix + 1),
                    "train_time_ix": int(train_time_ix),
                    "test_time_ix": int(test_time_ix),
                    "n_correct": int(repeat_perm_n_correct[train_time_ix, test_time_ix]),
                    "n_test_trials": repeat_n_test,
                    "accuracy": float(repeat_perm_n_correct[train_time_ix, test_time_ix] / repeat_n_test),
                    "balanced_accuracy": float(repeat_perm_n_correct[train_time_ix, test_time_ix] / repeat_n_test),
                    "chance_level": float(chance_level),
                }
            )

    return (
        repeat_n_correct / repeat_n_test,
        repeat_perm_n_correct / repeat_n_test,
        pd.DataFrame(repeat_accuracy_rows),
        int(np.mean(binned_trial_counts)),
    )


def run_hyperplane_repeat(
    data: np.ndarray,
    labels: np.ndarray,
    group_labels: np.ndarray,
    trial_ids: np.ndarray,
    group_order: list[str],
    label_order: list[str],
    cfg: DecodingConfig | dict,
    repeat_ix: int,
    repeat_seed: int,
) -> pd.DataFrame:
    """Run one held-out hyperplane-distance repeat on a balanced trial subset."""

    if isinstance(cfg, dict):
        cfg = cfg_from_state(cfg)

    classifier = build_classifier(cfg)
    repeat_rng = np.random.default_rng(repeat_seed)
    train_mask = training_row_mask(labels, label_order)
    train_pool_idx = np.flatnonzero(train_mask)
    train_pool_labels = labels[train_mask]

    if len(train_pool_labels) == 0:
        raise ValueError("No trials matched the configured training labels for hyperplane output.")

    balanced_idx = sample_balanced_indices(train_pool_labels, repeat_rng)
    balanced_global_idx = train_pool_idx[balanced_idx]

    balanced_data = data[balanced_global_idx]
    balanced_labels = labels[balanced_global_idx]

    min_trial_count = np.unique(balanced_labels, return_counts=True)[1].min()
    if min_trial_count < cfg.decode.n_splits:
        raise ValueError(
            "Not enough balanced trials for the requested number of folds. "
            f"Minimum trials per condition was {min_trial_count}, but n_splits={cfg.decode.n_splits}."
        )

    splitter = StratifiedKFold(
        n_splits=cfg.decode.n_splits,
        shuffle=True,
        random_state=cfg.decode.random_state + repeat_ix,
    )

    trial_rows = []
    for train_idx, _ in splitter.split(balanced_data, balanced_labels):
        X_train_single = balanced_data[train_idx]
        y_train = balanced_labels[train_idx]
        train_global_idx = balanced_global_idx[train_idx]

        X_train, y_train_binned = make_balanced_trial_bins(
            data=X_train_single,
            labels=y_train,
            trial_bin_size=cfg.decode.trial_bin_size,
            rng=repeat_rng,
        )

        trial_distance = hyperplane_one_fold(
            X_train=X_train,
            y_train=y_train_binned,
            X_test=data,
            classifier=classifier,
            label_order=label_order,
        )

        for row_ix, (trial_id, group_label, distance) in enumerate(zip(trial_ids, group_labels, trial_distance)):
            if row_ix in train_global_idx:
                continue
            if group_label in group_order:
                trial_rows.append(
                    {
                        "trial_id": trial_id,
                        "condition": group_label,
                        "distance": distance,
                    }
                )

    if len(trial_rows) == 0:
        return pd.DataFrame(columns=["trial_id", "condition", "distance"])

    repeat_df = pd.DataFrame(trial_rows)
    return (
        repeat_df.groupby(["trial_id", "condition"], as_index=False)["distance"]
        .apply(lambda x: np.mean(np.vstack(x), axis=0))
    )


def decode_one_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classifier,
    label_order: list[str],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Decode one cross-validation fold across time.

    Parameters
    ----------
    X_train : np.ndarray
        Training data with shape ``(n_train, n_channels, n_times)``.
    X_test : np.ndarray
        Held-out test data with shape ``(n_test, n_channels, n_times)``.
    y_train : np.ndarray
        Labels for the binned training data.
    y_test : np.ndarray
        Labels for the held-out test trials.
    classifier : object
        Scikit-learn compatible classifier.
    label_order : list[str]
        Training label order used to align outputs.
    rng : np.random.Generator
        Random generator used for the shuffled baseline.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Fold-level accuracy summaries, channel summaries, correct-count vectors
        for the real and permutation models, and the number of held-out trials.
    """

    n_times = X_train.shape[2]
    n_channels = X_train.shape[1]
    n_labels = len(label_order)
    acc = np.empty(n_times, dtype=float)
    acc_shuff = np.empty(n_times, dtype=float)
    conf_mat = np.empty((n_labels, n_labels, n_times), dtype=float)
    weights = np.empty((n_channels, n_times), dtype=float)
    patterns = np.empty((n_channels, n_times), dtype=float)
    correct_counts = np.empty(n_times, dtype=float)
    perm_correct_counts = np.empty(n_times, dtype=float)

    for time_ix in range(n_times):
        train_tp = X_train[:, :, time_ix]
        test_tp = X_test[:, :, time_ix]

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_tp)
        test_scaled = scaler.transform(test_tp)

        model = clone(classifier)
        model.fit(train_scaled, y_train)

        pred = model.predict(test_scaled)
        correct = pred == y_test

        perm_model = clone(classifier)
        perm_model.fit(train_scaled, rng.permutation(y_train))
        perm_pred = perm_model.predict(test_scaled)
        perm_correct = perm_pred == y_test

        acc[time_ix] = np.mean(correct)
        acc_shuff[time_ix] = np.mean(perm_correct)
        conf_mat[:, :, time_ix] = confusion_matrix(y_test, pred, labels=label_order)
        sign = binary_pattern_sign(model, label_order)
        weights[:, time_ix] = sign * get_binary_weights(model, scaler)
        patterns[:, time_ix] = sign * compute_haufe_pattern(train_tp, train_scaled, model)
        correct_counts[time_ix] = np.sum(correct)
        perm_correct_counts[time_ix] = np.sum(perm_correct)

    return acc, acc_shuff, conf_mat, weights, patterns, correct_counts, perm_correct_counts, int(len(y_test))


def decode_generalization_one_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classifier,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Decode one training window across multiple test windows.

    Parameters
    ----------
    X_train : np.ndarray
        Binned training data with shape ``(n_train, n_channels)`` from one
        training window.
    X_test : np.ndarray
        Held-out test data with shape ``(n_test, n_channels, n_test_times)``.
    y_train : np.ndarray
        Labels for the binned training data.
    y_test : np.ndarray
        Labels for the held-out test trials.
    classifier : object
        Scikit-learn compatible classifier.
    rng : np.random.Generator
        Random generator used for the shuffled baseline.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]
        Real accuracy, permutation accuracy, correct counts, permutation
        correct counts, and the number of held-out trials.
    """

    n_test_times = X_test.shape[2]
    acc = np.empty(n_test_times, dtype=float)
    acc_shuff = np.empty(n_test_times, dtype=float)
    correct_counts = np.empty(n_test_times, dtype=float)
    perm_correct_counts = np.empty(n_test_times, dtype=float)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)

    model = clone(classifier)
    model.fit(train_scaled, y_train)

    perm_model = clone(classifier)
    perm_model.fit(train_scaled, rng.permutation(y_train))

    for test_time_ix in range(n_test_times):
        test_tp = X_test[:, :, test_time_ix]
        test_scaled = scaler.transform(test_tp)

        pred = model.predict(test_scaled)
        perm_pred = perm_model.predict(test_scaled)

        correct = pred == y_test
        perm_correct = perm_pred == y_test

        acc[test_time_ix] = np.mean(correct)
        acc_shuff[test_time_ix] = np.mean(perm_correct)
        correct_counts[test_time_ix] = np.sum(correct)
        perm_correct_counts[test_time_ix] = np.sum(perm_correct)

    return acc, acc_shuff, correct_counts, perm_correct_counts, int(len(y_test))


def hyperplane_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    classifier,
    label_order: list[str],
) -> np.ndarray:
    """Return trial-level decision values across time for one fold."""

    n_times = X_train.shape[2]
    trial_distance = np.empty((len(X_test), n_times), dtype=float)

    for time_ix in range(n_times):
        train_tp = X_train[:, :, time_ix]
        test_tp = X_test[:, :, time_ix]

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_tp)
        test_scaled = scaler.transform(test_tp)

        model = clone(classifier)
        model.fit(train_scaled, y_train)

        distance = model.decision_function(test_scaled)
        if np.ndim(distance) > 1:
            distance = distance[:, 0]

        if len(label_order) == 2 and model.classes_[0] != label_order[0]:
            if list(model.classes_) == label_order[::-1]:
                distance = -distance

        trial_distance[:, time_ix] = distance

    return trial_distance


def average_hyperplane_repeats(repeat_dist: list[pd.DataFrame]) -> pd.DataFrame:
    """Average trial-level hyperplane distances over decoding repeats.

    Parameters
    ----------
    repeat_dist : list[pd.DataFrame]
        One trial-level distance table per repeat. Each table must contain
        `trial_id`, `condition`, and `distance`.

    Returns
    -------
    pd.DataFrame
        One row per trial and condition, with distances averaged across
        decoding repeats.
    """

    if len(repeat_dist) == 0:
        return pd.DataFrame(columns=["trial_id", "condition", "distance"])

    repeat_long = []
    for repeat_ix, repeat_df in enumerate(repeat_dist):
        repeat_copy = repeat_df.copy()
        repeat_copy["repeat"] = repeat_ix
        repeat_long.append(repeat_copy)

    combined = pd.concat(repeat_long, ignore_index=True)
    averaged = (
        combined.groupby(["trial_id", "condition"], as_index=False)["distance"]
        .apply(lambda x: np.mean(np.vstack(x), axis=0))
    )
    return averaged
