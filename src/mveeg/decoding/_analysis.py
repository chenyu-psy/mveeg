"""Repeated cross-validated decoding for one prepared EEG subject."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from ._aggregation import combine_all, combine_mean, summarize_repeat
from ._models import (
    haufe_patterns,
    make_classifier,
    native_evidence,
    pattern_components,
    target_evidence,
)
from ._prepare import average_training_trials, sample_balanced


@dataclass
class SubjectDecoding:
    """All public result rows produced for one subject."""

    tables: dict[str, pd.DataFrame]
    classifier: dict[str, object]
    pattern_components: list[list[str]]


@dataclass
class _RepeatDecoding:
    """Raw fold-level rows and classifier geometry for one CV repeat."""

    repeat: int
    tables: dict[str, pd.DataFrame]
    classes: list[str]
    evidence_shape: list[int]
    pattern_components: list[list[str]]


def decode_subject(
    *,
    subject: str,
    data: np.ndarray,
    class_labels: np.ndarray,
    evidence_labels: np.ndarray,
    generalization_labels: np.ndarray,
    condition_values: np.ndarray,
    trial_ids: np.ndarray,
    times: np.ndarray,
    class_order: list[str],
    classifier: str,
    classifier_parameters: dict[str, object],
    folds: int,
    repeats: int,
    trial_averaging: int,
    permutations: int,
    seed: int,
    output: str,
    n_jobs: int = 1,
    progress: bool = True,
) -> SubjectDecoding:
    """Decode one subject and return rows at the requested CV granularity."""

    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer.")
    if not isinstance(progress, bool):
        raise TypeError("progress must be bool.")
    labels = np.asarray(class_labels, dtype=object)
    evidence = np.asarray(evidence_labels, dtype=object)
    generalization = np.asarray(generalization_labels, dtype=object)
    conditions = np.asarray(condition_values, dtype=object)
    if any(len(values) != len(labels) for values in (evidence, generalization, conditions)):
        raise ValueError("Decoding trial labels must match the prepared data rows.")
    training_mask = pd.notna(labels)
    evidence_mask = pd.notna(evidence)
    generalization_mask = pd.notna(generalization)
    training_rows = np.flatnonzero(training_mask)
    training_labels = labels[training_mask]
    counts = {label: int(np.sum(training_labels == label)) for label in class_order}
    if any(count < folds for count in counts.values()):
        raise ValueError(f"Every class needs at least folds={folds} trials; counts were {counts}.")

    shared = {
        "subject": subject,
        "data": data,
        "labels": labels,
        "evidence_mask": evidence_mask,
        "generalization_labels": generalization,
        "generalization_mask": generalization_mask,
        "condition_values": conditions,
        "generalization_conditions": list(pd.unique(conditions[generalization_mask])),
        "training_mask": training_mask,
        "training_rows": training_rows,
        "training_labels": training_labels,
        "trial_ids": trial_ids,
        "times": times,
        "class_order": class_order,
        "classifier": classifier,
        "classifier_parameters": classifier_parameters,
        "folds": folds,
        "trial_averaging": trial_averaging,
        "permutations": permutations,
        "subject_seed": _subject_seed(seed, subject),
    }
    effective_jobs = min(n_jobs, repeats)
    if effective_jobs == 1:
        repeat_results = (
            _decode_repeat(repeat=repeat, **shared) for repeat in range(1, repeats + 1)
        )
    else:
        repeat_results = Parallel(
            n_jobs=effective_jobs,
            prefer="processes",
            return_as="generator_unordered",
        )(delayed(_decode_repeat)(repeat=repeat, **shared) for repeat in range(1, repeats + 1))

    all_results: dict[int, dict[str, pd.DataFrame]] = {}
    mean_results: dict[int, dict[str, pd.DataFrame]] = {}
    fitted_classes: list[str] | None = None
    fitted_evidence_shape: list[int] | None = None
    fitted_components: list[list[str]] | None = None
    for result in tqdm(
        repeat_results,
        total=repeats,
        desc=f"Decoding sub-{subject}",
        unit="repeat",
        disable=not progress,
    ):
        fitted_classes, fitted_evidence_shape, fitted_components = _check_geometry(
            fitted_classes,
            fitted_evidence_shape,
            fitted_components,
            result.classes,
            result.evidence_shape,
            result.pattern_components,
        )
        if output == "all":
            all_results[result.repeat] = result.tables
        else:
            mean_results[result.repeat] = summarize_repeat(result.tables)

    if fitted_classes is None or fitted_evidence_shape is None or fitted_components is None:
        raise RuntimeError("No decoding model was fitted.")
    if output == "all":
        tables = combine_all(all_results)
    else:
        tables = combine_mean(mean_results)
    return SubjectDecoding(
        tables=tables,
        classifier={
            "name": classifier,
            "parameters": classifier_parameters,
            "classes": fitted_classes,
            "evidence_shape": fitted_evidence_shape,
        },
        pattern_components=fitted_components,
    )


def _decode_repeat(
    *,
    repeat: int,
    subject: str,
    data: np.ndarray,
    labels: np.ndarray,
    evidence_mask: np.ndarray,
    generalization_labels: np.ndarray,
    generalization_mask: np.ndarray,
    condition_values: np.ndarray,
    generalization_conditions: list[object],
    training_mask: np.ndarray,
    training_rows: np.ndarray,
    training_labels: np.ndarray,
    trial_ids: np.ndarray,
    times: np.ndarray,
    class_order: list[str],
    classifier: str,
    classifier_parameters: dict[str, object],
    folds: int,
    trial_averaging: int,
    permutations: int,
    subject_seed: int,
) -> _RepeatDecoding:
    """Run one complete balanced repeated-CV iteration."""

    accuracy_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    pattern_rows: list[dict[str, object]] = []
    generalization_rows: list[dict[str, object]] = []
    fitted_classes: list[str] | None = None
    fitted_evidence_shape: list[int] | None = None
    fitted_components: list[list[str]] | None = None

    repeat_rng = _rng(subject_seed, repeat, 0)
    relative_balanced = sample_balanced(training_labels, class_order, repeat_rng)
    balanced_rows = training_rows[relative_balanced]
    balanced_labels = labels[balanced_rows]
    split_seed = int(_rng(subject_seed, repeat, 1).integers(0, 2**31 - 1))
    splits = list(
        StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=split_seed,
        ).split(balanced_rows, balanced_labels)
    )
    permuted_labels = {
        permutation: _valid_permutation(
            balanced_labels,
            splits,
            class_order,
            trial_averaging,
            _rng(subject_seed, repeat, 10_000 + permutation),
        )
        for permutation in range(1, permutations + 1)
    }

    for fold_index, (train_index, test_index) in enumerate(splits):
        fold = fold_index + 1
        train_rows = balanced_rows[train_index]
        test_rows = balanced_rows[test_index]
        averaged_data, averaged_labels = average_training_trials(
            data[train_rows],
            labels[train_rows],
            class_order=class_order,
            size=trial_averaging,
            rng=_rng(subject_seed, repeat, fold, 0),
        )
        evidence_eligible = evidence_mask & ~training_mask
        evidence_eligible = evidence_eligible.copy()
        evidence_eligible[test_rows] = evidence_mask[test_rows]
        evidence_rows_for_fold = np.flatnonzero(evidence_eligible)
        generalization_eligible = generalization_mask & ~training_mask
        generalization_eligible = generalization_eligible.copy()
        generalization_eligible[test_rows] = generalization_mask[test_rows]
        condition_rows = [
            (
                condition,
                np.flatnonzero(generalization_eligible & (condition_values == condition)),
            )
            for condition in generalization_conditions
        ]
        condition_rows = [(condition, rows) for condition, rows in condition_rows if len(rows)]

        for time_index, time in enumerate(times):
            raw_training = averaged_data[:, :, time_index]
            scaler = StandardScaler()
            scaled_training = scaler.fit_transform(raw_training)
            model = make_classifier(classifier, classifier_parameters)
            model.fit(scaled_training, averaged_labels)

            model_classes = [str(value) for value in model.classes_]
            components = pattern_components(model, classifier)
            evidence_values, evidence_shape = native_evidence(
                model,
                scaler.transform(data[evidence_rows_for_fold, :, time_index]),
            )
            fitted_classes, fitted_evidence_shape, fitted_components = _check_geometry(
                fitted_classes,
                fitted_evidence_shape,
                fitted_components,
                model_classes,
                evidence_shape,
                components,
            )

            scaled_test = scaler.transform(data[test_rows, :, time_index])
            predicted = model.predict(scaled_test)
            actual = labels[test_rows]
            correct = int(np.sum(predicted == actual))
            accuracy_rows.append(
                _accuracy_row(subject, repeat, fold, int(time), 0, correct, len(test_rows))
            )

            matrix = confusion_matrix(actual, predicted, labels=class_order)
            for actual_index, actual_label in enumerate(class_order):
                for predicted_index, predicted_label in enumerate(class_order):
                    confusion_rows.append(
                        {
                            "subject_index": subject,
                            "repeat": repeat,
                            "fold": fold,
                            "time": int(time),
                            "actual": actual_label,
                            "predicted": predicted_label,
                            "count": int(matrix[actual_index, predicted_index]),
                        }
                    )

            for row, value in zip(evidence_rows_for_fold, evidence_values):
                evidence_rows.append(
                    {
                        "subject_index": subject,
                        "epoch_index": int(trial_ids[row]),
                        "repeat": repeat,
                        "fold": fold,
                        "time": int(time),
                        "evidence": _public_evidence(value, evidence_shape),
                    }
                )

            patterns = haufe_patterns(raw_training, scaled_training, model)
            for component, component_values in enumerate(patterns):
                for channel, value in enumerate(component_values):
                    pattern_rows.append(
                        {
                            "subject_index": subject,
                            "repeat": repeat,
                            "fold": fold,
                            "time": int(time),
                            "channel_index": channel,
                            "component": component,
                            "pattern": float(value),
                        }
                    )

            _append_generalization(
                generalization_rows,
                subject=subject,
                repeat=repeat,
                fold=fold,
                train_time=int(time),
                permutation=0,
                model=model,
                scaler=scaler,
                data=data,
                times=times,
                condition_rows=condition_rows,
                actual_labels=generalization_labels,
            )

        for permutation, shuffled in permuted_labels.items():
            permuted_data, permuted_training_labels = average_training_trials(
                data[train_rows],
                shuffled[train_index],
                class_order=class_order,
                size=trial_averaging,
                rng=_rng(subject_seed, repeat, fold, permutation),
            )
            permuted_test_labels = shuffled[test_index]
            for time_index, time in enumerate(times):
                scaler = StandardScaler()
                scaled_training = scaler.fit_transform(permuted_data[:, :, time_index])
                model = make_classifier(classifier, classifier_parameters)
                model.fit(scaled_training, permuted_training_labels)
                test_values = scaler.transform(data[test_rows, :, time_index])
                predicted = model.predict(test_values)
                correct = int(np.sum(predicted == permuted_test_labels))
                accuracy_rows.append(
                    _accuracy_row(
                        subject,
                        repeat,
                        fold,
                        int(time),
                        permutation,
                        correct,
                        len(test_rows),
                    )
                )
                _append_generalization(
                    generalization_rows,
                    subject=subject,
                    repeat=repeat,
                    fold=fold,
                    train_time=int(time),
                    permutation=permutation,
                    model=model,
                    scaler=scaler,
                    data=data,
                    times=times,
                    condition_rows=condition_rows,
                    actual_labels=generalization_labels,
                )

    if fitted_classes is None or fitted_evidence_shape is None or fitted_components is None:
        raise RuntimeError("No decoding model was fitted.")
    tables = {
        "accuracy": pd.DataFrame(accuracy_rows),
        "classifier_evidence": pd.DataFrame(evidence_rows),
        "confusion_matrix": pd.DataFrame(confusion_rows),
        "patterns": pd.DataFrame(pattern_rows),
    }
    if generalization_conditions:
        tables["generalization"] = pd.DataFrame(generalization_rows)
    return _RepeatDecoding(
        repeat=repeat,
        tables=tables,
        classes=fitted_classes,
        evidence_shape=fitted_evidence_shape,
        pattern_components=fitted_components,
    )


def _check_geometry(
    saved_classes: list[str] | None,
    saved_shape: list[int] | None,
    saved_components: list[list[str]] | None,
    classes: list[str],
    shape: list[int],
    components: list[list[str]],
) -> tuple[list[str], list[int], list[list[str]]]:
    if saved_classes is None:
        return classes, shape, components
    if saved_classes != classes or saved_shape != shape or saved_components != components:
        raise RuntimeError("Classifier output geometry changed between CV models.")
    return saved_classes, saved_shape, saved_components


def _accuracy_row(
    subject: str,
    repeat: int,
    fold: int,
    time: int,
    permutation: int,
    n_correct: int,
    n_trials: int,
) -> dict[str, object]:
    return {
        "subject_index": subject,
        "repeat": repeat,
        "fold": fold,
        "time": time,
        "permutation": permutation,
        "accuracy": n_correct / n_trials,
        "n_correct": n_correct,
        "n_trials": n_trials,
    }


def _generalization_row(
    subject: str,
    condition: object,
    repeat: int,
    fold: int,
    train_time: int,
    test_time: int,
    permutation: int,
    n_correct: int,
    n_trials: int,
    target_evidence_value: float,
) -> dict[str, object]:
    row = _accuracy_row(subject, repeat, fold, train_time, permutation, n_correct, n_trials)
    row["condition"] = condition
    row["train_time"] = row.pop("time")
    row["test_time"] = test_time
    row["target_evidence"] = target_evidence_value
    return row


def _append_generalization(
    output: list[dict[str, object]],
    *,
    subject: str,
    repeat: int,
    fold: int,
    train_time: int,
    permutation: int,
    model,
    scaler: StandardScaler,
    data: np.ndarray,
    times: np.ndarray,
    condition_rows: list[tuple[object, np.ndarray]],
    actual_labels: np.ndarray,
) -> None:
    """Evaluate selected conditions at every test time with one fitted model."""

    for condition, rows in condition_rows:
        actual = actual_labels[rows]
        for test_time_index, test_time in enumerate(times):
            test_data = scaler.transform(data[rows, :, test_time_index])
            predicted = model.predict(test_data)
            evidence = target_evidence(model, test_data, actual)
            output.append(
                _generalization_row(
                    subject,
                    condition,
                    repeat,
                    fold,
                    train_time,
                    int(test_time),
                    permutation,
                    int(np.sum(predicted == actual)),
                    len(rows),
                    float(evidence.mean()),
                )
            )


def _public_evidence(value: np.ndarray | np.floating, shape: list[int]) -> float | list[float]:
    array = np.asarray(value, dtype=float)
    return float(array) if shape == [] else array.tolist()


def _subject_seed(seed: int, subject: str) -> int:
    encoded = np.frombuffer(subject.encode("utf-8"), dtype=np.uint8)
    return int(np.random.SeedSequence([seed, *encoded.tolist()]).generate_state(1)[0])


def _rng(seed: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *parts]))


def _valid_permutation(
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    class_order: list[str],
    trial_averaging: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one global label shuffle that every training fold can fit."""

    for _ in range(1000):
        shuffled = rng.permutation(labels)
        if all(
            all(np.sum(shuffled[train] == label) >= trial_averaging for label in class_order)
            for train, _ in splits
        ):
            return shuffled
    raise ValueError(
        "Could not draw a permutation with every class represented in each training fold."
    )
