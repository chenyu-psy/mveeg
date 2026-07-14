"""Trial selection and training-only averaging for decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_DROP_TYPES = ("eog", "eyegaze", "pupil", "misc")
RESERVED_TRIAL_COLUMNS = {"subject", "trial", "class", "evidence_group"}


def validate_groups(
    classes: Mapping[str, Sequence[object]],
    evidence: Mapping[str, Sequence[object]] | None,
) -> tuple[dict[str, list[object]], dict[str, list[object]]]:
    """Validate training and evidence mappings and preserve their order."""

    class_map = _validate_mapping("classes", classes, minimum=2)
    evidence_map = class_map.copy() if evidence is None else _validate_mapping(
        "evidence", evidence, minimum=1
    )
    class_values = [value for values in class_map.values() for value in values]
    evidence_values = {value for values in evidence_map.values() for value in values}
    missing = [value for value in class_values if value not in evidence_values]
    if missing:
        raise ValueError(f"evidence must include every classes value; missing {missing}.")
    return class_map, evidence_map


def validate_generalization(
    classes: dict[str, list[object]],
    generalization: Mapping[str, Sequence[object]] | None,
) -> dict[str, list[object]] | None:
    """Validate independent temporal-generalization conditions."""

    if generalization is None:
        return None
    generalization_map = _validate_mapping("generalization", generalization, minimum=1)
    unknown = [label for label in generalization_map if label not in classes]
    if unknown:
        raise ValueError(f"generalization keys must be classifier classes; unknown {unknown}.")
    return generalization_map


def select_trials(
    metadata: pd.DataFrame,
    *,
    target: str,
    classes: dict[str, list[object]],
    evidence: dict[str, list[object]],
    generalization: dict[str, list[object]] | None,
    qc: str | None,
    keep: Sequence[object],
    exclude: Mapping[str, Sequence[object] | str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select analysis trials and return their independent decoding labels."""

    metadata = _drop_redundant_subject(metadata)
    if target not in metadata.columns:
        raise ValueError(f"target column {target!r} is missing after transform_metadata.")
    collisions = sorted(RESERVED_TRIAL_COLUMNS.intersection(metadata.columns))
    if collisions:
        raise ValueError(f"Metadata uses reserved decoding columns: {collisions}.")
    mask = np.ones(len(metadata), dtype=bool)
    if qc is not None:
        if qc not in metadata.columns:
            raise ValueError(f"QC column {qc!r} is missing; label artifacts or set qc=None.")
        mask &= metadata[qc].isin(tuple(keep)).to_numpy()
    for column, rule in exclude.items():
        if column not in metadata.columns:
            raise ValueError(f"Trial exclusion column {column!r} is missing.")
        if rule == "notna":
            mask &= metadata[column].isna().to_numpy()
        else:
            mask &= ~metadata[column].isin(tuple(rule)).to_numpy()

    values = metadata[target].to_numpy(dtype=object)
    class_labels = _map_values(values, classes)
    evidence_labels = _map_values(values, evidence)
    generalization_labels = (
        np.full(len(values), None, dtype=object)
        if generalization is None
        else _map_values(values, generalization)
    )
    mask &= pd.notna(evidence_labels) | pd.notna(generalization_labels)
    if not np.any(mask):
        raise ValueError("No trials remain after selection and decoding mappings.")

    selected = metadata.loc[mask].reset_index(drop=True).copy()
    class_labels = class_labels[mask]
    evidence_labels = evidence_labels[mask]
    generalization_labels = generalization_labels[mask]
    original_rows = np.flatnonzero(mask)
    return selected, class_labels, evidence_labels, generalization_labels, original_rows


def _drop_redundant_subject(metadata: pd.DataFrame) -> pd.DataFrame:
    """Remove a behavior-level subject alias when it repeats subject_index."""

    if "subject" not in metadata.columns:
        return metadata
    subject = metadata["subject"].astype("string").str.strip()
    subject_index = metadata["subject_index"].astype("string").str.strip()
    same = subject.eq(subject_index).all()
    if not same:
        subject_number = pd.to_numeric(subject, errors="coerce")
        index_number = pd.to_numeric(subject_index, errors="coerce")
        same = subject_number.notna().all() and index_number.notna().all()
        same = same and np.array_equal(subject_number, index_number)
    if same:
        return metadata.drop(columns="subject")
    return metadata


def sample_balanced(
    labels: np.ndarray,
    class_order: Sequence[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the same number of single trials from every class."""

    counts = {label: int(np.sum(labels == label)) for label in class_order}
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Every class needs at least one trial; counts were {counts}.")
    size = min(counts.values())
    chosen = [
        rng.choice(np.flatnonzero(labels == label), size=size, replace=False)
        for label in class_order
    ]
    return rng.permutation(np.concatenate(chosen))


def average_training_trials(
    data: np.ndarray,
    labels: np.ndarray,
    *,
    class_order: Sequence[str],
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebalance and average only the training fold into pseudotrials."""

    counts = {label: int(np.sum(labels == label)) for label in class_order}
    usable = min(counts.values())
    usable -= usable % size
    if usable == 0:
        raise ValueError(
            f"No complete training averages can be formed with trial_averaging={size}; "
            f"fold counts were {counts}."
        )
    blocks = []
    output_labels: list[str] = []
    for label in class_order:
        indices = rng.permutation(np.flatnonzero(labels == label))[:usable]
        block = data[indices]
        if size > 1:
            block = block.reshape(usable // size, size, *data.shape[1:]).mean(axis=1)
        blocks.append(block)
        output_labels.extend([label] * len(block))
    output = np.concatenate(blocks)
    order = rng.permutation(len(output))
    return output[order], np.asarray(output_labels, dtype=object)[order]


def _validate_mapping(
    name: str,
    mapping: Mapping[str, Sequence[object]],
    *,
    minimum: int,
) -> dict[str, list[object]]:
    if not isinstance(mapping, Mapping) or len(mapping) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} named groups.")
    output: dict[str, list[object]] = {}
    seen: list[object] = []
    for label, raw_values in mapping.items():
        if not isinstance(label, str) or label.strip() == "":
            raise ValueError(f"{name} labels must be non-empty strings.")
        if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
            raise TypeError(f"{name}[{label!r}] must be a non-string sequence.")
        values = list(raw_values)
        if not values:
            raise ValueError(f"{name}[{label!r}] cannot be empty.")
        overlap = [
            value
            for index, value in enumerate(values)
            if value in seen or value in values[:index]
        ]
        if overlap:
            raise ValueError(f"{name} values must map to one group; duplicated {overlap}.")
        output[label] = values
        seen.extend(values)
    return output


def _map_values(values: np.ndarray, mapping: dict[str, list[object]]) -> np.ndarray:
    labels = np.full(len(values), None, dtype=object)
    for label, raw_values in mapping.items():
        labels[np.isin(values, raw_values)] = label
    return labels
