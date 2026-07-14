"""Researcher-facing decoding configuration and subject orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
import secrets
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .._shared.metadata import (
    load_subject_epochs_and_metadata,
    metadata_transform_spec,
    transform_metadata,
)
from .._shared.time_windows import average_time_windows, build_time_bins
from .._shared.topography import build_topography_coord_table
from ..prep.dataset import fingerprint, fingerprint_files
from .analysis import SubjectDecoding, decode_subject
from .models import classifier_settings
from .prepare import (
    DEFAULT_DROP_TYPES,
    select_trials,
    validate_generalization,
    validate_groups,
)
from .storage import (
    completed_subjects,
    initialize_store,
    mark_failed,
    mark_pending,
    read_analysis,
    result_path,
    subject_state,
    write_subject,
)


def initialize_analysis_state(pipeline) -> None:
    """Attach lazy modeling defaults to a dataset handle."""

    pipeline._metadata_transform = None
    pipeline._metadata_transform_spec = None
    pipeline._trial_selection = {
        "qc": "final_status",
        "keep": ["accepted"],
        "exclude": {},
    }
    pipeline._epoch_preparation = {
        "crop": [0.0, 0.8],
        "time_bin": 50,
        "drop_channel_types": list(DEFAULT_DROP_TYPES),
        "drop_channels": [],
    }
    pipeline._classifier = {
        "name": "logistic_regression",
        "parameters": classifier_settings("logistic_regression", {}),
    }
    pipeline._cv = {
        "folds": 5,
        "repeats": 20,
        "trial_averaging": 5,
        "permutations": 0,
        "seed": None,
    }


def set_metadata_transform(
    pipeline,
    transform: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    name: str,
    version: str,
) -> None:
    """Register a row-preserving analysis metadata transform."""

    pipeline._metadata_transform_spec = metadata_transform_spec(
        transform,
        name=name,
        version=version,
    )
    pipeline._metadata_transform = transform


def set_trial_selection(
    pipeline,
    *,
    qc: str | None,
    keep: Sequence[object],
    exclude: Mapping[str, Sequence[object] | str] | None,
) -> None:
    """Register trial quality and metadata exclusions."""

    if qc is not None and (not isinstance(qc, str) or qc.strip() == ""):
        raise ValueError("qc must be a non-empty column name or None.")
    normalized_exclude: dict[str, tuple[object, ...] | str] = {}
    for column, rule in (exclude or {}).items():
        if not isinstance(column, str) or column.strip() == "":
            raise ValueError("exclude keys must be non-empty metadata column names.")
        if rule == "notna":
            normalized_exclude[column] = "notna"
        elif isinstance(rule, (str, bytes)) or not isinstance(rule, Sequence):
            raise TypeError("exclude rules must be sequences or 'notna'.")
        else:
            normalized_exclude[column] = tuple(rule)
    pipeline._trial_selection = {
        "qc": qc,
        "keep": list(keep),
        "exclude": normalized_exclude,
    }


def set_epoch_preparation(
    pipeline,
    *,
    crop: tuple[float, float] | None,
    time_bin: int,
    drop_channel_types: Sequence[str],
    drop_channels: Sequence[str],
) -> None:
    """Register the major epoch preparation choices used by models."""

    if crop is not None:
        if len(crop) != 2 or not np.all(np.isfinite(crop)) or crop[0] >= crop[1]:
            raise ValueError("crop must contain finite increasing start/end times in seconds.")
        crop_value = [float(crop[0]), float(crop[1])]
    else:
        crop_value = None
    if not isinstance(time_bin, (int, np.integer)) or int(time_bin) < 1:
        raise ValueError("time_bin must be a positive integer in milliseconds.")
    pipeline._epoch_preparation = {
        "crop": crop_value,
        "time_bin": int(time_bin),
        "drop_channel_types": list(drop_channel_types),
        "drop_channels": list(drop_channels),
    }


def set_classifier(pipeline, *, classifier: str, parameters: dict[str, object]) -> None:
    """Register one supported classifier and its native parameters."""

    pipeline._classifier = {
        "name": classifier,
        "parameters": classifier_settings(classifier, parameters),
    }


def set_cv(
    pipeline,
    *,
    folds: int,
    repeats: int,
    trial_averaging: int,
    permutations: int,
    seed: int | None,
) -> None:
    """Register the fixed repeated stratified decoding design."""

    integers = {
        "folds": folds,
        "repeats": repeats,
        "trial_averaging": trial_averaging,
        "permutations": permutations,
    }
    if any(not isinstance(value, (int, np.integer)) for value in integers.values()):
        raise TypeError("CV settings must be integers.")
    if folds < 2 or repeats < 1 or trial_averaging < 1 or permutations < 0:
        raise ValueError(
            "folds must be >=2, repeats/trial_averaging >=1, and permutations >=0."
        )
    if seed is not None and (not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a non-negative integer or None.")
    pipeline._cv = {**integers, "seed": None if seed is None else int(seed)}


def decode(
    pipeline,
    *,
    target: str,
    classes: Mapping[str, Sequence[object]],
    evidence: Mapping[str, Sequence[object]] | None,
    generalization: Mapping[str, Sequence[object]] | None,
    output: str,
    file: str | Path,
    recompute: str,
    n_jobs: int,
    progress: bool,
) -> None:
    """Execute the registered decoding analysis and write its DuckDB file."""

    if not isinstance(target, str) or target.strip() == "":
        raise ValueError("target must be a non-empty post-transform metadata column.")
    if output not in {"mean", "all"}:
        raise ValueError("output must be 'mean' or 'all'.")
    if recompute not in {"never", "changed", "all"}:
        raise ValueError("recompute must be 'never', 'changed', or 'all'.")
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer.")
    if not isinstance(progress, bool):
        raise TypeError("progress must be bool.")
    class_map, evidence_map = validate_groups(classes, evidence)
    generalization_map = validate_generalization(class_map, generalization)
    path = result_path(file)
    existing = read_analysis(path)
    seed = pipeline._cv["seed"]
    if seed is None:
        seed = int(existing["seed"]) if existing is not None else secrets.randbelow(2**63 - 1)
    package_version = _version()

    config = {
        "version": package_version,
        "dataset": {
            "root": str(pipeline.root),
            "task": pipeline.task,
            "stage": pipeline.stage,
        },
        "metadata_transform": pipeline._metadata_transform_spec,
        "trial_selection": pipeline._trial_selection,
        "epoch_preparation": pipeline._epoch_preparation,
        "classifier": pipeline._classifier,
        "cv": {**pipeline._cv, "seed": seed},
        "target": target,
        "classes": class_map,
        "evidence": evidence_map,
        "generalization": generalization_map,
        "output": output,
    }
    analysis_fingerprint = fingerprint(config)
    reset = False
    if existing is not None and existing["fingerprint"] != analysis_fingerprint:
        if recompute != "all":
            raise ValueError(
                "Existing result file uses incompatible analysis settings; "
                "use another file or recompute='all'."
            )
        reset = True
    initialize_store(
        path,
        version=package_version,
        output=output,
        generalization=generalization_map,
        seed=seed,
        config=config,
        fingerprint=analysis_fingerprint,
        reset=reset,
    )

    fingerprints = {
        subject: _input_fingerprint(pipeline, subject)
        for subject in pipeline.subject_indices
    }
    saved = subject_state(path)
    subjects = []
    for subject in pipeline.subject_indices:
        status, previous = saved.get(subject, ("missing", None))
        changed = previous != fingerprints[subject]
        if recompute == "all" or status != "complete" or (recompute == "changed" and changed):
            subjects.append(subject)
        elif recompute == "never" and changed:
            warnings.warn(
                f"Input changed for subject {subject}, but recompute='never' reuses the result.",
                stacklevel=2,
            )

    def work(subject: str):
        prepared = _prepare_subject(
            pipeline,
            subject=subject,
            target=target,
            classes=class_map,
            evidence=evidence_map,
            generalization=generalization_map,
        )
        decoded = decode_subject(
            subject=subject,
            data=prepared["data"],
            class_labels=prepared["class_labels"],
            evidence_labels=prepared["evidence_labels"],
            generalization_labels=prepared["generalization_labels"],
            condition_values=prepared["condition_values"],
            trial_ids=prepared["trials"]["trial"].to_numpy(dtype=int),
            times=prepared["time_bins"]["time"].to_numpy(dtype=int),
            class_order=list(class_map),
            classifier=pipeline._classifier["name"],
            classifier_parameters=pipeline._classifier["parameters"],
            folds=pipeline._cv["folds"],
            repeats=pipeline._cv["repeats"],
            trial_averaging=pipeline._cv["trial_averaging"],
            permutations=pipeline._cv["permutations"],
            seed=seed,
            output=output,
            n_jobs=n_jobs,
            progress=progress,
        )
        return prepared, decoded

    for subject in subjects:
        mark_pending(path, subject, fingerprints[subject])
        _write_results(
            path,
            [(subject, _capture(work, subject))],
            fingerprints,
        )

    if completed_subjects(path) == 0:
        raise RuntimeError(f"No subject completed decoding; inspect the subjects table in {path}.")
    return None


def _prepare_subject(
    pipeline,
    *,
    subject: str,
    target: str,
    classes: dict[str, list[object]],
    evidence: dict[str, list[object]],
    generalization: dict[str, list[object]] | None,
) -> dict[str, object]:
    epochs, metadata = load_subject_epochs_and_metadata(
        pipeline.root,
        subject,
        preload=True,
    )
    metadata = transform_metadata(metadata, pipeline._metadata_transform)
    selected, class_labels, evidence_labels, generalization_labels, rows = select_trials(
        metadata,
        target=target,
        classes=classes,
        evidence=evidence,
        generalization=generalization,
        **pipeline._trial_selection,
    )
    preparation = pipeline._epoch_preparation
    drop = {
        channel
        for channel, channel_type in zip(epochs.ch_names, epochs.get_channel_types())
        if channel_type in preparation["drop_channel_types"]
    }
    drop.update(channel for channel in preparation["drop_channels"] if channel in epochs.ch_names)
    if drop:
        epochs.drop_channels(sorted(drop))
    if preparation["crop"] is not None:
        epochs.crop(
            tmin=preparation["crop"][0],
            tmax=preparation["crop"][1],
            include_tmax=True,
        )
    time_bins, masks = build_time_bins(epochs.times, preparation["time_bin"])
    data = average_time_windows(epochs.get_data(copy=True)[rows], masks)
    channels = build_topography_coord_table(info=epochs.info, channels=list(epochs.ch_names))

    trials = selected.rename(
        columns={"subject_index": "subject", "epoch_index": "trial"}
    )
    trials.insert(2, "class", class_labels)
    trials.insert(3, "evidence_group", evidence_labels)
    leading = ["subject", "trial", "class", "evidence_group"]
    trials = trials[[*leading, *[column for column in trials.columns if column not in leading]]]
    return {
        "data": data,
        "class_labels": class_labels,
        "evidence_labels": evidence_labels,
        "generalization_labels": generalization_labels,
        "condition_values": selected[target].to_numpy(dtype=object),
        "trials": trials,
        "channels": channels,
        "time_bins": time_bins,
    }


def _write_results(path: Path, results, fingerprints: dict[str, str]) -> None:
    for subject, (value, error) in results:
        if error is not None:
            mark_failed(path, subject, fingerprints[subject], str(error))
            continue
        prepared, decoded = value
        try:
            _name_pattern_channels(decoded, prepared["channels"])
            write_subject(
                path,
                subject=subject,
                fingerprint=fingerprints[subject],
                trials=prepared["trials"],
                tables=decoded.tables,
                classifier=decoded.classifier,
                components=decoded.pattern_components,
                channels=prepared["channels"],
                time_bins=prepared["time_bins"],
            )
        except Exception as error:
            mark_failed(path, subject, fingerprints[subject], str(error))


def _name_pattern_channels(decoded: SubjectDecoding, channels: pd.DataFrame) -> None:
    table = decoded.tables["patterns"].copy()
    names = channels["channel"].tolist()
    table["channel"] = table.pop("channel_index").map(dict(enumerate(names)))
    keys = ["subject"]
    if "repeat" in table:
        keys.extend(["repeat", "fold"])
    decoded.tables["patterns"] = table[
        [*keys, "time", "channel", "component", "pattern"]
    ]


def _input_fingerprint(pipeline, subject: str) -> str:
    row = pipeline.manifest.loc[
        pipeline.manifest["subject_index"].astype(str).eq(subject)
    ].iloc[0]
    paths = [path for path in pipeline.subject_paths(subject).values() if path.exists()]
    return fingerprint(
        {
            "manifest_input": row["input_fingerprint"],
            "manifest_pipeline": row["pipeline_fingerprint"],
            "files": fingerprint_files(paths, root=pipeline.root),
        }
    )


def _capture(function, subject: str):
    try:
        return function(subject), None
    except Exception as error:
        return None, error


def _version() -> str:
    try:
        return version("mveeg")
    except PackageNotFoundError:
        return "0.3.0"
