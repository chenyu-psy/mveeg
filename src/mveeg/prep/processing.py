"""Dataset orchestration for signal preprocessing and artifact labeling."""

from __future__ import annotations

import tempfile
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .._dataset.manifest import MANIFEST_COLUMNS, read_json, relative_paths
from .._dataset.store import (
    RECOMPUTE_VALUES,
    DatasetBuilder,
    write_json_atomic,
    write_table_atomic,
)
from .._provenance import fingerprint, fingerprint_files
from .artifacts import (
    _join_reasons,
    build_artifact_table,
    read_artifact_table,
    write_artifact_table,
)
from .gaze import normalize_gaze_geometry
from .pipeline.dataset import DatasetPipeline, open_pipeline
from .quality import (
    apply_autoreject,
    check_eligibility,
    label_artifact_rules,
    load_quality_state,
    save_quality_state,
)
from .quality.eligibility import _gaze_rule_requires_geometry


def _preprocess_epochs(
    prepared: DatasetPipeline,
    output_dir: str | Path,
    *,
    eligibility: Mapping[str, object],
    autoreject: Mapping[str, object] | None = None,
    recompute: str = "never",
) -> DatasetPipeline:
    """Check eligibility, optionally apply AutoReject, and save all epochs.

    AutoReject is fit only on eligible epochs. Neither eligibility nor
    AutoReject removes a row from the preprocessed dataset.
    """

    source = prepared
    if source.stage != "prepared":
        raise ValueError(f"preprocess_epochs requires a prepared dataset, found {source.stage!r}.")
    source_provenance = read_json(source.root / "provenance.json")
    gaze_geometry = _gaze_geometry_from_provenance(
        source_provenance,
        required=(
            "gaze" in eligibility
            and _gaze_rule_requires_geometry(eligibility["gaze"], context="eligibility.gaze")
        ),
        context="eligibility.gaze",
        pipeline_spec=False,
    )
    autoreject_config = None if autoreject is None else dict(autoreject)
    if autoreject_config is not None:
        autoreject_config.setdefault("random_state", 0)
    pipeline_spec = {
        "kind": "preprocess_epochs",
        "source_pipeline_fingerprint": source_provenance["pipeline_fingerprint"],
        "gaze_geometry": gaze_geometry,
        "eligibility": dict(eligibility),
        "autoreject": autoreject_config,
    }
    builder = DatasetBuilder(
        output_dir,
        task=source.task,
        stage="preprocessed",
        pipeline_fingerprint=fingerprint(pipeline_spec),
        pipeline_spec=pipeline_spec,
        recompute=recompute,
        subject_indices=source.subject_indices,
        complete_subject_set=True,
    )
    try:
        for subject_index in source.subject_indices:
            source_paths = [
                source.path_for_subject(subject_index, kind)
                for kind in ("epochs", "events", "eeg_json")
            ]
            input_fingerprint = fingerprint_files(source_paths, root=source.root)
            saved_quality_path = quality_state_path(output_dir, subject_index, source.task)
            should_write = builder.should_write(subject_index, input_fingerprint)
            if not should_write and saved_quality_path.exists():
                builder.record_reused(subject_index)
                continue

            artifact_path = (
                builder.root
                / relative_paths(str(subject_index), source.task, "preprocessed")["artifacts_path"]
            )
            if artifact_path.exists() and read_artifact_table(artifact_path)["reviewed"].any():
                if recompute == "changed":
                    raise RuntimeError(
                        f"Preprocessing subject {subject_index} would discard reviewed artifact "
                        "decisions. Back up the artifact sidecar and use recompute='all' to "
                        "confirm replacement."
                    )
                if recompute == "all":
                    warnings.warn(
                        f"Preprocessing subject {subject_index} with recompute='all' discards "
                        "reviewed artifact decisions.",
                        stacklevel=2,
                    )

            epochs = source.load_epochs(subject_index, preload=True)
            eligibility_result = check_eligibility(
                epochs,
                eligibility,
                gaze_geometry=gaze_geometry,
            )
            autoreject_result = apply_autoreject(
                epochs,
                eligibility_result.eligible,
                autoreject_config,
            )
            if len(autoreject_result.epochs) != len(epochs):
                raise RuntimeError("Signal preprocessing changed the number of epochs.")
            builder.write_subject(
                subject_index,
                autoreject_result.epochs,
                input_fingerprint=input_fingerprint,
            )
            staged_quality_path = quality_state_path(
                builder.working_root, subject_index, source.task
            )
            _save_quality_state_atomic(staged_quality_path, eligibility_result, autoreject_result)
        summary = builder.finish()
        result = DatasetPipeline(output_dir, run_summary=summary)
    except Exception:
        builder.abort()
        raise
    _extend_provenance(
        result.root,
        {"source_dataset": source.root.as_posix()},
    )
    return result


def label_artifacts(
    pipeline: DatasetPipeline | str | Path,
    *,
    reject: Mapping[str, object],
    review: Mapping[str, object],
    ignore_channels: Sequence[str] = (),
    recompute: str = "all",
) -> DatasetPipeline:
    """Create or refresh artifact sidecars.

    Parameters
    ----------
    pipeline : DatasetPipeline or path-like
        Preprocessed dataset to label.
    reject, review : mapping
        Automatic rejection and review rule configurations.
    ignore_channels : sequence of str
        Channels retained in sidecars but excluded from trial aggregation.
    recompute : {"never", "changed", "all"}
        Reuse existing sidecars, refresh them only when labeling rules changed,
        or always refresh them. Missing sidecars are generated.

    Returns
    -------
    DatasetPipeline
        Refreshed dataset with artifact paths recorded in its manifest.

    """

    if recompute not in RECOMPUTE_VALUES:
        raise ValueError(f"recompute must be one of {sorted(RECOMPUTE_VALUES)}.")
    dataset = open_pipeline(pipeline) if not isinstance(pipeline, DatasetPipeline) else pipeline
    if dataset.stage != "preprocessed":
        raise ValueError(
            f"label_artifacts requires a preprocessed dataset, found {dataset.stage!r}."
        )
    if "gaze" in reject:
        raise ValueError("reject.gaze is unsupported; hard gaze rules belong in eligibility.gaze.")
    provenance = read_json(dataset.root / "provenance.json")
    gaze_geometry = _gaze_geometry_from_provenance(
        provenance,
        required=(
            "gaze" in review and _gaze_rule_requires_geometry(review["gaze"], context="review.gaze")
        ),
        context="review.gaze",
        pipeline_spec=True,
    )
    labeling = {
        "reject": dict(reject),
        "review": dict(review),
        "ignore_channels": list(ignore_channels),
        "gaze_geometry": gaze_geometry,
    }
    labeling_fingerprint = fingerprint(labeling)
    previous_fingerprint = provenance.get("artifact_labeling", {}).get("fingerprint")
    labeling_changed = previous_fingerprint != labeling_fingerprint
    manifest = dataset.manifest
    artifact_paths = {
        subject: dataset.path_for_subject(subject, "artifacts")
        for subject in dataset.subject_indices
    }
    existing_count = sum(path.exists() for path in artifact_paths.values())
    if recompute == "never" and labeling_changed and 0 < existing_count < len(artifact_paths):
        raise RuntimeError(
            "recompute='never' cannot generate missing artifact sidecars with labeling rules "
            "that differ from existing sidecars."
        )
    if recompute == "never" and labeling_changed and existing_count:
        warnings.warn(
            "Artifact labeling configuration changed, but recompute='never' preserves existing "
            "sidecars.",
            stacklevel=2,
        )
    wrote = False
    for subject_index in dataset.subject_indices:
        artifact_path = artifact_paths[subject_index]
        if artifact_path.exists() and (
            recompute == "never" or (recompute == "changed" and not labeling_changed)
        ):
            continue
        epochs = dataset.load_epochs(subject_index, preload=True)
        quality_path = quality_state_path(dataset.root, subject_index, dataset.task)
        if not quality_path.exists():
            raise FileNotFoundError(
                f"Missing AutoReject/eligibility state for subject {subject_index}: {quality_path}."
            )
        eligibility_result, autoreject_result = load_quality_state(quality_path)
        _validate_autoreject_state(epochs, autoreject_result)
        try:
            rule_result = label_artifact_rules(
                epochs,
                eligibility_result,
                autoreject_result["bad_epochs"],
                autoreject_labels=autoreject_result["labels"],
                reject_config=reject,
                review_config=review,
                ignore_channels=ignore_channels,
                gaze_geometry=gaze_geometry,
            )
        except ValueError as error:
            raise ValueError(f"Subject {subject_index}: {error}") from error
        previous = read_artifact_table(artifact_path) if artifact_path.exists() else None
        channel_reasons = _append_autoreject_channel_reasons(
            epochs,
            rule_result.rejected_reasons,
            autoreject_result,
        )
        table = build_artifact_table(
            subject_index,
            epochs.metadata["epoch_index"].to_numpy(dtype=int),
            epochs.ch_names,
            rejected_reasons=channel_reasons,
            review_reasons=rule_result.review_reasons,
            epoch_rejected=rule_result.epoch_rejected,
            epoch_review=rule_result.epoch_review,
            epoch_reasons={"autoreject_bad_epoch": autoreject_result["bad_epochs"]},
            ignore_channels=ignore_channels,
            previous=previous,
        )
        write_artifact_table(table, artifact_path)
        wrote = True
        relative = artifact_path.relative_to(dataset.root).as_posix()
        manifest.loc[
            manifest["subject_index"].astype(str).eq(str(subject_index)),
            "artifacts_path",
        ] = relative
    if not wrote:
        return dataset.refresh()

    write_table_atomic(manifest[MANIFEST_COLUMNS], dataset.root / "manifest.tsv")
    _extend_provenance(
        dataset.root,
        {
            "artifact_labeling": {
                "fingerprint": labeling_fingerprint,
                **labeling,
            }
        },
    )
    return dataset.refresh()


def _append_autoreject_channel_reasons(
    epochs,
    reasons,
    autoreject_state: Mapping[str, object],
) -> np.ndarray:
    """Add AutoReject EEG labels to a full epoch-by-channel reason matrix."""

    output = np.asarray(reasons, dtype=object).copy()
    labels = np.asarray(autoreject_state["labels"], dtype=np.int8)
    eeg_channels = [str(channel) for channel in autoreject_state["eeg_channels"]]
    channel_positions = {channel: index for index, channel in enumerate(epochs.ch_names)}
    for eeg_position, channel in enumerate(eeg_channels):
        channel_position = channel_positions[channel]
        for epoch_position in np.flatnonzero(labels[:, eeg_position] > 0):
            output[epoch_position, channel_position] = _join_reasons(
                output[epoch_position, channel_position],
                "autoreject_bad_channel",
            )
        for epoch_position in np.flatnonzero(labels[:, eeg_position] == 2):
            output[epoch_position, channel_position] = _join_reasons(
                output[epoch_position, channel_position],
                "autoreject_interpolated",
            )
    return output


def quality_state_path(
    root: str | Path,
    subject_index: str | int,
    task: str,
) -> Path:
    """Return the compressed eligibility/AutoReject state path."""

    subject = str(subject_index)
    if subject.startswith("sub-"):
        subject = subject[4:]
    elif subject.startswith("sub"):
        subject = subject[3:]
    return (
        Path(root).expanduser().resolve()
        / f"sub-{subject}"
        / "eeg"
        / f"sub-{subject}_task-{task}_desc-autoreject.npz"
    )


def _save_quality_state_atomic(path: Path, eligibility, autoreject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        save_quality_state(temporary, eligibility, autoreject)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _extend_provenance(root: Path, updates: Mapping[str, object]) -> None:
    provenance = read_json(root / "provenance.json")
    provenance.update(updates)
    write_json_atomic(provenance, root / "provenance.json")


def _validate_autoreject_state(epochs, state: Mapping[str, object]) -> None:
    """Reject stale dense AutoReject state before keyed artifact labeling."""

    n_epochs = len(epochs)
    eeg_channels = tuple(
        channel
        for channel, kind in zip(epochs.ch_names, epochs.get_channel_types())
        if kind == "eeg"
    )
    saved_channels = tuple(str(channel) for channel in state["eeg_channels"])
    if saved_channels != eeg_channels:
        raise ValueError(
            "AutoReject state EEG channels do not match the preprocessed epochs: "
            f"saved={saved_channels}, current={eeg_channels}."
        )
    expected_epoch_vector = (n_epochs,)
    for key in ("bad_epochs", "interpolated_channels"):
        if state[key].shape != expected_epoch_vector:
            raise ValueError(
                f"AutoReject state {key!r} has shape {state[key].shape}; "
                f"expected {expected_epoch_vector}."
            )
    expected_labels = (n_epochs, len(eeg_channels))
    if state["labels"].shape != expected_labels:
        raise ValueError(
            "AutoReject state channel labels have shape "
            f"{state['labels'].shape}; expected {expected_labels}."
        )


def _gaze_geometry_from_provenance(
    provenance: Mapping[str, object],
    *,
    required: bool,
    context: str,
    pipeline_spec: bool,
) -> dict[str, float | int] | None:
    """Read canonical geometry from its stage-specific provenance location."""

    container: object = provenance.get("pipeline", {}) if pipeline_spec else provenance
    if not isinstance(container, Mapping):
        raise TypeError("Dataset provenance pipeline must be a mapping.")
    raw_geometry = container.get("gaze_geometry")
    if raw_geometry is None:
        if required:
            raise ValueError(f"{context} requires gaze_geometry in dataset provenance.")
        return None
    return _normalize_geometry_mapping(raw_geometry)


def _normalize_geometry_mapping(value: object) -> dict[str, float | int]:
    """Normalize a provenance mapping while rejecting legacy geometry keys."""

    if not isinstance(value, Mapping):
        raise TypeError("gaze_geometry in dataset provenance must be a mapping.")
    try:
        return normalize_gaze_geometry(**dict(value))
    except TypeError as error:
        raise ValueError(
            "gaze_geometry must contain exactly viewing_distance_cm, "
            "screen_width_cm, and screen_width_px."
        ) from error
