"""Dataset-root loading and trial-metadata contracts for model workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from numbers import Number
from pathlib import Path, PurePosixPath

import mne
import numpy as np
import pandas as pd

from .io_filters import apply_trial_filters, channels_to_drop_by_rule


IDENTITY_COLUMNS = ("subject_index", "epoch_index")
ARTIFACT_METADATA_COLUMNS = (
    "initial_status",
    "final_status",
    "epoch_reasons",
)
_PATH_COLUMNS = ("epochs_path", "events_path", "eeg_json_path", "artifacts_path")
_METADATA_FLOAT_ATOL = 5e-4


def transform_metadata(
    metadata: pd.DataFrame,
    metadata_transform: Callable[[pd.DataFrame], pd.DataFrame] | None,
) -> pd.DataFrame:
    """Apply a row-preserving transform to trial metadata.

    The transform may add or change analysis columns, but it must return a
    DataFrame with the same ``subject_index``/``epoch_index`` sequence.
    """

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame.")
    _validate_identity(metadata, context="metadata")

    output = metadata.copy() if metadata_transform is None else metadata_transform(metadata.copy())
    if not isinstance(output, pd.DataFrame):
        raise TypeError("metadata_transform must return a pandas DataFrame.")
    if len(output) != len(metadata):
        raise ValueError("metadata_transform must preserve the number of trial rows.")
    _validate_identity(output, context="metadata_transform output")
    missing = sorted(set(metadata.columns).difference(output.columns))
    if missing:
        raise ValueError(
            "metadata_transform cannot remove existing columns: "
            f"{missing}."
        )

    before = metadata.loc[:, IDENTITY_COLUMNS].reset_index(drop=True)
    after = output.loc[:, IDENTITY_COLUMNS].reset_index(drop=True)
    if not after.equals(before):
        raise ValueError(
            "metadata_transform must preserve subject_index and epoch_index values and order."
        )
    return output.reset_index(drop=True)


def assign_metadata_variables(
    metadata: pd.DataFrame,
    variables: Mapping[str, Callable[[pd.DataFrame], object]],
) -> pd.DataFrame:
    """Create analysis variables from ordered DataFrame-to-column functions.

    Each function receives the current metadata table and returns one scalar or
    one trial-aligned column. Variables are evaluated in mapping order, so a
    later function may use variables created earlier in the same call.
    """

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame.")
    if not isinstance(variables, Mapping):
        raise TypeError("variables must be a mapping from column names to callables.")
    _validate_identity(metadata, context="metadata")

    output = metadata.copy().reset_index(drop=True)
    identity = output.loc[:, IDENTITY_COLUMNS].copy()
    for name, function in variables.items():
        if not isinstance(name, str) or name.strip() == "":
            raise ValueError("Metadata variable names must be non-empty strings.")
        if name in IDENTITY_COLUMNS:
            raise ValueError(
                "transform_metadata cannot replace subject_index or epoch_index."
            )
        if not callable(function):
            raise TypeError(f"Metadata variable {name!r} must be defined by a callable.")

        value = function(output.copy())
        if isinstance(value, pd.DataFrame):
            raise TypeError(
                f"Metadata variable {name!r} must return one column, not a DataFrame."
            )
        if not np.isscalar(value):
            try:
                value_length = len(value)
            except TypeError as error:
                raise TypeError(
                    f"Metadata variable {name!r} must return a scalar or one column."
                ) from error
            if value_length != len(output):
                raise ValueError(
                    f"Metadata variable {name!r} returned {value_length} values for "
                    f"{len(output)} trials."
                )
        output[name] = value

    if not output.loc[:, IDENTITY_COLUMNS].equals(identity):
        raise ValueError(
            "transform_metadata must preserve subject_index and epoch_index values and order."
        )
    return output


def metadata_transform_spec(
    metadata_transform: Callable[[pd.DataFrame], pd.DataFrame] | None,
    *,
    name: str | None,
    version: str | None,
) -> dict[str, str] | None:
    """Validate and return the stable fingerprint fields for a transform."""

    if metadata_transform is None:
        if name is not None or version is not None:
            raise ValueError(
                "metadata_transform_name/version require metadata_transform."
            )
        return None
    if not callable(metadata_transform):
        raise TypeError("metadata_transform must be callable.")
    if name is None or str(name).strip() == "":
        raise ValueError("metadata_transform_name is required when using metadata_transform.")
    if version is None or str(version).strip() == "":
        raise ValueError("metadata_transform_version is required when using metadata_transform.")
    return {"name": str(name).strip(), "version": str(version).strip()}


def load_dataset_manifest(dataset_root: str | Path) -> pd.DataFrame:
    """Read and validate a prepared/preprocessed dataset manifest."""

    dataset_root = Path(dataset_root)
    manifest_path = dataset_root / "manifest.tsv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    manifest = pd.read_csv(
        manifest_path,
        sep="\t",
        dtype={"subject_index": "string"},
    )
    required = {"subject_index", "task", "stage", *_PATH_COLUMNS}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"manifest.tsv is missing required columns: {missing}")
    if manifest["subject_index"].isna().any() or manifest["subject_index"].str.strip().eq("").any():
        raise ValueError("manifest.tsv subject_index values cannot be missing or empty.")
    manifest["subject_index"] = manifest["subject_index"].astype(str)
    if manifest["subject_index"].duplicated().any():
        duplicates = sorted(
            manifest.loc[manifest["subject_index"].duplicated(False), "subject_index"].unique()
        )
        raise ValueError(f"manifest.tsv contains duplicate subject_index values: {duplicates}")
    return manifest.reset_index(drop=True)


def load_validated_dataset_manifest(dataset_root: str | Path) -> pd.DataFrame:
    """Read a complete mveeg dataset contract without loading signal data."""

    root = Path(dataset_root).expanduser().resolve()
    manifest = load_dataset_manifest(root)
    if len(manifest) == 0:
        raise ValueError(f"Dataset manifest is empty: {root / 'manifest.tsv'}.")
    required = {"input_fingerprint", "pipeline_fingerprint"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"manifest.tsv is missing required columns: {missing}")

    task = _single_manifest_value(manifest, "task")
    stage = _single_manifest_value(manifest, "stage")
    pipeline_fingerprint = _single_manifest_value(manifest, "pipeline_fingerprint")
    provenance = _read_json_object(root / "provenance.json")
    description = _read_json_object(root / "dataset_description.json")
    if provenance.get("pipeline_fingerprint") != pipeline_fingerprint:
        raise ValueError("Manifest and provenance contain different pipeline fingerprints.")
    if provenance.get("task") != task or provenance.get("stage") != stage:
        raise ValueError("Manifest and provenance disagree on task or processing stage.")
    if description.get("Task") != task or description.get("Stage") != stage:
        raise ValueError("Manifest and dataset_description disagree on task or processing stage.")
    return manifest


def list_dataset_subjects(dataset_root: str | Path) -> list[str]:
    """Return subject indices in manifest order."""

    return load_dataset_manifest(dataset_root)["subject_index"].tolist()


def subject_dataset_paths(
    dataset_root: str | Path,
    subject_index: str | int,
) -> dict[str, Path | None]:
    """Resolve one manifest row to dataset-root-contained file paths."""

    dataset_root = Path(dataset_root)
    subject = _normalize_subject_index(subject_index)
    manifest = load_dataset_manifest(dataset_root)
    rows = manifest.loc[manifest["subject_index"] == subject]
    if len(rows) == 0:
        raise KeyError(f"subject_index '{subject}' is not present in manifest.tsv.")
    row = rows.iloc[0]
    resolved: dict[str, Path | None] = {}
    for column in _PATH_COLUMNS:
        value = row.get(column)
        relative = None if pd.isna(value) or str(value).strip() == "" else str(value).strip()
        if relative is None:
            if column == "artifacts_path":
                resolved[column] = None
                continue
            raise ValueError(
                f"manifest.tsv has an empty {column} for subject_index '{subject}'."
            )
        resolved[column] = _resolve_dataset_path(dataset_root, relative)
    return resolved


def load_subject_epochs_and_metadata(
    dataset_root: str | Path,
    subject_index: str | int,
    *,
    preload: bool,
) -> tuple[mne.Epochs, pd.DataFrame]:
    """Load one subject and key-merge trial-level artifact fields."""

    subject = _normalize_subject_index(subject_index)
    paths = subject_dataset_paths(dataset_root, subject)
    epochs_path = _required_file(paths["epochs_path"], "epochs")
    epochs = mne.read_epochs(epochs_path, preload=preload, verbose="ERROR")
    base_metadata = _load_subject_events(paths, subject)
    if len(base_metadata) != len(epochs):
        raise ValueError(
            f"events.tsv has {len(base_metadata)} rows but the epochs file has {len(epochs)} epochs "
            f"for subject_index '{subject}'."
        )
    if epochs.metadata is None:
        raise ValueError(f"Epochs file has no identity metadata: {epochs_path}")
    fif_metadata = _normalize_identity(epochs.metadata, context="epochs metadata")
    _validate_metadata_mirror(fif_metadata, base_metadata)
    with mne.use_log_level("ERROR"):
        epochs.metadata = base_metadata

    return epochs, _merge_subject_artifacts(base_metadata, paths)


def load_subject_metadata(
    dataset_root: str | Path,
    subject_index: str | int,
) -> pd.DataFrame:
    """Load QC-merged subject metadata without opening the epochs file."""

    subject = _normalize_subject_index(subject_index)
    paths = subject_dataset_paths(dataset_root, subject)
    metadata = _load_subject_events(paths, subject)
    return _merge_subject_artifacts(metadata, paths)


def _load_subject_events(paths: dict[str, Path | None], subject: str) -> pd.DataFrame:
    events_path = _required_file(paths["events_path"], "events")
    metadata = pd.read_csv(
        events_path,
        sep="\t",
        dtype={"subject_index": "string"},
    )
    metadata = _normalize_identity(metadata, context="events.tsv")
    if set(metadata["subject_index"]) != {subject}:
        raise ValueError(
            f"events.tsv contains subject_index values other than '{subject}'."
        )
    return metadata


def _merge_subject_artifacts(
    metadata: pd.DataFrame,
    paths: dict[str, Path | None],
) -> pd.DataFrame:
    artifacts_path = paths["artifacts_path"]
    if artifacts_path is None:
        return metadata

    artifacts = pd.read_csv(
        _required_file(artifacts_path, "artifacts"),
        sep="\t",
        dtype={"subject_index": "string"},
    )
    artifacts = _normalize_identity(artifacts, context="artifacts.tsv")
    return _merge_artifact_metadata(metadata, artifacts)


def validate_metadata_mirror(
    epochs_metadata: pd.DataFrame,
    events_metadata: pd.DataFrame,
) -> None:
    """Validate two serialized copies of immutable base trial metadata."""

    normalized_epochs = _normalize_identity(
        epochs_metadata,
        context="epochs metadata",
    )
    normalized_events = _normalize_identity(
        events_metadata,
        context="events.tsv",
    )
    _validate_metadata_mirror(normalized_epochs, normalized_events)


def load_subject_metadata_table(
    subject_index: str | int,
    cfg,
    *,
    metadata_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Load QC-merged metadata and apply the analysis transform."""

    _, metadata = load_subject_epochs_and_metadata(
        cfg.dataset.data_dir,
        subject_index,
        preload=False,
    )
    return transform_metadata(metadata, metadata_transform)


def load_subject_data_with_filters(
    subject_index: str | int,
    cfg,
    *,
    return_metadata: bool = False,
    metadata_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple:
    """Load one subject, transform metadata, then apply configured filters."""

    epochs, metadata = load_subject_epochs_and_metadata(
        cfg.dataset.data_dir,
        subject_index,
        preload=True,
    )
    epoch_cfg = cfg.decode if hasattr(cfg, "decode") else cfg.epoch
    chans_to_drop = channels_to_drop_by_rule(epochs, epoch_cfg)
    if chans_to_drop:
        epochs.drop_channels(chans_to_drop)
    if epoch_cfg.crop_time is not None:
        epochs.crop(
            tmin=epoch_cfg.crop_time[0],
            tmax=epoch_cfg.crop_time[1],
            include_tmax=True,
        )

    metadata = transform_metadata(metadata, metadata_transform)
    qc_col = cfg.filters.qc_col
    if qc_col is not None and qc_col not in metadata.columns:
        raise ValueError(
            f"Configured QC column '{qc_col}' is missing; label artifacts first or set qc_col=None."
        )
    keep_mask = apply_trial_filters(metadata, cfg)
    if not np.any(keep_mask):
        raise ValueError(
            f"No usable trials remain for subject {subject_index} after filtering."
        )
    filtered_epochs = epochs[np.flatnonzero(keep_mask)]
    filtered_metadata = metadata.loc[keep_mask].reset_index(drop=True)

    if hasattr(cfg, "label_for_metadata_row"):
        labels = cfg.label_for_metadata_row(filtered_metadata)
        train_label_order = cfg.train_label_order()
    else:
        source_values = filtered_metadata[cfg.conditions.cond_col].to_numpy(dtype=object)
        if isinstance(cfg.conditions.train_cond, dict):
            labels = np.full(len(filtered_metadata), "", dtype=object)
            for label, group_values in cfg.conditions.train_cond.items():
                labels[np.isin(source_values, group_values)] = label
            train_label_order = list(cfg.conditions.train_cond)
        else:
            labels = source_values
            train_label_order = list(cfg.conditions.train_cond)

    missing_labels = sorted(
        set(label for label in np.unique(labels) if label != "") - set(train_label_order)
    )
    if missing_labels:
        raise ValueError(
            "Found training labels not listed in configured train_cond for subject "
            f"{subject_index}: {missing_labels}"
        )

    output = (
        filtered_epochs.get_data(copy=True),
        labels,
        filtered_epochs.times.copy(),
        filtered_epochs.ch_names.copy(),
    )
    return (*output, filtered_metadata) if return_metadata else output


def load_subject_info_with_channel_drop(subject_index: str | int, cfg) -> mne.Info:
    """Load one subject's channel info after configured channel removal."""

    paths = subject_dataset_paths(cfg.dataset.data_dir, subject_index)
    epochs = mne.read_epochs(
        _required_file(paths["epochs_path"], "epochs"),
        preload=False,
        verbose="ERROR",
    )
    epoch_cfg = cfg.decode if hasattr(cfg, "decode") else cfg.epoch
    chans_to_drop = channels_to_drop_by_rule(epochs, epoch_cfg)
    if chans_to_drop:
        epochs.drop_channels(chans_to_drop)
    return epochs.info.copy()


def _merge_artifact_metadata(
    metadata: pd.DataFrame,
    artifacts: pd.DataFrame,
) -> pd.DataFrame:
    missing = sorted(set(ARTIFACT_METADATA_COLUMNS).difference(artifacts.columns))
    if missing:
        raise ValueError(f"artifacts.tsv is missing required summary columns: {missing}")
    fields = list(ARTIFACT_METADATA_COLUMNS)
    collisions = sorted(set(fields).intersection(metadata.columns))
    if collisions:
        raise ValueError(
            "Trial artifact fields must not be duplicated in base metadata: "
            f"{collisions}"
        )
    artifact_rows = artifacts.loc[:, [*IDENTITY_COLUMNS, *fields]]
    merged = metadata.merge(
        artifact_rows,
        on=list(IDENTITY_COLUMNS),
        how="outer",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[
            merged["_merge"].ne("both"), [*IDENTITY_COLUMNS, "_merge"]
        ].to_dict(orient="records")
        raise ValueError(
            "events.tsv and artifacts.tsv must contain exactly the same trial keys. "
            f"Mismatches: {missing[:10]}"
        )
    merged = merged.drop(columns="_merge")
    expected = metadata.loc[:, IDENTITY_COLUMNS].reset_index(drop=True)
    observed = merged.loc[:, IDENTITY_COLUMNS].reset_index(drop=True)
    if not observed.equals(expected):
        raise ValueError("Artifact merge changed trial order.")
    return merged


def _single_manifest_value(manifest: pd.DataFrame, column: str) -> str:
    values = manifest[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"Dataset manifest must contain exactly one {column} value.")
    return str(values[0])


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _normalize_identity(metadata: pd.DataFrame, *, context: str) -> pd.DataFrame:
    metadata = metadata.copy()
    missing = sorted(set(IDENTITY_COLUMNS).difference(metadata.columns))
    if missing:
        raise ValueError(f"{context} is missing identity columns: {missing}")
    if metadata["subject_index"].isna().any():
        raise ValueError(f"{context} subject_index cannot contain missing values.")
    metadata["subject_index"] = metadata["subject_index"].astype(str)
    if metadata["subject_index"].str.strip().eq("").any():
        raise ValueError(f"{context} subject_index cannot contain empty values.")
    epoch_index = pd.to_numeric(metadata["epoch_index"], errors="raise")
    if epoch_index.isna().any() or not np.equal(epoch_index, np.floor(epoch_index)).all():
        raise ValueError(f"{context} epoch_index must contain integers without missing values.")
    metadata["epoch_index"] = epoch_index.astype(np.int64)
    _validate_identity(metadata, context=context)
    return metadata.reset_index(drop=True)


def _validate_metadata_mirror(
    epochs_metadata: pd.DataFrame,
    events_metadata: pd.DataFrame,
) -> None:
    """Require FIF metadata and events.tsv to be the same base metadata mirror."""

    if len(epochs_metadata) != len(events_metadata):
        raise ValueError(
            "Epochs metadata and events.tsv must contain the same number of trial rows."
        )
    if list(epochs_metadata.columns) != list(events_metadata.columns):
        raise ValueError(
            "Epochs metadata and events.tsv must contain the same metadata columns "
            "in the same order."
        )
    epochs_identity = epochs_metadata.loc[:, IDENTITY_COLUMNS]
    events_identity = events_metadata.loc[:, IDENTITY_COLUMNS]
    if not epochs_identity.equals(events_identity):
        detail = _first_metadata_mismatch(epochs_metadata, events_metadata)
        raise ValueError(
            "Epochs metadata and events.tsv must contain the same trial identity "
            f"in the same order. {detail}"
        )
    try:
        pd.testing.assert_frame_equal(
            epochs_metadata.reset_index(drop=True),
            events_metadata.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=_METADATA_FLOAT_ATOL,
            check_categorical=False,
        )
    except AssertionError as error:
        detail = _first_metadata_mismatch(epochs_metadata, events_metadata)
        raise ValueError(
            "Epochs metadata and events.tsv must contain the same base metadata "
            f"values in the same trial order. {detail}"
        ) from error


def _first_metadata_mismatch(
    epochs_metadata: pd.DataFrame,
    events_metadata: pd.DataFrame,
) -> str:
    for column in epochs_metadata.columns:
        for position, (fif_value, events_value) in enumerate(
            zip(epochs_metadata[column], events_metadata[column], strict=True)
        ):
            fif_missing = pd.isna(fif_value)
            events_missing = pd.isna(events_value)
            if fif_missing or events_missing:
                matches = bool(fif_missing and events_missing)
            elif isinstance(fif_value, Number) and isinstance(events_value, Number):
                matches = bool(
                    np.isclose(
                        fif_value,
                        events_value,
                        rtol=0,
                        atol=_METADATA_FLOAT_ATOL,
                    )
                )
            else:
                matches = bool(fif_value == events_value)
            if not matches:
                epoch_index = int(events_metadata.iloc[position]["epoch_index"])
                return (
                    f"First mismatch at row {position} (epoch_index={epoch_index!r}), "
                    f"column {column!r}: FIF={fif_value!r}, "
                    f"events.tsv={events_value!r}."
                )
    return ""


def _validate_identity(metadata: pd.DataFrame, *, context: str) -> None:
    missing = sorted(set(IDENTITY_COLUMNS).difference(metadata.columns))
    if missing:
        raise ValueError(f"{context} is missing identity columns: {missing}")
    if metadata.loc[:, IDENTITY_COLUMNS].isna().any(axis=None):
        raise ValueError(f"{context} identity columns cannot contain missing values.")
    if metadata.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(
            f"{context} must have unique subject_index + epoch_index keys."
        )


def _normalize_subject_index(subject_index: str | int) -> str:
    subject = str(subject_index).strip()
    return subject[4:] if subject.startswith("sub-") else subject


def _resolve_dataset_path(dataset_root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Manifest paths must be relative to dataset root: {relative_path}")
    return dataset_root.joinpath(*relative.parts)


def _required_file(path: Path | None, label: str) -> Path:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Subject {label} file not found: {path}")
    return path
