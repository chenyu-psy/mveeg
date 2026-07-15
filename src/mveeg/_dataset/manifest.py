"""Prepared-dataset manifest paths and read contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

import mne
import pandas as pd

from .metadata import merge_artifact_metadata, normalize_identity, validate_metadata_mirror

DATASET_SCHEMA_VERSION = 2
MANIFEST_COLUMNS = [
    "subject_index",
    "task",
    "stage",
    "epochs_path",
    "events_path",
    "eeg_json_path",
    "artifacts_path",
    "input_fingerprint",
    "pipeline_fingerprint",
    "n_epochs",
    "n_channels",
    "sampling_rate",
    "tmin",
    "tmax",
]
PATH_COLUMNS = {
    "epochs": "epochs_path",
    "events": "events_path",
    "eeg_json": "eeg_json_path",
    "artifacts": "artifacts_path",
}


def normalize_subject_index(value: str | int) -> str:
    """Normalize common subject-folder prefixes to a stable index."""

    subject = str(value).strip()
    if subject.startswith("sub-"):
        subject = subject[4:]
    elif subject.startswith("sub") and subject[3:].isdigit():
        subject = subject[3:]
    return validate_entity(subject, "subject_index")


def validate_entity(value: str, name: str) -> str:
    """Validate a compact filename entity."""

    value = str(value)
    if not value or re.fullmatch(r"[A-Za-z0-9]+", value) is None:
        raise ValueError(f"{name} must contain only letters and numbers, got {value!r}.")
    return value


def relative_paths(subject: str, task: str, stage: str) -> dict[str, str]:
    """Build standard root-relative POSIX paths for one subject."""

    stem = f"sub-{subject}_task-{task}_desc-{stage}"
    directory = Path(f"sub-{subject}") / "eeg"
    return {
        "epochs_path": (directory / f"{stem}_epo.fif").as_posix(),
        "events_path": (directory / f"sub-{subject}_task-{task}_events.tsv").as_posix(),
        "eeg_json_path": (directory / f"sub-{subject}_task-{task}_eeg.json").as_posix(),
        "artifacts_path": (directory / f"sub-{subject}_task-{task}_desc-artifacts.tsv").as_posix(),
    }


def read_manifest(root: str | Path, *, missing_ok: bool = False) -> pd.DataFrame:
    """Read a manifest with stable identifiers and ordered columns."""

    root = Path(root)
    path = root / "manifest.tsv"
    if not path.exists():
        if missing_ok:
            return pd.DataFrame(columns=MANIFEST_COLUMNS)
        raise FileNotFoundError(f"No mveeg manifest found at {path}.")
    manifest = pd.read_csv(
        path,
        sep="\t",
        keep_default_na=False,
        dtype={column: str for column in MANIFEST_COLUMNS[:9]},
    )
    missing = [column for column in MANIFEST_COLUMNS if column not in manifest]
    if missing:
        raise ValueError(f"Dataset manifest is missing columns: {missing}.")
    manifest = manifest.loc[:, MANIFEST_COLUMNS]
    if manifest["subject_index"].duplicated().any():
        duplicates = manifest.loc[
            manifest["subject_index"].duplicated(False), "subject_index"
        ].tolist()
        raise ValueError(f"Manifest has duplicate subject_index values: {duplicates}")
    return manifest


def single_value(table: pd.DataFrame, column: str) -> str:
    """Return one required dataset-wide manifest value."""

    values = table[column].astype(str).unique()
    if len(values) != 1:
        raise ValueError(f"Manifest must contain one {column!r} value, found {list(values)}.")
    return str(values[0])


def load_validated_manifest(root: str | Path) -> pd.DataFrame:
    """Validate the complete dataset contract without loading signal data."""

    root = Path(root).expanduser().resolve()
    manifest = read_manifest(root)
    if len(manifest) == 0:
        raise ValueError(f"Dataset manifest is empty: {root / 'manifest.tsv'}.")
    provenance = read_json(root / "provenance.json")
    description = read_json(root / "dataset_description.json")
    schema_version = provenance.get("schema_version")
    if schema_version != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Dataset schema {schema_version!r} is unsupported; regenerate with "
            f"schema {DATASET_SCHEMA_VERSION}."
        )
    task = single_value(manifest, "task")
    stage = single_value(manifest, "stage")
    pipeline_fingerprint = single_value(manifest, "pipeline_fingerprint")
    if provenance.get("pipeline_fingerprint") != pipeline_fingerprint:
        raise ValueError("Manifest and provenance contain different pipeline fingerprints.")
    if provenance.get("task") != task or provenance.get("stage") != stage:
        raise ValueError("Manifest and provenance disagree on task or processing stage.")
    if description.get("Task") != task or description.get("Stage") != stage:
        raise ValueError("Manifest and dataset_description disagree on task or stage.")
    return manifest


def subject_paths(
    root: str | Path,
    subject_index: str | int,
) -> dict[str, Path | None]:
    """Resolve one manifest row to root-contained paths."""

    root = Path(root).expanduser().resolve()
    subject = normalize_subject_index(subject_index)
    manifest = read_manifest(root)
    rows = manifest.loc[manifest["subject_index"].astype(str).eq(subject)]
    if len(rows) != 1:
        raise KeyError(f"subject_index {subject!r} is not present exactly once in manifest.tsv.")
    row = rows.iloc[0]
    output: dict[str, Path | None] = {}
    for column in PATH_COLUMNS.values():
        raw = str(row[column]).strip()
        if not raw:
            output[column] = None
            continue
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Manifest paths must remain inside dataset root: {raw}")
        output[column] = root.joinpath(*relative.parts)
    return output


def load_subject_epochs_and_metadata(
    root: str | Path,
    subject_index: str | int,
    *,
    preload: bool,
) -> tuple[mne.Epochs, pd.DataFrame]:
    """Load one subject, validate the mirror, and merge artifact fields."""

    subject = normalize_subject_index(subject_index)
    paths = subject_paths(root, subject)
    epochs_path = required_file(paths["epochs_path"], "epochs")
    events_path = required_file(paths["events_path"], "events")
    epochs = mne.read_epochs(epochs_path, preload=preload, verbose="ERROR")
    events = pd.read_csv(events_path, sep="\t", dtype={"subject_index": "string"})
    events = normalize_identity(events, context="events.tsv")
    if set(events["subject_index"]) != {subject}:
        raise ValueError(f"events.tsv contains subject_index values other than {subject!r}.")
    if len(events) != len(epochs):
        raise ValueError(
            f"events.tsv has {len(events)} rows but epochs has {len(epochs)} for {subject!r}."
        )
    if epochs.metadata is None:
        raise ValueError(f"Epochs file has no identity metadata: {epochs_path}")
    validate_metadata_mirror(epochs.metadata, events)
    with mne.use_log_level("ERROR"):
        epochs.metadata = events
    artifacts_path = paths["artifacts_path"]
    metadata = events
    if artifacts_path is not None:
        artifacts = pd.read_csv(
            required_file(artifacts_path, "artifacts"),
            sep="\t",
            dtype={"subject_index": "string"},
        )
        metadata = merge_artifact_metadata(events, artifacts)
    return epochs, metadata


def load_subject_metadata(root: str | Path, subject_index: str | int) -> pd.DataFrame:
    """Load events metadata and optional artifact fields without signal data."""

    paths = subject_paths(root, subject_index)
    events = pd.read_csv(
        required_file(paths["events_path"], "events"),
        sep="\t",
        dtype={"subject_index": "string"},
    )
    events = normalize_identity(events, context="events.tsv")
    if paths["artifacts_path"] is None:
        return events
    artifacts = pd.read_csv(
        required_file(paths["artifacts_path"], "artifacts"),
        sep="\t",
        dtype={"subject_index": "string"},
    )
    return merge_artifact_metadata(events, artifacts)


def read_json(path: str | Path, *, missing_ok: bool = False) -> dict[str, object]:
    """Read one JSON object."""

    path = Path(path)
    if not path.exists():
        if missing_ok:
            return {}
        raise FileNotFoundError(path)
    with path.open("r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def required_file(path: Path | None, label: str) -> Path:
    """Return an existing required path."""

    if path is None or not path.is_file():
        raise FileNotFoundError(f"Subject {label} file not found: {path}")
    return path
