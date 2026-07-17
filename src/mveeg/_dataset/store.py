"""Atomic prepared-dataset publication."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .._provenance import jsonable
from .manifest import (
    DATASET_SCHEMA_VERSION,
    MANIFEST_COLUMNS,
    normalize_subject_index,
    read_json,
    read_manifest,
    relative_paths,
    single_value,
    validate_entity,
)

RECOMPUTE_VALUES = {"never", "changed", "all"}


class DatasetBuilder:
    """Coordinate recompute decisions and atomic dataset publication."""

    def __init__(
        self,
        root: str | Path,
        *,
        task: str,
        stage: str,
        pipeline_fingerprint: str,
        pipeline_spec: Mapping[str, object],
        recompute: str,
        subject_indices: Sequence[str | int],
        complete_subject_set: bool,
    ):
        if recompute not in RECOMPUTE_VALUES:
            raise ValueError(f"recompute must be one of {sorted(RECOMPUTE_VALUES)}.")
        self.root = Path(root).expanduser().resolve()
        recover_transaction(self.root)
        self.task = validate_entity(task, "task")
        self.stage = validate_entity(stage, "stage")
        self.pipeline_fingerprint = pipeline_fingerprint
        self.pipeline_spec = dict(pipeline_spec)
        self.recompute = recompute
        self.subject_indices = tuple(normalize_subject_index(value) for value in subject_indices)
        self.complete_subject_set = complete_subject_set
        self.manifest = read_manifest(self.root, missing_ok=True)
        self._had_manifest = len(self.manifest) > 0
        self._stage_root: Path | None = None
        self._delete_relatives: set[Path] = set()
        self.summary: list[dict[str, object]] = []
        self._preserve_root_metadata = False
        self._force_all = False

        if len(self.manifest):
            if single_value(self.manifest, "task") != self.task:
                raise ValueError("A dataset root cannot mix task labels.")
            if single_value(self.manifest, "stage") != self.stage:
                raise ValueError("A dataset root cannot mix processing stages.")
            existing = set(self.manifest["subject_index"].astype(str))
            requested = set(self.subject_indices)
            if self.complete_subject_set and not existing.issubset(requested):
                absent = sorted(existing - requested)
                raise RuntimeError(
                    "The existing dataset contains subjects outside the current cohort: "
                    f"{absent}. Use a new output directory."
                )
        provenance = read_json(self.root / "provenance.json", missing_ok=True)
        self._gaze_geometry = provenance.get("gaze_geometry")
        manifest_fingerprint = (
            single_value(self.manifest, "pipeline_fingerprint") if len(self.manifest) else None
        )
        previous = provenance.get("pipeline_fingerprint") or manifest_fingerprint
        if previous and manifest_fingerprint and previous != manifest_fingerprint:
            raise ValueError("Manifest and provenance contain different pipeline fingerprints.")
        if previous and previous != self.pipeline_fingerprint:
            self._handle_global_change()

    def should_write(self, subject_index: str | int, input_fingerprint: str) -> bool:
        """Return whether one subject must be written under the recompute policy."""

        subject = normalize_subject_index(subject_index)
        previous = self.manifest.loc[self.manifest["subject_index"].astype(str).eq(subject)]
        if len(previous) == 0:
            if self._preserve_root_metadata:
                raise RuntimeError(
                    "Pipeline configuration changed, so recompute='never' cannot add subjects."
                )
            return True
        for column in ("epochs_path", "events_path", "eeg_json_path"):
            saved_path = self.root / str(previous.iloc[0][column])
            if not saved_path.exists():
                raise FileNotFoundError(
                    f"Manifest lists subject {subject}, but {column} is missing: {saved_path}."
                )
        previous_fingerprint = str(previous.iloc[0]["input_fingerprint"])
        if self.recompute == "never":
            if previous_fingerprint != input_fingerprint:
                warnings.warn(
                    f"Input changed for subject {subject}, but recompute='never' reuses it.",
                    stacklevel=2,
                )
            return False
        if self.recompute == "all" or self._force_all:
            return True
        return previous_fingerprint != input_fingerprint

    def record_reused(self, subject_index: str | int) -> None:
        """Record an existing subject reuse."""

        self.summary.append(
            {"subject_index": normalize_subject_index(subject_index), "status": "reused"}
        )

    def saved_input_fingerprint(self, subject_index: str | int) -> str | None:
        """Return the prior subject fingerprint without making a recompute decision."""

        subject = normalize_subject_index(subject_index)
        rows = self.manifest.loc[self.manifest["subject_index"].astype(str).eq(subject)]
        if len(rows) == 0:
            return None
        return str(rows.iloc[0]["input_fingerprint"])

    @property
    def working_root(self) -> Path:
        """Return the unpublished root for companion outputs."""

        return self._write_root()

    def remove_output(self, path: str | Path) -> None:
        """Remove a root-contained output in the same publication transaction."""

        candidate = Path(path)
        relative = candidate.relative_to(self.root) if candidate.is_absolute() else candidate
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("remove_output path must remain inside the dataset root.")
        if self._had_manifest:
            self._delete_relatives.add(relative)
        else:
            (self.root / relative).unlink(missing_ok=True)

    def write_subject(
        self,
        subject_index: str | int,
        epochs: mne.BaseEpochs,
        *,
        input_fingerprint: str,
    ) -> None:
        """Stage one subject and replace its manifest row."""

        subject = normalize_subject_index(subject_index)
        relative = relative_paths(subject, self.task, self.stage)
        write_root = self._write_root()
        absolute = {name: write_root / value for name, value in relative.items()}
        absolute["epochs_path"].parent.mkdir(parents=True, exist_ok=True)
        write_epochs_atomic(epochs, absolute["epochs_path"])
        write_table_atomic(epochs.metadata, absolute["events_path"])
        channel_types = epochs.get_channel_types()
        write_json_atomic(
            {
                "SamplingFrequency": float(epochs.info["sfreq"]),
                "EEGChannelCount": int(sum(kind == "eeg" for kind in channel_types)),
                "ChannelCount": len(epochs.ch_names),
                "EpochCount": len(epochs),
                "EpochTimeWindow": [float(epochs.tmin), float(epochs.tmax)],
                "ChannelNames": list(epochs.ch_names),
                "ChannelTypes": list(channel_types),
            },
            absolute["eeg_json_path"],
        )
        self.remove_output(self.root / relative["artifacts_path"])
        row = {
            "subject_index": subject,
            "task": self.task,
            "stage": self.stage,
            "epochs_path": relative["epochs_path"],
            "events_path": relative["events_path"],
            "eeg_json_path": relative["eeg_json_path"],
            "artifacts_path": "",
            "input_fingerprint": input_fingerprint,
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "n_epochs": len(epochs),
            "n_channels": len(epochs.ch_names),
            "sampling_rate": float(epochs.info["sfreq"]),
            "tmin": float(epochs.tmin),
            "tmax": float(epochs.tmax),
        }
        self.manifest = self.manifest.loc[~self.manifest["subject_index"].astype(str).eq(subject)]
        self.manifest = pd.concat([self.manifest, pd.DataFrame([row])], ignore_index=True)
        self.summary.append({"subject_index": subject, "status": "written"})

    def finish(self) -> pd.DataFrame:
        """Publish staged changes and return the build summary."""

        if len(self.manifest) == 0:
            raise ValueError("No subjects were available to write or reuse.")
        self.manifest = (
            self.manifest.loc[:, MANIFEST_COLUMNS]
            .sort_values("subject_index", kind="stable")
            .reset_index(drop=True)
        )
        if self._had_manifest and not any(row["status"] == "written" for row in self.summary):
            self.abort()
            return pd.DataFrame(self.summary)
        write_root = self._write_root()
        write_table_atomic(self.manifest, write_root / "manifest.tsv")
        if not self._preserve_root_metadata:
            provenance = {
                "schema_version": DATASET_SCHEMA_VERSION,
                "mveeg_version": version("mveeg"),
                "task": self.task,
                "stage": self.stage,
                "pipeline_fingerprint": self.pipeline_fingerprint,
                "pipeline": self.pipeline_spec,
            }
            if self._gaze_geometry is not None:
                provenance["gaze_geometry"] = self._gaze_geometry
            write_json_atomic(
                {
                    "Name": f"mveeg {self.task} {self.stage}",
                    "DatasetType": "derivative",
                    "Task": self.task,
                    "Stage": self.stage,
                    "GeneratedBy": [{"Name": "mveeg", "Version": version("mveeg")}],
                },
                write_root / "dataset_description.json",
            )
            write_json_atomic(provenance, write_root / "provenance.json")
        if self._stage_root is not None:
            publish_transaction(
                self.root,
                self._stage_root,
                delete_relatives=self._delete_relatives,
            )
            self._stage_root = None
            self._delete_relatives.clear()
        return pd.DataFrame(self.summary)

    def abort(self) -> None:
        """Discard unpublished staged files."""

        if self._stage_root is not None:
            shutil.rmtree(self._stage_root, ignore_errors=True)
            self._stage_root = None
        self._delete_relatives.clear()

    def _handle_global_change(self) -> None:
        if self.recompute == "never":
            self._preserve_root_metadata = True
            warnings.warn(
                "Pipeline configuration changed, but recompute='never' preserves results.",
                stacklevel=2,
            )
            return
        existing = set(self.manifest["subject_index"].astype(str))
        requested = set(self.subject_indices)
        complete = self.complete_subject_set and existing.issubset(requested)
        single = existing.issubset(requested) and len(existing) <= 1
        if not (complete or single):
            raise RuntimeError(
                "A global pipeline change requires rebuilding every existing subject. "
                "Use the dataset pipeline or a new output directory."
            )
        self._force_all = True

    def _write_root(self) -> Path:
        if not self._had_manifest:
            return self.root
        if self._stage_root is None:
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self._stage_root = Path(
                tempfile.mkdtemp(prefix=f".{self.root.name}.mveeg-stage-", dir=self.root.parent)
            )
        return self._stage_root


def fingerprint_epochs(epochs: mne.BaseEpochs) -> str:
    """Fingerprint epoch timing, events, metadata, channels, and signal values."""

    digest = sha256()
    digest.update(
        json.dumps(
            jsonable(
                {
                    "sfreq": float(epochs.info["sfreq"]),
                    "tmin": float(epochs.tmin),
                    "ch_names": epochs.ch_names,
                    "ch_types": epochs.get_channel_types(),
                    "events": epochs.events,
                    "metadata": None
                    if epochs.metadata is None
                    else epochs.metadata.to_dict(orient="list"),
                }
            ),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(np.ascontiguousarray(epochs.get_data(copy=False)).view(np.uint8))
    return digest.hexdigest()


def transaction_marker(root: Path) -> Path:
    return root.parent / f".{root.name}.mveeg-transaction.json"


def publish_transaction(
    root: Path,
    stage_root: Path,
    *,
    delete_relatives: Sequence[Path] = (),
) -> None:
    """Publish staged files with rollback metadata and manifest last."""

    root.parent.mkdir(parents=True, exist_ok=True)
    backup_root = Path(tempfile.mkdtemp(prefix=f".{root.name}.mveeg-backup-", dir=root.parent))
    staged = {path.relative_to(stage_root) for path in stage_root.rglob("*") if path.is_file()}
    deleted = set(delete_relatives).difference(staged)
    paths = list(staged | deleted)
    priority = {"dataset_description.json": 1, "provenance.json": 2, "manifest.tsv": 3}
    paths.sort(key=lambda path: (priority.get(path.as_posix(), 0), path.as_posix()))
    records = [
        {
            "path": path.as_posix(),
            "had_original": (root / path).exists(),
            "action": "replace" if path in staged else "delete",
        }
        for path in paths
    ]
    records_by_path = {record["path"]: record for record in records}
    marker = transaction_marker(root)
    payload = {
        "root": root.as_posix(),
        "stage_root": stage_root.as_posix(),
        "backup_root": backup_root.as_posix(),
        "records": records,
    }
    write_json_atomic(payload, marker)
    try:
        for relative in paths:
            record = records_by_path[relative.as_posix()]
            source = stage_root / relative
            target = root / relative
            backup = backup_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, backup)
            if record["action"] == "replace":
                os.replace(source, target)
    except Exception:
        rollback_transaction(payload)
        raise
    marker.unlink(missing_ok=True)
    shutil.rmtree(backup_root, ignore_errors=True)
    shutil.rmtree(stage_root, ignore_errors=True)


def recover_transaction(root: Path) -> None:
    """Roll back an interrupted publication before reading a dataset."""

    marker = transaction_marker(root)
    if not marker.exists():
        return
    payload = read_json(marker)
    if Path(str(payload.get("root", ""))).resolve() != root.resolve():
        raise RuntimeError(f"Transaction marker does not belong to dataset root {root}.")
    rollback_transaction(payload)


def rollback_transaction(payload: Mapping[str, object]) -> None:
    """Restore files described by an interrupted transaction marker."""

    root = Path(str(payload["root"]))
    stage_root = Path(str(payload["stage_root"]))
    backup_root = Path(str(payload["backup_root"]))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("Dataset transaction records must be a list.")
    for record in reversed(records):
        if not isinstance(record, Mapping):
            raise ValueError("Dataset transaction record must be a mapping.")
        relative = Path(str(record["path"]))
        target = root / relative
        source = stage_root / relative
        backup = backup_root / relative
        if bool(record["had_original"]):
            if backup.exists():
                target.unlink(missing_ok=True)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, target)
        elif not source.exists():
            target.unlink(missing_ok=True)
    transaction_marker(root).unlink(missing_ok=True)
    shutil.rmtree(backup_root, ignore_errors=True)
    shutil.rmtree(stage_root, ignore_errors=True)


def write_epochs_atomic(epochs: mne.BaseEpochs, path: Path) -> None:
    """Save epochs through a same-directory temporary FIF."""

    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}-", suffix="-epo.fif", delete=False
    )
    handle.close()
    temporary = Path(handle.name)
    try:
        epochs.save(temporary, overwrite=True, verbose="ERROR")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_table_atomic(table: pd.DataFrame | None, path: Path) -> None:
    """Write TSV through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame() if table is None else table
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tsv", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, sep="\t", index=False, na_rep="n/a")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(value: Mapping[str, object], path: Path) -> None:
    """Write a JSON object through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json", mode="w", delete=False
    ) as handle:
        json.dump(jsonable(value), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
