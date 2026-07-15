"""Standard prepared-dataset storage and lightweight access."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import warnings

import mne
import numpy as np
import pandas as pd

from .._shared.fingerprints import fingerprint, fingerprint_files, jsonable as _jsonable
from .._shared.metadata import validate_metadata_mirror
from .gaze import normalize_gaze_geometry


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
RECOMPUTE_VALUES = {"never", "changed", "all"}


class DatasetPipeline:
    """Lightweight handle for a prepared or preprocessed dataset.

    Parameters
    ----------
    root : path-like
        Dataset root containing ``manifest.tsv``.
    run_summary : pandas.DataFrame | None
        Optional status rows from the build that returned this handle.
    """

    def __init__(self, root: str | Path, *, run_summary: pd.DataFrame | None = None):
        """Open a dataset without loading any subject signal data."""
        self.root = Path(root).expanduser().resolve()
        _recover_transaction(self.root)
        self.run_summary = run_summary
        self._manifest = _read_manifest(self.root)
        if len(self._manifest) == 0:
            raise ValueError(f"Dataset manifest is empty: {self.root / 'manifest.tsv'}.")
        provenance = _read_json(self.root / "provenance.json")
        description = _read_json(self.root / "dataset_description.json")
        manifest_fingerprint = _single_value(self._manifest, "pipeline_fingerprint")
        if provenance.get("pipeline_fingerprint") != manifest_fingerprint:
            raise ValueError("Manifest and provenance contain different pipeline fingerprints.")
        if provenance.get("task") != self.task or provenance.get("stage") != self.stage:
            raise ValueError("Manifest and provenance disagree on task or processing stage.")
        if description.get("Task") != self.task or description.get("Stage") != self.stage:
            raise ValueError("Manifest and dataset_description disagree on task or processing stage.")

    @property
    def manifest(self) -> pd.DataFrame:
        """Return a copy of the dataset manifest."""
        return self._manifest.copy()

    @property
    def subject_indices(self) -> tuple[str, ...]:
        """Return subject indices in manifest order."""
        return tuple(self._manifest["subject_index"].astype(str))

    @property
    def task(self) -> str:
        """Return the dataset's single task label."""
        return _single_value(self._manifest, "task")

    @property
    def stage(self) -> str:
        """Return the dataset's single processing stage."""
        return _single_value(self._manifest, "stage")

    def refresh(self) -> DatasetPipeline:
        """Reload the manifest after another process updates the dataset."""
        self._manifest = _read_manifest(self.root)
        return self

    def path_for_subject(self, subject_index: str | int, kind: str = "epochs") -> Path:
        """Return one subject file path from the manifest.

        ``kind`` is one of ``epochs``, ``events``, ``eeg_json``, or
        ``artifacts``. Before artifact labeling, the expected standard artifact
        path is returned even though its manifest field is blank.
        """
        if kind not in PATH_COLUMNS:
            raise ValueError(f"Unknown dataset path kind {kind!r}; choose from {sorted(PATH_COLUMNS)}.")
        row = self._subject_row(subject_index)
        relative = str(row[PATH_COLUMNS[kind]])
        if kind == "artifacts" and relative == "":
            relative = _relative_paths(str(row["subject_index"]), self.task, self.stage)["artifacts_path"]
        return self.root / relative

    def subject_paths(self, subject_index: str | int) -> dict[str, Path]:
        """Return all standard paths for one subject."""
        return {kind: self.path_for_subject(subject_index, kind) for kind in PATH_COLUMNS}

    def load_epochs(self, subject_index: str | int, *, preload: bool = True) -> mne.Epochs:
        """Load one subject and validate stable epoch identity."""
        path = self.path_for_subject(subject_index, "epochs")
        if not path.exists():
            raise FileNotFoundError(f"Epoch file listed in manifest does not exist: {path}.")
        epochs = mne.read_epochs(path, preload=preload, verbose="ERROR")
        if epochs.metadata is None:
            raise ValueError(f"Saved epochs have no identity metadata: {path}.")
        required = {"subject_index", "epoch_index"}
        if not required.issubset(epochs.metadata.columns):
            raise ValueError(f"Saved epochs are missing identity columns: {sorted(required)}.")
        expected_subject = _normalize_subject_index(subject_index)
        observed = set(epochs.metadata["subject_index"].astype(str))
        if observed != {expected_subject}:
            raise ValueError(f"Saved subject identity {observed} does not match {expected_subject!r}.")
        epoch_index = epochs.metadata["epoch_index"].to_numpy(dtype=int)
        if not np.array_equal(epoch_index, np.arange(len(epochs), dtype=int)):
            raise ValueError("Saved epoch_index must be unique, zero-based, and sequential.")
        events_path = self.path_for_subject(subject_index, "events")
        if not events_path.exists():
            raise FileNotFoundError(f"Events file listed in manifest does not exist: {events_path}.")
        events_metadata = pd.read_csv(
            events_path,
            sep="\t",
            dtype={"subject_index": "string"},
        )
        validate_metadata_mirror(epochs.metadata, events_metadata)
        with mne.use_log_level("ERROR"):
            epochs.metadata = events_metadata
        return epochs

    def configure_gaze(
        self,
        *,
        viewing_distance_cm: float,
        screen_width_cm: float,
        screen_width_px: int,
    ) -> DatasetPipeline:
        """Persist gaze geometry for later degree-based quality rules."""
        if self.stage != "prepared":
            raise ValueError("configure_gaze is only available for prepared datasets.")
        geometry = normalize_gaze_geometry(
            viewing_distance_cm=viewing_distance_cm,
            screen_width_cm=screen_width_cm,
            screen_width_px=screen_width_px,
        )
        path = self.root / "provenance.json"
        provenance = _read_json(path)
        if provenance.get("gaze_geometry") == geometry:
            return self
        provenance["gaze_geometry"] = geometry
        _write_json_atomic(provenance, path)
        return self

    def label_artifacts(
        self,
        *,
        reject: Mapping[str, object],
        review: Mapping[str, object],
        ignore_channels: Sequence[str] = (),
    ) -> DatasetPipeline:
        """Create or refresh automatic artifact sidecars for this dataset."""
        from .processing import label_artifacts

        return label_artifacts(
            self,
            reject=reject,
            review=review,
            ignore_channels=ignore_channels,
        )

    def review_artifacts(
        self,
        *,
        subject_index: str | int,
        group_by: str | None = None,
        label: object | None = None,
        time_window: tuple[float | None, float | None] | None = None,
        hide_channels: Sequence[str] = (),
        scalings: Mapping[str, float] | None = None,
    ) -> None:
        """Open a blocking artifact review session and return after it closes.

        Parameters
        ----------
        subject_index : str or int
            Subject whose saved epochs and artifact sidecar are reviewed.
        group_by, label : object or None
            Both select one review group, or both are omitted to review all
            epochs.
        time_window : tuple or None
            Optional start and end times displayed from each epoch.
        hide_channels : sequence of str
            Saved channels omitted only from the review display.
        scalings : mapping or None
            Optional display multipliers keyed by MNE channel type.

        Returns
        -------
        None
            Closing the window releases preloaded epochs and GUI resources.

        Notes
        -----
        Pressing ``w`` commits visited decisions. Closing without ``w``
        discards pending edits.
        """
        from .review import ReviewSession, open_review_figure

        subject = _normalize_subject_index(subject_index)
        artifact_path = self.path_for_subject(subject, "artifacts")
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"No artifact sidecar exists for subject {subject}: {artifact_path}."
            )
        epochs = self.load_epochs(subject, preload=True)
        session = ReviewSession.from_path(
            artifact_path,
            subject_index=subject,
            metadata=epochs.metadata,
            group_by=group_by,
            label=label,
        )
        open_review_figure(
            session,
            epochs,
            time_window=time_window,
            hide_channels=hide_channels,
            scalings=scalings,
        )

    def _subject_row(self, subject_index: str | int) -> pd.Series:
        """Return one manifest row after normalizing its subject index."""
        subject = _normalize_subject_index(subject_index)
        rows = self._manifest.loc[self._manifest["subject_index"].astype(str).eq(subject)]
        if len(rows) != 1:
            raise KeyError(f"Subject {subject!r} is not present exactly once in the manifest.")
        return rows.iloc[0]


def open_pipeline(root: str | Path) -> DatasetPipeline:
    """Open a standard mveeg dataset without loading its epochs."""
    return DatasetPipeline(root)


@dataclass
class DatasetBuilder:
    """Coordinate recompute decisions and root metadata for one build."""

    root: Path
    task: str
    stage: str
    pipeline_fingerprint: str
    pipeline_spec: Mapping[str, object]
    recompute: str
    subject_indices: tuple[str, ...]
    complete_subject_set: bool

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
        """Validate the target dataset and establish recompute policy."""
        if recompute not in RECOMPUTE_VALUES:
            raise ValueError(f"recompute must be one of {sorted(RECOMPUTE_VALUES)}.")
        self.root = Path(root).expanduser().resolve()
        _recover_transaction(self.root)
        self.task = _validate_entity(task, "task")
        self.stage = _validate_entity(stage, "stage")
        self.pipeline_fingerprint = pipeline_fingerprint
        self.pipeline_spec = dict(pipeline_spec)
        self.recompute = recompute
        self.subject_indices = tuple(_normalize_subject_index(value) for value in subject_indices)
        self.complete_subject_set = complete_subject_set
        self.manifest = _read_manifest(self.root, missing_ok=True)
        self._had_manifest = len(self.manifest) > 0
        self._stage_root: Path | None = None
        self._delete_relatives: set[Path] = set()
        self.summary: list[dict[str, object]] = []
        self._preserve_root_metadata = False
        self._force_all = False

        if len(self.manifest):
            if _single_value(self.manifest, "task") != self.task:
                raise ValueError("A dataset root cannot mix task labels.")
            if _single_value(self.manifest, "stage") != self.stage:
                raise ValueError("A dataset root cannot mix processing stages.")
            existing_subjects = set(self.manifest["subject_index"].astype(str))
            requested_subjects = set(self.subject_indices)
            if self.complete_subject_set and not existing_subjects.issubset(requested_subjects):
                absent = sorted(existing_subjects - requested_subjects)
                raise RuntimeError(
                    "The existing dataset contains subjects outside the current cohort: "
                    f"{absent}. Use a new output directory instead of silently removing them."
                )
        provenance = _read_json(self.root / "provenance.json", missing_ok=True)
        self._gaze_geometry = provenance.get("gaze_geometry")
        if self._gaze_geometry is not None:
            if self.stage != "prepared" or not isinstance(self._gaze_geometry, Mapping):
                raise ValueError("gaze_geometry is only valid in prepared dataset provenance.")
            self._gaze_geometry = normalize_gaze_geometry(**self._gaze_geometry)
        manifest_fingerprint = (
            _single_value(self.manifest, "pipeline_fingerprint") if len(self.manifest) else None
        )
        previous_fingerprint = provenance.get("pipeline_fingerprint") or manifest_fingerprint
        if (
            previous_fingerprint
            and manifest_fingerprint
            and previous_fingerprint != manifest_fingerprint
        ):
            raise ValueError("Manifest and provenance contain different pipeline fingerprints.")
        if previous_fingerprint and previous_fingerprint != self.pipeline_fingerprint:
            self._handle_global_change()

    def should_write(self, subject_index: str | int, input_fingerprint: str) -> bool:
        """Return whether one subject must be computed under the chosen policy."""
        subject = _normalize_subject_index(subject_index)
        previous = self.manifest.loc[self.manifest["subject_index"].astype(str).eq(subject)]
        if len(previous) == 0:
            if self._preserve_root_metadata:
                raise RuntimeError(
                    "Pipeline configuration changed, so recompute='never' cannot add new subjects "
                    "to the existing dataset."
                )
            return True
        for column in ("epochs_path", "events_path", "eeg_json_path"):
            saved_path = self.root / str(previous.iloc[0][column])
            if not saved_path.exists():
                raise FileNotFoundError(
                    f"Manifest lists subject {subject}, but {column} is missing: {saved_path}."
                )
        if self.recompute == "never":
            if str(previous.iloc[0]["input_fingerprint"]) != input_fingerprint:
                warnings.warn(
                    f"Input changed for subject {subject}, but recompute='never' reuses the saved result.",
                    stacklevel=2,
                )
            return False
        if self.recompute == "all" or self._force_all:
            return True
        return str(previous.iloc[0]["input_fingerprint"]) != input_fingerprint

    def record_reused(self, subject_index: str | int) -> None:
        """Record that an existing subject was reused."""
        self.summary.append({"subject_index": _normalize_subject_index(subject_index), "status": "reused"})

    @property
    def working_root(self) -> Path:
        """Return the unpublished root where companion outputs must be written."""
        return self._write_root()

    def remove_output(self, path: str | Path) -> None:
        """Remove one root-contained output as part of the same publish transaction."""
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
        epochs: mne.Epochs,
        *,
        input_fingerprint: str,
    ) -> None:
        """Write one subject and replace its manifest row."""
        subject = _normalize_subject_index(subject_index)
        relative = _relative_paths(subject, self.task, self.stage)
        write_root = self._write_root()
        absolute = {name: write_root / value for name, value in relative.items()}
        absolute["epochs_path"].parent.mkdir(parents=True, exist_ok=True)
        _write_epochs_atomic(epochs, absolute["epochs_path"])
        _write_table_atomic(epochs.metadata, absolute["events_path"])
        channel_types = epochs.get_channel_types()
        sidecar = {
            "SamplingFrequency": float(epochs.info["sfreq"]),
            "EEGChannelCount": int(sum(kind == "eeg" for kind in channel_types)),
            "ChannelCount": len(epochs.ch_names),
            "EpochCount": len(epochs),
            "EpochTimeWindow": [float(epochs.tmin), float(epochs.tmax)],
            "ChannelNames": list(epochs.ch_names),
            "ChannelTypes": list(channel_types),
        }
        _write_json_atomic(sidecar, absolute["eeg_json_path"])
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
        self.manifest = self.manifest.loc[
            ~self.manifest["subject_index"].astype(str).eq(subject)
        ]
        self.manifest = pd.concat([self.manifest, pd.DataFrame([row])], ignore_index=True)
        self.summary.append({"subject_index": subject, "status": "written"})

    def finish(self) -> DatasetPipeline:
        """Atomically update root files and return a lightweight dataset handle."""
        if len(self.manifest) == 0:
            raise ValueError("No subjects were available to write or reuse.")
        self.manifest = self.manifest[MANIFEST_COLUMNS].sort_values(
            "subject_index", kind="stable"
        ).reset_index(drop=True)
        wrote_subject = any(row["status"] == "written" for row in self.summary)
        if self._had_manifest and not wrote_subject:
            self.abort()
            return DatasetPipeline(self.root, run_summary=pd.DataFrame(self.summary))
        write_root = self._write_root()
        _write_table_atomic(self.manifest, write_root / "manifest.tsv")
        if not self._preserve_root_metadata:
            provenance = {
                "schema_version": 1,
                "mveeg_version": _mveeg_version(),
                "task": self.task,
                "stage": self.stage,
                "pipeline_fingerprint": self.pipeline_fingerprint,
                "pipeline": self.pipeline_spec,
            }
            if self._gaze_geometry is not None:
                provenance["gaze_geometry"] = self._gaze_geometry
            _write_json_atomic(
                {
                    "Name": f"mveeg {self.task} {self.stage}",
                    "DatasetType": "derivative",
                    "Task": self.task,
                    "Stage": self.stage,
                    "GeneratedBy": [{"Name": "mveeg", "Version": _mveeg_version()}],
                },
                write_root / "dataset_description.json",
            )
            _write_json_atomic(
                provenance,
                write_root / "provenance.json",
            )
        if self._stage_root is not None:
            _publish_transaction(
                self.root,
                self._stage_root,
                delete_relatives=self._delete_relatives,
            )
            self._stage_root = None
            self._delete_relatives.clear()
        return DatasetPipeline(self.root, run_summary=pd.DataFrame(self.summary))

    def abort(self) -> None:
        """Discard unpublished staged files after a failed build."""
        if self._stage_root is not None:
            shutil.rmtree(self._stage_root, ignore_errors=True)
            self._stage_root = None
        self._delete_relatives.clear()

    def _handle_global_change(self) -> None:
        """Apply the global recompute rule after a pipeline change."""
        if self.recompute == "never":
            self._preserve_root_metadata = True
            warnings.warn(
                "Pipeline configuration changed, but recompute='never' preserves existing results.",
                stacklevel=2,
            )
            return
        existing = set(self.manifest["subject_index"].astype(str))
        requested = set(self.subject_indices)
        can_replace = self.complete_subject_set and existing.issubset(requested)
        can_replace_single = existing.issubset(requested) and len(existing) <= 1
        if not (can_replace or can_replace_single):
            raise RuntimeError(
                "A global pipeline change requires rebuilding every existing subject. "
                "Use the dataset-level pipeline or a new output directory."
            )
        self._force_all = True

    def _write_root(self) -> Path:
        """Return a staging root when an existing dataset is being updated."""
        if not self._had_manifest:
            return self.root
        if self._stage_root is None:
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self._stage_root = Path(
                tempfile.mkdtemp(prefix=f".{self.root.name}.mveeg-stage-", dir=self.root.parent)
            )
        return self._stage_root


def fingerprint_epochs(epochs: mne.Epochs) -> str:
    """Fingerprint in-memory epochs, events, metadata, timing, and signal values."""
    digest = sha256()
    digest.update(
        json.dumps(
            _jsonable(
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
    for epoch in epochs.get_data(copy=False):
        digest.update(np.ascontiguousarray(epoch).view(np.uint8))
    return digest.hexdigest()


def _normalize_subject_index(value: str | int) -> str:
    """Normalize common subject-folder prefixes to a stable index."""
    subject = str(value).strip()
    if subject.startswith("sub-"):
        subject = subject[4:]
    elif subject.startswith("sub") and subject[3:].isdigit():
        subject = subject[3:]
    return _validate_entity(subject, "subject_index")


def _validate_entity(value: str, name: str) -> str:
    """Validate one filename entity against the package's compact convention."""
    value = str(value)
    if not value or re.fullmatch(r"[A-Za-z0-9]+", value) is None:
        raise ValueError(f"{name} must contain only letters and numbers, got {value!r}.")
    return value


def _relative_paths(subject: str, task: str, stage: str) -> dict[str, str]:
    """Build all standard root-relative POSIX paths for one subject."""
    stem = f"sub-{subject}_task-{task}_desc-{stage}"
    directory = Path(f"sub-{subject}") / "eeg"
    return {
        "epochs_path": (directory / f"{stem}_epo.fif").as_posix(),
        "events_path": (directory / f"sub-{subject}_task-{task}_events.tsv").as_posix(),
        "eeg_json_path": (directory / f"sub-{subject}_task-{task}_eeg.json").as_posix(),
        "artifacts_path": (directory / f"sub-{subject}_task-{task}_desc-artifacts.tsv").as_posix(),
    }


def _read_manifest(root: Path, *, missing_ok: bool = False) -> pd.DataFrame:
    """Read a manifest with stable string identifiers and ordered columns."""
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
    return manifest[MANIFEST_COLUMNS]


def _single_value(table: pd.DataFrame, column: str) -> str:
    """Return one required dataset-wide manifest value."""
    values = table[column].astype(str).unique()
    if len(values) != 1:
        raise ValueError(f"Manifest must contain one {column!r} value, found {list(values)}.")
    return str(values[0])


def _transaction_marker(root: Path) -> Path:
    """Return the deterministic recovery marker for one dataset root."""
    return root.parent / f".{root.name}.mveeg-transaction.json"


def _publish_transaction(
    root: Path,
    stage_root: Path,
    *,
    delete_relatives: Sequence[Path] = (),
) -> None:
    """Publish staged files with rollback metadata and manifest last."""
    root.parent.mkdir(parents=True, exist_ok=True)
    backup_root = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.mveeg-backup-", dir=root.parent)
    )
    staged_paths = {
        path.relative_to(stage_root) for path in stage_root.rglob("*") if path.is_file()
    }
    delete_paths = set(delete_relatives).difference(staged_paths)
    relative_paths = list(staged_paths | delete_paths)
    priority = {"dataset_description.json": 1, "provenance.json": 2, "manifest.tsv": 3}
    relative_paths.sort(key=lambda path: (priority.get(path.as_posix(), 0), path.as_posix()))
    records = [
        {
            "path": path.as_posix(),
            "had_original": (root / path).exists(),
            "action": "replace" if path in staged_paths else "delete",
        }
        for path in relative_paths
    ]
    records_by_path = {record["path"]: record for record in records}
    marker = _transaction_marker(root)
    payload = {
        "root": root.as_posix(),
        "stage_root": stage_root.as_posix(),
        "backup_root": backup_root.as_posix(),
        "records": records,
    }
    _write_json_atomic(payload, marker)
    try:
        for relative in relative_paths:
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
        _rollback_transaction(payload)
        raise
    marker.unlink(missing_ok=True)
    shutil.rmtree(backup_root, ignore_errors=True)
    shutil.rmtree(stage_root, ignore_errors=True)


def _recover_transaction(root: Path) -> None:
    """Roll back an interrupted publish before reading or updating a dataset."""
    marker = _transaction_marker(root)
    if not marker.exists():
        return
    payload = _read_json(marker)
    if Path(str(payload.get("root", ""))).resolve() != root.resolve():
        raise RuntimeError(f"Transaction marker does not belong to dataset root {root}.")
    _rollback_transaction(payload)


def _rollback_transaction(payload: Mapping[str, object]) -> None:
    """Restore files described by one interrupted transaction marker."""
    root = Path(str(payload["root"]))
    stage_root = Path(str(payload["stage_root"]))
    backup_root = Path(str(payload["backup_root"]))
    records = payload.get("records", [])
    for record in reversed(records):
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
    _transaction_marker(root).unlink(missing_ok=True)
    shutil.rmtree(backup_root, ignore_errors=True)
    shutil.rmtree(stage_root, ignore_errors=True)


def _write_epochs_atomic(epochs: mne.Epochs, path: Path) -> None:
    """Save epochs through a same-directory temporary FIF."""
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}-", suffix="-epo.fif", delete=False
    )
    handle.close()
    temp_path = Path(handle.name)
    try:
        epochs.save(temp_path, overwrite=True, verbose="ERROR")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _write_table_atomic(table: pd.DataFrame | None, path: Path) -> None:
    """Write one TSV through a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame() if table is None else table
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tsv", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        frame.to_csv(temp_path, sep="\t", index=False, na_rep="n/a")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _write_json_atomic(value: Mapping[str, object], path: Path) -> None:
    """Write one JSON object through a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".json", mode="w", delete=False) as handle:
        json.dump(_jsonable(value), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _read_json(path: Path, *, missing_ok: bool = False) -> dict[str, object]:
    """Read one JSON object, optionally returning an empty object when absent."""
    if not path.exists():
        if missing_ok:
            return {}
        raise FileNotFoundError(path)
    with path.open("r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _mveeg_version() -> str:
    """Return the installed package version for provenance."""
    try:
        return version("mveeg")
    except PackageNotFoundError:
        return "0.3.0"
