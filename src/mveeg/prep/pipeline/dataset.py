"""Public handle for prepared and preprocessed datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from ..._dataset.manifest import (
    DATASET_SCHEMA_VERSION,
    PATH_COLUMNS,
    normalize_subject_index,
    read_json,
    read_manifest,
    relative_paths,
    single_value,
)
from ..._dataset.metadata import validate_metadata_mirror
from ..._dataset.store import recover_transaction, write_json_atomic
from ..gaze import normalize_gaze_geometry


class DatasetPipeline:
    """Lightweight handle for a manifest-backed mveeg dataset."""

    def __init__(self, root: str | Path, *, run_summary: pd.DataFrame | None = None):
        self.root = Path(root).expanduser().resolve()
        recover_transaction(self.root)
        self.run_summary = run_summary
        self._manifest = read_manifest(self.root)
        if len(self._manifest) == 0:
            raise ValueError(f"Dataset manifest is empty: {self.root / 'manifest.tsv'}.")
        provenance = read_json(self.root / "provenance.json")
        description = read_json(self.root / "dataset_description.json")
        schema_version = provenance.get("schema_version")
        if schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"Dataset schema {schema_version!r} is unsupported; regenerate with "
                f"mveeg schema {DATASET_SCHEMA_VERSION}."
            )
        manifest_fingerprint = single_value(self._manifest, "pipeline_fingerprint")
        if provenance.get("pipeline_fingerprint") != manifest_fingerprint:
            raise ValueError("Manifest and provenance contain different pipeline fingerprints.")
        if provenance.get("task") != self.task or provenance.get("stage") != self.stage:
            raise ValueError("Manifest and provenance disagree on task or processing stage.")
        if description.get("Task") != self.task or description.get("Stage") != self.stage:
            raise ValueError("Manifest and dataset_description disagree on task or stage.")

    @property
    def manifest(self) -> pd.DataFrame:
        """Return a manifest copy."""

        return self._manifest.copy()

    @property
    def subject_indices(self) -> tuple[str, ...]:
        """Return subject indices in manifest order."""

        return tuple(self._manifest["subject_index"].astype(str))

    @property
    def task(self) -> str:
        """Return the dataset task."""

        return single_value(self._manifest, "task")

    @property
    def stage(self) -> str:
        """Return the processing stage."""

        return single_value(self._manifest, "stage")

    def refresh(self) -> DatasetPipeline:
        """Reload the manifest after an update."""

        self._manifest = read_manifest(self.root)
        return self

    def path_for_subject(self, subject_index: str | int, kind: str = "epochs") -> Path:
        """Return one standard subject path."""

        if kind not in PATH_COLUMNS:
            raise ValueError(f"Unknown dataset path kind {kind!r}; choose {sorted(PATH_COLUMNS)}.")
        row = self._subject_row(subject_index)
        relative = str(row[PATH_COLUMNS[kind]])
        if kind == "artifacts" and not relative:
            relative = relative_paths(str(row["subject_index"]), self.task, self.stage)[
                "artifacts_path"
            ]
        return self.root / relative

    def subject_paths(self, subject_index: str | int) -> dict[str, Path]:
        """Return all standard paths for one subject."""

        return {kind: self.path_for_subject(subject_index, kind) for kind in PATH_COLUMNS}

    def load_epochs(self, subject_index: str | int, *, preload: bool = True) -> mne.Epochs:
        """Load one subject and validate its metadata mirror."""

        subject = normalize_subject_index(subject_index)
        path = self.path_for_subject(subject, "epochs")
        if not path.exists():
            raise FileNotFoundError(f"Epoch file listed in manifest does not exist: {path}.")
        epochs = mne.read_epochs(path, preload=preload, verbose="ERROR")
        if epochs.metadata is None:
            raise ValueError(f"Saved epochs have no identity metadata: {path}.")
        required = {"subject_index", "epoch_index"}
        if not required.issubset(epochs.metadata.columns):
            raise ValueError(f"Saved epochs are missing identity columns: {sorted(required)}.")
        if set(epochs.metadata["subject_index"].astype(str)) != {subject}:
            raise ValueError("Saved subject identity does not match the requested subject.")
        if not np.array_equal(
            epochs.metadata["epoch_index"].to_numpy(dtype=int),
            np.arange(len(epochs), dtype=int),
        ):
            raise ValueError("Saved epoch_index must be unique, zero-based, and sequential.")
        events_path = self.path_for_subject(subject, "events")
        events = pd.read_csv(events_path, sep="\t", dtype={"subject_index": "string"})
        validate_metadata_mirror(epochs.metadata, events)
        with mne.use_log_level("ERROR"):
            epochs.metadata = events
        return epochs

    def configure_gaze(
        self,
        *,
        viewing_distance_cm: float,
        screen_width_cm: float,
        screen_width_px: int,
    ) -> DatasetPipeline:
        """Persist gaze geometry for degree-based quality rules."""

        if self.stage != "prepared":
            raise ValueError("configure_gaze is only available for prepared datasets.")
        geometry = normalize_gaze_geometry(
            viewing_distance_cm=viewing_distance_cm,
            screen_width_cm=screen_width_cm,
            screen_width_px=screen_width_px,
        )
        path = self.root / "provenance.json"
        provenance = read_json(path)
        if provenance.get("gaze_geometry") != geometry:
            provenance["gaze_geometry"] = geometry
            write_json_atomic(provenance, path)
        return self

    def preprocess_epochs(
        self,
        output_dir: str | Path,
        *,
        eligibility: Mapping[str, object],
        autoreject: Mapping[str, object] | None = None,
        recompute: str = "never",
    ) -> DatasetPipeline:
        """Create a preprocessed dataset from this prepared dataset."""

        from ..processing import _preprocess_epochs

        return _preprocess_epochs(
            self,
            output_dir,
            eligibility=eligibility,
            autoreject=autoreject,
            recompute=recompute,
        )

    def label_artifacts(
        self,
        *,
        reject: Mapping[str, object],
        review: Mapping[str, object],
        ignore_channels: Sequence[str] = (),
        recompute: str = "all",
    ) -> DatasetPipeline:
        """Create or refresh artifact sidecars."""

        from ..processing import label_artifacts

        return label_artifacts(
            self,
            reject=reject,
            review=review,
            ignore_channels=ignore_channels,
            recompute=recompute,
        )

    def artifact_counts(self, *, status: str = "final_status") -> pd.DataFrame:
        """Return per-subject counts for automatic or final artifact status."""

        from ..artifacts import read_artifact_table

        if status not in {"initial_status", "final_status"}:
            raise ValueError("status must be 'initial_status' or 'final_status'.")
        rows = []
        for subject in self.subject_indices:
            path = self.path_for_subject(subject, "artifacts")
            if not path.exists():
                raise FileNotFoundError(f"No artifact sidecar exists for {subject}: {path}.")
            counts = read_artifact_table(path)[status].value_counts()
            rows.append(
                {
                    "subject": subject,
                    "accepted": int(counts.get("accepted", 0)),
                    "rejected": int(counts.get("rejected", 0)),
                    "review": int(counts.get("review", 0)),
                }
            )
        return pd.DataFrame(rows, columns=["subject", "accepted", "rejected", "review"])

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
        """Open a blocking Matplotlib artifact-review session."""

        from ..review.figure import open_review_figure
        from ..review.session import ReviewSession, _NoMatchingReviewEpochsError

        subject = normalize_subject_index(subject_index)
        artifact_path = self.path_for_subject(subject, "artifacts")
        if not artifact_path.exists():
            raise FileNotFoundError(f"No artifact sidecar exists for {subject}: {artifact_path}.")
        epochs = self.load_epochs(subject, preload=True)
        try:
            session = ReviewSession.from_path(
                artifact_path,
                subject_index=subject,
                metadata=epochs.metadata,
                group_by=group_by,
                label=label,
            )
        except _NoMatchingReviewEpochsError:
            print(f"No epochs are available for review in {group_by}={label!r}.")
            return
        open_review_figure(
            session,
            epochs,
            time_window=time_window,
            hide_channels=hide_channels,
            scalings=scalings,
        )

    def _subject_row(self, subject_index: str | int) -> pd.Series:
        subject = normalize_subject_index(subject_index)
        rows = self._manifest.loc[self._manifest["subject_index"].astype(str).eq(subject)]
        if len(rows) != 1:
            raise KeyError(f"Subject {subject!r} is not present exactly once in the manifest.")
        return rows.iloc[0]


def open_pipeline(root: str | Path) -> DatasetPipeline:
    """Open an mveeg dataset without loading signal data."""

    return DatasetPipeline(root)
