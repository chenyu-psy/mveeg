"""Subject-level eager pipeline for externally prepared in-memory data."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from . import steps
from .dataset import (
    DatasetBuilder,
    DatasetPipeline,
    _normalize_subject_index,
    fingerprint,
    fingerprint_epochs,
)
from .gaze import normalize_gaze_geometry


class ExternalPipeline:
    """Normalize one subject's in-memory data into an mveeg prepared dataset.

    Unlike :class:`~mveeg.prep.pipeline.RawPipeline`, every method acts
    immediately. Array inputs must use ``epochs, channels, times`` axis order.
    """

    def __init__(self, *, subject_index: str | int, data: np.ndarray | mne.BaseEpochs):
        """Store a copied in-memory array or MNE Epochs object."""
        self.subject_index = _normalize_subject_index(subject_index)
        self._gaze_geometry: dict[str, float | int] | None = None
        if isinstance(data, mne.BaseEpochs):
            self._array: np.ndarray | None = None
            self._epochs: mne.Epochs | None = data.copy().load_data()
            self._history: list[dict[str, object]] = [
                {
                    "step": "input_epochs",
                    "sampling_rate": float(data.info["sfreq"]),
                    "ch_names": list(data.ch_names),
                    "ch_types": data.get_channel_types(),
                    "tmin": float(data.tmin),
                    "tmax": float(data.tmax),
                }
            ]
        else:
            array = np.asarray(data)
            if array.ndim != 3:
                raise ValueError(
                    "External array data must have shape (epochs, channels, times)."
                )
            self._array = array.copy()
            self._epochs = None
            self._history = [{"step": "input_array", "n_channels": array.shape[1], "n_times": array.shape[2]}]

    def configure_gaze(
        self,
        *,
        viewing_distance_cm: float,
        screen_width_cm: float,
        screen_width_px: int,
    ) -> ExternalPipeline:
        """Store gaze geometry for later degree-based quality rules."""
        self._gaze_geometry = normalize_gaze_geometry(
            viewing_distance_cm=viewing_distance_cm,
            screen_width_cm=screen_width_cm,
            screen_width_px=screen_width_px,
        )
        return self

    @property
    def epochs(self) -> mne.Epochs:
        """Return a copy of the current epochs after ``make_epochs``."""
        return self._require_epochs().copy()

    def make_epochs(
        self,
        *,
        sampling_rate: float | None = None,
        ch_names: Sequence[str] | None = None,
        tmin: float = 0.0,
        events: Sequence[int] | np.ndarray | None = None,
        event_id: Mapping[str, int] | None = None,
        ch_types: str | Sequence[str] = "eeg",
        baseline: tuple[float | None, float | None] | None = None,
    ) -> ExternalPipeline:
        """Convert an external epoch array to :class:`mne.EpochsArray`.

        This method is a no-op when the input already was MNE Epochs. Array
        values follow MNE units (EEG in volts, gaze/pupil in their declared
        channel units).
        """
        if self._epochs is not None:
            return self
        assert self._array is not None
        if sampling_rate is None or ch_names is None:
            raise ValueError("Array inputs require sampling_rate and ch_names.")
        sampling_rate = steps._normalize_sampling_rate(sampling_rate)
        assert sampling_rate is not None
        if len(ch_names) != self._array.shape[1]:
            raise ValueError(
                f"ch_names has {len(ch_names)} entries for {self._array.shape[1]} channels."
            )
        mne_events = _external_events(events, self._array.shape[0])
        codes = sorted(set(mne_events[:, 2].astype(int)))
        if event_id is None:
            event_id = {f"event-{code}": code for code in codes}
        elif set(codes).difference(int(value) for value in event_id.values()):
            raise ValueError("event_id does not define every event code in events.")
        info = mne.create_info(
            list(ch_names),
            sfreq=sampling_rate,
            ch_types=ch_types,
        )
        self._epochs = mne.EpochsArray(
            self._array,
            info,
            events=mne_events,
            event_id=dict(event_id),
            tmin=float(tmin),
            baseline=baseline,
            verbose="ERROR",
        )
        self._array = None
        self._history.append(
            {
                "step": "make_epochs",
                "sampling_rate": sampling_rate,
                "ch_names": list(ch_names),
                "ch_types": ch_types,
                "tmin": float(tmin),
                "event_id": dict(event_id),
                "baseline": baseline,
            }
        )
        return self

    def merge_metadata(
        self,
        metadata: pd.DataFrame,
        *,
        epoch_key: str | None = None,
        metadata_key: str | None = None,
    ) -> ExternalPipeline:
        """Merge one table by strict row order or explicit unique keys."""
        self._epochs = steps.merge_metadata(
            self._require_epochs(),
            metadata,
            epoch_key=epoch_key,
            metadata_key=metadata_key,
        )
        self._history.append(
            {
                "step": "merge_metadata",
                "mode": "rows" if epoch_key is None else "keys",
                "epoch_key": epoch_key,
                "metadata_key": metadata_key,
            }
        )
        return self

    def transform_metadata(
        self,
        transform: Callable[[pd.DataFrame], pd.DataFrame],
        *,
        name: str,
        version: str,
        params: Mapping[str, object] | None = None,
    ) -> ExternalPipeline:
        """Apply a named, versioned, row-preserving metadata transform."""
        _validate_step_identity(name, version)
        self._epochs = steps.transform_metadata(self._require_epochs(), transform)
        self._history.append(
            {
                "step": "transform_metadata",
                "name": name,
                "version": str(version),
                "params": dict(params or {}),
            }
        )
        return self

    def select_epochs(
        self,
        *,
        include: Mapping[str, object] | None = None,
        exclude: Mapping[str, object] | None = None,
    ) -> ExternalPipeline:
        """Immediately select epochs using attached metadata."""
        self._epochs = steps.select_epochs(
            self._require_epochs(), include=include, exclude=exclude
        )
        self._history.append(
            {
                "step": "select_epochs",
                "include": None if include is None else dict(include),
                "exclude": None if exclude is None else dict(exclude),
            }
        )
        return self

    def drop_channels(self, ch_names: Sequence[str]) -> ExternalPipeline:
        """Immediately remove named channels from the current epochs."""
        self._epochs = steps.drop_channels(self._require_epochs(), ch_names)
        self._history.append({"step": "drop_channels", "ch_names": list(ch_names)})
        return self

    def add_epoch_step(
        self,
        function: Callable[[mne.Epochs], mne.Epochs],
        *,
        name: str,
        version: str,
        params: Mapping[str, object] | None = None,
    ) -> ExternalPipeline:
        """Immediately apply a named, versioned custom epoch operation."""
        _validate_step_identity(name, version)
        output = function(self._require_epochs())
        if not isinstance(output, mne.BaseEpochs):
            raise TypeError("A custom epoch step must return an MNE Epochs object.")
        self._epochs = output
        self._history.append(
            {
                "step": "custom_epoch",
                "name": name,
                "version": str(version),
                "params": dict(params or {}),
            }
        )
        return self

    def build_epochs(
        self,
        output_dir: str | Path,
        *,
        task: str,
        recompute: str = "never",
    ) -> DatasetPipeline:
        """Write this subject using the standard prepared-dataset contract."""
        epochs = steps.assign_identity(self._require_epochs(), self.subject_index)
        pipeline_spec = {"kind": "external", "steps": self._history}
        builder = DatasetBuilder(
            output_dir,
            task=task,
            stage="prepared",
            pipeline_fingerprint=fingerprint(pipeline_spec),
            pipeline_spec=pipeline_spec,
            recompute=recompute,
            subject_indices=[self.subject_index],
            complete_subject_set=False,
        )
        try:
            input_fingerprint = fingerprint_epochs(epochs)
            if builder.should_write(self.subject_index, input_fingerprint):
                builder.write_subject(
                    self.subject_index,
                    epochs,
                    input_fingerprint=input_fingerprint,
                )
            else:
                builder.record_reused(self.subject_index)
            result = builder.finish()
            if self._gaze_geometry is not None:
                result.configure_gaze(**self._gaze_geometry)
            return result
        except Exception:
            builder.abort()
            raise

    def _require_epochs(self) -> mne.Epochs:
        """Return current epochs or explain the required array conversion."""
        if self._epochs is None:
            raise RuntimeError("Array input must call make_epochs before this operation.")
        return self._epochs


def init_external(
    *, subject_index: str | int, data: np.ndarray | mne.BaseEpochs
) -> ExternalPipeline:
    """Create a subject-level eager pipeline from in-memory external data."""
    return ExternalPipeline(subject_index=subject_index, data=data)


def _external_events(events: Sequence[int] | np.ndarray | None, n_epochs: int) -> np.ndarray:
    """Normalize external epoch event labels to MNE's three-column layout."""
    if events is None:
        return np.column_stack(
            [np.arange(n_epochs, dtype=int), np.zeros(n_epochs, dtype=int), np.ones(n_epochs, dtype=int)]
        )
    array = np.asarray(events)
    if array.ndim == 1:
        if len(array) != n_epochs:
            raise ValueError("One-dimensional events must contain one code per epoch.")
        return np.column_stack(
            [np.arange(n_epochs, dtype=int), np.zeros(n_epochs, dtype=int), array.astype(int)]
        )
    if array.shape != (n_epochs, 3):
        raise ValueError(f"Events must have shape ({n_epochs}, 3) or ({n_epochs},).")
    return array.astype(int, copy=False)


def _validate_step_identity(name: str, version: str) -> None:
    """Require meaningful custom-step provenance fields."""
    if not str(name).strip() or not str(version).strip():
        raise ValueError("Custom steps require non-empty name and version values.")
