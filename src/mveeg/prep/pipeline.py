"""Dataset-level lazy preprocessing pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import warnings

import mne
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import steps
from .dataset import (
    DatasetBuilder,
    DatasetPipeline,
    _normalize_subject_index,
    fingerprint,
    fingerprint_files,
)
from .gaze import normalize_gaze_geometry


@dataclass
class _Operation:
    """One lazy operation plus its reproducibility metadata."""

    kind: str
    params: dict[str, object] = field(default_factory=dict)
    function: Callable | None = None

    def spec(self) -> dict[str, object]:
        """Return the serializable portion used in provenance and fingerprints."""
        return {"step": self.kind, **self.params}


class RawPipeline:
    """Register and execute a dataset-level raw preprocessing workflow.

    Methods register work and return ``self``. No subject data are loaded until
    :meth:`build_epochs` executes the registered sequence.
    """

    def __init__(self, input_dir: str | Path, *, subject_pattern: str = "sub*"):
        """Create a lazy pipeline rooted at subject folders in ``input_dir``."""
        self.input_dir = Path(input_dir).expanduser().resolve()
        self.subject_pattern = subject_pattern
        self._phase = "load"
        self._eeg_loader: dict[str, object] | None = None
        self._eye_loader: dict[str, object] | None = None
        self._behavior_loader: dict[str, object] | None = None
        self._operations: list[_Operation] = []
        self._gaze_geometry: dict[str, float | int] | None = None

    def configure_gaze(
        self,
        *,
        viewing_distance_cm: float,
        screen_width_cm: float,
        screen_width_px: int,
    ) -> RawPipeline:
        """Store gaze geometry for later degree-based quality rules."""
        self._gaze_geometry = normalize_gaze_geometry(
            viewing_distance_cm=viewing_distance_cm,
            screen_width_cm=screen_width_cm,
            screen_width_px=screen_width_px,
        )
        return self

    def load_eeg(
        self,
        pattern: str = "*.vhdr",
        *,
        preload: bool = False,
        reader: Callable | None = None,
        **reader_kwargs: object,
    ) -> RawPipeline:
        """Register continuous EEG loading from each subject folder.

        Multiple matching recordings are concatenated alphabetically. ``reader``
        defaults to MNE's format-dispatching ``read_raw`` and must follow the
        same path-first calling convention when customized.
        """
        self._require_load_phase("load_eeg")
        if self._eeg_loader is not None:
            raise RuntimeError("load_eeg can only be registered once.")
        self._eeg_loader = {
            "pattern": pattern,
            "preload": preload,
            "reader": reader,
            "reader_kwargs": dict(reader_kwargs),
        }
        return self

    def load_eyelink(
        self,
        pattern: str = "*.asc",
        *,
        reader: Callable | None = None,
        **reader_kwargs: object,
    ) -> RawPipeline:
        """Register optional EyeLink ASCII loading from each subject folder."""
        if self._eye_loader is not None:
            raise RuntimeError("load_eyelink can only be registered once.")
        self._eye_loader = {
            "pattern": pattern,
            "reader": reader,
            "reader_kwargs": dict(reader_kwargs),
        }
        return self

    def load_behavior(
        self,
        pattern: str = "*_beh.csv",
        *,
        include: Mapping[str, object] | None = None,
        sep: str | None = None,
    ) -> RawPipeline:
        """Register behavior loading and its required pre-alignment row filter."""
        if self._behavior_loader is not None:
            raise RuntimeError("load_behavior can only be registered once.")
        self._behavior_loader = {
            "pattern": pattern,
            "include": None if include is None else dict(include),
            "sep": sep,
        }
        return self

    def filter_eeg(
        self,
        *,
        l_freq: float | None = None,
        h_freq: float | None = None,
        **kwargs: object,
    ) -> RawPipeline:
        """Register continuous EEG filtering before epoch construction."""
        self._require_raw_phase("filter_eeg")
        self._phase = "raw"
        self._operations.append(
            _Operation("filter_eeg", {"l_freq": l_freq, "h_freq": h_freq, "kwargs": dict(kwargs)})
        )
        return self

    def add_raw_step(
        self,
        function: Callable[[mne.io.BaseRaw], mne.io.BaseRaw],
        *,
        name: str,
        version: str,
        params: Mapping[str, object] | None = None,
    ) -> RawPipeline:
        """Register a reproducible custom operation on continuous EEG data."""
        self._require_raw_phase("add_raw_step")
        _validate_step_identity(name, version)
        self._phase = "raw"
        self._operations.append(
            _Operation(
                "custom_raw",
                {"name": name, "version": str(version), "params": dict(params or {})},
                function,
            )
        )
        return self

    def make_epochs(
        self,
        *,
        event_id: Mapping[str, int],
        time_window: tuple[float, float],
        trial_sequences: Mapping[int, Sequence[int] | Sequence[Sequence[int]]] | None = None,
        time_zero: int | Mapping[int, int] | None = None,
        baseline: tuple[float | None, float | None] | None = None,
        sampling_rate: float | None = None,
        **kwargs: object,
    ) -> RawPipeline:
        """Register the single operation that converts continuous EEG to epochs."""
        self._require_raw_phase("make_epochs")
        resolved_window = steps._normalize_time_window(time_window)
        resolved_rate = steps._normalize_sampling_rate(sampling_rate)
        if trial_sequences is None:
            if time_zero is not None:
                raise ValueError("time_zero is only valid when trial_sequences is provided.")
            resolved_sequences = None
            resolved_time_zero = None
        else:
            if not trial_sequences:
                raise ValueError("trial_sequences cannot be empty.")
            resolved_sequences = dict(trial_sequences)
            resolved_time_zero = steps._resolve_time_zero(
                resolved_sequences,
                time_zero,
            )
        self._phase = "epochs"
        self._operations.append(
            _Operation(
                "make_epochs",
                {
                    "event_id": dict(event_id),
                    "time_window": resolved_window,
                    "trial_sequences": resolved_sequences,
                    "time_zero": resolved_time_zero,
                    "baseline": baseline,
                    "sampling_rate": resolved_rate,
                    "kwargs": dict(kwargs),
                },
            )
        )
        return self

    def sync_eyelink(self) -> RawPipeline:
        """Register EyeLink synchronization using the epoch construction config."""
        self._require_epoch_phase("sync_eyelink")
        if self._eye_loader is None:
            raise RuntimeError("sync_eyelink requires load_eyelink to be registered first.")
        self._operations.append(_Operation("sync_eyelink"))
        return self

    def align_behavior(self) -> RawPipeline:
        """Register strict count-and-order behavior attachment."""
        self._require_epoch_phase("align_behavior")
        if self._behavior_loader is None:
            raise RuntimeError("align_behavior requires load_behavior to be registered first.")
        self._operations.append(_Operation("align_behavior"))
        return self

    def select_epochs(
        self,
        *,
        include: Mapping[str, object] | None = None,
        exclude: Mapping[str, object] | None = None,
    ) -> RawPipeline:
        """Register metadata-driven post-alignment epoch selection."""
        self._require_epoch_phase("select_epochs")
        self._operations.append(
            _Operation(
                "select_epochs",
                {
                    "include": None if include is None else dict(include),
                    "exclude": None if exclude is None else dict(exclude),
                },
            )
        )
        return self

    def drop_channels(self, ch_names: Sequence[str]) -> RawPipeline:
        """Register channel removal after epoch construction."""
        self._require_epoch_phase("drop_channels")
        self._operations.append(_Operation("drop_channels", {"ch_names": list(ch_names)}))
        return self

    def transform_metadata(
        self,
        transform: Callable[[pd.DataFrame], pd.DataFrame],
        *,
        name: str,
        version: str,
        params: Mapping[str, object] | None = None,
    ) -> RawPipeline:
        """Register a named, versioned, row-preserving metadata transform."""
        self._require_epoch_phase("transform_metadata")
        _validate_step_identity(name, version)
        self._operations.append(
            _Operation(
                "transform_metadata",
                {"name": name, "version": str(version), "params": dict(params or {})},
                transform,
            )
        )
        return self

    def add_epoch_step(
        self,
        function: Callable[[mne.Epochs], mne.Epochs],
        *,
        name: str,
        version: str,
        params: Mapping[str, object] | None = None,
    ) -> RawPipeline:
        """Register a reproducible custom operation on constructed epochs."""
        self._require_epoch_phase("add_epoch_step")
        _validate_step_identity(name, version)
        self._operations.append(
            _Operation(
                "custom_epoch",
                {"name": name, "version": str(version), "params": dict(params or {})},
                function,
            )
        )
        return self

    def build_epochs(
        self,
        output_dir: str | Path,
        *,
        task: str,
        exclude_subjects: Sequence[str | int] | None = None,
        recompute: str = "never",
        progress: bool = True,
    ) -> DatasetPipeline:
        """Execute all registered steps and optionally show subject progress."""
        if self._eeg_loader is None:
            raise RuntimeError("build_epochs requires load_eeg.")
        if self._phase != "epochs":
            raise RuntimeError("build_epochs requires make_epochs.")
        subject_dirs = self._discover_subjects()
        excluded = {_normalize_subject_index(value) for value in (exclude_subjects or [])}
        selected = [item for item in subject_dirs if item[0] not in excluded]
        if not selected:
            raise ValueError("No subject folders remain after exclude_subjects.")

        pipeline_spec = self._pipeline_spec()
        pipeline_fingerprint = fingerprint(pipeline_spec)
        builder = DatasetBuilder(
            output_dir,
            task=task,
            stage="prepared",
            pipeline_fingerprint=pipeline_fingerprint,
            pipeline_spec=pipeline_spec,
            recompute=recompute,
            subject_indices=[subject for subject, _ in selected],
            complete_subject_set=True,
        )
        try:
            for subject_index, subject_dir in tqdm(
                selected,
                desc="Building epochs",
                unit="subject",
                disable=not progress,
            ):
                input_paths = self._input_paths(subject_dir)
                input_fingerprint = fingerprint_files(input_paths, root=self.input_dir)
                if not builder.should_write(subject_index, input_fingerprint):
                    builder.record_reused(subject_index)
                    continue
                epochs = self._process_subject(subject_dir)
                epochs = steps.assign_identity(epochs, subject_index)
                builder.write_subject(subject_index, epochs, input_fingerprint=input_fingerprint)
            result = builder.finish()
            if self._gaze_geometry is not None:
                result.configure_gaze(**self._gaze_geometry)
            return result
        except Exception:
            builder.abort()
            raise

    def _process_subject(self, subject_dir: Path) -> mne.Epochs:
        """Load and run the registered workflow for one subject."""
        raw = self._load_eeg(subject_dir)
        eye_raw = self._load_eye(subject_dir) if self._eye_loader is not None else None
        behavior = self._load_behavior(subject_dir) if self._behavior_loader is not None else None
        epochs: mne.Epochs | None = None
        epoch_config = next(
            operation.params for operation in self._operations if operation.kind == "make_epochs"
        )
        for operation in self._operations:
            params = operation.params
            if operation.kind == "filter_eeg":
                raw = steps.filter_eeg(
                    raw,
                    l_freq=params["l_freq"],
                    h_freq=params["h_freq"],
                    **params["kwargs"],
                )
            elif operation.kind == "custom_raw":
                raw = operation.function(raw)
                if not isinstance(raw, mne.io.BaseRaw):
                    raise TypeError("A custom raw step must return an MNE Raw object.")
            elif operation.kind == "make_epochs":
                epochs = steps.make_epochs(
                    raw,
                    event_id=params["event_id"],
                    time_window=params["time_window"],
                    trial_sequences=params["trial_sequences"],
                    time_zero=params["time_zero"],
                    baseline=params["baseline"],
                    sampling_rate=params["sampling_rate"],
                    **params["kwargs"],
                )
            elif operation.kind == "sync_eyelink":
                assert epochs is not None and eye_raw is not None
                epochs = steps.sync_eyelink(
                    epochs,
                    eye_raw,
                    event_id=epoch_config["event_id"],
                    time_window=(float(epochs.tmin), float(epochs.tmax)),
                    trial_sequences=epoch_config["trial_sequences"],
                    time_zero=epoch_config["time_zero"],
                    baseline=epoch_config["baseline"],
                    sampling_rate=float(epochs.info["sfreq"]),
                )
            elif operation.kind == "align_behavior":
                assert epochs is not None and behavior is not None
                epochs = steps.align_behavior(epochs, behavior)
            elif operation.kind == "select_epochs":
                assert epochs is not None
                epochs = steps.select_epochs(epochs, **params)
            elif operation.kind == "drop_channels":
                assert epochs is not None
                epochs = steps.drop_channels(epochs, params["ch_names"])
            elif operation.kind == "transform_metadata":
                assert epochs is not None
                epochs = steps.transform_metadata(epochs, operation.function)
            elif operation.kind == "custom_epoch":
                assert epochs is not None
                epochs = operation.function(epochs)
                if not isinstance(epochs, mne.BaseEpochs):
                    raise TypeError("A custom epoch step must return an MNE Epochs object.")
            else:
                raise RuntimeError(f"Unknown registered operation {operation.kind!r}.")
        if epochs is None:
            raise RuntimeError("The registered workflow did not create epochs.")
        return epochs

    def _load_eeg(self, subject_dir: Path) -> mne.io.BaseRaw:
        """Load and concatenate registered EEG files for one subject."""
        assert self._eeg_loader is not None
        paths = _matched_files(subject_dir, str(self._eeg_loader["pattern"]))
        reader = self._eeg_loader["reader"] or mne.io.read_raw
        kwargs = dict(self._eeg_loader["reader_kwargs"])
        raws = [
            reader(
                path,
                preload=bool(self._eeg_loader["preload"]),
                verbose="ERROR",
                **kwargs,
            )
            for path in paths
        ]
        return raws[0] if len(raws) == 1 else mne.concatenate_raws(raws, verbose="ERROR")

    def _load_eye(self, subject_dir: Path) -> mne.io.BaseRaw:
        """Load and concatenate registered EyeLink files for one subject."""
        assert self._eye_loader is not None
        paths = _matched_files(subject_dir, str(self._eye_loader["pattern"]))
        reader = self._eye_loader["reader"]
        kwargs = dict(self._eye_loader["reader_kwargs"])
        raws = []
        for path in paths:
            if reader is not None:
                raw = reader(path, verbose="ERROR", **kwargs)
            else:
                raw = _read_eyelink(path, **kwargs)
            raws.append(raw)
        return raws[0] if len(raws) == 1 else mne.concatenate_raws(raws, verbose="ERROR")

    def _load_behavior(self, subject_dir: Path) -> pd.DataFrame:
        """Load exactly one behavior table and apply its registered pre-filter."""
        assert self._behavior_loader is not None
        paths = _matched_files(subject_dir, str(self._behavior_loader["pattern"]))
        if len(paths) != 1:
            raise ValueError(
                f"Behavior pattern must match exactly one file in {subject_dir}, found {len(paths)}."
            )
        sep = self._behavior_loader["sep"]
        if sep is None:
            sep = "\t" if paths[0].suffix.lower() == ".tsv" else ","
        table = pd.read_csv(paths[0], sep=sep)
        return steps.filter_table(table, include=self._behavior_loader["include"])

    def _input_paths(self, subject_dir: Path) -> list[Path]:
        """Return every registered source file used for one subject."""
        loaders = [self._eeg_loader, self._eye_loader, self._behavior_loader]
        paths: list[Path] = []
        for loader in loaders:
            if loader is not None:
                for path in _matched_files(subject_dir, str(loader["pattern"])):
                    paths.extend(_source_dependencies(path))
        return sorted(set(paths))

    def _discover_subjects(self) -> list[tuple[str, Path]]:
        """Discover subject directories and reject normalized index collisions."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}.")
        pairs = [
            (_normalize_subject_index(path.name), path)
            for path in sorted(self.input_dir.glob(self.subject_pattern))
            if path.is_dir()
        ]
        if not pairs:
            raise FileNotFoundError(
                f"No subject directories matched {self.subject_pattern!r} in {self.input_dir}."
            )
        subjects = [subject for subject, _ in pairs]
        if len(subjects) != len(set(subjects)):
            raise ValueError("Subject directories produce duplicate normalized subject_index values.")
        return pairs

    def _pipeline_spec(self) -> dict[str, object]:
        """Return the complete subject-independent lazy workflow specification."""
        def loader_spec(loader: dict[str, object] | None) -> object:
            if loader is None:
                return None
            return dict(loader)

        return {
            "kind": "raw",
            "subject_pattern": self.subject_pattern,
            "load_eeg": loader_spec(self._eeg_loader),
            "load_eyelink": loader_spec(self._eye_loader),
            "load_behavior": loader_spec(self._behavior_loader),
            "steps": [operation.spec() for operation in self._operations],
        }

    def _require_load_phase(self, method: str) -> None:
        """Require continuous EEG loading before its processing begins."""
        if self._phase != "load":
            raise RuntimeError(f"{method} must be called before preprocessing steps.")

    def _require_raw_phase(self, method: str) -> None:
        """Require EEG loading and disallow raw steps after epoch construction."""
        if self._eeg_loader is None:
            raise RuntimeError(f"{method} requires load_eeg first.")
        if self._phase == "epochs":
            raise RuntimeError(f"{method} cannot run after make_epochs.")

    def _require_epoch_phase(self, method: str) -> None:
        """Require epoch construction before epoch-level steps."""
        if self._phase != "epochs":
            raise RuntimeError(f"{method} requires make_epochs first.")


def init_pipeline(input_dir: str | Path, *, subject_pattern: str = "sub*") -> RawPipeline:
    """Create a dataset-level lazy raw preprocessing pipeline."""
    return RawPipeline(input_dir, subject_pattern=subject_pattern)


def _matched_files(subject_dir: Path, pattern: str) -> list[Path]:
    """Return sorted matching files or raise a subject-specific error."""
    paths = sorted(path for path in subject_dir.glob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No files matched {pattern!r} in {subject_dir}.")
    return paths


def _source_dependencies(path: Path) -> list[Path]:
    """Include companion files used by common MNE header-based formats."""
    dependencies = [path]
    companion_suffixes = {
        ".vhdr": [".eeg", ".vmrk"],
        ".set": [".fdt"],
    }
    for suffix in companion_suffixes.get(path.suffix.lower(), []):
        companion = path.with_suffix(suffix)
        if companion.exists():
            dependencies.append(companion)
    return dependencies


def _read_eyelink(path: Path, **kwargs: object) -> mne.io.BaseRaw:
    """Read EyeLink ASCII with an internal binocular fallback."""
    try:
        return mne.io.read_raw_eyelink(path, verbose="ERROR", **kwargs)
    except (RuntimeError, ValueError) as mne_error:
        if kwargs:
            options = ", ".join(sorted(kwargs))
            raise ValueError(
                f"MNE could not read {path.name}, and mveeg's fallback EyeLink reader "
                f"cannot apply reader options: {options}."
            ) from mne_error
        try:
            raw = _read_eyelink_fallback(path)
        except (OSError, ValueError) as mveeg_error:
            raise ValueError(
                f"Could not read {path.name} as EyeLink ASCII. "
                f"MNE reader failed: {mne_error}; mveeg reader failed: {mveeg_error}"
            ) from mveeg_error

        message = str(mne_error)
        # MNE PR #13571 fixes signed samples being mistaken for the STATUS column.
        known_status_error = (
            "Expected the samples data in this file to have 7 columns of data, but got 6."
            in message
            and "xpos_left" in message
            and "pupil_right" in message
        )
        if not known_status_error:
            warnings.warn(
                f"MNE could not read {path.name}; using mveeg's internal EyeLink ASCII reader "
                f"({mne_error}).",
                stacklevel=2,
            )
        return raw


def _read_eyelink_fallback(path: Path) -> mne.io.RawArray:
    """Read binocular samples and message markers from an EyeLink ASCII file."""
    sfreq: float | None = None
    samples: list[list[float]] = []
    messages: list[tuple[int, str]] = []
    with path.open("r") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("SAMPLES"):
                parts = stripped.split()
                if "RATE" in parts:
                    sfreq = float(parts[parts.index("RATE") + 1])
            elif stripped.startswith("MSG"):
                parts = stripped.split(maxsplit=2)
                if len(parts) == 3:
                    messages.append((int(parts[1]), parts[2].split()[-1]))
            else:
                parts = stripped.split()
                if len(parts) >= 7 and parts[0].isdigit():
                    try:
                        row = [float(parts[0])]
                        row.extend(
                            np.nan if value == "." else float(value) for value in parts[1:7]
                        )
                        samples.append(row)
                    except ValueError:
                        continue
    if sfreq is None or not samples:
        raise ValueError(f"Could not find sampling rate and binocular samples in {path}.")
    array = np.asarray(samples)
    sample_clock = array[:, 0].astype(int)
    if np.any(np.diff(sample_clock) <= 0):
        raise ValueError(f"EyeLink sample timestamps are not strictly increasing in {path}.")
    sample_step = 1000.0 / sfreq
    sample_ix = np.rint((sample_clock - sample_clock[0]) / sample_step).astype(int)
    reconstructed_clock = sample_clock[0] + sample_ix * sample_step
    if not np.allclose(reconstructed_clock, sample_clock, atol=sample_step / 4):
        raise ValueError(f"EyeLink sample timestamps do not match the declared sampling rate in {path}.")
    data = np.full((6, sample_ix[-1] + 1), np.nan, dtype=float)
    data[:, sample_ix] = array[:, 1:].T
    info = mne.create_info(
        ["xpos_left", "ypos_left", "pupil_left", "xpos_right", "ypos_right", "pupil_right"],
        sfreq=sfreq,
        ch_types=["eyegaze", "eyegaze", "pupil", "eyegaze", "eyegaze", "pupil"],
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    if messages:
        messages = [
            (clock, description)
            for clock, description in messages
            if sample_clock[0] <= clock <= sample_clock[-1]
        ]
    if messages:
        event_clock = np.asarray([clock for clock, _ in messages])
        event_ix = np.rint((event_clock - sample_clock[0]) / sample_step).astype(int)
        event_ix = np.clip(event_ix, 0, data.shape[1] - 1)
        raw.set_annotations(
            mne.Annotations(
                onset=event_ix / sfreq,
                duration=np.zeros(len(messages)),
                description=[description for _, description in messages],
            )
        )
    return raw


def _validate_step_identity(name: str, version: str) -> None:
    """Require meaningful custom-step provenance fields."""
    if not str(name).strip() or not str(version).strip():
        raise ValueError("Custom steps require non-empty name and version values.")
