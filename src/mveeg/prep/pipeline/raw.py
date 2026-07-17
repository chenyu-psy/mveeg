"""Dataset-level lazy preprocessing pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import mne
import pandas as pd
from tqdm.auto import tqdm

from ..._dataset.manifest import normalize_subject_index
from ..._dataset.metadata import assign_metadata_variables, validate_metadata_variables
from ..._dataset.store import DatasetBuilder
from ..._provenance import fingerprint, fingerprint_files
from .. import events, steps
from ..eyelink import eyelink_files, read_eyelink
from ..gaze import normalize_gaze_geometry
from .dataset import DatasetPipeline


@dataclass
class _Operation:
    """One lazy operation plus its reproducibility metadata."""

    kind: str
    params: dict[str, object] = field(default_factory=dict)
    function: object | None = None

    def spec(self) -> dict[str, object]:
        """Return the serializable portion used in provenance and fingerprints."""
        params = dict(self.params)
        if self.kind == "make_epochs":
            for name in ("trial_sequences", "time_zero"):
                mapping = params.get(name)
                if isinstance(mapping, Mapping):
                    params[name] = {str(key): value for key, value in mapping.items()}
        return {"step": self.kind, **params}


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
        **reader_kwargs: object,
    ) -> RawPipeline:
        """Register continuous EEG loading through :func:`mne.io.read_raw`."""
        self._require_load_phase("load_eeg")
        if self._eeg_loader is not None:
            raise RuntimeError("load_eeg can only be registered once.")
        self._eeg_loader = {
            "pattern": pattern,
            "preload": preload,
            "reader_kwargs": dict(reader_kwargs),
        }
        return self

    def load_eyelink(self) -> RawPipeline:
        """Register automatic EyeLink ASC/EDF loading from each subject folder."""
        if self._eye_loader is not None:
            raise RuntimeError("load_eyelink can only be registered once.")
        self._eye_loader = {"format": "asc_or_edf"}
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
        resolved_window = events._normalize_time_window(time_window)
        resolved_rate = events._normalize_sampling_rate(sampling_rate)
        if trial_sequences is None:
            if time_zero is not None:
                raise ValueError("time_zero is only valid when trial_sequences is provided.")
            resolved_sequences = None
            resolved_time_zero = None
        else:
            if not trial_sequences:
                raise ValueError("trial_sequences cannot be empty.")
            resolved_sequences = dict(trial_sequences)
            resolved_time_zero = events._resolve_time_zero(
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
        **variables: Callable[[pd.DataFrame], object],
    ) -> RawPipeline:
        """Register ordered scalar or trial-aligned metadata variables."""
        self._require_epoch_phase("transform_metadata")
        validate_metadata_variables(variables)
        self._operations.append(
            _Operation(
                "transform_metadata",
                {"variables": list(variables)},
                dict(variables),
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
        excluded = {normalize_subject_index(value) for value in (exclude_subjects or [])}
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
                source_fingerprint = fingerprint_files(
                    self._input_paths(subject_dir), root=self.input_dir
                )
                has_metadata_transform = any(
                    operation.kind == "transform_metadata" for operation in self._operations
                )
                epochs: mne.Epochs | None = None
                if has_metadata_transform and recompute == "never":
                    saved = builder.saved_input_fingerprint(subject_index)
                    if saved is not None:
                        builder.should_write(subject_index, saved)
                        builder.record_reused(subject_index)
                        continue
                if has_metadata_transform:
                    epochs = self._process_subject(subject_dir, subject_index)
                    epochs = steps._ensure_identity(epochs, subject_index)
                    input_fingerprint = _metadata_output_fingerprint(
                        source_fingerprint, epochs.metadata
                    )
                else:
                    input_fingerprint = source_fingerprint
                if not builder.should_write(subject_index, input_fingerprint):
                    builder.record_reused(subject_index)
                    continue
                if epochs is None:
                    epochs = self._process_subject(subject_dir, subject_index)
                    epochs = steps._ensure_identity(epochs, subject_index)
                builder.write_subject(subject_index, epochs, input_fingerprint=input_fingerprint)
            summary = builder.finish()
            result = DatasetPipeline(output_dir, run_summary=summary)
            if self._gaze_geometry is not None:
                result.configure_gaze(**self._gaze_geometry)
            return result
        except Exception:
            builder.abort()
            raise

    def _process_subject(self, subject_dir: Path, subject_index: str) -> mne.Epochs:
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
                epochs = steps._ensure_identity(epochs, subject_index)
                assert epochs.metadata is not None
                variables = operation.function
                assert isinstance(variables, Mapping)
                metadata = assign_metadata_variables(epochs.metadata, variables)
                with mne.use_log_level("ERROR"):
                    epochs.metadata = metadata
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
        kwargs = dict(self._eeg_loader["reader_kwargs"])
        raws = [
            mne.io.read_raw(
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
        raws = [read_eyelink(path) for path in eyelink_files(subject_dir)]
        return raws[0] if len(raws) == 1 else mne.concatenate_raws(raws, verbose="ERROR")

    def _load_behavior(self, subject_dir: Path) -> pd.DataFrame:
        """Load exactly one behavior table and apply its registered pre-filter."""
        assert self._behavior_loader is not None
        paths = _matched_files(subject_dir, str(self._behavior_loader["pattern"]))
        if len(paths) != 1:
            raise ValueError(
                f"Behavior pattern must match exactly one file in {subject_dir}, "
                f"found {len(paths)}."
            )
        sep = self._behavior_loader["sep"]
        if sep is None:
            sep = "\t" if paths[0].suffix.lower() == ".tsv" else ","
        table = pd.read_csv(paths[0], sep=sep)
        return steps.filter_table(table, include=self._behavior_loader["include"])

    def _input_paths(self, subject_dir: Path) -> list[Path]:
        """Return every registered source file used for one subject."""
        loaders = [self._eeg_loader, self._behavior_loader]
        paths: list[Path] = []
        for loader in loaders:
            if loader is not None:
                for path in _matched_files(subject_dir, str(loader["pattern"])):
                    paths.extend(_source_dependencies(path))
        if self._eye_loader is not None:
            paths.extend(eyelink_files(subject_dir))
        return sorted(set(paths))

    def _discover_subjects(self) -> list[tuple[str, Path]]:
        """Discover subject directories and reject normalized index collisions."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}.")
        pairs = [
            (normalize_subject_index(path.name), path)
            for path in sorted(self.input_dir.glob(self.subject_pattern))
            if path.is_dir()
        ]
        if not pairs:
            raise FileNotFoundError(
                f"No subject directories matched {self.subject_pattern!r} in {self.input_dir}."
            )
        subjects = [subject for subject, _ in pairs]
        if len(subjects) != len(set(subjects)):
            raise ValueError(
                "Subject directories produce duplicate normalized subject_index values."
            )
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


def _metadata_output_fingerprint(
    source_fingerprint: str,
    metadata: pd.DataFrame | None,
) -> str:
    """Fingerprint raw inputs together with actual derived metadata values."""

    if metadata is None:
        raise ValueError("transform_metadata requires epochs metadata.")
    return fingerprint(
        {
            "source_fingerprint": source_fingerprint,
            "metadata": {
                "columns": list(metadata.columns),
                "dtypes": [str(metadata[column].dtype) for column in metadata],
                "values": metadata.to_json(
                    orient="split",
                    date_format="iso",
                    double_precision=15,
                ),
            },
        }
    )


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


def _validate_step_identity(name: str, version: str) -> None:
    """Require meaningful custom-step provenance fields."""
    if not str(name).strip() or not str(version).strip():
        raise ValueError("Custom steps require non-empty name and version values.")
