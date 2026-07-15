"""Researcher-facing encoding configuration and subject orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import secrets
import warnings

import numpy as np
import pandas as pd

from .._shared.fingerprints import fingerprint, fingerprint_files
from .._shared.metadata import (
    assign_metadata_variables,
    load_subject_epochs_and_metadata,
    load_subject_metadata,
    load_validated_dataset_manifest,
    subject_dataset_paths,
)
from .._shared.time_windows import average_time_windows, build_time_bins
from .._shared.topography import build_topography_coord_table
from .analysis import SubjectEncoding, encode_subject
from .prepare import DEFAULT_DROP_TYPES, select_trials, validate_groups
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


class EncodingPipeline:
    """Lazy encoding analysis bound to one manifest-backed EEG dataset."""

    def __init__(self, dataset: str | Path):
        self.dataset = Path(dataset).expanduser().resolve()
        self._refresh_dataset()
        self._metadata_variables: dict[str, Callable[[pd.DataFrame], object]] = {}
        self._trial_selection = {
            "qc": "final_status",
            "keep": ["accepted"],
            "exclude": {},
        }
        self._epoch_preparation = {
            "crop": [0.0, 0.8],
            "time_bin": 50,
            "drop_channel_types": list(DEFAULT_DROP_TYPES),
            "drop_channels": [],
        }
        self._model = {"penalty": {"component": 1.0, "condition": 0.1}}
        self._cv = {"folds": 5, "seed": None}

    def transform_metadata(
        self,
        **variables: Callable[[pd.DataFrame], object],
    ) -> EncodingPipeline:
        """Register ordered DataFrame-to-column analysis variables."""

        if not variables:
            raise ValueError("transform_metadata requires at least one variable.")
        for name, function in variables.items():
            if not isinstance(name, str) or name.strip() == "":
                raise ValueError("Metadata variable names must be non-empty strings.")
            if name in {"subject_index", "epoch_index"}:
                raise ValueError(
                    "transform_metadata cannot replace subject_index or epoch_index."
                )
            if not callable(function):
                raise TypeError(f"Metadata variable {name!r} must be defined by a callable.")
        self._metadata_variables = dict(variables)
        return self

    def select_trials(
        self,
        *,
        qc: str | None = "final_status",
        keep: Sequence[object] = ("accepted",),
        exclude: Mapping[str, Sequence[object] | str] | None = None,
    ) -> EncodingPipeline:
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
        self._trial_selection = {
            "qc": qc,
            "keep": list(keep),
            "exclude": normalized_exclude,
        }
        return self

    def prepare_epochs(
        self,
        *,
        crop: tuple[float, float] | None = (0.0, 0.8),
        time_bin: int = 50,
        drop_channel_types: Sequence[str] = DEFAULT_DROP_TYPES,
        drop_channels: Sequence[str] = (),
    ) -> EncodingPipeline:
        """Register epoch cropping, time bins, and analysis channels."""

        if crop is not None:
            if len(crop) != 2 or not np.all(np.isfinite(crop)) or crop[0] >= crop[1]:
                raise ValueError(
                    "crop must contain finite increasing start/end times in seconds."
                )
            crop_value = [float(crop[0]), float(crop[1])]
        else:
            crop_value = None
        if not isinstance(time_bin, (int, np.integer)) or int(time_bin) < 1:
            raise ValueError("time_bin must be a positive integer in milliseconds.")
        self._epoch_preparation = {
            "crop": crop_value,
            "time_bin": int(time_bin),
            "drop_channel_types": list(drop_channel_types),
            "drop_channels": list(drop_channels),
        }
        return self

    def setup_model(
        self,
        *,
        penalty: Mapping[str, float] | None = None,
    ) -> EncodingPipeline:
        """Configure component and condition ridge penalties."""

        values = {"component": 1.0, "condition": 0.1} if penalty is None else dict(penalty)
        if set(values) != {"component", "condition"}:
            raise ValueError("penalty must contain exactly 'component' and 'condition'.")
        try:
            values = {name: float(value) for name, value in values.items()}
        except (TypeError, ValueError) as error:
            raise TypeError("penalty values must be numeric.") from error
        if any(not np.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError("component and condition penalties must be finite positive values.")
        self._model = {"penalty": values}
        return self

    def setup_cv(
        self,
        *,
        folds: int = 5,
        seed: int | None = None,
    ) -> EncodingPipeline:
        """Configure one stratified K-fold partition for held-out expression."""

        if not isinstance(folds, (int, np.integer)) or int(folds) < 2:
            raise ValueError("folds must be an integer of at least 2.")
        if seed is not None and (
            not isinstance(seed, (int, np.integer)) or int(seed) < 0
        ):
            raise ValueError("seed must be a non-negative integer or None.")
        self._cv = {"folds": int(folds), "seed": None if seed is None else int(seed)}
        return self

    def encode(
        self,
        *,
        formula: str,
        target: str,
        conditions: Mapping[str, Sequence[object]],
        expression: Mapping[str, Sequence[object]] | None = None,
        file: str | Path,
        recompute: str = "never",
        n_jobs: int = 1,
        progress: bool = True,
    ) -> None:
        """Fit encoding and write the documented DuckDB result tables."""

        self._refresh_dataset()
        if not isinstance(formula, str) or formula.strip() == "":
            raise ValueError("formula must be a non-empty RHS component formula.")
        if not isinstance(target, str) or target.strip() == "":
            raise ValueError("target must be a non-empty post-transform metadata column.")
        if recompute not in {"never", "changed", "all"}:
            raise ValueError("recompute must be 'never', 'changed', or 'all'.")
        if not isinstance(n_jobs, int) or n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer.")
        if not isinstance(progress, bool):
            raise TypeError("progress must be bool.")

        condition_map, expression_map = validate_groups(conditions, expression)
        path = result_path(file)
        existing = read_analysis(path)
        seed = self._cv["seed"]
        if seed is None:
            seed = int(existing["seed"]) if existing is not None else secrets.randbelow(2**63 - 1)
        package_version = _version()
        config = {
            "version": package_version,
            "dataset": {"root": str(self.dataset), "task": self.task, "stage": self.stage},
            "metadata_variables": list(self._metadata_variables),
            "trial_selection": self._trial_selection,
            "epoch_preparation": self._epoch_preparation,
            "model": self._model,
            "cv": {**self._cv, "seed": seed},
            "formula": formula,
            "target": target,
            "conditions": condition_map,
            "expression": expression_map,
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
            seed=seed,
            config=config,
            fingerprint=analysis_fingerprint,
            reset=reset,
        )

        fingerprints = {
            subject: _input_fingerprint(self, subject) for subject in self.subject_indices
        }
        saved = subject_state(path)
        subjects = []
        for subject in self.subject_indices:
            status, previous = saved.get(subject, ("missing", None))
            changed = previous != fingerprints[subject]
            if recompute == "all" or status != "complete" or (
                recompute == "changed" and changed
            ):
                subjects.append(subject)
            elif recompute == "never" and changed:
                warnings.warn(
                    f"Input changed for subject {subject}, but recompute='never' reuses the result.",
                    stacklevel=2,
                )

        def work(subject: str):
            prepared = _prepare_subject(
                self,
                subject=subject,
                target=target,
                conditions=condition_map,
                expression=expression_map,
            )
            encoded = encode_subject(
                subject=subject,
                data=prepared["data"],
                metadata=prepared["metadata"],
                training_labels=prepared["training_labels"],
                expression_labels=prepared["expression_labels"],
                trial_ids=prepared["trials"]["trial"].to_numpy(dtype=int),
                times=prepared["time_bins"]["time"].to_numpy(dtype=int),
                condition_order=list(condition_map),
                formula=formula,
                penalty=self._model["penalty"],
                folds=self._cv["folds"],
                seed=seed,
                n_jobs=n_jobs,
                progress=progress,
            )
            return prepared, encoded

        for subject in subjects:
            mark_pending(path, subject, fingerprints[subject])
            prepared_encoded, error = _capture(work, subject)
            if error is not None:
                mark_failed(path, subject, fingerprints[subject], str(error))
                continue
            prepared, encoded = prepared_encoded
            try:
                _name_coefficient_channels(encoded, prepared["channels"])
                write_subject(
                    path,
                    subject=subject,
                    fingerprint=fingerprints[subject],
                    trials=prepared["trials"],
                    tables=encoded.tables,
                    predictors=encoded.predictors,
                    channels=prepared["channels"],
                    time_bins=prepared["time_bins"],
                )
            except Exception as error:
                mark_failed(path, subject, fingerprints[subject], str(error))

        if completed_subjects(path) == 0:
            raise RuntimeError(
                f"No subject completed encoding; inspect the subjects table in {path}."
            )
        return None

    @property
    def subject_indices(self) -> tuple[str, ...]:
        return tuple(self._manifest["subject_index"].astype(str))

    def _refresh_dataset(self) -> None:
        self._manifest = load_validated_dataset_manifest(self.dataset)
        self.task = str(self._manifest["task"].iloc[0])
        self.stage = str(self._manifest["stage"].iloc[0])
        if self.stage not in {"prepared", "preprocessed"}:
            raise ValueError(
                "Encoding requires a prepared or preprocessed dataset, "
                f"found stage {self.stage!r}."
            )


def init_pipeline(dataset: str | Path) -> EncodingPipeline:
    """Initialize encoding for one standard mveeg dataset root."""

    return EncodingPipeline(dataset)


def _prepare_subject(
    pipeline: EncodingPipeline,
    *,
    subject: str,
    target: str,
    conditions: dict[str, list[object]],
    expression: dict[str, list[object]],
) -> dict[str, object]:
    epochs, metadata = load_subject_epochs_and_metadata(
        pipeline.dataset,
        subject,
        preload=True,
    )
    metadata = assign_metadata_variables(metadata, pipeline._metadata_variables)
    selected, training_labels, expression_labels, rows = select_trials(
        metadata,
        target=target,
        conditions=conditions,
        expression=expression,
        **pipeline._trial_selection,
    )
    preparation = pipeline._epoch_preparation
    drop = {
        channel
        for channel, channel_type in zip(epochs.ch_names, epochs.get_channel_types())
        if channel_type in preparation["drop_channel_types"]
    }
    drop.update(
        channel for channel in preparation["drop_channels"] if channel in epochs.ch_names
    )
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

    trials = selected.rename(columns={"subject_index": "subject", "epoch_index": "trial"})
    trials.insert(2, "training_group", training_labels)
    trials.insert(3, "expression_group", expression_labels)
    leading = ["subject", "trial", "training_group", "expression_group"]
    trials = trials[[*leading, *[column for column in trials.columns if column not in leading]]]
    return {
        "data": data,
        "metadata": selected,
        "training_labels": training_labels,
        "expression_labels": expression_labels,
        "trials": trials,
        "channels": channels,
        "time_bins": time_bins,
    }


def _name_coefficient_channels(encoded: SubjectEncoding, channels: pd.DataFrame) -> None:
    table = encoded.tables["coefficients"].copy()
    names = dict(enumerate(channels["channel"].tolist()))
    table["channel"] = table.pop("channel_index").map(names)
    encoded.tables["coefficients"] = table[
        ["subject", "time", "channel", "predictor", "beta"]
    ]


def _input_fingerprint(pipeline: EncodingPipeline, subject: str) -> str:
    row = pipeline._manifest.loc[
        pipeline._manifest["subject_index"].astype(str).eq(subject)
    ].iloc[0]
    paths = [
        path
        for path in subject_dataset_paths(pipeline.dataset, subject).values()
        if path is not None and path.exists()
    ]
    metadata_variables = None
    if pipeline._metadata_variables:
        metadata = load_subject_metadata(pipeline.dataset, subject)
        metadata = assign_metadata_variables(metadata, pipeline._metadata_variables)
        names = list(pipeline._metadata_variables)
        table = metadata.loc[:, names]
        metadata_variables = {
            "columns": names,
            "dtypes": [str(table[name].dtype) for name in names],
            "values": table.to_json(
                orient="split",
                date_format="iso",
                double_precision=15,
                default_handler=repr,
            ),
        }
    return fingerprint(
        {
            "manifest_input": row["input_fingerprint"],
            "manifest_pipeline": row["pipeline_fingerprint"],
            "files": fingerprint_files(paths, root=pipeline.dataset),
            "metadata_variables": metadata_variables,
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
