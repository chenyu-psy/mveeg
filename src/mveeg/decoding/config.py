"""Configuration objects for the EEG decoding workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
DEFAULT_DROP_TYPES = ("eog", "eyegaze", "pupil", "misc")


@dataclass
class DatasetConfig:
    """Dataset paths and derivative naming used by the decoding workflow."""

    data_dir: str | Path
    experiment_name: str
    subject_prefix: str = "sub"
    derivative_dirname: str = "derivatives"
    derivative_datatype: str = "eeg"
    derivative_label: str = "preprocessed"


@dataclass
class ConditionConfig:
    """Training and testing condition definitions for one decoding analysis."""

    train_cond: dict[str, list[str]] | list[str]
    test_cond: dict[str, list[str]] | list[str]
    cond_col: str


@dataclass
class TrialFilterConfig:
    """Trial-level inclusion and exclusion rules for decoding."""

    qc_col: str | None = "final_qc_category"
    keep_qc: tuple[str, ...] = ("accepted",)
    exclude_metadata: dict[str, tuple | str] | None = None


@dataclass
class DecodeParamConfig:
    """Time-window, cross-validation, and channel settings for decoding."""

    crop_time: tuple[float, float] | None = (0.0, 0.8)
    time_window_ms: int = 50
    trial_bin_size: int = 5
    n_splits: int = 5
    n_repeats: int = 20
    n_jobs: int = 1
    random_state: int = RANDOM_SEED
    drop_channel_types: tuple[str, ...] = DEFAULT_DROP_TYPES
    drop_channels: tuple[str, ...] = ()


@dataclass
class ModelConfig:
    """Model specification kept visible at the script level."""

    classifier_spec: dict | None = None
    classifier: object | None = None

    def __post_init__(self) -> None:
        """Fill in defaults for the structured classifier specification."""

        if self.classifier_spec is None:
            self.classifier_spec = {
                "backend": "sklearn",
                "model_name": "logistic_regression",
                "model_params": {
                    "solver": "lbfgs",
                    "max_iter": 1000,
                },
            }
        else:
            self.classifier_spec = dict(self.classifier_spec)
            if "backend" not in self.classifier_spec:
                self.classifier_spec["backend"] = "sklearn"
            if "model_name" not in self.classifier_spec:
                self.classifier_spec["model_name"] = "logistic_regression"
            if "model_params" not in self.classifier_spec:
                self.classifier_spec["model_params"] = {}


@dataclass
class DecodingConfig:
    """Grouped settings for one decoding analysis.

    Parameters
    ----------
    dataset : DatasetConfig
        Paths and derivative naming for the current dataset.
    conditions : ConditionConfig
        Training and testing condition mappings.
    filters : TrialFilterConfig
        Trial-level quality-control and exclusion rules.
    decode : DecodeParamConfig
        Time-window, cross-validation, and channel settings.
    model : ModelConfig
        Model specification shown in the analysis script.
    """

    dataset: DatasetConfig
    conditions: ConditionConfig
    filters: TrialFilterConfig
    decode: DecodeParamConfig
    model: ModelConfig

    def __post_init__(self) -> None:
        """Validate decoding settings after initialization."""

        if isinstance(self.conditions.train_cond, dict):
            if len(self.conditions.train_cond) == 0:
                raise ValueError("train_cond must contain at least one decoding label.")
            for label, group_values in self.conditions.train_cond.items():
                if not isinstance(group_values, list):
                    raise ValueError(
                        f"train_cond['{label}'] must be a list so the grouped trial types stay explicit."
                    )
                if len(group_values) == 0:
                    raise ValueError(
                        f"train_cond['{label}'] must contain at least one trial group."
                    )
        else:
            if len(self.conditions.train_cond) == 0:
                raise ValueError("train_cond must contain at least one decoding label.")

        if isinstance(self.conditions.test_cond, dict):
            if len(self.conditions.test_cond) == 0:
                raise ValueError("test_cond must contain at least one output group.")
            for label, group_values in self.conditions.test_cond.items():
                if not isinstance(group_values, list):
                    raise ValueError(
                        f"test_cond['{label}'] must be a list so the grouped trial types stay explicit."
                    )
                if len(group_values) == 0:
                    raise ValueError(
                        f"test_cond['{label}'] must contain at least one trial group."
                    )
        else:
            if len(self.conditions.test_cond) == 0:
                raise ValueError("test_cond must contain at least one trial group.")

        if isinstance(self.conditions.train_cond, dict):
            missing_test_groups = sorted(set(self.train_group_values()) - set(self.test_group_values()))
            if len(missing_test_groups) > 0:
                raise ValueError(
                    "Every grouped value in train_cond must also appear in test_cond. "
                    f"Missing values: {missing_test_groups}"
                )

        if self.decode.n_jobs < 1:
            raise ValueError("n_jobs must be at least 1.")

    def train_label_order(self) -> list[str]:
        """Return training labels in the configured order."""

        if isinstance(self.conditions.train_cond, dict):
            return list(self.conditions.train_cond)
        return list(self.conditions.train_cond)

    def train_group_values(self) -> list[str]:
        """Return trial groups used to define the training labels."""

        if isinstance(self.conditions.train_cond, dict):
            train_groups = []
            for group_values in self.conditions.train_cond.values():
                for value in group_values:
                    if value not in train_groups:
                        train_groups.append(value)
            return train_groups
        return list(self.conditions.train_cond)

    def test_group_order(self) -> list[str]:
        """Return the kept test groups in plotting order."""

        if isinstance(self.conditions.test_cond, dict):
            return list(self.conditions.test_cond)
        return list(self.conditions.test_cond)

    def test_group_values(self) -> list[str]:
        """Return the raw trial groups kept for testing and output."""

        if isinstance(self.conditions.test_cond, dict):
            test_groups = []
            for group_values in self.conditions.test_cond.values():
                for value in group_values:
                    if value not in test_groups:
                        test_groups.append(value)
            return test_groups
        return list(self.conditions.test_cond)

    def test_group_for_metadata_row(self, metadata: pd.DataFrame) -> np.ndarray:
        """Return output-group labels for the filtered metadata rows."""

        source_values = metadata[self.conditions.cond_col].to_numpy(dtype=object)

        if not isinstance(self.conditions.test_cond, dict):
            return source_values

        grouped_labels = np.empty(len(metadata), dtype=object)
        grouped_labels[:] = ""
        for label, group_values in self.conditions.test_cond.items():
            row_mask = np.isin(source_values, group_values)
            grouped_labels[row_mask] = label
        return grouped_labels

    def label_for_metadata_row(self, metadata: pd.DataFrame) -> np.ndarray:
        """Return training labels for the filtered metadata rows."""

        raw_labels = metadata["label"].to_numpy(dtype=object)

        if not isinstance(self.conditions.train_cond, dict):
            return raw_labels

        source_values = metadata[self.conditions.cond_col].to_numpy(dtype=object)

        mapped_labels = np.empty(len(metadata), dtype=object)
        mapped_labels[:] = ""
        for label, group_values in self.conditions.train_cond.items():
            row_mask = np.isin(source_values, group_values)
            mapped_labels[row_mask] = label
        return mapped_labels

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dictionary for logging and worker state."""

        config_dict = asdict(self)
        config_dict["dataset"]["data_dir"] = str(config_dict["dataset"]["data_dir"])
        return config_dict

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-safe dictionary for saved configuration files."""

        config_dict = self.to_dict()
        config_dict["model"]["classifier"] = None
        return config_dict
