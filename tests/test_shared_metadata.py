"""Tests for the shared dataset-root metadata contract."""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from mveeg._shared.metadata import (
    load_subject_data_with_filters,
    load_subject_epochs_and_metadata,
    metadata_transform_spec,
    transform_metadata,
)
from mveeg.decoding.io import load_subject_decoding_data
from mveeg.encoding.workflow_model import _load_subject_arrays
from mveeg.decoding.config import (
    ConditionConfig,
    DatasetConfig,
    DecodeParamConfig,
    DecodingConfig,
    ModelConfig,
    TrialFilterConfig,
)


def test_transform_metadata_preserves_trial_identity():
    metadata = pd.DataFrame(
        {
            "subject_index": ["001", "001"],
            "epoch_index": [0, 1],
            "value": [2, 4],
        }
    )

    output = transform_metadata(
        metadata,
        lambda frame: frame.assign(double=frame["value"] * 2),
    )

    assert output["double"].tolist() == [4, 8]
    assert "double" not in metadata.columns


@pytest.mark.parametrize(
    "metadata_transform, error",
    [
        (lambda frame: frame.iloc[::-1], "values and order"),
        (lambda frame: frame.iloc[:-1], "number of trial rows"),
        (lambda frame: frame.drop(columns="epoch_index"), "identity columns"),
        (lambda frame: frame.to_dict(), "pandas DataFrame"),
    ],
)
def test_transform_metadata_rejects_identity_changes(metadata_transform, error):
    metadata = pd.DataFrame(
        {"subject_index": ["001", "001"], "epoch_index": [0, 1]}
    )

    with pytest.raises((TypeError, ValueError), match=error):
        transform_metadata(metadata, metadata_transform)


def test_transform_metadata_rejects_existing_column_removal():
    metadata = pd.DataFrame(
        {
            "subject_index": ["001", "001"],
            "epoch_index": [0, 1],
            "condition": ["a", "b"],
        }
    )

    with pytest.raises(ValueError, match="cannot remove existing columns"):
        transform_metadata(metadata, lambda frame: frame.drop(columns="condition"))


def test_metadata_transform_requires_stable_fingerprint_fields():
    transform = lambda frame: frame  # noqa: E731

    with pytest.raises(ValueError, match="metadata_transform_name"):
        metadata_transform_spec(transform, name=None, version="1")
    assert metadata_transform_spec(transform, name="derive_load", version="1") == {
        "name": "derive_load",
        "version": "1",
    }


def test_dataset_loader_key_merges_artifacts_before_transform_and_filters(tmp_path):
    root = _write_dataset(tmp_path)
    cfg = DecodingConfig(
        dataset=DatasetConfig(data_dir=root, experiment_name="task"),
        conditions=ConditionConfig(
            train_cond={"a": ["a"], "b": ["b"]},
            test_cond={"a": ["a"], "b": ["b"]},
            cond_col="condition",
        ),
        filters=TrialFilterConfig(qc_col="final_status", keep_qc=("accepted",)),
        decode=DecodeParamConfig(
            crop_time=None,
            n_repeats=1,
            drop_channel_types=(),
        ),
        model=ModelConfig(),
    )

    data, labels, _times, _channels, metadata = load_subject_data_with_filters(
        "001",
        cfg,
        return_metadata=True,
        metadata_transform=lambda frame: frame.assign(
            condition=frame["raw_condition"].str.lower()
        ),
    )

    assert data.shape[0] == 2
    assert labels.tolist() == ["a", "b"]
    assert metadata["epoch_index"].tolist() == [0, 3]
    assert metadata["final_status"].tolist() == ["accepted", "accepted"]
    assert "reviewed" not in metadata.columns
    assert "channel_Fz" not in metadata.columns


def test_dataset_loader_rejects_artifact_key_mismatch(tmp_path):
    root = _write_dataset(tmp_path)
    artifacts_path = next(root.glob("sub-001/eeg/*artifacts.tsv"))
    artifacts = pd.read_csv(artifacts_path, sep="\t", dtype={"subject_index": "string"})
    artifacts.iloc[:-1].to_csv(artifacts_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="exactly the same trial keys"):
        load_subject_epochs_and_metadata(root, "001", preload=False)


def test_dataset_loader_rejects_nonidentity_metadata_mirror_mismatch(tmp_path):
    root = _write_dataset(tmp_path)
    events_path = next(root.glob("sub-001/eeg/*events.tsv"))
    events = pd.read_csv(events_path, sep="\t", dtype={"subject_index": "string"})
    events.loc[1, "raw_condition"] = "changed"
    events.to_csv(events_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="same base metadata values"):
        load_subject_epochs_and_metadata(root, "001", preload=False)


def test_dataset_loader_accepts_mne_float_rounding_and_uses_events_metadata(tmp_path):
    root = _write_dataset(tmp_path)

    epochs, metadata = load_subject_epochs_and_metadata(root, "001", preload=False)

    assert epochs.metadata["continuous_value"].reset_index(drop=True).equals(
        metadata["continuous_value"]
    )


def test_dataset_loader_rejects_float_mismatch_beyond_experiment_precision(tmp_path):
    root = _write_dataset(tmp_path)
    events_path = next(root.glob("sub-001/eeg/*events.tsv"))
    events = pd.read_csv(events_path, sep="\t", dtype={"subject_index": "string"})
    events.loc[1, "continuous_value"] += 0.0006
    events.to_csv(events_path, sep="\t", index=False)

    with pytest.raises(
        ValueError,
        match=r"epoch_index=1.*column 'continuous_value'.*FIF=.*events.tsv=",
    ):
        load_subject_epochs_and_metadata(root, "001", preload=False)


def test_dataset_loader_does_not_discover_unlisted_artifact_sidecar(tmp_path):
    root = _write_dataset(tmp_path)
    manifest_path = root / "manifest.tsv"
    manifest = pd.read_csv(manifest_path, sep="\t", dtype={"subject_index": "string"})
    manifest["artifacts_path"] = ""
    manifest.to_csv(manifest_path, sep="\t", index=False)

    _, metadata = load_subject_epochs_and_metadata(root, "001", preload=False)

    assert "final_status" not in metadata.columns


def test_decoding_loader_calls_shared_subject_index_api(tmp_path):
    root = _write_dataset(tmp_path)
    cfg = DecodingConfig(
        dataset=DatasetConfig(data_dir=root, experiment_name="task"),
        conditions=ConditionConfig(
            train_cond={"a": ["a"], "b": ["b"]},
            test_cond={"a": ["a"], "b": ["b"]},
            cond_col="condition",
        ),
        filters=TrialFilterConfig(qc_col="final_status", keep_qc=("accepted",)),
        decode=DecodeParamConfig(crop_time=None, n_repeats=1, drop_channel_types=()),
        model=ModelConfig(),
    )

    data, labels, *_ = load_subject_decoding_data(
        "001",
        cfg,
        metadata_transform=lambda frame: frame.assign(
            condition=frame["raw_condition"].str.lower()
        ),
    )

    assert data.shape[0] == 2
    assert labels.tolist() == ["a", "b"]


def test_encoding_loader_calls_shared_subject_index_api(tmp_path):
    root = _write_dataset(tmp_path)
    cfg = DecodingConfig(
        dataset=DatasetConfig(data_dir=root, experiment_name="task"),
        conditions=ConditionConfig(
            train_cond={"a": ["A"], "b": ["B"]},
            test_cond={"a": ["A"], "b": ["B"]},
            cond_col="raw_condition",
        ),
        filters=TrialFilterConfig(qc_col="final_status", keep_qc=("accepted",)),
        decode=DecodeParamConfig(crop_time=None, n_repeats=1, drop_channel_types=()),
        model=ModelConfig(),
    )

    data, conditions, times, channels, metadata = _load_subject_arrays(
        subject_id="001",
        loader_cfg=cfg,
        source_condition_col="raw_condition",
        source_to_condition={"A": "a", "B": "b"},
        time_window_ms=20,
    )

    assert data.shape == (2, 2, 2)
    assert conditions.tolist() == ["a", "b"]
    assert times.tolist() == [0.01, 0.03]
    assert channels == ["Fz", "Cz"]
    assert metadata["epoch_index"].tolist() == [0, 3]


def _write_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    eeg_dir = root / "sub-001" / "eeg"
    eeg_dir.mkdir(parents=True)
    stem = "sub-001_task-task_desc-preprocessed"
    task_stem = "sub-001_task-task"
    epochs_path = eeg_dir / f"{stem}_epo.fif"
    events_path = eeg_dir / f"{task_stem}_events.tsv"
    eeg_json_path = eeg_dir / f"{task_stem}_eeg.json"
    artifacts_path = eeg_dir / f"{task_stem}_desc-artifacts.tsv"

    info = mne.create_info(["Fz", "Cz"], sfreq=100, ch_types="eeg")
    events = np.column_stack([np.arange(4) * 10, np.zeros(4, dtype=int), np.ones(4, dtype=int)])
    metadata = pd.DataFrame(
        {
            "subject_index": ["001"] * 4,
            "epoch_index": [0, 1, 2, 3],
            "raw_condition": ["A", "B", "A", "B"],
            "continuous_value": [
                0.8275291992661313,
                0.593530612259512,
                0.5481671573183453,
                0.8483204297152385,
            ],
        }
    )
    epochs = mne.EpochsArray(
        np.arange(40, dtype=float).reshape(4, 2, 5),
        info,
        events=events,
        event_id={"trial": 1},
        tmin=0,
        metadata=metadata,
        verbose=False,
    )
    epochs.save(epochs_path, overwrite=True, verbose=False)

    metadata.to_csv(events_path, sep="\t", index=False)
    eeg_json_path.write_text("{}", encoding="utf-8")
    pd.DataFrame(
        {
            "subject_index": ["001"] * 4,
            "epoch_index": [3, 1, 0, 2],
            "initial_status": ["accepted", "rejected", "accepted", "review"],
            "final_status": ["accepted", "rejected", "accepted", "review"],
            "epoch_reasons": [pd.NA, "large_p2p", pd.NA, "high_frequency_noise"],
            "reviewed": [False] * 4,
            "channel_Fz": [pd.NA, "large_p2p", pd.NA, "high_frequency_noise"],
        }
    ).to_csv(artifacts_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "subject_index": ["001"],
            "task": ["task"],
            "stage": ["preprocessed"],
            "n_epochs": [4],
            "epochs_path": [epochs_path.relative_to(root).as_posix()],
            "events_path": [events_path.relative_to(root).as_posix()],
            "eeg_json_path": [eeg_json_path.relative_to(root).as_posix()],
            "artifacts_path": [artifacts_path.relative_to(root).as_posix()],
            "input_fingerprint": ["input"],
            "pipeline_fingerprint": ["pipeline"],
        }
    ).to_csv(root / "manifest.tsv", sep="\t", index=False)
    return root
