"""Focused tests for the external and manifest sampling-rate contract."""

from __future__ import annotations

import json

import mne
import numpy as np
import pandas as pd
import pytest

from mveeg.prep.dataset import MANIFEST_COLUMNS, open_pipeline
from mveeg.prep.external import init_external


def test_external_array_uses_sampling_rate_in_api_provenance_and_manifest(tmp_path):
    dataset = (
        init_external(subject_index="4001", data=np.zeros((3, 2, 40)))
        .make_epochs(
            sampling_rate=250,
            ch_names=["Cz", "Pz"],
            tmin=-0.2,
        )
        .build_epochs(tmp_path / "prepared", task="task")
    )

    epochs = dataset.load_epochs("4001")
    provenance = json.loads((dataset.root / "provenance.json").read_text())
    eeg_json = json.loads(dataset.path_for_subject("4001", "eeg_json").read_text())
    make_step = provenance["pipeline"]["steps"][-1]

    assert epochs.info["sfreq"] == 250
    assert epochs.tmin == pytest.approx(-0.2)
    assert "sfreq" not in MANIFEST_COLUMNS
    assert dataset.manifest["sampling_rate"].tolist() == [250.0]
    assert make_step["sampling_rate"] == 250.0
    assert "sfreq" not in make_step
    assert eeg_json["SamplingFrequency"] == 250.0


def test_external_mne_input_history_uses_sampling_rate(tmp_path):
    info = mne.create_info(["Cz"], sfreq=128, ch_types="eeg")
    epochs = mne.EpochsArray(
        np.zeros((2, 1, 20)),
        info,
        verbose="ERROR",
    )

    dataset = init_external(subject_index="4001", data=epochs).build_epochs(
        tmp_path / "prepared",
        task="task",
    )
    provenance = json.loads((dataset.root / "provenance.json").read_text())
    input_step = provenance["pipeline"]["steps"][0]

    assert input_step["sampling_rate"] == 128.0
    assert "sfreq" not in input_step
    assert dataset.manifest["sampling_rate"].tolist() == [128.0]


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, np.bool_(True), "100"])
def test_external_array_rejects_invalid_sampling_rate(value):
    external = init_external(subject_index="4001", data=np.zeros((2, 1, 10)))

    with pytest.raises(ValueError, match="finite positive"):
        external.make_epochs(sampling_rate=value, ch_names=["Cz"])


def test_external_array_has_no_sfreq_compatibility_keyword():
    external = init_external(subject_index="4001", data=np.zeros((2, 1, 10)))

    with pytest.raises(TypeError, match="sfreq"):
        external.make_epochs(sfreq=100, ch_names=["Cz"])


def test_manifest_reader_rejects_legacy_sfreq_column(tmp_path):
    dataset = (
        init_external(subject_index="4001", data=np.zeros((2, 1, 10)))
        .make_epochs(sampling_rate=100, ch_names=["Cz"])
        .build_epochs(tmp_path / "prepared", task="task")
    )
    manifest_path = dataset.root / "manifest.tsv"
    legacy = pd.read_csv(manifest_path, sep="\t").rename(
        columns={"sampling_rate": "sfreq"}
    )
    legacy.to_csv(manifest_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="sampling_rate"):
        open_pipeline(dataset.root)
