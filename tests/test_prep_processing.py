"""Integration tests for the 0.3 preprocessing and artifact lifecycle."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mveeg.prep.artifacts import (
    build_artifact_table,
    read_artifact_table,
    write_artifact_table,
)
from mveeg.prep.external import init_external
from mveeg.prep.processing import (
    _append_autoreject_channel_reasons,
    preprocess_epochs,
    quality_state_path,
)
from mveeg.prep.quality import load_quality_state


GAZE_GEOMETRY = {
    "viewing_distance_cm": 80.0,
    "screen_width_cm": 53.2,
    "screen_width_px": 1920,
}


def _prepared_dataset(tmp_path):
    rng = np.random.default_rng(12)
    external = init_external(
        subject_index="4001",
        data=rng.normal(scale=2e-6, size=(4, 2, 60)),
    )
    return (
        external.make_epochs(sampling_rate=100, ch_names=["Cz", "Pz"], tmin=-0.2)
        .build_epochs(tmp_path / "prepared", task="task")
    )


def _eligibility_config(absolute_value=1e-3):
    return {
        "time_window": (-0.2, 0.39),
        "eeg": {
            "p2p": 1e-3,
            "step": 1e-3,
            "absolute_value": absolute_value,
            "bad_channels": 2,
        },
    }


def _review_config(threshold=1e-3):
    return {
        "time_window": (-0.2, 0.39),
        "eeg": {
            "p2p": threshold,
            "step": threshold,
            "absolute_value": threshold,
            "bad_channels": 1,
        },
    }


def _set_prepared_gaze_geometry(dataset, geometry):
    path = dataset.root / "provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["gaze_geometry"] = geometry
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_preprocess_label_relabel_and_dataset_review_method(
    tmp_path, monkeypatch, capsys
):
    prepared = _prepared_dataset(tmp_path)

    preprocessed = preprocess_epochs(
        prepared,
        tmp_path / "preprocessed",
        eligibility=_eligibility_config(),
        autoreject=None,
    )

    saved = preprocessed.load_epochs("4001")
    assert len(saved) == 4
    assert saved.metadata["epoch_index"].tolist() == [0, 1, 2, 3]
    state_path = quality_state_path(preprocessed.root, "4001", "task")
    eligibility, autoreject = load_quality_state(state_path)
    assert eligibility.eligible.tolist() == [True] * 4
    assert autoreject["labels"].shape == (4, 2)

    preprocessed.label_artifacts(reject={}, review=_review_config())
    assert capsys.readouterr().out.split() == [
        "subject",
        "accepted",
        "rejected",
        "review",
        "4001",
        "4",
        "0",
        "0",
    ]
    artifact_path = preprocessed.path_for_subject("4001", "artifacts")
    first = read_artifact_table(artifact_path)
    assert first["initial_status"].tolist() == ["accepted"] * 4

    first.loc[0, ["final_status", "reviewed"]] = ["rejected", True]
    write_artifact_table(first, artifact_path)
    preprocessed.label_artifacts(reject={}, review=_review_config(threshold=0))
    relabeled = read_artifact_table(artifact_path)
    assert relabeled.loc[0, "initial_status"] == "review"
    assert relabeled.loc[0, "final_status"] == "rejected"
    assert bool(relabeled.loc[0, "reviewed"])
    assert relabeled.loc[1:, "final_status"].tolist() == ["review"] * 3

    captured = {}

    def fake_open(session, epochs, **kwargs):
        captured["session"] = session
        captured["epochs"] = epochs
        captured["kwargs"] = kwargs
        return "browser"

    monkeypatch.setattr("mveeg.prep.review.open_review_figure", fake_open)
    result = preprocessed.review_artifacts(
        subject_index="4001",
        group_by="final_status",
        label="review",
        time_window=(-0.1, 0.2),
        hide_channels=["Pz"],
        scalings={"eeg": 1},
    )

    assert result is None
    assert captured["session"].target_epoch_indices == (1, 2, 3)
    assert captured["kwargs"]["hide_channels"] == ["Pz"]


def test_changed_preprocessing_replaces_dense_state_and_removes_stale_artifacts(tmp_path):
    prepared = _prepared_dataset(tmp_path)
    output = tmp_path / "preprocessed"
    first = preprocess_epochs(
        prepared,
        output,
        eligibility=_eligibility_config(),
        autoreject=None,
    )
    first.label_artifacts(reject={}, review=_review_config())
    artifact_path = first.path_for_subject("4001", "artifacts")
    assert artifact_path.exists()

    changed = preprocess_epochs(
        prepared,
        output,
        eligibility=_eligibility_config(absolute_value=5e-4),
        autoreject=None,
        recompute="changed",
    )

    assert not artifact_path.exists()
    _, state = load_quality_state(quality_state_path(changed.root, "4001", "task"))
    assert state["labels"].shape == (4, 2)


def test_autoreject_channel_reasons_do_not_change_authoritative_status(tmp_path):
    """AutoReject channel diagnostics remain separate from epoch decisions."""

    epochs = _prepared_dataset(tmp_path).load_epochs("4001")
    reasons = np.full((len(epochs), len(epochs.ch_names)), "", dtype=object)
    labels = np.array(
        [
            [1, 2],
            [0, 0],
            [-1, -1],
            [0, 0],
        ],
        dtype=np.int8,
    )
    combined = _append_autoreject_channel_reasons(
        epochs,
        reasons,
        {"labels": labels, "eeg_channels": np.array(["Cz", "Pz"])},
    )
    table = build_artifact_table(
        "4001",
        epochs.metadata["epoch_index"],
        epochs.ch_names,
        rejected_reasons=combined,
        epoch_rejected=np.zeros(len(epochs), dtype=bool),
        epoch_review=np.zeros(len(epochs), dtype=bool),
    )

    assert table["initial_status"].tolist() == ["accepted"] * len(epochs)
    assert table.loc[0, "channel_Cz"] == "autoreject_bad_channel"
    assert table.loc[0, "channel_Pz"] == (
        "autoreject_bad_channel;autoreject_interpolated"
    )
    assert "autoreject_interpolated" in table.loc[0, "epoch_reasons"]


def test_changed_preprocessing_fingerprints_all_prepared_subject_files(tmp_path):
    prepared = _prepared_dataset(tmp_path)
    output = tmp_path / "preprocessed"
    preprocess_epochs(
        prepared,
        output,
        eligibility=_eligibility_config(),
        autoreject=None,
    )

    prepared.path_for_subject("4001", "eeg_json").write_text(
        '{"manually_changed": true}\n',
        encoding="utf-8",
    )
    changed = preprocess_epochs(
        prepared,
        output,
        eligibility=_eligibility_config(),
        autoreject=None,
        recompute="changed",
    )

    assert changed.run_summary.to_dict("records") == [
        {"subject_index": "4001", "status": "written"}
    ]


def test_gaze_geometry_is_carried_in_pipeline_and_artifact_fingerprints(tmp_path):
    prepared = _prepared_dataset(tmp_path)
    _set_prepared_gaze_geometry(prepared, GAZE_GEOMETRY)

    preprocessed = preprocess_epochs(
        prepared,
        tmp_path / "preprocessed",
        eligibility=_eligibility_config(),
        autoreject=None,
    )
    provenance_path = preprocessed.root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    assert "gaze_geometry" not in provenance
    assert provenance["pipeline"]["gaze_geometry"] == GAZE_GEOMETRY

    preprocessed.label_artifacts(reject={}, review=_review_config())
    labeled = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert labeled["artifact_labeling"]["gaze_geometry"] == GAZE_GEOMETRY


def test_preprocess_gaze_rule_requires_prepared_geometry_before_loading_subject(tmp_path):
    prepared = _prepared_dataset(tmp_path)
    eligibility = {
        **_eligibility_config(),
        "gaze": {"deviation_deg": 1.25, "shift_deg": 0.75},
    }

    with pytest.raises(ValueError, match="requires gaze_geometry"):
        preprocess_epochs(
            prepared,
            tmp_path / "preprocessed",
            eligibility=eligibility,
            autoreject=None,
        )

    assert not (tmp_path / "preprocessed" / "manifest.tsv").exists()


def test_missing_only_gaze_rule_does_not_require_geometry(tmp_path):
    rng = np.random.default_rng(24)
    data = np.zeros((2, 5, 60))
    data[:, 0] = rng.normal(scale=2e-6, size=(2, 60))
    data[1, 1:5, :10] = np.nan
    prepared = (
        init_external(subject_index="4001", data=data)
        .make_epochs(
            sampling_rate=100,
            ch_names=[
                "Cz",
                "xpos_left",
                "xpos_right",
                "ypos_left",
                "ypos_right",
            ],
            ch_types=["eeg", "eyegaze", "eyegaze", "eyegaze", "eyegaze"],
            tmin=-0.2,
        )
        .build_epochs(tmp_path / "prepared", task="task")
    )
    eligibility = {
        **_eligibility_config(),
        "gaze": {"max_missing_fraction": 0.1},
    }

    preprocessed = preprocess_epochs(
        prepared,
        tmp_path / "preprocessed",
        eligibility=eligibility,
        autoreject=None,
    )
    state, _ = load_quality_state(
        quality_state_path(preprocessed.root, "4001", "task")
    )

    assert state.eligible.tolist() == [True, False]


def test_label_gaze_contract_is_validated_before_subject_writes(tmp_path):
    prepared = _prepared_dataset(tmp_path)
    preprocessed = preprocess_epochs(
        prepared,
        tmp_path / "preprocessed",
        eligibility=_eligibility_config(),
        autoreject=None,
    )
    gaze_rule = {"deviation_deg": 1.0, "shift_deg": 0.75}

    with pytest.raises(ValueError, match="review.gaze requires gaze_geometry"):
        preprocessed.label_artifacts(
            reject={},
            review={**_review_config(), "gaze": gaze_rule},
        )
    with pytest.raises(ValueError, match="reject.gaze is unsupported"):
        preprocessed.label_artifacts(
            reject={"gaze": gaze_rule},
            review=_review_config(),
        )
    assert not preprocessed.path_for_subject("4001", "artifacts").exists()


def test_recompute_never_does_not_replace_persisted_gaze_geometry(tmp_path):
    prepared = _prepared_dataset(tmp_path)
    _set_prepared_gaze_geometry(prepared, GAZE_GEOMETRY)
    output = tmp_path / "preprocessed"
    preprocess_epochs(
        prepared,
        output,
        eligibility=_eligibility_config(),
        autoreject=None,
    )
    changed_geometry = {
        "viewing_distance_cm": 70.0,
        "screen_width_cm": 50.0,
        "screen_width_px": 1600,
    }
    _set_prepared_gaze_geometry(prepared, changed_geometry)

    with pytest.warns(UserWarning, match="Pipeline configuration changed"):
        reused = preprocess_epochs(
            prepared,
            output,
            eligibility=_eligibility_config(),
            autoreject=None,
            recompute="never",
        )

    persisted = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert reused.run_summary.to_dict("records") == [
        {"subject_index": "4001", "status": "reused"}
    ]
    assert persisted["pipeline"]["gaze_geometry"] == GAZE_GEOMETRY
    assert "gaze_geometry" not in persisted


def test_preprocess_rejects_legacy_geometry_even_without_gaze_rules(tmp_path):
    prepared = _prepared_dataset(tmp_path)
    _set_prepared_gaze_geometry(
        prepared,
        {"distance_mm": 800, "width_mm": 532, "resolution_x": 1920},
    )

    with pytest.raises(ValueError, match="viewing_distance_cm"):
        preprocess_epochs(
            prepared,
            tmp_path / "preprocessed",
            eligibility=_eligibility_config(),
            autoreject=None,
        )


def test_hf_reference_error_names_subject_and_channel_before_sidecar_write(tmp_path):
    rng = np.random.default_rng(48)
    repeated = np.repeat(
        rng.normal(scale=2e-6, size=(1, 2, 60)),
        repeats=4,
        axis=0,
    )
    prepared = (
        init_external(subject_index="4001", data=repeated)
        .make_epochs(sampling_rate=100, ch_names=["Cz", "Pz"], tmin=-0.2)
        .build_epochs(tmp_path / "prepared", task="task")
    )
    preprocessed = preprocess_epochs(
        prepared,
        tmp_path / "preprocessed",
        eligibility=_eligibility_config(),
        autoreject=None,
    )
    hf_noise = {
        "band": (20, 40),
        "window_duration": 0.6,
        "z_threshold": 6,
        "min_noisy_fraction": 0.2,
        "bad_channels": 1,
    }

    with pytest.raises(ValueError, match="Subject 4001.*channel 'Cz'"):
        preprocessed.label_artifacts(
            reject={"time_window": (-0.2, 0.39), "hf_noise": hf_noise},
            review=_review_config(),
        )

    assert not preprocessed.path_for_subject("4001", "artifacts").exists()
