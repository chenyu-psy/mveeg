"""Topography coordinate contract."""

import mne
import numpy as np

from mveeg._analysis.topography import build_topography_coord_table


def test_topography_uses_public_mne_layout_and_preserves_requested_order(monkeypatch):
    info = mne.create_info(["F3", "F4", "Pz"], sfreq=100, ch_types="eeg")
    info.set_montage("standard_1020")
    requested = ["Pz", "F4", "F3"]
    original = mne.channels.make_eeg_layout
    calls = []

    def capture_layout(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(mne.channels, "make_eeg_layout", capture_layout)
    table = build_topography_coord_table(info=info, channels=requested)

    assert calls == [{"exclude": []}]
    assert table["channel"].tolist() == requested
    assert np.isfinite(table[["x", "y"]]).all(axis=None)
    assert table[["x", "y"]].min().eq(0).all()
    assert table[["x", "y"]].max().eq(1).all()


def test_single_channel_topography_is_finite_and_centered():
    info = mne.create_info(["Cz"], sfreq=100, ch_types="eeg")
    info.set_montage("standard_1020")

    table = build_topography_coord_table(info=info, channels=["Cz"])

    assert table.to_dict("records") == [{"channel": "Cz", "x": 0.5, "y": 0.5}]
