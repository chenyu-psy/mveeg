"""Stable, unitless topography coordinates from MNE's public layout API."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd


def build_topography_coord_table(*, info: mne.Info, channels: list[str]) -> pd.DataFrame:
    """Return normalized layout-box centers in the requested channel order."""

    missing = [channel for channel in channels if channel not in info["ch_names"]]
    if missing:
        raise ValueError(f"Topography channels are missing from MNE info: {missing}")
    picks = mne.pick_channels(info["ch_names"], include=channels, ordered=True)
    picked_info = mne.pick_info(info, picks, copy=True)
    try:
        layout = mne.channels.make_eeg_layout(picked_info, exclude=[])
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            "Could not create a topography layout; saved epochs need electrode positions."
        ) from error
    positions = {
        name: position[:2] + position[2:] / 2
        for name, position in zip(layout.names, layout.pos, strict=True)
    }
    try:
        coordinates = np.asarray([positions[channel] for channel in channels], dtype=float)
    except KeyError as error:
        raise ValueError(f"MNE layout omitted channel {error.args[0]!r}.") from error
    if coordinates.shape != (len(channels), 2) or not np.isfinite(coordinates).all():
        raise ValueError("MNE layout returned invalid topography coordinates.")
    minimum = coordinates.min(axis=0)
    span = coordinates.max(axis=0) - minimum
    normalized = np.full_like(coordinates, 0.5)
    varying = span > 0
    normalized[:, varying] = (coordinates[:, varying] - minimum[varying]) / span[varying]
    return pd.DataFrame({"channel": channels, "x": normalized[:, 0], "y": normalized[:, 1]})
