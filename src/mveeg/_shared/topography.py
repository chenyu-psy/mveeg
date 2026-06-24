"""Shared topography table helpers."""

from __future__ import annotations

import mne
from mne.channels.layout import _find_topomap_coords
import numpy as np
import pandas as pd


def build_topography_coord_table(
    *,
    info: mne.Info,
    channels: list[str],
) -> pd.DataFrame:
    """Export MNE-projected electrode coordinates for R topography plotting.

    Parameters
    ----------
    info : mne.Info
        Channel metadata after applying the same channel-drop rules as the data.
    channels : list[str]
        Channel names that should appear in the output table.

    Returns
    -------
    pd.DataFrame
        Table with ``channel``, ``x``, and ``y`` columns. Coordinates are in
        millimeters, matching the R topography input files.
    """

    missing_channels = [ch_name for ch_name in channels if ch_name not in info["ch_names"]]
    if len(missing_channels) > 0:
        raise ValueError(
            "Topography coordinate export could not find these channels in the MNE info: "
            f"{missing_channels}"
        )

    picks = [info["ch_names"].index(ch_name) for ch_name in channels]
    try:
        coords = _find_topomap_coords(info, picks=picks)
    except (AttributeError, RuntimeError, ValueError) as err:
        raise ValueError(
            "Could not export topography coordinates. Make sure the saved epochs "
            "include electrode montage positions."
        ) from err

    coords = np.asarray(coords, dtype=float)
    if coords.shape != (len(channels), 2):
        raise ValueError(
            f"Expected topography coordinates to have shape {(len(channels), 2)}, "
            f"but found {coords.shape}."
        )
    if np.any(~np.isfinite(coords)):
        raise ValueError(
            "Topography coordinates contained missing or non-finite values. "
            "Make sure the saved epochs include electrode montage positions."
        )

    return pd.DataFrame(
        {
            "channel": channels,
            "x": coords[:, 0] * 1000.0,
            "y": coords[:, 1] * 1000.0,
        }
    )
