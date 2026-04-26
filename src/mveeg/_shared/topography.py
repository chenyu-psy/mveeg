"""Shared scalp-topography plotting and export helpers."""

from __future__ import annotations

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import mne
import numpy as np
import pandas as pd


def _build_noninteractive_figure(
    figsize: tuple[float, float] = (5.0, 4.8),
) -> tuple[Figure, Axes]:
    """Return an Agg-backed figure that is safe for script export helpers.

    Parameters
    ----------
    figsize : tuple[float, float], optional
        Figure size in inches.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes created without registering with ``matplotlib.pyplot``.
    """

    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    axes = fig.add_subplot(111)
    return fig, axes


def plot_scalp_topography(
    *,
    channel_values: pd.DataFrame,
    info: mne.Info,
    value_col: str,
    title: str | None = None,
    cmap: str = "RdBu_r",
    vlim: tuple[float, float] | None = None,
    axes=None,
    show_colorbar: bool = True,
    colorbar_label: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot one scalp topography from channel-level values.

    Parameters
    ----------
    channel_values : pd.DataFrame
        Table with one row per channel and at least ``channel`` plus ``value_col``.
    info : mne.Info
        MNE info used for channel positions.
    value_col : str
        Column containing numeric values to render.
    title : str | None
        Optional title shown above the map.
    cmap : str
        Matplotlib colormap name.
    vlim : tuple[float, float] | None
        Optional fixed color range.
    axes : matplotlib axes | None
        Existing axes for drawing; creates a new figure when ``None``.
    show_colorbar : bool
        Whether to draw a colorbar.
    colorbar_label : str | None
        Optional colorbar label.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes containing the topography.
    """

    expected_channels = info["ch_names"]
    value_map = channel_values.set_index("channel")[value_col].to_dict()

    missing_channels = [ch for ch in expected_channels if ch not in value_map]
    extra_channels = [ch for ch in value_map if ch not in expected_channels]
    if len(missing_channels) > 0 or len(extra_channels) > 0:
        raise ValueError(
            "Topography channels did not match expected channels. "
            f"Missing: {missing_channels}; extra: {extra_channels}."
        )

    values = np.asarray([value_map[ch] for ch in expected_channels], dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("Topography values must all be finite.")

    if axes is None:
        fig, axes = _build_noninteractive_figure()
    else:
        fig = axes.figure

    if vlim is None:
        value_limit = float(np.max(np.abs(values)))
        topomap_vlim = (-value_limit, value_limit)
    else:
        topomap_vlim = vlim

    image, _ = mne.viz.plot_topomap(
        values,
        info,
        axes=axes,
        cmap=cmap,
        vlim=topomap_vlim,
        contours=0,
        sensors=False,
        mask=np.ones(len(values), dtype=bool),
        mask_params={
            "marker": "o",
            "linestyle": "None",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "linewidth": 1.5,
            "markersize": 6,
        },
        show=False,
    )
    # Some datasets place edge electrodes exactly on the head outline. Let
    # marker strokes extend beyond the axes limits so perimeter electrodes are
    # still drawn as complete circles.
    for artist in [*axes.collections, *axes.lines, *axes.patches]:
        artist.set_clip_on(False)

    if title is not None:
        axes.set_title(title)

    if show_colorbar:
        colorbar_axes = inset_axes(
            axes,
            width="70%",
            height="5%",
            loc="lower center",
            bbox_to_anchor=(0.0, -0.12, 1.0, 1.0),
            bbox_transform=axes.transAxes,
            borderpad=0,
        )
        colorbar = fig.colorbar(image, cax=colorbar_axes, orientation="horizontal")
        if colorbar_label is not None:
            colorbar.set_label(colorbar_label)

    return fig, axes


def save_window_topography_set(
    *,
    channel_df: pd.DataFrame,
    info: mne.Info,
    output_dir: str | Path,
    windows_ms: dict[str, tuple[int, int]],
    value_col: str,
    filename_prefix: str,
    title_prefix: str | None = None,
    colorbar_label: str | None = None,
    zscore_within_window: bool = False,
) -> pd.DataFrame:
    """Save one topography PNG per time window and return a manifest table.

    Parameters
    ----------
    channel_df : pd.DataFrame
        Long table with ``channel``, ``time_ms``, and ``value_col`` columns.
    info : mne.Info
        MNE info used for channel locations.
    output_dir : str | Path
        Output folder where PNG files are written.
    windows_ms : dict[str, tuple[int, int]]
        Named time windows in milliseconds.
    value_col : str
        Column name holding channel values to visualize.
    filename_prefix : str
        Prefix for generated filenames.
    title_prefix : str | None
        Optional title prefix shown on each figure.
    colorbar_label : str | None
        Optional colorbar label.
    zscore_within_window : bool
        Whether to z-score channel values within each window before plotting.

    Returns
    -------
    pd.DataFrame
        Manifest with one row per saved topography.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for window_name, (start_ms, end_ms) in windows_ms.items():
        window_rows = channel_df["time_ms"].between(start_ms, end_ms, inclusive="both")
        window_data = channel_df.loc[window_rows].copy()
        if len(window_data) == 0:
            raise ValueError(
                f"No channel values were found between {start_ms} ms and {end_ms} ms."
            )

        averaged = (
            window_data.groupby("channel", as_index=False)[value_col]
            .mean()
            .sort_values("channel")
            .reset_index(drop=True)
        )
        if zscore_within_window:
            values = averaged[value_col].to_numpy(dtype=float)
            value_std = values.std(ddof=0)
            if value_std == 0:
                averaged[value_col] = 0.0
            else:
                averaged[value_col] = (values - values.mean()) / value_std

        title = None
        if title_prefix is not None:
            title = f"{title_prefix}: {window_name} ({start_ms}-{end_ms} ms)"

        figure_path = output_dir / f"{filename_prefix}_{window_name}.png"
        fig, _ = plot_scalp_topography(
            channel_values=averaged,
            info=info,
            value_col=value_col,
            title=title,
            colorbar_label=colorbar_label,
        )
        try:
            fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        finally:
            fig.clear()

        rows.append(
            {
                "window_name": window_name,
                "start_ms": int(start_ms),
                "end_ms": int(end_ms),
                "topography_png": str(figure_path),
            }
        )

    return pd.DataFrame(rows)
