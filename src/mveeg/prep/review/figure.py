"""Stateful, explicitly committed review sessions for artifact sidecars."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..artifacts import KEY_COLUMNS
from .session import ReviewSession


class MatplotlibReviewBrowser:
    """Minimal Matplotlib view bound to a :class:`ReviewSession`.

    Click an epoch to toggle accepted/rejected. ``r`` toggles reason labels,
    arrow keys move between windows, and ``w`` commits.
    """

    def __init__(
        self,
        session: ReviewSession,
        epochs,
        *,
        time_window: tuple[float | None, float | None] | None = None,
        hide_channels: Sequence[str] = (),
        scalings: Mapping[str, float] | None = None,
    ) -> None:
        """Validate keyed epoch alignment and prepare browser state."""
        if epochs.metadata is None:
            raise ValueError("Epochs metadata must contain subject_index and epoch_index.")
        missing = [column for column in KEY_COLUMNS if column not in epochs.metadata.columns]
        if missing:
            raise ValueError(f"Epochs metadata are missing identity columns: {missing}.")

        metadata = epochs.metadata.loc[:, list(KEY_COLUMNS)].copy()
        metadata["subject_index"] = metadata["subject_index"].astype(str)
        numeric_epoch = pd.to_numeric(metadata["epoch_index"], errors="coerce")
        if numeric_epoch.isna().any() or not np.equal(numeric_epoch, np.floor(numeric_epoch)).all():
            raise ValueError("Epochs metadata epoch_index must contain integers.")
        metadata["epoch_index"] = numeric_epoch.astype(int)
        if metadata.duplicated().any():
            raise ValueError("Epochs metadata identity keys must be unique.")

        epoch_positions = {
            (row.subject_index, int(row.epoch_index)): position
            for position, row in enumerate(metadata.itertuples(index=False))
        }
        artifact_keys = set(
            map(
                tuple,
                session.artifacts.loc[:, list(KEY_COLUMNS)].itertuples(index=False, name=None),
            )
        )
        if artifact_keys != set(epoch_positions):
            raise ValueError("Epochs metadata keys must match artifact keys exactly.")

        hidden = {str(channel) for channel in hide_channels}
        unknown = sorted(hidden.difference(epochs.ch_names))
        if unknown:
            raise ValueError(f"hide_channels contains unknown channels: {unknown}.")
        self.picks = [
            index for index, channel in enumerate(epochs.ch_names) if channel not in hidden
        ]
        if not self.picks:
            raise ValueError("At least one channel must remain visible.")

        self.session = session
        self.epochs = epochs
        self._epoch_positions = epoch_positions
        self.channel_names = [epochs.ch_names[index] for index in self.picks]
        self.channel_types = [epochs.get_channel_types()[index] for index in self.picks]
        self.scalings = dict(scalings or {})
        if any(not np.isfinite(value) or value <= 0 for value in self.scalings.values()):
            raise ValueError("scalings must contain positive finite values.")

        start, end = (None, None) if time_window is None else time_window
        start = epochs.times[0] if start is None else start
        end = epochs.times[-1] if end is None else end
        self._time_mask = (epochs.times >= start) & (epochs.times <= end)
        if start > end or not self._time_mask.any():
            raise ValueError("time_window does not overlap the epoch time axis.")

        self.figure = None
        self.axes = None
        self._flag_checkbox = None
        self._flag_callback_id = None
        self._canvas_callback_ids: list[int] = []
        self._show_flags_on_accepted = False
        self._show_reasons = False
        self._displayed_epochs: tuple[int, ...] = ()
        self._base_spacing: float | None = None
        self._spacing_scale = 1.0
        self._samples_per_epoch = int(self._time_mask.sum())
        window_times = epochs.times[self._time_mask]
        self._zero_sample = (
            int(np.argmin(np.abs(window_times)))
            if window_times[0] <= 0 <= window_times[-1]
            else None
        )
        self._event_name_by_code = {int(code): str(name) for name, code in epochs.event_id.items()}

    def open(self, *, show: bool = True) -> MatplotlibReviewBrowser:
        """Create the figure, connect controls, and optionally block on show."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons

        self.figure, self.axes = plt.subplots(figsize=(14, 8))
        self.figure.subplots_adjust(left=0.08, right=0.995, top=0.85, bottom=0.10)
        checkbox_axes = self.figure.add_axes([0.39, 0.015, 0.22, 0.05])
        self._flag_checkbox = CheckButtons(checkbox_axes, ["Show flags on accepted"], [False])
        self._flag_callback_id = self._flag_checkbox.on_clicked(self._on_flag_toggle)
        manager = self.figure.canvas.manager
        default_key_handler = getattr(manager, "key_press_handler_id", None)
        if default_key_handler is not None:
            self.figure.canvas.mpl_disconnect(default_key_handler)
        self._canvas_callback_ids = [
            self.figure.canvas.mpl_connect("button_press_event", self._on_click),
            self.figure.canvas.mpl_connect("key_press_event", self._on_key),
            self.figure.canvas.mpl_connect("close_event", self._on_close),
        ]
        self._draw()
        if show:
            plt.show()
        return self

    def _draw(self) -> None:
        """Draw the current window and mark every shown epoch visited."""
        rows = self.session.mark_displayed(window_start=self.session.current_index)
        self._displayed_epochs = tuple(rows["epoch_index"].to_numpy(dtype=int))

        positions = [
            self._epoch_positions[(self.session.subject_index, epoch_index)]
            for epoch_index in self._displayed_epochs
        ]
        data = self.epochs[positions].get_data()[:, self.picks][:, :, self._time_mask]
        channel_scale = np.array(
            [self.scalings.get(channel_type, 1.0) for channel_type in self.channel_types]
        )
        data = data * channel_scale[np.newaxis, :, np.newaxis]
        if self._base_spacing is None:
            finite = np.abs(data[np.isfinite(data)])
            amplitude = float(np.percentile(finite, 95)) if len(finite) else 0.0
            self._base_spacing = 3 * amplitude if amplitude > 0 else 1.0
        spacing = self._base_spacing * self._spacing_scale

        self.axes.clear()
        x = np.arange(len(self._displayed_epochs) * self._samples_per_epoch)
        offsets = np.arange(len(self.picks) - 1, -1, -1) * spacing
        for segment, epoch_index in enumerate(self._displayed_epochs):
            if self.session.get_status(epoch_index) == "rejected":
                start = segment * self._samples_per_epoch
                self.axes.axvspan(
                    start,
                    start + self._samples_per_epoch,
                    color="#edb74a",
                    alpha=0.4,
                )
        for channel, offset in enumerate(offsets):
            self.axes.plot(
                x,
                data[:, channel].reshape(-1) + offset,
                color="#000000",
                linewidth=0.75,
            )
        for segment, (epoch_index, row) in enumerate(
            zip(self._displayed_epochs, rows.to_dict(orient="records"))
        ):
            status = self.session.get_status(epoch_index)
            if status != "rejected" and not self._show_flags_on_accepted:
                continue
            start = segment * self._samples_per_epoch
            segment_x = start + np.arange(self._samples_per_epoch)
            for channel, offset in enumerate(offsets):
                reason = row.get(f"channel_{self.channel_names[channel]}", pd.NA)
                if pd.notna(reason) and str(reason):
                    color, linestyle = self._reason_style(reason)
                    self.axes.plot(
                        segment_x,
                        data[segment, channel] + offset,
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.0,
                    )
        for boundary in range(1, len(self._displayed_epochs)):
            self.axes.axvline(
                boundary * self._samples_per_epoch,
                color="#000000",
                linewidth=3,
                zorder=5,
            )
        if self._zero_sample is not None:
            for segment in range(len(self._displayed_epochs)):
                self.axes.axvline(
                    segment * self._samples_per_epoch + self._zero_sample,
                    color="#FF00FF",
                    linestyle="-",
                    linewidth=1.0,
                    zorder=4,
                    label="time-lock event" if segment == 0 else "_time-lock event",
                )

        centers = (
            np.arange(len(self._displayed_epochs)) * self._samples_per_epoch
            + self._samples_per_epoch / 2
        )
        event_names = [
            self._event_name_by_code.get(
                int(self.epochs.events[position, 2]),
                str(int(self.epochs.events[position, 2])),
            )
            for position in positions
        ]
        self.axes.set_xticks([])
        for center, epoch, event_name in zip(centers, self._displayed_epochs, event_names):
            self.axes.text(
                center,
                1.02,
                f"Trial {epoch}\n{event_name}\n{self.session.get_status(epoch)}",
                transform=self.axes.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
                clip_on=False,
            )
        self.axes.set_yticks(offsets, self.channel_names)
        self.axes.set_xlim(0, self.session.window_size * self._samples_per_epoch)
        self.axes.set_ylim(-spacing, offsets[0] + spacing)
        if self._show_reasons:
            self._draw_reasons(rows, centers, offsets)
        progress, total = self.session.progress
        self.axes.text(
            1.0,
            1.12,
            f"Progress {progress} / {total}",
            transform=self.axes.transAxes,
            ha="right",
            va="bottom",
        )
        reason_action = "hide reasons" if self._show_reasons else "show reasons"
        self.figure.suptitle(
            "Click trial to toggle accepted/rejected; "
            f"r={reason_action}; arrows=navigate; +/-=scale; w=save",
            y=0.98,
        )
        self.figure.canvas.draw_idle()

    @staticmethod
    def _reason_style(reason: object) -> tuple[str, str]:
        """Return a distinct line style for interpolated AutoReject channels."""

        codes = {code.strip() for code in str(reason).split(";")}
        if "autoreject_interpolated" in codes:
            return "#0072B2", "--"
        return "#FF0000", "-"

    def _draw_reasons(self, rows: pd.DataFrame, centers: np.ndarray, offsets: np.ndarray) -> None:
        """Annotate all visible epoch- and channel-level reason codes."""
        for segment, row in enumerate(rows.to_dict(orient="records")):
            epoch_reason = row["epoch_reasons"]
            if pd.notna(epoch_reason) and str(epoch_reason):
                self.axes.text(
                    centers[segment],
                    0.98,
                    str(epoch_reason),
                    transform=self.axes.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#7F0000",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.8,
                        "pad": 1,
                    },
                )
            start = segment * self._samples_per_epoch
            for channel, offset in zip(self.channel_names, offsets):
                reason = row.get(f"channel_{channel}", pd.NA)
                if pd.notna(reason) and str(reason):
                    color, _ = self._reason_style(reason)
                    self.axes.text(
                        start + 2,
                        offset,
                        str(reason),
                        color=color,
                        ha="left",
                        va="bottom",
                        fontsize=7,
                    )

    def _on_click(self, event) -> None:
        """Toggle the epoch under a left mouse click."""
        if event.inaxes is not self.axes or event.xdata is None or event.button != 1:
            return
        position = int(event.xdata // self._samples_per_epoch)
        if 0 <= position < len(self._displayed_epochs):
            epoch_index = self._displayed_epochs[position]
            status = self.session.get_status(epoch_index)
            self.session.set_status(epoch_index, "accepted" if status == "rejected" else "rejected")
            self._draw()

    def _on_flag_toggle(self, _label: str) -> None:
        """Redraw after changing accepted-trial channel highlighting."""
        self._show_flags_on_accepted = bool(self._flag_checkbox.get_status()[0])
        self._draw()

    def _on_key(self, event) -> None:
        """Handle display scale, reasons, navigation, and explicit commit."""

        if event.key in {"+", "="}:
            self._spacing_scale *= 1.2
            self._draw()
            return
        if event.key in {"-", "_"}:
            self._spacing_scale /= 1.2
            self._draw()
            return
        if event.key == "r":
            self._show_reasons = not self._show_reasons
            self._draw()
            return
        if event.key == "right":
            last_start = max(0, len(self.session.target_epoch_indices) - 1)
            self.session.current_index = min(
                self.session.current_index + self.session.window_size,
                last_start,
            )
            self._draw()
            return
        if event.key == "left":
            self.session.current_index = max(
                0, self.session.current_index - self.session.window_size
            )
            self._draw()
            return
        if event.key == "w":
            self.session.commit()
            self._draw()

    def _on_close(self, _event) -> None:
        """Discard uncommitted edits and release all session GUI resources."""
        if self.session is None:
            return

        self.session.close()
        if self._flag_checkbox is not None and self._flag_callback_id is not None:
            self._flag_checkbox.disconnect(self._flag_callback_id)
        if self.figure is not None:
            for callback_id in self._canvas_callback_ids:
                self.figure.canvas.mpl_disconnect(callback_id)

        self._flag_callback_id = None
        self._canvas_callback_ids.clear()
        self.epochs = None
        self.session = None
        self.figure = None
        self.axes = None
        self._flag_checkbox = None


def open_review_figure(
    session: ReviewSession,
    epochs,
    *,
    time_window: tuple[float | None, float | None] | None = None,
    hide_channels: Sequence[str] = (),
    scalings: Mapping[str, float] | None = None,
    show: bool = True,
) -> MatplotlibReviewBrowser:
    """Open the minimal Matplotlib frontend for an existing review session."""
    session._print_group_counts()
    browser = MatplotlibReviewBrowser(
        session,
        epochs,
        time_window=time_window,
        hide_channels=hide_channels,
        scalings=scalings,
    )
    return browser.open(show=show)
