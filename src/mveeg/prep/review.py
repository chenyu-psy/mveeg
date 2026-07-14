"""Stateful, explicitly committed review sessions for artifact sidecars."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import (
    ARTIFACT_COLUMNS,
    ARTIFACT_STATUSES,
    KEY_COLUMNS,
    read_artifact_table,
    validate_artifact_table,
    write_artifact_table,
)


class ReviewSession:
    """Hold one subject's manual artifact review until an explicit commit.

    Parameters
    ----------
    artifacts : pandas.DataFrame
        Canonical artifact sidecar for one subject.
    subject_index : str or int
        Subject that must match every artifact row.
    metadata : pandas.DataFrame or None
        Keyed epoch metadata used only when ``group_by`` is not an artifact
        fixed field.
    group_by, label : object or None
        Both must be supplied to select one group, or both omitted to review
        all epochs.
    window_size : int
        Default number of epochs marked as displayed together.
    artifact_path : path-like or None
        Optional sidecar path written by :meth:`commit`.
    on_commit : callable or None
        Alternative callback receiving the committed canonical table.
    """

    def __init__(
        self,
        artifacts: pd.DataFrame,
        *,
        subject_index: str | int,
        metadata: pd.DataFrame | None = None,
        group_by: str | None = None,
        label: object | None = None,
        window_size: int = 5,
        artifact_path: str | Path | None = None,
        on_commit: Callable[[pd.DataFrame], None] | None = None,
    ) -> None:
        """Initialize an isolated in-memory review session."""
        if (group_by is None) != (label is None):
            raise ValueError("group_by and label must be supplied together or both omitted.")
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        if artifact_path is not None and on_commit is not None:
            raise ValueError("Configure either artifact_path or on_commit, not both.")

        self._artifacts = validate_artifact_table(artifacts)
        self.subject_index = str(subject_index)
        stored_subject = self._artifacts["subject_index"].iat[0]
        if stored_subject != self.subject_index:
            raise ValueError(
                f"Artifact subject {stored_subject!r} does not match "
                f"subject_index {self.subject_index!r}."
            )

        self.group_by = group_by
        self.label = label
        self.window_size = window_size
        self._artifact_path = Path(artifact_path) if artifact_path is not None else None
        self._on_commit = on_commit
        self._group_values: pd.Series | None = None
        self._target_positions = self._select_target_positions(metadata)
        if len(self._target_positions) == 0:
            raise ValueError(f"No epochs match {group_by!r} == {label!r}.")

        reviewed = self._artifacts.loc[self._target_positions, "reviewed"].to_numpy(dtype=bool)
        pending = np.flatnonzero(~reviewed)
        self.start_index = int(pending[0]) if len(pending) else 0
        self.current_index = self.start_index
        self._visited: set[int] = set()
        self._pending: dict[int, str] = {}
        self._closed = False

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        subject_index: str | int,
        metadata: pd.DataFrame | None = None,
        group_by: str | None = None,
        label: object | None = None,
        window_size: int = 5,
    ) -> "ReviewSession":
        """Load a sidecar and configure commits to write back to that path."""
        return cls(
            read_artifact_table(path),
            subject_index=subject_index,
            metadata=metadata,
            group_by=group_by,
            label=label,
            window_size=window_size,
            artifact_path=path,
        )

    def _select_target_positions(self, metadata: pd.DataFrame | None) -> np.ndarray:
        """Resolve the session group using fixed artifact fields or metadata."""
        if self.group_by is None:
            return np.arange(len(self._artifacts), dtype=int)
        if self.group_by in ARTIFACT_COLUMNS:
            values = self._artifacts[self.group_by]
        else:
            values = self._metadata_group_values(metadata)
        self._group_values = values.reset_index(drop=True)
        return np.flatnonzero(values.eq(self.label).fillna(False).to_numpy(dtype=bool))

    def _metadata_group_values(self, metadata: pd.DataFrame | None) -> pd.Series:
        """Align one metadata grouping column to artifact keys."""
        if metadata is None or self.group_by not in metadata.columns:
            raise ValueError(
                f"group_by {self.group_by!r} is neither an artifact fixed field "
                "nor a supplied metadata column."
            )
        missing = [column for column in KEY_COLUMNS if column not in metadata.columns]
        if missing:
            raise ValueError(f"metadata are missing identity columns: {missing}.")

        keyed = metadata.loc[:, [*KEY_COLUMNS, self.group_by]].copy()
        if keyed["subject_index"].isna().any():
            raise ValueError("metadata subject_index cannot be missing.")
        keyed["subject_index"] = keyed["subject_index"].astype(str)
        numeric_epoch = pd.to_numeric(keyed["epoch_index"], errors="coerce")
        if numeric_epoch.isna().any() or not np.equal(numeric_epoch, np.floor(numeric_epoch)).all():
            raise ValueError("metadata epoch_index must contain integers.")
        keyed["epoch_index"] = numeric_epoch.astype(int)
        keyed = keyed.loc[keyed["subject_index"].eq(self.subject_index)]
        if keyed.loc[:, list(KEY_COLUMNS)].duplicated().any():
            raise ValueError("metadata identity keys must be unique.")

        artifact_keys = pd.MultiIndex.from_frame(self._artifacts.loc[:, list(KEY_COLUMNS)])
        metadata_keys = pd.MultiIndex.from_frame(keyed.loc[:, list(KEY_COLUMNS)])
        missing_keys = artifact_keys.difference(metadata_keys).tolist()
        extra_keys = metadata_keys.difference(artifact_keys).tolist()
        if missing_keys or extra_keys:
            raise ValueError(
                "metadata keys must match artifact keys exactly for the subject; "
                f"missing={missing_keys}, extra={extra_keys}."
            )
        return keyed.set_index(list(KEY_COLUMNS))[self.group_by].reindex(artifact_keys).reset_index(drop=True)

    def _ensure_open(self) -> None:
        """Reject operations after a session has been closed."""
        if self._closed:
            raise RuntimeError("The review session is closed.")

    @property
    def artifacts(self) -> pd.DataFrame:
        """Return the last committed artifact table as a defensive copy."""
        return self._artifacts.copy()

    @property
    def target_epoch_indices(self) -> tuple[int, ...]:
        """Return epoch identities included in this review group."""
        values = self._artifacts.loc[self._target_positions, "epoch_index"]
        return tuple(values.to_numpy(dtype=int))

    @property
    def visited_epoch_indices(self) -> tuple[int, ...]:
        """Return visited epoch identities in target-group order."""
        return tuple(
            epoch_index
            for epoch_index in self.target_epoch_indices
            if epoch_index in self._visited
        )

    @property
    def pending_changes(self) -> dict[int, str]:
        """Return unsaved manual final-status edits keyed by epoch identity."""
        return dict(self._pending)

    @property
    def current_epoch_index(self) -> int:
        """Return the first epoch identity in the current display window."""
        return self.target_epoch_indices[self.current_index]

    @property
    def is_complete(self) -> bool:
        """Return whether every epoch in the selected group has been reviewed."""
        reviewed = self._artifacts.loc[self._target_positions, "reviewed"]
        return bool(reviewed.all())

    @property
    def progress(self) -> tuple[int, int]:
        """Return persisted-or-visited and total epoch counts for this group."""
        target = self._artifacts.loc[self._target_positions]
        persisted = set(target.loc[target["reviewed"], "epoch_index"].to_numpy(dtype=int))
        return len(persisted.union(self._visited)), len(target)

    def _print_group_counts(self) -> None:
        """Print all labels for the current grouping exactly once per GUI open."""
        if self.group_by is None:
            print(f"all: {len(self._artifacts)}")
            return

        assert self._group_values is not None
        values = self._group_values.astype("object").where(
            self._group_values.notna(), "<NA>"
        )
        counts = values.value_counts(sort=False)
        if self.group_by in {"initial_status", "final_status"}:
            items = [
                (status, counts[status])
                for status in ARTIFACT_STATUSES
                if status in counts
            ]
        else:
            items = list(counts.items())
        print(f"{self.group_by}:")
        for value, count in items:
            print(f"  {value}: {int(count)}")

    def working_table(self) -> pd.DataFrame:
        """Return artifact rows with in-memory status edits applied."""
        working = self._artifacts.copy()
        for epoch_index, status in self._pending.items():
            mask = (
                working["subject_index"].eq(self.subject_index)
                & working["epoch_index"].eq(epoch_index)
            )
            working.loc[mask, "final_status"] = status
        return working

    def mark_displayed(
        self,
        indices: Sequence[int] | int | None = None,
        *,
        window_start: int | None = None,
        window_size: int | None = None,
    ) -> pd.DataFrame:
        """Mark explicitly displayed epochs or one target-local window visited.

        ``indices`` are stable ``epoch_index`` values. ``window_start`` is a
        zero-based position within the selected group, not a global row index.
        """
        self._ensure_open()
        if indices is not None and window_start is not None:
            raise ValueError("Use indices or window_start, not both.")

        target_epochs = self.target_epoch_indices
        position_by_epoch = {epoch: position for position, epoch in enumerate(target_epochs)}
        if indices is not None:
            raw_indices = [indices] if np.isscalar(indices) else list(indices)
            epoch_indices = []
            for raw_index in raw_indices:
                numeric = pd.to_numeric(pd.Series([raw_index]), errors="coerce").iat[0]
                if pd.isna(numeric) or numeric != np.floor(numeric):
                    raise ValueError("Displayed indices must be integer epoch_index values.")
                epoch_index = int(numeric)
                if epoch_index not in position_by_epoch:
                    raise ValueError(f"epoch_index {epoch_index} is outside the selected group.")
                epoch_indices.append(epoch_index)
            if not epoch_indices:
                raise ValueError("At least one displayed epoch is required.")
            self.current_index = position_by_epoch[epoch_indices[0]]
        else:
            start = self.current_index if window_start is None else window_start
            size = self.window_size if window_size is None else window_size
            if not isinstance(start, int) or isinstance(start, bool) or not 0 <= start < len(target_epochs):
                raise ValueError("window_start is outside the selected group.")
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ValueError("window_size must be a positive integer.")
            self.current_index = start
            epoch_indices = list(target_epochs[start : start + size])

        newly_visited = set(epoch_indices).difference(self._visited)
        for epoch_index in newly_visited:
            row = self._artifacts.loc[self._artifacts["epoch_index"].eq(epoch_index)].iloc[0]
            if not bool(row["reviewed"]) and row["final_status"] == "review":
                self._pending.setdefault(epoch_index, "rejected")
        self._visited.update(epoch_indices)
        working = self.working_table().set_index("epoch_index")
        return working.loc[epoch_indices].reset_index()

    def set_status(self, epoch_index: int, status: str) -> None:
        """Stage one visited epoch's manual final status in memory."""
        self._ensure_open()
        if status not in ARTIFACT_STATUSES:
            raise ValueError(f"status must be one of {ARTIFACT_STATUSES}.")
        if epoch_index not in self._visited:
            raise ValueError("An epoch must be displayed before its status can be edited.")
        self._pending[int(epoch_index)] = status

    def get_status(self, epoch_index: int) -> str:
        """Return one epoch's current in-memory final status."""
        if epoch_index not in self.target_epoch_indices:
            raise ValueError(f"epoch_index {epoch_index} is outside the selected group.")
        if epoch_index in self._pending:
            return self._pending[epoch_index]
        row = self._artifacts.loc[self._artifacts["epoch_index"].eq(epoch_index)]
        return str(row["final_status"].iat[0])

    def commit(
        self,
        *,
        path: str | Path | None = None,
        on_commit: Callable[[pd.DataFrame], None] | None = None,
    ) -> pd.DataFrame:
        """Save staged decisions and mark only visited epochs reviewed.

        The method writes only when called explicitly. A configured or
        per-call path and callback are mutually exclusive. Without either,
        the committed state remains in memory and is returned.
        """
        self._ensure_open()
        resolved_path = Path(path) if path is not None else self._artifact_path
        resolved_callback = on_commit if on_commit is not None else self._on_commit
        if resolved_path is not None and resolved_callback is not None:
            raise ValueError("Commit to a path or callback, not both.")

        committed = self.working_table()
        visited_mask = (
            committed["subject_index"].eq(self.subject_index)
            & committed["epoch_index"].isin(self._visited)
        )
        committed.loc[visited_mask, "reviewed"] = True
        committed = validate_artifact_table(committed)
        if resolved_path is not None:
            write_artifact_table(committed, resolved_path)
        elif resolved_callback is not None:
            resolved_callback(committed.copy())

        self._artifacts = committed
        self._pending.clear()
        return committed.copy()

    def close(self) -> None:
        """Close without writing, leaving uncommitted edits discarded."""
        self._closed = True
        self._pending.clear()


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
            map(tuple, session.artifacts.loc[:, list(KEY_COLUMNS)].itertuples(index=False, name=None))
        )
        if artifact_keys != set(epoch_positions):
            raise ValueError("Epochs metadata keys must match artifact keys exactly.")

        hidden = {str(channel) for channel in hide_channels}
        unknown = sorted(hidden.difference(epochs.ch_names))
        if unknown:
            raise ValueError(f"hide_channels contains unknown channels: {unknown}.")
        self.picks = [index for index, channel in enumerate(epochs.ch_names) if channel not in hidden]
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
        self._event_name_by_code = {
            int(code): str(name) for name, code in epochs.event_id.items()
        }

    def open(self, *, show: bool = True) -> "MatplotlibReviewBrowser":
        """Create the figure, connect controls, and optionally block on show."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons

        self.figure, self.axes = plt.subplots(figsize=(14, 8))
        self.figure.subplots_adjust(
            left=0.08, right=0.995, top=0.85, bottom=0.10
        )
        checkbox_axes = self.figure.add_axes([0.39, 0.015, 0.22, 0.05])
        self._flag_checkbox = CheckButtons(
            checkbox_axes, ["Show flags on accepted"], [False]
        )
        self._flag_callback_id = self._flag_checkbox.on_clicked(
            self._on_flag_toggle
        )
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
        for center, epoch, event_name in zip(
            centers, self._displayed_epochs, event_names
        ):
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
        self.axes.set_xlim(
            0, len(self._displayed_epochs) * self._samples_per_epoch
        )
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
            self.session.set_status(
                epoch_index, "accepted" if status == "rejected" else "rejected"
            )
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
