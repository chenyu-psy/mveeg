"""Stateful, explicitly committed review sessions for artifact sidecars."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import (
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
    ) -> ReviewSession:
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
        return (
            keyed.set_index(list(KEY_COLUMNS))[self.group_by]
            .reindex(artifact_keys)
            .reset_index(drop=True)
        )

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
            epoch_index for epoch_index in self.target_epoch_indices if epoch_index in self._visited
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
        values = self._group_values.astype("object").where(self._group_values.notna(), "<NA>")
        counts = values.value_counts(sort=False)
        if self.group_by in {"initial_status", "final_status"}:
            items = [(status, counts[status]) for status in ARTIFACT_STATUSES if status in counts]
        else:
            items = list(counts.items())
        print(f"{self.group_by}:")
        for value, count in items:
            print(f"  {value}: {int(count)}")

    def working_table(self) -> pd.DataFrame:
        """Return artifact rows with in-memory status edits applied."""
        working = self._artifacts.copy()
        for epoch_index, status in self._pending.items():
            mask = working["subject_index"].eq(self.subject_index) & working["epoch_index"].eq(
                epoch_index
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
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not 0 <= start < len(target_epochs)
            ):
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
        visited_mask = committed["subject_index"].eq(self.subject_index) & committed[
            "epoch_index"
        ].isin(self._visited)
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
