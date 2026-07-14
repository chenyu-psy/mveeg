"""Pure preprocessing steps shared by raw and external pipelines."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
import re

import mne
import numpy as np
import pandas as pd
from mne.baseline import rescale


IDENTITY_COLUMNS = ("subject_index", "epoch_index")


def filter_table(
    table: pd.DataFrame,
    *,
    include: Mapping[str, object] | None = None,
    exclude: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Select table rows using explicit column rules.

    Parameters
    ----------
    table : pandas.DataFrame
        Trial-level input table.
    include, exclude : mapping | None
        Column-to-value rules. A collection means membership and a callable
        receives the column and must return a Boolean mask. Include rules are
        combined with AND. Rows matching every exclude rule are removed.

    Returns
    -------
    pandas.DataFrame
        A copied table with a fresh positional index.
    """
    output = table.copy()
    if include:
        mask = np.ones(len(output), dtype=bool)
        for column, expected in include.items():
            mask &= _column_mask(output, column, expected)
        output = output.loc[mask]
    if exclude:
        mask = np.ones(len(output), dtype=bool)
        for column, expected in exclude.items():
            mask &= _column_mask(output, column, expected)
        output = output.loc[~mask]
    return output.reset_index(drop=True)


def filter_eeg(
    raw: mne.io.BaseRaw,
    *,
    l_freq: float | None = None,
    h_freq: float | None = None,
    **kwargs: object,
) -> mne.io.BaseRaw:
    """Return a filtered copy of a continuous EEG recording."""
    output = raw.copy()
    if l_freq is not None or h_freq is not None:
        output.load_data(verbose="ERROR").filter(l_freq=l_freq, h_freq=h_freq, **kwargs)
    return output


def make_epochs(
    raw: mne.io.BaseRaw,
    *,
    event_id: Mapping[str, int],
    time_window: tuple[float, float],
    trial_sequences: Mapping[int, Sequence[int] | Sequence[Sequence[int]]] | None = None,
    time_zero: int | Mapping[int, int] | None = None,
    baseline: tuple[float | None, float | None] | None = None,
    sampling_rate: float | None = None,
    events: np.ndarray | pd.DataFrame | None = None,
    **kwargs: object,
) -> mne.Epochs:
    """Create epochs from annotations or an explicit MNE event table.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous signal data.
    event_id : mapping
        Event names to integer codes. With ``trial_sequences``, only entries
        whose codes are sequence keys become epoch conditions.
    time_window : tuple of float
        Epoch bounds in seconds relative to ``time_zero``.
    trial_sequences : mapping | None
        Trial code to one required event sequence, or to alternative sequences.
        Relative event times are attached as metadata.
    time_zero : int | mapping | None
        Target event code shared by all sequences, or one target code per trial
        code. ``None`` uses each trial-sequence key as its target event.
    baseline : tuple | None
        Baseline interval passed to MNE.
    sampling_rate : float | None
        Optional target sampling rate. Integer decimation is used when possible.
    events : array | DataFrame | None
        Optional event rows. Arrays use MNE's ``sample, zero, value`` layout;
        DataFrames require ``sample`` and ``value`` columns.
    **kwargs
        Additional arguments forwarded to :class:`mne.Epochs`.

    Returns
    -------
    mne.Epochs
        Preloaded epochs with event-relative metadata when sequences are used.
    """
    tmin, tmax = _normalize_time_window(time_window)
    sampling_rate = _normalize_sampling_rate(sampling_rate)
    if trial_sequences is None and time_zero is not None:
        raise ValueError("time_zero is only valid when trial_sequences is provided.")
    if trial_sequences is not None and not trial_sequences:
        raise ValueError("trial_sequences cannot be empty.")

    raw_events = extract_events(raw, event_id=event_id) if events is None else _normalize_events(events)
    if trial_sequences is not None:
        matches = _find_trial_matches(raw_events, trial_sequences, time_zero)
        if not matches:
            raise ValueError("No event sequence matched the configured trial_sequences.")
        epoch_events = np.asarray(
            [[match.lock_sample, 0, match.trial_code] for match in matches], dtype=int
        )
        metadata = _event_metadata(matches, event_id, float(raw.info["sfreq"]))
        epoch_event_id = {
            name: int(code) for name, code in event_id.items() if int(code) in trial_sequences
        }
    else:
        keep_codes = {int(code) for code in event_id.values()}
        epoch_events = raw_events[np.isin(raw_events[:, 2], list(keep_codes))]
        metadata = None
        epoch_event_id = {name: int(code) for name, code in event_id.items()}
    if len(epoch_events) == 0:
        raise ValueError("No events matched event_id.")

    native_sfreq = float(raw.info["sfreq"])
    decim = 1
    resample_after = False
    if sampling_rate is not None and not np.isclose(sampling_rate, native_sfreq):
        ratio = native_sfreq / sampling_rate
        if ratio >= 1 and np.isclose(ratio, round(ratio)):
            decim = int(round(ratio))
        else:
            resample_after = True

    epochs = mne.Epochs(
        raw,
        epoch_events,
        event_id=epoch_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        metadata=metadata,
        preload=True,
        decim=decim,
        on_missing="ignore",
        verbose="ERROR",
        **kwargs,
    )
    if sampling_rate is not None and resample_after:
        epochs.resample(sampling_rate, verbose="ERROR")
    return epochs


def sync_eyelink(
    epochs: mne.Epochs,
    eye_raw: mne.io.BaseRaw,
    *,
    event_id: Mapping[str, int],
    time_window: tuple[float, float] | None = None,
    trial_sequences: Mapping[int, Sequence[int] | Sequence[Sequence[int]]] | None = None,
    time_zero: int | Mapping[int, int] | None = None,
    baseline: tuple[float | None, float | None] | None = None,
    sampling_rate: float | None = None,
) -> mne.Epochs:
    """Epoch EyeLink data, align trial codes, and append its channels.

    Alignment uses chronological trial codes. Exact prefix and suffix matches
    are preferred so extra warm-up trials are handled deterministically.
    """
    resolved_window = (
        (float(epochs.tmin), float(epochs.tmax))
        if time_window is None
        else time_window
    )
    resolved_rate = float(epochs.info["sfreq"]) if sampling_rate is None else sampling_rate
    eye_epochs = make_epochs(
        eye_raw,
        event_id=event_id,
        time_window=resolved_window,
        trial_sequences=trial_sequences,
        time_zero=time_zero,
        baseline=None,
        sampling_rate=resolved_rate,
        reject=None,
        flat=None,
        reject_by_annotation=False,
    )
    eeg_ix, eye_ix = _ordered_code_alignment(epochs.events[:, 2], eye_epochs.events[:, 2])
    eeg_epochs = epochs[eeg_ix]
    eye_epochs = eye_epochs[eye_ix]

    if baseline is not None:
        channel_types = np.asarray(eye_epochs.get_channel_types())
        picks = np.flatnonzero(np.isin(channel_types, ["eyegaze", "pupil"]))
        if len(picks):
            rescale(
                eye_epochs._data,
                eye_epochs.times,
                baseline,
                copy=False,
                picks=picks,
                verbose="ERROR",
            )
    if "DIN" in eye_epochs.ch_names:
        eye_epochs.drop_channels(["DIN"])
    if len(eeg_epochs.times) != len(eye_epochs.times) or not np.allclose(
        eeg_epochs.times, eye_epochs.times
    ):
        raise RuntimeError("EEG and EyeLink epochs have different sample times after resampling.")
    output = eeg_epochs.copy()
    output.add_channels([eye_epochs], force_update_info=True)
    return output


def align_behavior(epochs: mne.Epochs, behavior: pd.DataFrame) -> mne.Epochs:
    """Attach behavior rows to epochs strictly by count and current order.

    No label, condition, or key matching is attempted. Any count mismatch or
    duplicate column name is an error that must be resolved before alignment.
    """
    if len(epochs) != len(behavior):
        raise ValueError(
            "Behavior alignment requires equal row counts in the existing order: "
            f"epochs={len(epochs)}, behavior={len(behavior)}."
        )
    base = _metadata(epochs)
    collisions = sorted(set(base.columns).intersection(behavior.columns))
    if collisions:
        raise ValueError(f"Behavior columns already exist in epoch metadata: {collisions}.")
    metadata = pd.concat(
        [base.reset_index(drop=True), behavior.reset_index(drop=True)], axis=1
    )
    return _with_metadata(epochs, metadata)


def select_epochs(
    epochs: mne.Epochs,
    *,
    include: Mapping[str, object] | None = None,
    exclude: Mapping[str, object] | None = None,
) -> mne.Epochs:
    """Select epochs from metadata while keeping signal, events, and rows aligned."""
    if not include and not exclude:
        return epochs.copy()
    metadata = _metadata(epochs)
    if "__mveeg_position__" in metadata:
        raise ValueError("Metadata column '__mveeg_position__' is reserved by mveeg.")
    selected = filter_table(
        metadata.assign(__mveeg_position__=np.arange(len(metadata))),
        include=include,
        exclude=exclude,
    )
    positions = selected.pop("__mveeg_position__").to_numpy(dtype=int)
    if len(positions) == 0:
        raise ValueError("Epoch selection produced an empty dataset.")
    return epochs[positions]


def drop_channels(epochs: mne.Epochs, ch_names: Sequence[str]) -> mne.Epochs:
    """Return a copy of epochs without the named channels."""
    return epochs.copy().drop_channels(list(ch_names))


def transform_metadata(
    epochs: mne.Epochs,
    transform: Callable[[pd.DataFrame], pd.DataFrame],
) -> mne.Epochs:
    """Apply a row-preserving metadata transformation.

    The transform may add or change non-identity columns. It must preserve row
    order, row count, all existing columns, and the two identity columns.
    """
    metadata = _metadata(epochs)
    marker = "__mveeg_row_order__"
    if marker in metadata:
        raise ValueError(f"Metadata column {marker!r} is reserved by mveeg.")
    marked = metadata.copy()
    marked[marker] = np.arange(len(marked), dtype=int)
    output = transform(marked.copy())
    if not isinstance(output, pd.DataFrame):
        raise TypeError("transform_metadata callable must return a pandas DataFrame.")
    if marker not in output or not np.array_equal(output[marker], marked[marker]):
        raise ValueError("transform_metadata must preserve metadata row order.")
    if len(output) != len(metadata):
        raise ValueError("transform_metadata must preserve metadata row count.")
    missing = sorted(set(metadata.columns).difference(output.columns))
    if missing:
        raise ValueError(f"transform_metadata cannot remove existing columns: {missing}.")
    for column in IDENTITY_COLUMNS:
        if column not in metadata and column in output:
            raise ValueError(f"transform_metadata cannot create reserved column {column!r}.")
        if column in metadata and not output[column].reset_index(drop=True).equals(
            metadata[column].reset_index(drop=True)
        ):
            raise ValueError(f"transform_metadata cannot modify {column!r}.")
    output = output.drop(columns=marker).reset_index(drop=True)
    return _with_metadata(epochs, output)


def merge_metadata(
    epochs: mne.Epochs,
    metadata: pd.DataFrame,
    *,
    epoch_key: str | None = None,
    metadata_key: str | None = None,
) -> mne.Epochs:
    """Merge an external metadata table by strict row order or unique keys."""
    if (epoch_key is None) != (metadata_key is None):
        raise ValueError("epoch_key and metadata_key must be provided together.")
    base = _metadata(epochs)
    incoming = metadata.copy().reset_index(drop=True)
    reserved = sorted(set(IDENTITY_COLUMNS).intersection(incoming.columns))
    if reserved:
        raise ValueError(f"External metadata cannot define reserved identity columns: {reserved}.")
    allowed_overlap = {epoch_key} if epoch_key == metadata_key and epoch_key is not None else set()
    collisions = sorted((set(base.columns) & set(incoming.columns)) - allowed_overlap)
    if collisions:
        raise ValueError(f"Metadata columns already exist: {collisions}.")

    if epoch_key is None:
        if len(base) != len(incoming):
            raise ValueError(
                "Row-wise metadata merge requires equal counts: "
                f"epochs={len(base)}, metadata={len(incoming)}."
            )
        additions = incoming
    else:
        if epoch_key not in base:
            raise ValueError(f"Epoch metadata does not contain key {epoch_key!r}.")
        if metadata_key not in incoming:
            raise ValueError(f"External metadata does not contain key {metadata_key!r}.")
        if base[epoch_key].isna().any() or incoming[metadata_key].isna().any():
            raise ValueError("Metadata merge keys cannot contain missing values.")
        if not base[epoch_key].is_unique or not incoming[metadata_key].is_unique:
            raise ValueError("Metadata merge keys must be unique on both sides.")
        base_keys = pd.Index(base[epoch_key])
        incoming_keys = pd.Index(incoming[metadata_key])
        if set(base_keys) != set(incoming_keys):
            missing = list(base_keys.difference(incoming_keys))
            extra = list(incoming_keys.difference(base_keys))
            raise ValueError(f"Metadata merge keys differ; missing={missing}, extra={extra}.")
        additions = incoming.set_index(metadata_key).loc[base_keys].reset_index()
        if epoch_key == metadata_key:
            additions = additions.drop(columns=[metadata_key])

    output = pd.concat([base.reset_index(drop=True), additions.reset_index(drop=True)], axis=1)
    return _with_metadata(epochs, output)


def assign_identity(epochs: mne.Epochs, subject_index: str | int) -> mne.Epochs:
    """Attach the stable subject index and fresh zero-based epoch index."""
    metadata = _metadata(epochs)
    existing = sorted(set(IDENTITY_COLUMNS).intersection(metadata.columns))
    if existing:
        raise ValueError(f"Identity columns are generated at build time and already exist: {existing}.")
    metadata["subject_index"] = str(subject_index)
    metadata["epoch_index"] = np.arange(len(metadata), dtype=int)
    identity = list(IDENTITY_COLUMNS)
    metadata = metadata[identity + [column for column in metadata if column not in identity]]
    return _with_metadata(epochs, metadata)


def extract_events(
    raw: mne.io.BaseRaw,
    *,
    event_id: Mapping[str, int] | None = None,
) -> np.ndarray:
    """Extract integer events from raw annotations.

    Configured labels, numeric descriptions, and BrainVision-style descriptions
    ending in an integer code are supported.
    """
    if event_id is None:
        events, _ = mne.events_from_annotations(raw, event_id="auto", verbose="ERROR")
        return events.astype(int)

    configured = {str(name): int(code) for name, code in event_id.items()}
    allowed_codes = set(configured.values())

    def parse(description: str) -> int | None:
        if description in configured:
            return configured[description]
        trailing_code = re.search(r"(-?\d+)\s*$", description)
        if trailing_code is None:
            return None
        code = int(trailing_code.group(1))
        return code if code in allowed_codes else None

    events, _ = mne.events_from_annotations(raw, event_id=parse, verbose="ERROR")
    return events.astype(int)


@dataclass(frozen=True)
class _TrialMatch:
    """One configured trial sequence found in an event stream."""

    trial_code: int
    lock_sample: int
    samples: tuple[int, ...]
    codes: tuple[int, ...]


def _column_mask(table: pd.DataFrame, column: str, expected: object) -> np.ndarray:
    """Build one Boolean selection mask."""
    if column not in table:
        raise ValueError(f"Metadata does not contain selection column {column!r}.")
    values = table[column]
    if callable(expected):
        mask = expected(values)
    elif expected is None:
        mask = values.isna()
    elif isinstance(expected, (str, bytes)) or np.isscalar(expected):
        mask = values.eq(expected)
    else:
        mask = values.isin(expected)
    if isinstance(mask, pd.Series):
        mask = mask.fillna(False)
    mask = np.asarray(mask)
    if mask.shape != (len(table),):
        raise ValueError(f"Selection rule for {column!r} did not return one Boolean per row.")
    return mask.astype(bool)


def _metadata(epochs: mne.Epochs) -> pd.DataFrame:
    """Return copied positional metadata, creating an empty table if absent."""
    if epochs.metadata is None:
        return pd.DataFrame(index=np.arange(len(epochs)))
    return epochs.metadata.copy().reset_index(drop=True)


def _with_metadata(epochs: mne.Epochs, metadata: pd.DataFrame) -> mne.Epochs:
    """Return copied epochs with metadata assigned quietly."""
    output = epochs.copy()
    with mne.use_log_level("ERROR"):
        output.metadata = metadata.reset_index(drop=True)
    return output


def _normalize_events(events: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Normalize an explicit event table to MNE's integer layout."""
    if isinstance(events, pd.DataFrame):
        required = {"sample", "value"}
        if not required.issubset(events.columns):
            raise ValueError("Event DataFrame must contain 'sample' and 'value' columns.")
        middle = events["duration"] if "duration" in events else np.zeros(len(events))
        array = np.column_stack([events["sample"], middle, events["value"]])
    else:
        array = np.asarray(events)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Events must have shape (n_events, 3).")
    return array.astype(int, copy=False)


def _normalize_time_window(time_window: object) -> tuple[float, float]:
    """Return two finite, increasing epoch bounds."""
    try:
        values = tuple(time_window)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("time_window must contain exactly two numeric values.") from error
    if len(values) != 2 or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in values
    ):
        raise ValueError("time_window must contain exactly two numeric values.")
    tmin, tmax = (float(value) for value in values)
    if not np.isfinite(tmin) or not np.isfinite(tmax):
        raise ValueError("time_window endpoints must be finite.")
    if tmin >= tmax:
        raise ValueError("time_window must have a start earlier than its end.")
    return tmin, tmax


def _normalize_sampling_rate(sampling_rate: object | None) -> float | None:
    """Return a finite positive sampling rate without accepting booleans."""
    if sampling_rate is None:
        return None
    if isinstance(sampling_rate, (bool, np.bool_)) or not isinstance(sampling_rate, Real):
        raise ValueError("sampling_rate must be a finite positive number.")
    value = float(sampling_rate)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("sampling_rate must be a finite positive number.")
    return value


def _sequence_options(sequence: Sequence[int] | Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    """Normalize one sequence configuration to explicit alternatives."""
    if len(sequence) == 0:
        return []
    first = sequence[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return [tuple(int(value) for value in option) for option in sequence]  # type: ignore[arg-type]
    return [tuple(int(value) for value in sequence)]  # type: ignore[arg-type]


def _find_trial_matches(
    events: np.ndarray,
    trial_sequences: Mapping[int, Sequence[int] | Sequence[Sequence[int]]],
    time_zero: int | Mapping[int, int] | None,
) -> list[_TrialMatch]:
    """Find configured trial sequences without changing their chronology."""
    targets = _resolve_time_zero(trial_sequences, time_zero)
    matches: list[_TrialMatch] = []
    codes = events[:, 2]
    for start in range(len(events)):
        found: _TrialMatch | None = None
        for configured_trial_code, configured in trial_sequences.items():
            trial_code = int(configured_trial_code)
            for option in _sequence_options(configured):
                stop = start + len(option)
                if stop > len(events) or not np.array_equal(codes[start:stop], option):
                    continue
                lock = option.index(targets[trial_code])
                found = _TrialMatch(
                    trial_code=trial_code,
                    lock_sample=int(events[start + lock, 0]),
                    samples=tuple(int(value) for value in events[start:stop, 0]),
                    codes=option,
                )
                break
            if found is not None:
                break
        if found is not None:
            matches.append(found)
    return matches


def _resolve_time_zero(
    trial_sequences: Mapping[int, Sequence[int] | Sequence[Sequence[int]]],
    time_zero: int | Mapping[int, int] | None,
) -> dict[int, int]:
    """Resolve and validate one target event code for every trial sequence."""

    def event_code(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer event code.")
        return int(value)

    trial_codes = {
        event_code(code, "trial_sequences key") for code in trial_sequences
    }
    if time_zero is None:
        targets = {code: code for code in trial_codes}
    elif isinstance(time_zero, Mapping):
        targets = {
            event_code(code, "time_zero key"): event_code(target, "time_zero target")
            for code, target in time_zero.items()
        }
        if set(targets) != trial_codes:
            missing = sorted(trial_codes.difference(targets))
            extra = sorted(set(targets).difference(trial_codes))
            raise ValueError(
                "time_zero mapping must contain exactly the trial_sequences keys; "
                f"missing={missing}, extra={extra}."
            )
    else:
        target = event_code(time_zero, "time_zero")
        targets = {code: target for code in trial_codes}

    for configured_trial_code, configured in trial_sequences.items():
        trial_code = int(configured_trial_code)
        options = _sequence_options(configured)
        if not options:
            raise ValueError(f"Trial sequence {trial_code} has no alternatives.")
        target = targets[trial_code]
        for option in options:
            count = option.count(target)
            if count != 1:
                raise ValueError(
                    f"time_zero event {target} must occur exactly once in every "
                    f"alternative for trial code {trial_code}; found {count} in {option}."
                )
    return targets


def _event_metadata(
    matches: Sequence[_TrialMatch],
    event_id: Mapping[str, int],
    sfreq: float,
) -> pd.DataFrame:
    """Build trial rows containing event times relative to each time lock."""
    names = {int(code): str(name) for name, code in event_id.items()}
    rows: list[dict[str, float | int | str]] = []
    for match in matches:
        row: dict[str, float | int | str] = {
            "event_name": names.get(match.trial_code, str(match.trial_code)),
            "event_code": match.trial_code,
        }
        for sample, code in zip(match.samples, match.codes, strict=True):
            name = names.get(code, str(code))
            row[name] = (sample - match.lock_sample) / sfreq
        rows.append(row)
    return pd.DataFrame(rows)


def _ordered_code_alignment(
    eeg_codes: Sequence[int], eye_codes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Align two chronological trial-code streams, preferring edge matches."""
    eeg = np.asarray(eeg_codes, dtype=int)
    eye = np.asarray(eye_codes, dtype=int)
    if np.array_equal(eeg, eye):
        indices = np.arange(len(eeg), dtype=int)
        return indices, indices
    if len(eye) > len(eeg):
        eye_prefix = np.array_equal(eye[: len(eeg)], eeg)
        eye_suffix = np.array_equal(eye[-len(eeg) :], eeg)
        if eye_prefix != eye_suffix:
            eye_ix = (
                np.arange(len(eeg), dtype=int)
                if eye_prefix
                else np.arange(len(eye) - len(eeg), len(eye), dtype=int)
            )
            return np.arange(len(eeg), dtype=int), eye_ix
        eye_ix = _unique_subsequence_indices(eeg, eye)
        return np.arange(len(eeg), dtype=int), eye_ix
    if len(eeg) > len(eye):
        eeg_prefix = np.array_equal(eeg[: len(eye)], eye)
        eeg_suffix = np.array_equal(eeg[-len(eye) :], eye)
        if eeg_prefix != eeg_suffix:
            eeg_ix = (
                np.arange(len(eye), dtype=int)
                if eeg_prefix
                else np.arange(len(eeg) - len(eye), len(eeg), dtype=int)
            )
            return eeg_ix, np.arange(len(eye), dtype=int)
        eeg_ix = _unique_subsequence_indices(eye, eeg)
        return eeg_ix, np.arange(len(eye), dtype=int)
    raise RuntimeError("Equal-length EEG and EyeLink trial codes differ.")


def _unique_subsequence_indices(shorter: np.ndarray, longer: np.ndarray) -> np.ndarray:
    """Return the only chronological embedding, rejecting absent or ambiguous matches."""
    counts = np.zeros(len(shorter) + 1, dtype=np.uint8)
    counts[0] = 1
    for code in longer:
        for index in range(len(shorter) - 1, -1, -1):
            if shorter[index] == code:
                counts[index + 1] = min(2, int(counts[index + 1]) + int(counts[index]))
    if counts[-1] != 1:
        problem = "ambiguous" if counts[-1] > 1 else "incomplete"
        raise RuntimeError(f"EEG and EyeLink trial-code alignment is {problem}.")
    indices: list[int] = []
    cursor = 0
    for code in shorter:
        matches = np.flatnonzero(longer[cursor:] == code)
        if len(matches) == 0:
            raise RuntimeError("EEG and EyeLink trial-code alignment is incomplete.")
        cursor += int(matches[0])
        indices.append(cursor)
        cursor += 1
    return np.asarray(indices, dtype=int)
