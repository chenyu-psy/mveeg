"""Pure preprocessing steps shared by raw and external pipelines."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import mne
import numpy as np
import pandas as pd


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
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in values
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

    trial_codes = {event_code(code, "trial_sequences key") for code in trial_sequences}
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
