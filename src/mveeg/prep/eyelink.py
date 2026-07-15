"""EyeLink discovery, EDF conversion, and strict ASC reading."""

from __future__ import annotations

import shutil
import subprocess
import warnings
from pathlib import Path

import mne
import numpy as np


def eyelink_files(subject_dir: Path) -> list[Path]:
    """Return ASC files, converting EDF files without ASC counterparts."""

    files = sorted(path for path in subject_dir.iterdir() if path.is_file())
    asc_paths = [path for path in files if path.suffix.lower() == ".asc"]
    edf_paths = [path for path in files if path.suffix.lower() == ".edf"]
    if not asc_paths and not edf_paths:
        raise FileNotFoundError(f"No EyeLink .asc or .edf files found in {subject_dir}.")
    asc_stems = {path.stem.casefold() for path in asc_paths}
    pending = [path for path in edf_paths if path.stem.casefold() not in asc_stems]
    if not pending:
        return asc_paths
    converter = shutil.which("edf2asc")
    if converter is None:
        raise FileNotFoundError(
            "EyeLink EDF files require the `edf2asc` converter, but it is not on PATH: "
            f"{subject_dir}."
        )
    for edf_path in pending:
        result = subprocess.run(
            [converter, str(edf_path)], capture_output=True, text=True, check=False
        )
        generated = [
            path
            for path in subject_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".asc"
            and path.stem.casefold() == edf_path.stem.casefold()
        ]
        if not generated:
            raise RuntimeError(
                f"edf2asc did not create an ASC file for {edf_path.name} "
                f"(exit status {result.returncode}).\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    return sorted(
        path for path in subject_dir.iterdir() if path.is_file() and path.suffix.lower() == ".asc"
    )


def read_eyelink(path: Path) -> mne.io.BaseRaw:
    """Prefer MNE's public reader and fall back for the known signed-sample bug."""

    try:
        return mne.io.read_raw_eyelink(path, verbose="ERROR")
    except (RuntimeError, ValueError) as mne_error:
        try:
            raw = read_eyelink_asc_fallback(path)
        except (OSError, ValueError) as fallback_error:
            raise ValueError(
                f"Could not read {path.name} as EyeLink ASCII. "
                f"MNE reader failed: {mne_error}; fallback failed: {fallback_error}"
            ) from fallback_error
        message = str(mne_error)
        known_status_error = (
            "Expected the samples data in this file to have 7 columns of data, but got 6."
            in message
            and "xpos_left" in message
            and "pupil_right" in message
        )
        if not known_status_error:
            warnings.warn(
                f"MNE could not read {path.name}; using mveeg's strict ASC fallback ({mne_error}).",
                stacklevel=2,
            )
        return raw


def read_eyelink_asc_fallback(path: Path) -> mne.io.RawArray:
    """Read binocular samples and message markers from EyeLink ASCII."""

    sfreq: float | None = None
    samples: list[list[float]] = []
    messages: list[tuple[int, str]] = []
    with path.open("r") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("SAMPLES"):
                parts = stripped.split()
                if "RATE" in parts:
                    sfreq = float(parts[parts.index("RATE") + 1])
            elif stripped.startswith("MSG"):
                parts = stripped.split(maxsplit=2)
                if len(parts) == 3:
                    messages.append((int(parts[1]), parts[2].split()[-1]))
            else:
                parts = stripped.split()
                if len(parts) >= 7 and parts[0].isdigit():
                    try:
                        row = [float(parts[0])]
                        row.extend(np.nan if value == "." else float(value) for value in parts[1:7])
                        samples.append(row)
                    except ValueError:
                        continue
    if sfreq is None or not samples:
        raise ValueError(f"Could not find sampling rate and binocular samples in {path}.")
    array = np.asarray(samples)
    sample_clock = array[:, 0].astype(int)
    if np.any(np.diff(sample_clock) <= 0):
        raise ValueError(f"EyeLink sample timestamps are not strictly increasing in {path}.")
    sample_step = 1000.0 / sfreq
    sample_ix = np.rint((sample_clock - sample_clock[0]) / sample_step).astype(int)
    reconstructed = sample_clock[0] + sample_ix * sample_step
    if not np.allclose(reconstructed, sample_clock, atol=sample_step / 4):
        raise ValueError(f"EyeLink timestamps do not match the sampling rate in {path}.")
    data = np.full((6, sample_ix[-1] + 1), np.nan, dtype=float)
    data[:, sample_ix] = array[:, 1:].T
    info = mne.create_info(
        ["xpos_left", "ypos_left", "pupil_left", "xpos_right", "ypos_right", "pupil_right"],
        sfreq=sfreq,
        ch_types=["eyegaze", "eyegaze", "pupil", "eyegaze", "eyegaze", "pupil"],
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    messages = [
        (clock, description)
        for clock, description in messages
        if sample_clock[0] <= clock <= sample_clock[-1]
    ]
    if messages:
        event_clock = np.asarray([clock for clock, _ in messages])
        event_ix = np.rint((event_clock - sample_clock[0]) / sample_step).astype(int)
        event_ix = np.clip(event_ix, 0, data.shape[1] - 1)
        raw.set_annotations(
            mne.Annotations(
                onset=event_ix / sfreq,
                duration=np.zeros(len(messages)),
                description=[description for _, description in messages],
            )
        )
    return raw
