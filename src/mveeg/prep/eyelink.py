"""EyeLink discovery, EDF conversion, and ASC reading."""

from __future__ import annotations

import shutil
import subprocess
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


def read_eyelink(path: Path) -> mne.io.RawArray:
    """Read gaze, pupil, and message markers from EyeLink ASCII.

    The recorded eyes come from the ``SAMPLES`` header, so binocular files give
    left and right channels and monocular files give channels for one eye.
    Extra sample columns (velocity, resolution, status flags) are ignored.
    """

    sfreq: float | None = None
    eyes: tuple[str, ...] | None = None
    samples: list[list[float]] = []
    messages: list[tuple[int, str]] = []
    with path.open("r") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("SAMPLES"):
                parts = stripped.split()
                if "RATE" in parts:
                    sfreq = float(parts[parts.index("RATE") + 1])
                header_eyes = tuple(eye for eye in ("left", "right") if eye.upper() in parts)
                if header_eyes:
                    if eyes is not None and header_eyes != eyes:
                        raise ValueError(
                            f"EyeLink recording blocks in {path} track different eyes: "
                            f"{list(eyes)} and {list(header_eyes)}."
                        )
                    eyes = header_eyes
            elif stripped.startswith("MSG"):
                parts = stripped.split(maxsplit=2)
                if len(parts) == 3:
                    messages.append((int(parts[1]), parts[2].split()[-1]))
            elif eyes is not None:
                parts = stripped.split()
                n_values = 3 * len(eyes)
                if len(parts) > n_values and parts[0].isdigit():
                    try:
                        row = [float(parts[0])]
                        row.extend(
                            np.nan if value == "." else float(value)
                            for value in parts[1 : n_values + 1]
                        )
                        samples.append(row)
                    except ValueError:
                        continue
    if sfreq is None or eyes is None or not samples:
        raise ValueError(f"Could not find sampling rate, tracked eyes, and samples in {path}.")
    array = np.asarray(samples)
    sample_clock = array[:, 0].astype(int)
    if np.any(np.diff(sample_clock) <= 0):
        raise ValueError(f"EyeLink sample timestamps are not strictly increasing in {path}.")
    sample_step = 1000.0 / sfreq
    sample_ix = np.rint((sample_clock - sample_clock[0]) / sample_step).astype(int)
    reconstructed = sample_clock[0] + sample_ix * sample_step
    if not np.allclose(reconstructed, sample_clock, atol=sample_step / 4):
        raise ValueError(f"EyeLink timestamps do not match the sampling rate in {path}.")
    data = np.full((3 * len(eyes), sample_ix[-1] + 1), np.nan, dtype=float)
    data[:, sample_ix] = array[:, 1:].T
    info = mne.create_info(
        [f"{kind}_{eye}" for eye in eyes for kind in ("xpos", "ypos", "pupil")],
        sfreq=sfreq,
        ch_types=["eyegaze", "eyegaze", "pupil"] * len(eyes),
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
