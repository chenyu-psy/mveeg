"""Stable provenance serialization and fingerprints."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path

import numpy as np


def jsonable(value: object) -> object:
    """Convert supported provenance values to deterministic JSON values.

    Provenance accepts only stable data. Callables, sets, and arbitrary
    objects are rejected instead of being serialized with ``repr``.
    """

    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Provenance mapping keys must be strings.")
            output[key] = jsonable(item)
        return output
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Provenance numbers must be finite.")
        return value
    raise TypeError(
        "Unsupported provenance value "
        f"{type(value).__name__}; use JSON-like, pathlib.Path, or NumPy values."
    )


def fingerprint(value: object) -> str:
    """Return a stable SHA-256 fingerprint for supported provenance values."""

    payload = json.dumps(
        jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_files(paths: Sequence[str | Path], *, root: str | Path) -> str:
    """Fingerprint file paths, sizes, and mtimes relative to ``root``."""

    base = Path(root).expanduser().resolve()
    records: list[dict[str, object]] = []
    for path in sorted(Path(item).expanduser().resolve() for item in paths):
        try:
            relative = path.relative_to(base)
        except ValueError as error:
            raise ValueError(f"Fingerprint path is outside root {base}: {path}") from error
        stat = path.stat()
        records.append(
            {
                "path": relative.as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return fingerprint(records)
