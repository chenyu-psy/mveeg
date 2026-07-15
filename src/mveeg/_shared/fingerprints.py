"""Stable fingerprints shared by dataset-producing and model workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def fingerprint(value: object) -> str:
    """Return a stable SHA-256 fingerprint for JSON-like configuration."""

    payload = json.dumps(
        jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_files(
    paths: Sequence[str | Path],
    *,
    root: str | Path,
) -> str:
    """Fingerprint file identity, size, and modification time relative to a root."""

    base = Path(root).resolve()
    records = []
    for raw_path in sorted(Path(path).resolve() for path in paths):
        stat = raw_path.stat()
        records.append(
            {
                "path": raw_path.relative_to(base).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return fingerprint(records)


def jsonable(value: object) -> object:
    """Convert common analysis objects to stable JSON-compatible values."""

    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return [jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)
