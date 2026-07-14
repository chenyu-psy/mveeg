"""Multivariate encoding and decoding models for EEG research."""

from . import decoding, encoding, io, prep
from ._shared.metadata import transform_metadata

__version__ = "0.3.0"

__all__ = [
    "decoding",
    "encoding",
    "io",
    "prep",
    "transform_metadata",
]
