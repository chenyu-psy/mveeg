"""Multivariate encoding and decoding models for EEG research."""

from importlib.metadata import version

from . import decoding, encoding, prep

__version__ = version("mveeg")

__all__ = ["decoding", "encoding", "prep", "__version__"]
