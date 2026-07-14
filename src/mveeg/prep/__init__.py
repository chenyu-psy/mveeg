"""Build, preprocess, and review manifest-backed EEG datasets."""

from . import steps
from .dataset import DatasetPipeline, open_pipeline
from .external import ExternalPipeline, init_external
from .pipeline import RawPipeline, init_pipeline
from .processing import preprocess_epochs

__all__ = [
    "DatasetPipeline",
    "ExternalPipeline",
    "RawPipeline",
    "init_external",
    "init_pipeline",
    "open_pipeline",
    "preprocess_epochs",
    "steps",
]
