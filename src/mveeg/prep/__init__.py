"""EEG preparation helpers for importing, cleaning, and reviewing data.

The public name is short on purpose: scripts can use ``mveeg.prep`` without
long import lines, while the module still contains the preprocessing workflow,
quality-control helpers, and manual review visualizer.
"""

from . import core, epoched_mat, qc, visualizer, workflow

__all__ = ["core", "workflow", "qc", "epoched_mat", "visualizer"]
