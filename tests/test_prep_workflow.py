"""Tests for general preprocessing workflow defaults."""

from mveeg.prep.workflow import PreprocessWorkflow


def test_preprocess_workflow_does_not_assume_reference_channel():
    """A new workflow should not assume a project-specific reference channel."""
    flow = PreprocessWorkflow()

    assert flow.reref_channels is None
