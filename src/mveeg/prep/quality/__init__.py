"""Signal eligibility, AutoReject, artifact rules, and sidecar state."""

from .autoreject import AutorejectResult, apply_autoreject
from .eligibility import EligibilityResult, check_eligibility
from .rules import ArtifactRuleResult, label_artifact_rules
from .state import QUALITY_SCHEMA_VERSION, load_quality_state, save_quality_state

__all__ = [
    "ArtifactRuleResult",
    "AutorejectResult",
    "EligibilityResult",
    "QUALITY_SCHEMA_VERSION",
    "apply_autoreject",
    "check_eligibility",
    "label_artifact_rules",
    "load_quality_state",
    "save_quality_state",
]
