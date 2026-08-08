"""Outcome-calibrated pre-VLA controller handoff research package.

The package is deliberately importable without LIBERO, MuJoCo, SAM3, Pi0.5,
CUDA, pandas, scikit-learn, or matplotlib. Environment integrations and optional
research dependencies are imported only by the entry points that need them.
"""

from rpent.research.handoff.types import (
    CandidateDecisionRecord,
    CandidateGeometry,
    ControllerIdentity,
    CostRecord,
    FeatureAvailability,
    FeatureProvenance,
    GovernorState,
    HandoffAction,
    HandoffDecision,
    HandoffState,
    OutcomeEstimate,
    OutcomeLabels,
    OutcomeRecord,
    SkillIdentity,
    TargetContext,
    TargetEstimate,
    TerminationReason,
    TrialIdentity,
)

__all__ = [
    "CandidateDecisionRecord",
    "CandidateGeometry",
    "ControllerIdentity",
    "CostRecord",
    "FeatureAvailability",
    "FeatureProvenance",
    "GovernorState",
    "HandoffAction",
    "HandoffDecision",
    "HandoffState",
    "OutcomeEstimate",
    "OutcomeLabels",
    "OutcomeRecord",
    "SkillIdentity",
    "TargetContext",
    "TargetEstimate",
    "TerminationReason",
    "TrialIdentity",
]

