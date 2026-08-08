"""Configuration-driven experiment orchestration for controller handoff.

This package stays importable without LIBERO, CUDA, scikit-learn, pandas, or
matplotlib.  Runtime-specific launchers consume the strict manifests defined
here; importing the experiment surface never starts a service or subprocess.
"""

from rpent.research.handoff.experiments.config import (
    ConditionSpec,
    ExecutionLayer,
    ExperimentConfig,
    PlannerConfig,
    RuntimeConfig,
    TaskSpec,
    load_experiment_config,
)
from rpent.research.handoff.experiments.full_agent import (
    FullAgentChildPlan,
    build_child_plan,
    build_full_agent_command,
)
from rpent.research.handoff.experiments.lifecycle import (
    LifecycleJournal,
    TrialEventType,
    derive_resume_states,
    resumable_trials,
)
from rpent.research.handoff.experiments.manifest import (
    ExperimentManifest,
    TrialManifest,
    expand_manifest,
    load_manifest,
    write_manifest,
)
from rpent.research.handoff.experiments.preflight import (
    PreflightReport,
    run_offline_preflight,
)

__all__ = [
    "ConditionSpec",
    "ExecutionLayer",
    "ExperimentConfig",
    "ExperimentManifest",
    "FullAgentChildPlan",
    "LifecycleJournal",
    "PlannerConfig",
    "PreflightReport",
    "RuntimeConfig",
    "TaskSpec",
    "TrialEventType",
    "TrialManifest",
    "build_child_plan",
    "build_full_agent_command",
    "derive_resume_states",
    "expand_manifest",
    "load_experiment_config",
    "load_manifest",
    "resumable_trials",
    "run_offline_preflight",
    "write_manifest",
]
