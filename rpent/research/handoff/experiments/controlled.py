"""Controller-only experiment execution with no planner in the causal path.

The controlled runner deliberately owns only experiment semantics.  A thin
runtime adapter is responsible for creating a fresh task/reset and for
delegating the eventual learned-controller call to the existing primitive.
This keeps fake-adapter tests and server execution on the same governor API.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import Field

from rpent.research.handoff.experiments.config import ExecutionLayer
from rpent.research.handoff.experiments.manifest import TrialManifest
from rpent.research.handoff.governor import (
    GovernorAdapter,
    GovernorInvocation,
    GovernorRunResult,
    HandoffGovernor,
)
from rpent.research.handoff.types import (
    ControllerIdentity,
    HandoffRecord,
    OutcomeRecord,
    SkillIdentity,
    TrialIdentity,
)


class ControlledReset(HandoffRecord):
    """Runtime-confirmed identity of one fresh controlled reset."""

    reset_id: str | None = None
    episode_id: str
    metadata: dict[str, object] = Field(default_factory=dict)


@runtime_checkable
class ControlledAdapter(GovernorAdapter, Protocol):
    """Governor adapter with the one extra fresh-reset operation."""

    def reset_for_trial(self, trial: TrialManifest) -> ControlledReset:
        """Create a fresh reset before any policy observation or action."""


GovernorFactory = Callable[[TrialManifest], HandoffGovernor]
AdapterFactory = Callable[[TrialManifest], ControlledAdapter]


@dataclass(frozen=True, slots=True)
class ControlledRunResult:
    trial_id: str
    outcome: OutcomeRecord
    governor_result: GovernorRunResult


class ControlledRunner:
    """Run fixed task/target/skill cells while varying only handoff policy."""

    def __init__(
        self,
        *,
        governor_factory: GovernorFactory,
        adapter_factory: AdapterFactory,
        run_id: str,
        completed_trial_ids: Sequence[str] = (),
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        self._governor_factory = governor_factory
        self._adapter_factory = adapter_factory
        self._run_id = run_id
        self._completed = set(completed_trial_ids)

    def pending(self, trials: Sequence[TrialManifest]) -> tuple[TrialManifest, ...]:
        selected: list[TrialManifest] = []
        for trial in trials:
            self._validate_trial(trial)
            if trial.trial_id not in self._completed:
                selected.append(trial)
        return tuple(selected)

    def run(
        self,
        trials: Sequence[TrialManifest],
        *,
        limit: int | None = None,
    ) -> tuple[ControlledRunResult, ...]:
        pending = list(self.pending(trials))
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            pending = pending[:limit]
        results: list[ControlledRunResult] = []
        for trial in pending:
            result = self.run_one(trial)
            results.append(result)
            self._completed.add(trial.trial_id)
        return tuple(results)

    def run_one(self, trial: TrialManifest) -> ControlledRunResult:
        self._validate_trial(trial)
        adapter = self._adapter_factory(trial)
        reset = adapter.reset_for_trial(trial)
        if not isinstance(reset, ControlledReset):
            reset = ControlledReset.model_validate(reset)

        skill = SkillIdentity(
            name=trial.task.skill_name,
            semantic_target=trial.task.target_description,
            learned_controller="pi0.5",
        )
        controller = ControllerIdentity(
            method=trial.condition.method,
            implementation_version="rpent-controlled-runner/v1",
            checkpoint_id=trial.condition.checkpoint_id,
            configuration_id=trial.configuration_id,
        )
        invocation = GovernorInvocation(
            identity=TrialIdentity(
                run_id=self._run_id,
                episode_id=reset.episode_id,
                trial_id=trial.trial_id,
                invocation_id=f"{trial.trial_id}-vla-0001",
                suite=trial.task.suite,
                task_id=trial.task.task,
                seed=trial.task.seed,
                reset_id=reset.reset_id or trial.task.reset_id,
                repeat_index=trial.repeat_index,
            ),
            skill=skill,
            controller=controller,
            vla_kwargs={"prompt": trial.task.skill_prompt},
            metadata={
                "execution_layer": ExecutionLayer.CONTROLLED.value,
                "experiment_id": trial.experiment_id,
                "condition_name": trial.condition.name,
                "target_id": trial.task.target_id,
                "target_description": trial.task.target_description,
                "runtime_reset": reset.metadata,
                "source_revision": trial.source_revision,
            },
        )
        governor = self._governor_factory(trial)
        governor_result = governor.run(adapter, invocation)
        return ControlledRunResult(
            trial_id=trial.trial_id,
            outcome=governor_result.outcome,
            governor_result=governor_result,
        )

    @staticmethod
    def _validate_trial(trial: TrialManifest) -> None:
        if trial.execution_layer is not ExecutionLayer.CONTROLLED:
            raise ValueError(f"trial is not controller-controlled: {trial.trial_id}")
        if trial.condition.method == "original_harness":
            raise ValueError("Original Harness belongs to the full-agent system layer")
        if not trial.condition.handoff_enabled:
            raise ValueError(
                "controlled policy trials must explicitly enable the isolated "
                "handoff runtime"
            )
        if trial.handoff_config_path is None:
            raise ValueError("controlled trial has no resolved handoff config")


__all__ = [
    "AdapterFactory",
    "ControlledAdapter",
    "ControlledReset",
    "ControlledRunResult",
    "ControlledRunner",
    "GovernorFactory",
]
