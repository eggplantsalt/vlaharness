"""Bounded, recoverable, append-logged local handoff governor."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rpent.research.handoff.candidates import ObjectRelativeCandidateGenerator
from rpent.research.handoff.policies import HandoffPolicy, PolicyContext
from rpent.research.handoff.types import (
    CandidateGeometry,
    ControllerIdentity,
    CostRecord,
    FailureMode,
    GovernorState,
    GovernorTransition,
    HandoffAction,
    HandoffDecision,
    HandoffState,
    LabelSource,
    OutcomeLabels,
    OutcomeRecord,
    OutcomeSignal,
    SkillIdentity,
    TerminationReason,
    TerminationRecord,
    TimingRecord,
    TrialIdentity,
    unavailable_signal,
)


class FallbackMode(str, Enum):
    FORCE_HANDOFF = "force_handoff"
    ABORT = "abort"
    BASELINE = "baseline"


class GovernorCancelled(RuntimeError):
    """Cancellation signal adapters may raise at safe action boundaries."""


class EpisodeStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    terminated: bool = False
    truncated: bool = False


class StageResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    success: bool
    steps: int = Field(ge=0, le=1)
    distance_m: float = Field(ge=0.0)
    elapsed_s: float = Field(ge=0.0)
    achieved_state: HandoffState | None = None
    error: str | None = None

    @field_validator("error")
    @classmethod
    def validate_error(cls, value: str | None) -> str | None:
        if value == "":
            raise ValueError("stage error must be non-empty or null")
        return value


class VLAExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    result: dict[str, Any] = Field(default_factory=dict)
    elapsed_s: float = Field(ge=0.0)
    exception: str | None = None
    invocations: int = Field(default=1, ge=0)
    chunks: int | None = Field(default=None, ge=0)
    env_actions: int | None = Field(default=None, ge=0)


class GovernorInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    identity: TrialIdentity
    skill: SkillIdentity
    controller: ControllerIdentity
    vla_kwargs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    initial_costs: CostRecord = Field(default_factory=CostRecord)

    @field_validator("vla_kwargs", "metadata")
    @classmethod
    def validate_json_mapping(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("governor mappings must contain finite JSON values") from exc
        return value


class GovernorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    max_analytic_steps: int = Field(default=20, ge=0)
    max_staging_distance_m: float = Field(default=0.30, ge=0.0)
    max_local_time_s: float = Field(default=30.0, gt=0.0)
    max_step_distance_m: float = Field(default=0.025, gt=0.0)
    max_target_age_s: float = Field(default=2.0, ge=0.0)
    fallback_mode: FallbackMode = FallbackMode.FORCE_HANDOFF
    budget_fallback_mode: FallbackMode = FallbackMode.FORCE_HANDOFF
    propagate_cancellation: bool = True


class GovernorRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: OutcomeRecord
    planner_result: dict[str, Any]
    transitions: tuple[GovernorTransition, ...]


@runtime_checkable
class GovernorAdapter(Protocol):
    def observe(self, skill: SkillIdentity) -> HandoffState:
        """Extract a deployment-realistic current state from whitelisted inputs."""

    def episode_status(self) -> EpisodeStatus:
        """Return independently latched task termination/truncation."""

    def raise_if_cancelled(self) -> None:
        """Raise GovernorCancelled (or the host cancellation type) if requested."""

    def stage(
        self, candidate: CandidateGeometry, max_step_m: float
    ) -> StageResult:
        """Execute at most one bounded analytic action."""

    def execute_vla(self, invocation: GovernorInvocation) -> VLAExecutionResult:
        """Delegate to the unchanged frozen learned-controller primitive."""

    def label_outcome(self, result: VLAExecutionResult) -> OutcomeLabels:
        """Label the VLA invocation without substituting label semantics."""


@runtime_checkable
class ResearchSink(Protocol):
    def append_decision(self, decision: HandoffDecision) -> None:
        """Durably append one candidate decision."""

    def append_outcome(self, outcome: OutcomeRecord) -> None:
        """Durably append the final outcome, including failure paths."""


class NullResearchSink:
    """Explicit no-op sink for fake tests; production config should use JSONL."""

    def append_decision(self, decision: HandoffDecision) -> None:
        del decision

    def append_outcome(self, outcome: OutcomeRecord) -> None:
        del outcome


@dataclass
class InMemoryResearchSink:
    decisions: list[HandoffDecision] = field(default_factory=list)
    outcomes: list[OutcomeRecord] = field(default_factory=list)

    def append_decision(self, decision: HandoffDecision) -> None:
        self.decisions.append(decision)

    def append_outcome(self, outcome: OutcomeRecord) -> None:
        self.outcomes.append(outcome)


def _record_id(identity: TrialIdentity, reason: TerminationReason) -> str:
    canonical = json.dumps(
        [identity.model_dump(mode="json"), reason.value],
        sort_keys=True,
        separators=(",", ":"),
    )
    return "outcome-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _is_cancellation(exc: BaseException) -> bool:
    return isinstance(exc, GovernorCancelled) or exc.__class__.__name__ in {
        "ToolCancelled",
        "CancelledError",
    }


def _status_labels(status: EpisodeStatus) -> OutcomeLabels:
    return OutcomeLabels(
        primitive_success=unavailable_signal(
            "no VLA primitive completed on this governor path"
        ),
        skill_success=unavailable_signal(
            "no skill evaluator result is available on this governor path"
        ),
        task_success=OutcomeSignal(
            value=status.terminated,
            source=LabelSource.OFFICIAL_TERMINATION,
            definition="official environment termination at governor exit",
        ),
        episode_truncated=OutcomeSignal(
            value=status.truncated,
            source=LabelSource.RUNTIME,
            definition="environment truncation at governor exit",
        ),
        llm_finish=unavailable_signal(
            "local controller runner has no LLM finish declaration"
        ),
    )


class HandoffGovernor:
    """Local observe/decide/one-stage/re-observe/handoff state machine."""

    def __init__(
        self,
        *,
        policy: HandoffPolicy,
        candidate_generator: ObjectRelativeCandidateGenerator,
        config: GovernorConfig,
        sink: ResearchSink | None = None,
        fallback_policy: HandoffPolicy | None = None,
        monotonic=time.monotonic,
    ) -> None:
        if config.fallback_mode is FallbackMode.BASELINE and fallback_policy is None:
            raise ValueError("baseline fallback mode requires fallback_policy")
        self.policy = policy
        self.candidate_generator = candidate_generator
        self.config = config
        self.sink = sink or NullResearchSink()
        self.fallback_policy = fallback_policy
        self._monotonic = monotonic

    def run(
        self, adapter: GovernorAdapter, invocation: GovernorInvocation
    ) -> GovernorRunResult:
        started = self._monotonic()
        current_governor_state = GovernorState.INIT
        transitions: list[GovernorTransition] = []
        decisions: list[HandoffDecision] = []
        last_state: HandoffState | None = None
        analytic_steps = invocation.initial_costs.analytic_steps
        analytic_distance = invocation.initial_costs.analytic_distance_m
        analytic_time = invocation.initial_costs.analytic_time_s
        observe_time = 0.0
        decide_time = 0.0
        active_policy = self.policy
        using_fallback_policy = False
        vla_result: VLAExecutionResult | None = None
        handoff_started = False
        vla_started_s: float | None = None
        cancellation_to_raise: BaseException | None = None

        def transition(to_state: GovernorState, reason: str) -> None:
            nonlocal current_governor_state
            transitions.append(
                GovernorTransition(
                    sequence=len(transitions),
                    from_state=current_governor_state,
                    to_state=to_state,
                    elapsed_s=max(0.0, self._monotonic() - started),
                    reason=reason,
                )
            )
            current_governor_state = to_state

        def elapsed() -> float:
            return max(0.0, self._monotonic() - started)

        def finalize(
            *,
            reason: TerminationReason,
            failure_mode: FailureMode,
            final_state: GovernorState,
            handoff_occurred: bool,
            labels: OutcomeLabels,
            status: EpisodeStatus,
            message: str | None,
            planner_result: Mapping[str, Any] | None = None,
        ) -> GovernorRunResult:
            transition(GovernorState.RECORD_OUTCOME, "persist final governor outcome")
            ended = self._monotonic()
            vla_invocations = vla_result.invocations if vla_result is not None else 0
            vla_elapsed = vla_result.elapsed_s if vla_result is not None else 0.0
            vla_chunks = vla_result.chunks if vla_result is not None else None
            vla_actions = vla_result.env_actions if vla_result is not None else None
            total_actions = (
                analytic_steps + vla_actions if vla_actions is not None else None
            )
            outcome = OutcomeRecord(
                record_id=_record_id(invocation.identity, reason),
                identity=invocation.identity,
                skill=invocation.skill,
                controller=invocation.controller,
                pre_handoff_state=last_state if handoff_occurred else None,
                handoff_occurred=handoff_occurred,
                decision_trace=tuple(decisions),
                labels=labels,
                costs=CostRecord(
                    analytic_steps=analytic_steps,
                    analytic_distance_m=analytic_distance,
                    analytic_time_s=analytic_time,
                    vla_invocations=vla_invocations,
                    vla_chunks=vla_chunks,
                    vla_env_actions=vla_actions,
                    vla_time_s=vla_elapsed,
                    total_env_actions=total_actions,
                    total_elapsed_s=max(0.0, ended - started),
                ),
                timing=TimingRecord(
                    started_monotonic_s=started,
                    ended_monotonic_s=ended,
                    observe_time_s=observe_time,
                    decide_time_s=decide_time,
                    record_time_s=None,
                ),
                termination=TerminationRecord(
                    reason=reason,
                    failure_mode=failure_mode,
                    final_governor_state=final_state,
                    episode_terminated=status.terminated,
                    episode_truncated=status.truncated,
                    message=message,
                ),
                setup_record_id=invocation.metadata.get("setup_record_id"),
                privileged_label_record_id=invocation.metadata.get(
                    "privileged_label_record_id"
                ),
                source_revision=invocation.metadata.get("source_revision"),
                metadata={
                    key: value
                    for key, value in invocation.metadata.items()
                    if key
                    not in {
                        "setup_record_id",
                        "privileged_label_record_id",
                        "source_revision",
                    }
                },
            )
            self.sink.append_outcome(outcome)
            transition(final_state, message or reason.value)
            result_payload = dict(planner_result or {})
            result_payload.update(
                {
                    "handoff_record_id": outcome.record_id,
                    "handoff_occurred": handoff_occurred,
                    "handoff_termination_reason": reason.value,
                    "handoff_failure_mode": failure_mode.value,
                    "analytic_steps": analytic_steps,
                    "analytic_distance_m": analytic_distance,
                }
            )
            return GovernorRunResult(
                outcome=outcome,
                planner_result=result_payload,
                transitions=tuple(transitions),
            )

        def current_status() -> EpisodeStatus:
            status = adapter.episode_status()
            if not isinstance(status, EpisodeStatus):
                status = EpisodeStatus.model_validate(status)
            return status

        def execute_handoff(
            *, reason_on_success: TerminationReason = TerminationReason.HANDOFF_COMPLETED
        ) -> GovernorRunResult:
            nonlocal handoff_started, vla_result, vla_started_s
            if last_state is None:
                status = current_status()
                return finalize(
                    reason=TerminationReason.INVALID_STATE,
                    failure_mode=FailureMode.INVALID_INPUT,
                    final_state=GovernorState.ABORT,
                    handoff_occurred=False,
                    labels=_status_labels(status),
                    status=status,
                    message="cannot hand off without a valid observed state",
                )
            adapter.raise_if_cancelled()
            status = current_status()
            if status.terminated:
                return finalize(
                    reason=TerminationReason.EPISODE_TERMINATED_BEFORE_VLA,
                    failure_mode=FailureMode.TERMINATION_BEFORE_VLA,
                    final_state=GovernorState.TERMINATED,
                    handoff_occurred=False,
                    labels=_status_labels(status),
                    status=status,
                    message="episode terminated before the first VLA inference",
                )
            if status.truncated:
                return finalize(
                    reason=TerminationReason.EPISODE_TRUNCATED_BEFORE_VLA,
                    failure_mode=FailureMode.TRUNCATION,
                    final_state=GovernorState.TRUNCATED,
                    handoff_occurred=False,
                    labels=_status_labels(status),
                    status=status,
                    message="episode truncated before the first VLA inference",
                )
            transition(GovernorState.HANDOFF, "policy transferred control to frozen VLA")
            transition(GovernorState.EXECUTE_VLA, "execute unchanged learned primitive")
            handoff_started = True
            vla_started_s = self._monotonic()
            try:
                vla_result = adapter.execute_vla(invocation)
                if not isinstance(vla_result, VLAExecutionResult):
                    vla_result = VLAExecutionResult.model_validate(vla_result)
            except Exception as exc:
                if _is_cancellation(exc):
                    raise
                status = current_status()
                return finalize(
                    reason=TerminationReason.VLA_FAILURE,
                    failure_mode=(
                        FailureMode.RPC
                        if "rpc" in exc.__class__.__name__.lower()
                        else FailureMode.VLA
                    ),
                    final_state=GovernorState.VLA_FAILURE,
                    handoff_occurred=True,
                    labels=_status_labels(status),
                    status=status,
                    message=f"VLA execution raised {exc.__class__.__name__}: {exc}",
                )
            status = current_status()
            if vla_result.exception:
                return finalize(
                    reason=TerminationReason.VLA_FAILURE,
                    failure_mode=(
                        FailureMode.RPC
                        if "rpc" in vla_result.exception.lower()
                        else FailureMode.VLA
                    ),
                    final_state=GovernorState.VLA_FAILURE,
                    handoff_occurred=True,
                    labels=_status_labels(status),
                    status=status,
                    message=vla_result.exception,
                    planner_result=vla_result.result,
                )
            try:
                labels = adapter.label_outcome(vla_result)
            except Exception as exc:
                if _is_cancellation(exc):
                    raise
                labels = _status_labels(status)
                message = f"outcome labeler failed: {exc}"
            else:
                message = None
            return finalize(
                reason=reason_on_success,
                failure_mode=FailureMode.NONE,
                final_state=GovernorState.DONE,
                handoff_occurred=True,
                labels=labels,
                status=status,
                message=message,
                planner_result=vla_result.result,
            )

        def failure_or_fallback(
            *,
            reason: TerminationReason,
            failure_mode: FailureMode,
            state: GovernorState,
            message: str,
            budget: bool = False,
        ) -> GovernorRunResult | None:
            nonlocal active_policy, using_fallback_policy
            mode = (
                self.config.budget_fallback_mode if budget else self.config.fallback_mode
            )
            status = current_status()
            if status.terminated or status.truncated:
                final = (
                    GovernorState.TERMINATED
                    if status.terminated
                    else GovernorState.TRUNCATED
                )
                terminal_reason = (
                    TerminationReason.EPISODE_TERMINATED_BEFORE_VLA
                    if status.terminated
                    else TerminationReason.EPISODE_TRUNCATED_BEFORE_VLA
                )
                return finalize(
                    reason=terminal_reason,
                    failure_mode=(
                        FailureMode.TERMINATION_BEFORE_VLA
                        if status.terminated
                        else FailureMode.TRUNCATION
                    ),
                    final_state=final,
                    handoff_occurred=False,
                    labels=_status_labels(status),
                    status=status,
                    message=message,
                )
            if mode is FallbackMode.FORCE_HANDOFF and last_state is not None:
                transition(GovernorState.FALLBACK, message)
                return execute_handoff(
                    reason_on_success=TerminationReason.FALLBACK_FORCED_HANDOFF
                )
            if (
                mode is FallbackMode.BASELINE
                and self.fallback_policy is not None
                and not using_fallback_policy
                and not budget
            ):
                transition(GovernorState.FALLBACK, message)
                active_policy = self.fallback_policy
                using_fallback_policy = True
                transition(GovernorState.OBSERVE, "re-observe under configured baseline fallback")
                return None
            return finalize(
                reason=reason,
                failure_mode=failure_mode,
                final_state=state,
                handoff_occurred=False,
                labels=_status_labels(status),
                status=status,
                message=message,
            )

        transition(GovernorState.OBSERVE, "start local physical handoff loop")
        result: GovernorRunResult | None = None
        try:
            while result is None:
                adapter.raise_if_cancelled()
                status = current_status()
                if status.terminated:
                    result = finalize(
                        reason=TerminationReason.EPISODE_TERMINATED_BEFORE_VLA,
                        failure_mode=FailureMode.TERMINATION_BEFORE_VLA,
                        final_state=GovernorState.TERMINATED,
                        handoff_occurred=False,
                        labels=_status_labels(status),
                        status=status,
                        message="episode terminated during analytic staging",
                    )
                    break
                if status.truncated:
                    result = finalize(
                        reason=TerminationReason.EPISODE_TRUNCATED_BEFORE_VLA,
                        failure_mode=FailureMode.TRUNCATION,
                        final_state=GovernorState.TRUNCATED,
                        handoff_occurred=False,
                        labels=_status_labels(status),
                        status=status,
                        message="episode truncated during analytic staging",
                    )
                    break
                if last_state is not None and (
                    analytic_steps >= self.config.max_analytic_steps
                    or analytic_distance >= self.config.max_staging_distance_m
                    or elapsed() >= self.config.max_local_time_s
                ):
                    result = failure_or_fallback(
                        reason=TerminationReason.BUDGET_EXHAUSTED,
                        failure_mode=FailureMode.TIMEOUT,
                        state=GovernorState.ABORT,
                        message="local analytic staging budget exhausted",
                        budget=True,
                    )
                    continue

                observe_started = self._monotonic()
                try:
                    observed = adapter.observe(invocation.skill)
                    if not isinstance(observed, HandoffState):
                        observed = HandoffState.model_validate(observed)
                    last_state = observed
                    policy_requires_target = bool(
                        getattr(active_policy, "requires_target", True)
                    )
                    if policy_requires_target and (
                        observed.target is None
                        or observed.target.estimate.position_m is None
                        or observed.target.estimate.age_s > self.config.max_target_age_s
                    ):
                        raise ValueError("target estimate is unavailable or stale")
                except Exception as exc:
                    if _is_cancellation(exc):
                        raise
                    observe_time += max(0.0, self._monotonic() - observe_started)
                    result = failure_or_fallback(
                        reason=TerminationReason.PERCEPTION_FAILURE,
                        failure_mode=FailureMode.PERCEPTION,
                        state=GovernorState.PERCEPTION_FAILURE,
                        message=f"observation/target provider failed: {exc}",
                    )
                    continue
                observe_time += max(0.0, self._monotonic() - observe_started)
                transition(GovernorState.DECIDE, "evaluate current and future candidates")
                decision_started = self._monotonic()
                try:
                    candidates = self.candidate_generator.generate(last_state)
                    context = PolicyContext(
                        current_state=last_state,
                        candidates=candidates,
                        decision_sequence=len(decisions),
                        previous_action=(decisions[-1].action if decisions else None),
                    )
                    decision = active_policy.decide(context)
                except Exception as exc:
                    decide_time += max(0.0, self._monotonic() - decision_started)
                    result = failure_or_fallback(
                        reason=TerminationReason.INVALID_MODEL_PREDICTION,
                        failure_mode=FailureMode.INVALID_PREDICTION,
                        state=GovernorState.ABORT,
                        message=f"policy/candidate evaluation failed: {exc}",
                    )
                    continue
                decide_time += max(0.0, self._monotonic() - decision_started)
                decisions.append(decision)
                self.sink.append_decision(decision)

                if decision.action is HandoffAction.HANDOFF_NOW:
                    result = execute_handoff()
                    continue
                if decision.action is HandoffAction.ABORT:
                    status = current_status()
                    result = finalize(
                        reason=TerminationReason.ABORTED,
                        failure_mode=FailureMode.NONE,
                        final_state=GovernorState.ABORT,
                        handoff_occurred=False,
                        labels=_status_labels(status),
                        status=status,
                        message=decision.rationale,
                    )
                    continue
                if decision.action is HandoffAction.FALLBACK:
                    result = failure_or_fallback(
                        reason=TerminationReason.ABORTED,
                        failure_mode=FailureMode.UNKNOWN,
                        state=GovernorState.ABORT,
                        message=decision.rationale,
                    )
                    continue

                selected = next(
                    (
                        item.candidate
                        for item in decision.candidates
                        if item.selected
                    ),
                    None,
                )
                if selected is None or selected.kind == "current":
                    result = failure_or_fallback(
                        reason=TerminationReason.INVALID_MODEL_PREDICTION,
                        failure_mode=FailureMode.INVALID_PREDICTION,
                        state=GovernorState.ABORT,
                        message="CONTINUE decision did not select a future candidate",
                    )
                    continue
                transition(GovernorState.STAGE, "execute one bounded analytic adjustment")
                try:
                    adapter.raise_if_cancelled()
                    stage_result = adapter.stage(
                        selected,
                        min(
                            self.config.max_step_distance_m,
                            max(
                                0.0,
                                self.config.max_staging_distance_m
                                - analytic_distance,
                            ),
                        ),
                    )
                    if not isinstance(stage_result, StageResult):
                        stage_result = StageResult.model_validate(stage_result)
                except Exception as exc:
                    if _is_cancellation(exc):
                        raise
                    result = failure_or_fallback(
                        reason=TerminationReason.STAGING_FAILURE,
                        failure_mode=FailureMode.STAGING,
                        state=GovernorState.STAGING_FAILURE,
                        message=f"analytic stage action raised: {exc}",
                    )
                    continue
                analytic_steps += stage_result.steps
                analytic_distance += stage_result.distance_m
                analytic_time += stage_result.elapsed_s
                if stage_result.distance_m > self.config.max_step_distance_m + 1e-9:
                    result = failure_or_fallback(
                        reason=TerminationReason.STAGING_FAILURE,
                        failure_mode=FailureMode.STAGING,
                        state=GovernorState.STAGING_FAILURE,
                        message="adapter exceeded the configured one-step distance bound",
                    )
                    continue
                if not stage_result.success or stage_result.steps != 1:
                    result = failure_or_fallback(
                        reason=TerminationReason.STAGING_FAILURE,
                        failure_mode=FailureMode.STAGING,
                        state=GovernorState.STAGING_FAILURE,
                        message=stage_result.error or "analytic stage action did not succeed",
                    )
                    continue
                transition(GovernorState.OBSERVE, "stage completed; re-observe real state")
        except BaseException as exc:
            if not _is_cancellation(exc):
                raise
            cancellation_to_raise = exc
            try:
                status = current_status()
            except Exception:
                status = EpisodeStatus()
            if handoff_started and vla_result is None:
                # Control was transferred and the learned primitive was
                # entered, but cancellation prevented its normal return.  An
                # invocation attempt and elapsed time are still causal costs;
                # chunk/action counts remain explicitly unavailable.
                vla_result = VLAExecutionResult(
                    result={},
                    elapsed_s=max(
                        0.0,
                        self._monotonic() - (vla_started_s or self._monotonic()),
                    ),
                    exception="cancelled before VLA execution returned",
                    invocations=1,
                    chunks=None,
                    env_actions=None,
                )
            result = finalize(
                reason=TerminationReason.CANCELLED,
                failure_mode=FailureMode.CANCELLATION,
                final_state=GovernorState.CANCELLED,
                handoff_occurred=handoff_started,
                labels=_status_labels(status),
                status=status,
                message=str(exc) or "governor cancelled",
            )

        assert result is not None
        if cancellation_to_raise is not None and self.config.propagate_cancellation:
            raise cancellation_to_raise
        return result


def build_governor(
    config: Mapping[str, Any],
    *,
    model=None,
    sink: ResearchSink | None = None,
    expected_model_artifact_id: str | None = None,
) -> HandoffGovernor:
    """Build a governor from resolved JSON config.

    Policy/model artifact dispatch is imported lazily so baseline RPent does not
    load research dependencies. ``config`` must contain ``governor``,
    ``candidate_generator``, and ``policy`` mappings. The full dispatch lives in
    :func:`rpent.research.handoff.runtime.build_runtime_policy`.
    """
    from rpent.research.handoff.candidates import CandidateGeneratorConfig
    from rpent.research.handoff.runtime import build_runtime_policy

    governor_config = GovernorConfig.model_validate(config.get("governor", {}))
    generator_config = CandidateGeneratorConfig.model_validate(
        config.get("candidate_generator", {})
    )
    policy, fallback_policy = build_runtime_policy(
        config,
        model=model,
        expected_model_artifact_id=expected_model_artifact_id,
    )
    return HandoffGovernor(
        policy=policy,
        fallback_policy=fallback_policy,
        candidate_generator=ObjectRelativeCandidateGenerator(generator_config),
        config=governor_config,
        sink=sink,
    )
