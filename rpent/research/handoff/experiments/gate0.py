"""Gate-0 kill-test collector over a real or fake controller adapter."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rpent.research.handoff.candidates import (
    CandidateGeneratorConfig,
    ObjectRelativeCandidateGenerator,
    wrist_yaw_pitch,
)
from rpent.research.handoff.experiments.sampling import (
    Gate0Sample,
    Gate0SamplerConfig,
    generate_gate0_samples,
    sample_world_position,
)
from rpent.research.handoff.governor import (
    EpisodeStatus,
    GovernorAdapter,
    GovernorConfig,
    GovernorInvocation,
    GovernorRunResult,
    HandoffGovernor,
    ResearchSink,
)
from rpent.research.handoff.policies import DirectHandoffPolicy
from rpent.research.handoff.privileged import ExperimentSetupRecord
from rpent.research.handoff.types import (
    CandidateGeometry,
    ControllerIdentity,
    CostRecord,
    FailureMode,
    GovernorState,
    LabelSource,
    OutcomeLabels,
    OutcomeRecord,
    OutcomeSignal,
    SkillIdentity,
    TerminationReason,
    TerminationRecord,
    TimingRecord,
    TrialIdentity,
    outcome_record_id,
    unavailable_signal,
)


class Gate0Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    sampler: Gate0SamplerConfig
    staging_tolerance_m: float = Field(default=0.012, gt=0.0)
    staging_orientation_tolerance_rad: float = Field(default=0.05, gt=0.0)
    staging_step_m: float = Field(default=0.025, gt=0.0)
    max_staging_steps: int = Field(default=80, ge=1)
    max_staging_distance_m: float = Field(default=0.5, gt=0.0)
    max_staging_time_s: float = Field(default=30.0, gt=0.0)


class Gate0Setup(BaseModel):
    """Privileged controlled setup, physically separate from online state."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    record: ExperimentSetupRecord
    target_position_m: tuple[float, float, float]
    reset_id: str | None = None


@runtime_checkable
class Gate0Adapter(GovernorAdapter, Protocol):
    def reset_for_trial(
        self,
        identity: TrialIdentity,
        skill: SkillIdentity,
        sample: Gate0Sample,
    ) -> Gate0Setup:
        """Perform a fresh reset and return label/setup-only target context."""

    def current_eef_position_m(self) -> tuple[float, float, float]:
        """Read whitelisted proprioception without object ground truth."""

    def current_eef_quaternion_xyzw(self) -> tuple[float, float, float, float]:
        """Read measured EEF orientation without object ground truth."""


@runtime_checkable
class SetupSink(Protocol):
    def append_setup(self, setup: ExperimentSetupRecord) -> None:
        """Persist setup records in a separate append-only namespace."""


class NullSetupSink:
    def append_setup(self, setup: ExperimentSetupRecord) -> None:
        del setup


@dataclass(frozen=True, slots=True)
class Gate0RunIdentity:
    run_id: str
    suite: str
    task_id: int | str
    seed: int
    episode_prefix: str = "gate0"
    source_revision: str | None = None
    configuration_id: str | None = None
    execution_plan_id: str | None = None
    runtime_attestation_id: str | None = None
    runtime_attestation_sha256: str | None = None


def _matched_cohort_id(
    identity: TrialIdentity,
    *,
    skill: SkillIdentity,
    source_revision: str | None,
) -> str | None:
    """Identify candidates executed from the same pinned reset/repeat cell."""
    if identity.reset_id is None:
        return None
    payload = {
        "schema_version": "rpent.handoff-gate0-matched-cohort/v1",
        "run_id": identity.run_id,
        "suite": identity.suite,
        "task_id": identity.task_id,
        "seed": identity.seed,
        "reset_id": identity.reset_id,
        "repeat_index": identity.repeat_index,
        "skill": skill.model_dump(mode="json", exclude_none=False),
        "source_revision": source_revision,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return "gate0-cohort-" + hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()[:20]


def _failure_labels(status: EpisodeStatus) -> OutcomeLabels:
    return OutcomeLabels(
        primitive_success=unavailable_signal("VLA primitive was not invoked"),
        skill_success=unavailable_signal("VLA skill was not invoked"),
        task_success=OutcomeSignal(
            value=status.terminated,
            source=LabelSource.OFFICIAL_TERMINATION,
            definition="official task termination after setup/staging failure",
        ),
        episode_truncated=OutcomeSignal(
            value=status.truncated,
            source=LabelSource.RUNTIME,
            definition="episode truncation after setup/staging failure",
        ),
        llm_finish=unavailable_signal("Gate-0 does not invoke an LLM planner"),
    )


def _failure_outcome(
    *,
    identity: TrialIdentity,
    skill: SkillIdentity,
    controller: ControllerIdentity,
    setup_record_id: str | None,
    reason: TerminationReason,
    failure_mode: FailureMode,
    final_state: GovernorState,
    message: str,
    status: EpisodeStatus,
    started_s: float,
    staging_steps: int,
    staging_distance_m: float,
    staging_time_s: float,
    source_revision: str | None,
    metadata: Mapping[str, Any],
) -> OutcomeRecord:
    ended = time.monotonic()
    return OutcomeRecord(
        record_id=outcome_record_id(identity),
        identity=identity,
        skill=skill,
        controller=controller,
        pre_handoff_state=None,
        handoff_occurred=False,
        labels=_failure_labels(status),
        costs=CostRecord(
            analytic_steps=staging_steps,
            analytic_distance_m=staging_distance_m,
            analytic_time_s=staging_time_s,
            vla_invocations=0,
            vla_time_s=0.0,
            total_elapsed_s=max(0.0, ended - started_s),
        ),
        timing=TimingRecord(
            started_monotonic_s=started_s,
            ended_monotonic_s=ended,
        ),
        termination=TerminationRecord(
            reason=reason,
            failure_mode=failure_mode,
            final_governor_state=final_state,
            episode_terminated=status.terminated,
            episode_truncated=status.truncated,
            message=message,
        ),
        setup_record_id=setup_record_id,
        source_revision=source_revision,
        metadata={"execution_layer": "gate0", **dict(metadata)},
    )


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _is_timeout_failure(exc: BaseException) -> bool:
    return any(
        isinstance(item, TimeoutError)
        or "timeout" in item.__class__.__name__.lower()
        or "timed out" in str(item).lower()
        for item in _exception_chain(exc)
    )


def _is_rpc_failure(exc: BaseException) -> bool:
    tokens = ("rpc", "http", "socket", "transport", "connection")
    return any(
        isinstance(item, (ConnectionError, BrokenPipeError))
        or any(token in item.__class__.__name__.lower() for token in tokens)
        for item in _exception_chain(exc)
    )


class Gate0Collector:
    """Fresh-reset candidate staging followed by unchanged frozen-VLA execution."""

    def __init__(
        self,
        *,
        adapter: Gate0Adapter,
        config: Gate0Config,
        skill: SkillIdentity,
        controller: ControllerIdentity,
        run_identity: Gate0RunIdentity,
        outcome_sink: ResearchSink,
        setup_sink: SetupSink | None = None,
        completed_trial_ids: Sequence[str] = (),
        vla_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.skill = skill
        self.controller = controller
        self.run_identity = run_identity
        self.outcome_sink = outcome_sink
        self.setup_sink = setup_sink or NullSetupSink()
        self.completed_trial_ids = set(completed_trial_ids)
        self.vla_kwargs = dict(vla_kwargs or {})
        self.vla_kwargs.setdefault(
            "prompt", self.skill.semantic_target or self.skill.name
        )

    def samples(self) -> tuple[Gate0Sample, ...]:
        return generate_gate0_samples(self.config.sampler)

    def _identity(self, sample: Gate0Sample) -> TrialIdentity:
        return TrialIdentity(
            run_id=self.run_identity.run_id,
            episode_id=(
                f"{self.run_identity.episode_prefix}-episode-{sample.sample_id}"
            ),
            trial_id=f"{self.run_identity.episode_prefix}-trial-{sample.sample_id}",
            invocation_id=f"{self.run_identity.episode_prefix}-vla-{sample.sample_id}",
            candidate_id=sample.candidate_id,
            suite=self.run_identity.suite,
            task_id=self.run_identity.task_id,
            seed=self.run_identity.seed,
            repeat_index=sample.repeat_index,
        )

    def collect(self, *, limit: int | None = None) -> tuple[OutcomeRecord, ...]:
        outcomes: list[OutcomeRecord] = []
        pending = [
            sample
            for sample in self.samples()
            if self._identity(sample).trial_id not in self.completed_trial_ids
        ]
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            pending = pending[:limit]
        for sample in pending:
            result = self._collect_one(sample)
            outcomes.append(result)
            self.completed_trial_ids.add(result.identity.trial_id)
        return tuple(outcomes)

    def _collect_one(self, sample: Gate0Sample) -> OutcomeRecord:
        identity = self._identity(sample)
        started = time.monotonic()
        staging_steps = 0
        staging_distance = 0.0
        staging_time = 0.0
        setup: Gate0Setup | None = None
        resolved_identity = identity
        cohort_id: str | None = None
        phase = "reset_or_setup"
        try:
            setup = self.adapter.reset_for_trial(identity, self.skill, sample)
            if not isinstance(setup, Gate0Setup):
                setup = Gate0Setup.model_validate(setup)
            resolved_identity = identity.model_copy(update={"reset_id": setup.reset_id})
            cohort_id = _matched_cohort_id(
                resolved_identity,
                skill=self.skill,
                source_revision=self.run_identity.source_revision,
            )
            phase = "setup_persistence"
            self.setup_sink.append_setup(setup.record)
            phase = "staging"
            desired_position = sample_world_position(
                sample,
                target_position_m=setup.target_position_m,
                approach_axis_world=self.config.sampler.approach_axis_world,
            )
            desired = CandidateGeometry(
                candidate_id=sample.candidate_id,
                kind="perturbation",
                eef_position_m=desired_position,
                target_relative_position_m=tuple(
                    float(desired_value - target_value)
                    for desired_value, target_value in zip(
                        desired_position, setup.target_position_m
                    )
                ),
                wrist_yaw_rad=sample.wrist_yaw_rad,
                wrist_pitch_rad=sample.wrist_pitch_rad,
                requested_standoff_m=sample.standoff_m,
            )
            while True:
                current = self.adapter.current_eef_position_m()
                quaternion = self.adapter.current_eef_quaternion_xyzw()
                current_yaw, current_pitch = wrist_yaw_pitch(quaternion)
                position_error = float(
                    np.linalg.norm(
                        np.asarray(desired_position) - np.asarray(current)
                    )
                )
                yaw_error = abs(
                    math.atan2(
                        math.sin(sample.wrist_yaw_rad - current_yaw),
                        math.cos(sample.wrist_yaw_rad - current_yaw),
                    )
                )
                pitch_error = abs(
                    math.atan2(
                        math.sin(sample.wrist_pitch_rad - current_pitch),
                        math.cos(sample.wrist_pitch_rad - current_pitch),
                    )
                )
                if (
                    position_error <= self.config.staging_tolerance_m
                    and yaw_error <= self.config.staging_orientation_tolerance_rad
                    and pitch_error
                    <= self.config.staging_orientation_tolerance_rad
                ):
                    break
                if (
                    staging_steps >= self.config.max_staging_steps
                    or staging_distance >= self.config.max_staging_distance_m
                    or time.monotonic() - started >= self.config.max_staging_time_s
                ):
                    raise RuntimeError(
                        "requested Gate-0 candidate was not reached within staging "
                        "budget: "
                        f"position_error_m={position_error:.6g}, "
                        f"yaw_error_rad={yaw_error:.6g}, "
                        f"pitch_error_rad={pitch_error:.6g}"
                    )
                stage = self.adapter.stage(desired, self.config.staging_step_m)
                staging_steps += stage.steps
                staging_distance += stage.distance_m
                staging_time += stage.elapsed_s
                if not stage.success or stage.steps != 1:
                    raise RuntimeError(stage.error or "Gate-0 analytic staging failed")
                status = self.adapter.episode_status()
                if status.terminated or status.truncated:
                    raise RuntimeError("episode ended before Gate-0 VLA invocation")
        except Exception as exc:
            try:
                status = self.adapter.episode_status()
            except Exception:
                status = EpisodeStatus()
            if exc.__class__.__name__ in {"ToolCancelled", "CancelledError", "GovernorCancelled"}:
                reason = TerminationReason.CANCELLED
                mode = FailureMode.CANCELLATION
                state = GovernorState.CANCELLED
            elif _is_timeout_failure(exc):
                reason = TerminationReason.TIMEOUT
                mode = FailureMode.TIMEOUT
                state = GovernorState.ABORT
            elif _is_rpc_failure(exc):
                reason = TerminationReason.RPC_FAILURE
                mode = FailureMode.RPC
                state = GovernorState.ABORT
            elif status.terminated:
                reason = TerminationReason.EPISODE_TERMINATED_BEFORE_VLA
                mode = FailureMode.TERMINATION_BEFORE_VLA
                state = GovernorState.TERMINATED
            elif status.truncated:
                reason = TerminationReason.EPISODE_TRUNCATED_BEFORE_VLA
                mode = FailureMode.TRUNCATION
                state = GovernorState.TRUNCATED
            elif phase == "staging":
                reason = TerminationReason.STAGING_FAILURE
                mode = FailureMode.STAGING
                state = GovernorState.STAGING_FAILURE
            else:
                # Reset/setup extraction and durable setup persistence occur
                # before deployment perception. Collapsing those failures into
                # "perception" would corrupt the failure analysis.
                reason = TerminationReason.INVALID_STATE
                mode = (
                    FailureMode.INVALID_INPUT
                    if phase == "reset_or_setup"
                    else FailureMode.UNKNOWN
                )
                state = GovernorState.ABORT
            outcome = _failure_outcome(
                identity=resolved_identity,
                skill=self.skill,
                controller=self.controller,
                setup_record_id=setup.record.record_id if setup is not None else None,
                reason=reason,
                failure_mode=mode,
                final_state=state,
                message=str(exc),
                status=status,
                started_s=started,
                staging_steps=staging_steps,
                staging_distance_m=staging_distance,
                staging_time_s=staging_time,
                source_revision=self.run_identity.source_revision,
                metadata=self._execution_metadata(sample, cohort_id=cohort_id),
            )
            self.outcome_sink.append_outcome(outcome)
            return outcome

        direct_governor = HandoffGovernor(
            policy=DirectHandoffPolicy(),
            candidate_generator=ObjectRelativeCandidateGenerator(
                CandidateGeneratorConfig(
                    standoff_distances_m=(0.01,),
                    max_candidates=1,
                )
            ),
            config=GovernorConfig(
                max_analytic_steps=staging_steps,
                max_staging_distance_m=max(
                    staging_distance, self.config.staging_step_m
                ),
                max_local_time_s=max(
                    self.config.max_staging_time_s,
                    time.monotonic() - started + 1.0,
                ),
            ),
            sink=self.outcome_sink,
        )
        invocation = GovernorInvocation(
            identity=resolved_identity,
            skill=self.skill,
            controller=self.controller,
            vla_kwargs=self.vla_kwargs,
            metadata={
                "setup_record_id": setup.record.record_id,
                "source_revision": self.run_identity.source_revision,
                "execution_layer": "gate0",
                "requested_sample_id": sample.sample_id,
                "candidate_id": sample.candidate_id,
                **self._execution_metadata(sample, cohort_id=cohort_id),
            },
            initial_costs=CostRecord(
                analytic_steps=staging_steps,
                analytic_distance_m=staging_distance,
                analytic_time_s=staging_time,
            ),
        )
        result: GovernorRunResult = direct_governor.run(self.adapter, invocation)
        return result.outcome

    def _execution_metadata(
        self,
        sample: Gate0Sample,
        *,
        cohort_id: str | None,
    ) -> dict[str, Any]:
        return {
            "gate0_configuration_id": self.run_identity.configuration_id,
            "execution_plan_id": self.run_identity.execution_plan_id,
            "runtime_attestation_id": self.run_identity.runtime_attestation_id,
            "runtime_attestation_sha256": (
                self.run_identity.runtime_attestation_sha256
            ),
            "gate0_matched_cohort_id": cohort_id,
            "gate0_candidate_id": sample.candidate_id,
            "gate0_repeat_index": sample.repeat_index,
            "requested_sample_id": sample.sample_id,
        }
