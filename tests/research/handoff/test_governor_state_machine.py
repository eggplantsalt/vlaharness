from __future__ import annotations

from dataclasses import dataclass

import pytest

from rpent.research.handoff.candidates import (
    CandidateGeneratorConfig,
    ObjectRelativeCandidateGenerator,
)
from rpent.research.handoff.governor import (
    EpisodeStatus,
    FallbackMode,
    GovernorConfig,
    GovernorInvocation,
    HandoffGovernor,
    InMemoryResearchSink,
    StageResult,
    VLAExecutionResult,
)
from rpent.research.handoff.policies import DirectHandoffPolicy, FixedDistancePolicy
from rpent.research.handoff.types import (
    ControllerIdentity,
    FailureMode,
    FeatureAvailability,
    FeatureProvenance,
    GovernorState,
    LabelSource,
    OutcomeLabels,
    OutcomeSignal,
    SkillIdentity,
    TargetContext,
    TargetEstimate,
    TerminationReason,
    TrialIdentity,
    HandoffState,
    outcome_record_id,
    unavailable_signal,
)


def _state(z: float, sequence: int = 0) -> HandoffState:
    return HandoffState(
        state_id=f"state-{sequence}-{z}",
        observation_sequence=sequence,
        observed_elapsed_s=float(sequence),
        eef_position_m=(0.0, 0.0, z),
        eef_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        gripper_opening_m=0.08,
        skill=SkillIdentity(name="pick", semantic_target="mug"),
        target=TargetContext(
            target_id="mug",
            description="mug",
            estimate=TargetEstimate(
                estimate_id=f"target-{sequence}",
                position_m=(0.0, 0.0, 0.0),
                frame="world",
                provider="fake",
                availability=FeatureAvailability.DEPLOYMENT_PERCEPTION,
                observation_sequence=sequence,
            ),
        ),
        provenance=(
            FeatureProvenance(feature_name="eef_position_m", availability=FeatureAvailability.DEPLOYMENT_SENSOR, source="fake", unit="m", frame="world"),
            FeatureProvenance(feature_name="eef_quaternion_xyzw", availability=FeatureAvailability.DEPLOYMENT_SENSOR, source="fake", unit="unit", frame="world"),
            FeatureProvenance(feature_name="gripper_opening_m", availability=FeatureAvailability.DEPLOYMENT_SENSOR, source="fake", unit="m", frame="gripper"),
            FeatureProvenance(feature_name="skill", availability=FeatureAvailability.DERIVED_DEPLOYMENT, source="fake", unit="categorical", frame="semantic"),
            FeatureProvenance(feature_name="target_position_m", availability=FeatureAvailability.DEPLOYMENT_PERCEPTION, source="fake", unit="m", frame="world"),
        ),
    )


def _invocation() -> GovernorInvocation:
    return GovernorInvocation(
        identity=TrialIdentity(run_id="run", episode_id="ep", trial_id="trial", invocation_id="inv", candidate_id="candidate", suite="suite", task_id=0, seed=0, reset_id="reset"),
        skill=SkillIdentity(name="pick", semantic_target="mug"),
        controller=ControllerIdentity(method="test", implementation_version="v1", configuration_id="cfg"),
        vla_kwargs={"prompt": "pick mug"},
    )


@dataclass
class _FakeAdapter:
    z: float = 0.2
    stage_success: bool = True
    cancel_in_vla: bool = False
    vla_calls: int = 0
    stage_calls: int = 0

    def observe(self, skill):
        del skill
        return _state(self.z, self.stage_calls)

    def episode_status(self):
        return EpisodeStatus()

    def raise_if_cancelled(self):
        return None

    def stage(self, candidate, max_step_m):
        del max_step_m
        self.stage_calls += 1
        if not self.stage_success:
            return StageResult(success=False, steps=0, distance_m=0.0, elapsed_s=0.01, error="controller rejected action")
        before = self.z
        self.z = candidate.eef_position_m[2]
        return StageResult(success=True, steps=1, distance_m=abs(before - self.z), elapsed_s=0.01)

    def execute_vla(self, invocation):
        del invocation
        self.vla_calls += 1
        if self.cancel_in_vla:
            raise RuntimeErrorWithCancellationName("cancel after VLA entry")
        return VLAExecutionResult(result={"success": False}, elapsed_s=0.2, invocations=1, chunks=2, env_actions=10)

    def label_outcome(self, result):
        del result
        return OutcomeLabels(
            primitive_success=OutcomeSignal(value=False, source=LabelSource.PRIMITIVE_HEURISTIC, definition="fake primitive"),
            skill_success=unavailable_signal("independent skill evaluator unavailable"),
            task_success=OutcomeSignal(value=True, source=LabelSource.OFFICIAL_TERMINATION, definition="independent fake task signal"),
        )


class RuntimeErrorWithCancellationName(Exception):
    pass


RuntimeErrorWithCancellationName.__name__ = "CancelledError"


class _LabelFailureAdapter(_FakeAdapter):
    def label_outcome(self, result):
        del result
        raise ValueError("fixture labeler failure")


class _VlaExceptionAdapter(_FakeAdapter):
    def __init__(self, exception: BaseException) -> None:
        super().__init__()
        self.exception = exception

    def execute_vla(self, invocation):
        del invocation
        self.vla_calls += 1
        raise self.exception


def _governor(policy, sink, *, fallback=FallbackMode.ABORT):
    return HandoffGovernor(
        policy=policy,
        candidate_generator=ObjectRelativeCandidateGenerator(
            CandidateGeneratorConfig(standoff_distances_m=(0.08,), max_candidates=1)
        ),
        config=GovernorConfig(
            max_analytic_steps=3,
            max_staging_distance_m=0.5,
            max_step_distance_m=0.5,
            fallback_mode=fallback,
            budget_fallback_mode=fallback,
            propagate_cancellation=False,
        ),
        sink=sink,
    )


def test_fake_env_vla_end_to_end_stages_reobserves_and_hands_off() -> None:
    sink = InMemoryResearchSink()
    adapter = _FakeAdapter()
    result = _governor(FixedDistancePolicy(0.08), sink).run(adapter, _invocation())
    assert adapter.stage_calls == 1
    assert adapter.vla_calls == 1
    assert result.outcome.handoff_occurred
    assert result.outcome.labels.primitive_success.value is False
    assert result.outcome.labels.task_success.value is True
    states = [transition.to_state for transition in result.transitions]
    assert GovernorState.STAGE in states
    assert GovernorState.EXECUTE_VLA in states
    assert states[-1] is GovernorState.DONE


def test_staging_failure_is_not_vla_failure_or_training_negative() -> None:
    sink = InMemoryResearchSink()
    adapter = _FakeAdapter(stage_success=False)
    result = _governor(FixedDistancePolicy(0.08), sink).run(adapter, _invocation())
    assert adapter.vla_calls == 0
    assert not result.outcome.handoff_occurred
    assert result.outcome.costs.vla_invocations == 0
    assert result.outcome.termination.reason is TerminationReason.STAGING_FAILURE
    assert result.outcome.termination.failure_mode is FailureMode.STAGING
    assert result.outcome.labels.primitive_success.value is None


def test_cancellation_after_handoff_preserves_attempt_and_pre_handoff_state() -> None:
    sink = InMemoryResearchSink()
    adapter = _FakeAdapter(cancel_in_vla=True)
    result = _governor(DirectHandoffPolicy(), sink).run(adapter, _invocation())
    assert result.outcome.handoff_occurred
    assert result.outcome.pre_handoff_state is not None
    assert result.outcome.costs.vla_invocations == 1
    assert result.outcome.termination.reason is TerminationReason.CANCELLED
    assert result.outcome.termination.failure_mode is FailureMode.CANCELLATION


def test_label_failure_after_vla_is_an_explicit_nontraining_outcome() -> None:
    sink = InMemoryResearchSink()
    result = _governor(DirectHandoffPolicy(), sink).run(
        _LabelFailureAdapter(),
        _invocation(),
    )

    assert result.outcome.handoff_occurred
    assert result.outcome.costs.vla_invocations == 1
    assert result.outcome.labels.primitive_success.value is None
    assert result.outcome.termination.reason is TerminationReason.OUTCOME_LABEL_FAILURE
    assert result.outcome.termination.failure_mode is FailureMode.OUTCOME_LABEL
    assert result.outcome.termination.final_governor_state is GovernorState.OUTCOME_LABEL_FAILURE


@pytest.mark.parametrize(
    ("exception", "expected_reason", "expected_mode"),
    (
        (TimeoutError("VLA timed out"), TerminationReason.TIMEOUT, FailureMode.TIMEOUT),
        (
            ConnectionError("RPC transport disconnected"),
            TerminationReason.RPC_FAILURE,
            FailureMode.RPC,
        ),
    ),
)
def test_vla_transport_exceptions_are_classified_and_counted(
    exception,
    expected_reason,
    expected_mode,
) -> None:
    sink = InMemoryResearchSink()
    result = _governor(DirectHandoffPolicy(), sink).run(
        _VlaExceptionAdapter(exception),
        _invocation(),
    )

    assert result.outcome.handoff_occurred
    assert result.outcome.costs.vla_invocations == 1
    assert result.outcome.termination.reason is expected_reason
    assert result.outcome.termination.failure_mode is expected_mode
    assert result.outcome.labels.primitive_success.value is None


def test_outcome_record_identity_is_result_independent_and_retry_stable() -> None:
    identity = _invocation().identity
    retry_context = identity.model_copy(
        update={
            "candidate_id": "different-candidate-detail",
            "reset_id": "same-invocation-observed-differently",
            "repeat_index": 9,
        }
    )

    assert outcome_record_id(identity) == outcome_record_id(retry_context)
    assert outcome_record_id(identity) != outcome_record_id(
        identity.model_copy(update={"invocation_id": "inv-2"})
    )
