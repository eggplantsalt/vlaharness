"""Strict versioned records for controller-handoff research.

Authoritative experiment records are Pydantic models with forbidden extra
fields, finite-number validation, explicit units/frames/provenance, and stable
canonical JSON. Free-form dictionaries are allowed only in the explicitly
non-authoritative ``metadata`` fields and must themselves be JSON serializable.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

STATE_SCHEMA_VERSION = "rpent.handoff-state/v1"
OUTCOME_SCHEMA_VERSION = "rpent.handoff-outcome/v1"
DECISION_SCHEMA_VERSION = "rpent.handoff-decision/v1"
PRIVILEGED_SCHEMA_VERSION = "rpent.handoff-privileged/v1"


class HandoffRecord(BaseModel):
    """Base for deterministic, immutable, strict research records."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    def canonical_json(self) -> str:
        """Return deterministic UTF-8-safe JSON suitable for hashing/JSONL."""
        return json.dumps(
            self.model_dump(mode="json", exclude_none=False),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> Self:
        """Parse a strict record from JSON."""
        return cls.model_validate_json(value)


def _non_empty(value: str, name: str) -> str:
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _json_metadata(value: dict[str, Any]) -> dict[str, Any]:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain finite JSON values only") from exc
    return value


class FeatureAvailability(str, Enum):
    """Where a feature comes from and whether online policy may consume it."""

    DEPLOYMENT_SENSOR = "deployment_sensor"
    DEPLOYMENT_PERCEPTION = "deployment_perception"
    DERIVED_DEPLOYMENT = "derived_deployment"
    SIMULATOR_PRIVILEGED = "simulator_privileged"
    EXPERIMENT_SETUP = "experiment_setup"

    @property
    def online_allowed(self) -> bool:
        return self in {
            self.DEPLOYMENT_SENSOR,
            self.DEPLOYMENT_PERCEPTION,
            self.DERIVED_DEPLOYMENT,
        }


class LabelSource(str, Enum):
    PRIMITIVE_HEURISTIC = "primitive_heuristic"
    SKILL_EVALUATOR = "skill_evaluator"
    OFFICIAL_TERMINATION = "official_termination"
    PRIVILEGED_EVALUATOR = "privileged_evaluator"
    PLANNER_DECLARATION = "planner_declaration"
    RUNTIME = "runtime"
    UNAVAILABLE = "unavailable"


class HandoffAction(str, Enum):
    CONTINUE = "continue"
    HANDOFF_NOW = "handoff_now"
    ABORT = "abort"
    FALLBACK = "fallback"


class GovernorState(str, Enum):
    INIT = "init"
    OBSERVE = "observe"
    DECIDE = "decide"
    STAGE = "stage"
    HANDOFF = "handoff"
    EXECUTE_VLA = "execute_vla"
    RECORD_OUTCOME = "record_outcome"
    FALLBACK = "fallback"
    ABORT = "abort"
    CANCELLED = "cancelled"
    TERMINATED = "terminated"
    TRUNCATED = "truncated"
    PERCEPTION_FAILURE = "perception_failure"
    STAGING_FAILURE = "staging_failure"
    VLA_FAILURE = "vla_failure"
    DONE = "done"


class TerminationReason(str, Enum):
    COMPLETED = "completed"
    HANDOFF_COMPLETED = "handoff_completed"
    ABORTED = "aborted"
    FALLBACK_FORCED_HANDOFF = "fallback_forced_handoff"
    FALLBACK_BASELINE = "fallback_baseline"
    CANCELLED = "cancelled"
    EPISODE_TERMINATED_BEFORE_VLA = "episode_terminated_before_vla"
    EPISODE_TRUNCATED_BEFORE_VLA = "episode_truncated_before_vla"
    PERCEPTION_FAILURE = "perception_failure"
    STAGING_FAILURE = "staging_failure"
    VLA_FAILURE = "vla_failure"
    RPC_FAILURE = "rpc_failure"
    TIMEOUT = "timeout"
    INVALID_STATE = "invalid_state"
    INVALID_MODEL_PREDICTION = "invalid_model_prediction"
    BUDGET_EXHAUSTED = "budget_exhausted"
    UNKNOWN = "unknown"


class FailureMode(str, Enum):
    NONE = "none"
    PERCEPTION = "perception"
    STAGING = "staging"
    VLA = "vla"
    RPC = "rpc"
    TIMEOUT = "timeout"
    CANCELLATION = "cancellation"
    TERMINATION_BEFORE_VLA = "termination_before_vla"
    TRUNCATION = "truncation"
    INVALID_INPUT = "invalid_input"
    INVALID_PREDICTION = "invalid_prediction"
    UNKNOWN = "unknown"


class SkillIdentity(HandoffRecord):
    name: str
    semantic_target: str | None = None
    learned_controller: str = "pi0.5"

    @field_validator("name", "learned_controller")
    @classmethod
    def validate_names(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)


class ControllerIdentity(HandoffRecord):
    method: str
    implementation_version: str
    checkpoint_id: str | None = None
    configuration_id: str

    @field_validator("method", "implementation_version", "configuration_id")
    @classmethod
    def validate_names(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)


class TrialIdentity(HandoffRecord):
    run_id: str
    episode_id: str
    trial_id: str
    invocation_id: str
    candidate_id: str | None = None
    suite: str
    task_id: int | str
    seed: int
    reset_id: str | None = None
    repeat_index: int = Field(default=0, ge=0)

    @field_validator("run_id", "episode_id", "trial_id", "invocation_id", "suite")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)


class FeatureProvenance(HandoffRecord):
    """Provenance for one named field in a policy state or feature vector."""

    feature_name: str
    availability: FeatureAvailability
    source: str
    unit: str
    frame: str
    derivation: str | None = None
    provider_version: str | None = None

    @field_validator("feature_name", "source", "unit", "frame")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)


class VisualGeometry(HandoffRecord):
    mask_area_fraction: float = Field(ge=0.0, le=1.0)
    valid_depth_fraction: float = Field(ge=0.0, le=1.0)
    image_centroid_rc_normalized: tuple[float, float]
    camera_name: str

    @field_validator("image_centroid_rc_normalized")
    @classmethod
    def validate_centroid(cls, value: tuple[float, float]) -> tuple[float, float]:
        if any(component < 0.0 or component > 1.0 for component in value):
            raise ValueError("normalized image centroid components must be in [0, 1]")
        return value

    @field_validator("camera_name")
    @classmethod
    def validate_camera(cls, value: str) -> str:
        return _non_empty(value, "camera_name")


class TargetEstimate(HandoffRecord):
    estimate_id: str
    position_m: tuple[float, float, float] | None
    frame: str
    provider: str
    availability: FeatureAvailability
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    observation_sequence: int = Field(ge=0)
    age_s: float = Field(default=0.0, ge=0.0)
    visual_geometry: VisualGeometry | None = None
    unavailable_reason: str | None = None

    @field_validator("estimate_id", "frame", "provider")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.position_m is None and not self.unavailable_reason:
            raise ValueError("an unavailable target estimate needs unavailable_reason")
        if self.position_m is not None and self.unavailable_reason:
            raise ValueError("available target estimate cannot have unavailable_reason")
        return self


class TargetContext(HandoffRecord):
    target_id: str
    description: str
    estimate: TargetEstimate
    approach_axis_target_frame: tuple[float, float, float] | None = None

    @field_validator("target_id", "description")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)


class HandoffState(HandoffRecord):
    schema_version: Literal[STATE_SCHEMA_VERSION] = STATE_SCHEMA_VERSION
    state_id: str
    observation_sequence: int = Field(ge=0)
    observed_elapsed_s: float = Field(ge=0.0)
    eef_position_m: tuple[float, float, float]
    eef_quaternion_xyzw: tuple[float, float, float, float]
    gripper_opening_m: float = Field(ge=0.0)
    skill: SkillIdentity
    target: TargetContext | None = None
    provenance: tuple[FeatureProvenance, ...]

    @field_validator("state_id")
    @classmethod
    def validate_state_id(cls, value: str) -> str:
        return _non_empty(value, "state_id")

    @field_validator("eef_quaternion_xyzw")
    @classmethod
    def validate_quaternion(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        norm_sq = sum(component * component for component in value)
        if norm_sq < 1e-12:
            raise ValueError("EEF quaternion must have non-zero norm")
        return value

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        names = [item.feature_name for item in self.provenance]
        if len(names) != len(set(names)):
            raise ValueError("handoff state provenance feature names must be unique")
        required = {
            "eef_position_m",
            "eef_quaternion_xyzw",
            "gripper_opening_m",
            "skill",
        }
        if self.target is not None:
            required.add("target_position_m")
        missing = required.difference(names)
        if missing:
            raise ValueError(f"handoff state missing provenance for {sorted(missing)}")
        return self


class OutcomeEstimate(HandoffRecord):
    mean_success_probability: float = Field(ge=0.0, le=1.0)
    epistemic_std: float = Field(ge=0.0)
    conservative_success_probability: float = Field(ge=0.0, le=1.0)
    lower_quantile_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    upper_quantile_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    ensemble_size: int = Field(default=1, ge=1)
    calibrated: bool = False

    @model_validator(mode="after")
    def validate_quantiles(self) -> Self:
        if (
            self.lower_quantile_probability is not None
            and self.upper_quantile_probability is not None
            and self.lower_quantile_probability > self.upper_quantile_probability
        ):
            raise ValueError("lower quantile exceeds upper quantile")
        return self


class CandidateGeometry(HandoffRecord):
    candidate_id: str
    kind: Literal["current", "standoff", "perturbation", "canonical", "retrieved"]
    eef_position_m: tuple[float, float, float]
    target_relative_position_m: tuple[float, float, float] | None = None
    wrist_yaw_rad: float | None = None
    wrist_pitch_rad: float | None = None
    wrist_yaw_delta_rad: float = 0.0
    wrist_pitch_delta_rad: float = 0.0
    requested_standoff_m: float | None = Field(default=None, ge=0.0)
    approximated_features: tuple[str, ...] = ()

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, value: str) -> str:
        return _non_empty(value, "candidate_id")


class CandidateDecisionRecord(HandoffRecord):
    schema_version: Literal[DECISION_SCHEMA_VERSION] = DECISION_SCHEMA_VERSION
    decision_sequence: int = Field(ge=0)
    candidate: CandidateGeometry
    estimate: OutcomeEstimate | None = None
    handoff_cost: float = Field(ge=0.0)
    staging_cost: float = Field(ge=0.0)
    total_cost: float = Field(ge=0.0)
    selected: bool
    feasible: bool = True
    infeasible_reason: str | None = None

    @model_validator(mode="after")
    def validate_feasibility(self) -> Self:
        if self.feasible and self.infeasible_reason:
            raise ValueError("feasible candidate cannot have infeasible_reason")
        if not self.feasible and not self.infeasible_reason:
            raise ValueError("infeasible candidate needs infeasible_reason")
        return self


class HandoffDecision(HandoffRecord):
    schema_version: Literal[DECISION_SCHEMA_VERSION] = DECISION_SCHEMA_VERSION
    decision_id: str
    state_id: str
    decision_sequence: int = Field(ge=0)
    action: HandoffAction
    selected_candidate_id: str | None = None
    candidates: tuple[CandidateDecisionRecord, ...]
    rationale: str
    policy_name: str

    @field_validator("decision_id", "state_id", "rationale", "policy_name")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)

    @model_validator(mode="after")
    def validate_selected_candidate(self) -> Self:
        selected = [item.candidate.candidate_id for item in self.candidates if item.selected]
        if len(selected) > 1:
            raise ValueError("a handoff decision can select at most one candidate")
        if selected:
            if self.selected_candidate_id != selected[0]:
                raise ValueError("selected_candidate_id disagrees with candidate records")
        elif self.selected_candidate_id is not None:
            raise ValueError("selected_candidate_id has no selected candidate record")
        return self


class OutcomeSignal(HandoffRecord):
    value: bool | None
    source: LabelSource
    definition: str
    evaluator_id: str | None = None

    @field_validator("definition")
    @classmethod
    def validate_definition(cls, value: str) -> str:
        return _non_empty(value, "definition")

    @model_validator(mode="after")
    def validate_unknown(self) -> Self:
        if self.value is None and self.source is not LabelSource.UNAVAILABLE:
            raise ValueError("unknown outcome signal must use source=unavailable")
        if self.value is not None and self.source is LabelSource.UNAVAILABLE:
            raise ValueError("available outcome signal cannot use source=unavailable")
        return self


def unavailable_signal(definition: str) -> OutcomeSignal:
    return OutcomeSignal(value=None, source=LabelSource.UNAVAILABLE, definition=definition)


class OutcomeLabels(HandoffRecord):
    primitive_success: OutcomeSignal = Field(
        default_factory=lambda: unavailable_signal("primitive-specific success")
    )
    skill_success: OutcomeSignal = Field(
        default_factory=lambda: unavailable_signal("skill-specific success")
    )
    task_success: OutcomeSignal = Field(
        default_factory=lambda: unavailable_signal("official task success")
    )
    episode_truncated: OutcomeSignal = Field(
        default_factory=lambda: unavailable_signal("episode truncation")
    )
    llm_finish: OutcomeSignal = Field(
        default_factory=lambda: unavailable_signal("LLM finish declaration")
    )

    def target_value(self, target: str) -> bool | None:
        """Return exactly the requested label; never substitute another signal."""
        if target not in type(self).model_fields:
            raise ValueError(f"unknown training target label: {target!r}")
        signal = getattr(self, target)
        if not isinstance(signal, OutcomeSignal):
            raise ValueError(f"field is not an outcome signal: {target!r}")
        return signal.value


class CostRecord(HandoffRecord):
    analytic_steps: int = Field(default=0, ge=0)
    analytic_distance_m: float = Field(default=0.0, ge=0.0)
    analytic_time_s: float = Field(default=0.0, ge=0.0)
    # Null is reserved for artifact-only/post-run summaries where execution
    # began but the underlying runtime did not expose a trustworthy inference
    # count (for example, cancellation inside an Original Harness Pi0 call).
    vla_invocations: int | None = Field(default=0, ge=0)
    vla_chunks: int | None = Field(default=None, ge=0)
    vla_env_actions: int | None = Field(default=None, ge=0)
    vla_time_s: float = Field(default=0.0, ge=0.0)
    total_env_actions: int | None = Field(default=None, ge=0)
    total_elapsed_s: float = Field(default=0.0, ge=0.0)
    planner_time_s: float | None = Field(default=None, ge=0.0)
    llm_turns: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class TimingRecord(HandoffRecord):
    started_monotonic_s: float = Field(ge=0.0)
    ended_monotonic_s: float = Field(ge=0.0)
    observe_time_s: float = Field(default=0.0, ge=0.0)
    decide_time_s: float = Field(default=0.0, ge=0.0)
    record_time_s: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.ended_monotonic_s < self.started_monotonic_s:
            raise ValueError("timing end precedes start")
        return self


class TerminationRecord(HandoffRecord):
    reason: TerminationReason
    failure_mode: FailureMode = FailureMode.NONE
    final_governor_state: GovernorState
    episode_terminated: bool
    episode_truncated: bool
    message: str | None = None

class OutcomeRecord(HandoffRecord):
    schema_version: Literal[OUTCOME_SCHEMA_VERSION] = OUTCOME_SCHEMA_VERSION
    record_id: str
    identity: TrialIdentity
    skill: SkillIdentity
    controller: ControllerIdentity
    pre_handoff_state: HandoffState | None
    handoff_occurred: bool
    decision_trace: tuple[HandoffDecision, ...] = ()
    labels: OutcomeLabels
    costs: CostRecord
    timing: TimingRecord
    termination: TerminationRecord
    setup_record_id: str | None = None
    privileged_label_record_id: str | None = None
    source_revision: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("record_id")
    @classmethod
    def validate_record_id(cls, value: str) -> str:
        return _non_empty(value, "record_id")

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _json_metadata(value)

    @model_validator(mode="after")
    def validate_handoff(self) -> Self:
        if self.handoff_occurred and self.pre_handoff_state is None:
            raise ValueError("handoff outcome requires the achieved pre-handoff state")
        if (
            not self.handoff_occurred
            and self.costs.vla_invocations not in (None, 0)
        ):
            raise ValueError("VLA invocation count is nonzero without handoff")
        if self.termination.failure_mode is FailureMode.STAGING and self.handoff_occurred:
            raise ValueError("staging failure cannot be recorded as a VLA handoff outcome")
        return self


class GovernorTransition(HandoffRecord):
    sequence: int = Field(ge=0)
    from_state: GovernorState
    to_state: GovernorState
    elapsed_s: float = Field(ge=0.0)
    reason: str

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return _non_empty(value, "reason")
