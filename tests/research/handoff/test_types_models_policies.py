from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from rpent.research.handoff.candidates import (
    CandidateFeaturePredictor,
    CandidateFeaturePredictorConfig,
    CandidateGeneratorConfig,
    ObjectRelativeCandidateGenerator,
)
from rpent.research.handoff.features import FeaturePreset, make_feature_spec
from rpent.research.handoff.model import ConstantOutcomeModel, ModelCompatibilityError
from rpent.research.handoff.policies import (
    DirectHandoffPolicy,
    OutcomeCalibratedSwitchingPolicy,
    PolicyContext,
    RiskAwareSwitchingConfig,
)
from rpent.research.handoff.types import (
    FeatureAvailability,
    FeatureProvenance,
    HandoffAction,
    HandoffState,
    LabelSource,
    OutcomeEstimate,
    OutcomeLabels,
    OutcomeSignal,
    SkillIdentity,
    TargetContext,
    TargetEstimate,
)


def _state(*, eef_z: float = 0.2, target_available: bool = True) -> HandoffState:
    estimate = TargetEstimate(
        estimate_id="target-0",
        position_m=(0.0, 0.0, 0.0) if target_available else None,
        frame="world",
        provider="fake-perception/v1",
        availability=FeatureAvailability.DEPLOYMENT_PERCEPTION,
        confidence=0.8 if target_available else None,
        observation_sequence=0,
        unavailable_reason=None if target_available else "disabled",
    )
    return HandoffState(
        state_id=f"state-{eef_z}",
        observation_sequence=0,
        observed_elapsed_s=0.0,
        eef_position_m=(0.0, 0.0, eef_z),
        eef_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        gripper_opening_m=0.08,
        skill=SkillIdentity(name="pick", semantic_target="red mug"),
        target=TargetContext(
            target_id="red-mug", description="red mug", estimate=estimate
        ),
        provenance=(
            FeatureProvenance(feature_name="eef_position_m", availability=FeatureAvailability.DEPLOYMENT_SENSOR, source="fake", unit="m", frame="world"),
            FeatureProvenance(feature_name="eef_quaternion_xyzw", availability=FeatureAvailability.DEPLOYMENT_SENSOR, source="fake", unit="unit_quaternion", frame="world"),
            FeatureProvenance(feature_name="gripper_opening_m", availability=FeatureAvailability.DEPLOYMENT_SENSOR, source="fake", unit="m", frame="gripper"),
            FeatureProvenance(feature_name="skill", availability=FeatureAvailability.DERIVED_DEPLOYMENT, source="fake", unit="categorical", frame="semantic"),
            FeatureProvenance(feature_name="target_position_m", availability=FeatureAvailability.DEPLOYMENT_PERCEPTION, source="fake", unit="m", frame="world"),
        ),
    )


def test_schema_round_trip_is_canonical_and_extra_fields_fail() -> None:
    state = _state()
    assert HandoffState.from_json(state.canonical_json()) == state
    payload = state.model_dump(mode="json")
    payload["simulator_contact"] = True
    with pytest.raises(ValidationError):
        HandoffState.model_validate(payload)


def test_outcome_labels_never_substitute_task_for_primitive_success() -> None:
    labels = OutcomeLabels(
        primitive_success=OutcomeSignal(value=False, source=LabelSource.PRIMITIVE_HEURISTIC, definition="grasp heuristic"),
        task_success=OutcomeSignal(value=True, source=LabelSource.OFFICIAL_TERMINATION, definition="official termination"),
    )
    assert labels.target_value("primitive_success") is False
    assert labels.target_value("task_success") is True


def test_direct_policy_handles_current_only_without_target_geometry() -> None:
    state = _state(target_available=False)
    candidates = ObjectRelativeCandidateGenerator(
        CandidateGeneratorConfig(standoff_distances_m=(0.08,), max_candidates=1)
    ).generate(state)
    assert len(candidates) == 1
    decision = DirectHandoffPolicy().decide(
        PolicyContext(current_state=state, candidates=candidates, decision_sequence=0)
    )
    assert decision.action is HandoffAction.HANDOFF_NOW


def test_constant_model_rejects_feature_identity_mismatch() -> None:
    spec = make_feature_spec(FeaturePreset.ABSOLUTE, skill_vocabulary=("pick",))
    model = ConstantOutcomeModel(
        probability=0.5,
        feature_spec_id=spec.spec_id,
        feature_names=spec.names,
    )

    @dataclass(frozen=True)
    class WrongVector:
        spec_id: str = "wrong"
        names: tuple[str, ...] = spec.names
        values: tuple[float, ...] = tuple(0.0 for _ in spec.names)

    with pytest.raises(ModelCompatibilityError):
        model.predict_one(WrongVector())


class _PositionBuilder:
    @dataclass(frozen=True)
    class Vector:
        spec_id: str
        names: tuple[str, ...]
        values: tuple[float, ...]

    def build(self, state: HandoffState) -> Vector:
        return self.Vector("position", ("z",), (state.eef_position_m[2],))


class _UncertainPositionModel:
    feature_spec_id = "position"
    feature_names = ("z",)

    def predict_one(self, features) -> OutcomeEstimate:
        future = features.values[0] < 0.15
        return OutcomeEstimate(
            mean_success_probability=0.9 if future else 0.6,
            epistemic_std=0.3 if future else 0.05,
            conservative_success_probability=0.2 if future else 0.55,
            lower_quantile_probability=0.2 if future else 0.55,
            upper_quantile_probability=0.95 if future else 0.65,
            ensemble_size=20,
            calibrated=True,
        )


def test_uncertainty_mode_changes_risk_aware_switching_decision() -> None:
    state = _state(eef_z=0.2)
    candidates = ObjectRelativeCandidateGenerator(
        CandidateGeneratorConfig(
            standoff_distances_m=(0.08,),
            xyz_perturbations_m=((0.0, 0.0, 0.0),),
            max_candidates=1,
        )
    ).generate(state)
    context = PolicyContext(state, candidates, 0)
    common = dict(
        model=_UncertainPositionModel(),
        feature_builder=_PositionBuilder(),
        predictor=CandidateFeaturePredictor(
            CandidateFeaturePredictorConfig(visual_prediction="mark_unavailable")
        ),
    )
    mean = OutcomeCalibratedSwitchingPolicy(
        **common,
        config=RiskAwareSwitchingConfig(
            vla_cost=0.0,
            failure_cost=1.0,
            stage_cost_per_m=0.5,
            probability_mode="mean",
        ),
    ).decide(context)
    conservative = OutcomeCalibratedSwitchingPolicy(
        **common,
        config=RiskAwareSwitchingConfig(
            vla_cost=0.0,
            failure_cost=1.0,
            stage_cost_per_m=0.5,
            probability_mode="conservative",
        ),
    ).decide(context)
    assert mean.action is HandoffAction.CONTINUE
    assert conservative.action is HandoffAction.HANDOFF_NOW
