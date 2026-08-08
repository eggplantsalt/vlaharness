from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from rpent.research.handoff.features import (
    FeatureBuilder,
    FeatureCompatibilityError,
    FeaturePreset,
    make_feature_spec,
)
from rpent.research.handoff.privileged import (
    ExperimentSetupRecord,
    PrivilegedEvaluatorRecord,
    PrivilegedEvaluatorSignal,
    PrivilegedObservationRecord,
    PrivilegedValue,
    ProvenanceFirewallError,
    SetupValue,
    reject_privileged_policy_value,
)
from rpent.research.handoff.types import (
    CandidateGeometry,
    FeatureAvailability,
    FeatureProvenance,
    HandoffState,
    SkillIdentity,
    TargetContext,
    TargetEstimate,
    TrialIdentity,
    VisualGeometry,
)


def _source(
    name: str,
    availability: FeatureAvailability,
    *,
    unit: str = "unitless",
    frame: str = "world",
) -> FeatureProvenance:
    return FeatureProvenance(
        feature_name=name,
        availability=availability,
        source=f"test:{name}",
        unit=unit,
        frame=frame,
    )


def _state() -> HandoffState:
    target = TargetEstimate(
        estimate_id="target-estimate-1",
        position_m=(0.4, 0.1, 0.2),
        frame="world",
        provider="test-perception",
        availability=FeatureAvailability.DEPLOYMENT_PERCEPTION,
        confidence=0.8,
        observation_sequence=3,
        visual_geometry=VisualGeometry(
            mask_area_fraction=0.2,
            valid_depth_fraction=0.75,
            image_centroid_rc_normalized=(0.25, 0.6),
            camera_name="agentview",
        ),
    )
    return HandoffState(
        state_id="state-1",
        observation_sequence=3,
        observed_elapsed_s=1.25,
        eef_position_m=(0.5, 0.0, 0.3),
        eef_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        gripper_opening_m=0.06,
        skill=SkillIdentity(name="pick", semantic_target="bowl"),
        target=TargetContext(
            target_id="bowl",
            description="black bowl",
            estimate=target,
        ),
        provenance=(
            _source(
                "eef_position_m", FeatureAvailability.DEPLOYMENT_SENSOR, unit="m"
            ),
            _source(
                "eef_quaternion_xyzw", FeatureAvailability.DEPLOYMENT_SENSOR
            ),
            _source(
                "gripper_opening_m",
                FeatureAvailability.DEPLOYMENT_SENSOR,
                unit="m",
                frame="gripper",
            ),
            _source("skill", FeatureAvailability.DERIVED_DEPLOYMENT, frame="semantic"),
            _source(
                "target_position_m",
                FeatureAvailability.DEPLOYMENT_PERCEPTION,
                unit="m",
            ),
            _source(
                "target_confidence", FeatureAvailability.DEPLOYMENT_PERCEPTION
            ),
            _source(
                "mask_area_fraction",
                FeatureAvailability.DEPLOYMENT_PERCEPTION,
                frame="camera",
            ),
            _source(
                "valid_depth_fraction",
                FeatureAvailability.DEPLOYMENT_PERCEPTION,
                frame="camera",
            ),
            _source(
                "image_centroid_rc_normalized",
                FeatureAvailability.DEPLOYMENT_PERCEPTION,
                frame="camera",
            ),
        ),
    )


def _identity() -> TrialIdentity:
    return TrialIdentity(
        run_id="run",
        episode_id="episode",
        trial_id="trial",
        invocation_id="invocation",
        candidate_id="candidate",
        suite="libero_test",
        task_id=0,
        seed=1,
        reset_id="reset",
    )


def test_feature_presets_are_ordered_fingerprinted_and_numpy_free() -> None:
    state = _state()
    for preset in FeaturePreset:
        first = make_feature_spec(preset, skill_vocabulary=("push", "pick"))
        second = make_feature_spec(preset, skill_vocabulary=("pick", "push"))
        assert first.names == second.names
        assert first.fingerprint == second.fingerprint

        vector = FeatureBuilder(first).build(state)
        assert vector.spec_id == first.spec_id
        assert vector.spec_fingerprint == first.fingerprint
        assert vector.names == first.names
        assert all(type(value) is float for value in vector.values)
        assert all(math.isfinite(value) for value in vector.values)
        first.validate_vector(vector)


def test_target_relative_values_and_skill_one_hot_are_explicit() -> None:
    spec = make_feature_spec(
        FeaturePreset.TARGET_RELATIVE,
        skill_vocabulary=("pick", "push"),
    )
    vector = FeatureBuilder().build(_state(), spec)
    values = dict(zip(vector.names, vector.values, strict=True))

    assert values["eef_minus_target_x_m"] == pytest.approx(0.1)
    assert values["eef_minus_target_y_m"] == pytest.approx(-0.1)
    assert values["eef_minus_target_z_m"] == pytest.approx(0.1)
    assert values["skill=pick"] == 1.0
    assert values["skill=push"] == 0.0


def test_missing_visual_values_use_finite_fill_and_availability_indicators() -> None:
    state = _state()
    estimate = state.target.estimate.model_copy(
        update={"confidence": None, "visual_geometry": None}
    )
    target = state.target.model_copy(update={"estimate": estimate})
    minimal_provenance = tuple(
        item
        for item in state.provenance
        if item.feature_name
        not in {
            "target_confidence",
            "mask_area_fraction",
            "valid_depth_fraction",
            "image_centroid_rc_normalized",
        }
    )
    state = state.model_copy(
        update={"target": target, "provenance": minimal_provenance}
    )
    spec = make_feature_spec(
        FeaturePreset.TARGET_RELATIVE_VISUAL,
        skill_vocabulary=("pick",),
    )
    vector = FeatureBuilder(spec).build(state)
    values = dict(zip(vector.names, vector.values, strict=True))

    for name in (
        "target_confidence",
        "mask_area_fraction",
        "valid_depth_fraction",
        "target_centroid_row",
        "target_centroid_col",
    ):
        assert values[name] == 0.0
    for name in (
        "target_confidence_available",
        "mask_area_fraction_available",
        "valid_depth_fraction_available",
        "target_centroid_available",
    ):
        assert values[name] == 0.0
    assert all(math.isfinite(value) for value in vector.values)


def test_online_firewall_rejects_privileged_source_provenance() -> None:
    state = _state()
    replaced = tuple(
        _source("eef_position_m", FeatureAvailability.SIMULATOR_PRIVILEGED, unit="m")
        if item.feature_name == "eef_position_m"
        else item
        for item in state.provenance
    )
    state = state.model_copy(update={"provenance": replaced})
    spec = make_feature_spec(FeaturePreset.ABSOLUTE, skill_vocabulary=("pick",))

    with pytest.raises(ProvenanceFirewallError, match="non-deployment"):
        FeatureBuilder(spec).build(state)


def test_online_firewall_checks_nested_target_availability_independently() -> None:
    state = _state()
    estimate = state.target.estimate.model_copy(
        update={"availability": FeatureAvailability.SIMULATOR_PRIVILEGED}
    )
    state = state.model_copy(
        update={"target": state.target.model_copy(update={"estimate": estimate})}
    )
    spec = make_feature_spec(
        FeaturePreset.TARGET_RELATIVE,
        skill_vocabulary=("pick",),
    )

    with pytest.raises(ProvenanceFirewallError, match="target estimate"):
        FeatureBuilder(spec).build(state)


def test_builder_requires_bound_or_per_call_spec() -> None:
    with pytest.raises(FeatureCompatibilityError, match="requires"):
        FeatureBuilder().build(_state())


def test_setup_privileged_and_evaluator_records_are_separate_and_strict() -> None:
    identity = _identity()
    setup = ExperimentSetupRecord(
        record_id="setup-1",
        identity=identity,
        setup_provider="gate0",
        requested_candidate=CandidateGeometry(
            candidate_id="candidate",
            kind="standoff",
            eef_position_m=(0.5, 0.0, 0.3),
        ),
        values=(
            SetupValue(
                name="requested_standoff",
                values=(0.1,),
                unit="m",
                frame="target",
                source="gate0-config",
            ),
        ),
    )
    privileged = PrivilegedObservationRecord(
        record_id="privileged-1",
        identity=identity,
        values=(
            PrivilegedValue(
                name="simulator_object_position",
                values=(0.4, 0.1, 0.2),
                unit="m",
                frame="world",
                source="simulator_raw_obs",
            ),
        ),
    )
    evaluator = PrivilegedEvaluatorRecord(
        record_id="evaluator-1",
        identity=identity,
        evaluator_id="correct-object-grasp",
        evaluator_version="v1",
        signals=(
            PrivilegedEvaluatorSignal(
                name="correct_object_grasped",
                value=True,
                definition="simulator-only object identity evaluator",
            ),
        ),
    )

    for record in (setup, privileged, evaluator):
        with pytest.raises(ProvenanceFirewallError):
            reject_privileged_policy_value(record)

    with pytest.raises(ValidationError):
        ExperimentSetupRecord(
            record_id="setup-2",
            identity=identity,
            setup_provider="gate0",
            unexpected_privileged_pose=(1, 2, 3),
        )
