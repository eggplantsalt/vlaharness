from __future__ import annotations

from pathlib import Path

import pytest

from rpent.research.handoff.artifacts import (
    SourceIdentity,
    load_model_artifact,
    save_model_artifact,
)
from rpent.research.handoff.dataset import OutcomeDataset
from rpent.research.handoff.features import FeaturePreset, make_feature_spec
from rpent.research.handoff.model import ModelCompatibilityError
from rpent.research.handoff.training import OutcomeTrainingConfig, train_outcome_model
from rpent.research.handoff.types import (
    ControllerIdentity,
    CostRecord,
    FailureMode,
    FeatureAvailability,
    FeatureProvenance,
    GovernorState,
    HandoffState,
    LabelSource,
    OutcomeLabels,
    OutcomeRecord,
    OutcomeSignal,
    SkillIdentity,
    TargetContext,
    TargetEstimate,
    TerminationReason,
    TerminationRecord,
    TimingRecord,
    TrialIdentity,
)


def _record(group: int, success: bool) -> OutcomeRecord:
    suffix = "success" if success else "failure"
    identity = TrialIdentity(
        run_id="run",
        episode_id=f"episode-{group}",
        trial_id=f"trial-{group}-{suffix}",
        invocation_id=f"invocation-{group}-{suffix}",
        candidate_id=f"candidate-{group}",
        suite="suite",
        task_id=0,
        seed=group,
        reset_id=f"reset-{group}",
    )
    skill = SkillIdentity(name="pick", semantic_target="mug")
    state = HandoffState(
        state_id=f"state-{group}-{suffix}",
        observation_sequence=0,
        observed_elapsed_s=0.0,
        eef_position_m=(group * 0.001, 0.0, 0.08),
        eef_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        gripper_opening_m=0.08 if success else 0.02,
        skill=skill,
        target=TargetContext(
            target_id="mug",
            description="mug",
            estimate=TargetEstimate(
                estimate_id=f"target-{group}",
                position_m=(0.0, 0.0, 0.0),
                frame="world",
                provider="fake",
                availability=FeatureAvailability.DEPLOYMENT_PERCEPTION,
                observation_sequence=0,
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
    return OutcomeRecord(
        record_id=f"outcome-{group}-{suffix}",
        identity=identity,
        skill=skill,
        controller=ControllerIdentity(method="gate0", implementation_version="v1", configuration_id="cfg"),
        pre_handoff_state=state,
        handoff_occurred=True,
        labels=OutcomeLabels(primitive_success=OutcomeSignal(value=success, source=LabelSource.PRIMITIVE_HEURISTIC, definition="fake")),
        costs=CostRecord(vla_invocations=1),
        timing=TimingRecord(started_monotonic_s=0.0, ended_monotonic_s=1.0),
        termination=TerminationRecord(reason=TerminationReason.HANDOFF_COMPLETED, failure_mode=FailureMode.NONE, final_governor_state=GovernorState.DONE, episode_terminated=False, episode_truncated=False),
    )


def test_training_split_calibration_target_and_artifact_compatibility(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    records = tuple(
        _record(group, success)
        for group in range(24)
        for success in (False, True)
    )
    dataset = OutcomeDataset.from_records(records)
    spec = make_feature_spec(FeaturePreset.ABSOLUTE, skill_vocabulary=("pick",))
    config = OutcomeTrainingConfig(
        target_label="primitive_success",
        estimator_kind="logistic",
        calibration_method="none",
        bootstrap_ensemble_size=1,
        random_state=7,
    )
    result = train_outcome_model(dataset, feature_spec=spec, config=config)
    assert result.report.target_label == "primitive_success"
    assert result.report.train.successes == result.report.train.failures
    artifact_dir = tmp_path / "model"
    manifest = save_model_artifact(
        artifact_dir,
        model=result.model,
        feature_spec=spec,
        training_target_label="primitive_success",
        dataset_fingerprint=dataset.fingerprint,
        training_configuration=config.model_dump(mode="json"),
        source_identity=SourceIdentity(git_revision="test"),
        calibration_method=config.calibration_method,
        split_assignment_fingerprint=result.assignment.fingerprint,
        training_record_ids=tuple(
            record.record_id for record in result.train.records
        ),
        calibration_record_ids=tuple(
            record.record_id for record in result.calibration.records
        ),
        held_out_record_ids=tuple(
            record.record_id for record in result.test.records
        ),
    )
    loaded, loaded_manifest = load_model_artifact(
        artifact_dir,
        trusted=True,
        expected_feature_spec=spec,
        expected_dataset_fingerprint=dataset.fingerprint,
    )
    assert loaded.feature_spec_id == spec.spec_id
    assert loaded_manifest == manifest
    incompatible = make_feature_spec(
        FeaturePreset.TARGET_RELATIVE, skill_vocabulary=("pick",)
    )
    with pytest.raises(ModelCompatibilityError):
        load_model_artifact(
            artifact_dir,
            trusted=True,
            expected_feature_spec=incompatible,
        )
