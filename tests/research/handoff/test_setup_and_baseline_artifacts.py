from __future__ import annotations

from pathlib import Path

import pytest

from rpent.research.handoff.baseline_data import (
    build_positive_reference_artifact,
    load_positive_reference_artifact,
    write_positive_reference_artifact,
)
from rpent.research.handoff.dataset import OutcomeDataset
from rpent.research.handoff.experiments.setup_data import (
    SetupJsonlWriter,
    read_setup_records,
)
from rpent.research.handoff.privileged import (
    ExperimentSetupRecord,
    ProvenanceFirewallError,
    SetupValue,
)
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


def _identity(index: int) -> TrialIdentity:
    return TrialIdentity(
        run_id="run",
        episode_id=f"episode-{index}",
        trial_id=f"trial-{index}",
        invocation_id=f"invocation-{index}",
        candidate_id=f"candidate-{index}",
        suite="suite",
        task_id=0,
        seed=index,
        reset_id=f"reset-{index}",
    )


def _state(index: int) -> HandoffState:
    skill = SkillIdentity(name="pick", semantic_target="mug")
    return HandoffState(
        state_id=f"state-{index}",
        observation_sequence=0,
        observed_elapsed_s=0.0,
        eef_position_m=(0.0, 0.0, 0.08 + index * 0.001),
        eef_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        gripper_opening_m=0.08,
        skill=skill,
        target=TargetContext(
            target_id="mug",
            description="mug",
            estimate=TargetEstimate(
                estimate_id=f"target-{index}",
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


def _outcome(index: int, success: bool) -> OutcomeRecord:
    skill = SkillIdentity(name="pick", semantic_target="mug")
    return OutcomeRecord(
        record_id=f"outcome-{index}",
        identity=_identity(index),
        skill=skill,
        controller=ControllerIdentity(method="gate0", implementation_version="v1", configuration_id="cfg"),
        pre_handoff_state=_state(index),
        handoff_occurred=True,
        labels=OutcomeLabels(
            primitive_success=OutcomeSignal(value=success, source=LabelSource.PRIMITIVE_HEURISTIC, definition="fake"),
        ),
        costs=CostRecord(vla_invocations=1),
        timing=TimingRecord(started_monotonic_s=0.0, ended_monotonic_s=1.0),
        termination=TerminationRecord(reason=TerminationReason.HANDOFF_COMPLETED, failure_mode=FailureMode.NONE, final_governor_state=GovernorState.DONE, episode_terminated=False, episode_truncated=False),
    )


def test_setup_writer_is_idempotent_and_rejects_conflicting_retry(tmp_path: Path) -> None:
    path = tmp_path / "setups.jsonl"
    writer = SetupJsonlWriter(path)
    record = ExperimentSetupRecord(
        record_id="setup-1",
        identity=_identity(1),
        setup_provider="fake/v1",
        values=(SetupValue(name="target_position_m", values=(0.0, 0.0, 0.0), unit="m", frame="world", source="simulator"),),
    )
    writer.append_setup(record)
    writer.append_setup(record)
    assert read_setup_records(path) == (record,)
    conflict = record.model_copy(update={"notes": ("different",)})
    with pytest.raises(ValueError, match="conflicting setup retry"):
        writer.append_setup(conflict)


def test_positive_reference_artifact_uses_only_requested_positive_label(tmp_path: Path) -> None:
    dataset = OutcomeDataset.from_records((_outcome(1, True), _outcome(2, False)))
    artifact = build_positive_reference_artifact(
        dataset, target="primitive_success"
    )
    assert len(artifact.references) == 1
    assert artifact.references[0].reference_id.endswith("outcome-1")
    path = write_positive_reference_artifact(artifact, tmp_path / "positive.json")
    assert load_positive_reference_artifact(path) == artifact


def test_positive_reference_artifact_rejects_privileged_geometry() -> None:
    outcome = _outcome(1, True)
    state = outcome.pre_handoff_state
    assert state is not None and state.target is not None
    privileged_target = state.target.model_copy(
        update={
            "estimate": state.target.estimate.model_copy(
                update={
                    "availability": FeatureAvailability.SIMULATOR_PRIVILEGED,
                    "provider": "simulator-object-pose",
                }
            )
        }
    )
    privileged_provenance = tuple(
        item.model_copy(
            update={
                "availability": FeatureAvailability.SIMULATOR_PRIVILEGED,
                "source": "simulator-object-pose",
            }
        )
        if item.feature_name == "target_position_m"
        else item
        for item in state.provenance
    )
    privileged = outcome.model_copy(
        update={
            "pre_handoff_state": state.model_copy(
                update={
                    "target": privileged_target,
                    "provenance": privileged_provenance,
                }
            )
        }
    )

    with pytest.raises(ProvenanceFirewallError, match="non-deployment"):
        build_positive_reference_artifact(
            (privileged,), target="primitive_success"
        )
