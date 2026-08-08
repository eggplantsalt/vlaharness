from __future__ import annotations

from pathlib import Path

import pytest

from rpent.research.handoff.dataset import (
    DatasetConflictError,
    DatasetCorruptionError,
    DatasetResearchSink,
    ExclusionReason,
    OutcomeDataset,
    OutcomeJsonlWriter,
    TrainingTarget,
    dataset_fingerprint,
    extract_labeled_outcomes,
    read_decision_records,
    read_outcome_records,
    scan_decision_jsonl,
    scan_outcome_jsonl,
)
from rpent.research.handoff.splits import (
    GroupConstraint,
    GroupLeakageError,
    GroupSplitConfig,
    SplitAssignment,
    SplitEntry,
    SplitName,
    apply_split_assignment,
    connected_group_ids,
    split_outcomes,
    verify_no_group_leakage,
)
from rpent.research.handoff.types import (
    CandidateDecisionRecord,
    CandidateGeometry,
    ControllerIdentity,
    CostRecord,
    FailureMode,
    FeatureAvailability,
    FeatureProvenance,
    GovernorState,
    HandoffAction,
    HandoffDecision,
    HandoffState,
    LabelSource,
    OutcomeEstimate,
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


def _state(index: int) -> HandoffState:
    def provenance(
        name: str,
        availability: FeatureAvailability,
        unit: str,
        frame: str,
    ) -> FeatureProvenance:
        return FeatureProvenance(
            feature_name=name,
            availability=availability,
            source=f"fixture:{name}",
            unit=unit,
            frame=frame,
        )

    return HandoffState(
        state_id=f"state-{index}",
        observation_sequence=index,
        observed_elapsed_s=float(index),
        eef_position_m=(0.4 + index * 0.001, 0.0, 0.3),
        eef_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        gripper_opening_m=0.06,
        skill=SkillIdentity(name="pick", semantic_target="bowl"),
        provenance=(
            provenance(
                "eef_position_m", FeatureAvailability.DEPLOYMENT_SENSOR, "m", "world"
            ),
            provenance(
                "eef_quaternion_xyzw",
                FeatureAvailability.DEPLOYMENT_SENSOR,
                "unitless",
                "world",
            ),
            provenance(
                "gripper_opening_m",
                FeatureAvailability.DEPLOYMENT_SENSOR,
                "m",
                "gripper",
            ),
            provenance(
                "skill",
                FeatureAvailability.DERIVED_DEPLOYMENT,
                "categorical",
                "semantic",
            ),
        ),
    )


def _labels(*, skill_value: bool | None = True) -> OutcomeLabels:
    skill_signal = (
        unavailable_signal("skill-specific outcome")
        if skill_value is None
        else OutcomeSignal(
            value=skill_value,
            source=LabelSource.SKILL_EVALUATOR,
            definition="fixture skill evaluator",
        )
    )
    return OutcomeLabels(
        primitive_success=OutcomeSignal(
            value=True,
            source=LabelSource.PRIMITIVE_HEURISTIC,
            definition="fixture primitive heuristic",
        ),
        skill_success=skill_signal,
        task_success=OutcomeSignal(
            value=False,
            source=LabelSource.OFFICIAL_TERMINATION,
            definition="fixture official termination",
        ),
        episode_truncated=OutcomeSignal(
            value=False,
            source=LabelSource.RUNTIME,
            definition="fixture truncation",
        ),
        llm_finish=unavailable_signal("planner not used"),
    )


def _outcome(
    index: int,
    *,
    handoff: bool = True,
    vla_invocations: int | None = None,
    failure: FailureMode = FailureMode.NONE,
    reason: TerminationReason = TerminationReason.HANDOFF_COMPLETED,
    final_state: GovernorState = GovernorState.DONE,
    skill_value: bool | None = True,
    episode: str | None = None,
    reset: str | None = None,
    candidate: str | None = None,
    seed: int | None = None,
) -> OutcomeRecord:
    if vla_invocations is None:
        vla_invocations = int(handoff)
    return OutcomeRecord(
        record_id=f"outcome-{index}",
        identity=TrialIdentity(
            run_id="run-1",
            episode_id=episode or f"episode-{index}",
            trial_id=f"trial-{index}",
            invocation_id=f"invocation-{index}",
            candidate_id=candidate or f"candidate-{index}",
            suite="libero_test",
            task_id=0,
            seed=index if seed is None else seed,
            reset_id=reset or f"reset-{index}",
            repeat_index=index,
        ),
        skill=SkillIdentity(name="pick", semantic_target="bowl"),
        controller=ControllerIdentity(
            method="test-policy",
            implementation_version="v1",
            checkpoint_id="pi0.5-test",
            configuration_id="config-1",
        ),
        pre_handoff_state=_state(index) if handoff else None,
        handoff_occurred=handoff,
        labels=_labels(skill_value=skill_value),
        costs=CostRecord(vla_invocations=vla_invocations, total_elapsed_s=2.0),
        timing=TimingRecord(started_monotonic_s=1.0, ended_monotonic_s=3.0),
        termination=TerminationRecord(
            reason=reason,
            failure_mode=failure,
            final_governor_state=final_state,
            episode_terminated=False,
            episode_truncated=False,
        ),
    )


def _decision(index: int) -> HandoffDecision:
    candidate = CandidateGeometry(
        candidate_id=f"candidate-{index}",
        kind="current",
        eef_position_m=(0.4, 0.0, 0.3),
    )
    candidate_record = CandidateDecisionRecord(
        decision_sequence=index,
        candidate=candidate,
        estimate=OutcomeEstimate(
            mean_success_probability=0.8,
            epistemic_std=0.1,
            conservative_success_probability=0.6,
        ),
        handoff_cost=1.0,
        staging_cost=0.0,
        total_cost=1.0,
        selected=True,
    )
    return HandoffDecision(
        decision_id=f"decision-{index}",
        state_id=f"state-{index}",
        decision_sequence=index,
        action=HandoffAction.HANDOFF_NOW,
        selected_candidate_id=candidate.candidate_id,
        candidates=(candidate_record,),
        rationale="current candidate minimizes configured cost",
        policy_name="test-policy",
    )


def _append_torn_tail(path: Path) -> None:
    with path.open("ab") as stream:
        stream.write(b'{"incomplete":')


def test_research_sink_is_idempotent_conflict_checked_and_torn_tail_resumable(
    tmp_path,
) -> None:
    sink = DatasetResearchSink(tmp_path)
    decision = _decision(0)
    outcome = _outcome(0)

    sink.append_decision(decision)
    sink.append_outcome(outcome)
    sink.append_decision(decision)
    sink.append_outcome(outcome)

    with pytest.raises(DatasetConflictError):
        sink.append_decision(decision.model_copy(update={"rationale": "changed"}))
    with pytest.raises(DatasetConflictError):
        sink.append_outcome(outcome.model_copy(update={"metadata": {"changed": True}}))

    _append_torn_tail(sink.decision_path)
    _append_torn_tail(sink.outcome_path)
    assert scan_decision_jsonl(sink.decision_path).partial_tail_bytes > 0
    assert scan_outcome_jsonl(sink.outcome_path).partial_tail_bytes > 0

    resumed = DatasetResearchSink(tmp_path)
    resumed.append_decision(_decision(1))
    resumed.append_outcome(_outcome(1))
    assert [item.decision_id for item in read_decision_records(sink.decision_path)] == [
        "decision-0",
        "decision-1",
    ]
    assert [item.record_id for item in read_outcome_records(sink.outcome_path)] == [
        "outcome-0",
        "outcome-1",
    ]


def test_complete_malformed_line_is_never_silently_ignored(tmp_path) -> None:
    path = tmp_path / "outcomes.jsonl"
    path.write_bytes(b"{}\n")
    with pytest.raises(DatasetCorruptionError):
        OutcomeJsonlWriter(path)


def test_outcome_reader_and_fingerprint_are_strict_and_order_independent(
    tmp_path,
) -> None:
    records = (_outcome(0), _outcome(1), _outcome(2))
    writer = OutcomeJsonlWriter(tmp_path / "outcomes.jsonl")
    for record in records:
        writer.append(record)

    dataset = OutcomeDataset.from_jsonl(tmp_path / "outcomes.jsonl")
    assert dataset.records == records
    assert dataset.fingerprint == dataset_fingerprint(tuple(reversed(records)))
    assert OutcomeDataset.from_records(records).fingerprint == dataset.fingerprint

    with pytest.raises(DatasetCorruptionError):
        OutcomeDataset.from_records((records[0], records[0]))


def test_label_extraction_never_turns_non_vla_failures_into_negatives() -> None:
    success = _outcome(0)
    no_handoff = _outcome(
        1,
        handoff=False,
        reason=TerminationReason.ABORTED,
        final_state=GovernorState.ABORT,
    )
    staging = _outcome(
        2,
        handoff=False,
        failure=FailureMode.STAGING,
        reason=TerminationReason.STAGING_FAILURE,
        final_state=GovernorState.STAGING_FAILURE,
    )
    perception = _outcome(
        3,
        handoff=False,
        failure=FailureMode.PERCEPTION,
        reason=TerminationReason.PERCEPTION_FAILURE,
        final_state=GovernorState.PERCEPTION_FAILURE,
    )
    no_invocation = _outcome(4, handoff=True, vla_invocations=0)
    unknown = _outcome(5, skill_value=None)

    result = extract_labeled_outcomes(
        (success, no_handoff, staging, perception, no_invocation, unknown),
        target=TrainingTarget.SKILL_SUCCESS,
    )
    assert [(item.record.record_id, item.value) for item in result.included] == [
        ("outcome-0", True)
    ]
    reasons = {item.record_id: item.reason for item in result.excluded}
    assert reasons == {
        "outcome-1": ExclusionReason.NO_HANDOFF,
        "outcome-2": ExclusionReason.STAGING_FAILURE,
        "outcome-3": ExclusionReason.PERCEPTION_FAILURE,
        "outcome-4": ExclusionReason.NO_VLA_INVOCATION,
        "outcome-5": ExclusionReason.TARGET_LABEL_UNAVAILABLE,
    }


def _grouped_records() -> tuple[OutcomeRecord, ...]:
    records = []
    for component in range(6):
        for repeat in range(2):
            index = component * 2 + repeat
            records.append(
                _outcome(
                    index,
                    episode=f"episode-group-{component}",
                    reset=f"reset-group-{component}",
                    candidate=f"candidate-group-{component}",
                    seed=component,
                )
            )
    return tuple(records)


def test_connected_group_split_blocks_episode_reset_and_candidate_leakage() -> None:
    records = _grouped_records()
    config = GroupSplitConfig(seed=17)
    result = split_outcomes(records, config)
    reversed_result = split_outcomes(tuple(reversed(records)), config)

    assert result.assignment == reversed_result.assignment
    assert result.train and result.calibration and result.test
    verify_no_group_leakage(records, result.assignment, config.constraints)

    groups = connected_group_ids(records, config.constraints)
    for component in range(6):
        left = f"outcome-{component * 2}"
        right = f"outcome-{component * 2 + 1}"
        assert groups[left] == groups[right]
    assert len(set(groups.values())) == 6


def test_saved_assignment_materializes_the_exact_eligible_cohort() -> None:
    records = _grouped_records()
    result = split_outcomes(records, GroupSplitConfig(seed=17))
    unrelated = _outcome(100)

    partitions = apply_split_assignment((*records, unrelated), result.assignment)

    assert partitions[SplitName.TRAIN] == result.train
    assert partitions[SplitName.CALIBRATION] == result.calibration
    assert partitions[SplitName.TEST] == result.test
    assert all(
        unrelated not in partition for partition in partitions.values()
    )


def test_saved_assignment_rejects_changed_record_contents() -> None:
    records = list(_grouped_records())
    result = split_outcomes(records, GroupSplitConfig(seed=17))
    records[0] = records[0].model_copy(update={"metadata": {"tampered": True}})

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        apply_split_assignment(records, result.assignment)


def test_connected_groups_include_transitive_links_across_constraints() -> None:
    records = (
        _outcome(
            20,
            episode="shared-episode",
            candidate="candidate-a",
            reset="reset-a",
        ),
        _outcome(
            21,
            episode="shared-episode",
            candidate="shared-candidate",
            reset="reset-b",
        ),
        _outcome(
            22,
            episode="other-episode",
            candidate="shared-candidate",
            reset="shared-reset",
        ),
        _outcome(
            23,
            episode="episode-d",
            candidate="candidate-d",
            reset="shared-reset",
        ),
    )
    constraints = (
        GroupConstraint(name="episode", fields=("identity.episode_id",)),
        GroupConstraint(name="candidate", fields=("identity.candidate_id",)),
        GroupConstraint(name="reset", fields=("identity.reset_id",)),
    )
    groups = connected_group_ids(records, constraints)
    assert len(set(groups.values())) == 1


def test_leakage_verifier_rejects_a_manually_corrupted_assignment() -> None:
    records = _grouped_records()
    config = GroupSplitConfig(seed=9)
    result = split_outcomes(records, config)
    entries = list(result.assignment.entries)
    current = entries[0].split
    replacement = SplitName.TEST if current is not SplitName.TEST else SplitName.TRAIN
    entries[0] = SplitEntry(record_id=entries[0].record_id, split=replacement)
    corrupted = SplitAssignment(
        dataset_fingerprint=result.assignment.dataset_fingerprint,
        config_fingerprint=result.assignment.config_fingerprint,
        entries=tuple(entries),
    )

    with pytest.raises(GroupLeakageError, match="leaks"):
        verify_no_group_leakage(records, corrupted, config.constraints)
