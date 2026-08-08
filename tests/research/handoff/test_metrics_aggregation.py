from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpent.research.handoff.evaluation.aggregate import (
    aggregate_outcomes,
    read_outcome_jsonl,
    write_aggregation,
)
from rpent.research.handoff.evaluation.metrics import (
    evaluate_binary_predictions,
    evaluate_controller_records,
    evaluate_system_records,
    handoff_regret,
)
from rpent.research.handoff.evaluation.oracle import (
    OracleCostConfig,
    annotate_matched_oracle_costs,
)
from rpent.research.handoff.evaluation import plotting
from rpent.research.handoff.evaluation.plotting import (
    plot_gate0_landscape,
    plot_method_success_cost,
)
from rpent.research.handoff.evaluation.statistics import grouped_bootstrap_interval
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
    TerminationReason,
    TerminationRecord,
    TimingRecord,
    TrialIdentity,
)


def test_binary_metrics_and_single_class_edges_are_explicit() -> None:
    evaluation = evaluate_binary_predictions(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.35, 0.8],
        uncertainties=[0.1, 0.3, 0.2, 0.05],
    )
    assert evaluation.auroc.value == pytest.approx(0.75)
    assert evaluation.auprc.value == pytest.approx(5 / 6)
    assert evaluation.brier.value == pytest.approx(0.158125)
    assert evaluation.log_loss.value is not None
    assert evaluation.calibration.expected_calibration_error.value is not None
    assert evaluation.risk_coverage is not None

    single_class = evaluate_binary_predictions([1, 1], [0.2, 0.9])
    assert single_class.auroc.value is None
    assert "single-class" in (single_class.auroc.reason or "")


def test_grouped_bootstrap_and_regret_are_deterministic() -> None:
    first = grouped_bootstrap_interval(
        [0, 1, 1, 1],
        ["reset-a", "reset-a", "reset-b", "reset-b"],
        iterations=100,
        seed=7,
    )
    second = grouped_bootstrap_interval(
        [0, 1, 1, 1],
        ["reset-a", "reset-a", "reset-b", "reset-b"],
        iterations=100,
        seed=7,
    )
    assert first == second
    assert first.n_groups == 2
    assert handoff_regret([4.0, 2.0], [3.0, 1.5]).mean_regret.value == pytest.approx(0.75)


def _record(index: int, *, success: bool, data_status: str = "observed") -> OutcomeRecord:
    skill = SkillIdentity(name="pick", semantic_target="cup")
    state = HandoffState(
        state_id=f"state-{index}",
        observation_sequence=0,
        observed_elapsed_s=0.0,
        eef_position_m=(0.1, 0.2, 0.3),
        eef_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        gripper_opening_m=0.04,
        skill=skill,
        provenance=tuple(
            FeatureProvenance(
                feature_name=name,
                availability=(
                    FeatureAvailability.DERIVED_DEPLOYMENT
                    if name == "skill"
                    else FeatureAvailability.DEPLOYMENT_SENSOR
                ),
                source="fixture",
                unit=unit,
                frame=frame,
            )
            for name, unit, frame in (
                ("eef_position_m", "m", "world"),
                ("eef_quaternion_xyzw", "unitless", "world"),
                ("gripper_opening_m", "m", "gripper"),
                ("skill", "categorical", "semantic"),
            )
        ),
    )
    labels = OutcomeLabels(
        primitive_success=OutcomeSignal(
            value=success,
            source=LabelSource.PRIMITIVE_HEURISTIC,
            definition="fixture primitive success",
        ),
        skill_success=OutcomeSignal(
            value=success,
            source=LabelSource.SKILL_EVALUATOR,
            definition="fixture skill success",
        ),
        task_success=OutcomeSignal(
            value=success,
            source=LabelSource.OFFICIAL_TERMINATION,
            definition="fixture official task success",
        ),
        episode_truncated=OutcomeSignal(
            value=False,
            source=LabelSource.RUNTIME,
            definition="fixture truncation",
        ),
    )
    return OutcomeRecord(
        record_id=f"record-{index}",
        identity=TrialIdentity(
            run_id="run",
            episode_id=f"episode-{index}",
            trial_id=f"trial-{index}",
            invocation_id=f"invocation-{index}",
            candidate_id=f"candidate-{index}",
            suite="libero_object",
            task_id=1,
            seed=index,
            reset_id=f"reset-{index}",
        ),
        skill=skill,
        controller=ControllerIdentity(
            method="ours",
            implementation_version="v1",
            configuration_id="cfg",
        ),
        pre_handoff_state=state,
        handoff_occurred=True,
        labels=labels,
        costs=CostRecord(
            analytic_steps=index + 1,
            analytic_distance_m=0.01 * index,
            analytic_time_s=0.1,
            vla_invocations=1,
            vla_time_s=0.2,
            total_elapsed_s=0.3,
        ),
        timing=TimingRecord(started_monotonic_s=0.0, ended_monotonic_s=0.3),
        termination=TerminationRecord(
            reason=TerminationReason.HANDOFF_COMPLETED,
            failure_mode=FailureMode.NONE,
            final_governor_state=GovernorState.DONE,
            episode_terminated=success,
            episode_truncated=False,
        ),
        metadata={
            "data_status": data_status,
            "chosen_cost": 2.0 + index,
            "oracle_cost": 1.5 + index,
        },
    )


def test_outcome_aggregation_writes_real_csv_and_json(tmp_path) -> None:
    complete = _record(0, success=True)
    incomplete_source = _record(1, success=False)
    incomplete = incomplete_source.model_copy(
        update={
            "labels": incomplete_source.labels.model_copy(
                update={
                    "task_success": OutcomeSignal(
                        value=None,
                        source=LabelSource.UNAVAILABLE,
                        definition="fixture incomplete official outcome",
                    )
                }
            ),
            "metadata": {
                **incomplete_source.metadata,
                "incomplete_execution": True,
                "denominator_eligible": True,
                "system_attempt_success": False,
            },
        }
    )
    records = (complete, incomplete)
    source = tmp_path / "outcomes.jsonl"
    source.write_text(
        "".join(record.canonical_json() + "\n" for record in records),
        encoding="utf-8",
    )
    loaded = read_outcome_jsonl(source)
    result = aggregate_outcomes(
        loaded,
        target_label="skill_success",
        bootstrap_iterations=50,
    )
    artifacts = write_aggregation(result, loaded, tmp_path / "aggregate")

    assert result.n_records == 2
    assert result.schema_version == "rpent.handoff-aggregation/v2"
    assert result.overall.controller_metrics["skill_success_rate"].value == 0.5
    assert incomplete.labels.task_success.value is None
    assert result.overall.system_attempt_success_interval.n_observations == 2
    assert result.overall.system_attempt_success_interval.estimate == pytest.approx(0.5)
    assert result.overall.handoff_regret.value == 0.5
    assert (tmp_path / "aggregate" / "results.csv").is_file()
    summary = json.loads((tmp_path / "aggregate" / "summary.json").read_text())
    assert summary["n_records"] == 2
    assert artifacts.calibration_csv is None


def test_aggregation_refuses_cross_layer_or_cross_scope_pooling() -> None:
    controlled = _record(0, success=True).model_copy(
        update={"metadata": {"execution_layer": "controlled"}}
    )
    full_agent = _record(1, success=False).model_copy(
        update={
            "metadata": {
                "execution_layer": "full_agent",
                "record_scope": "full_agent_episode",
            }
        }
    )

    with pytest.raises(ValueError, match="mixed execution layers or record scopes"):
        aggregate_outcomes(
            (controlled, full_agent),
            target_label="skill_success",
            bootstrap_iterations=10,
        )


def _controlled_record(
    index: int,
    *,
    condition: str,
    reset_id: str,
) -> OutcomeRecord:
    source = _record(index, success=index % 2 == 0)
    return source.model_copy(
        update={
            "identity": source.identity.model_copy(
                update={
                    "seed": 0,
                    "reset_id": reset_id,
                    "repeat_index": 0,
                }
            ),
            "controller": source.controller.model_copy(
                update={
                    "method": f"method-{condition}",
                    "configuration_id": f"config-{condition}",
                }
            ),
            "metadata": {
                **source.metadata,
                "condition": condition,
                "execution_layer": "controlled",
                "record_scope": "handoff_invocation",
                "protocol_adherent": True,
            },
        }
    )


def test_controlled_aggregation_requires_complete_matched_reset_panel() -> None:
    first = _controlled_record(0, condition="direct", reset_id="shared-reset")
    second = _controlled_record(1, condition="ours", reset_id="shared-reset")

    result = aggregate_outcomes(
        (first, second),
        target_label="skill_success",
        bootstrap_iterations=10,
    )

    assert result.n_records == 2
    assert {group.group["condition"] for group in result.per_method} == {
        "direct",
        "ours",
    }

    mismatched = second.model_copy(
        update={
            "identity": second.identity.model_copy(
                update={"reset_id": "different-reset"}
            )
        }
    )
    with pytest.raises(ValueError, match="complete matched-reset policy panel"):
        aggregate_outcomes(
            (first, mismatched),
            target_label="skill_success",
            bootstrap_iterations=10,
        )


def test_controller_metrics_use_explicit_target_and_schema_cost_fields() -> None:
    source = _record(0, success=True)
    labels = source.labels.model_copy(
        update={
            "skill_success": source.labels.skill_success.model_copy(
                update={"value": False}
            )
        }
    )
    record = source.model_copy(
        update={
            "labels": labels,
            "costs": source.costs.model_copy(
                update={
                    "intervention_count": 0,
                    "system_analytic_time_s": 0.75,
                    "recovery_retry_cost": 2.0,
                }
            ),
        }
    )

    skill = evaluate_controller_records((record,), target_label="skill_success")
    primitive = evaluate_controller_records(
        (record,), target_label="primitive_success"
    )
    system = evaluate_system_records((record,))

    assert skill["vla_success_per_handoff"].value == 0.0
    assert skill["failed_vla_calls_per_success"].value is None
    assert primitive["vla_success_per_handoff"].value == 1.0
    assert primitive["failed_vla_calls_per_success"].value == 0.0
    assert skill["intervention_count"].value == 0.0
    assert system["analytic_time_s"].value == pytest.approx(0.75)
    assert system["recovery_retry_cost"].value == pytest.approx(2.0)


def test_cost_record_new_fields_are_optional_for_legacy_json() -> None:
    legacy = CostRecord.model_validate(
        {
            "analytic_steps": 1,
            "analytic_distance_m": 0.1,
            "analytic_time_s": 0.2,
            "vla_invocations": 1,
            "vla_time_s": 0.3,
            "total_elapsed_s": 0.5,
        }
    )

    assert legacy.system_analytic_time_s is None
    assert legacy.intervention_count is None
    assert legacy.recovery_retry_cost is None
    assert CostRecord.model_validate_json(legacy.canonical_json()) == legacy


def test_success_cost_plot_uses_selected_target_not_task_success(
    tmp_path,
    monkeypatch,
) -> None:
    source = _record(0, success=True)
    record = source.model_copy(
        update={
            "labels": source.labels.model_copy(
                update={
                    "skill_success": source.labels.skill_success.model_copy(
                        update={"value": False}
                    ),
                    "task_success": source.labels.task_success.model_copy(
                        update={"value": True}
                    ),
                }
            )
        }
    )
    result = aggregate_outcomes(
        (record,),
        target_label="skill_success",
        bootstrap_iterations=10,
    )

    class FakeFigure:
        def savefig(self, path, **_kwargs):
            Path(path).write_bytes(b"fixture plot")

    class FakeAxis:
        def __init__(self):
            self.points = []
            self.settings = {}

        def scatter(self, x, y, **_kwargs):
            self.points.append((tuple(x), tuple(y)))

        def annotate(self, *_args, **_kwargs):
            return None

        def set(self, **kwargs):
            self.settings.update(kwargs)

    axis = FakeAxis()

    class FakePyplot:
        @staticmethod
        def subplots(**_kwargs):
            return FakeFigure(), axis

        @staticmethod
        def close(_figure):
            return None

    monkeypatch.setattr(plotting, "_pyplot", lambda: FakePyplot())
    output = plot_method_success_cost(result, tmp_path / "success-cost.png")

    assert output.is_file()
    assert len(axis.points) == 1
    assert axis.points[0][0][0] == pytest.approx(0.3)
    assert axis.points[0][1] == (0.0,)
    assert axis.settings["ylabel"] == "skill_success rate"


def test_plotting_refuses_empty_and_synthetic_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="empty"):
        plot_gate0_landscape([], tmp_path / "empty.png")
    with pytest.raises(ValueError, match="non-observed"):
        plot_gate0_landscape(
            [
                {
                    "data_status": "synthetic",
                    "target_relative_x_m": 0.0,
                    "target_relative_y_m": 0.0,
                    "skill_success": True,
                }
            ],
            tmp_path / "fake.png",
        )


def test_posthoc_oracle_uses_only_unique_candidates_in_exact_matched_context() -> None:
    first = _record(0, success=True).model_copy(
        update={
            "metadata": {
                "data_status": "observed",
                "execution_layer": "gate0",
                "record_scope": "handoff_invocation",
            }
        }
    )
    second_source = _record(1, success=False)
    second = second_source.model_copy(
        update={
            "identity": second_source.identity.model_copy(
                update={
                    "run_id": first.identity.run_id,
                    "suite": first.identity.suite,
                    "task_id": first.identity.task_id,
                    "seed": first.identity.seed,
                    "reset_id": first.identity.reset_id,
                    "repeat_index": first.identity.repeat_index,
                    "candidate_id": "candidate-alternative",
                }
            ),
            "controller": first.controller,
            "skill": first.skill,
            "metadata": {
                "data_status": "observed",
                "execution_layer": "gate0",
                "record_scope": "handoff_invocation",
            },
        }
    )
    config = OracleCostConfig(
        target_label="skill_success",
        failure_penalty=100.0,
        analytic_distance_weight=1.0,
    )
    policy_source = _record(2, success=False)
    policy = policy_source.model_copy(
        update={
            "identity": policy_source.identity.model_copy(
                update={
                    "suite": first.identity.suite,
                    "task_id": first.identity.task_id,
                    "seed": first.identity.seed,
                    "reset_id": first.identity.reset_id,
                    "repeat_index": first.identity.repeat_index,
                }
            ),
            "skill": first.skill,
            "controller": policy_source.controller.model_copy(
                update={"checkpoint_id": first.controller.checkpoint_id}
            ),
            "metadata": {
                "data_status": "observed",
                "condition": "controlled-policy",
                "execution_layer": "controlled",
                "record_scope": "handoff_invocation",
            },
        }
    )

    annotated = annotate_matched_oracle_costs(
        (first, second),
        config,
        policy_records=(policy,),
    )

    assert annotated.matched_groups == 1
    assert annotated.annotated_records == 2
    assert annotated.annotated_policy_records == 1
    assert annotated.records[0].metadata["oracle_cost"] == 0.0
    assert annotated.records[1].metadata["chosen_cost"] > 100.0
    assert all(
        record.metadata["oracle_is_posthoc"] is True
        and record.metadata["oracle_policy_eligible"] is False
        for record in annotated.records
    )
    annotated_policy = annotated.policy_records[0]
    policy_regret = handoff_regret(
        [annotated_policy.metadata["chosen_cost"]],
        [annotated_policy.metadata["oracle_cost"]],
    )
    assert policy_regret.mean_regret.value is not None
    assert policy_regret.mean_regret.value > 100.0
    assert annotated_policy.metadata["oracle_candidate_set_source"] == (
        "matched_gate0_landscape"
    )

    duplicate_candidate = second.model_copy(
        update={
            "identity": second.identity.model_copy(
                update={"candidate_id": first.identity.candidate_id}
            )
        }
    )
    with pytest.raises(ValueError, match="duplicate candidate"):
        annotate_matched_oracle_costs((first, duplicate_candidate), config)

    wrong_cohort = second.model_copy(
        update={
            "metadata": {
                **second.metadata,
                "gate0_matched_cohort_id": "gate0-cohort-wrong",
                "gate0_candidate_id": second.identity.candidate_id,
            }
        }
    )
    with pytest.raises(ValueError, match="declared matched cohort"):
        annotate_matched_oracle_costs((first, wrong_cohort), config)
