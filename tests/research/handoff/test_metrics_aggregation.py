from __future__ import annotations

import json

import pytest

from rpent.research.handoff.evaluation.aggregate import (
    aggregate_outcomes,
    read_outcome_jsonl,
    write_aggregation,
)
from rpent.research.handoff.evaluation.metrics import (
    evaluate_binary_predictions,
    handoff_regret,
)
from rpent.research.handoff.evaluation.plotting import plot_gate0_landscape
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
    records = (_record(0, success=True), _record(1, success=False))
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
    assert result.overall.controller_metrics["skill_success_rate"].value == 0.5
    assert result.overall.handoff_regret.value == 0.5
    assert (tmp_path / "aggregate" / "results.csv").is_file()
    summary = json.loads((tmp_path / "aggregate" / "summary.json").read_text())
    assert summary["n_records"] == 2
    assert artifacts.calibration_csv is None


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
