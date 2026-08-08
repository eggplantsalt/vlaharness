"""Strict outcome loading and paper-oriented CSV/JSON aggregation."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from rpent.research.handoff.evaluation.metrics import (
    BinaryEvaluation,
    MetricValue,
    evaluate_binary_predictions,
    evaluate_controller_records,
    evaluate_system_records,
    handoff_regret,
)
from rpent.research.handoff.evaluation.statistics import (
    BootstrapInterval,
    grouped_bootstrap_interval,
)
from rpent.research.handoff.types import HandoffRecord, OutcomeRecord

AGGREGATION_SCHEMA_VERSION = "rpent.handoff-aggregation/v1"


class AggregateGroup(HandoffRecord):
    """Metrics for one deterministic grouping of outcome records."""

    group: dict[str, str | int]
    n_records: int = Field(gt=0)
    controller_metrics: dict[str, MetricValue]
    system_metrics: dict[str, MetricValue]
    target_success_interval: BootstrapInterval
    predictive_metrics: BinaryEvaluation | None = None
    predictive_unavailable_reason: str | None = None
    handoff_regret: MetricValue
    failure_counts: dict[str, int]

    @field_validator("failure_counts")
    @classmethod
    def validate_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("failure counts must be non-negative")
        return value


class AggregationResult(HandoffRecord):
    schema_version: Literal[AGGREGATION_SCHEMA_VERSION] = AGGREGATION_SCHEMA_VERSION
    n_records: int = Field(gt=0)
    target_label: str | None = None
    contains_non_observed_data: bool
    data_status_counts: dict[str, int]
    overall: AggregateGroup
    per_method: tuple[AggregateGroup, ...]
    per_task: tuple[AggregateGroup, ...]
    per_method_task: tuple[AggregateGroup, ...]


class AggregationArtifacts(HandoffRecord):
    output_dir: str
    tidy_csv: str
    summary_json: str
    per_method_csv: str
    per_task_csv: str
    failure_breakdown_csv: str
    calibration_csv: str | None = None


def _strict_json_line(raw: str, *, source: Path, line_number: int) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"invalid outcome JSON in {source} at line {line_number}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"outcome line {line_number} in {source} is not an object")
    return value


def read_outcome_jsonl(
    paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> tuple[OutcomeRecord, ...]:
    """Read canonical OutcomeRecords and reject blank/corrupt/duplicate rows."""
    sources = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    if not sources:
        raise ValueError("at least one outcome JSONL path is required")
    records: list[OutcomeRecord] = []
    seen: set[str] = set()
    for source_value in sources:
        source = Path(source_value)
        with source.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line:
                    raise ValueError(
                        f"blank outcome line in {source} at line {line_number}"
                    )
                payload = _strict_json_line(
                    line,
                    source=source,
                    line_number=line_number,
                )
                try:
                    record = OutcomeRecord.model_validate(payload)
                except Exception as exc:
                    raise ValueError(
                        f"invalid OutcomeRecord in {source} at line {line_number}: {exc}"
                    ) from exc
                if record.record_id in seen:
                    raise ValueError(f"duplicate outcome record ID: {record.record_id}")
                seen.add(record.record_id)
                records.append(record)
    if not records:
        raise ValueError("outcome dataset is empty")
    return tuple(records)


def _data_status(record: OutcomeRecord) -> str:
    explicit = record.metadata.get("data_status")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit:
            raise ValueError(f"invalid data_status in {record.record_id}")
        return explicit
    if any(bool(record.metadata.get(key)) for key in ("synthetic", "fake", "mock")):
        return "synthetic"
    return "observed"


def _selected_estimate(record: OutcomeRecord):
    if record.metadata.get("record_scope") == "full_agent_episode":
        # An episode may contain several handoff decisions; its final selected
        # estimate is not a single prediction for the episode-level label.
        return None
    if not record.decision_trace:
        return None
    decision = record.decision_trace[-1]
    selected = [candidate for candidate in decision.candidates if candidate.selected]
    return selected[0].estimate if len(selected) == 1 else None


def _finite_metadata_number(record: OutcomeRecord, key: str) -> float | None:
    value = record.metadata.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metadata {key!r} in {record.record_id} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"metadata {key!r} in {record.record_id} must be finite")
    return number


def outcome_to_tidy_row(record: OutcomeRecord) -> dict[str, Any]:
    """Flatten authoritative fields while retaining distinct outcome labels."""
    state = record.pre_handoff_state
    target_position = (
        state.target.estimate.position_m
        if state is not None and state.target is not None
        else None
    )
    selected_estimate = _selected_estimate(record)
    condition = record.metadata.get(
        "condition", record.metadata.get("condition_name")
    )
    execution_layer = record.metadata.get("execution_layer", "unspecified")
    record_scope = record.metadata.get("record_scope", "handoff_invocation")
    row: dict[str, Any] = {
        "record_id": record.record_id,
        "run_id": record.identity.run_id,
        "episode_id": record.identity.episode_id,
        "trial_id": record.identity.trial_id,
        "invocation_id": record.identity.invocation_id,
        "candidate_id": record.identity.candidate_id,
        "suite": record.identity.suite,
        "task": record.identity.task_id,
        "seed": record.identity.seed,
        "reset_id": record.identity.reset_id,
        "repeat_index": record.identity.repeat_index,
        "method": record.controller.method,
        "implementation_version": record.controller.implementation_version,
        "configuration_id": record.controller.configuration_id,
        "condition": condition,
        "execution_layer": execution_layer,
        "record_scope": record_scope,
        "protocol_adherent": record.metadata.get("protocol_adherent"),
        "checkpoint_id": record.controller.checkpoint_id,
        "skill": record.skill.name,
        "semantic_target": record.skill.semantic_target,
        "handoff_occurred": record.handoff_occurred,
        "primitive_success": record.labels.primitive_success.value,
        "skill_success": record.labels.skill_success.value,
        "task_success": record.labels.task_success.value,
        "episode_truncated": record.labels.episode_truncated.value,
        "llm_finish": record.labels.llm_finish.value,
        "termination_reason": record.termination.reason.value,
        "failure_mode": record.termination.failure_mode.value,
        "final_governor_state": record.termination.final_governor_state.value,
        "analytic_steps": record.costs.analytic_steps,
        "analytic_distance_m": record.costs.analytic_distance_m,
        "analytic_time_s": record.costs.analytic_time_s,
        "vla_invocations": record.costs.vla_invocations,
        "vla_chunks": record.costs.vla_chunks,
        "vla_env_actions": record.costs.vla_env_actions,
        "vla_time_s": record.costs.vla_time_s,
        "total_env_actions": record.costs.total_env_actions,
        "total_elapsed_s": record.costs.total_elapsed_s,
        "planner_time_s": record.costs.planner_time_s,
        "llm_turns": record.costs.llm_turns,
        "input_tokens": record.costs.input_tokens,
        "output_tokens": record.costs.output_tokens,
        "system_analytic_time_s": record.costs.system_analytic_time_s,
        "intervention_count": record.costs.intervention_count,
        "recovery_retry_cost": record.costs.recovery_retry_cost,
        "eef_x_m": state.eef_position_m[0] if state is not None else None,
        "eef_y_m": state.eef_position_m[1] if state is not None else None,
        "eef_z_m": state.eef_position_m[2] if state is not None else None,
        "target_x_m": target_position[0] if target_position is not None else None,
        "target_y_m": target_position[1] if target_position is not None else None,
        "target_z_m": target_position[2] if target_position is not None else None,
        "predicted_success_probability": (
            selected_estimate.mean_success_probability
            if selected_estimate is not None
            else None
        ),
        "conservative_success_probability": (
            selected_estimate.conservative_success_probability
            if selected_estimate is not None
            else None
        ),
        "epistemic_uncertainty": (
            selected_estimate.epistemic_std if selected_estimate is not None else None
        ),
        "chosen_cost": _finite_metadata_number(record, "chosen_cost"),
        "oracle_cost": _finite_metadata_number(record, "oracle_cost"),
        "representation": record.metadata.get("representation"),
        "evidence_mode": record.metadata.get("evidence_mode"),
        "decision_mode": record.metadata.get("decision_mode"),
        "uncertainty_mode": record.metadata.get("uncertainty_mode"),
        "hierarchy_mode": record.metadata.get("hierarchy_mode"),
        "data_status": _data_status(record),
    }
    if state is not None and target_position is not None:
        row.update(
            {
                "target_relative_x_m": state.eef_position_m[0] - target_position[0],
                "target_relative_y_m": state.eef_position_m[1] - target_position[1],
                "target_relative_z_m": state.eef_position_m[2] - target_position[2],
            }
        )
    else:
        row.update(
            {
                "target_relative_x_m": None,
                "target_relative_y_m": None,
                "target_relative_z_m": None,
            }
        )
    return row


def _predictive_metrics(
    records: Sequence[OutcomeRecord],
    target_label: str | None,
) -> tuple[BinaryEvaluation | None, str | None]:
    if target_label is None:
        return None, "target label was not explicitly selected"
    predictive_identities = {
        (
            str(
                record.metadata.get(
                    "condition", record.metadata.get("condition_name")
                )
            ),
            record.controller.configuration_id,
            str(record.controller.checkpoint_id),
            record.skill.name,
            target_label,
        )
        for record in records
        if _selected_estimate(record) is not None
    }
    if len(predictive_identities) > 1:
        return None, (
            "predictive/calibration metrics refuse pooled condition, controller, "
            "checkpoint, skill, or target identities; aggregate one predictive "
            "identity at a time"
        )
    labels: list[bool] = []
    probabilities: list[float] = []
    uncertainties: list[float] = []
    missing = 0
    for record in records:
        label = record.labels.target_value(target_label)
        estimate = _selected_estimate(record)
        if label is None or estimate is None:
            missing += 1
            continue
        labels.append(label)
        probabilities.append(estimate.mean_success_probability)
        uncertainties.append(estimate.epistemic_std)
    if not labels:
        return None, (
            f"no records jointly contain label {target_label!r} and a selected estimate"
        )
    evaluation = evaluate_binary_predictions(
        labels,
        probabilities,
        uncertainties=uncertainties,
    )
    reason = f"excluded {missing} records with unavailable label/estimate" if missing else None
    return evaluation, reason


def _success_interval(
    records: Sequence[OutcomeRecord],
    *,
    target_label: str | None,
    iterations: int,
    seed: int,
) -> BootstrapInterval:
    values: list[bool] = []
    groups: list[str] = []
    selected_label = target_label or "task_success"
    for record in records:
        value = record.labels.target_value(selected_label)
        if value is None:
            continue
        values.append(value)
        groups.append(
            "|".join(
                (
                    record.identity.suite,
                    str(record.identity.task_id),
                    record.identity.reset_id
                    or record.identity.episode_id
                    or record.identity.trial_id,
                )
            )
        )
    return grouped_bootstrap_interval(
        values,
        groups,
        iterations=iterations,
        seed=seed,
    )


def _aggregate_group(
    records: Sequence[OutcomeRecord],
    group: dict[str, str | int],
    *,
    target_label: str | None,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> AggregateGroup:
    predictive, predictive_reason = _predictive_metrics(records, target_label)
    chosen: list[float] = []
    oracle: list[float] = []
    for record in records:
        chosen_value = _finite_metadata_number(record, "chosen_cost")
        oracle_value = _finite_metadata_number(record, "oracle_cost")
        if chosen_value is None or oracle_value is None:
            continue
        chosen.append(chosen_value)
        oracle.append(oracle_value)
    regret = handoff_regret(chosen, oracle).mean_regret
    return AggregateGroup(
        group=group,
        n_records=len(records),
        controller_metrics=evaluate_controller_records(
            records,
            target_label=target_label,
        ),
        system_metrics=evaluate_system_records(records),
        target_success_interval=_success_interval(
            records,
            target_label=target_label,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
        predictive_metrics=predictive,
        predictive_unavailable_reason=predictive_reason,
        handoff_regret=regret,
        failure_counts=dict(
            sorted(Counter(record.termination.failure_mode.value for record in records).items())
        ),
    )


def _group_records(
    records: Sequence[OutcomeRecord],
    keys: tuple[str, ...],
) -> list[tuple[dict[str, str | int], list[OutcomeRecord]]]:
    grouped: dict[tuple[str | int, ...], list[OutcomeRecord]] = defaultdict(list)
    for record in records:
        values: list[str | int] = []
        for key in keys:
            if key == "method":
                values.append(record.controller.method)
            elif key == "configuration_id":
                values.append(record.controller.configuration_id)
            elif key == "execution_layer":
                value = record.metadata.get("execution_layer", "unspecified")
                values.append(str(value))
            elif key == "record_scope":
                value = record.metadata.get(
                    "record_scope", "handoff_invocation"
                )
                values.append(str(value))
            elif key == "condition":
                value = record.metadata.get(
                    "condition", record.metadata.get("condition_name")
                )
                values.append(
                    str(value)
                    if value is not None
                    else f"unlabeled:{record.controller.configuration_id}"
                )
            elif key == "suite":
                values.append(record.identity.suite)
            elif key == "task":
                values.append(record.identity.task_id)
            else:
                raise ValueError(f"unsupported aggregate group key: {key}")
        grouped[tuple(values)].append(record)
    result = []
    for values in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        result.append((dict(zip(keys, values, strict=True)), grouped[values]))
    return result


def _verify_controlled_reset_pairing(records: Sequence[OutcomeRecord]) -> None:
    """Fail closed unless each policy sees the same realized reset per context."""
    conditions = {
        str(
            record.metadata.get(
                "condition", record.metadata.get("condition_name")
            )
        )
        for record in records
    }
    if len(conditions) <= 1:
        return
    paired: dict[
        tuple[str, str, int, int],
        dict[str, list[OutcomeRecord]],
    ] = defaultdict(lambda: defaultdict(list))
    for record in records:
        context = (
            record.identity.suite,
            str(record.identity.task_id),
            record.identity.seed,
            record.identity.repeat_index,
        )
        condition = str(
            record.metadata.get(
                "condition", record.metadata.get("condition_name")
            )
        )
        paired[context][condition].append(record)
    violations: dict[str, Any] = {}
    for context, by_condition in paired.items():
        counts = {name: len(values) for name, values in by_condition.items()}
        reset_ids = {
            record.identity.reset_id
            for values in by_condition.values()
            for record in values
        }
        if (
            set(by_condition) != conditions
            or any(count != 1 for count in counts.values())
            or None in reset_ids
            or len(reset_ids) != 1
        ):
            violations["|".join(map(str, context))] = {
                "expected_conditions": sorted(conditions),
                "observed_counts": dict(sorted(counts.items())),
                "reset_ids": sorted(str(value) for value in reset_ids),
            }
    if violations:
        raise ValueError(
            "controlled comparison is not a complete matched-reset policy panel: "
            + json.dumps(violations, sort_keys=True)
        )


def aggregate_outcomes(
    records: Sequence[OutcomeRecord],
    *,
    target_label: str | None,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 0,
) -> AggregationResult:
    """Aggregate real or explicitly marked non-observed OutcomeRecords."""
    if not records:
        raise ValueError("refusing to aggregate an empty outcome set")
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("outcome records contain duplicate record IDs")
    statuses = Counter(_data_status(record) for record in records)
    layers = {
        str(record.metadata.get("execution_layer", "unspecified"))
        for record in records
    }
    scopes = {
        str(record.metadata.get("record_scope", "handoff_invocation"))
        for record in records
    }
    if len(layers) != 1 or len(scopes) != 1:
        raise ValueError(
            "aggregation refuses mixed execution layers or record scopes; "
            f"layers={sorted(layers)}, scopes={sorted(scopes)}"
        )
    if layers == {"controlled"}:
        _verify_controlled_reset_pairing(records)
    protocol_violations = [
        record.record_id
        for record in records
        if record.metadata.get("protocol_adherent") is False
    ]
    if protocol_violations:
        raise ValueError(
            "per-protocol aggregation refuses protocol violations; summarize "
            "them separately for intention-to-treat analysis: "
            + ", ".join(protocol_violations[:10])
        )

    def aggregate_many(keys: tuple[str, ...]) -> tuple[AggregateGroup, ...]:
        return tuple(
            _aggregate_group(
                grouped_records,
                group,
                target_label=target_label,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
            )
            for group, grouped_records in _group_records(records, keys)
        )

    return AggregationResult(
        n_records=len(records),
        target_label=target_label,
        contains_non_observed_data=any(status != "observed" for status in statuses),
        data_status_counts=dict(sorted(statuses.items())),
        overall=_aggregate_group(
            records,
            {"scope": "pooled_inventory_not_method_comparison"},
            target_label=target_label,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        ),
        per_method=aggregate_many(
            (
                "execution_layer",
                "record_scope",
                "condition",
                "method",
                "configuration_id",
            )
        ),
        per_task=aggregate_many(
            (
                "execution_layer",
                "record_scope",
                "suite",
                "task",
            )
        ),
        per_method_task=aggregate_many(
            (
                "execution_layer",
                "record_scope",
                "condition",
                "method",
                "configuration_id",
                "suite",
                "task",
            )
        ),
    )


def _csv_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    fieldnames = sorted({key for row in rows for key in row})
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_scalar(row.get(key)) for key in fieldnames})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def _atomic_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _aggregate_group_row(group: AggregateGroup) -> dict[str, Any]:
    row: dict[str, Any] = {**group.group, "n_records": group.n_records}
    for prefix, metrics in (
        ("controller", group.controller_metrics),
        ("system", group.system_metrics),
    ):
        for name, metric in metrics.items():
            row[f"{prefix}.{name}"] = metric.value
            row[f"{prefix}.{name}.reason"] = metric.reason
    row.update(
        {
            "target_success_ci.estimate": group.target_success_interval.estimate,
            "target_success_ci.lower": group.target_success_interval.lower,
            "target_success_ci.upper": group.target_success_interval.upper,
            "target_success_ci.n_groups": group.target_success_interval.n_groups,
            "handoff_regret": group.handoff_regret.value,
            "handoff_regret.reason": group.handoff_regret.reason,
        }
    )
    return row


def write_aggregation(
    result: AggregationResult,
    records: Sequence[OutcomeRecord],
    output_dir: str | os.PathLike[str],
) -> AggregationArtifacts:
    """Write tidy data and machine-readable summaries; never invent rows."""
    if not records or len(records) != result.n_records:
        raise ValueError("aggregation result and source records disagree")
    destination = Path(output_dir)
    tidy_rows = [outcome_to_tidy_row(record) for record in records]
    tidy_path = _atomic_csv(destination / "results.csv", tidy_rows)
    summary_path = _atomic_json(
        destination / "summary.json",
        result.model_dump(mode="json", exclude_none=False),
    )
    per_method_path = _atomic_csv(
        destination / "per_method.csv",
        [_aggregate_group_row(group) for group in result.per_method],
    )
    per_task_path = _atomic_csv(
        destination / "per_task.csv",
        [_aggregate_group_row(group) for group in result.per_task],
    )
    failure_rows = [
        {
            **group.group,
            "failure_mode": failure_mode,
            "count": count,
            "n_records": group.n_records,
        }
        for group in result.per_method_task
        for failure_mode, count in group.failure_counts.items()
    ]
    failure_path = _atomic_csv(
        destination / "failure_breakdown.csv",
        failure_rows,
    )
    calibration_path: Path | None = None
    predictive = result.overall.predictive_metrics
    if predictive is not None and predictive.calibration.bins:
        calibration_path = _atomic_csv(
            destination / "calibration.csv",
            [bin_record.model_dump(mode="json") for bin_record in predictive.calibration.bins],
        )
    return AggregationArtifacts(
        output_dir=str(destination.resolve()),
        tidy_csv=str(tidy_path.resolve()),
        summary_json=str(summary_path.resolve()),
        per_method_csv=str(per_method_path.resolve()),
        per_task_csv=str(per_task_path.resolve()),
        failure_breakdown_csv=str(failure_path.resolve()),
        calibration_csv=(str(calibration_path.resolve()) if calibration_path else None),
    )


def load_aggregate_write(
    paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    *,
    target_label: str | None,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 0,
) -> tuple[AggregationResult, AggregationArtifacts]:
    records = read_outcome_jsonl(paths)
    result = aggregate_outcomes(
        records,
        target_label=target_label,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    return result, write_aggregation(result, records, output_dir)


__all__ = [
    "AggregateGroup",
    "AggregationArtifacts",
    "AggregationResult",
    "aggregate_outcomes",
    "load_aggregate_write",
    "outcome_to_tidy_row",
    "read_outcome_jsonl",
    "write_aggregation",
]
