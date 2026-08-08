"""Post-hoc matched-context oracle costs for Gate-0 outcome records.

The oracle is an evaluation upper bound, never a deployment policy. It selects
the lowest *realized* configured cost among candidates executed under the same
source-verified reset/repeat/controller context and annotates copies of the
observed records for regret analysis.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.types import HandoffRecord, OutcomeRecord

ORACLE_CONFIG_SCHEMA_VERSION = "rpent.handoff-oracle-cost/v1"
_LABELS = {
    "primitive_success",
    "skill_success",
    "task_success",
    "episode_truncated",
    "llm_finish",
}


class OracleCostConfig(HandoffRecord):
    """Explicit scalarization used only for post-hoc upper-bound evaluation."""

    schema_version: Literal[ORACLE_CONFIG_SCHEMA_VERSION] = (
        ORACLE_CONFIG_SCHEMA_VERSION
    )
    target_label: str = "primitive_success"
    failure_penalty: float = Field(default=100.0, gt=0.0)
    analytic_distance_weight: float = Field(default=1.0, ge=0.0)
    analytic_time_weight: float = Field(default=0.0, ge=0.0)
    analytic_step_weight: float = Field(default=0.0, ge=0.0)
    vla_time_weight: float = Field(default=0.0, ge=0.0)
    vla_invocation_weight: float = Field(default=0.0, ge=0.0)
    minimum_matched_candidates: int = Field(default=2, ge=2)

    @field_validator("target_label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if value not in _LABELS:
            raise ValueError(f"unsupported oracle target label: {value!r}")
        return value

    @model_validator(mode="after")
    def validate_cost_signal(self) -> Self:
        weights = (
            self.analytic_distance_weight,
            self.analytic_time_weight,
            self.analytic_step_weight,
            self.vla_time_weight,
            self.vla_invocation_weight,
        )
        if not any(weight > 0.0 for weight in weights):
            raise ValueError(
                "oracle cost needs at least one non-failure cost weight"
            )
        return self

    @property
    def configuration_id(self) -> str:
        return "oracle-" + hashlib.sha256(
            self.canonical_json().encode("utf-8")
        ).hexdigest()[:20]


@dataclass(frozen=True, slots=True)
class OracleAnnotationResult:
    records: tuple[OutcomeRecord, ...]
    policy_records: tuple[OutcomeRecord, ...]
    matched_groups: int
    annotated_records: int
    annotated_policy_records: int
    eligible_unmatched_records: int
    eligible_unmatched_policy_records: int
    ineligible_records: int
    ineligible_policy_records: int
    configuration_id: str


def _match_key(record: OutcomeRecord) -> tuple[str, ...]:
    return (
        _matched_cohort_id(record),
        record.skill.name,
        record.controller.method,
        record.controller.implementation_version,
        str(record.controller.checkpoint_id),
        str(record.controller.configuration_id),
    )


def _matched_cohort_id(record: OutcomeRecord) -> str:
    """Validate or derive the exact pinned reset/repeat cohort identity."""
    identity = record.identity
    if identity.reset_id is None:
        raise ValueError("matched Gate-0 oracle record lacks reset identity")
    payload = {
        "schema_version": "rpent.handoff-gate0-matched-cohort/v1",
        "run_id": identity.run_id,
        "suite": identity.suite,
        "task_id": identity.task_id,
        "seed": identity.seed,
        "reset_id": identity.reset_id,
        "repeat_index": identity.repeat_index,
        "skill": record.skill.model_dump(mode="json", exclude_none=False),
        "source_revision": record.source_revision,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    derived = "gate0-cohort-" + hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()[:20]
    declared = record.metadata.get("gate0_matched_cohort_id")
    if declared is not None and declared != derived:
        raise ValueError(
            "Gate-0 declared matched cohort does not bind its reset/repeat context"
        )
    return derived


def _candidate_id(record: OutcomeRecord) -> str:
    candidate = record.identity.candidate_id
    if candidate is None:
        raise ValueError("matched Gate-0 oracle record lacks candidate identity")
    declared = record.metadata.get("gate0_candidate_id")
    if declared is not None and declared != candidate:
        raise ValueError(
            "Gate-0 declared candidate identity disagrees with TrialIdentity"
        )
    return candidate


def _policy_context_key(record: OutcomeRecord) -> tuple[str, ...]:
    """Match a policy rollout to a Gate-0 landscape without requiring run IDs."""
    identity = record.identity
    return (
        identity.suite,
        str(identity.task_id),
        str(identity.seed),
        str(identity.reset_id),
        str(identity.repeat_index),
        record.skill.name,
        str(record.controller.checkpoint_id),
    )


def _group_id(key: tuple[str, ...]) -> str:
    canonical = json.dumps(
        key,
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return "oracle-group-" + hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()[:20]


def _eligible(record: OutcomeRecord, config: OracleCostConfig) -> bool:
    return (
        record.metadata.get("execution_layer") == "gate0"
        and record.metadata.get("record_scope", "handoff_invocation")
        == "handoff_invocation"
        and record.handoff_occurred
        and record.identity.reset_id is not None
        and record.identity.candidate_id is not None
        and record.costs.vla_invocations is not None
        and record.costs.vla_invocations > 0
        and record.labels.target_value(config.target_label) is not None
        and _cost_telemetry_available(record, config)
    )


def _policy_eligible(record: OutcomeRecord, config: OracleCostConfig) -> bool:
    return (
        record.metadata.get("execution_layer") == "controlled"
        and record.metadata.get("record_scope", "handoff_invocation")
        == "handoff_invocation"
        and record.handoff_occurred
        and record.identity.reset_id is not None
        and record.costs.vla_invocations is not None
        and record.costs.vla_invocations > 0
        and record.labels.target_value(config.target_label) is not None
        and _cost_telemetry_available(record, config)
    )


def _cost_telemetry_available(
    record: OutcomeRecord,
    config: OracleCostConfig,
) -> bool:
    required = (
        (config.analytic_distance_weight, record.costs.analytic_distance_m),
        (config.analytic_time_weight, record.costs.analytic_time_s),
        (config.analytic_step_weight, record.costs.analytic_steps),
        (config.vla_time_weight, record.costs.vla_time_s),
        (config.vla_invocation_weight, record.costs.vla_invocations),
    )
    return all(weight == 0.0 or value is not None for weight, value in required)


def _realized_cost(record: OutcomeRecord, config: OracleCostConfig) -> float:
    success = record.labels.target_value(config.target_label)
    assert success is not None
    invocations = record.costs.vla_invocations
    assert invocations is not None
    analytic_distance = record.costs.analytic_distance_m
    analytic_time = record.costs.analytic_time_s
    analytic_steps = record.costs.analytic_steps
    vla_time = record.costs.vla_time_s
    if not _cost_telemetry_available(record, config):
        raise ValueError(
            f"record {record.record_id} lacks weighted oracle-cost telemetry"
        )
    return float(
        (0.0 if success else config.failure_penalty)
        + config.analytic_distance_weight * float(analytic_distance or 0.0)
        + config.analytic_time_weight * float(analytic_time or 0.0)
        + config.analytic_step_weight * float(analytic_steps or 0)
        + config.vla_time_weight * float(vla_time or 0.0)
        + config.vla_invocation_weight * invocations
    )


def annotate_matched_oracle_costs(
    records: Sequence[OutcomeRecord],
    config: OracleCostConfig,
    *,
    policy_records: Sequence[OutcomeRecord] = (),
) -> OracleAnnotationResult:
    """Annotate Gate-0 and matched controlled choices without online leakage."""
    if not records:
        raise ValueError("oracle annotation requires non-empty outcomes")
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("oracle input contains duplicate record IDs")
    policy_record_ids = [record.record_id for record in policy_records]
    if len(policy_record_ids) != len(set(policy_record_ids)):
        raise ValueError("oracle policy input contains duplicate record IDs")
    overlap = sorted(set(record_ids).intersection(policy_record_ids))
    if overlap:
        raise ValueError(f"landscape/policy oracle inputs overlap: {overlap[:10]}")

    grouped: dict[tuple[str, ...], list[OutcomeRecord]] = defaultdict(list)
    ineligible = 0
    for record in records:
        if _eligible(record, config):
            grouped[_match_key(record)].append(record)
        else:
            ineligible += 1

    annotations: dict[str, dict[str, object]] = {}
    context_groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    unmatched = 0
    matched_groups = 0
    for key, group in sorted(grouped.items()):
        candidate_ids = [_candidate_id(record) for record in group]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(
                "matched oracle context contains duplicate candidate identities: "
                f"{_group_id(key)}"
            )
        if len(group) < config.minimum_matched_candidates:
            unmatched += len(group)
            continue
        matched_groups += 1
        costs = {record.record_id: _realized_cost(record, config) for record in group}
        oracle_cost = min(costs.values())
        winners = sorted(
            _candidate_id(record)
            for record in group
            if costs[record.record_id] == oracle_cost
        )
        group_id = _group_id(key)
        context_key = _policy_context_key(group[0])
        context_groups[context_key].append(
            {
                "oracle_cost": oracle_cost,
                "oracle_candidate_id": winners[0],
                "oracle_tied_candidate_ids": winners,
                "oracle_group_id": group_id,
                "oracle_matched_cohort_id": _matched_cohort_id(group[0]),
                "oracle_group_size": len(group),
            }
        )
        for record in group:
            annotations[record.record_id] = {
                "chosen_cost": costs[record.record_id],
                "oracle_cost": oracle_cost,
                "oracle_candidate_id": winners[0],
                "oracle_tied_candidate_ids": winners,
                "oracle_group_id": group_id,
                "oracle_matched_cohort_id": _matched_cohort_id(record),
                "oracle_group_size": len(group),
                "oracle_target_label": config.target_label,
                "oracle_cost_configuration_id": config.configuration_id,
                "oracle_is_posthoc": True,
                "oracle_policy_eligible": False,
            }

    if not annotations:
        raise ValueError(
            "no matched Gate-0 context contains the configured minimum number "
            "of eligible candidates"
        )
    output = tuple(
        record.model_copy(
            update={"metadata": {**record.metadata, **annotations[record.record_id]}}
        )
        if record.record_id in annotations
        else record
        for record in records
    )
    ambiguous_contexts = {
        key: [str(item["oracle_group_id"]) for item in groups]
        for key, groups in context_groups.items()
        if len(groups) != 1
    }
    if ambiguous_contexts:
        raise ValueError(
            "multiple Gate-0 candidate sets match one policy context: "
            + json.dumps(
                {"|".join(key): value for key, value in ambiguous_contexts.items()},
                sort_keys=True,
            )
        )
    policy_annotations: dict[str, dict[str, object]] = {}
    unmatched_policy = 0
    ineligible_policy = 0
    for record in policy_records:
        if not _policy_eligible(record, config):
            ineligible_policy += 1
            continue
        groups = context_groups.get(_policy_context_key(record), [])
        if not groups:
            unmatched_policy += 1
            continue
        oracle_group = groups[0]
        policy_annotations[record.record_id] = {
            "chosen_cost": _realized_cost(record, config),
            **oracle_group,
            "oracle_target_label": config.target_label,
            "oracle_cost_configuration_id": config.configuration_id,
            "oracle_is_posthoc": True,
            "oracle_policy_eligible": False,
            "oracle_candidate_set_source": "matched_gate0_landscape",
        }
    annotated_policy = tuple(
        record.model_copy(
            update={
                "metadata": {
                    **record.metadata,
                    **policy_annotations[record.record_id],
                }
            }
        )
        if record.record_id in policy_annotations
        else record
        for record in policy_records
    )
    return OracleAnnotationResult(
        records=output,
        policy_records=annotated_policy,
        matched_groups=matched_groups,
        annotated_records=len(annotations),
        annotated_policy_records=len(policy_annotations),
        eligible_unmatched_records=unmatched,
        eligible_unmatched_policy_records=unmatched_policy,
        ineligible_records=ineligible,
        ineligible_policy_records=ineligible_policy,
        configuration_id=config.configuration_id,
    )


__all__ = [
    "ORACLE_CONFIG_SCHEMA_VERSION",
    "OracleAnnotationResult",
    "OracleCostConfig",
    "annotate_matched_oracle_costs",
]
