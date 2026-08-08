"""Dependency-light predictive, controller, system, and regret metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from statistics import mean, median
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.types import (
    FailureMode,
    GovernorState,
    HandoffRecord,
    OutcomeRecord,
    TerminationReason,
)

METRICS_SCHEMA_VERSION = "rpent.handoff-metrics/v1"


class MetricValue(HandoffRecord):
    """A scalar metric that is explicitly unavailable when undefined."""

    name: str
    value: float | None
    n: int = Field(ge=0)
    reason: str | None = None
    unit: str = "fraction"

    @field_validator("name", "unit")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.value is None and not self.reason:
            raise ValueError("unavailable metric needs a reason")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("metric value must be finite")
        return self


class CalibrationBin(HandoffRecord):
    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    count: int = Field(gt=0)
    mean_probability: float = Field(ge=0.0, le=1.0)
    empirical_success: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.upper <= self.lower:
            raise ValueError("calibration bin upper bound must exceed lower")
        return self


class CalibrationResult(HandoffRecord):
    bins: tuple[CalibrationBin, ...]
    expected_calibration_error: MetricValue


class RiskCoveragePoint(HandoffRecord):
    coverage: float = Field(gt=0.0, le=1.0)
    risk: float = Field(ge=0.0, le=1.0)
    uncertainty_threshold: float = Field(ge=0.0)
    retained: int = Field(gt=0)


class RiskCoverageResult(HandoffRecord):
    points: tuple[RiskCoveragePoint, ...]
    area_under_risk_coverage: MetricValue


class BinaryEvaluation(HandoffRecord):
    schema_version: Literal[METRICS_SCHEMA_VERSION] = METRICS_SCHEMA_VERSION
    accuracy: MetricValue
    auroc: MetricValue
    auprc: MetricValue
    brier: MetricValue
    log_loss: MetricValue
    calibration: CalibrationResult
    risk_coverage: RiskCoverageResult | None = None


class RegretResult(HandoffRecord):
    mean_regret: MetricValue
    median_regret: MetricValue
    regrets: tuple[float, ...]


def unavailable_metric(
    name: str,
    reason: str,
    *,
    n: int = 0,
    unit: str = "fraction",
) -> MetricValue:
    return MetricValue(name=name, value=None, n=n, reason=reason, unit=unit)


def _metric(name: str, value: float, n: int, *, unit: str = "fraction") -> MetricValue:
    return MetricValue(name=name, value=float(value), n=n, unit=unit)


def _normalize_binary_inputs(
    labels: Sequence[bool | int | float],
    probabilities: Sequence[int | float],
) -> tuple[list[int], list[float]]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have equal length")
    ys: list[int] = []
    ps: list[float] = []
    for index, (label, probability) in enumerate(zip(labels, probabilities, strict=True)):
        if isinstance(label, bool):
            y = int(label)
        elif isinstance(label, (int, float)) and float(label) in {0.0, 1.0}:
            y = int(label)
        else:
            raise ValueError(f"label {index} is not binary: {label!r}")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError(f"probability {index} is not numeric")
        p = float(probability)
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"probability {index} must be finite and in [0, 1]")
        ys.append(y)
        ps.append(p)
    return ys, ps


def accuracy_score(
    labels: Sequence[bool | int | float],
    probabilities: Sequence[int | float],
    *,
    threshold: float = 0.5,
) -> MetricValue:
    ys, ps = _normalize_binary_inputs(labels, probabilities)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not ys:
        return unavailable_metric("accuracy", "no labeled examples")
    correct = sum(int((p >= threshold) == bool(y)) for y, p in zip(ys, ps, strict=True))
    return _metric("accuracy", correct / len(ys), len(ys))


def roc_auc_score(
    labels: Sequence[bool | int | float],
    probabilities: Sequence[int | float],
) -> MetricValue:
    """Tie-aware Mann-Whitney AUROC without a scikit-learn dependency."""
    ys, ps = _normalize_binary_inputs(labels, probabilities)
    n = len(ys)
    if not n:
        return unavailable_metric("auroc", "no labeled examples")
    positives = sum(ys)
    negatives = n - positives
    if positives == 0 or negatives == 0:
        return unavailable_metric(
            "auroc",
            "AUROC is undefined for a single-class target",
            n=n,
        )
    ordered = sorted(enumerate(ps), key=lambda item: item[1])
    ranks = [0.0] * n
    cursor = 0
    while cursor < n:
        end = cursor + 1
        while end < n and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            ranks[ordered[position][0]] = average_rank
        cursor = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, ys, strict=True) if label)
    auc = (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
    return _metric("auroc", auc, n)


def average_precision_score(
    labels: Sequence[bool | int | float],
    probabilities: Sequence[int | float],
) -> MetricValue:
    """Threshold-grouped average precision, reported as AUPRC."""
    ys, ps = _normalize_binary_inputs(labels, probabilities)
    n = len(ys)
    positives = sum(ys)
    if not n:
        return unavailable_metric("auprc", "no labeled examples")
    if positives == 0:
        return unavailable_metric(
            "auprc",
            "AUPRC is undefined when the target has no positive examples",
            n=n,
        )
    ordered = sorted(zip(ps, ys, strict=True), key=lambda item: item[0], reverse=True)
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    cursor = 0
    while cursor < n:
        threshold = ordered[cursor][0]
        end = cursor
        group_positive = 0
        while end < n and ordered[end][0] == threshold:
            group_positive += ordered[end][1]
            end += 1
        group_size = end - cursor
        true_positive += group_positive
        false_positive += group_size - group_positive
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        cursor = end
    return _metric("auprc", area, n)


def brier_score(
    labels: Sequence[bool | int | float],
    probabilities: Sequence[int | float],
) -> MetricValue:
    ys, ps = _normalize_binary_inputs(labels, probabilities)
    if not ys:
        return unavailable_metric("brier", "no labeled examples", unit="score")
    value = mean((p - y) ** 2 for y, p in zip(ys, ps, strict=True))
    return _metric("brier", value, len(ys), unit="score")


def binary_log_loss(
    labels: Sequence[bool | int | float],
    probabilities: Sequence[int | float],
    *,
    epsilon: float = 1e-15,
) -> MetricValue:
    ys, ps = _normalize_binary_inputs(labels, probabilities)
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")
    if not ys:
        return unavailable_metric("log_loss", "no labeled examples", unit="nats")
    losses = []
    for y, p in zip(ys, ps, strict=True):
        clipped = min(max(p, epsilon), 1.0 - epsilon)
        losses.append(-(y * math.log(clipped) + (1 - y) * math.log(1.0 - clipped)))
    return _metric("log_loss", mean(losses), len(losses), unit="nats")


def calibration_curve(
    labels: Sequence[bool | int | float],
    probabilities: Sequence[int | float],
    *,
    n_bins: int = 10,
) -> CalibrationResult:
    ys, ps = _normalize_binary_inputs(labels, probabilities)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if not ys:
        return CalibrationResult(
            bins=(),
            expected_calibration_error=unavailable_metric(
                "ece", "no labeled examples"
            ),
        )
    grouped: list[list[tuple[int, float]]] = [[] for _ in range(n_bins)]
    for y, p in zip(ys, ps, strict=True):
        index = min(int(p * n_bins), n_bins - 1)
        grouped[index].append((y, p))
    bins: list[CalibrationBin] = []
    weighted_error = 0.0
    for index, values in enumerate(grouped):
        if not values:
            continue
        mean_probability = mean(item[1] for item in values)
        empirical_success = mean(item[0] for item in values)
        weighted_error += len(values) / len(ys) * abs(mean_probability - empirical_success)
        bins.append(
            CalibrationBin(
                lower=index / n_bins,
                upper=(index + 1) / n_bins,
                count=len(values),
                mean_probability=mean_probability,
                empirical_success=empirical_success,
            )
        )
    return CalibrationResult(
        bins=tuple(bins),
        expected_calibration_error=_metric("ece", weighted_error, len(ys)),
    )


def risk_coverage_curve(
    labels: Sequence[bool | int | float],
    probabilities: Sequence[int | float],
    uncertainties: Sequence[int | float],
    *,
    threshold: float = 0.5,
) -> RiskCoverageResult:
    ys, ps = _normalize_binary_inputs(labels, probabilities)
    if len(uncertainties) != len(ys):
        raise ValueError("uncertainties must match labels length")
    normalized_uncertainty: list[float] = []
    for index, uncertainty in enumerate(uncertainties):
        if isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float)):
            raise ValueError(f"uncertainty {index} is not numeric")
        value = float(uncertainty)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"uncertainty {index} must be finite and non-negative")
        normalized_uncertainty.append(value)
    if not ys:
        return RiskCoverageResult(
            points=(),
            area_under_risk_coverage=unavailable_metric(
                "aurc", "no labeled examples", unit="risk"
            ),
        )
    ordered = sorted(
        range(len(ys)),
        key=lambda index: (normalized_uncertainty[index], index),
    )
    points: list[RiskCoveragePoint] = []
    errors = 0
    retained = 0
    area = 0.0
    previous_coverage = 0.0
    cursor = 0
    while cursor < len(ordered):
        uncertainty = normalized_uncertainty[ordered[cursor]]
        end = cursor
        while (
            end < len(ordered)
            and normalized_uncertainty[ordered[end]] == uncertainty
        ):
            index = ordered[end]
            errors += int((ps[index] >= threshold) != bool(ys[index]))
            retained += 1
            end += 1
        coverage = retained / len(ys)
        risk = errors / retained
        points.append(
            RiskCoveragePoint(
                coverage=coverage,
                risk=risk,
                uncertainty_threshold=uncertainty,
                retained=retained,
            )
        )
        area += (coverage - previous_coverage) * risk
        previous_coverage = coverage
        cursor = end
    return RiskCoverageResult(
        points=tuple(points),
        area_under_risk_coverage=_metric("aurc", area, len(ys), unit="risk"),
    )


def evaluate_binary_predictions(
    labels: Sequence[bool | int | float],
    probabilities: Sequence[int | float],
    *,
    uncertainties: Sequence[int | float] | None = None,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> BinaryEvaluation:
    """Compute all required predictive metrics with safe edge-case behavior."""
    return BinaryEvaluation(
        accuracy=accuracy_score(labels, probabilities, threshold=threshold),
        auroc=roc_auc_score(labels, probabilities),
        auprc=average_precision_score(labels, probabilities),
        brier=brier_score(labels, probabilities),
        log_loss=binary_log_loss(labels, probabilities),
        calibration=calibration_curve(labels, probabilities, n_bins=n_bins),
        risk_coverage=(
            risk_coverage_curve(
                labels,
                probabilities,
                uncertainties,
                threshold=threshold,
            )
            if uncertainties is not None
            else None
        ),
    )


def _bool_rate(name: str, values: Iterable[bool | None]) -> MetricValue:
    available = [bool(value) for value in values if value is not None]
    if not available:
        return unavailable_metric(name, "telemetry/label unavailable")
    return _metric(name, mean(available), len(available))


def _numeric_summary(
    name: str,
    values: Iterable[int | float | None],
    *,
    unit: str,
    quantile: float | None = None,
) -> MetricValue:
    available = [float(value) for value in values if value is not None]
    if not available:
        return unavailable_metric(name, "telemetry unavailable", unit=unit)
    if any(not math.isfinite(value) for value in available):
        raise ValueError(f"non-finite telemetry for {name}")
    if quantile is None:
        value = mean(available)
    else:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        ordered = sorted(available)
        position = quantile * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            value = ordered[lower]
        else:
            fraction = position - lower
            value = ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    return _metric(name, value, len(available), unit=unit)


def _numeric_total(
    name: str,
    values: Iterable[int | float | None],
    *,
    unit: str,
) -> MetricValue:
    materialized = list(values)
    available = [float(value) for value in materialized if value is not None]
    if not materialized or len(available) != len(materialized):
        return unavailable_metric(
            name,
            "authoritative telemetry is unavailable for one or more records",
            n=len(available),
            unit=unit,
        )
    if any(not math.isfinite(value) for value in available):
        raise ValueError(f"non-finite telemetry for {name}")
    return _metric(name, sum(available), len(available), unit=unit)


def _ratio_metric(
    name: str,
    numerator: int,
    denominator: int,
    *,
    n: int,
    unit: str = "ratio",
) -> MetricValue:
    if denominator <= 0:
        return unavailable_metric(
            name,
            "denominator has no observed successes",
            n=n,
            unit=unit,
        )
    return _metric(name, numerator / denominator, n, unit=unit)


def evaluate_controller_records(
    records: Sequence[OutcomeRecord],
    *,
    target_label: str | None = None,
) -> dict[str, MetricValue]:
    """Aggregate controller-level outcomes without collapsing label semantics."""
    handoff_records = [record for record in records if record.handoff_occurred]
    target_values = (
        [record.labels.target_value(target_label) for record in handoff_records]
        if target_label is not None
        else []
    )
    if target_label is None:
        vla_success_metric = unavailable_metric(
            "vla_success_per_handoff",
            "an explicit outcome target label is required",
            n=len(handoff_records),
        )
        failed_per_success = unavailable_metric(
            "failed_vla_calls_per_success",
            "an explicit outcome target label is required",
            n=len(handoff_records),
        )
    else:
        target_successes = sum(value is True for value in target_values)
        failed_calls = sum(
            value is False
            or record.termination.failure_mode
            in {
                FailureMode.VLA,
                FailureMode.RPC,
                FailureMode.TIMEOUT,
                FailureMode.CANCELLATION,
            }
            for record, value in zip(
                handoff_records, target_values, strict=True
            )
        )
        vla_success_metric = _bool_rate(
            "vla_success_per_handoff", target_values
        )
        failed_per_success = _ratio_metric(
            "failed_vla_calls_per_success",
            failed_calls,
            target_successes,
            n=len(handoff_records),
        )
    return {
        "primitive_success_rate": _bool_rate(
            "primitive_success_rate",
            (record.labels.primitive_success.value for record in records),
        ),
        "skill_success_rate": _bool_rate(
            "skill_success_rate",
            (record.labels.skill_success.value for record in records),
        ),
        "task_success_rate": _bool_rate(
            "task_success_rate",
            (record.labels.task_success.value for record in records),
        ),
        "vla_success_per_handoff": vla_success_metric,
        "vla_failure_rate": _bool_rate(
            "vla_failure_rate",
            (
                record.termination.failure_mode is FailureMode.VLA
                for record in handoff_records
            ),
        ),
        "rpc_failure_rate": _bool_rate(
            "rpc_failure_rate",
            (
                record.termination.failure_mode is FailureMode.RPC
                for record in handoff_records
            ),
        ),
        "timeout_rate": _bool_rate(
            "timeout_rate",
            (
                record.termination.failure_mode is FailureMode.TIMEOUT
                for record in handoff_records
            ),
        ),
        "failed_vla_calls_per_success": failed_per_success,
        "staging_failure_rate": _bool_rate(
            "staging_failure_rate",
            (
                record.termination.failure_mode is FailureMode.STAGING
                for record in records
            ),
        ),
        "perception_failure_rate": _bool_rate(
            "perception_failure_rate",
            (
                record.termination.failure_mode is FailureMode.PERCEPTION
                for record in records
            ),
        ),
        "truncation_rate": _bool_rate(
            "truncation_rate",
            (record.termination.episode_truncated for record in records),
        ),
        "handoff_rate": _bool_rate(
            "handoff_rate", (record.handoff_occurred for record in records)
        ),
        "fallback_rate": _bool_rate(
            "fallback_rate",
            (
                record.termination.final_governor_state is GovernorState.FALLBACK
                or record.termination.reason
                in {
                    TerminationReason.FALLBACK_BASELINE,
                    TerminationReason.FALLBACK_FORCED_HANDOFF,
                }
                for record in records
            ),
        ),
        "abort_rate": _bool_rate(
            "abort_rate",
            (
                record.termination.final_governor_state is GovernorState.ABORT
                or record.termination.reason is TerminationReason.ABORTED
                for record in records
            ),
        ),
        "mean_vla_invocations": _numeric_summary(
            "mean_vla_invocations",
            (record.costs.vla_invocations for record in records),
            unit="count",
        ),
        "vla_invocation_count": _numeric_total(
            "vla_invocation_count",
            (record.costs.vla_invocations for record in records),
            unit="count",
        ),
        "mean_analytic_steps": _numeric_summary(
            "mean_analytic_steps",
            (record.costs.analytic_steps for record in records),
            unit="count",
        ),
        "mean_analytic_distance_m": _numeric_summary(
            "mean_analytic_distance_m",
            (record.costs.analytic_distance_m for record in records),
            unit="m",
        ),
        "mean_analytic_time_s": _numeric_summary(
            "mean_analytic_time_s",
            (record.costs.analytic_time_s for record in records),
            unit="s",
        ),
        "mean_vla_time_s": _numeric_summary(
            "mean_vla_time_s",
            (record.costs.vla_time_s for record in records),
            unit="s",
        ),
        "mean_total_env_actions": _numeric_summary(
            "mean_total_env_actions",
            (record.costs.total_env_actions for record in records),
            unit="count",
        ),
        "mean_total_elapsed_s": _numeric_summary(
            "mean_total_elapsed_s",
            (record.costs.total_elapsed_s for record in records),
            unit="s",
        ),
        "intervention_count": _numeric_total(
            "intervention_count",
            (record.costs.intervention_count for record in records),
            unit="count",
        ),
    }


def evaluate_system_records(
    records: Sequence[OutcomeRecord],
) -> dict[str, MetricValue]:
    """Aggregate full-system telemetry, preserving unavailable fields as null."""
    return {
        "task_success_rate": _bool_rate(
            "task_success_rate",
            (record.labels.task_success.value for record in records),
        ),
        "llm_turns_per_episode": _numeric_summary(
            "llm_turns_per_episode",
            (record.costs.llm_turns for record in records),
            unit="count",
        ),
        "input_tokens_per_episode": _numeric_summary(
            "input_tokens_per_episode",
            (record.costs.input_tokens for record in records),
            unit="tokens",
        ),
        "output_tokens_per_episode": _numeric_summary(
            "output_tokens_per_episode",
            (record.costs.output_tokens for record in records),
            unit="tokens",
        ),
        "planner_time_s": _numeric_summary(
            "planner_time_s",
            (record.costs.planner_time_s for record in records),
            unit="s",
        ),
        "analytic_time_s": _numeric_summary(
            "analytic_time_s",
            (record.costs.system_analytic_time_s for record in records),
            unit="s",
        ),
        "recovery_retry_cost": _numeric_summary(
            "recovery_retry_cost",
            (record.costs.recovery_retry_cost for record in records),
            unit="cost",
        ),
        "wall_clock_p50_s": _numeric_summary(
            "wall_clock_p50_s",
            (record.costs.total_elapsed_s for record in records),
            unit="s",
            quantile=0.50,
        ),
        "wall_clock_p95_s": _numeric_summary(
            "wall_clock_p95_s",
            (record.costs.total_elapsed_s for record in records),
            unit="s",
            quantile=0.95,
        ),
    }


def handoff_regret(
    chosen_costs: Sequence[int | float],
    oracle_costs: Sequence[int | float],
) -> RegretResult:
    """Compute matched-candidate regret J(chosen)-J(oracle)."""
    if len(chosen_costs) != len(oracle_costs):
        raise ValueError("chosen and oracle costs must be matched one-to-one")
    regrets: list[float] = []
    for index, (chosen, oracle) in enumerate(
        zip(chosen_costs, oracle_costs, strict=True)
    ):
        if isinstance(chosen, bool) or isinstance(oracle, bool):
            raise ValueError(f"regret costs at index {index} must be numeric")
        chosen_value = float(chosen)
        oracle_value = float(oracle)
        if not math.isfinite(chosen_value) or not math.isfinite(oracle_value):
            raise ValueError(f"regret costs at index {index} must be finite")
        regrets.append(chosen_value - oracle_value)
    if not regrets:
        unavailable = unavailable_metric(
            "mean_handoff_regret",
            "no matched candidate sets",
            unit="cost",
        )
        return RegretResult(
            mean_regret=unavailable,
            median_regret=unavailable_metric(
                "median_handoff_regret",
                "no matched candidate sets",
                unit="cost",
            ),
            regrets=(),
        )
    return RegretResult(
        mean_regret=_metric(
            "mean_handoff_regret", mean(regrets), len(regrets), unit="cost"
        ),
        median_regret=_metric(
            "median_handoff_regret", median(regrets), len(regrets), unit="cost"
        ),
        regrets=tuple(regrets),
    )


__all__ = [
    "BinaryEvaluation",
    "CalibrationBin",
    "CalibrationResult",
    "MetricValue",
    "RegretResult",
    "RiskCoveragePoint",
    "RiskCoverageResult",
    "accuracy_score",
    "average_precision_score",
    "binary_log_loss",
    "brier_score",
    "calibration_curve",
    "evaluate_binary_predictions",
    "evaluate_controller_records",
    "evaluate_system_records",
    "handoff_regret",
    "risk_coverage_curve",
    "roc_auc_score",
    "unavailable_metric",
]
