"""Offline evaluation, statistics, aggregation, and plotting surfaces."""

from rpent.research.handoff.evaluation.aggregate import (
    AggregationResult,
    aggregate_outcomes,
    read_outcome_jsonl,
    write_aggregation,
)
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

__all__ = [
    "AggregationResult",
    "BinaryEvaluation",
    "BootstrapInterval",
    "MetricValue",
    "aggregate_outcomes",
    "evaluate_binary_predictions",
    "evaluate_controller_records",
    "evaluate_system_records",
    "grouped_bootstrap_interval",
    "handoff_regret",
    "read_outcome_jsonl",
    "write_aggregation",
]
