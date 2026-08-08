"""Lazy, data-honest plotting for observed experiment artifacts."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rpent.research.handoff.evaluation.aggregate import AggregationResult


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plotting requires the optional matplotlib research dependency"
        ) from exc
    return plt


def _observed_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not rows:
        raise ValueError("refusing to plot empty input")
    invalid = [
        index
        for index, row in enumerate(rows)
        if row.get("data_status") != "observed"
        or bool(row.get("synthetic"))
        or bool(row.get("fake"))
        or bool(row.get("mock"))
    ]
    if invalid:
        raise ValueError(
            "refusing to plot non-observed/fake input rows: "
            f"{invalid[:10]}"
        )
    return list(rows)


def _finite(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"plot field {key!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"plot field {key!r} must be finite")
    return number


def _save_figure(figure, output_path: str | os.PathLike[str]) -> Path:
    destination = Path(output_path)
    if not destination.suffix:
        raise ValueError("plot output path needs a file extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.stem}.tmp{destination.suffix}"
    )
    figure.savefig(temporary, bbox_inches="tight", dpi=180)
    os.replace(temporary, destination)
    return destination


def _require_observed_aggregation(result: AggregationResult) -> None:
    if result.n_records <= 0:
        raise ValueError("refusing to plot an empty aggregation")
    if result.contains_non_observed_data:
        raise ValueError(
            "refusing to plot aggregation containing synthetic/mock/fake records"
        )


def plot_calibration_curve(
    result: AggregationResult,
    output_path: str | os.PathLike[str],
) -> Path:
    """Plot observed overall calibration against the identity line."""
    _require_observed_aggregation(result)
    predictive = result.overall.predictive_metrics
    if predictive is None or not predictive.calibration.bins:
        raise ValueError(
            "aggregation has no single-identity calibration data to plot; "
            "aggregate one condition/controller/checkpoint/skill/target at a time"
        )
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    bins = predictive.calibration.bins
    axis.plot(
        [record.mean_probability for record in bins],
        [record.empirical_success for record in bins],
        marker="o",
        label="observed",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="ideal")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted success", ylabel="Empirical success")
    axis.legend()
    try:
        return _save_figure(figure, output_path)
    finally:
        plt.close(figure)


def plot_method_success_cost(
    result: AggregationResult,
    output_path: str | os.PathLike[str],
) -> Path:
    """Plot the explicitly selected outcome target against observed time cost."""
    _require_observed_aggregation(result)
    metric_by_target = {
        "primitive_success": "primitive_success_rate",
        "skill_success": "skill_success_rate",
        "task_success": "task_success_rate",
    }
    metric_name = metric_by_target.get(str(result.target_label))
    if metric_name is None:
        raise ValueError(
            "success-cost plotting requires target_label to be one of "
            "primitive_success, skill_success, or task_success"
        )
    rows = []
    for group in result.per_method:
        method = str(group.group["method"])
        condition = str(group.group.get("condition", "unlabeled"))
        configuration = str(group.group.get("configuration_id", "unknown"))
        label = f"{condition} | {method} | {configuration[:10]}"
        success = group.controller_metrics[metric_name].value
        if group.group.get("execution_layer") == "full_agent":
            analytic_time = group.system_metrics["analytic_time_s"].value
        else:
            analytic_time = group.controller_metrics["mean_analytic_time_s"].value
        vla_time = group.controller_metrics["mean_vla_time_s"].value
        if success is None or analytic_time is None or vla_time is None:
            continue
        rows.append((label, analytic_time + vla_time, success))
    if not rows:
        raise ValueError("no method has jointly available success and cost")
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(6.2, 4.8))
    for label, cost, success in rows:
        axis.scatter([cost], [success], s=48)
        axis.annotate(label, (cost, success), xytext=(4, 4), textcoords="offset points")
    axis.set(
        xlabel="Mean analytic + VLA time (s)",
        ylabel=f"{result.target_label} rate",
        ylim=(0, 1),
    )
    try:
        return _save_figure(figure, output_path)
    finally:
        plt.close(figure)


def plot_handoff_regret(
    result: AggregationResult,
    output_path: str | os.PathLike[str],
) -> Path:
    """Plot mean matched-candidate regret for methods with oracle metadata."""
    _require_observed_aggregation(result)
    rows = [
        (
            f"{group.group.get('condition', 'unlabeled')} | "
            f"{group.group['method']} | "
            f"{str(group.group.get('configuration_id', 'unknown'))[:10]}",
            group.handoff_regret.value,
        )
        for group in result.per_method
        if group.handoff_regret.value is not None
    ]
    if not rows:
        raise ValueError("aggregation has no matched handoff regret data")
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    names = [name for name, _ in rows]
    values = [float(value) for _, value in rows if value is not None]
    axis.bar(names, values)
    axis.set(ylabel="Mean handoff regret (cost)")
    axis.tick_params(axis="x", rotation=30)
    try:
        return _save_figure(figure, output_path)
    finally:
        plt.close(figure)


def plot_gate0_landscape(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | os.PathLike[str],
    *,
    x_key: str = "target_relative_x_m",
    y_key: str = "target_relative_y_m",
    value_key: str = "skill_success",
) -> Path:
    """Scatter the measured Gate-0 competence landscape; no interpolation."""
    observed = _observed_rows(rows)
    points = []
    for row in observed:
        value = row.get(value_key)
        if value is None:
            continue
        if isinstance(value, bool):
            z = float(value)
        else:
            z = _finite(row, value_key)
        points.append((_finite(row, x_key), _finite(row, y_key), z))
    if not points:
        raise ValueError("Gate-0 rows contain no measured outcome points")
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(5.8, 5.0))
    artist = axis.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        c=[point[2] for point in points],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set(xlabel=x_key, ylabel=y_key)
    figure.colorbar(artist, ax=axis, label=value_key)
    try:
        return _save_figure(figure, output_path)
    finally:
        plt.close(figure)


def plot_ablation(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | os.PathLike[str],
    *,
    factor_key: str,
    value_key: str,
) -> Path:
    """Plot observed mean metric by representation/uncertainty/evidence factor."""
    observed = _observed_rows(rows)
    confounders = (
        "execution_layer",
        "record_scope",
        "method",
        "representation",
        "evidence_mode",
        "decision_mode",
        "uncertainty_mode",
        "hierarchy_mode",
    )
    mixed = {
        key: sorted({str(row.get(key)) for row in observed})
        for key in confounders
        if key != factor_key
        and len({str(row.get(key)) for row in observed}) > 1
    }
    if mixed:
        raise ValueError(
            "ablation input mixes uncontrolled factors; filter rows first: "
            f"{mixed}"
        )
    grouped: dict[str, list[float]] = {}
    for row in observed:
        factor = row.get(factor_key)
        value = row.get(value_key)
        if factor is None or value is None:
            continue
        grouped.setdefault(str(factor), []).append(
            float(value) if isinstance(value, bool) else _finite(row, value_key)
        )
    if not grouped:
        raise ValueError("ablation rows contain no complete factor/value pairs")
    names = sorted(grouped)
    values = [sum(grouped[name]) / len(grouped[name]) for name in names]
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    axis.bar(names, values)
    axis.set(xlabel=factor_key, ylabel=f"Mean {value_key}")
    axis.tick_params(axis="x", rotation=30)
    try:
        return _save_figure(figure, output_path)
    finally:
        plt.close(figure)


__all__ = [
    "plot_ablation",
    "plot_calibration_curve",
    "plot_gate0_landscape",
    "plot_handoff_regret",
    "plot_method_success_cost",
]
