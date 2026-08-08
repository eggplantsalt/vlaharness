"""Grouped bootstrap utilities that preserve episode/reset correlation."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Hashable, Sequence
from statistics import mean, median
from typing import Literal, Self

from pydantic import Field, model_validator

from rpent.research.handoff.types import HandoffRecord


class BootstrapInterval(HandoffRecord):
    """Point estimate and percentile interval with explicit limitations."""

    statistic: Literal["mean", "median"]
    estimate: float | None
    lower: float | None
    upper: float | None
    confidence: float = Field(gt=0.0, lt=1.0)
    n_observations: int = Field(ge=0)
    n_groups: int = Field(ge=0)
    iterations: int = Field(ge=0)
    seed: int
    reason: str | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        values = (self.estimate, self.lower, self.upper)
        if any(value is None for value in values):
            if not all(value is None for value in values):
                raise ValueError("bootstrap estimate and bounds are jointly available")
            if not self.reason:
                raise ValueError("unavailable bootstrap interval needs a reason")
        else:
            assert self.estimate is not None
            assert self.lower is not None
            assert self.upper is not None
            if not all(math.isfinite(value) for value in values if value is not None):
                raise ValueError("bootstrap interval must be finite")
            if self.lower > self.upper:
                raise ValueError("bootstrap lower bound exceeds upper bound")
        return self


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile needs at least one value")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _statistic(values: Sequence[float], name: Literal["mean", "median"]) -> float:
    if name == "mean":
        return mean(values)
    return median(values)


def grouped_bootstrap_interval(
    values: Sequence[int | float | bool],
    groups: Sequence[Hashable],
    *,
    statistic: Literal["mean", "median"] = "mean",
    confidence: float = 0.95,
    iterations: int = 2000,
    seed: int = 0,
) -> BootstrapInterval:
    """Resample whole groups with replacement and compute a percentile CI."""
    if len(values) != len(groups):
        raise ValueError("values and groups must have equal length")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    normalized: list[float] = []
    grouped: dict[Hashable, list[float]] = defaultdict(list)
    for index, (value, group) in enumerate(zip(values, groups, strict=True)):
        if not isinstance(value, (int, float, bool)):
            raise ValueError(f"value {index} is not numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"value {index} is not finite")
        try:
            hash(group)
        except TypeError as exc:
            raise ValueError(f"group {index} is not hashable") from exc
        normalized.append(number)
        grouped[group].append(number)
    if not normalized:
        return BootstrapInterval(
            statistic=statistic,
            estimate=None,
            lower=None,
            upper=None,
            confidence=confidence,
            n_observations=0,
            n_groups=0,
            iterations=0,
            seed=seed,
            reason="no observations",
        )
    group_keys = sorted(grouped, key=lambda value: (type(value).__name__, repr(value)))
    estimate = _statistic(normalized, statistic)
    if len(group_keys) == 1:
        return BootstrapInterval(
            statistic=statistic,
            estimate=estimate,
            lower=estimate,
            upper=estimate,
            confidence=confidence,
            n_observations=len(normalized),
            n_groups=1,
            iterations=iterations,
            seed=seed,
            reason=(
                "only one independent group; interval has no between-group information"
            ),
        )
    generator = random.Random(seed)
    bootstrap_values: list[float] = []
    for _ in range(iterations):
        sample: list[float] = []
        for _group_index in range(len(group_keys)):
            sampled_key = group_keys[generator.randrange(len(group_keys))]
            sample.extend(grouped[sampled_key])
        bootstrap_values.append(_statistic(sample, statistic))
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        statistic=statistic,
        estimate=estimate,
        lower=_quantile(bootstrap_values, alpha),
        upper=_quantile(bootstrap_values, 1.0 - alpha),
        confidence=confidence,
        n_observations=len(normalized),
        n_groups=len(group_keys),
        iterations=iterations,
        seed=seed,
    )


__all__ = ["BootstrapInterval", "grouped_bootstrap_interval"]
