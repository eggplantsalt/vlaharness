"""Deterministic grid, random, and Latin-hypercube-like Gate-0 sampling."""

from __future__ import annotations

import hashlib
import itertools
import json
from enum import Enum
from typing import Literal, Self

import numpy as np
from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.types import HandoffRecord

GATE0_SAMPLE_SCHEMA_VERSION = "rpent.handoff-gate0-sample/v1"


class SamplingMode(str, Enum):
    GRID = "grid"
    RANDOM = "random"
    LATIN_HYPERCUBE = "latin_hypercube"


class SampleRange(HandoffRecord):
    minimum: float
    maximum: float
    grid_count: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.minimum > self.maximum:
            raise ValueError("sample range minimum exceeds maximum")
        if self.grid_count > 1 and self.minimum == self.maximum:
            raise ValueError("constant sample range must use grid_count=1")
        return self

    def grid(self) -> tuple[float, ...]:
        if self.grid_count == 1:
            return ((self.minimum + self.maximum) / 2.0,)
        return tuple(
            float(value)
            for value in np.linspace(self.minimum, self.maximum, self.grid_count)
        )


class Gate0SamplerConfig(HandoffRecord):
    mode: SamplingMode
    relative_x_m: SampleRange
    relative_y_m: SampleRange
    relative_z_m: SampleRange
    standoff_m: SampleRange = Field(
        default_factory=lambda: SampleRange(minimum=0.0, maximum=0.0)
    )
    wrist_yaw_rad: SampleRange = Field(
        default_factory=lambda: SampleRange(minimum=0.0, maximum=0.0)
    )
    wrist_pitch_rad: SampleRange = Field(
        default_factory=lambda: SampleRange(minimum=0.0, maximum=0.0)
    )
    approach_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0)
    random_samples: int = Field(default=64, ge=1)
    repeats: int = Field(default=1, ge=1)
    seed: int = 0
    maximum_total_trials: int = Field(default=100_000, ge=1)

    @field_validator("approach_axis_world")
    @classmethod
    def validate_axis(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if float(np.linalg.norm(value)) < 1e-12:
            raise ValueError("approach axis must have non-zero norm")
        return value


class Gate0Sample(HandoffRecord):
    schema_version: Literal[GATE0_SAMPLE_SCHEMA_VERSION] = GATE0_SAMPLE_SCHEMA_VERSION
    sample_id: str
    sample_index: int = Field(ge=0)
    repeat_index: int = Field(ge=0)
    relative_xyz_m: tuple[float, float, float]
    standoff_m: float = Field(ge=0.0)
    wrist_yaw_rad: float
    wrist_pitch_rad: float


def _sample_id(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "gate0-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _rows_grid(config: Gate0SamplerConfig) -> list[tuple[float, ...]]:
    dimensions = (
        config.relative_x_m.grid(),
        config.relative_y_m.grid(),
        config.relative_z_m.grid(),
        config.standoff_m.grid(),
        config.wrist_yaw_rad.grid(),
        config.wrist_pitch_rad.grid(),
    )
    return [tuple(float(value) for value in row) for row in itertools.product(*dimensions)]


def _ranges(config: Gate0SamplerConfig) -> tuple[SampleRange, ...]:
    return (
        config.relative_x_m,
        config.relative_y_m,
        config.relative_z_m,
        config.standoff_m,
        config.wrist_yaw_rad,
        config.wrist_pitch_rad,
    )


def _rows_random(config: Gate0SamplerConfig) -> list[tuple[float, ...]]:
    rng = np.random.default_rng(config.seed)
    rows = []
    for _ in range(config.random_samples):
        rows.append(
            tuple(
                float(rng.uniform(item.minimum, item.maximum))
                if item.minimum != item.maximum
                else float(item.minimum)
                for item in _ranges(config)
            )
        )
    return rows


def _rows_latin_hypercube(config: Gate0SamplerConfig) -> list[tuple[float, ...]]:
    rng = np.random.default_rng(config.seed)
    count = config.random_samples
    columns: list[np.ndarray] = []
    for item in _ranges(config):
        if item.minimum == item.maximum:
            columns.append(np.full(count, item.minimum, dtype=np.float64))
            continue
        strata = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        rng.shuffle(strata)
        columns.append(item.minimum + strata * (item.maximum - item.minimum))
    return [
        tuple(float(columns[column][row]) for column in range(len(columns)))
        for row in range(count)
    ]


def generate_gate0_samples(config: Gate0SamplerConfig) -> tuple[Gate0Sample, ...]:
    """Expand deterministic requested conditions and stable resumable IDs."""
    if config.mode is SamplingMode.GRID:
        rows = _rows_grid(config)
    elif config.mode is SamplingMode.RANDOM:
        rows = _rows_random(config)
    else:
        rows = _rows_latin_hypercube(config)
    total = len(rows) * config.repeats
    if total > config.maximum_total_trials:
        raise ValueError(
            f"sampler expands to {total} trials, exceeding "
            f"maximum_total_trials={config.maximum_total_trials}"
        )
    samples: list[Gate0Sample] = []
    sample_index = 0
    for repeat_index in range(config.repeats):
        for row in rows:
            x, y, z, standoff, yaw, pitch = row
            payload = {
                "sampler": config.canonical_json(),
                "row": row,
                "repeat_index": repeat_index,
            }
            samples.append(
                Gate0Sample(
                    sample_id=_sample_id(payload),
                    sample_index=sample_index,
                    repeat_index=repeat_index,
                    relative_xyz_m=(x, y, z),
                    standoff_m=standoff,
                    wrist_yaw_rad=yaw,
                    wrist_pitch_rad=pitch,
                )
            )
            sample_index += 1
    return tuple(samples)


def sample_world_position(
    sample: Gate0Sample,
    *,
    target_position_m: tuple[float, float, float],
    approach_axis_world: tuple[float, float, float],
) -> tuple[float, float, float]:
    axis = np.asarray(approach_axis_world, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    world = (
        np.asarray(target_position_m, dtype=np.float64)
        + np.asarray(sample.relative_xyz_m, dtype=np.float64)
        + axis * sample.standoff_m
    )
    return tuple(float(value) for value in world)  # type: ignore[return-value]

