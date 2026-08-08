"""Object-relative handoff candidate generation and honest feature prediction."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rpent.research.handoff.types import (
    CandidateGeometry,
    FeatureAvailability,
    FeatureProvenance,
    HandoffState,
)


class CandidateGenerationError(ValueError):
    """Raised when candidates cannot be constructed from observed state."""


class CandidateGeneratorConfig(BaseModel):
    """Configurable object-relative approach/standoff candidate lattice."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    approach_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0)
    standoff_distances_m: tuple[float, ...] = (0.04, 0.08, 0.12)
    xyz_perturbations_m: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)
    wrist_yaw_offsets_rad: tuple[float, ...] = (0.0,)
    wrist_pitch_offsets_rad: tuple[float, ...] = (0.0,)
    workspace_min_m: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    workspace_max_m: tuple[float, float, float] = (1.0, 1.0, 2.0)
    max_candidates: int = Field(default=128, ge=1)

    @field_validator("standoff_distances_m")
    @classmethod
    def validate_standoffs(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(distance < 0.0 for distance in value):
            raise ValueError("standoff distances must be a non-empty non-negative list")
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> "CandidateGeneratorConfig":
        if any(low >= high for low, high in zip(self.workspace_min_m, self.workspace_max_m)):
            raise ValueError("workspace bounds must be strictly increasing")
        norm = math.sqrt(sum(value * value for value in self.approach_axis_world))
        if norm < 1e-12:
            raise ValueError("approach axis must have non-zero norm")
        if not self.xyz_perturbations_m:
            raise ValueError("at least one XYZ perturbation is required")
        if not self.wrist_yaw_offsets_rad or not self.wrist_pitch_offsets_rad:
            raise ValueError("orientation offset sets must be non-empty")
        requested = (
            len(self.standoff_distances_m)
            * len(self.xyz_perturbations_m)
            * len(self.wrist_yaw_offsets_rad)
            * len(self.wrist_pitch_offsets_rad)
        )
        if requested > self.max_candidates:
            raise ValueError(
                f"candidate lattice has {requested} future candidates, exceeding "
                f"max_candidates={self.max_candidates}"
            )
        return self


def _rotation_matrix_from_quaternion(q: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in q)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def wrist_yaw_pitch(q: tuple[float, float, float, float]) -> tuple[float, float]:
    """Match RPent's world-yaw and gripper-down pitch definitions."""
    matrix = _rotation_matrix_from_quaternion(q)
    yaw = float(math.atan2(matrix[1, 0], matrix[0, 0]))
    pitch = float(math.atan2(matrix[1, 2], -matrix[2, 2]))
    return yaw, pitch


def _quat_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    result = (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )
    norm = math.sqrt(sum(value * value for value in result))
    return tuple(float(value / norm) for value in result)  # type: ignore[return-value]


def _orientation_delta(
    yaw_delta: float, pitch_delta: float
) -> tuple[float, float, float, float]:
    yaw_half = yaw_delta / 2.0
    pitch_half = pitch_delta / 2.0
    yaw_q = (0.0, 0.0, math.sin(yaw_half), math.cos(yaw_half))
    pitch_q = (math.sin(pitch_half), 0.0, 0.0, math.cos(pitch_half))
    return _quat_multiply(yaw_q, pitch_q)


def _candidate_id(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "candidate-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class ObjectRelativeCandidateGenerator:
    """Generate candidate 0 plus bounded target-relative future states."""

    def __init__(self, config: CandidateGeneratorConfig) -> None:
        self.config = config

    def generate(self, state: HandoffState) -> tuple[CandidateGeometry, ...]:
        current_yaw, current_pitch = wrist_yaw_pitch(state.eef_quaternion_xyzw)
        current_target_relative = None
        if state.target is not None and state.target.estimate.position_m is not None:
            current_target_relative = tuple(
                float(current - target)
                for current, target in zip(
                    state.eef_position_m, state.target.estimate.position_m
                )
            )
        current = CandidateGeometry(
            candidate_id="candidate-current",
            kind="current",
            eef_position_m=state.eef_position_m,
            target_relative_position_m=current_target_relative,
            wrist_yaw_rad=current_yaw,
            wrist_pitch_rad=current_pitch,
        )
        if state.target is None or state.target.estimate.position_m is None:
            # Candidate 0 is always exact and is sufficient for a direct
            # handoff policy. Policies that require future/object-relative
            # candidates fail explicitly at their own decision boundary.
            return (current,)

        target = np.asarray(state.target.estimate.position_m, dtype=np.float64)
        axis = np.asarray(self.config.approach_axis_world, dtype=np.float64)
        axis /= np.linalg.norm(axis)
        low = np.asarray(self.config.workspace_min_m, dtype=np.float64)
        high = np.asarray(self.config.workspace_max_m, dtype=np.float64)
        future: list[CandidateGeometry] = []
        for standoff in self.config.standoff_distances_m:
            for perturbation in self.config.xyz_perturbations_m:
                relative = axis * float(standoff) + np.asarray(
                    perturbation, dtype=np.float64
                )
                position = target + relative
                if np.any(position < low) or np.any(position > high):
                    continue
                for yaw_offset in self.config.wrist_yaw_offsets_rad:
                    for pitch_offset in self.config.wrist_pitch_offsets_rad:
                        payload = {
                            "position": [float(value) for value in position],
                            "yaw_offset": float(yaw_offset),
                            "pitch_offset": float(pitch_offset),
                            "standoff": float(standoff),
                        }
                        future.append(
                            CandidateGeometry(
                                candidate_id=_candidate_id(payload),
                                kind=(
                                    "standoff"
                                    if all(abs(value) < 1e-12 for value in perturbation)
                                    else "perturbation"
                                ),
                                eef_position_m=tuple(
                                    float(value) for value in position
                                ),
                                target_relative_position_m=tuple(
                                    float(value) for value in relative
                                ),
                                wrist_yaw_rad=current_yaw + float(yaw_offset),
                                wrist_pitch_rad=current_pitch + float(pitch_offset),
                                wrist_yaw_delta_rad=float(yaw_offset),
                                wrist_pitch_delta_rad=float(pitch_offset),
                                requested_standoff_m=float(standoff),
                            )
                        )
        if not future:
            raise CandidateGenerationError(
                "all configured future candidates lie outside workspace bounds"
            )
        return (current, *future)


class CandidateFeaturePredictorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    visual_prediction: Literal["hold_current", "mark_unavailable"] = "hold_current"


class CandidateFeaturePredictor:
    """Predict only defensible candidate fields and label every approximation."""

    def __init__(self, config: CandidateFeaturePredictorConfig) -> None:
        self.config = config

    def predict(self, current: HandoffState, candidate: CandidateGeometry) -> HandoffState:
        if candidate.kind == "current":
            return current
        orientation = _quat_multiply(
            _orientation_delta(
                candidate.wrist_yaw_delta_rad,
                candidate.wrist_pitch_delta_rad,
            ),
            current.eef_quaternion_xyzw,
        )
        target = current.target
        if target is not None and self.config.visual_prediction == "mark_unavailable":
            estimate = target.estimate.model_copy(
                update={"visual_geometry": None}
            )
            target = target.model_copy(update={"estimate": estimate})

        replaced_names = {"eef_position_m", "eef_quaternion_xyzw"}
        provenance = tuple(
            item
            for item in current.provenance
            if item.feature_name not in replaced_names
        ) + (
            FeatureProvenance(
                feature_name="eef_position_m",
                availability=FeatureAvailability.DERIVED_DEPLOYMENT,
                source="candidate_feature_predictor",
                unit="m",
                frame="world",
                derivation="configured target-relative candidate geometry",
                provider_version="v1",
            ),
            FeatureProvenance(
                feature_name="eef_quaternion_xyzw",
                availability=FeatureAvailability.DERIVED_DEPLOYMENT,
                source="candidate_feature_predictor",
                unit="unit_quaternion",
                frame="world",
                derivation="bounded configured yaw/pitch delta",
                provider_version="v1",
            ),
        )
        return current.model_copy(
            update={
                "state_id": f"predicted:{current.state_id}:{candidate.candidate_id}",
                "eef_position_m": candidate.eef_position_m,
                "eef_quaternion_xyzw": orientation,
                "target": target,
                "provenance": provenance,
            }
        )
