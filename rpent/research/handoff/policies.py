"""Unified handoff policies, baselines, and risk-aware reference method."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rpent.research.handoff.candidates import CandidateFeaturePredictor
from rpent.research.handoff.model import OutcomeModel
from rpent.research.handoff.types import (
    CandidateDecisionRecord,
    CandidateGeometry,
    HandoffAction,
    HandoffDecision,
    HandoffState,
    OutcomeEstimate,
)


class PolicyError(ValueError):
    """Raised when a policy cannot make a valid bounded decision."""


class FeatureBuilderLike(Protocol):
    def build(self, state: HandoffState):
        """Build a deployment-safe feature vector."""


@dataclass(frozen=True)
class PolicyContext:
    current_state: HandoffState
    candidates: tuple[CandidateGeometry, ...]
    decision_sequence: int
    previous_action: HandoffAction | None = None

    def __post_init__(self) -> None:
        if not self.candidates:
            raise PolicyError("policy context needs at least candidate 0")
        first = self.candidates[0]
        if first.kind != "current" or first.candidate_id != "candidate-current":
            raise PolicyError("candidate 0 must be the observed current state")
        if first.eef_position_m != self.current_state.eef_position_m:
            raise PolicyError("candidate 0 position must equal the observed state")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise PolicyError("candidate IDs must be unique")


@runtime_checkable
class HandoffPolicy(Protocol):
    name: str

    def decide(self, context: PolicyContext) -> HandoffDecision:
        """Choose between current handoff and one future staging candidate."""


def _decision_id(
    policy_name: str, state_id: str, sequence: int, selected_id: str
) -> str:
    payload = json.dumps(
        [policy_name, state_id, sequence, selected_id], separators=(",", ":")
    )
    return "decision-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


def _orientation_cost(candidate: CandidateGeometry) -> float:
    return abs(candidate.wrist_yaw_delta_rad) + abs(candidate.wrist_pitch_delta_rad)


def _record(
    context: PolicyContext,
    candidate: CandidateGeometry,
    *,
    selected: bool,
    estimate: OutcomeEstimate | None = None,
    handoff_cost: float = 0.0,
    staging_cost: float = 0.0,
    total_cost: float = 0.0,
    feasible: bool = True,
    infeasible_reason: str | None = None,
) -> CandidateDecisionRecord:
    return CandidateDecisionRecord(
        decision_sequence=context.decision_sequence,
        candidate=candidate,
        estimate=estimate,
        handoff_cost=max(0.0, float(handoff_cost)),
        staging_cost=max(0.0, float(staging_cost)),
        total_cost=max(0.0, float(total_cost)),
        selected=selected,
        feasible=feasible,
        infeasible_reason=infeasible_reason,
    )


def _decision(
    policy_name: str,
    context: PolicyContext,
    selected_index: int,
    records: Sequence[CandidateDecisionRecord],
    rationale: str,
    *,
    force_action: HandoffAction | None = None,
) -> HandoffDecision:
    if selected_index < 0 or selected_index >= len(context.candidates):
        raise PolicyError("selected candidate index is outside candidate set")
    selected_id = context.candidates[selected_index].candidate_id
    normalized = tuple(
        item.model_copy(update={"selected": index == selected_index})
        for index, item in enumerate(records)
    )
    action = force_action or (
        HandoffAction.HANDOFF_NOW if selected_index == 0 else HandoffAction.CONTINUE
    )
    return HandoffDecision(
        decision_id=_decision_id(
            policy_name,
            context.current_state.state_id,
            context.decision_sequence,
            selected_id,
        ),
        state_id=context.current_state.state_id,
        decision_sequence=context.decision_sequence,
        action=action,
        selected_candidate_id=selected_id,
        candidates=normalized,
        rationale=rationale,
        policy_name=policy_name,
    )


class DirectHandoffPolicy:
    """Direct frozen Pi0.5 baseline with no analytic staging."""

    name = "direct_frozen_pi0"
    requires_target = False

    def decide(self, context: PolicyContext) -> HandoffDecision:
        records = [
            _record(context, candidate, selected=index == 0)
            for index, candidate in enumerate(context.candidates)
        ]
        return _decision(
            self.name,
            context,
            0,
            records,
            "direct baseline hands off at the observed current state",
        )


class FixedCanonicalPolicy:
    """Stage toward a fixed target-relative canonical pre-contact geometry."""

    name = "fixed_canonical_precontact"

    def __init__(
        self,
        target_relative_position_m: tuple[float, float, float],
        *,
        tolerance_m: float = 0.015,
    ) -> None:
        if tolerance_m <= 0.0:
            raise ValueError("canonical tolerance must be positive")
        self.target_relative_position_m = target_relative_position_m
        self.tolerance_m = tolerance_m

    def decide(self, context: PolicyContext) -> HandoffDecision:
        candidates = context.candidates
        distances = [
            math.inf
            if candidate.target_relative_position_m is None
            else _distance(
                candidate.target_relative_position_m,
                self.target_relative_position_m,
            )
            for candidate in candidates
        ]
        if not math.isfinite(distances[0]):
            raise PolicyError("canonical baseline requires target-relative state")
        if distances[0] <= self.tolerance_m:
            selected = 0
        elif len(distances) < 2:
            raise PolicyError("canonical target is not reached and no future candidate exists")
        else:
            selected = int(np.argmin(distances[1:])) + 1
        records = [
            _record(
                context,
                candidate,
                selected=index == selected,
                staging_cost=0.0 if index == 0 else distance,
                total_cost=distance,
                feasible=math.isfinite(distance),
                infeasible_reason=None if math.isfinite(distance) else "missing target-relative geometry",
            )
            for index, (candidate, distance) in enumerate(zip(candidates, distances))
        ]
        return _decision(
            self.name,
            context,
            selected,
            records,
            "handoff inside canonical tolerance; otherwise stage toward nearest canonical candidate",
        )


class FixedDistancePolicy:
    """Hand off at a configured target standoff distance."""

    name = "fixed_distance"

    def __init__(self, distance_m: float, *, tolerance_m: float = 0.015) -> None:
        if distance_m < 0.0 or tolerance_m <= 0.0:
            raise ValueError("fixed distance and tolerance must be valid")
        self.distance_m = distance_m
        self.tolerance_m = tolerance_m

    @staticmethod
    def _standoff(candidate: CandidateGeometry) -> float:
        if candidate.target_relative_position_m is None:
            return math.inf
        return float(np.linalg.norm(candidate.target_relative_position_m))

    def decide(self, context: PolicyContext) -> HandoffDecision:
        errors = [abs(self._standoff(candidate) - self.distance_m) for candidate in context.candidates]
        if not math.isfinite(errors[0]):
            raise PolicyError("fixed-distance baseline requires target-relative geometry")
        if errors[0] <= self.tolerance_m:
            selected = 0
        elif len(errors) < 2:
            raise PolicyError("fixed distance is not reached and no future candidate exists")
        else:
            selected = int(np.argmin(errors[1:])) + 1
        records = [
            _record(
                context,
                candidate,
                selected=index == selected,
                staging_cost=0.0 if index == 0 else error,
                total_cost=error,
                feasible=math.isfinite(error),
                infeasible_reason=None if math.isfinite(error) else "missing target-relative geometry",
            )
            for index, (candidate, error) in enumerate(zip(context.candidates, errors))
        ]
        return _decision(
            self.name,
            context,
            selected,
            records,
            "handoff when measured standoff is within the fixed-distance tolerance",
        )


class PositiveReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    reference_id: str
    target_relative_position_m: tuple[float, float, float]
    wrist_yaw_rad: float | None = None
    wrist_pitch_rad: float | None = None


def _reference_distance(candidate: CandidateGeometry, reference: PositiveReference) -> float:
    if candidate.target_relative_position_m is None:
        return math.inf
    distance = _distance(
        candidate.target_relative_position_m, reference.target_relative_position_m
    )
    if candidate.wrist_yaw_rad is not None and reference.wrist_yaw_rad is not None:
        distance += 0.05 * abs(candidate.wrist_yaw_rad - reference.wrist_yaw_rad)
    if candidate.wrist_pitch_rad is not None and reference.wrist_pitch_rad is not None:
        distance += 0.05 * abs(candidate.wrist_pitch_rad - reference.wrist_pitch_rad)
    return distance


class PositiveNearestSuccessPolicy:
    """Positive-only nearest-success retrieval baseline."""

    name = "positive_nearest_success"

    def __init__(
        self, references: Sequence[PositiveReference], *, handoff_radius: float = 0.03
    ) -> None:
        if not references:
            raise ValueError("positive retrieval requires at least one success")
        if handoff_radius <= 0.0:
            raise ValueError("handoff radius must be positive")
        self.references = tuple(references)
        self.handoff_radius = handoff_radius

    def decide(self, context: PolicyContext) -> HandoffDecision:
        distances = [
            min(_reference_distance(candidate, reference) for reference in self.references)
            for candidate in context.candidates
        ]
        if distances[0] <= self.handoff_radius:
            selected = 0
        elif len(distances) < 2:
            raise PolicyError("retrieval radius is not reached and no future candidate exists")
        else:
            selected = int(np.argmin(distances[1:])) + 1
        records = [
            _record(
                context,
                candidate,
                selected=index == selected,
                staging_cost=0.0 if index == 0 else distance,
                total_cost=distance,
                feasible=math.isfinite(distance),
                infeasible_reason=None if math.isfinite(distance) else "missing target-relative geometry",
            )
            for index, (candidate, distance) in enumerate(zip(context.candidates, distances))
        ]
        return _decision(
            self.name,
            context,
            selected,
            records,
            "positive-only nearest-success retrieval with a fixed support radius",
        )


class _ModelPolicy:
    def __init__(
        self,
        model: OutcomeModel,
        feature_builder: FeatureBuilderLike,
        predictor: CandidateFeaturePredictor,
    ) -> None:
        self.model = model
        self.feature_builder = feature_builder
        self.predictor = predictor

    def _estimates(self, context: PolicyContext) -> list[OutcomeEstimate]:
        estimates: list[OutcomeEstimate] = []
        for candidate in context.candidates:
            predicted = self.predictor.predict(context.current_state, candidate)
            features = self.feature_builder.build(predicted)
            estimate = self.model.predict_one(features)
            estimates.append(estimate)
        return estimates


class CompetenceThresholdPolicy(_ModelPolicy):
    """Probability-threshold competence baseline without optimal stopping."""

    name = "competence_probability_threshold"

    def __init__(self, *args, threshold: float = 0.7, conservative: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("competence threshold must be in [0, 1]")
        self.threshold = threshold
        self.conservative = conservative

    def decide(self, context: PolicyContext) -> HandoffDecision:
        estimates = self._estimates(context)
        scores = [
            estimate.conservative_success_probability
            if self.conservative
            else estimate.mean_success_probability
            for estimate in estimates
        ]
        if scores[0] >= self.threshold:
            selected = 0
        elif len(scores) < 2:
            raise PolicyError("competence threshold is unmet and no future candidate exists")
        else:
            selected = int(np.argmax(scores[1:])) + 1
        records = [
            _record(
                context,
                candidate,
                selected=index == selected,
                estimate=estimate,
                total_cost=1.0 - score,
            )
            for index, (candidate, estimate, score) in enumerate(
                zip(context.candidates, estimates, scores)
            )
        ]
        return _decision(
            self.name,
            context,
            selected,
            records,
            "handoff when current competence crosses a fixed threshold; otherwise seek maximum predicted competence",
        )


class CompetenceProjectionPolicy(_ModelPolicy):
    """Static competence-region projection baseline."""

    name = "competence_projection"

    def __init__(self, *args, threshold: float = 0.7, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("competence threshold must be in [0, 1]")
        self.threshold = threshold

    def decide(self, context: PolicyContext) -> HandoffDecision:
        estimates = self._estimates(context)
        scores = [item.mean_success_probability for item in estimates]
        if scores[0] >= self.threshold:
            selected = 0
        else:
            if len(scores) < 2:
                raise PolicyError(
                    "competence region is unmet and no future candidate exists"
                )
            eligible = [
                index
                for index, score in enumerate(scores[1:], start=1)
                if score >= self.threshold
            ]
            selected = min(
                eligible,
                key=lambda index: _distance(
                    context.current_state.eef_position_m,
                    context.candidates[index].eef_position_m,
                ),
                default=int(np.argmax(scores[1:])) + 1,
            )
        records = []
        for index, (candidate, estimate) in enumerate(zip(context.candidates, estimates)):
            stage_distance = (
                0.0
                if index == 0
                else _distance(
                    context.current_state.eef_position_m, candidate.eef_position_m
                )
            )
            records.append(
                _record(
                    context,
                    candidate,
                    selected=index == selected,
                    estimate=estimate,
                    staging_cost=stage_distance,
                    total_cost=stage_distance,
                )
            )
        return _decision(
            self.name,
            context,
            selected,
            records,
            "project to the nearest static competence-threshold candidate",
        )


class PositiveSupportRegionPolicy:
    """Positive-only ellipsoidal support/bridge-style baseline."""

    name = "positive_support_region"

    def __init__(
        self,
        references: Sequence[PositiveReference],
        *,
        bandwidth_m: tuple[float, float, float] = (0.04, 0.04, 0.04),
    ) -> None:
        if not references:
            raise ValueError("support region requires positive references")
        if any(value <= 0.0 for value in bandwidth_m):
            raise ValueError("support bandwidths must be positive")
        self.references = tuple(references)
        self.bandwidth = np.asarray(bandwidth_m, dtype=float)

    def _support_distance(self, candidate: CandidateGeometry) -> float:
        if candidate.target_relative_position_m is None:
            return math.inf
        point = np.asarray(candidate.target_relative_position_m, dtype=float)
        return min(
            float(
                np.linalg.norm(
                    (point - np.asarray(reference.target_relative_position_m))
                    / self.bandwidth
                )
            )
            for reference in self.references
        )

    def decide(self, context: PolicyContext) -> HandoffDecision:
        distances = [self._support_distance(candidate) for candidate in context.candidates]
        if distances[0] <= 1.0:
            selected = 0
        elif len(distances) < 2:
            raise PolicyError("support region is not reached and no future candidate exists")
        else:
            selected = int(np.argmin(distances[1:])) + 1
        records = [
            _record(
                context,
                candidate,
                selected=index == selected,
                staging_cost=0.0 if index == 0 else distance,
                total_cost=distance,
                feasible=math.isfinite(distance),
                infeasible_reason=None if math.isfinite(distance) else "missing target-relative geometry",
            )
            for index, (candidate, distance) in enumerate(zip(context.candidates, distances))
        ]
        return _decision(
            self.name,
            context,
            selected,
            records,
            "positive-only static support-region membership/projection",
        )


class RiskAwareSwitchingConfig(BaseModel):
    """Working-hypothesis cost configuration; every term is experiment-owned."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    vla_cost: float = Field(default=0.1, ge=0.0)
    failure_cost: float = Field(default=1.0, ge=0.0)
    stage_fixed_cost: float = Field(default=0.0, ge=0.0)
    stage_cost_per_m: float = Field(default=1.0, ge=0.0)
    orientation_cost_per_rad: float = Field(default=0.0, ge=0.0)
    hysteresis_cost: float = Field(default=0.0, ge=0.0)
    probability_mode: str = "conservative"


class OutcomeCalibratedSwitchingPolicy(_ModelPolicy):
    """Risk-aware receding-horizon comparison of handoff now vs continue."""

    name = "outcome_calibrated_switching"

    def __init__(self, *args, config: RiskAwareSwitchingConfig, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if config.probability_mode not in {"mean", "conservative"}:
            raise ValueError("probability_mode must be mean or conservative")
        self.config = config

    def decide(self, context: PolicyContext) -> HandoffDecision:
        estimates = self._estimates(context)
        records: list[CandidateDecisionRecord] = []
        totals: list[float] = []
        for index, (candidate, estimate) in enumerate(zip(context.candidates, estimates)):
            probability = (
                estimate.mean_success_probability
                if self.config.probability_mode == "mean"
                else estimate.conservative_success_probability
            )
            handoff_cost = self.config.vla_cost + self.config.failure_cost * (
                1.0 - probability
            )
            stage_cost = 0.0
            if index > 0:
                stage_cost = (
                    self.config.stage_fixed_cost
                    + self.config.stage_cost_per_m
                    * _distance(
                        context.current_state.eef_position_m,
                        candidate.eef_position_m,
                    )
                    + self.config.orientation_cost_per_rad
                    * _orientation_cost(candidate)
                )
            total = handoff_cost + stage_cost
            totals.append(total)
            approximations = candidate.approximated_features
            if index > 0 and not approximations:
                approximation = (
                    "visual_geometry_held_current"
                    if self.predictor.config.visual_prediction == "hold_current"
                    else "future_visual_geometry_unavailable"
                )
                candidate = candidate.model_copy(
                    update={"approximated_features": (approximation,)}
                )
            records.append(
                _record(
                    context,
                    candidate,
                    selected=False,
                    estimate=estimate,
                    handoff_cost=handoff_cost,
                    staging_cost=stage_cost,
                    total_cost=total,
                )
            )
        best = int(np.argmin(totals))
        if best != 0 and totals[0] - totals[best] <= self.config.hysteresis_cost:
            best = 0
        return _decision(
            self.name,
            context,
            best,
            records,
            "minimize configurable VLA, failure, staging, orientation, uncertainty, and hysteresis costs; execute one stage action then re-observe",
        )


class PostHocOraclePolicy:
    """Privileged matched-candidate upper bound; forbidden in deployment mode."""

    name = "posthoc_oracle_upper_bound"

    def __init__(
        self,
        actual_cost_by_candidate: Mapping[str, float],
        *,
        allow_privileged: bool = False,
    ) -> None:
        if not allow_privileged:
            raise PermissionError(
                "post-hoc oracle requires explicit allow_privileged=True and must not "
                "be used in the deployment policy path"
            )
        self.costs = {key: float(value) for key, value in actual_cost_by_candidate.items()}
        if any(not math.isfinite(value) or value < 0.0 for value in self.costs.values()):
            raise ValueError("oracle candidate costs must be finite and non-negative")

    def decide(self, context: PolicyContext) -> HandoffDecision:
        available = [
            (index, self.costs[candidate.candidate_id])
            for index, candidate in enumerate(context.candidates)
            if candidate.candidate_id in self.costs
        ]
        if not available:
            raise PolicyError("oracle has no outcomes for this matched candidate set")
        selected, _ = min(available, key=lambda item: item[1])
        records = [
            _record(
                context,
                candidate,
                selected=index == selected,
                total_cost=self.costs.get(candidate.candidate_id, 0.0),
                feasible=candidate.candidate_id in self.costs,
                infeasible_reason=(
                    None
                    if candidate.candidate_id in self.costs
                    else "candidate lacks a matched realized oracle outcome"
                ),
            )
            for index, candidate in enumerate(context.candidates)
        ]
        return _decision(
            self.name,
            context,
            selected,
            records,
            "evaluation-only selection using matched realized candidate costs",
        )
