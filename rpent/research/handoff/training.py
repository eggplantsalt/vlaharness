"""Explicit-label, group-split training/calibration orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rpent.research.handoff.dataset import (
    ExclusionReason,
    OutcomeDataset,
    TrainingTarget,
    dataset_fingerprint,
    extract_labeled_outcomes,
)
from rpent.research.handoff.features import FeatureBuilder, FeatureSpec, FeatureVector
from rpent.research.handoff.model import (
    BootstrapOutcomeModel,
    CalibratedOutcomeModel,
    ModelTrainingError,
    OutcomeModel,
    ProbabilityCalibrator,
    SklearnProbabilityModel,
)
from rpent.research.handoff.splits import (
    GroupSplitConfig,
    SplitAssignment,
    SplitName,
    connected_group_ids,
    split_outcomes,
)
from rpent.research.handoff.types import HandoffRecord, OutcomeRecord

TRAINING_REPORT_SCHEMA_VERSION = "rpent.handoff-training-report/v1"


class OutcomeTrainingConfig(BaseModel):
    """Complete, serializable training and uncertainty configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    target_label: TrainingTarget
    estimator_kind: Literal["logistic", "hist_gradient_boosting"] = "logistic"
    calibration_method: Literal["none", "platt", "isotonic"] = "platt"
    bootstrap_ensemble_size: int = Field(default=20, ge=1)
    uncertainty_beta: float = Field(default=1.0, ge=0.0)
    lower_quantile: float = Field(default=0.1, ge=0.0, le=1.0)
    upper_quantile: float = Field(default=0.9, ge=0.0, le=1.0)
    random_state: int = 0
    max_iter: int = Field(default=500, ge=1)
    split: GroupSplitConfig = Field(default_factory=GroupSplitConfig)

    @model_validator(mode="after")
    def validate_quantiles(self) -> "OutcomeTrainingConfig":
        if self.lower_quantile > self.upper_quantile:
            raise ValueError("uncertainty quantiles must be ordered")
        return self


class PartitionSummary(HandoffRecord):
    examples: int = Field(ge=0)
    successes: int = Field(ge=0)
    failures: int = Field(ge=0)


class TrainingReport(HandoffRecord):
    schema_version: Literal[TRAINING_REPORT_SCHEMA_VERSION] = (
        TRAINING_REPORT_SCHEMA_VERSION
    )
    raw_dataset_fingerprint: str
    eligible_dataset_fingerprint: str
    feature_spec_id: str
    feature_spec_fingerprint: str
    target_label: str
    estimator_kind: str
    calibration_method: str
    ensemble_size: int = Field(ge=1)
    split_assignment_fingerprint: str
    train: PartitionSummary
    calibration: PartitionSummary
    test: PartitionSummary
    exclusion_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class PreparedPartition:
    records: tuple[OutcomeRecord, ...]
    features: tuple[FeatureVector, ...]
    labels: tuple[int, ...]

    @property
    def matrix(self) -> np.ndarray:
        if not self.features:
            return np.empty((0, 0), dtype=np.float64)
        return np.asarray([vector.values for vector in self.features], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: OutcomeModel
    feature_spec: FeatureSpec
    assignment: SplitAssignment
    train: PreparedPartition
    calibration: PreparedPartition
    test: PreparedPartition
    report: TrainingReport


def _summary(labels: Sequence[int]) -> PartitionSummary:
    successes = sum(int(value == 1) for value in labels)
    failures = sum(int(value == 0) for value in labels)
    return PartitionSummary(
        examples=len(labels), successes=successes, failures=failures
    )


def _partition(
    records: Sequence[OutcomeRecord],
    *,
    labels_by_record: dict[str, int],
    builder: FeatureBuilder,
) -> PreparedPartition:
    ordered = tuple(sorted(records, key=lambda record: record.record_id))
    vectors = tuple(builder.build(record.pre_handoff_state) for record in ordered)
    labels = tuple(labels_by_record[record.record_id] for record in ordered)
    return PreparedPartition(records=ordered, features=vectors, labels=labels)


def _require_two_classes(partition: PreparedPartition, name: str) -> None:
    if len(set(partition.labels)) < 2:
        raise ModelTrainingError(
            f"{name} partition needs both success and failure examples; "
            "collect more independent outcomes or change the split seed"
        )


def train_outcome_model(
    dataset: OutcomeDataset | Sequence[OutcomeRecord],
    *,
    feature_spec: FeatureSpec,
    config: OutcomeTrainingConfig,
) -> TrainingResult:
    """Train on exactly one requested label with no semantic substitutions."""
    if isinstance(dataset, OutcomeDataset):
        raw_records = dataset.records
        raw_fingerprint = dataset.fingerprint
    else:
        raw_records = tuple(dataset)
        raw_fingerprint = dataset_fingerprint(raw_records)
    extraction = extract_labeled_outcomes(
        raw_records, target=config.target_label
    )
    eligible_records = tuple(item.record for item in extraction.included)
    if not eligible_records:
        raise ModelTrainingError(
            f"no VLA outcomes contain requested label {config.target_label.value!r}"
        )
    labels_by_record = {
        item.record.record_id: int(item.value) for item in extraction.included
    }
    split = split_outcomes(eligible_records, config.split)
    builder = FeatureBuilder(feature_spec)
    train = _partition(split.train, labels_by_record=labels_by_record, builder=builder)
    calibration = _partition(
        split.calibration, labels_by_record=labels_by_record, builder=builder
    )
    test = _partition(split.test, labels_by_record=labels_by_record, builder=builder)
    _require_two_classes(train, "training")
    if config.calibration_method != "none":
        _require_two_classes(calibration, "calibration")

    if config.bootstrap_ensemble_size >= 2:
        component_ids = connected_group_ids(
            train.records, constraints=config.split.constraints
        )
        model: OutcomeModel = BootstrapOutcomeModel(
            estimator_kind=config.estimator_kind,
            feature_spec_id=feature_spec.spec_id,
            feature_names=feature_spec.names,
            ensemble_size=config.bootstrap_ensemble_size,
            random_state=config.random_state,
            uncertainty_beta=config.uncertainty_beta,
            lower_quantile=config.lower_quantile,
            upper_quantile=config.upper_quantile,
            calibration_method=config.calibration_method,
            max_iter=config.max_iter,
        ).fit(
            train.matrix,
            train.labels,
            groups=[component_ids[record.record_id] for record in train.records],
            calibration_x=(
                calibration.matrix
                if config.calibration_method != "none"
                else None
            ),
            calibration_y=(
                calibration.labels
                if config.calibration_method != "none"
                else None
            ),
        )
    else:
        base = SklearnProbabilityModel(
            estimator_kind=config.estimator_kind,
            feature_spec_id=feature_spec.spec_id,
            feature_names=feature_spec.names,
            random_state=config.random_state,
            max_iter=config.max_iter,
        ).fit(train.matrix, train.labels)
        if config.calibration_method == "none":
            model = base
        else:
            model = CalibratedOutcomeModel(
                base_model=base,
                calibrator=ProbabilityCalibrator(method=config.calibration_method),
            ).fit_calibrator(calibration.features, calibration.labels)

    exclusion_counts = {reason.value: 0 for reason in ExclusionReason}
    for excluded in extraction.excluded:
        exclusion_counts[excluded.reason.value] += 1
    report = TrainingReport(
        raw_dataset_fingerprint=raw_fingerprint,
        eligible_dataset_fingerprint=dataset_fingerprint(eligible_records),
        feature_spec_id=feature_spec.spec_id,
        feature_spec_fingerprint=feature_spec.fingerprint,
        target_label=config.target_label.value,
        estimator_kind=config.estimator_kind,
        calibration_method=config.calibration_method,
        ensemble_size=config.bootstrap_ensemble_size,
        split_assignment_fingerprint=split.assignment.fingerprint,
        train=_summary(train.labels),
        calibration=_summary(calibration.labels),
        test=_summary(test.labels),
        exclusion_counts=exclusion_counts,
    )
    return TrainingResult(
        model=model,
        feature_spec=feature_spec,
        assignment=split.assignment,
        train=train,
        calibration=calibration,
        test=test,
        report=report,
    )


def partition_for(
    result: TrainingResult, split: SplitName
) -> PreparedPartition:
    if split is SplitName.TRAIN:
        return result.train
    if split is SplitName.CALIBRATION:
        return result.calibration
    return result.test

