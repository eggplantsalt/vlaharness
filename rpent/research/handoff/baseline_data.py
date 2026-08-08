"""Versioned positive-only reference artifacts for retrieval baselines."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal, Sequence, Self

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.candidates import wrist_yaw_pitch
from rpent.research.handoff.dataset import (
    OutcomeDataset,
    TrainingTarget,
    extract_labeled_outcomes,
)
from rpent.research.handoff.policies import PositiveReference
from rpent.research.handoff.privileged import require_online_provenance
from rpent.research.handoff.types import HandoffRecord, OutcomeRecord

POSITIVE_REFERENCE_SCHEMA_VERSION = "rpent.handoff-positive-references/v1"


class PositiveReferenceBuildSettings(HandoffRecord):
    """Deterministic choices that define how references are selected/built."""

    maximum_references: int | None = Field(default=None, ge=1)
    selection_order: Literal["record_id_ascending"] = "record_id_ascending"
    geometry: Literal["target_relative_position_and_wrist_yaw_pitch/v1"] = (
        "target_relative_position_and_wrist_yaw_pitch/v1"
    )


class PositiveReferenceArtifact(HandoffRecord):
    schema_version: Literal[POSITIVE_REFERENCE_SCHEMA_VERSION] = (
        POSITIVE_REFERENCE_SCHEMA_VERSION
    )
    artifact_id: str
    dataset_fingerprint: str
    target_label: TrainingTarget
    deployment_provenance_verified: Literal[True]
    build_settings: PositiveReferenceBuildSettings
    source_record_ids: tuple[str, ...]
    references: tuple[PositiveReference, ...]
    excluded_counts: dict[str, int] = Field(default_factory=dict)

    @field_validator("artifact_id", "dataset_fingerprint")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        if not self.references:
            raise ValueError("positive reference artifact cannot be empty")
        ids = [item.reference_id for item in self.references]
        if len(ids) != len(set(ids)):
            raise ValueError("positive reference IDs must be unique")
        if (
            not self.source_record_ids
            or self.source_record_ids != tuple(sorted(self.source_record_ids))
            or len(self.source_record_ids) != len(set(self.source_record_ids))
        ):
            raise ValueError(
                "positive reference source record IDs must be non-empty, unique, "
                "and sorted"
            )
        expected_ids = tuple(
            f"positive-{record_id}" for record_id in self.source_record_ids
        )
        if tuple(ids) != expected_ids:
            raise ValueError(
                "positive references must align one-to-one with source record IDs"
            )
        if any(not reason or count < 0 for reason, count in self.excluded_counts.items()):
            raise ValueError("excluded_counts needs non-empty reasons and non-negative counts")
        return self


def _artifact_id(artifact: PositiveReferenceArtifact) -> str:
    payload = json.dumps(
        artifact.model_dump(
            mode="json",
            exclude={"artifact_id"},
            exclude_none=False,
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"positive-references-{digest[:20]}"


def build_positive_reference_artifact(
    dataset: OutcomeDataset | Sequence[OutcomeRecord],
    *,
    target: TrainingTarget | str,
    maximum_references: int | None = None,
) -> PositiveReferenceArtifact:
    resolved = (
        dataset
        if isinstance(dataset, OutcomeDataset)
        else OutcomeDataset.from_records(dataset)
    )
    extracted = extract_labeled_outcomes(resolved.records, target=target)
    positives = [item for item in extracted.included if item.value]
    positives.sort(key=lambda item: item.record.record_id)
    if maximum_references is not None:
        if maximum_references < 1:
            raise ValueError("maximum_references must be positive")
    excluded: dict[str, int] = {}
    for item in extracted.excluded:
        excluded[item.reason.value] = excluded.get(item.reason.value, 0) + 1
    negative_count = sum(not item.value for item in extracted.included)
    if negative_count:
        excluded["negative_target_label"] = negative_count

    eligible: list[tuple[str, PositiveReference]] = []
    for labeled in positives:
        state = labeled.state
        if state.target is None or state.target.estimate.position_m is None:
            excluded["missing_target_geometry"] = (
                excluded.get("missing_target_geometry", 0) + 1
            )
            continue
        # These coordinates become deployment-time retrieval inputs after the
        # artifact drops the full state. Verify every consumed source before
        # that provenance can no longer be inspected.
        require_online_provenance(
            state,
            (
                "eef_position_m",
                "eef_quaternion_xyzw",
                "target_position_m",
            ),
        )
        relative = tuple(
            float(eef - target_value)
            for eef, target_value in zip(
                state.eef_position_m, state.target.estimate.position_m
            )
        )
        yaw, pitch = wrist_yaw_pitch(state.eef_quaternion_xyzw)
        eligible.append(
            (
                labeled.record.record_id,
                PositiveReference(
                    reference_id=f"positive-{labeled.record.record_id}",
                    target_relative_position_m=relative,
                    wrist_yaw_rad=yaw,
                    wrist_pitch_rad=pitch,
                ),
            )
        )
    selected = eligible
    if maximum_references is not None:
        selected = eligible[:maximum_references]
        omitted_count = len(eligible) - len(selected)
        if omitted_count:
            excluded["maximum_references"] = omitted_count
    source_record_ids = tuple(record_id for record_id, _ in selected)
    references = tuple(reference for _, reference in selected)
    artifact_without_id = PositiveReferenceArtifact(
        artifact_id="pending",
        dataset_fingerprint=resolved.fingerprint,
        target_label=TrainingTarget(target),
        deployment_provenance_verified=True,
        build_settings=PositiveReferenceBuildSettings(
            maximum_references=maximum_references,
        ),
        source_record_ids=source_record_ids,
        references=references,
        excluded_counts=excluded,
    )
    return artifact_without_id.model_copy(
        update={"artifact_id": _artifact_id(artifact_without_id)}
    )


def write_positive_reference_artifact(
    artifact: PositiveReferenceArtifact,
    path: str | os.PathLike[str],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(
            artifact.model_dump(mode="json", exclude_none=False),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def load_positive_reference_artifact(
    path: str | os.PathLike[str],
) -> PositiveReferenceArtifact:
    source = Path(path)
    artifact = PositiveReferenceArtifact.model_validate_json(
        source.read_text(encoding="utf-8")
    )
    if artifact.artifact_id != _artifact_id(artifact):
        raise ValueError(
            "positive-reference artifact identity mismatch; content, build "
            "settings, exclusions, or source identity may have been modified"
        )
    return artifact


__all__ = [
    "POSITIVE_REFERENCE_SCHEMA_VERSION",
    "PositiveReferenceArtifact",
    "PositiveReferenceBuildSettings",
    "build_positive_reference_artifact",
    "load_positive_reference_artifact",
    "write_positive_reference_artifact",
]
