"""Ordered, provenance-checked deployment feature specifications.

The feature layer is intentionally NumPy-free.  It produces immutable tuples
with stable names and fingerprints so model artifacts can reject reordered or
otherwise incompatible inputs before prediction.
"""

from __future__ import annotations

import hashlib
import math
from enum import Enum
from typing import Literal, Self, Sequence

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.privileged import (
    ProvenanceFirewallError,
    require_online_provenance,
)
from rpent.research.handoff.types import (
    FeatureAvailability,
    FeatureProvenance,
    HandoffRecord,
    HandoffState,
    STATE_SCHEMA_VERSION,
)

FEATURE_SPEC_SCHEMA_VERSION = "rpent.handoff-feature-spec/v1"
FEATURE_VECTOR_SCHEMA_VERSION = "rpent.handoff-feature-vector/v1"


class FeaturePreset(str, Enum):
    """Required representation ablations."""

    ABSOLUTE = "absolute"
    TARGET_RELATIVE = "target_relative"
    TARGET_RELATIVE_VISUAL = "target_relative_visual"
    DEPLOYMENT_FULL = "deployment_full"


class FeatureKind(str, Enum):
    EEF_POSITION = "eef_position"
    EEF_QUATERNION = "eef_quaternion"
    GRIPPER_OPENING = "gripper_opening"
    TARGET_POSITION = "target_position"
    TARGET_RELATIVE_POSITION = "target_relative_position"
    TARGET_CONFIDENCE = "target_confidence"
    TARGET_CONFIDENCE_AVAILABLE = "target_confidence_available"
    MASK_AREA = "mask_area"
    MASK_AREA_AVAILABLE = "mask_area_available"
    VALID_DEPTH = "valid_depth"
    VALID_DEPTH_AVAILABLE = "valid_depth_available"
    IMAGE_CENTROID = "image_centroid"
    IMAGE_CENTROID_AVAILABLE = "image_centroid_available"
    SKILL_ONE_HOT = "skill_one_hot"


_COMPONENT_KINDS = {
    FeatureKind.EEF_POSITION,
    FeatureKind.EEF_QUATERNION,
    FeatureKind.TARGET_POSITION,
    FeatureKind.TARGET_RELATIVE_POSITION,
    FeatureKind.IMAGE_CENTROID,
}


class FeatureField(HandoffRecord):
    """One ordered scalar model input."""

    name: str
    kind: FeatureKind
    unit: str
    frame: str
    component: int | None = Field(default=None, ge=0)
    skill_name: str | None = None
    missing_fill_value: float = 0.0

    @field_validator("name", "unit", "frame")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_kind_arguments(self) -> Self:
        if self.kind in _COMPONENT_KINDS and self.component is None:
            raise ValueError(f"{self.kind.value} feature needs a component")
        if self.kind not in _COMPONENT_KINDS and self.component is not None:
            raise ValueError(f"{self.kind.value} feature cannot have a component")
        if self.kind is FeatureKind.SKILL_ONE_HOT:
            if not self.skill_name:
                raise ValueError("skill one-hot feature needs skill_name")
        elif self.skill_name is not None:
            raise ValueError("skill_name is valid only for skill one-hot features")
        component_limits = {
            FeatureKind.EEF_POSITION: 3,
            FeatureKind.EEF_QUATERNION: 4,
            FeatureKind.TARGET_POSITION: 3,
            FeatureKind.TARGET_RELATIVE_POSITION: 3,
            FeatureKind.IMAGE_CENTROID: 2,
        }
        limit = component_limits.get(self.kind)
        if limit is not None and self.component is not None and self.component >= limit:
            raise ValueError(
                f"component {self.component} out of range for {self.kind.value}"
            )
        return self


class FeatureSpec(HandoffRecord):
    """Serializable ordered feature contract stored with every model."""

    schema_version: Literal[FEATURE_SPEC_SCHEMA_VERSION] = FEATURE_SPEC_SCHEMA_VERSION
    spec_id: str
    preset: FeaturePreset
    state_schema_version: Literal[STATE_SCHEMA_VERSION] = STATE_SCHEMA_VERSION
    fields: tuple[FeatureField, ...]
    skill_vocabulary: tuple[str, ...] = ()

    @field_validator("spec_id")
    @classmethod
    def validate_spec_id(cls, value: str) -> str:
        if not value:
            raise ValueError("spec_id must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        if not self.fields:
            raise ValueError("feature specification cannot be empty")
        names = self.names
        if len(names) != len(set(names)):
            raise ValueError("feature field names must be unique")
        if (
            tuple(sorted(set(self.skill_vocabulary))) != self.skill_vocabulary
            or any(not name for name in self.skill_vocabulary)
        ):
            raise ValueError("skill_vocabulary must contain unique, sorted names")
        declared_skills = tuple(
            field.skill_name
            for field in self.fields
            if field.kind is FeatureKind.SKILL_ONE_HOT
        )
        if declared_skills != self.skill_vocabulary:
            raise ValueError(
                "skill one-hot field order must exactly match skill_vocabulary"
            )
        return self

    @property
    def names(self) -> tuple[str, ...]:
        """Ordered scalar names consumed by estimators."""
        return tuple(field.name for field in self.fields)

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint of the complete ordered contract."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def validate_vector(self, vector: FeatureVector) -> None:
        """Reject feature reorderings and cross-spec vectors."""
        if vector.spec_id != self.spec_id:
            raise FeatureCompatibilityError(
                f"feature spec id mismatch: {vector.spec_id!r} != {self.spec_id!r}"
            )
        if vector.spec_fingerprint != self.fingerprint:
            raise FeatureCompatibilityError("feature specification fingerprint mismatch")
        if vector.names != self.names:
            raise FeatureCompatibilityError("feature name/order mismatch")


class FeatureVector(HandoffRecord):
    """One finite, ordered, policy-safe numeric vector."""

    schema_version: Literal[FEATURE_VECTOR_SCHEMA_VERSION] = FEATURE_VECTOR_SCHEMA_VERSION
    spec_id: str
    spec_fingerprint: str
    state_id: str
    names: tuple[str, ...]
    values: tuple[float, ...]
    provenance: tuple[FeatureProvenance, ...]

    @field_validator("spec_id", "spec_fingerprint", "state_id")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        size = len(self.names)
        if not size:
            raise ValueError("feature vector cannot be empty")
        if len(set(self.names)) != size:
            raise ValueError("feature vector names must be unique")
        if len(self.values) != size or len(self.provenance) != size:
            raise ValueError("feature names, values, and provenance must align")
        provenance_names = tuple(item.feature_name for item in self.provenance)
        if provenance_names != self.names:
            raise ValueError("feature provenance must follow exact vector order")
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("feature vector contains NaN or infinity")
        forbidden = [
            item.feature_name
            for item in self.provenance
            if not item.availability.online_allowed
        ]
        if forbidden:
            raise ProvenanceFirewallError(
                f"feature vector contains non-deployment provenance: {forbidden}"
            )
        return self


class FeatureCompatibilityError(ValueError):
    """Raised when a vector does not match its model's exact feature contract."""


class FeatureBuilder:
    """Build finite vectors while checking every consumed source field."""

    provider_version = FEATURE_VECTOR_SCHEMA_VERSION

    def __init__(self, spec: FeatureSpec | None = None) -> None:
        self.spec = spec

    def build(
        self,
        state: HandoffState,
        spec: FeatureSpec | None = None,
    ) -> FeatureVector:
        """Build with an argument spec or the spec bound at construction."""
        selected_spec = spec or self.spec
        if selected_spec is None:
            raise FeatureCompatibilityError(
                "FeatureBuilder requires a bound or per-call FeatureSpec"
            )
        spec = selected_spec
        if state.schema_version != spec.state_schema_version:
            raise FeatureCompatibilityError(
                "handoff state schema is incompatible with feature specification"
            )
        values: list[float] = []
        provenance: list[FeatureProvenance] = []
        for field in spec.fields:
            value, sources = self._extract(field, state, spec)
            if not math.isfinite(value):
                raise ValueError(f"feature {field.name!r} is not finite")
            values.append(float(value))
            provenance.append(self._output_provenance(field, sources))
        vector = FeatureVector(
            spec_id=spec.spec_id,
            spec_fingerprint=spec.fingerprint,
            state_id=state.state_id,
            names=spec.names,
            values=tuple(values),
            provenance=tuple(provenance),
        )
        spec.validate_vector(vector)
        return vector

    def _extract(
        self,
        field: FeatureField,
        state: HandoffState,
        spec: FeatureSpec,
    ) -> tuple[float, tuple[FeatureProvenance, ...]]:
        kind = field.kind
        component = field.component
        if kind is FeatureKind.EEF_POSITION:
            sources = require_online_provenance(state, ("eef_position_m",))
            return state.eef_position_m[_component(component)], sources
        if kind is FeatureKind.EEF_QUATERNION:
            sources = require_online_provenance(state, ("eef_quaternion_xyzw",))
            return state.eef_quaternion_xyzw[_component(component)], sources
        if kind is FeatureKind.GRIPPER_OPENING:
            sources = require_online_provenance(state, ("gripper_opening_m",))
            return state.gripper_opening_m, sources
        if kind is FeatureKind.TARGET_POSITION:
            target_position = _target_position(state)
            sources = require_online_provenance(state, ("target_position_m",))
            return target_position[_component(component)], sources
        if kind is FeatureKind.TARGET_RELATIVE_POSITION:
            target_position = _target_position(state)
            sources = require_online_provenance(
                state, ("eef_position_m", "target_position_m")
            )
            index = _component(component)
            return state.eef_position_m[index] - target_position[index], sources
        if kind is FeatureKind.TARGET_CONFIDENCE:
            target = _target(state)
            if target.confidence is None:
                sources = require_online_provenance(state, ("target_position_m",))
                return field.missing_fill_value, sources
            sources = require_online_provenance(state, ("target_confidence",))
            return target.confidence, sources
        if kind is FeatureKind.TARGET_CONFIDENCE_AVAILABLE:
            target = _target(state)
            source_name = (
                "target_confidence"
                if target.confidence is not None
                else "target_position_m"
            )
            sources = require_online_provenance(state, (source_name,))
            return float(target.confidence is not None), sources
        if kind in {
            FeatureKind.MASK_AREA,
            FeatureKind.MASK_AREA_AVAILABLE,
            FeatureKind.VALID_DEPTH,
            FeatureKind.VALID_DEPTH_AVAILABLE,
            FeatureKind.IMAGE_CENTROID,
            FeatureKind.IMAGE_CENTROID_AVAILABLE,
        }:
            return self._extract_visual(field, state)
        if kind is FeatureKind.SKILL_ONE_HOT:
            sources = require_online_provenance(state, ("skill",))
            if state.skill.name not in spec.skill_vocabulary:
                raise FeatureCompatibilityError(
                    f"skill {state.skill.name!r} is outside feature vocabulary"
                )
            return float(state.skill.name == field.skill_name), sources
        raise AssertionError(f"unhandled feature kind: {kind}")

    def _extract_visual(
        self,
        field: FeatureField,
        state: HandoffState,
    ) -> tuple[float, tuple[FeatureProvenance, ...]]:
        target = _target(state)
        visual = target.visual_geometry
        source_by_kind = {
            FeatureKind.MASK_AREA: "mask_area_fraction",
            FeatureKind.MASK_AREA_AVAILABLE: "mask_area_fraction",
            FeatureKind.VALID_DEPTH: "valid_depth_fraction",
            FeatureKind.VALID_DEPTH_AVAILABLE: "valid_depth_fraction",
            FeatureKind.IMAGE_CENTROID: "image_centroid_rc_normalized",
            FeatureKind.IMAGE_CENTROID_AVAILABLE: "image_centroid_rc_normalized",
        }
        source_name = source_by_kind[field.kind]
        if visual is None:
            sources = require_online_provenance(state, ("target_position_m",))
            if field.kind in {
                FeatureKind.MASK_AREA_AVAILABLE,
                FeatureKind.VALID_DEPTH_AVAILABLE,
                FeatureKind.IMAGE_CENTROID_AVAILABLE,
            }:
                return 0.0, sources
            return field.missing_fill_value, sources

        sources = require_online_provenance(state, (source_name,))
        if field.kind is FeatureKind.MASK_AREA:
            return visual.mask_area_fraction, sources
        if field.kind is FeatureKind.VALID_DEPTH:
            return visual.valid_depth_fraction, sources
        if field.kind is FeatureKind.IMAGE_CENTROID:
            return visual.image_centroid_rc_normalized[_component(field.component)], sources
        return 1.0, sources

    def _output_provenance(
        self,
        field: FeatureField,
        sources: tuple[FeatureProvenance, ...],
    ) -> FeatureProvenance:
        if len(sources) == 1 and field.kind in {
            FeatureKind.EEF_POSITION,
            FeatureKind.EEF_QUATERNION,
            FeatureKind.GRIPPER_OPENING,
            FeatureKind.TARGET_POSITION,
        }:
            availability = sources[0].availability
        else:
            availability = FeatureAvailability.DERIVED_DEPLOYMENT
        source_names = ",".join(item.feature_name for item in sources)
        providers = ",".join(sorted({item.source for item in sources}))
        return FeatureProvenance(
            feature_name=field.name,
            availability=availability,
            source=f"feature_builder[{providers}]",
            unit=field.unit,
            frame=field.frame,
            derivation=f"{field.kind.value} from {source_names}",
            provider_version=self.provider_version,
        )


def _component(value: int | None) -> int:
    if value is None:
        raise AssertionError("validated component is unexpectedly absent")
    return value


def _target(state: HandoffState):
    if state.target is None:
        raise FeatureCompatibilityError("feature specification requires a target")
    if not state.target.estimate.availability.online_allowed:
        raise ProvenanceFirewallError(
            "feature specification rejected non-deployment target estimate"
        )
    return state.target.estimate


def _target_position(state: HandoffState) -> tuple[float, float, float]:
    target = _target(state)
    if target.position_m is None:
        raise FeatureCompatibilityError(
            "target-relative feature requires an available target position"
        )
    return target.position_m


def make_feature_spec(
    preset: FeaturePreset | str,
    *,
    skill_vocabulary: Sequence[str],
) -> FeatureSpec:
    """Build one required ablation spec with deterministic field ordering."""
    preset = FeaturePreset(preset)
    supplied_skills = tuple(skill_vocabulary)
    if (
        not supplied_skills
        or any(not name.strip() for name in supplied_skills)
        or len(supplied_skills) != len(set(supplied_skills))
    ):
        raise ValueError("skill_vocabulary must contain non-empty skill names")
    skills = tuple(sorted(supplied_skills))

    absolute = (
        _field("eef_x_m", FeatureKind.EEF_POSITION, "m", "world", 0),
        _field("eef_y_m", FeatureKind.EEF_POSITION, "m", "world", 1),
        _field("eef_z_m", FeatureKind.EEF_POSITION, "m", "world", 2),
    )
    orientation_and_gripper = (
        _field("eef_quat_x", FeatureKind.EEF_QUATERNION, "unitless", "world", 0),
        _field("eef_quat_y", FeatureKind.EEF_QUATERNION, "unitless", "world", 1),
        _field("eef_quat_z", FeatureKind.EEF_QUATERNION, "unitless", "world", 2),
        _field("eef_quat_w", FeatureKind.EEF_QUATERNION, "unitless", "world", 3),
        _field("gripper_opening_m", FeatureKind.GRIPPER_OPENING, "m", "gripper"),
    )
    target_position = (
        _field("target_x_m", FeatureKind.TARGET_POSITION, "m", "world", 0),
        _field("target_y_m", FeatureKind.TARGET_POSITION, "m", "world", 1),
        _field("target_z_m", FeatureKind.TARGET_POSITION, "m", "world", 2),
    )
    relative = (
        _field("eef_minus_target_x_m", FeatureKind.TARGET_RELATIVE_POSITION, "m", "target_origin_world_axes", 0),
        _field("eef_minus_target_y_m", FeatureKind.TARGET_RELATIVE_POSITION, "m", "target_origin_world_axes", 1),
        _field("eef_minus_target_z_m", FeatureKind.TARGET_RELATIVE_POSITION, "m", "target_origin_world_axes", 2),
    )
    visual = (
        _field("target_confidence", FeatureKind.TARGET_CONFIDENCE, "probability", "none"),
        _field("target_confidence_available", FeatureKind.TARGET_CONFIDENCE_AVAILABLE, "boolean", "none"),
        _field("mask_area_fraction", FeatureKind.MASK_AREA, "fraction", "camera"),
        _field("mask_area_fraction_available", FeatureKind.MASK_AREA_AVAILABLE, "boolean", "none"),
        _field("valid_depth_fraction", FeatureKind.VALID_DEPTH, "fraction", "camera"),
        _field("valid_depth_fraction_available", FeatureKind.VALID_DEPTH_AVAILABLE, "boolean", "none"),
        _field("target_centroid_row", FeatureKind.IMAGE_CENTROID, "normalized", "camera", 0),
        _field("target_centroid_col", FeatureKind.IMAGE_CENTROID, "normalized", "camera", 1),
        _field("target_centroid_available", FeatureKind.IMAGE_CENTROID_AVAILABLE, "boolean", "none"),
    )
    skill_fields = tuple(
        FeatureField(
            name=f"skill={skill}",
            kind=FeatureKind.SKILL_ONE_HOT,
            unit="boolean",
            frame="semantic",
            skill_name=skill,
        )
        for skill in skills
    )

    if preset is FeaturePreset.ABSOLUTE:
        fields = absolute + orientation_and_gripper + skill_fields
    elif preset is FeaturePreset.TARGET_RELATIVE:
        fields = relative + orientation_and_gripper + skill_fields
    elif preset is FeaturePreset.TARGET_RELATIVE_VISUAL:
        fields = relative + orientation_and_gripper + visual + skill_fields
    else:
        fields = (
            absolute
            + orientation_and_gripper
            + target_position
            + relative
            + visual
            + skill_fields
        )
    return FeatureSpec(
        spec_id=f"rpent.handoff.features/{preset.value}/v1",
        preset=preset,
        fields=fields,
        skill_vocabulary=skills,
    )


def _field(
    name: str,
    kind: FeatureKind,
    unit: str,
    frame: str,
    component: int | None = None,
) -> FeatureField:
    return FeatureField(
        name=name,
        kind=kind,
        unit=unit,
        frame=frame,
        component=component,
    )
