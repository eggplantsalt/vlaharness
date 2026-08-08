"""Privileged-data records and the deployment-policy provenance firewall.

Simulator state, experimental setup, and evaluator outputs deliberately use
record types that cannot be mistaken for :class:`~.types.HandoffState`.  The
online feature builder accepts only ``HandoffState`` and calls the firewall in
this module for every source field it consumes.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import field_validator, model_validator

from rpent.research.handoff.types import (
    CandidateGeometry,
    FeatureAvailability,
    FeatureProvenance,
    HandoffRecord,
    HandoffState,
    TrialIdentity,
)

SETUP_SCHEMA_VERSION = "rpent.handoff-experiment-setup/v1"
PRIVILEGED_OBSERVATION_SCHEMA_VERSION = "rpent.handoff-privileged-observation/v1"
PRIVILEGED_EVALUATOR_SCHEMA_VERSION = "rpent.handoff-privileged-evaluator/v1"


class ProvenanceFirewallError(ValueError):
    """Raised when an online feature is missing or has forbidden provenance."""


def _non_empty(value: str, name: str) -> str:
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


class SetupValue(HandoffRecord):
    """One experiment-only numeric value, never an online policy input."""

    name: str
    values: tuple[float, ...]
    unit: str
    frame: str
    source: str
    availability: Literal[FeatureAvailability.EXPERIMENT_SETUP] = (
        FeatureAvailability.EXPERIMENT_SETUP
    )

    @field_validator("name", "unit", "frame", "source")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("setup value must contain at least one component")
        return value


class PrivilegedValue(HandoffRecord):
    """One simulator-only observation retained for labels or diagnostics."""

    name: str
    values: tuple[float, ...]
    unit: str
    frame: str
    source: str
    availability: Literal[FeatureAvailability.SIMULATOR_PRIVILEGED] = (
        FeatureAvailability.SIMULATOR_PRIVILEGED
    )

    @field_validator("name", "unit", "frame", "source")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("privileged value must contain at least one component")
        return value


class ExperimentSetupRecord(HandoffRecord):
    """Controlled trial setup kept separate from deployment observations."""

    schema_version: Literal[SETUP_SCHEMA_VERSION] = SETUP_SCHEMA_VERSION
    record_id: str
    identity: TrialIdentity
    setup_provider: str
    requested_candidate: CandidateGeometry | None = None
    values: tuple[SetupValue, ...] = ()
    notes: tuple[str, ...] = ()

    @field_validator("record_id", "setup_provider")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)

    @model_validator(mode="after")
    def validate_unique_names(self) -> Self:
        _require_unique_names(self.values, "experiment setup")
        return self


class PrivilegedObservationRecord(HandoffRecord):
    """Simulator-only values stored for diagnostics, labels, or oracle use."""

    schema_version: Literal[PRIVILEGED_OBSERVATION_SCHEMA_VERSION] = (
        PRIVILEGED_OBSERVATION_SCHEMA_VERSION
    )
    record_id: str
    identity: TrialIdentity
    values: tuple[PrivilegedValue, ...]

    @field_validator("record_id")
    @classmethod
    def validate_record_id(cls, value: str) -> str:
        return _non_empty(value, "record_id")

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        if not self.values:
            raise ValueError("privileged observation record cannot be empty")
        _require_unique_names(self.values, "privileged observation")
        return self


class PrivilegedEvaluatorSignal(HandoffRecord):
    """One evaluator label whose deployment policy must never consume it."""

    name: str
    value: bool | None
    definition: str
    unavailable_reason: str | None = None

    @field_validator("name", "definition")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.value is None and not self.unavailable_reason:
            raise ValueError("unavailable evaluator signal needs unavailable_reason")
        if self.value is not None and self.unavailable_reason:
            raise ValueError("available evaluator signal cannot have unavailable_reason")
        return self


class PrivilegedEvaluatorRecord(HandoffRecord):
    """Skill-specific or simulator evaluator output stored label-side only."""

    schema_version: Literal[PRIVILEGED_EVALUATOR_SCHEMA_VERSION] = (
        PRIVILEGED_EVALUATOR_SCHEMA_VERSION
    )
    record_id: str
    identity: TrialIdentity
    evaluator_id: str
    evaluator_version: str
    signals: tuple[PrivilegedEvaluatorSignal, ...]

    @field_validator("record_id", "evaluator_id", "evaluator_version")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)

    @model_validator(mode="after")
    def validate_signals(self) -> Self:
        if not self.signals:
            raise ValueError("privileged evaluator record cannot be empty")
        _require_unique_names(self.signals, "privileged evaluator")
        return self


def _require_unique_names(values: tuple[object, ...], context: str) -> None:
    names = [getattr(item, "name") for item in values]
    if len(names) != len(set(names)):
        raise ValueError(f"{context} names must be unique")


def provenance_index(state: HandoffState) -> dict[str, FeatureProvenance]:
    """Return the state's strict name-to-provenance map."""
    return {item.feature_name: item for item in state.provenance}


def require_online_provenance(
    state: HandoffState,
    feature_names: tuple[str, ...] | list[str] | set[str],
) -> tuple[FeatureProvenance, ...]:
    """Validate and return provenance for every online-consumed source.

    Missing entries fail closed. Simulator-privileged and experiment-setup
    provenance fail closed even if the numeric value itself looks innocuous.
    Nested target availability is checked independently so a caller cannot
    relabel a privileged target merely by forging a deployment-looking entry in
    ``state.provenance``.
    """
    names = tuple(feature_names)
    index = provenance_index(state)
    missing = sorted(set(names).difference(index))
    if missing:
        raise ProvenanceFirewallError(
            f"online feature sources have no provenance: {missing}"
        )

    selected = tuple(index[name] for name in names)
    forbidden = [
        f"{item.feature_name} ({item.availability.value}, source={item.source})"
        for item in selected
        if not item.availability.online_allowed
    ]
    if forbidden:
        raise ProvenanceFirewallError(
            "online policy rejected non-deployment provenance: "
            + ", ".join(forbidden)
        )

    target_sources = {
        "target_position_m",
        "target_confidence",
        "mask_area_fraction",
        "valid_depth_fraction",
        "image_centroid_rc_normalized",
    }
    if target_sources.intersection(names):
        if state.target is None:
            raise ProvenanceFirewallError("target feature requested without a target")
        availability = state.target.estimate.availability
        if not availability.online_allowed:
            raise ProvenanceFirewallError(
                "online policy rejected target estimate availability "
                f"{availability.value!r} from {state.target.estimate.provider!r}"
            )
    return selected


def reject_privileged_policy_value(value: object) -> None:
    """Fail if a label/setup record is accidentally routed to online policy."""
    if isinstance(
        value,
        (
            ExperimentSetupRecord,
            PrivilegedObservationRecord,
            PrivilegedEvaluatorRecord,
        ),
    ):
        raise ProvenanceFirewallError(
            f"{type(value).__name__} is label/setup-side and cannot be policy input"
        )
