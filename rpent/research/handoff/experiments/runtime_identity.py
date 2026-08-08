"""Pre-action runtime checkpoint attestation for research executions.

This module is intentionally lightweight: it calls only the read-only probe
methods already exposed by the Pi0.5 and SAM3 clients.  It never imports a
model implementation, resets an environment, or performs inference.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.experiments.config import stable_identifier
from rpent.research.handoff.types import HandoffRecord

RUNTIME_ATTESTATION_SCHEMA_VERSION = "rpent.handoff-runtime-attestation/v1"
_PROBE_SCHEMA_VERSION = "rpent.runtime-probe/v1"


class CheckpointObservation(HandoffRecord):
    """One manifest expectation compared with one live server response."""

    component: Literal["pi0.5_vla", "sam3"]
    expected_checkpoint_id: str
    observed_checkpoint_id: str
    expected_checkpoint_path: str | None = None
    observed_checkpoint_path: str
    external_endpoint: bool
    probe_sha256: str
    probe_payload: dict[str, Any]

    @field_validator(
        "expected_checkpoint_id",
        "observed_checkpoint_id",
        "observed_checkpoint_path",
        "probe_sha256",
    )
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value


class RuntimeCheckpointAttestation(HandoffRecord):
    """Run-local proof that live clients matched configured identities."""

    schema_version: Literal[RUNTIME_ATTESTATION_SCHEMA_VERSION] = (
        RUNTIME_ATTESTATION_SCHEMA_VERSION
    )
    attestation_id: str
    trial_id: str
    manifest_id: str | None = None
    plan_id: str | None = None
    source_revision: str | None = None
    observed_before_reset_or_action: Literal[True] = True
    observations: tuple[CheckpointObservation, ...]

    @field_validator("attestation_id", "trial_id")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_identity(self):
        components = tuple(item.component for item in self.observations)
        if components != ("pi0.5_vla", "sam3"):
            raise ValueError(
                "runtime attestation must contain ordered Pi0.5 and SAM3 observations"
            )
        expected = stable_identifier(
            "runtime-attestation",
            self.model_dump(
                mode="json",
                exclude={"attestation_id"},
                exclude_none=False,
            ),
        )
        if self.attestation_id != expected:
            raise ValueError(
                "runtime attestation identity does not bind its complete payload"
            )
        return self


def _canonical_payload(value: Mapping[str, Any], *, component: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{component} runtime probe returned non-finite/non-JSON metadata"
        ) from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{component} runtime probe did not return an object")
    return decoded


def _resolved_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _observe_checkpoint(
    client: Any,
    *,
    component: Literal["pi0.5_vla", "sam3"],
    expected_id: str | None,
    expected_path: str | None,
    external_endpoint: bool,
) -> CheckpointObservation:
    if expected_id is None or not expected_id.strip():
        raise ValueError(f"{component} expected checkpoint ID is required")
    runtime_probe = getattr(client, "runtime_probe", None)
    if not callable(runtime_probe):
        raise RuntimeError(f"{component} client has no runtime_probe()")
    raw = runtime_probe()
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"{component} runtime_probe() did not return an object")
    payload = _canonical_payload(raw, component=component)
    if payload.get("schema_version") != _PROBE_SCHEMA_VERSION:
        raise RuntimeError(f"{component} runtime probe schema mismatch")
    if payload.get("component") != component:
        raise RuntimeError(
            f"runtime probe component mismatch: expected {component!r}, "
            f"got {payload.get('component')!r}"
        )
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"{component} runtime probe lacks checkpoint metadata")
    observed_id = checkpoint.get("configured_id")
    observed_path = checkpoint.get("path")
    if not isinstance(observed_id, str) or not observed_id:
        raise RuntimeError(f"{component} live checkpoint configured_id is missing")
    if observed_id != expected_id:
        raise RuntimeError(
            f"{component} live checkpoint ID mismatch: "
            f"expected {expected_id!r}, got {observed_id!r}"
        )
    if not isinstance(observed_path, str) or not observed_path:
        raise RuntimeError(f"{component} live checkpoint path is missing")
    if checkpoint.get("exists") is not True:
        raise RuntimeError(f"{component} live checkpoint path does not exist")
    if not external_endpoint:
        if expected_path is None or not expected_path.strip():
            raise ValueError(f"local {component} runtime needs a configured path")
        if _resolved_path(observed_path) != _resolved_path(expected_path):
            raise RuntimeError(
                f"{component} live checkpoint path mismatch: "
                f"expected {str(_resolved_path(expected_path))!r}, "
                f"got {str(_resolved_path(observed_path))!r}"
            )
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return CheckpointObservation(
        component=component,
        expected_checkpoint_id=expected_id,
        observed_checkpoint_id=observed_id,
        expected_checkpoint_path=expected_path,
        observed_checkpoint_path=observed_path,
        external_endpoint=external_endpoint,
        probe_sha256=hashlib.sha256(canonical).hexdigest(),
        probe_payload=payload,
    )


def attest_runtime_checkpoint_clients(
    primitives_kwargs: Mapping[str, Any],
    runtime: Any,
    *,
    trial_id: str,
    manifest_id: str | None = None,
    plan_id: str | None = None,
    source_revision: str | None = None,
) -> RuntimeCheckpointAttestation:
    """Fail closed unless both connected services match runtime expectations."""
    try:
        model_client = primitives_kwargs["model"]
        sam3_client = primitives_kwargs["sam3_client"]
    except KeyError as exc:
        raise RuntimeError(
            f"runtime primitives omitted checkpoint client {exc.args[0]!r}"
        ) from exc
    observations = (
        _observe_checkpoint(
            model_client,
            component="pi0.5_vla",
            expected_id=runtime.pi05_checkpoint_id,
            expected_path=runtime.pi05_checkpoint_path,
            external_endpoint=runtime.vla_endpoint is not None,
        ),
        _observe_checkpoint(
            sam3_client,
            component="sam3",
            expected_id=runtime.sam3_checkpoint_id,
            expected_path=runtime.sam3_checkpoint_path,
            external_endpoint=runtime.sam3_endpoint is not None,
        ),
    )
    pending = RuntimeCheckpointAttestation.model_construct(
        schema_version=RUNTIME_ATTESTATION_SCHEMA_VERSION,
        attestation_id="pending",
        trial_id=trial_id,
        manifest_id=manifest_id,
        plan_id=plan_id,
        source_revision=source_revision,
        observed_before_reset_or_action=True,
        observations=observations,
    )
    payload = pending.model_dump(
        mode="json", exclude={"attestation_id"}, exclude_none=False
    )
    return RuntimeCheckpointAttestation(
        **payload,
        attestation_id=stable_identifier("runtime-attestation", payload),
    )


def write_runtime_attestation(
    attestation: RuntimeCheckpointAttestation,
    path: str | os.PathLike[str],
) -> Path:
    """Persist once; an identical existing attestation is idempotent."""
    destination = Path(path).expanduser().resolve()
    canonical = (attestation.canonical_json() + "\n").encode("utf-8")
    if destination.exists():
        existing = destination.read_bytes()
        if existing != canonical:
            raise FileExistsError(
                f"runtime attestation already exists with different content: {destination}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(canonical)
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def load_runtime_attestation(
    path: str | os.PathLike[str],
) -> RuntimeCheckpointAttestation:
    return RuntimeCheckpointAttestation.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def verify_runtime_attestation_binding(
    path: str | os.PathLike[str],
    *,
    trial_id: str,
    manifest_id: str,
    plan_id: str,
    source_revision: str,
    expected_attestation_id: str | None = None,
    expected_sha256: str | None = None,
) -> tuple[RuntimeCheckpointAttestation, str]:
    """Strictly bind one immutable attestation file to an execution identity."""
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"runtime attestation is unreadable: {source}") from exc
    try:
        attestation = RuntimeCheckpointAttestation.model_validate_json(text)
    except ValueError as exc:
        raise RuntimeError(f"runtime attestation is invalid: {source}") from exc
    canonical = (attestation.canonical_json() + "\n").encode("utf-8")
    if raw != canonical:
        raise RuntimeError("runtime attestation bytes are not canonical")
    expected_fields = {
        "trial_id": trial_id,
        "manifest_id": manifest_id,
        "plan_id": plan_id,
        "source_revision": source_revision,
    }
    mismatches = {
        field: {"expected": expected, "actual": getattr(attestation, field)}
        for field, expected in expected_fields.items()
        if getattr(attestation, field) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"runtime attestation disagrees with execution identity: {mismatches}"
        )
    if (
        expected_attestation_id is not None
        and attestation.attestation_id != expected_attestation_id
    ):
        raise RuntimeError("runtime attestation ID disagrees with bound sidecar")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError("runtime attestation file hash disagrees with bound sidecar")
    return attestation, digest


__all__ = [
    "CheckpointObservation",
    "RUNTIME_ATTESTATION_SCHEMA_VERSION",
    "RuntimeCheckpointAttestation",
    "attest_runtime_checkpoint_clients",
    "load_runtime_attestation",
    "verify_runtime_attestation_binding",
    "write_runtime_attestation",
]
