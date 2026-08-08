"""Lazy, dependency-injected probes for Linux/GPU runtime unknowns.

Importing this module does not import LIBERO, MuJoCo, torch, Pi0.5, SAM3, or
numpy, start a subprocess, contact an endpoint, or mutate an environment.  The
caller supplies already-created clients.  Expensive inference and diagnostics
that can mutate an environment/model session are separately and explicitly
authorized through :class:`RuntimeProbeOptions`.

Probe results are experiment metadata, never deployment-policy inputs.  In
particular, the diagnostic state sample intentionally carries an
``experiment_only`` provenance supplied by the env server.
"""

from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import math
import platform
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any, Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.types import HandoffRecord

RUNTIME_PROBE_SCHEMA_VERSION = "rpent.handoff-runtime-probe/v1"


class ProbeStatus(str, Enum):
    """Epistemic status of one runtime fact."""

    OBSERVED = "observed"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    REQUIRES_DIAGNOSTIC = "requires_diagnostic"


class ProbeSafety(str, Enum):
    """Strongest operation used to obtain a fact."""

    READ_ONLY = "read_only"
    INFERENCE_ONLY = "inference_only"
    DESTRUCTIVE_ENVIRONMENT = "destructive_environment"
    ISOLATED_MODEL_DIAGNOSTIC = "isolated_model_diagnostic"


class HiddenStateConclusion(str, Enum):
    """Protocol-bounded conclusion from a controlled statefulness diagnostic."""

    STATE_DEPENDENCE_OBSERVED = "state_dependence_observed"
    NO_STATE_DEPENDENCE_OBSERVED = "no_state_dependence_observed"
    INCONCLUSIVE = "inconclusive"


class ProbeError(HandoffRecord):
    exception_type: str
    message: str

    @field_validator("exception_type", "message")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value


def _validate_json(value: Any, *, name: str) -> Any:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values") from exc
    return value


class ProbeFact(HandoffRecord):
    """One named fact, including missing evidence and diagnostic requirements."""

    name: str
    component: Literal["host", "env", "vla", "sam3", "diagnostic"]
    status: ProbeStatus
    safety: ProbeSafety = ProbeSafety.READ_ONLY
    source: str
    value: Any | None = None
    error: ProbeError | None = None
    detail: str | None = None
    policy_eligible: Literal[False] = False

    @field_validator("name", "source")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("detail")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise ValueError("detail must be non-empty when present")
        return value

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: Any) -> Any:
        return _validate_json(value, name="probe fact value")

    @model_validator(mode="after")
    def validate_status_payload(self) -> Self:
        if self.status is ProbeStatus.OBSERVED and self.value is None:
            raise ValueError("observed facts require a value")
        if self.status is ProbeStatus.ERROR:
            if self.error is None:
                raise ValueError("error facts require structured error details")
            if self.value is not None:
                raise ValueError("error facts cannot contain a value")
        elif self.error is not None:
            raise ValueError("only error facts may contain an error")
        if self.status is ProbeStatus.UNAVAILABLE and self.value is not None:
            raise ValueError("unavailable facts cannot contain a value")
        return self


class RuntimeProbeOptions(HandoffRecord):
    """Explicit opt-ins for operations beyond server metadata inspection."""

    run_host_gpu_discovery: bool = False
    run_vla_inference: bool = False
    run_sam3_inference: bool = False
    run_destructive_chunk_diagnostic: bool = False
    run_hidden_state_diagnostic: bool = False
    fresh_env_reset_confirmed: bool = False
    isolated_env_trial_confirmed: bool = False
    isolated_model_session_confirmed: bool = False

    @model_validator(mode="after")
    def validate_diagnostic_authorization(self) -> Self:
        if self.run_destructive_chunk_diagnostic and not (
            self.fresh_env_reset_confirmed and self.isolated_env_trial_confirmed
        ):
            raise ValueError(
                "destructive chunk diagnostic requires both a fresh reset and "
                "an isolated throwaway env trial"
            )
        if self.run_hidden_state_diagnostic and not (
            self.isolated_model_session_confirmed
        ):
            raise ValueError(
                "hidden-state diagnostic requires an isolated model session"
            )
        if self.run_vla_inference and not self.isolated_model_session_confirmed:
            raise ValueError(
                "VLA inference probe requires an isolated model session because "
                "episode-local model state is runtime-unverified"
            )
        return self


class HiddenStateDiagnosticResult(HandoffRecord):
    """Evidence contract for a caller-owned Pi0.5 statefulness experiment.

    A conclusion beyond ``inconclusive`` is accepted only when both reset
    behavior and stochasticity were controlled.  Even a negative result means
    only that no dependence was observed under the recorded protocol; it is
    not a proof that every model implementation is stateless.
    """

    protocol: str
    conclusion: HiddenStateConclusion
    repetitions: int = Field(ge=2)
    isolated_model_session: bool
    reset_controlled: bool
    stochasticity_controlled: bool
    evidence: dict[str, Any] = Field(default_factory=dict)

    @field_validator("protocol")
    @classmethod
    def validate_protocol(cls, value: str) -> str:
        if not value:
            raise ValueError("protocol must be non-empty")
        return value

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json(value, name="hidden-state evidence")

    @model_validator(mode="after")
    def validate_conclusion_strength(self) -> Self:
        if self.conclusion is not HiddenStateConclusion.INCONCLUSIVE and not (
            self.isolated_model_session
            and self.reset_controlled
            and self.stochasticity_controlled
        ):
            raise ValueError(
                "a conclusive hidden-state claim requires isolation plus "
                "controlled resets and stochasticity"
            )
        return self


REQUIRED_RUNTIME_FACTS = (
    "host.versions",
    "host.cuda_gpu",
    "env.endpoint_health",
    "env.versions",
    "env.observation_keys",
    "env.state_vector_shape",
    "env.state_field_values",
    "env.camera_views",
    "env.controller_environment_config",
    "env.cuda_gpu",
    "env.reset_identity",
    "env.termination_truncation_arrays",
    "vla.endpoint_health",
    "vla.versions",
    "vla.configured_action_shape",
    "vla.actual_action_shape",
    "vla.actual_chunk_size",
    "vla.controller_config",
    "vla.cuda_gpu",
    "vla.model_checkpoint_identity",
    "sam3.endpoint_health",
    "sam3.versions",
    "sam3.current_observation_contract",
    "sam3.current_observation_acceptance",
    "sam3.cuda_gpu",
    "sam3.model_checkpoint_identity",
    "diagnostic.chunk_done_mid_chunk_behavior",
    "diagnostic.vla_hidden_episode_state",
)


class RuntimeProbeReport(HandoffRecord):
    """Complete status ledger for all currently identified runtime unknowns."""

    schema_version: Literal[RUNTIME_PROBE_SCHEMA_VERSION] = (
        RUNTIME_PROBE_SCHEMA_VERSION
    )
    options: RuntimeProbeOptions
    facts: tuple[ProbeFact, ...]

    @model_validator(mode="after")
    def validate_fact_coverage(self) -> Self:
        names = [fact.name for fact in self.facts]
        if len(names) != len(set(names)):
            raise ValueError("runtime probe facts must have unique names")
        missing = sorted(set(REQUIRED_RUNTIME_FACTS).difference(names))
        extra = sorted(set(names).difference(REQUIRED_RUNTIME_FACTS))
        if missing or extra:
            raise ValueError(
                f"runtime probe fact coverage mismatch: missing={missing}, "
                f"extra={extra}"
            )
        return self

    def fact(self, name: str) -> ProbeFact:
        for fact in self.facts:
            if fact.name == name:
                return fact
        raise KeyError(name)

    @property
    def errors(self) -> tuple[ProbeFact, ...]:
        return tuple(fact for fact in self.facts if fact.status is ProbeStatus.ERROR)

    @property
    def pending_diagnostics(self) -> tuple[ProbeFact, ...]:
        return tuple(
            fact
            for fact in self.facts
            if fact.status is ProbeStatus.REQUIRES_DIAGNOSTIC
        )


class RuntimeProbeClient(Protocol):
    def runtime_probe(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]: ...


class EnvDiagnosticClient(RuntimeProbeClient, Protocol):
    def diagnostic_chunk_step(self, actions: Any) -> tuple[Any, Any, Any, Any, Any]: ...


class Sam3ProbeClient(RuntimeProbeClient, Protocol):
    def segment_image(self, image: Any, **kwargs: Any) -> Any: ...


HiddenStateDiagnostic = Callable[[Any], HiddenStateDiagnosticResult | Mapping[str, Any]]
GpuProbe = Callable[[], Mapping[str, Any]]
HealthCheck = Callable[[], Any]


def probe_local_versions(
    distributions: Sequence[str] = ("rpent", "numpy", "pydantic", "torch"),
) -> dict[str, Any]:
    """Read installed distribution metadata without importing those packages."""

    packages: dict[str, str | None] = {}
    for name in distributions:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def probe_nvidia_smi(
    *,
    runner: Callable[..., Any] | None = None,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Query GPU identity/memory with a fixed, non-shell subprocess command.

    The helper is never called implicitly unless ``run_host_gpu_discovery`` is
    enabled.  ``runner`` exists for tests and managed server environments.
    """

    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout_s must be finite and positive")
    if runner is None:
        import subprocess

        runner = subprocess.run
    command = (
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,driver_version",
        "--format=csv,noheader,nounits",
    )
    completed = runner(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=float(timeout_s),
        shell=False,
    )
    return_code = int(completed.returncode)
    if return_code != 0:
        message = str(getattr(completed, "stderr", "")).strip()
        raise RuntimeError(
            f"nvidia-smi exited with code {return_code}: {message[:500]}"
        )
    reader = csv.reader(io.StringIO(str(completed.stdout)))
    devices: list[dict[str, Any]] = []
    for row_number, row in enumerate(reader, start=1):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) != 6:
            raise RuntimeError(
                f"nvidia-smi row {row_number} has {len(row)} fields, expected 6"
            )
        try:
            devices.append(
                {
                    "index": int(row[0].strip()),
                    "name": row[1].strip(),
                    "uuid": row[2].strip(),
                    "memory_total_mib": int(row[3].strip()),
                    "memory_used_mib": int(row[4].strip()),
                    "driver_version": row[5].strip(),
                }
            )
        except ValueError as exc:
            raise RuntimeError(
                f"could not parse nvidia-smi row {row_number}: {row!r}"
            ) from exc
    return {"tool": "nvidia-smi", "device_count": len(devices), "devices": devices}


_MISSING = object()


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _error(exc: Exception) -> ProbeError:
    return ProbeError(exception_type=type(exc).__name__, message=str(exc) or repr(exc))


def _observed(
    name: str,
    component: Literal["host", "env", "vla", "sam3", "diagnostic"],
    value: Any,
    *,
    source: str,
    safety: ProbeSafety = ProbeSafety.READ_ONLY,
    detail: str | None = None,
) -> ProbeFact:
    try:
        return ProbeFact(
            name=name,
            component=component,
            status=ProbeStatus.OBSERVED,
            safety=safety,
            source=source,
            value=value,
            detail=detail,
        )
    except Exception as exc:
        return ProbeFact(
            name=name,
            component=component,
            status=ProbeStatus.ERROR,
            safety=safety,
            source=source,
            error=_error(exc),
            detail="probe returned a non-finite or non-JSON-compatible value",
        )


def _missing_fact(
    name: str,
    component: Literal["host", "env", "vla", "sam3", "diagnostic"],
    *,
    source: str,
    detail: str,
) -> ProbeFact:
    return ProbeFact(
        name=name,
        component=component,
        status=ProbeStatus.UNAVAILABLE,
        source=source,
        detail=detail,
    )


def _requires(
    name: str,
    component: Literal["host", "env", "vla", "sam3", "diagnostic"],
    *,
    source: str,
    detail: str,
    value: Any | None = None,
    safety: ProbeSafety = ProbeSafety.READ_ONLY,
) -> ProbeFact:
    return ProbeFact(
        name=name,
        component=component,
        status=ProbeStatus.REQUIRES_DIAGNOSTIC,
        source=source,
        detail=detail,
        value=value,
        safety=safety,
    )


def _error_fact(
    name: str,
    component: Literal["host", "env", "vla", "sam3", "diagnostic"],
    exc: Exception,
    *,
    source: str,
    safety: ProbeSafety = ProbeSafety.READ_ONLY,
    detail: str | None = None,
) -> ProbeFact:
    return ProbeFact(
        name=name,
        component=component,
        status=ProbeStatus.ERROR,
        safety=safety,
        source=source,
        error=_error(exc),
        detail=detail,
    )


def _payload_fact(
    name: str,
    component: Literal["env", "vla", "sam3"],
    payload: Mapping[str, Any] | None,
    payload_error: Exception | None,
    *keys: str,
    detail: str,
) -> ProbeFact:
    source = f"{component}_client.runtime_probe"
    if payload_error is not None:
        return _error_fact(name, component, payload_error, source=source)
    if payload is None:
        return _missing_fact(
            name,
            component,
            source=source,
            detail=f"{component} client was not supplied",
        )
    value = _nested(payload, *keys)
    if value is _MISSING or value is None:
        return _missing_fact(name, component, source=source, detail=detail)
    return _observed(name, component, value, source=source)


def _call_runtime_probe(
    client: RuntimeProbeClient | None,
    *,
    argument: Any = _MISSING,
) -> tuple[Mapping[str, Any] | None, Exception | None]:
    if client is None:
        return None, None
    try:
        if argument is _MISSING:
            payload = client.runtime_probe()
        else:
            payload = client.runtime_probe(argument)
        if not isinstance(payload, Mapping):
            raise TypeError("runtime_probe must return a mapping")
        return payload, None
    except Exception as exc:
        return None, exc


def _endpoint_health_fact(
    component: Literal["env", "vla", "sam3"],
    client: Any,
    payload: Mapping[str, Any] | None,
    payload_error: Exception | None,
    health_check: HealthCheck | None,
) -> ProbeFact:
    name = f"{component}.endpoint_health"
    if client is None:
        return _missing_fact(
            name,
            component,
            source="injected_client",
            detail=f"{component} client was not supplied",
        )
    callback = health_check
    if callback is None:
        candidate = getattr(client, "healthz", None)
        if callable(candidate):
            callback = candidate
    if callback is not None:
        try:
            response = callback()
            return _observed(
                name,
                component,
                {"reachable": True, "response": response},
                source="health_check",
            )
        except Exception as exc:
            return _error_fact(name, component, exc, source="health_check")
    if payload_error is not None:
        return _error_fact(
            name,
            component,
            payload_error,
            source=f"{component}_client.runtime_probe",
        )
    if payload is not None:
        return _observed(
            name,
            component,
            {"reachable": True, "evidence": "runtime_probe_response"},
            source=f"{component}_client.runtime_probe",
        )
    return _missing_fact(
        name,
        component,
        source="injected_client",
        detail="no health check or runtime-probe evidence was available",
    )


def _camera_fact(
    payload: Mapping[str, Any] | None,
    payload_error: Exception | None,
) -> ProbeFact:
    name = "env.camera_views"
    source = "env_client.runtime_probe"
    if payload_error is not None:
        return _error_fact(name, "env", payload_error, source=source)
    if payload is None:
        return _missing_fact(
            name, "env", source=source, detail="env client was not supplied"
        )
    fields = payload.get("policy_observation_fields")
    if not isinstance(fields, Mapping):
        return _missing_fact(
            name,
            "env",
            source=source,
            detail="env probe did not describe policy observation fields",
        )
    aliases = {
        "main": "main_images",
        "wrist": "wrist_images",
        "extra": "extra_view_images",
    }
    canonical = {}
    for view, key in aliases.items():
        descriptor = fields.get(key)
        canonical[view] = {
            "observation_key": key,
            "available": bool(
                isinstance(descriptor, Mapping)
                and descriptor.get("available", True)
            ),
            "descriptor": descriptor if isinstance(descriptor, Mapping) else None,
        }
    image_like_keys = sorted(
        str(key)
        for key in fields
        if "image" in str(key).lower() or "rgb" in str(key).lower()
    )
    return _observed(
        name,
        "env",
        {"canonical_views": canonical, "image_like_keys": image_like_keys},
        source=source,
    )


def _env_state_shape_fact(
    payload: Mapping[str, Any] | None,
    payload_error: Exception | None,
) -> ProbeFact:
    name = "env.state_vector_shape"
    source = "env_client.runtime_probe"
    if payload_error is not None:
        return _error_fact(name, "env", payload_error, source=source)
    if payload is None:
        return _missing_fact(
            name, "env", source=source, detail="env client was not supplied"
        )
    descriptor = _nested(payload, "policy_observation_fields", "states")
    if descriptor is _MISSING or not isinstance(descriptor, Mapping):
        return _missing_fact(
            name,
            "env",
            source=source,
            detail="env probe did not describe the states field",
        )
    return _observed(name, "env", descriptor, source=source)


def _env_state_values_fact(
    payload: Mapping[str, Any] | None,
    payload_error: Exception | None,
) -> ProbeFact:
    name = "env.state_field_values"
    source = "env_client.runtime_probe"
    if payload_error is not None:
        return _error_fact(name, "env", payload_error, source=source)
    if payload is None:
        return _missing_fact(
            name, "env", source=source, detail="env client was not supplied"
        )
    sample = payload.get("state_sample")
    provenance = payload.get("state_sample_provenance")
    if sample is None:
        return _missing_fact(
            name,
            "env",
            source=source,
            detail="env probe did not expose a diagnostic state sample",
        )
    return _observed(
        name,
        "env",
        {
            "values": sample,
            "provenance": provenance,
            "policy_eligible": False,
        },
        source=source,
        detail="experiment-only diagnostic; never a policy feature source",
    )


def _observation_keys_fact(
    payload: Mapping[str, Any] | None,
    payload_error: Exception | None,
) -> ProbeFact:
    name = "env.observation_keys"
    source = "env_client.runtime_probe"
    if payload_error is not None:
        return _error_fact(name, "env", payload_error, source=source)
    if payload is None:
        return _missing_fact(
            name, "env", source=source, detail="env client was not supplied"
        )
    policy_fields = payload.get("policy_observation_fields")
    raw_fields = payload.get("raw_observation_fields")
    if not isinstance(policy_fields, Mapping) or not isinstance(raw_fields, Mapping):
        return _missing_fact(
            name,
            "env",
            source=source,
            detail="env probe did not describe both policy and raw observation keys",
        )
    return _observed(
        name,
        "env",
        {
            "policy_keys": sorted(str(key) for key in policy_fields),
            "raw_keys": sorted(str(key) for key in raw_fields),
            "raw_values_included": False,
        },
        source=source,
    )


def _env_config_fact(
    payload: Mapping[str, Any] | None,
    payload_error: Exception | None,
) -> ProbeFact:
    name = "env.controller_environment_config"
    source = "env_client.runtime_probe"
    if payload_error is not None:
        return _error_fact(name, "env", payload_error, source=source)
    if payload is None:
        return _missing_fact(
            name, "env", source=source, detail="env client was not supplied"
        )
    available = {
        key: payload[key]
        for key in ("server_meta", "runtime_meta", "environment_class")
        if key in payload and payload[key] is not None
    }
    if not available:
        return _missing_fact(
            name,
            "env",
            source=source,
            detail="env probe exposed no controller/environment configuration",
        )
    return _observed(name, "env", available, source=source)


def _reset_identity_fact(
    payload: Mapping[str, Any] | None,
    payload_error: Exception | None,
) -> ProbeFact:
    name = "env.reset_identity"
    source = "env_client.runtime_probe"
    if payload_error is not None:
        return _error_fact(name, "env", payload_error, source=source)
    if payload is None:
        return _missing_fact(
            name, "env", source=source, detail="env client was not supplied"
        )
    runtime_meta = payload.get("runtime_meta")
    server_meta = payload.get("server_meta")
    reset_id = (
        runtime_meta.get("reset_id")
        if isinstance(runtime_meta, Mapping)
        else None
    )
    context = {
        key: server_meta[key]
        for key in ("suite", "task", "seed")
        if isinstance(server_meta, Mapping) and key in server_meta
    }
    if reset_id is None:
        return _requires(
            name,
            "env",
            source=source,
            detail="runtime_meta.reset_id is absent; configure/reset with an identity",
            value={"reset_id": None, "context": context},
        )
    return _observed(
        name,
        "env",
        {"reset_id": reset_id, "context": context},
        source=source,
    )


def _versions_fact(
    component: Literal["env", "vla", "sam3"],
    payload: Mapping[str, Any] | None,
    payload_error: Exception | None,
) -> ProbeFact:
    name = f"{component}.versions"
    source = f"{component}_client.runtime_probe"
    if payload_error is not None:
        return _error_fact(name, component, payload_error, source=source)
    if payload is None:
        return _missing_fact(
            name,
            component,
            source=source,
            detail=f"{component} client was not supplied",
        )
    value = {
        key: payload[key]
        for key in ("python", "packages")
        if key in payload and payload[key] is not None
    }
    if not value:
        return _missing_fact(
            name,
            component,
            source=source,
            detail=f"{component} probe exposed no version metadata",
        )
    return _observed(name, component, value, source=source)


def _identity_fact(
    component: Literal["vla", "sam3"],
    payload: Mapping[str, Any] | None,
    payload_error: Exception | None,
) -> ProbeFact:
    name = f"{component}.model_checkpoint_identity"
    source = f"{component}_client.runtime_probe"
    if payload_error is not None:
        return _error_fact(name, component, payload_error, source=source)
    if payload is None:
        return _missing_fact(
            name,
            component,
            source=source,
            detail=f"{component} client was not supplied",
        )
    model_class = payload.get("model_class")
    checkpoint = payload.get("checkpoint")
    value = {
        "model_class": model_class,
        "checkpoint": checkpoint,
    }
    if model_class is None and checkpoint is None:
        return _missing_fact(
            name,
            component,
            source=source,
            detail=f"{component} probe exposed no model/checkpoint identity",
        )
    if model_class is None or checkpoint is None:
        missing = "model_class" if model_class is None else "checkpoint"
        return _requires(
            name,
            component,
            source=source,
            detail=f"{component} probe identity is partial; {missing} is absent",
            value=value,
        )
    return _observed(name, component, value, source=source)


def _actual_action_shape(
    payload: Mapping[str, Any] | None,
    payload_error: Exception | None,
    *,
    inference_requested: bool,
) -> ProbeFact:
    name = "vla.actual_action_shape"
    source = "vla_client.runtime_probe"
    if payload_error is not None:
        return _error_fact(
            name,
            "vla",
            payload_error,
            source=source,
            safety=(
                ProbeSafety.INFERENCE_ONLY
                if inference_requested
                else ProbeSafety.READ_ONLY
            ),
        )
    if payload is None:
        return _missing_fact(
            name, "vla", source=source, detail="VLA client was not supplied"
        )
    shape = _nested(payload, "inference_probe", "action_shape")
    if shape is _MISSING or shape is None:
        shape = payload.get("actual_action_shape")
    if shape is None or shape is _MISSING:
        return _requires(
            name,
            "vla",
            source=source,
            detail=(
                "enable run_vla_inference and supply a deployment-shaped "
                "observation; returned actions are not executed"
            ),
            value={"configured_action_shape": payload.get("configured_action_shape")},
            safety=ProbeSafety.INFERENCE_ONLY,
        )
    return _observed(
        name,
        "vla",
        {"shape": shape, "executed_in_environment": False},
        source=source,
        safety=ProbeSafety.INFERENCE_ONLY,
    )


def _actual_chunk_size(action_shape: ProbeFact) -> ProbeFact:
    name = "vla.actual_chunk_size"
    source = "derived_from:vla.actual_action_shape"
    if action_shape.status is ProbeStatus.ERROR:
        assert action_shape.error is not None
        return ProbeFact(
            name=name,
            component="vla",
            status=ProbeStatus.ERROR,
            safety=action_shape.safety,
            source=source,
            error=action_shape.error,
        )
    if action_shape.status is ProbeStatus.UNAVAILABLE:
        return _missing_fact(
            name,
            "vla",
            source=source,
            detail="actual action shape is unavailable",
        )
    if action_shape.status is ProbeStatus.REQUIRES_DIAGNOSTIC:
        return _requires(
            name,
            "vla",
            source=source,
            detail="actual chunk size requires the non-executed VLA inference probe",
            value=action_shape.value,
            safety=ProbeSafety.INFERENCE_ONLY,
        )
    try:
        value = action_shape.value
        if not isinstance(value, Mapping):
            raise TypeError("actual action-shape fact is not a mapping")
        shape = value.get("shape")
        if (
            not isinstance(shape, Sequence)
            or isinstance(shape, (str, bytes))
            or len(shape) not in {2, 3}
        ):
            raise ValueError(f"unexpected VLA action shape: {shape!r}")
        dimensions = [int(item) for item in shape]
        if any(item <= 0 for item in dimensions):
            raise ValueError(f"VLA action shape must be positive: {dimensions}")
        chunk_size = dimensions[-2]
        return _observed(
            name,
            "vla",
            {"chunk_size": chunk_size, "action_shape": dimensions},
            source=source,
            safety=ProbeSafety.INFERENCE_ONLY,
        )
    except Exception as exc:
        return _error_fact(
            name,
            "vla",
            exc,
            source=source,
            safety=ProbeSafety.INFERENCE_ONLY,
        )


def _sam_acceptance_fact(
    sam3_client: Sam3ProbeClient | None,
    *,
    run_inference: bool,
    image: Any,
    text_prompt: str | None,
) -> ProbeFact:
    name = "sam3.current_observation_acceptance"
    source = "sam3_client.segment_image"
    if sam3_client is None:
        return _missing_fact(
            name, "sam3", source=source, detail="SAM3 client was not supplied"
        )
    if not run_inference:
        return _requires(
            name,
            "sam3",
            source=source,
            detail=(
                "enable run_sam3_inference and supply the current RGB observation "
                "plus a text prompt"
            ),
            safety=ProbeSafety.INFERENCE_ONLY,
        )
    if image is None or not isinstance(text_prompt, str) or not text_prompt.strip():
        return _error_fact(
            name,
            "sam3",
            ValueError("SAM3 inference requires an image and non-empty text prompt"),
            source=source,
            safety=ProbeSafety.INFERENCE_ONLY,
        )
    try:
        result = sam3_client.segment_image(image, text_prompt=text_prompt.strip())
        if isinstance(result, Mapping):
            getter = result.get
        else:
            getter = lambda key, default=None: getattr(result, key, default)
        summary = {
            "accepted_current_observation": True,
            "found": bool(getter("found", False)),
            "score": getter("score"),
            "mask_shape": getter("mask_shape"),
            "reason": getter("reason"),
            "environment_advanced": False,
        }
        return _observed(
            name,
            "sam3",
            summary,
            source=source,
            safety=ProbeSafety.INFERENCE_ONLY,
        )
    except Exception as exc:
        return _error_fact(
            name,
            "sam3",
            exc,
            source=source,
            safety=ProbeSafety.INFERENCE_ONLY,
        )


def _to_list(value: Any) -> Any:
    converter = getattr(value, "tolist", None)
    return converter() if callable(converter) else value


def _infer_shape(value: Any) -> list[int]:
    explicit = getattr(value, "shape", None)
    if explicit is not None:
        return [int(item) for item in explicit]
    result: list[int] = []
    current = value
    while isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
        result.append(len(current))
        if not current:
            break
        current = current[0]
    return result


def _array_summary(value: Any, *, include_values: bool) -> dict[str, Any]:
    converted = _to_list(value)
    result: dict[str, Any] = {
        "shape": _infer_shape(value),
        "dtype": str(getattr(value, "dtype", type(value).__name__)),
    }
    if include_values:
        _validate_json(converted, name="diagnostic array values")
        result["values"] = converted
    return result


def _first_true_index(value: Any) -> int | None:
    converted = _to_list(value)
    if not isinstance(converted, Sequence) or isinstance(converted, (str, bytes)):
        converted = [converted]
    for index, item in enumerate(converted):
        if bool(item):
            return index
    return None


def _sequence_length(value: Any) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


def _chunk_diagnostic_facts(
    env_client: EnvDiagnosticClient | None,
    *,
    enabled: bool,
    actions: Any,
) -> tuple[ProbeFact, ProbeFact]:
    arrays_name = "env.termination_truncation_arrays"
    behavior_name = "diagnostic.chunk_done_mid_chunk_behavior"
    source = "env_client.diagnostic_chunk_step"
    if env_client is None:
        unavailable = _missing_fact(
            arrays_name,
            "env",
            source=source,
            detail="env client was not supplied",
        )
        behavior = _missing_fact(
            behavior_name,
            "diagnostic",
            source=source,
            detail="env client was not supplied",
        )
        return unavailable, behavior
    if not enabled:
        arrays = _requires(
            arrays_name,
            "env",
            source=source,
            detail=(
                "termination/truncation array shape requires an explicitly "
                "authorized throwaway chunk diagnostic"
            ),
            safety=ProbeSafety.DESTRUCTIVE_ENVIRONMENT,
        )
        behavior = _requires(
            behavior_name,
            "diagnostic",
            source=source,
            detail=(
                "mid-chunk done behavior cannot be established by introspection; "
                "run a targeted fresh-reset throwaway trial"
            ),
            safety=ProbeSafety.DESTRUCTIVE_ENVIRONMENT,
        )
        return arrays, behavior
    if actions is None:
        exc = ValueError("chunk diagnostic actions were not supplied")
        return (
            _error_fact(
                arrays_name,
                "env",
                exc,
                source=source,
                safety=ProbeSafety.DESTRUCTIVE_ENVIRONMENT,
            ),
            _error_fact(
                behavior_name,
                "diagnostic",
                exc,
                source=source,
                safety=ProbeSafety.DESTRUCTIVE_ENVIRONMENT,
            ),
        )
    diagnostic_method = getattr(env_client, "diagnostic_chunk_step", None)
    if not callable(diagnostic_method):
        exc = AttributeError("env client has no diagnostic_chunk_step method")
        return (
            _error_fact(
                arrays_name,
                "env",
                exc,
                source=source,
                safety=ProbeSafety.DESTRUCTIVE_ENVIRONMENT,
            ),
            _error_fact(
                behavior_name,
                "diagnostic",
                exc,
                source=source,
                safety=ProbeSafety.DESTRUCTIVE_ENVIRONMENT,
            ),
        )
    try:
        result = diagnostic_method(actions)
        if not isinstance(result, tuple) or len(result) != 5:
            raise TypeError("chunk diagnostic must return a five-item tuple")
        observations, _reward, terminated, truncated, _info = result
        term_summary = _array_summary(terminated, include_values=True)
        trunc_summary = _array_summary(truncated, include_values=True)
        arrays = _observed(
            arrays_name,
            "env",
            {"terminated": term_summary, "truncated": trunc_summary},
            source=source,
            safety=ProbeSafety.DESTRUCTIVE_ENVIRONMENT,
        )

        requested_actions = _sequence_length(actions)
        observations_returned = _sequence_length(observations)
        term_index = _first_true_index(terminated)
        trunc_index = _first_true_index(truncated)
        done_indices = [index for index in (term_index, trunc_index) if index is not None]
        first_done = min(done_indices) if done_indices else None
        entries_after_done = (
            max(0, observations_returned - first_done - 1)
            if observations_returned is not None and first_done is not None
            else None
        )
        evidence = {
            "requested_actions": requested_actions,
            "observations_returned": observations_returned,
            "first_terminated_index": term_index,
            "first_truncated_index": trunc_index,
            "first_done_index": first_done,
            "rpc_entries_returned_after_first_done": entries_after_done,
            "physical_step_continuation_established": False,
        }
        if first_done is None:
            behavior = _requires(
                behavior_name,
                "diagnostic",
                source=source,
                detail=(
                    "diagnostic chunk did not trigger done; use a controlled trial "
                    "that triggers termination before its final action"
                ),
                value=evidence,
                safety=ProbeSafety.DESTRUCTIVE_ENVIRONMENT,
            )
        else:
            behavior = _requires(
                behavior_name,
                "diagnostic",
                source=source,
                detail=(
                    "RPC return behavior was observed, but returned frame/vector "
                    "lengths cannot distinguish physical post-done stepping from "
                    "external padding; inspect/instrument the installed RLinf runtime"
                ),
                value=evidence,
                safety=ProbeSafety.DESTRUCTIVE_ENVIRONMENT,
            )
        return arrays, behavior
    except Exception as exc:
        return (
            _error_fact(
                arrays_name,
                "env",
                exc,
                source=source,
                safety=ProbeSafety.DESTRUCTIVE_ENVIRONMENT,
            ),
            _error_fact(
                behavior_name,
                "diagnostic",
                exc,
                source=source,
                safety=ProbeSafety.DESTRUCTIVE_ENVIRONMENT,
            ),
        )


def _hidden_state_fact(
    vla_client: Any,
    *,
    enabled: bool,
    diagnostic: HiddenStateDiagnostic | None,
) -> ProbeFact:
    name = "diagnostic.vla_hidden_episode_state"
    source = "injected_hidden_state_diagnostic"
    if vla_client is None:
        return _missing_fact(
            name,
            "diagnostic",
            source=source,
            detail="VLA client was not supplied",
        )
    if not enabled:
        return _requires(
            name,
            "diagnostic",
            source=source,
            detail=(
                "hidden episode state requires an isolated controlled protocol; "
                "repeated identical outputs alone cannot separate state from "
                "stochasticity"
            ),
            safety=ProbeSafety.ISOLATED_MODEL_DIAGNOSTIC,
        )
    if diagnostic is None:
        return _error_fact(
            name,
            "diagnostic",
            ValueError("hidden-state diagnostic callback was not supplied"),
            source=source,
            safety=ProbeSafety.ISOLATED_MODEL_DIAGNOSTIC,
        )
    try:
        raw_result = diagnostic(vla_client)
        result = (
            raw_result
            if isinstance(raw_result, HiddenStateDiagnosticResult)
            else HiddenStateDiagnosticResult.model_validate(raw_result)
        )
        if result.conclusion is HiddenStateConclusion.INCONCLUSIVE:
            return _requires(
                name,
                "diagnostic",
                source=source,
                detail="controlled hidden-state diagnostic was inconclusive",
                value=result.model_dump(mode="json"),
                safety=ProbeSafety.ISOLATED_MODEL_DIAGNOSTIC,
            )
        return _observed(
            name,
            "diagnostic",
            result.model_dump(mode="json"),
            source=source,
            safety=ProbeSafety.ISOLATED_MODEL_DIAGNOSTIC,
        )
    except Exception as exc:
        return _error_fact(
            name,
            "diagnostic",
            exc,
            source=source,
            safety=ProbeSafety.ISOLATED_MODEL_DIAGNOSTIC,
        )


def run_runtime_probes(
    *,
    env_client: EnvDiagnosticClient | None = None,
    vla_client: RuntimeProbeClient | None = None,
    sam3_client: Sam3ProbeClient | None = None,
    options: RuntimeProbeOptions | None = None,
    vla_probe_observation: Mapping[str, Any] | None = None,
    sam3_probe_image: Any = None,
    sam3_text_prompt: str | None = None,
    chunk_diagnostic_actions: Any = None,
    health_checks: Mapping[str, HealthCheck] | None = None,
    host_gpu_probe: GpuProbe | None = None,
    hidden_state_diagnostic: HiddenStateDiagnostic | None = None,
    local_version_probe: Callable[[], Mapping[str, Any]] = probe_local_versions,
) -> RuntimeProbeReport:
    """Collect a complete, structured ledger from injected runtime clients.

    The default path is read-only.  ``run_vla_inference`` and
    ``run_sam3_inference`` opt into inference that does not step the env; VLA
    inference also requires an isolated model session because statefulness is
    not assumed either way.
    ``run_destructive_chunk_diagnostic`` additionally requires fresh-reset and
    isolated-trial confirmations.  Hidden-state diagnostics require a caller-
    supplied controlled protocol and isolated model-session confirmation.
    """

    resolved_options = options or RuntimeProbeOptions()
    health = dict(health_checks or {})
    facts: dict[str, ProbeFact] = {}

    try:
        facts["host.versions"] = _observed(
            "host.versions",
            "host",
            local_version_probe(),
            source="local_version_probe",
        )
    except Exception as exc:
        facts["host.versions"] = _error_fact(
            "host.versions", "host", exc, source="local_version_probe"
        )

    if resolved_options.run_host_gpu_discovery:
        gpu_callback = host_gpu_probe or probe_nvidia_smi
        try:
            facts["host.cuda_gpu"] = _observed(
                "host.cuda_gpu",
                "host",
                gpu_callback(),
                source=(
                    "injected_host_gpu_probe"
                    if host_gpu_probe is not None
                    else "nvidia-smi"
                ),
            )
        except Exception as exc:
            facts["host.cuda_gpu"] = _error_fact(
                "host.cuda_gpu", "host", exc, source="host_gpu_probe"
            )
    else:
        facts["host.cuda_gpu"] = _requires(
            "host.cuda_gpu",
            "host",
            source="host_gpu_probe",
            detail="enable run_host_gpu_discovery to query CUDA device memory",
        )

    env_payload, env_error = _call_runtime_probe(env_client)
    vla_argument = (
        vla_probe_observation
        if resolved_options.run_vla_inference
        else _MISSING
    )
    if resolved_options.run_vla_inference and vla_probe_observation is None:
        vla_payload = None
        vla_error: Exception | None = ValueError(
            "run_vla_inference requires vla_probe_observation"
        )
    else:
        vla_payload, vla_error = _call_runtime_probe(
            vla_client, argument=vla_argument
        )
    sam_payload, sam_error = _call_runtime_probe(sam3_client)

    facts["env.endpoint_health"] = _endpoint_health_fact(
        "env",
        env_client,
        env_payload,
        env_error,
        health.get("env"),
    )
    facts["env.versions"] = _versions_fact("env", env_payload, env_error)
    facts["env.observation_keys"] = _observation_keys_fact(env_payload, env_error)
    facts["env.state_vector_shape"] = _env_state_shape_fact(env_payload, env_error)
    facts["env.state_field_values"] = _env_state_values_fact(env_payload, env_error)
    facts["env.camera_views"] = _camera_fact(env_payload, env_error)
    facts["env.controller_environment_config"] = _env_config_fact(
        env_payload, env_error
    )
    facts["env.cuda_gpu"] = _payload_fact(
        "env.cuda_gpu",
        "env",
        env_payload,
        env_error,
        "cuda",
        detail="env probe exposed no CUDA/GPU metadata",
    )
    facts["env.reset_identity"] = _reset_identity_fact(env_payload, env_error)

    term_fact, chunk_behavior = _chunk_diagnostic_facts(
        env_client,
        enabled=resolved_options.run_destructive_chunk_diagnostic,
        actions=chunk_diagnostic_actions,
    )
    facts["env.termination_truncation_arrays"] = term_fact
    facts["diagnostic.chunk_done_mid_chunk_behavior"] = chunk_behavior

    facts["vla.endpoint_health"] = _endpoint_health_fact(
        "vla",
        vla_client,
        vla_payload,
        vla_error,
        health.get("vla"),
    )
    facts["vla.versions"] = _versions_fact("vla", vla_payload, vla_error)
    facts["vla.configured_action_shape"] = _payload_fact(
        "vla.configured_action_shape",
        "vla",
        vla_payload,
        vla_error,
        "configured_action_shape",
        detail="VLA probe exposed no configured action shape",
    )
    action_shape = _actual_action_shape(
        vla_payload,
        vla_error,
        inference_requested=resolved_options.run_vla_inference,
    )
    facts["vla.actual_action_shape"] = action_shape
    facts["vla.actual_chunk_size"] = _actual_chunk_size(action_shape)
    facts["vla.controller_config"] = _payload_fact(
        "vla.controller_config",
        "vla",
        vla_payload,
        vla_error,
        "model_config",
        detail="VLA probe exposed no model/controller configuration",
    )
    facts["vla.cuda_gpu"] = _payload_fact(
        "vla.cuda_gpu",
        "vla",
        vla_payload,
        vla_error,
        "cuda",
        detail="VLA probe exposed no CUDA/GPU metadata",
    )
    facts["vla.model_checkpoint_identity"] = _identity_fact(
        "vla", vla_payload, vla_error
    )

    facts["sam3.endpoint_health"] = _endpoint_health_fact(
        "sam3",
        sam3_client,
        sam_payload,
        sam_error,
        health.get("sam3"),
    )
    facts["sam3.versions"] = _versions_fact("sam3", sam_payload, sam_error)
    facts["sam3.current_observation_contract"] = _payload_fact(
        "sam3.current_observation_contract",
        "sam3",
        sam_payload,
        sam_error,
        "image_contract",
        detail="SAM3 probe exposed no current-observation image contract",
    )
    facts["sam3.current_observation_acceptance"] = _sam_acceptance_fact(
        sam3_client,
        run_inference=resolved_options.run_sam3_inference,
        image=sam3_probe_image,
        text_prompt=sam3_text_prompt,
    )
    facts["sam3.cuda_gpu"] = _payload_fact(
        "sam3.cuda_gpu",
        "sam3",
        sam_payload,
        sam_error,
        "cuda",
        detail="SAM3 probe exposed no CUDA/GPU metadata",
    )
    facts["sam3.model_checkpoint_identity"] = _identity_fact(
        "sam3", sam_payload, sam_error
    )

    facts["diagnostic.vla_hidden_episode_state"] = _hidden_state_fact(
        vla_client,
        enabled=resolved_options.run_hidden_state_diagnostic,
        diagnostic=hidden_state_diagnostic,
    )

    return RuntimeProbeReport(
        options=resolved_options,
        facts=tuple(facts[name] for name in REQUIRED_RUNTIME_FACTS),
    )


__all__ = [
    "EnvDiagnosticClient",
    "GpuProbe",
    "HealthCheck",
    "HiddenStateConclusion",
    "HiddenStateDiagnostic",
    "HiddenStateDiagnosticResult",
    "ProbeError",
    "ProbeFact",
    "ProbeSafety",
    "ProbeStatus",
    "REQUIRED_RUNTIME_FACTS",
    "RUNTIME_PROBE_SCHEMA_VERSION",
    "RuntimeProbeClient",
    "RuntimeProbeOptions",
    "RuntimeProbeReport",
    "Sam3ProbeClient",
    "probe_local_versions",
    "probe_nvidia_smi",
    "run_runtime_probes",
]
