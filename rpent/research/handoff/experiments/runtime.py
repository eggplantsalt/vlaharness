"""Runtime-only helpers for isolated controller-handoff experiment children.

Importing this module does not import LIBERO, MuJoCo, CUDA, SAM3, Pi0.5, or
the trainable model stack.  Those dependencies are resolved only inside the
explicit child entry points.

Gate-0 adapter factories use the ``module:callable`` contract.  The callable
is invoked with keyword arguments ``job``, ``adapter_config``,
``gate0_config``, and ``output_dir``.  It may return the adapter directly or a
mapping with an ``adapter`` entry and optional ``setup_sink`` and ``cleanup``
entries.  ``cleanup`` must be a zero-argument callable.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Self

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.experiments.config import RuntimeConfig
from rpent.research.handoff.experiments.probes import (
    ProbeStatus,
    RuntimeProbeArtifact,
)
from rpent.research.handoff.types import HandoffRecord


GATE0_JOB_SCHEMA_VERSION = "rpent.handoff-gate0-job/v2"
CONTROLLED_CHILD_PLAN_SCHEMA_VERSION = "rpent.handoff-controlled-plan/v1"
EXECUTION_CONFIRMATION = "I_UNDERSTAND_SERVER_EXECUTION"
_GATE0_REQUIRED_PROBE_FACTS = frozenset(
    {
        "env.endpoint_health",
        "env.reset_identity",
        "vla.endpoint_health",
        "vla.model_checkpoint_identity",
        "sam3.endpoint_health",
        "sam3.model_checkpoint_identity",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Gate0RuntimeProbeReference(HandoffRecord):
    """One probe file and the facts Gate-0 requires from that file."""

    name: str
    path: str
    required_observed_facts: tuple[str, ...]

    @field_validator("name", "path")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("required_observed_facts")
    @classmethod
    def validate_facts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item for item in value):
            raise ValueError("a Gate-0 probe reference needs non-empty fact names")
        if len(value) != len(set(value)):
            raise ValueError("Gate-0 probe fact names must be unique")
        return value


class Gate0RuntimeProbeBinding(HandoffRecord):
    """Resolved, byte-bound probe evidence embedded in a Gate-0 job."""

    name: str
    resolved_path: str
    sha256: str
    required_observed_facts: tuple[str, ...]
    artifact: RuntimeProbeArtifact

    @field_validator("name", "resolved_path", "sha256")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value


class Gate0ExternalBindings(HandoffRecord):
    """All mutable files that define one resolved Gate-0 job."""

    source_job_path: str
    source_job_sha256: str
    handoff_config_path: str
    handoff_config_sha256: str
    handoff_canonical_sha256: str
    handoff_configuration_id: str
    handoff_controller_configuration_id: str
    handoff_controller_method: str
    handoff_controller_implementation_version: str
    handoff_checkpoint_id: str | None = None
    handoff_metadata_pi05_checkpoint_id: str | None = None
    handoff_metadata_sam3_checkpoint_id: str | None = None
    runtime_probes: tuple[Gate0RuntimeProbeBinding, ...]

    @field_validator(
        "source_job_path",
        "source_job_sha256",
        "handoff_config_path",
        "handoff_config_sha256",
        "handoff_canonical_sha256",
        "handoff_configuration_id",
        "handoff_controller_configuration_id",
        "handoff_controller_method",
        "handoff_controller_implementation_version",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_probe_names(self) -> Self:
        names = [binding.name for binding in self.runtime_probes]
        if not names:
            raise ValueError("Gate-0 needs content-bound runtime probe artifacts")
        if len(names) != len(set(names)):
            raise ValueError("Gate-0 runtime probe names must be unique")
        return self


def _finite_json_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values only") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # defensive; the input is a Mapping
        raise ValueError(f"{name} must be a JSON object")
    return decoded


def _factory_path(value: str, name: str) -> str:
    module_name, separator, attribute = value.strip().partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"{name} must use the 'module:callable' form")
    return value.strip()


class Gate0JobSpec(HandoffRecord):
    """Strict, portable Gate-0 collection job.

    The nested ``gate0`` object is validated as ``Gate0Config`` only in the
    execution child, keeping offline CLI import free of NumPy/runtime modules.
    """

    schema_version: Literal[GATE0_JOB_SCHEMA_VERSION] = GATE0_JOB_SCHEMA_VERSION
    output_dir: str
    adapter_factory: str
    adapter_config: dict[str, Any] = Field(default_factory=dict)
    gate0: dict[str, Any]
    vla_kwargs: dict[str, Any] = Field(default_factory=dict)
    run_id: str
    episode_prefix: str = "gate0"
    suite: str
    task_id: int | str
    seed: int = Field(ge=0)
    target_id: str
    target_description: str
    skill_name: str
    skill_prompt: str
    controller_method: str
    controller_implementation_version: str = "rpent-gate0/v1"
    checkpoint_id: str
    configuration_id: str | None = None
    source_revision: str
    runtime_probes: tuple[Gate0RuntimeProbeReference, ...] = ()
    external_bindings: Gate0ExternalBindings | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "output_dir",
        "run_id",
        "episode_prefix",
        "suite",
        "target_id",
        "target_description",
        "skill_name",
        "skill_prompt",
        "controller_method",
        "controller_implementation_version",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("adapter_factory")
    @classmethod
    def validate_factory(cls, value: str) -> str:
        return _factory_path(value, "adapter_factory")

    @field_validator("adapter_config", "gate0", "vla_kwargs", "metadata")
    @classmethod
    def validate_mapping(cls, value: dict[str, Any], info) -> dict[str, Any]:
        resolved = _finite_json_mapping(value, info.field_name)
        if info.field_name == "metadata":
            reserved = sorted(
                {
                    "execution_plan_id",
                    "runtime_attestation_id",
                    "runtime_attestation_sha256",
                }.intersection(resolved)
            )
            if reserved:
                raise ValueError(
                    "Gate-0 metadata cannot set execution-reserved fields: "
                    + ", ".join(reserved)
                )
        if info.field_name == "vla_kwargs":
            protected = sorted(
                {"prompt", "target_id", "target_description"}.intersection(
                    resolved
                )
            )
            if protected:
                raise ValueError(
                    "vla_kwargs cannot override manifest-bound fields: "
                    + ", ".join(protected)
                )
            if "max_chunks" in resolved:
                max_chunks = resolved["max_chunks"]
                if (
                    isinstance(max_chunks, bool)
                    or not isinstance(max_chunks, int)
                    or max_chunks < 1
                ):
                    raise ValueError(
                        "vla_kwargs.max_chunks must be a positive integer"
                    )
        return resolved

    @field_validator("configuration_id")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is not None and not value.strip():
            raise ValueError(f"{info.field_name} must be null or non-empty")
        return value

    @field_validator("checkpoint_id", "source_revision")
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_runtime_identity(self) -> Self:
        runtime_value = self.adapter_config.get("runtime")
        if not isinstance(runtime_value, dict):
            raise ValueError("adapter_config.runtime is required for Gate-0")
        runtime = RuntimeConfig.model_validate(runtime_value)
        for name in ("pi05_checkpoint_id", "sam3_checkpoint_id"):
            if getattr(runtime, name) is None:
                raise ValueError(f"adapter_config.runtime.{name} is required")
        for endpoint_name, path_name in (
            ("vla_endpoint", "pi05_checkpoint_path"),
            ("sam3_endpoint", "sam3_checkpoint_path"),
        ):
            if (
                getattr(runtime, endpoint_name) is None
                and getattr(runtime, path_name) is None
            ):
                raise ValueError(
                    f"local Gate-0 runtime requires adapter_config.runtime.{path_name}"
                )
        if runtime.pi05_checkpoint_id != self.checkpoint_id:
            raise ValueError(
                "Gate-0 checkpoint_id disagrees with runtime Pi0.5 checkpoint ID"
            )
        if self.external_bindings is not None:
            handoff_path = self.adapter_config.get("handoff_config")
            if handoff_path != self.external_bindings.handoff_config_path:
                raise ValueError(
                    "adapter handoff config path disagrees with external binding"
                )
            optional_id = self.external_bindings.handoff_checkpoint_id
            if (
                self.external_bindings.handoff_controller_method
                != self.controller_method
            ):
                raise ValueError(
                    "handoff controller method disagrees with Gate-0 job"
                )
            if optional_id is not None and optional_id != self.checkpoint_id:
                raise ValueError(
                    "handoff config checkpoint_id disagrees with Gate-0 job"
                )
            metadata_pi05 = (
                self.external_bindings.handoff_metadata_pi05_checkpoint_id
            )
            metadata_sam3 = (
                self.external_bindings.handoff_metadata_sam3_checkpoint_id
            )
            if metadata_pi05 not in (None, runtime.pi05_checkpoint_id):
                raise ValueError(
                    "handoff metadata Pi0.5 checkpoint ID disagrees with Gate-0"
                )
            if metadata_sam3 not in (None, runtime.sam3_checkpoint_id):
                raise ValueError(
                    "handoff metadata SAM3 checkpoint ID disagrees with Gate-0"
                )
        return self

    @property
    def stable_configuration_id(self) -> str:
        if self.external_bindings is None:
            raise ValueError(
                "Gate-0 configuration identity requires load_gate0_job() to bind "
                "the handoff config and runtime probes"
            )
        scientific = self.model_dump(
            mode="json",
            exclude={
                "configuration_id",
                "output_dir",
                "run_id",
                "episode_prefix",
                "metadata",
            },
        )
        computed = "gate0-" + hashlib.sha256(
            json.dumps(
                scientific,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()[:20]
        if self.configuration_id is not None and self.configuration_id != computed:
            raise ValueError(
                "configured Gate-0 configuration_id does not bind the resolved job"
            )
        return computed


def _resolved_reference(value: str, *, source: Path, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} not found: {resolved}")
    return resolved


def _handoff_binding(path: Path) -> dict[str, Any]:
    """Parse the handoff config and return both byte and canonical identities."""
    from robots.libero.handoff_runtime import load_handoff_runtime_config

    handoff = load_handoff_runtime_config(path)
    canonical = json.dumps(
        handoff.canonical_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    optional_metadata_ids: dict[str, str | None] = {}
    for key, output_key in (
        ("pi05_checkpoint_id", "handoff_metadata_pi05_checkpoint_id"),
        ("sam3_checkpoint_id", "handoff_metadata_sam3_checkpoint_id"),
    ):
        value = handoff.metadata.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"handoff metadata {key} must be a non-empty string")
        optional_metadata_ids[output_key] = value
    return {
        "handoff_config_path": str(path),
        "handoff_config_sha256": _sha256_file(path),
        "handoff_canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "handoff_configuration_id": handoff.configuration_id,
        "handoff_controller_configuration_id": (
            handoff.controller_configuration_id
        ),
        "handoff_controller_method": handoff.controller_method,
        "handoff_controller_implementation_version": (
            handoff.controller_implementation_version
        ),
        "handoff_checkpoint_id": handoff.checkpoint_id,
        **optional_metadata_ids,
    }


def _validate_probe_binding(
    reference: Gate0RuntimeProbeReference,
    *,
    path: Path,
    runtime: RuntimeConfig,
) -> Gate0RuntimeProbeBinding:
    from rpent.research.handoff.experiments.config import load_strict_json

    artifact = RuntimeProbeArtifact.model_validate(load_strict_json(path))
    if not artifact.readiness_ok or not artifact.ok:
        raise ValueError(f"Gate-0 runtime probe is not ready: {reference.name}")
    if artifact.checkpoint_identity_mismatches:
        raise ValueError(
            f"Gate-0 runtime probe reports checkpoint mismatches: {reference.name}"
        )
    if not set(reference.required_observed_facts).issubset(
        artifact.required_observed_facts
    ):
        raise ValueError(
            f"Gate-0 probe {reference.name!r} was not captured with all of its "
            "job-required facts declared as required"
        )
    facts = {fact.name: fact for fact in artifact.report.facts}
    unknown = sorted(set(reference.required_observed_facts).difference(facts))
    if unknown:
        raise ValueError(
            f"Gate-0 probe {reference.name!r} names unknown facts: {unknown}"
        )
    missing = sorted(
        name
        for name in reference.required_observed_facts
        if facts[name].status is not ProbeStatus.OBSERVED
    )
    if missing:
        raise ValueError(
            f"Gate-0 probe {reference.name!r} lacks observed facts: {missing}"
        )
    expected_ids = {
        "vla": runtime.pi05_checkpoint_id,
        "sam3": runtime.sam3_checkpoint_id,
    }
    expected_paths = {
        "vla": runtime.pi05_checkpoint_path,
        "sam3": runtime.sam3_checkpoint_path,
    }
    external = {
        "vla": runtime.vla_endpoint is not None,
        "sam3": runtime.sam3_endpoint is not None,
    }
    for component in ("vla", "sam3"):
        fact = facts[f"{component}.model_checkpoint_identity"]
        if fact.status is not ProbeStatus.OBSERVED:
            continue
        if not isinstance(fact.value, Mapping):
            raise ValueError(
                f"{component} checkpoint fact in {reference.name!r} is not an object"
            )
        checkpoint = fact.value.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ValueError(
                f"{component} checkpoint fact in {reference.name!r} lacks metadata"
            )
        if checkpoint.get("exists") is not True:
            raise ValueError(
                f"{component} checkpoint in {reference.name!r} does not exist"
            )
        observed_id = checkpoint.get("configured_id")
        if observed_id != expected_ids[component]:
            raise ValueError(
                f"{component} checkpoint ID in {reference.name!r} disagrees "
                "with the Gate-0 runtime"
            )
        observed_path = checkpoint.get("path")
        if not isinstance(observed_path, str) or not observed_path:
            raise ValueError(
                f"{component} checkpoint path in {reference.name!r} is missing"
            )
        if not external[component]:
            configured_path = expected_paths[component]
            if configured_path is None or (
                Path(observed_path).expanduser().resolve()
                != Path(configured_path).expanduser().resolve()
            ):
                raise ValueError(
                    f"{component} checkpoint path in {reference.name!r} disagrees "
                    "with the local Gate-0 runtime"
                )
    return Gate0RuntimeProbeBinding(
        name=reference.name,
        resolved_path=str(path),
        sha256=_sha256_file(path),
        required_observed_facts=reference.required_observed_facts,
        artifact=artifact,
    )


def _resolved_gate0_bindings(
    job: Gate0JobSpec,
    *,
    source: Path,
) -> tuple[dict[str, Any], Gate0ExternalBindings]:
    adapter = dict(job.adapter_config)
    handoff_value = adapter.get("handoff_config")
    if not isinstance(handoff_value, str) or not handoff_value:
        raise ValueError("adapter_config.handoff_config is required for Gate-0")
    handoff_path = _resolved_reference(
        handoff_value,
        source=source,
        name="Gate-0 handoff config",
    )
    runtime = RuntimeConfig.model_validate(adapter["runtime"])
    runtime_updates: dict[str, str] = {}
    for field_name in ("pi05_checkpoint_path", "sam3_checkpoint_path"):
        value = getattr(runtime, field_name)
        if value is None:
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = source.parent / candidate
        runtime_updates[field_name] = str(candidate.resolve())
    if runtime_updates:
        runtime = runtime.model_copy(update=runtime_updates)
    adapter["runtime"] = runtime.model_dump(mode="json", exclude_none=False)
    adapter["handoff_config"] = str(handoff_path)

    resolved_references: list[Gate0RuntimeProbeReference] = []
    probe_bindings: list[Gate0RuntimeProbeBinding] = []
    for reference in job.runtime_probes:
        probe_path = _resolved_reference(
            reference.path,
            source=source,
            name=f"Gate-0 runtime probe {reference.name!r}",
        )
        resolved = reference.model_copy(update={"path": str(probe_path)})
        resolved_references.append(resolved)
        probe_bindings.append(
            _validate_probe_binding(resolved, path=probe_path, runtime=runtime)
        )
    required = {
        fact
        for reference in resolved_references
        for fact in reference.required_observed_facts
    }
    missing_required = sorted(_GATE0_REQUIRED_PROBE_FACTS.difference(required))
    if missing_required:
        raise ValueError(
            "Gate-0 runtime probes do not bind required observed facts: "
            + ", ".join(missing_required)
        )
    observed_checkpoint_components = {
        component
        for component in ("vla", "sam3")
        if any(
            binding.artifact.report.fact(
                f"{component}.model_checkpoint_identity"
            ).status
            is ProbeStatus.OBSERVED
            for binding in probe_bindings
        )
    }
    if observed_checkpoint_components != {"vla", "sam3"}:
        raise ValueError(
            "Gate-0 probes need observed Pi0.5 and SAM3 checkpoint identities"
        )
    bindings = Gate0ExternalBindings(
        source_job_path=str(source),
        source_job_sha256=_sha256_file(source),
        runtime_probes=tuple(probe_bindings),
        **_handoff_binding(handoff_path),
    )
    return {
        "adapter_config": adapter,
        "runtime_probes": tuple(resolved_references),
    }, bindings


def load_gate0_job(path: str | os.PathLike[str]) -> Gate0JobSpec:
    """Load a Gate-0 job and bind every mutable execution input by bytes."""
    from rpent.research.handoff.experiments.config import load_strict_json

    unresolved = re.compile(
        r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*|%[A-Za-z_][A-Za-z0-9_]*%"
    )

    def expand(value: Any) -> Any:
        if isinstance(value, str):
            resolved = os.path.expandvars(value)

            def replace_percent(match: re.Match[str]) -> str:
                variable = match.group(0)[1:-1]
                return os.environ.get(variable, match.group(0))

            resolved = re.sub(
                r"%[A-Za-z_][A-Za-z0-9_]*%",
                replace_percent,
                resolved,
            )
            match = unresolved.search(resolved)
            if match is not None:
                raise ValueError(
                    "unresolved environment variable "
                    f"{match.group(0)!r} in Gate-0 job value {value!r}"
                )
            return resolved
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    source = Path(path).expanduser().resolve()
    job = Gate0JobSpec.model_validate(expand(load_strict_json(source)))
    updates, bindings = _resolved_gate0_bindings(job, source=source)
    output = Path(job.output_dir).expanduser()
    if not output.is_absolute():
        output = source.parent / output
    resolved = job.model_copy(
        update={
            **updates,
            "output_dir": str(output.resolve()),
            "external_bindings": bindings,
        }
    )
    # Force the final model-level cross-checks and configured-ID assertion.
    Gate0JobSpec.model_validate(resolved.model_dump(mode="json", exclude_none=False))
    resolved.stable_configuration_id
    return resolved


def gate0_runtime_environment(job: Gate0JobSpec) -> dict[str, str]:
    """Return the four checkpoint variables bound into the child plan."""
    runtime = RuntimeConfig.model_validate(job.adapter_config["runtime"])
    values = {
        "PI05_CHECKPOINT_PATH": runtime.pi05_checkpoint_path,
        "SAM3_CHECKPOINT_PATH": runtime.sam3_checkpoint_path,
        "RPENT_PI05_CHECKPOINT_ID": runtime.pi05_checkpoint_id,
        "RPENT_SAM3_CHECKPOINT_ID": runtime.sam3_checkpoint_id,
    }
    missing = sorted(name for name, value in values.items() if value is None)
    if missing:
        # Gate-0 deliberately binds paths even for attached endpoints.  The
        # live attestation treats endpoint paths as observational evidence and
        # checks the immutable ID, while retaining the declared path in the job.
        raise ValueError(
            "Gate-0 child environment lacks checkpoint bindings: "
            + ", ".join(missing)
        )
    return {name: str(value) for name, value in values.items()}


def verify_gate0_job_external_bindings(
    job: Gate0JobSpec,
    *,
    repo_root: str | os.PathLike[str] | None = None,
) -> None:
    """Recheck bound bytes, parsed identities, facts, and current source."""
    bindings = job.external_bindings
    if bindings is None:
        raise ValueError("Gate-0 job has no external artifact bindings")
    source = Path(bindings.source_job_path)
    if not source.is_file() or _sha256_file(source) != bindings.source_job_sha256:
        raise ValueError("bound Gate-0 source job bytes changed")
    handoff = Path(bindings.handoff_config_path)
    if not handoff.is_file():
        raise FileNotFoundError(f"bound Gate-0 handoff config is missing: {handoff}")
    current_handoff = _handoff_binding(handoff)
    expected_handoff = {
        "handoff_config_path": bindings.handoff_config_path,
        "handoff_config_sha256": bindings.handoff_config_sha256,
        "handoff_canonical_sha256": bindings.handoff_canonical_sha256,
        "handoff_configuration_id": bindings.handoff_configuration_id,
        "handoff_controller_configuration_id": (
            bindings.handoff_controller_configuration_id
        ),
        "handoff_controller_method": bindings.handoff_controller_method,
        "handoff_controller_implementation_version": (
            bindings.handoff_controller_implementation_version
        ),
        "handoff_checkpoint_id": bindings.handoff_checkpoint_id,
        "handoff_metadata_pi05_checkpoint_id": (
            bindings.handoff_metadata_pi05_checkpoint_id
        ),
        "handoff_metadata_sam3_checkpoint_id": (
            bindings.handoff_metadata_sam3_checkpoint_id
        ),
    }
    if current_handoff != expected_handoff:
        raise ValueError("bound Gate-0 handoff config identity changed")
    runtime = RuntimeConfig.model_validate(job.adapter_config["runtime"])
    references = {reference.name: reference for reference in job.runtime_probes}
    if set(references) != {binding.name for binding in bindings.runtime_probes}:
        raise ValueError("Gate-0 probe references disagree with bound probes")
    for binding in bindings.runtime_probes:
        path = Path(binding.resolved_path)
        if not path.is_file() or _sha256_file(path) != binding.sha256:
            raise ValueError(
                f"bound Gate-0 runtime probe bytes changed: {binding.name}"
            )
        current = _validate_probe_binding(
            references[binding.name],
            path=path,
            runtime=runtime,
        )
        if current != binding:
            raise ValueError(
                f"bound Gate-0 runtime probe content changed: {binding.name}"
            )
    if repo_root is not None:
        from rpent.research.handoff.experiments.manifest import (
            compute_source_revision,
        )

        current_source = compute_source_revision(repo_root)
        if current_source != job.source_revision:
            raise ValueError(
                "current repository bytes disagree with Gate-0 source_revision: "
                f"expected={job.source_revision!r}, actual={current_source!r}"
            )


def gate0_resume_anchor(output_dir: str | os.PathLike[str]) -> dict[str, str]:
    """Checksum every authoritative prior-run artifact before planning."""
    root = Path(output_dir).expanduser().resolve()
    paths = [
        root / "gate0_run_manifest.json",
        root / "online" / "outcomes.jsonl",
        root / "online" / "decisions.jsonl",
        root / "privileged" / "setups.jsonl",
        root / "collection_summary.json",
    ]
    for directory in ("attempts", "collection_attempts", "runtime_identity"):
        candidate = root / directory
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.json")))
    telemetry = root / "telemetry"
    if telemetry.is_dir():
        paths.extend(
            sorted(path for path in telemetry.rglob("*") if path.is_file())
        )
    return {
        str(path.relative_to(root)).replace("\\", "/"): _sha256_file(path)
        for path in sorted(set(paths))
        if path.is_file()
    }


def gate0_plan_id(
    job: Gate0JobSpec,
    *,
    repo_root: str | os.PathLike[str],
    limit: int | None,
    resume: bool,
    resume_anchor: Mapping[str, str] | None = None,
) -> str:
    """Bind one physical attempt to the immutable job and prior output state."""
    if limit is not None and limit < 0:
        raise ValueError("Gate-0 plan limit must be non-negative")
    anchor = dict(
        resume_anchor
        if resume_anchor is not None
        else gate0_resume_anchor(job.output_dir)
    )
    payload = {
        "schema_version": "rpent.handoff-gate0-child-plan/v2",
        "job_configuration_id": job.stable_configuration_id,
        "repo_root": str(Path(repo_root).expanduser().resolve()),
        "limit": limit,
        "resume": resume,
        "resume_anchor": dict(sorted(anchor.items())),
        "env_overrides": gate0_runtime_environment(job),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return "gate0-plan-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def load_object(import_path: str) -> Any:
    """Resolve a validated ``module:attribute`` path with a useful error."""
    value = _factory_path(import_path, "import path")
    module_name, _, attribute = value.partition(":")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ImportError(
            f"configured attribute {attribute!r} is missing from {module_name!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class Gate0AdapterBundle:
    adapter: Any
    setup_sink: Any | None = None
    runtime_attestation: Any | None = None
    runtime_attestation_sha256: str | None = None
    cleanup: Callable[[], None] | None = None


def instantiate_gate0_adapter(
    job: Gate0JobSpec,
    *,
    gate0_config: Any,
    output_dir: Path,
) -> Gate0AdapterBundle:
    """Invoke and validate the configured Gate-0 adapter factory contract."""
    factory = load_object(job.adapter_factory)
    if not callable(factory):
        raise TypeError(f"Gate-0 adapter factory is not callable: {job.adapter_factory}")
    try:
        produced = factory(
            job=job,
            adapter_config=dict(job.adapter_config),
            gate0_config=gate0_config,
            output_dir=output_dir,
        )
    except TypeError as exc:
        raise TypeError(
            "Gate-0 adapter factory must accept keyword arguments job, "
            "adapter_config, gate0_config, and output_dir; factory raised: "
            f"{exc}"
        ) from exc

    if isinstance(produced, Gate0AdapterBundle):
        bundle = produced
    elif isinstance(produced, Mapping):
        if "adapter" not in produced:
            raise TypeError("Gate-0 adapter factory mapping omitted 'adapter'")
        bundle = Gate0AdapterBundle(
            adapter=produced["adapter"],
            setup_sink=produced.get("setup_sink"),
            runtime_attestation=produced.get("runtime_attestation"),
            runtime_attestation_sha256=produced.get(
                "runtime_attestation_sha256"
            ),
            cleanup=produced.get("cleanup"),
        )
    else:
        bundle = Gate0AdapterBundle(adapter=produced)
    if bundle.cleanup is not None and not callable(bundle.cleanup):
        raise TypeError("Gate-0 adapter cleanup must be a zero-argument callable")
    return bundle


class SetupJsonlSink:
    """Conflict-detecting append-only storage for privileged setup records."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        from rpent.research.handoff.privileged import ExperimentSetupRecord

        self.path = Path(path)
        self._record_type = ExperimentSetupRecord
        self._lock = threading.Lock()
        self._records: dict[str, str] = {}
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, start=1):
                line = raw.strip()
                if not line:
                    raise ValueError(
                        f"blank setup JSONL line in {self.path} at {line_number}"
                    )
                try:
                    record = self._record_type.model_validate_json(line)
                except Exception as exc:
                    raise ValueError(
                        f"invalid setup record in {self.path} at {line_number}: {exc}"
                    ) from exc
                canonical = record.canonical_json()
                if record.record_id in self._records:
                    raise ValueError(
                        f"duplicate setup record ID in {self.path}: {record.record_id}"
                    )
                self._records[record.record_id] = canonical

    @property
    def records(self) -> tuple[Any, ...]:
        """Return validated existing records in durable file order."""
        return tuple(
            self._record_type.model_validate_json(canonical)
            for canonical in self._records.values()
        )

    def append_setup(self, setup: Any) -> None:
        record = self._record_type.model_validate(setup)
        canonical = record.canonical_json()
        with self._lock:
            previous = self._records.get(record.record_id)
            if previous is not None:
                if previous != canonical:
                    raise ValueError(
                        f"setup record ID conflict: {record.record_id}"
                    )
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(canonical)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._records[record.record_id] = canonical


class ControlledChildPlan(HandoffRecord):
    """One shell-free, process-isolated controlled-trial invocation."""

    schema_version: Literal[CONTROLLED_CHILD_PLAN_SCHEMA_VERSION] = (
        CONTROLLED_CHILD_PLAN_SCHEMA_VERSION
    )
    plan_id: str
    trial_id: str
    command: tuple[str, ...]
    cwd: str
    env_overrides: dict[str, str] = Field(default_factory=dict)
    output_dir: str

    @field_validator("plan_id", "trial_id", "cwd", "output_dir")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item or "\x00" in item for item in value):
            raise ValueError("controlled child command contains an invalid token")
        return value


def build_controlled_child_plan(
    trial: Any,
    *,
    manifest_path: str | os.PathLike[str],
    repo_root: str | os.PathLike[str],
    python_executable: str | None = None,
) -> ControlledChildPlan:
    """Build the real child command without starting services or writing files."""
    from rpent.research.handoff.experiments.config import ExecutionLayer

    if trial.execution_layer is not ExecutionLayer.CONTROLLED:
        raise ValueError(f"trial is not controlled: {trial.trial_id}")
    if not trial.condition.handoff_enabled or trial.handoff_config_path is None:
        raise ValueError(
            f"controlled trial is not explicitly handoff-enabled: {trial.trial_id}"
        )
    base_command = (
        python_executable or sys.executable,
        "-m",
        "rpent.research.handoff",
        "--traceback",
        "_controlled-child",
        "--manifest",
        str(Path(manifest_path).expanduser().resolve()),
        "--trial-id",
        trial.trial_id,
    )
    env_overrides: dict[str, str] = {}
    for field_name, variable in (
        ("pi05_checkpoint_path", "PI05_CHECKPOINT_PATH"),
        ("sam3_checkpoint_path", "SAM3_CHECKPOINT_PATH"),
        ("pi05_checkpoint_id", "RPENT_PI05_CHECKPOINT_ID"),
        ("sam3_checkpoint_id", "RPENT_SAM3_CHECKPOINT_ID"),
    ):
        value = getattr(trial.runtime, field_name)
        if value is not None:
            env_overrides[variable] = value
    payload = json.dumps(
        {
            "trial_id": trial.trial_id,
            "command": base_command,
            "cwd": str(Path(repo_root).expanduser().resolve()),
            "env_overrides": env_overrides,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    plan_id = "controlled-" + hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]
    command = (*base_command, "--plan-id", plan_id)
    return ControlledChildPlan(
        plan_id=plan_id,
        trial_id=trial.trial_id,
        command=command,
        cwd=str(Path(repo_root).expanduser().resolve()),
        env_overrides=env_overrides,
        output_dir=trial.output_dir,
    )


def execute_controlled_child_plan(
    plan: ControlledChildPlan,
    *,
    allow_execution: bool = False,
    capture_output: bool = False,
    timeout_s: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute a controlled child only after explicit caller authorization."""
    if not allow_execution:
        raise PermissionError(
            "controlled server execution is disabled; explicitly authorize the plan"
        )
    environment = os.environ.copy()
    environment.update(plan.env_overrides)
    return subprocess.run(
        list(plan.command),
        cwd=plan.cwd,
        env=environment,
        shell=False,
        check=False,
        capture_output=capture_output,
        text=True,
        timeout=timeout_s,
    )


def write_controlled_plans(
    plans: tuple[ControlledChildPlan, ...] | list[ControlledChildPlan],
    path: str | os.PathLike[str],
) -> Path:
    if not plans:
        raise ValueError("refusing to write an empty controlled plan")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(
            [plan.model_dump(mode="json", exclude_none=False) for plan in plans],
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def _write_json_immutable(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != canonical:
            raise FileExistsError(
                f"immutable runtime configuration already differs: {path}"
            )
        return path
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _resolve_trial_local_core_paths(
    core: Mapping[str, Any],
    *,
    source_config: Path,
) -> dict[str, Any]:
    """Preserve source-config path semantics after copying into a trial dir."""
    resolved = _finite_json_mapping(core, "source handoff core")

    def resolve_path(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty path string")
        expanded = os.path.expandvars(value)
        if "$" in expanded:
            raise ValueError(
                f"{name} contains an unresolved environment variable: {expanded!r}"
            )
        path = Path(expanded).expanduser()
        if not path.is_absolute():
            path = source_config.parent / path
        return str(path.resolve())

    if resolved.get("model_artifact") is not None:
        resolved["model_artifact"] = resolve_path(
            resolved["model_artifact"], "core.model_artifact"
        )
    for policy_name in ("policy", "fallback_policy"):
        policy = resolved.get(policy_name)
        if not isinstance(policy, dict):
            continue
        if policy.get("positive_references_file") is not None:
            policy["positive_references_file"] = resolve_path(
                policy["positive_references_file"],
                f"core.{policy_name}.positive_references_file",
            )
    return resolved


def write_resolved_handoff_config(
    trial: Any,
    *,
    execution_plan_id: str | None = None,
    manifest_id: str | None = None,
) -> Path:
    """Create a trial-local runtime config with immutable manifest identity."""
    from rpent.research.handoff.experiments.config import load_strict_json
    from rpent.research.handoff.experiments.manifest import (
        verify_trial_artifact_bindings,
    )

    if trial.handoff_config_path is None:
        raise ValueError("handoff-enabled trial has no source handoff config")
    verify_trial_artifact_bindings(trial)
    source = Path(trial.handoff_config_path)
    raw = load_strict_json(source)
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("source handoff metadata must be an object")
    metadata = {
        **metadata,
        "run_id": trial.experiment_id,
        "episode_id": trial.trial_id,
        "trial_id": trial.trial_id,
        "experiment_configuration_id": trial.configuration_id,
        "suite": trial.task.suite,
        "task": trial.task.task,
        "seed": trial.task.seed,
        "repeat_index": trial.repeat_index,
        "condition_name": trial.condition.name,
        "condition": trial.condition.name,
        "execution_layer": trial.execution_layer.value,
        "representation": trial.condition.feature_set.value,
        "evidence_mode": trial.condition.evidence.value,
        "decision_mode": trial.condition.decision.value,
        "uncertainty_mode": trial.condition.uncertainty.value,
        "hierarchy_mode": trial.condition.hierarchy.value,
        "training_target_label": trial.task.training_target_label,
        "source_revision": trial.source_revision,
        "manifest_id": manifest_id,
        "execution_plan_id": execution_plan_id,
        "artifact_bindings": dict(trial.artifact_bindings),
    }
    if trial.task.reset_id is None:
        metadata.pop("reset_id", None)
    else:
        metadata["reset_id"] = trial.task.reset_id
    core = raw.get("core")
    if not isinstance(core, dict):
        raise ValueError("source handoff core must be an object")
    if "model_artifact" in raw:
        raise ValueError(
            "source handoff config ambiguously combines core with a top-level "
            "model_artifact"
        )
    if trial.model_artifact_path is not None:
        from rpent.research.handoff.artifacts import ModelArtifactManifest

        artifact_manifest_path = Path(trial.model_artifact_path) / "manifest.json"
        artifact_manifest = ModelArtifactManifest.model_validate(
            load_strict_json(artifact_manifest_path)
        )
        if (
            trial.condition.model_artifact_id is not None
            and trial.condition.model_artifact_id != artifact_manifest.artifact_id
        ):
            raise ValueError(
                "condition model_artifact_id disagrees with the bound artifact"
            )
        model_artifact_id = artifact_manifest.artifact_id
        core = {
            **core,
            # Bind the manifest-resolved artifact here instead of relying on a
            # process-global environment placeholder.  This is part of the
            # scientific controller identity and is auditable per trial.
            "model_artifact": str(Path(trial.model_artifact_path).resolve()),
        }
    else:
        model_artifact_id = None
    core = _resolve_trial_local_core_paths(core, source_config=source.resolve())
    configured_checkpoint_id = trial.condition.checkpoint_id
    runtime_checkpoint_id = trial.runtime.pi05_checkpoint_id
    if (
        configured_checkpoint_id is not None
        and runtime_checkpoint_id is not None
        and configured_checkpoint_id != runtime_checkpoint_id
    ):
        raise ValueError(
            "condition checkpoint_id disagrees with runtime Pi0.5 checkpoint id"
        )
    checkpoint_id = runtime_checkpoint_id or configured_checkpoint_id
    metadata.update(
        {
            "pi05_checkpoint_id": checkpoint_id,
            "sam3_checkpoint_id": trial.runtime.sam3_checkpoint_id,
            "model_artifact_id": model_artifact_id,
        }
    )
    resolved = {
        **raw,
        "enabled": True,
        "controller_method": trial.condition.method,
        "checkpoint_id": checkpoint_id,
        "model_artifact_id": model_artifact_id,
        "core": core,
        "metadata": metadata,
    }
    destination = Path(trial.output_dir) / "resolved_handoff_runtime.json"
    return _write_json_immutable(destination, resolved)


def _controlled_tool_request(trial: Any) -> tuple[str, dict[str, Any]]:
    parameters = trial.condition.parameters
    configured_name = parameters.get("composite_tool")
    if configured_name is None:
        normalized_skill = trial.task.skill_name.strip().lower()
        if normalized_skill in {"pick", "pi0_pick", "handoff_pi0_pick"}:
            name = "pi0_pick"
        elif normalized_skill in {
            "contact",
            "doubled",
            "pi0_doubled",
            "handoff_pi0_doubled",
        }:
            name = "pi0_doubled"
        else:
            raise ValueError(
                "controlled condition must set parameters.composite_tool for "
                f"unrecognized skill {trial.task.skill_name!r}"
            )
    elif isinstance(configured_name, str):
        name = {
            "handoff_pi0_pick": "pi0_pick",
            "handoff_pi0_doubled": "pi0_doubled",
        }.get(configured_name, configured_name)
    else:
        raise ValueError("condition parameters.composite_tool must be a string")
    if name not in {"pi0_pick", "pi0_doubled"}:
        raise ValueError(f"unsupported controlled composite tool: {name!r}")
    configured_kwargs = parameters.get("tool_kwargs", {})
    if not isinstance(configured_kwargs, dict):
        raise ValueError("condition parameters.tool_kwargs must be an object")
    protected = {"prompt", "target_description", "target_id"}
    overlap = sorted(protected.intersection(configured_kwargs))
    if overlap:
        raise ValueError(
            "tool_kwargs cannot override manifest task identity fields: "
            + ", ".join(overlap)
        )
    request = {
        "prompt": trial.task.skill_prompt,
        "target_description": trial.task.target_description,
        "target_id": trial.task.target_id,
        **configured_kwargs,
    }
    _finite_json_mapping(request, "controlled composite request")
    return name, request


def _verify_controlled_outcome(
    trial: Any,
    runtime_config: Any,
    *,
    manifest_id: str,
    plan_id: str,
) -> tuple[Path, Any]:
    from rpent.research.handoff.types import ControllerIdentity, OutcomeRecord

    output_root = Path(trial.output_dir) / runtime_config.output_subdir
    path = output_root / "outcomes.jsonl"
    if not path.is_file():
        raise RuntimeError(f"controlled tool wrote no outcome JSONL: {path}")
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line:
                raise ValueError(f"blank outcome line in {path} at {line_number}")
            try:
                record = OutcomeRecord.model_validate_json(line)
            except Exception as exc:
                raise ValueError(
                    f"invalid controlled outcome in {path} at {line_number}: {exc}"
                ) from exc
            records.append(record)
    expected_controller = ControllerIdentity(
        method=runtime_config.controller_method,
        implementation_version=runtime_config.controller_implementation_version,
        checkpoint_id=runtime_config.checkpoint_id,
        configuration_id=runtime_config.controller_configuration_id,
    )
    invalid = [
        record.record_id
        for record in records
        if (
            record.identity.run_id != trial.experiment_id
            or record.identity.episode_id != trial.trial_id
            or record.identity.trial_id != trial.trial_id
            or record.identity.suite != trial.task.suite
            or str(record.identity.task_id) != str(trial.task.task)
            or record.identity.seed != trial.task.seed
            or record.identity.repeat_index != trial.repeat_index
            or record.identity.reset_id is None
            or record.skill.name != trial.task.skill_name
            or record.skill.semantic_target != trial.task.target_description
            or record.controller != expected_controller
            or record.source_revision != trial.source_revision
            or record.metadata.get("manifest_id") != manifest_id
            or record.metadata.get("execution_plan_id") != plan_id
            or record.metadata.get("representation")
            != trial.condition.feature_set.value
            or record.metadata.get("evidence_mode")
            != trial.condition.evidence.value
            or record.metadata.get("uncertainty_mode")
            != trial.condition.uncertainty.value
            or record.metadata.get("hierarchy_mode")
            != trial.condition.hierarchy.value
        )
    ]
    if invalid:
        raise RuntimeError(
            f"controlled outcomes disagree with resolved trial/config: {invalid[:10]}"
        )
    if len(records) != 1:
        raise RuntimeError(
            f"expected exactly one outcome for {trial.trial_id}, found {len(records)}"
        )
    sink_manifest_path = output_root / "manifest.json"
    if not sink_manifest_path.is_file():
        raise RuntimeError(f"controlled sink manifest is missing: {sink_manifest_path}")
    from rpent.research.handoff.experiments.config import load_strict_json

    sink_manifest = load_strict_json(sink_manifest_path)
    if (
        sink_manifest.get("configuration_id") != runtime_config.configuration_id
        or sink_manifest.get("controller_configuration_id")
        != runtime_config.controller_configuration_id
        or sink_manifest.get("configuration") != runtime_config.canonical_config
    ):
        raise RuntimeError("controlled sink manifest disagrees with resolved config")
    from rpent.research.handoff.experiments.runtime_identity import (
        load_runtime_attestation,
    )

    attestation_path = Path(trial.output_dir) / "runtime_identity.json"
    attestation = load_runtime_attestation(attestation_path)
    if (
        attestation.trial_id != trial.trial_id
        or attestation.manifest_id != manifest_id
        or attestation.plan_id != plan_id
        or attestation.source_revision != trial.source_revision
        or tuple(
            item.observed_checkpoint_id for item in attestation.observations
        )
        != (
            trial.runtime.pi05_checkpoint_id,
            trial.runtime.sam3_checkpoint_id,
        )
    ):
        raise RuntimeError("controlled runtime attestation disagrees with trial")
    return path, records[0]


def run_controlled_trial(
    manifest_path: str | os.PathLike[str],
    trial_id: str,
    plan_id: str,
) -> dict[str, Any]:
    """Run one manifest-bound controlled trial through the normal LIBERO toolkit."""
    # All environment/server-heavy imports are intentionally inside this child.
    import argparse

    from robots.libero.handoff_runtime import load_handoff_runtime_config
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.envs import get_env_spec, get_toolkit
    from rpent.research.handoff.experiments.config import ExecutionLayer
    from rpent.research.handoff.experiments.manifest import (
        load_manifest,
        verify_manifest_external_bindings,
    )
    from rpent.research.handoff.experiments.runtime_identity import (
        attest_runtime_checkpoint_clients,
        write_runtime_attestation,
    )
    from rpent.utils.logging import init_output_dir

    manifest = load_manifest(manifest_path)
    verify_manifest_external_bindings(
        manifest,
        repo_root=Path.cwd(),
        require_runtime_probes=True,
    )
    candidates = [trial for trial in manifest.trials if trial.trial_id == trial_id]
    if len(candidates) != 1:
        raise ValueError(
            f"manifest must contain exactly one trial {trial_id!r}; found {len(candidates)}"
        )
    trial = candidates[0]
    if trial.execution_layer is not ExecutionLayer.CONTROLLED:
        raise ValueError(f"trial is not controlled: {trial_id}")
    if not trial.condition.handoff_enabled:
        raise ValueError("controlled child refuses a handoff-disabled condition")
    expected_plan = build_controlled_child_plan(
        trial,
        manifest_path=manifest_path,
        repo_root=Path.cwd(),
        python_executable=sys.executable,
    )
    if expected_plan.plan_id != plan_id:
        raise ValueError("controlled child plan ID does not bind this invocation")

    output_path = Path(trial.output_dir)
    resolved_handoff_path = output_path / "resolved_handoff_runtime.json"
    existing_outcomes = tuple(output_path.rglob("outcomes.jsonl")) if output_path.exists() else ()
    if existing_outcomes:
        if len(existing_outcomes) != 1 or not resolved_handoff_path.is_file():
            raise RuntimeError("controlled output contains ambiguous terminal artifacts")
        handoff_config = load_handoff_runtime_config(resolved_handoff_path)
        outcome_path, existing_outcome = _verify_controlled_outcome(
            trial,
            handoff_config,
            manifest_id=manifest.manifest_id,
            plan_id=plan_id,
        )
        return {
            "trial_id": trial.trial_id,
            "tool_name": None,
            "outcome_jsonl": str(outcome_path.resolve()),
            "recipe_path": None,
            "resolved_handoff_config": str(resolved_handoff_path.resolve()),
            "resumed_existing_outcome": True,
            "cancelled": existing_outcome.termination.reason.value == "cancelled",
            "tool_error": None,
            "outcome_failure_mode": (
                existing_outcome.termination.failure_mode.value
            ),
        }
    stale_files = (
        tuple(path for path in output_path.rglob("*") if path.is_file())
        if output_path.exists()
        else ()
    )
    if stale_files:
        raise FileExistsError(
            "controlled retry found a partial attempt; preserve it and allocate "
            "a new trial identity: " + ", ".join(str(path) for path in stale_files[:20])
        )

    output_dir = init_output_dir(output_path)
    _write_json_immutable(
        output_path / "attempt.json",
        {
            "schema_version": "rpent.handoff-controlled-attempt/v1",
            "trial_id": trial.trial_id,
            "manifest_id": manifest.manifest_id,
            "plan_id": plan_id,
            "source_revision": trial.source_revision,
        },
    )
    resolved_handoff_path = write_resolved_handoff_config(
        trial,
        execution_plan_id=plan_id,
        manifest_id=manifest.manifest_id,
    )
    handoff_config = load_handoff_runtime_config(resolved_handoff_path)
    for variable, value in expected_plan.env_overrides.items():
        os.environ[variable] = value

    args = argparse.Namespace(
        output_dir=str(output_dir),
        suite=trial.task.suite,
        task=trial.task.task,
        seed=trial.task.seed,
        max_episode_steps=trial.runtime.max_episode_steps,
        libero_type=trial.runtime.libero_type,
        env_endpoint=trial.runtime.env_endpoint,
        vla_endpoint=trial.runtime.vla_endpoint,
        sam3_endpoint=trial.runtime.sam3_endpoint,
        cuda_device=trial.runtime.cuda_device,
        handoff_config=str(resolved_handoff_path),
    )
    dashboard = NullDashboardEventSink()
    env_spec = get_env_spec("libero")
    daemons: list[Any] = []
    toolkit: Any | None = None
    recipe_path: str | None = None
    tool_name, tool_request = _controlled_tool_request(trial)
    try:
        daemons, primitives_kwargs = env_spec.init_runtime(
            args,
            output_dir,
            dashboard,
        )
        attestation = attest_runtime_checkpoint_clients(
            primitives_kwargs,
            trial.runtime,
            trial_id=trial.trial_id,
            manifest_id=manifest.manifest_id,
            plan_id=plan_id,
            source_revision=trial.source_revision,
        )
        write_runtime_attestation(
            attestation,
            output_path / "runtime_identity.json",
        )
        toolkit = get_toolkit(
            "libero",
            primitives_kwargs=primitives_kwargs,
            dashboard_events=dashboard,
            video_path=str(output_dir / "episode.mp4"),
        )
        registered = {item["name"] for item in toolkit.get_tools_spec()}
        if tool_name not in registered:
            raise RuntimeError(
                f"opt-in Pi0 tool {tool_name!r} was not registered; "
                "verify enabled handoff configuration"
            )
        result = toolkit.execute_tool(tool_name, tool_request)
        cancelled = bool(
            isinstance(result.result, dict)
            and (
                result.result.get("code") == "tool_cancelled"
                or result.result.get("interrupted") is True
            )
        )
        tool_error = (
            str(result.result["error"])
            if isinstance(result.result, dict) and result.result.get("error")
            else None
        )
        outcome_path, outcome = _verify_controlled_outcome(
            trial,
            handoff_config,
            manifest_id=manifest.manifest_id,
            plan_id=plan_id,
        )
        cancelled = cancelled or outcome.termination.reason.value == "cancelled"
        recipe_path = toolkit.write_recipe(
            f"controlled_{trial.condition.name}_{trial.trial_id}"
        )
        return {
            "trial_id": trial.trial_id,
            "tool_name": tool_name,
            "outcome_jsonl": str(outcome_path.resolve()),
            "recipe_path": recipe_path,
            "resolved_handoff_config": str(resolved_handoff_path.resolve()),
            "cancelled": cancelled,
            "tool_error": tool_error,
            "outcome_failure_mode": outcome.termination.failure_mode.value,
        }
    finally:
        try:
            if toolkit is not None:
                toolkit.close()
        finally:
            stop_errors: list[Exception] = []
            for daemon in reversed(daemons):
                try:
                    daemon.stop()
                except Exception as exc:  # preserve attempts to stop all owners
                    stop_errors.append(exc)
            if stop_errors and sys.exc_info()[0] is None:
                raise RuntimeError(
                    "one or more owned runtime daemons could not be stopped: "
                    + "; ".join(str(error) for error in stop_errors)
                )


__all__ = [
    "CONTROLLED_CHILD_PLAN_SCHEMA_VERSION",
    "EXECUTION_CONFIRMATION",
    "GATE0_JOB_SCHEMA_VERSION",
    "ControlledChildPlan",
    "Gate0AdapterBundle",
    "Gate0ExternalBindings",
    "Gate0JobSpec",
    "Gate0RuntimeProbeBinding",
    "Gate0RuntimeProbeReference",
    "SetupJsonlSink",
    "build_controlled_child_plan",
    "execute_controlled_child_plan",
    "gate0_plan_id",
    "gate0_resume_anchor",
    "gate0_runtime_environment",
    "instantiate_gate0_adapter",
    "load_gate0_job",
    "load_object",
    "run_controlled_trial",
    "verify_gate0_job_external_bindings",
    "write_controlled_plans",
    "write_resolved_handoff_config",
]
