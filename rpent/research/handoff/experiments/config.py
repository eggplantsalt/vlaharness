"""Strict JSON configuration for reproducible handoff experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.types import (
    FeatureAvailability,
    HandoffRecord,
    LabelSource,
    OutcomeLabels,
)

EXPERIMENT_CONFIG_SCHEMA_VERSION = "rpent.handoff-experiment-config/v1"


def _non_empty(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _safe_identifier(value: str, field_name: str) -> str:
    value = _non_empty(value, field_name)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(
            f"{field_name} must be a path-safe identifier containing only "
            "letters, digits, dot, underscore, or hyphen"
        )
    return value


def _validate_json_value(value: Any, *, name: str) -> Any:
    """Reject non-JSON and non-finite method metadata deterministically."""
    try:
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values only") from exc
    return value


def canonical_json(value: Any) -> str:
    """Serialize a record or JSON value with stable key and number handling."""
    if isinstance(value, HandoffRecord):
        value = value.model_dump(mode="json", exclude_none=False)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_digest(value: Any) -> str:
    """Return the SHA-256 digest of the canonical JSON representation."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_identifier(prefix: str, value: Any, *, digest_chars: int = 20) -> str:
    """Return a readable content-derived identifier."""
    if digest_chars < 12 or digest_chars > 64:
        raise ValueError("digest_chars must be between 12 and 64")
    prefix = _non_empty(prefix, "prefix")
    return f"{prefix}-{stable_digest(value)[:digest_chars]}"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def load_strict_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load one strict JSON object, rejecting NaN/Infinity and duplicate keys."""
    source = Path(path)

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {source}: {key!r}")
            result[key] = value
        return result

    try:
        with source.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                parse_constant=_reject_json_constant,
                object_pairs_hook=object_pairs,
            )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"experiment config must be a JSON object: {source}")
    return value


class ExecutionLayer(str, Enum):
    """Which causal/system layer owns a trial."""

    GATE0 = "gate0"
    CONTROLLED = "controlled"
    FULL_AGENT = "full_agent"


class FeatureSetName(str, Enum):
    ABSOLUTE = "absolute"
    TARGET_RELATIVE = "target_relative"
    TARGET_RELATIVE_VISUAL = "target_relative_visual"
    DEPLOYMENT_FULL = "deployment_full"


class EvidenceMode(str, Enum):
    SUCCESS_ONLY = "success_only"
    SUCCESS_AND_FAILURE = "success_and_failure"


class DecisionMode(str, Enum):
    DIRECT = "direct"
    FIXED = "fixed"
    RETRIEVAL = "retrieval"
    THRESHOLD = "threshold"
    PROJECTION = "projection"
    SUPPORT_REGION = "support_region"
    ONLINE_SWITCHING = "online_switching"
    ORACLE = "oracle"


class UncertaintyMode(str, Enum):
    MEAN_ONLY = "mean_only"
    CONSERVATIVE = "conservative"


class HierarchyMode(str, Enum):
    PLANNER_MEDIATED = "planner_mediated"
    LOCAL_GOVERNOR = "local_governor"
    NOT_APPLICABLE = "not_applicable"


class RuntimeConfig(HandoffRecord):
    """Environment/service settings shared by resolved trials."""

    env_name: Literal["libero"] = "libero"
    libero_type: Literal["standard", "pro", "plus"] = "pro"
    max_episode_steps: int = Field(default=10_000, gt=0)
    cuda_device: int | None = Field(default=None, ge=0)
    env_endpoint: str | None = None
    vla_endpoint: str | None = None
    sam3_endpoint: str | None = None
    pi05_checkpoint_path: str | None = None
    sam3_checkpoint_path: str | None = None
    pi05_checkpoint_id: str | None = None
    sam3_checkpoint_id: str | None = None

    @field_validator(
        "env_endpoint",
        "vla_endpoint",
        "sam3_endpoint",
        "pi05_checkpoint_path",
        "sam3_checkpoint_path",
        "pi05_checkpoint_id",
        "sam3_checkpoint_id",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _non_empty(value, info.field_name)

    @field_validator("env_endpoint", "vla_endpoint", "sam3_endpoint")
    @classmethod
    def validate_endpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        endpoint = value.split("://", 1)[-1]
        host, separator, port = endpoint.rpartition(":")
        if not separator or not host:
            raise ValueError("endpoint must be [protocol://]host:port")
        try:
            port_number = int(port)
        except ValueError as exc:
            raise ValueError("endpoint port must be an integer") from exc
        if not 1 <= port_number <= 65535:
            raise ValueError("endpoint port must be in [1, 65535]")
        if "://" in value and value.split("://", 1)[0] not in {"http", "socket"}:
            raise ValueError("endpoint protocol must be http or socket")
        return value


class RuntimeProbeReference(HandoffRecord):
    """One named, content-bound runtime probe required by a matrix."""

    name: str
    path: str
    required_observed_facts: tuple[str, ...] = ()

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _safe_identifier(value, "runtime probe name")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _non_empty(value, "runtime probe path")

    @field_validator("required_observed_facts")
    @classmethod
    def validate_required_facts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item for item in value):
            raise ValueError("runtime probe fact names must be non-empty")
        if len(value) != len(set(value)):
            raise ValueError("runtime probe fact names must be unique")
        return value


class PlannerConfig(HandoffRecord):
    """Full-agent planner settings; ignored by controller-only runners."""

    backend: Literal["api", "claude_code", "codex"] = "api"
    model: str | None = None
    base_url: str | None = None
    max_turns: int = Field(default=100, gt=0)
    max_tokens: int = Field(default=8192, gt=0)
    planner_timeout_s: int | None = Field(default=None, gt=0)
    claude_code_max_budget_usd: float | None = Field(default=None, gt=0.0)
    no_images: bool = False

    @field_validator("model", "base_url")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _non_empty(value, info.field_name)


class TaskSpec(HandoffRecord):
    """One semantic task with the reset seeds to expand."""

    suite: str
    task: int = Field(ge=0)
    seeds: tuple[int, ...] = (0,)
    target_id: str
    target_description: str
    skill_name: str
    skill_prompt: str
    training_target_label: Literal[
        "primitive_success",
        "skill_success",
        "task_success",
        "episode_truncated",
        "llm_finish",
    ] = "skill_success"
    label_source: LabelSource
    task_config: str | None = None
    reset_id_template: str | None = None

    @field_validator(
        "suite",
        "target_id",
        "target_description",
        "skill_name",
        "skill_prompt",
    )
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("task_config", "reset_id_template")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _non_empty(value, info.field_name)

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("seeds must not be empty")
        if any(seed < 0 for seed in value):
            raise ValueError("seeds must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("seeds must be unique")
        return value

    @model_validator(mode="after")
    def validate_label(self) -> Self:
        if self.training_target_label not in OutcomeLabels.model_fields:
            raise ValueError("unknown training target label")
        if self.label_source is LabelSource.UNAVAILABLE:
            raise ValueError("configured training label source cannot be unavailable")
        return self


_RESERVED_CHILD_FLAGS = {
    "-i",
    "--dashboard",
    "--env",
    "--interactive",
    "--suite",
    "--task",
    "--seed",
    "--output-dir",
    "--handoff-config",
    "--research-trial-id",
    "--research-reset-identity-output",
    "--research-completion-output",
}


class ConditionSpec(HandoffRecord):
    """One method/ablation condition in the experiment matrix."""

    name: str
    execution_layer: ExecutionLayer
    method: str
    handoff_enabled: bool = False
    handoff_config: str | None = None
    model_artifact: str | None = None
    model_artifact_id: str | None = None
    checkpoint_id: str | None = None
    feature_set: FeatureSetName = FeatureSetName.DEPLOYMENT_FULL
    evidence: EvidenceMode = EvidenceMode.SUCCESS_AND_FAILURE
    decision: DecisionMode = DecisionMode.ONLINE_SWITCHING
    uncertainty: UncertaintyMode = UncertaintyMode.CONSERVATIVE
    hierarchy: HierarchyMode = HierarchyMode.LOCAL_GOVERNOR
    policy_feature_availability: tuple[FeatureAvailability, ...] = (
        FeatureAvailability.DEPLOYMENT_SENSOR,
        FeatureAvailability.DEPLOYMENT_PERCEPTION,
        FeatureAvailability.DERIVED_DEPLOYMENT,
    )
    parameters: dict[str, Any] = Field(default_factory=dict)
    extra_cli_args: tuple[str, ...] = ()

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _safe_identifier(value, "name")

    @field_validator("method")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)

    @field_validator(
        "handoff_config", "model_artifact", "model_artifact_id", "checkpoint_id"
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _non_empty(value, info.field_name)

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_value(value, name="parameters")

    @field_validator("extra_cli_args")
    @classmethod
    def validate_extra_cli_args(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for token in value:
            if not token or any(character in token for character in ("\x00", "\r", "\n")):
                raise ValueError("extra_cli_args contains an empty/invalid token")
            flag = token.split("=", 1)[0]
            if flag in _RESERVED_CHILD_FLAGS:
                raise ValueError(f"extra_cli_args cannot override reserved flag {flag}")
        return value

    @model_validator(mode="after")
    def validate_condition(self) -> Self:
        forbidden = [item.value for item in self.policy_feature_availability if not item.online_allowed]
        if forbidden:
            raise ValueError(
                "policy_feature_availability contains privileged/setup sources: "
                f"{sorted(forbidden)}"
            )
        if self.method == "original_harness":
            if self.execution_layer is not ExecutionLayer.FULL_AGENT:
                raise ValueError("original_harness is only a full_agent condition")
            if self.handoff_enabled or self.handoff_config is not None:
                raise ValueError("original_harness must not enable handoff")
        if not self.handoff_enabled and self.handoff_config is not None:
            raise ValueError("disabled handoff condition cannot set handoff_config")
        if (
            self.execution_layer
            in {ExecutionLayer.FULL_AGENT, ExecutionLayer.CONTROLLED}
            and self.handoff_enabled
            and self.handoff_config is None
        ):
            raise ValueError(
                "handoff-enabled full_agent/controlled condition needs handoff_config"
            )
        return self


class ExperimentConfig(HandoffRecord):
    """Top-level strict matrix configuration."""

    schema_version: Literal[EXPERIMENT_CONFIG_SCHEMA_VERSION] = (
        EXPERIMENT_CONFIG_SCHEMA_VERSION
    )
    experiment_id: str
    output_root: str
    repeats: int = Field(default=1, gt=0)
    tasks: tuple[TaskSpec, ...]
    conditions: tuple[ConditionSpec, ...]
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    runtime_probes: tuple[RuntimeProbeReference, ...] = ()
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    source_revision: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str) -> str:
        return _safe_identifier(value, "experiment_id")

    @field_validator("output_root")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        value = _non_empty(value, info.field_name)
        if "\x00" in value:
            raise ValueError("output_root contains a NUL byte")
        return value

    @field_validator("source_revision")
    @classmethod
    def validate_source_revision(cls, value: str) -> str:
        return _non_empty(value, "source_revision")

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_value(value, name="metadata")

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        if not self.tasks:
            raise ValueError("tasks must not be empty")
        if not self.conditions:
            raise ValueError("conditions must not be empty")
        names = [condition.name for condition in self.conditions]
        if len(names) != len(set(names)):
            raise ValueError("condition names must be unique")
        probe_names = [probe.name for probe in self.runtime_probes]
        if len(probe_names) != len(set(probe_names)):
            raise ValueError("runtime probe names must be unique")
        task_keys = [(task.suite, task.task) for task in self.tasks]
        if len(task_keys) != len(set(task_keys)):
            raise ValueError("suite/task entries must be unique")
        if self.runtime.pi05_checkpoint_id is None:
            raise ValueError("runtime.pi05_checkpoint_id is required")
        if self.runtime.sam3_checkpoint_id is None:
            raise ValueError("runtime.sam3_checkpoint_id is required")
        has_full_agent = any(
            condition.execution_layer is ExecutionLayer.FULL_AGENT
            for condition in self.conditions
        )
        if has_full_agent:
            if self.planner.model is None:
                raise ValueError(
                    "full-agent conditions require an explicit planner.model"
                )
            if self.planner.backend == "api" and self.planner.base_url is None:
                raise ValueError(
                    "API full-agent conditions require an explicit planner.base_url"
                )
        return self

    @property
    def configuration_id(self) -> str:
        """Stable identity of the complete normalized configuration."""
        return stable_identifier("cfg", self)


def load_experiment_config(path: str | os.PathLike[str]) -> ExperimentConfig:
    """Load and strictly validate one JSON experiment config."""
    def expand(value: Any) -> Any:
        if isinstance(value, str):
            resolved = os.path.expandvars(value)
            if re.search(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*|%[^%]+%", resolved):
                raise ValueError(f"unresolved environment variable in {value!r}")
            return resolved
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    return ExperimentConfig.model_validate(expand(load_strict_json(path)))


def write_experiment_config(
    config: ExperimentConfig,
    path: str | os.PathLike[str],
) -> Path:
    """Atomically write normalized configuration JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(
            config.model_dump(mode="json", exclude_none=False),
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


def resolve_reference(reference: str, *, config_path: str | os.PathLike[str] | None) -> Path:
    """Resolve a config-relative artifact reference without requiring it to exist."""
    candidate = Path(reference).expanduser()
    if candidate.is_absolute() or config_path is None:
        return candidate.resolve()
    return (Path(config_path).resolve().parent / candidate).resolve()


def is_finite_number(value: Any) -> bool:
    """Return true only for finite int/float values, excluding booleans."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


__all__ = [
    "ConditionSpec",
    "DecisionMode",
    "EvidenceMode",
    "ExecutionLayer",
    "ExperimentConfig",
    "FeatureSetName",
    "HierarchyMode",
    "PlannerConfig",
    "RuntimeConfig",
    "TaskSpec",
    "UncertaintyMode",
    "canonical_json",
    "load_experiment_config",
    "load_strict_json",
    "resolve_reference",
    "stable_digest",
    "stable_identifier",
    "write_experiment_config",
]
