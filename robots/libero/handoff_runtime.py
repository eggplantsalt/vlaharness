"""Opt-in runtime configuration and lazy core bindings for LIBERO handoff.

This module intentionally has no simulator, CUDA, SAM3, or Pi0.5 imports.  It
is safe to import while parsing CLI configuration, before heavyweight services
are started.  The default RPent path never imports the research core unless a
validated handoff configuration is explicitly enabled.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


HANDOFF_RUNTIME_SCHEMA_VERSION = "rpent.libero-handoff-runtime/v1"
DEFAULT_GOVERNOR_FACTORY = "rpent.research.handoff.governor:build_governor"


class HandoffConfigurationError(ValueError):
    """Raised before service startup when opt-in handoff config is invalid."""


class _FrozenDict(dict):
    """JSON-serializable dict that rejects mutation after configuration load."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("handoff runtime configuration is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


class _FrozenList(list):
    """JSON-serializable list that rejects mutation after configuration load."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("handoff runtime configuration is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return _FrozenList(_freeze_json(item) for item in value)
    return value


def _require_json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HandoffConfigurationError(f"{name} must be a JSON object")
    try:
        # Copy through JSON so callers cannot retain mutable/non-JSON values.
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise HandoffConfigurationError(
            f"{name} must contain finite JSON values only"
        ) from exc


def _validate_factory_path(value: Any, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip() or ":" not in value:
        raise HandoffConfigurationError(
            f"{name} must be an import path in 'module:callable' form"
        )
    if any(ord(char) < 32 for char in value):
        raise HandoffConfigurationError(f"{name} must not contain control characters")
    module_name, _, attr_name = value.strip().partition(":")
    if not module_name or not attr_name:
        raise HandoffConfigurationError(
            f"{name} must be an import path in 'module:callable' form"
        )
    return value.strip()


def _validate_output_subdir(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HandoffConfigurationError("output_subdir must be a non-empty string")
    if any(ord(char) < 32 for char in value):
        raise HandoffConfigurationError("output_subdir must not contain control characters")
    # Persist one platform-independent spelling so a config prepared on
    # Windows has the same canonical identity when executed on Linux.
    path = PurePosixPath(value.strip().replace("\\", "/"))
    if not path.parts or value.strip() == ".":
        raise HandoffConfigurationError(
            "output_subdir must name a dedicated research subdirectory"
        )
    if path.is_absolute() or ".." in path.parts:
        raise HandoffConfigurationError(
            "output_subdir must stay below the RPent run output directory"
        )
    invalid_windows_chars = set('<>:"|?*')
    reserved_windows_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
    if any(
        any(char in invalid_windows_chars for char in part)
        or any(ord(char) < 32 for char in part)
        or part.endswith((" ", "."))
        or part.split(".", 1)[0].upper() in reserved_windows_names
        for part in path.parts
    ):
        raise HandoffConfigurationError("output_subdir must be Windows-safe")
    reserved = {
        "images",
        "images_cam",
        "depths",
        "world",
        "images_cam_hi",
        "world_hi",
        "images_wrist",
        "depths_wrist",
        "world_wrist",
        "wrist_meta",
        "images_wrist_hi",
        "world_wrist_hi",
        "segments",
        "action_videos",
        "camera_meta.json",
        "states.json",
        "episode.mp4",
        "run.log",
        "env_server.log",
        "vla_server.log",
        "sam3_server.log",
    }
    if path.parts and path.parts[0].lower() in reserved:
        raise HandoffConfigurationError(
            "output_subdir collides with an Original RPent artifact directory"
        )
    if path.suffix.lower() in {".json", ".jsonl", ".log", ".mp4"}:
        raise HandoffConfigurationError(
            "output_subdir must name a directory, not a run artifact file"
        )
    return path.as_posix()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HandoffConfigurationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise HandoffConfigurationError(f"{name} must be a finite number")
    return result


def _position(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise HandoffConfigurationError(f"{name} must contain exactly three numbers")
    return [_finite_number(item, name) for item in value]


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HandoffConfigurationError(f"{name} must be a non-empty string")
    if any(ord(char) < 32 for char in value):
        raise HandoffConfigurationError(f"{name} must not contain control characters")
    return value.strip()


def _validate_target_provider(value: Any) -> dict[str, Any]:
    provider = _require_json_object(value, "target_provider")
    kind = provider.get("kind", "perception")
    if kind not in {"perception", "injected", "oracle"}:
        raise HandoffConfigurationError(
            "target_provider.kind must be perception, injected, or oracle"
        )
    allowed_by_kind = {
        "perception": {"kind", "camera", "min_score", "provider_version"},
        "injected": {
            "kind",
            "position_m",
            "availability",
            "provider",
            "confidence",
        },
        "oracle": {"kind", "position_key"},
    }
    unknown = sorted(set(provider).difference(allowed_by_kind[kind]))
    if unknown:
        raise HandoffConfigurationError(
            f"unknown target_provider fields for {kind}: {', '.join(unknown)}"
        )
    provider["kind"] = kind
    if kind == "perception":
        camera = provider.get("camera", "agentview")
        if camera not in {"agentview", "wrist"}:
            raise HandoffConfigurationError(
                "target_provider.camera must be agentview or wrist"
            )
        provider["camera"] = camera
        min_score = _finite_number(
            provider.get("min_score", 0.2), "target_provider.min_score"
        )
        if not 0.0 <= min_score <= 1.0:
            raise HandoffConfigurationError(
                "target_provider.min_score must be in [0, 1]"
            )
        provider["min_score"] = min_score
        provider["provider_version"] = _non_empty_string(
            provider.get("provider_version", "rpent-current-rgbd-sam3/v1"),
            "target_provider.provider_version",
        )
    elif kind == "injected":
        if provider.get("position_m") is not None:
            provider["position_m"] = _position(
                provider["position_m"], "target_provider.position_m"
            )
        availability = provider.get("availability", "deployment_perception")
        online_availability = {
            "deployment_sensor",
            "deployment_perception",
            "derived_deployment",
        }
        if availability not in online_availability:
            raise HandoffConfigurationError(
                "injected targets are online policy inputs and availability must be "
                "deployment_sensor, deployment_perception, or derived_deployment; "
                "use the separate oracle provider for privileged state"
            )
        provider["availability"] = availability
        provider["provider"] = _non_empty_string(
            provider.get("provider", "injected_precomputed/v1"),
            "target_provider.provider",
        )
        confidence = _finite_number(
            provider.get("confidence", 1.0), "target_provider.confidence"
        )
        if not 0.0 <= confidence <= 1.0:
            raise HandoffConfigurationError(
                "target_provider.confidence must be in [0, 1]"
            )
        provider["confidence"] = confidence
    else:
        provider["position_key"] = _non_empty_string(
            provider.get("position_key"), "target_provider.position_key"
        )
    return provider


_STAGE_DEFAULT_KEYS = {
    "stage_gripper",
    "action_scale",
    "stage_tolerance_m",
    "stage_pitch_step_rad",
    "stage_yaw_step_rad",
    "stage_orientation_tolerance_rad",
}
_TOOL_DEFAULT_KEYS = {
    "handoff_pi0_pick": _STAGE_DEFAULT_KEYS
    | {"max_chunks", "lift_thresh", "gripper_closed_thresh"},
    "handoff_pi0_doubled": _STAGE_DEFAULT_KEYS | {"max_chunks"},
}


def _validate_tool_defaults(value: Any) -> dict[str, Any]:
    defaults = _require_json_object(value, "tool_defaults")
    unknown_tools = sorted(set(defaults).difference(_TOOL_DEFAULT_KEYS))
    if unknown_tools:
        raise HandoffConfigurationError(
            "unknown tool_defaults entries: " + ", ".join(unknown_tools)
        )
    for tool_name, payload in defaults.items():
        tool = _require_json_object(payload, f"tool_defaults.{tool_name}")
        unknown = sorted(set(tool).difference(_TOOL_DEFAULT_KEYS[tool_name]))
        if unknown:
            raise HandoffConfigurationError(
                f"unknown tool_defaults.{tool_name} fields: {', '.join(unknown)}"
            )
        if "max_chunks" in tool:
            chunks = tool["max_chunks"]
            if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 1:
                raise HandoffConfigurationError(
                    f"tool_defaults.{tool_name}.max_chunks must be a positive integer"
                )
        for key in (
            "lift_thresh",
            "gripper_closed_thresh",
            "stage_tolerance_m",
            "stage_orientation_tolerance_rad",
        ):
            if key in tool:
                number = _finite_number(
                    tool[key], f"tool_defaults.{tool_name}.{key}"
                )
                if number < 0.0:
                    raise HandoffConfigurationError(
                        f"tool_defaults.{tool_name}.{key} must be non-negative"
                    )
                tool[key] = number
        for key in ("action_scale", "stage_pitch_step_rad", "stage_yaw_step_rad"):
            if key in tool:
                number = _finite_number(
                    tool[key], f"tool_defaults.{tool_name}.{key}"
                )
                if number <= 0.0:
                    raise HandoffConfigurationError(
                        f"tool_defaults.{tool_name}.{key} must be positive"
                    )
                tool[key] = number
        if "stage_gripper" in tool:
            gripper = _finite_number(
                tool["stage_gripper"],
                f"tool_defaults.{tool_name}.stage_gripper",
            )
            if not -1.0 <= gripper <= 1.0:
                raise HandoffConfigurationError(
                    f"tool_defaults.{tool_name}.stage_gripper must be in [-1, 1]"
                )
            tool["stage_gripper"] = gripper
        defaults[tool_name] = tool
    return defaults


def _resolve_config_path(value: Any, name: str, *, base_dir: Path) -> str:
    path_text = os.path.expandvars(_non_empty_string(value, name))
    if "$" in path_text or re.search(r"%[A-Za-z_][A-Za-z0-9_]*%", path_text):
        raise HandoffConfigurationError(
            f"{name} contains an unresolved environment variable: {path_text!r}"
        )
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _resolve_core_paths(core: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    resolved = _require_json_object(core, "core")
    if "model_artifact" in resolved and resolved["model_artifact"] is not None:
        resolved["model_artifact"] = _resolve_config_path(
            resolved["model_artifact"], "core.model_artifact", base_dir=base_dir
        )
    for policy_name in ("policy", "fallback_policy"):
        policy = resolved.get(policy_name)
        if policy is None:
            continue
        if not isinstance(policy, dict):
            # The governor factory's semantic preflight reports the full policy
            # schema error; there is no path to resolve in a non-object value.
            continue
        artifact = policy.get("positive_references_file")
        if artifact is not None:
            policy["positive_references_file"] = _resolve_config_path(
                artifact,
                f"core.{policy_name}.positive_references_file",
                base_dir=base_dir,
            )
    return resolved


def _validate_metadata(value: Any) -> dict[str, Any]:
    metadata = _require_json_object(value, "metadata")
    for key in ("run_id", "episode_id", "trial_id", "suite"):
        if key in metadata:
            metadata[key] = _non_empty_string(metadata[key], f"metadata.{key}")
    if "task" in metadata and (
        isinstance(metadata["task"], bool)
        or not isinstance(metadata["task"], (int, str))
        or (isinstance(metadata["task"], str) and not metadata["task"].strip())
    ):
        raise HandoffConfigurationError("metadata.task must be an integer or string")
    if "seed" in metadata and (
        isinstance(metadata["seed"], bool) or not isinstance(metadata["seed"], int)
    ):
        raise HandoffConfigurationError("metadata.seed must be an integer")
    if "repeat_index" in metadata and (
        isinstance(metadata["repeat_index"], bool)
        or not isinstance(metadata["repeat_index"], int)
        or metadata["repeat_index"] < 0
    ):
        raise HandoffConfigurationError(
            "metadata.repeat_index must be a non-negative integer"
        )
    if "reset_id" in metadata:
        reset_id = metadata["reset_id"]
        if (
            isinstance(reset_id, bool)
            or not isinstance(reset_id, (int, str))
            or (isinstance(reset_id, str) and not reset_id.strip())
        ):
            raise HandoffConfigurationError(
                "metadata.reset_id must be an integer or non-empty string"
            )
    return metadata


def _configuration_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class HandoffRuntimeConfig:
    """Strict, JSON-backed configuration for the LIBERO integration only."""

    source_path: Path
    schema_version: str
    enabled: bool
    governor_factory: str
    governor_api_module: str
    governor_config: dict[str, Any]
    target_provider: dict[str, Any]
    tool_defaults: dict[str, Any]
    instrumentation: bool
    output_subdir: str
    sink_factory: str | None
    sink_config: dict[str, Any]
    oracle_ablation: bool
    controller_method: str
    controller_implementation_version: str
    checkpoint_id: str | None
    model_artifact_id: str | None
    metadata: dict[str, Any]
    configuration_id: str
    controller_configuration_id: str
    canonical_config: dict[str, Any]

    @property
    def output_relative_path(self) -> Path:
        return Path(self.output_subdir)


_CONFIG_KEYS = {
    "schema_version",
    "enabled",
    "governor_factory",
    "governor_api_module",
    "core",
    "governor",
    "candidate_generator",
    "candidate_feature_predictor",
    "policy",
    "fallback_policy",
    "feature_spec",
    "model_artifact",
    "trusted_model_artifact",
    "target_provider",
    "tool_defaults",
    "instrumentation",
    "output_subdir",
    "sink_factory",
    "sink",
    "oracle_ablation",
    "controller_method",
    "controller_implementation_version",
    "checkpoint_id",
    "model_artifact_id",
    "metadata",
}


def load_handoff_runtime_config(path: str | os.PathLike[str]) -> HandoffRuntimeConfig:
    """Load and validate the explicit opt-in JSON config.

    This function performs no dynamic imports.  It is therefore suitable for
    CLI/config validation before the env, VLA, or SAM3 processes are launched.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise HandoffConfigurationError(f"handoff config not found: {source}")
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HandoffConfigurationError(
                    f"duplicate JSON key in handoff config: {key!r}"
                )
            result[key] = value
        return result

    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise HandoffConfigurationError(
            f"handoff config is not valid JSON ({source}:{exc.lineno}:{exc.colno}): "
            f"{exc.msg}"
        ) from exc
    config = _require_json_object(raw, "handoff config")
    unknown = sorted(set(config).difference(_CONFIG_KEYS))
    if unknown:
        raise HandoffConfigurationError(
            f"unknown handoff config fields: {', '.join(unknown)}"
        )

    schema_version = config.get("schema_version", HANDOFF_RUNTIME_SCHEMA_VERSION)
    if schema_version != HANDOFF_RUNTIME_SCHEMA_VERSION:
        raise HandoffConfigurationError(
            f"unsupported handoff schema_version {schema_version!r}; "
            f"expected {HANDOFF_RUNTIME_SCHEMA_VERSION!r}"
        )
    enabled = config.get("enabled", True)
    instrumentation = config.get("instrumentation", True)
    oracle_ablation = config.get("oracle_ablation", False)
    for field_name, value in (
        ("enabled", enabled),
        ("instrumentation", instrumentation),
        ("oracle_ablation", oracle_ablation),
    ):
        if not isinstance(value, bool):
            raise HandoffConfigurationError(f"{field_name} must be boolean")

    governor_factory = _validate_factory_path(
        config.get("governor_factory", DEFAULT_GOVERNOR_FACTORY),
        "governor_factory",
    )
    assert governor_factory is not None
    default_api_module = governor_factory.partition(":")[0]
    governor_api_module = _non_empty_string(
        config.get("governor_api_module", default_api_module),
        "governor_api_module",
    )

    target_provider = _validate_target_provider(
        config.get("target_provider", {"kind": "perception"}),
    )
    provider_kind = target_provider["kind"]
    if provider_kind == "oracle" and not oracle_ablation:
        raise HandoffConfigurationError(
            "target_provider.kind='oracle' requires oracle_ablation=true"
        )
    if provider_kind != "oracle" and oracle_ablation:
        raise HandoffConfigurationError(
            "oracle_ablation=true requires target_provider.kind='oracle'"
        )

    controller_method = config.get("controller_method")
    implementation_version = config.get(
        "controller_implementation_version", "rpent-libero-handoff/v1"
    )
    implementation_version = _non_empty_string(
        implementation_version, "controller_implementation_version"
    )
    if controller_method is not None:
        controller_method = _non_empty_string(controller_method, "controller_method")
    checkpoint_id = config.get("checkpoint_id")
    if checkpoint_id is not None:
        checkpoint_id = _non_empty_string(checkpoint_id, "checkpoint_id")
    model_artifact_id = config.get("model_artifact_id")
    if model_artifact_id is not None:
        model_artifact_id = _non_empty_string(
            model_artifact_id, "model_artifact_id"
        )

    core_config = config.get("core")
    if core_config is not None:
        core_config = _require_json_object(core_config, "core")
        duplicated = sorted(
            key
            for key in (
                "governor",
                "candidate_generator",
                "candidate_feature_predictor",
                "policy",
                "fallback_policy",
                "feature_spec",
                "model_artifact",
                "trusted_model_artifact",
            )
            if key in config
        )
        if duplicated:
            raise HandoffConfigurationError(
                "core cannot be combined with duplicate top-level core fields: "
                + ", ".join(duplicated)
            )
    else:
        core_config = {
            key: config[key]
            for key in (
                "governor",
                "candidate_generator",
                "candidate_feature_predictor",
                "policy",
                "fallback_policy",
                "feature_spec",
                "model_artifact",
                "trusted_model_artifact",
            )
            if key in config
        }
    core_config.setdefault("policy", {"name": "direct_frozen_pi0"})
    if controller_method is None:
        policy_config = core_config.get("policy")
        policy_name = (
            policy_config.get("name") if isinstance(policy_config, dict) else None
        )
        controller_method = (
            policy_name.strip()
            if isinstance(policy_name, str) and policy_name.strip()
            else "configured_handoff_policy"
        )
    core_config.setdefault(
        "feature_spec",
        {"preset": "deployment_full", "skill_vocabulary": ["pick", "contact"]},
    )
    core_config = _resolve_core_paths(core_config, base_dir=source.parent)
    if "trusted_model_artifact" in core_config and not isinstance(
        core_config["trusted_model_artifact"], bool
    ):
        raise HandoffConfigurationError("core.trusted_model_artifact must be boolean")

    canonical = {
        "schema_version": schema_version,
        "enabled": enabled,
        "governor_factory": governor_factory,
        "governor_api_module": governor_api_module.strip(),
        "core": core_config,
        "target_provider": target_provider,
        "tool_defaults": _validate_tool_defaults(config.get("tool_defaults", {})),
        "instrumentation": instrumentation,
        "output_subdir": _validate_output_subdir(
            config.get("output_subdir", "handoff")
        ),
        "sink_factory": _validate_factory_path(
            config.get("sink_factory"), "sink_factory", optional=True
        ),
        "sink": _require_json_object(config.get("sink", {}), "sink"),
        "oracle_ablation": oracle_ablation,
        "controller_method": controller_method.strip(),
        "controller_implementation_version": implementation_version.strip(),
        "checkpoint_id": checkpoint_id.strip() if checkpoint_id is not None else None,
        "model_artifact_id": (
            model_artifact_id.strip() if model_artifact_id is not None else None
        ),
        "metadata": _validate_metadata(config.get("metadata", {})),
    }
    configuration_id = _configuration_hash(canonical)
    controller_configuration_id = _configuration_hash(
        {
            "schema_version": schema_version,
            "governor_factory": governor_factory,
            "governor_api_module": canonical["governor_api_module"],
            "core": core_config,
            "target_provider": target_provider,
            "tool_defaults": canonical["tool_defaults"],
            "oracle_ablation": oracle_ablation,
            "controller_method": canonical["controller_method"],
            "controller_implementation_version": canonical[
                "controller_implementation_version"
            ],
            "checkpoint_id": canonical["checkpoint_id"],
            "model_artifact_id": canonical["model_artifact_id"],
        }
    )
    frozen_canonical = _freeze_json(canonical)
    return HandoffRuntimeConfig(
        source_path=source,
        schema_version=schema_version,
        enabled=enabled,
        governor_factory=governor_factory,
        governor_api_module=canonical["governor_api_module"],
        governor_config=frozen_canonical["core"],
        target_provider=frozen_canonical["target_provider"],
        tool_defaults=frozen_canonical["tool_defaults"],
        instrumentation=instrumentation,
        output_subdir=canonical["output_subdir"],
        sink_factory=canonical["sink_factory"],
        sink_config=frozen_canonical["sink"],
        oracle_ablation=oracle_ablation,
        controller_method=canonical["controller_method"],
        controller_implementation_version=canonical[
            "controller_implementation_version"
        ],
        checkpoint_id=canonical["checkpoint_id"],
        model_artifact_id=canonical["model_artifact_id"],
        metadata=frozen_canonical["metadata"],
        configuration_id=configuration_id,
        controller_configuration_id=controller_configuration_id,
        canonical_config=frozen_canonical,
    )


def load_object(import_path: str) -> Any:
    module_name, _, attr_name = import_path.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise HandoffConfigurationError(
            f"could not import configured module {module_name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise HandoffConfigurationError(
            f"configured object {attr_name!r} is missing from {module_name!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class CoreRuntimeAPI:
    """Lazily resolved constructor surface shared with the pure governor."""

    GovernorInvocation: type
    EpisodeStatus: type
    StageResult: type
    VLAExecutionResult: type
    GovernorRunResult: type

    @classmethod
    def load(cls, module_name: str) -> "CoreRuntimeAPI":
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise HandoffConfigurationError(
                f"could not import handoff core module {module_name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        values: dict[str, type] = {}
        for name in (
            "GovernorInvocation",
            "EpisodeStatus",
            "StageResult",
            "VLAExecutionResult",
            "GovernorRunResult",
        ):
            value = getattr(module, name, None)
            if not isinstance(value, type):
                raise HandoffConfigurationError(
                    f"handoff core module {module_name!r} does not expose class {name}"
                )
            values[name] = value
        return cls(**values)


def _record_json(value: Any) -> str:
    if hasattr(value, "canonical_json"):
        return value.canonical_json()
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=False)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class RuntimeEventSink:
    """Append-only enabled-run trace implementing the governor sink protocol."""

    def __init__(
        self,
        output_dir: Path,
        *,
        configuration_id: str,
        controller_configuration_id: str | None = None,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.configuration_id = configuration_id
        self.controller_configuration_id = controller_configuration_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._needs_trailing_newline: dict[str, bool] = {}
        if manifest is not None:
            self._ensure_manifest(manifest)
        for filename in ("decisions.jsonl", "outcomes.jsonl", "runtime_events.jsonl"):
            self._needs_trailing_newline[filename] = self._prepare_jsonl(filename)

    def _prepare_jsonl(self, filename: str) -> bool:
        """Validate prior complete lines and recover only a torn final write."""
        def reject_constant(token: str) -> None:
            raise ValueError(f"non-finite JSON constant is forbidden: {token}")

        path = self.output_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            return False
        data = path.read_bytes()
        lines = data.splitlines(keepends=True)
        offset = 0
        for index, raw_line in enumerate(lines):
            is_last = index == len(lines) - 1
            complete = raw_line.endswith((b"\n", b"\r"))
            content = raw_line.rstrip(b"\r\n")
            if not content:
                raise HandoffConfigurationError(
                    f"blank line in existing handoff JSONL: {path}:{index + 1}"
                )
            try:
                decoded = json.loads(
                    content.decode("utf-8"),
                    parse_constant=reject_constant,
                )
                if not isinstance(decoded, dict):
                    raise ValueError("JSONL record must be an object")
            except (UnicodeDecodeError, ValueError) as exc:
                if is_last and not complete:
                    with path.open("r+b") as handle:
                        handle.truncate(offset)
                        handle.flush()
                        os.fsync(handle.fileno())
                    return False
                raise HandoffConfigurationError(
                    f"invalid existing handoff JSONL: {path}:{index + 1}: {exc}"
                ) from exc
            offset += len(raw_line)
        return not lines[-1].endswith((b"\n", b"\r"))

    def _ensure_manifest(self, manifest: Mapping[str, Any]) -> None:
        path = self.output_dir / "manifest.json"
        text = json.dumps(
            dict(manifest),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise HandoffConfigurationError(
                    f"existing handoff manifest is unreadable: {path}: {exc}"
                ) from exc
            if (
                not isinstance(existing, dict)
                or existing.get("schema_version")
                != manifest.get("schema_version")
                or existing.get("configuration_id")
                != manifest.get("configuration_id")
                or existing.get("controller_configuration_id")
                != manifest.get("controller_configuration_id")
                or existing.get("configuration") != manifest.get("configuration")
            ):
                raise HandoffConfigurationError(
                    "existing handoff manifest disagrees with the enabled run "
                    f"configuration: {path}"
                )

    def _append_line(self, filename: str, value: Any) -> None:
        path = self.output_dir / filename
        line = _record_json(value)
        with self._lock, path.open("a", encoding="utf-8", newline="\n") as handle:
            if self._needs_trailing_newline.get(filename, False):
                handle.write("\n")
            handle.write(line)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            self._needs_trailing_newline[filename] = False

    def append_decision(self, decision: Any) -> None:
        self._append_line("decisions.jsonl", decision)

    def append_outcome(self, outcome: Any) -> None:
        self._append_line("outcomes.jsonl", outcome)

    def append_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self._append_line(
            "runtime_events.jsonl",
            {
                "schema_version": "rpent.libero-handoff-event/v1",
                "configuration_id": self.configuration_id,
                "controller_configuration_id": self.controller_configuration_id,
                "event_type": event_type,
                "payload": dict(payload),
            },
        )


class CompositeResearchSink:
    """Fan out core and runtime records without weakening either sink."""

    def __init__(self, sinks: list[Any]) -> None:
        self._sinks = tuple(sinks)

    def append_decision(self, decision: Any) -> None:
        for sink in self._sinks:
            sink.append_decision(decision)

    def append_outcome(self, outcome: Any) -> None:
        for sink in self._sinks:
            sink.append_outcome(outcome)

    def append_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        for sink in self._sinks:
            append = getattr(sink, "append_event", None)
            if append is not None:
                append(event_type, payload)


def build_research_sink(
    config: HandoffRuntimeConfig,
    *,
    run_output_dir: Path,
) -> Any:
    output_dir = resolve_handoff_output_dir(config, run_output_dir=run_output_dir)
    local_sink = RuntimeEventSink(
        output_dir,
        configuration_id=config.configuration_id,
        controller_configuration_id=config.controller_configuration_id,
        manifest={
            "schema_version": "rpent.libero-handoff-run/v1",
            "configuration_id": config.configuration_id,
            "controller_configuration_id": config.controller_configuration_id,
            "source_config_path": str(config.source_path),
            "configuration": config.canonical_config,
        },
    )
    if config.sink_factory is None:
        return local_sink
    factory = load_object(config.sink_factory)
    external = factory(
        _require_json_object(config.sink_config, "sink"),
        output_dir=local_sink.output_dir,
        configuration_id=config.configuration_id,
    )
    for name in ("append_decision", "append_outcome"):
        if not callable(getattr(external, name, None)):
            raise HandoffConfigurationError(
                f"configured sink does not implement {name}()"
            )
    return CompositeResearchSink([local_sink, external])


def resolve_handoff_output_dir(
    config: HandoffRuntimeConfig,
    *,
    run_output_dir: Path,
) -> Path:
    """Resolve and validate the dedicated sink path without creating it."""
    run_root = run_output_dir.resolve()
    output_dir = (run_root / config.output_relative_path).resolve()
    try:
        output_dir.relative_to(run_root)
    except ValueError as exc:
        raise HandoffConfigurationError(
            "resolved handoff output directory escapes the run output root"
        ) from exc
    if output_dir == run_root:
        raise HandoffConfigurationError(
            "handoff output directory must be dedicated below the run root"
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise HandoffConfigurationError(
            f"handoff output path is an existing non-directory: {output_dir}"
        )
    return output_dir


def build_governor(config: HandoffRuntimeConfig, *, sink: Any) -> Any:
    factory = load_object(config.governor_factory)
    governor = factory(
        _require_json_object(config.governor_config, "core"),
        model=None,
        sink=sink,
        expected_model_artifact_id=config.model_artifact_id,
    )
    if not callable(getattr(governor, "run", None)):
        raise HandoffConfigurationError(
            "configured governor must implement run(adapter, invocation)"
        )
    return governor


def validate_handoff_runtime_bindings(config: HandoffRuntimeConfig) -> None:
    """Resolve configured pure-Python factories before heavyweight startup."""
    if not config.enabled:
        return
    factory = load_object(config.governor_factory)
    if not callable(factory):
        raise HandoffConfigurationError("governor_factory is not callable")
    CoreRuntimeAPI.load(config.governor_api_module)

    class _ValidationSink:
        def append_decision(self, decision: Any) -> None:
            del decision

        def append_outcome(self, outcome: Any) -> None:
            del outcome

    # Construct once during preflight so policy names, candidate/governor
    # settings, model artifact compatibility, and required optional research
    # dependencies fail before env/VLA/SAM3 services start. Runtime constructs a
    # fresh instance with the durable run sink after clients are ready.
    try:
        governor = factory(
            _require_json_object(config.governor_config, "core"),
            model=None,
            sink=_ValidationSink(),
            expected_model_artifact_id=config.model_artifact_id,
        )
    except HandoffConfigurationError:
        raise
    except Exception as exc:
        raise HandoffConfigurationError(
            "invalid handoff core configuration: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not callable(getattr(governor, "run", None)):
        raise HandoffConfigurationError(
            "configured governor must implement run(adapter, invocation)"
        )
    if config.sink_factory is not None:
        sink_factory = load_object(config.sink_factory)
        if not callable(sink_factory):
            raise HandoffConfigurationError("sink_factory is not callable")
        try:
            inspect.signature(sink_factory).bind(
                _require_json_object(config.sink_config, "sink"),
                output_dir=Path("handoff-sink-preflight"),
                configuration_id=config.configuration_id,
            )
        except (TypeError, ValueError) as exc:
            raise HandoffConfigurationError(
                "configured sink_factory does not accept the required runtime "
                f"arguments: {exc}"
            ) from exc


__all__ = [
    "CoreRuntimeAPI",
    "DEFAULT_GOVERNOR_FACTORY",
    "HANDOFF_RUNTIME_SCHEMA_VERSION",
    "HandoffConfigurationError",
    "HandoffRuntimeConfig",
    "RuntimeEventSink",
    "build_governor",
    "build_research_sink",
    "load_handoff_runtime_config",
    "resolve_handoff_output_dir",
    "validate_handoff_runtime_bindings",
]
