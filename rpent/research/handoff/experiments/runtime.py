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
from typing import Any, Callable, Literal, Mapping

from pydantic import Field, field_validator

from rpent.research.handoff.types import HandoffRecord


GATE0_JOB_SCHEMA_VERSION = "rpent.handoff-gate0-job/v1"
CONTROLLED_CHILD_PLAN_SCHEMA_VERSION = "rpent.handoff-controlled-plan/v1"
EXECUTION_CONFIRMATION = "I_UNDERSTAND_SERVER_EXECUTION"


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
    checkpoint_id: str | None = None
    configuration_id: str | None = None
    source_revision: str | None = None
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

    @field_validator("checkpoint_id", "configuration_id", "source_revision")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is not None and not value.strip():
            raise ValueError(f"{info.field_name} must be null or non-empty")
        return value

    @property
    def stable_configuration_id(self) -> str:
        if self.configuration_id is not None:
            return self.configuration_id
        scientific = self.model_dump(
            mode="json",
            exclude={
                "configuration_id",
                "output_dir",
                "run_id",
                "episode_prefix",
                "source_revision",
                "metadata",
            },
        )
        return "gate0-" + hashlib.sha256(
            json.dumps(
                scientific,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()[:20]


def load_gate0_job(path: str | os.PathLike[str]) -> Gate0JobSpec:
    """Load a Gate-0 job and resolve its output directory relative to it."""
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
    output = Path(job.output_dir).expanduser()
    if not output.is_absolute():
        output = source.parent / output
    return job.model_copy(update={"output_dir": str(output.resolve())})


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
                previous = self._records.setdefault(record.record_id, canonical)
                if previous != canonical:
                    raise ValueError(
                        f"conflicting setup record ID in {self.path}: {record.record_id}"
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
    command = (
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
    payload = json.dumps(
        {
            "trial_id": trial.trial_id,
            "command": command,
            "cwd": str(Path(repo_root).expanduser().resolve()),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return ControlledChildPlan(
        plan_id="controlled-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20],
        trial_id=trial.trial_id,
        command=command,
        cwd=str(Path(repo_root).expanduser().resolve()),
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
    return subprocess.run(
        list(plan.command),
        cwd=plan.cwd,
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


def _write_json_atomic(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
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


def write_resolved_handoff_config(trial: Any) -> Path:
    """Create a trial-local runtime config with immutable manifest identity."""
    from rpent.research.handoff.experiments.config import load_strict_json

    if trial.handoff_config_path is None:
        raise ValueError("handoff-enabled trial has no source handoff config")
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
        "execution_layer": trial.execution_layer.value,
        "source_revision": trial.source_revision,
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
    return _write_json_atomic(destination, resolved)


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


def _verify_controlled_outcome(trial: Any, runtime_config: Any) -> tuple[Path, Any]:
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
    return path, records[0]


def run_controlled_trial(
    manifest_path: str | os.PathLike[str],
    trial_id: str,
) -> dict[str, Any]:
    """Run one manifest-bound controlled trial through the normal LIBERO toolkit."""
    # All environment/server-heavy imports are intentionally inside this child.
    import argparse

    from robots.libero.handoff_runtime import load_handoff_runtime_config
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.envs import get_env_spec, get_toolkit
    from rpent.research.handoff.experiments.config import ExecutionLayer
    from rpent.research.handoff.experiments.manifest import load_manifest
    from rpent.utils.logging import init_output_dir

    manifest = load_manifest(manifest_path)
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

    output_dir = init_output_dir(Path(trial.output_dir))
    resolved_handoff_path = write_resolved_handoff_config(trial)
    handoff_config = load_handoff_runtime_config(resolved_handoff_path)
    existing_outcome_path = (
        Path(trial.output_dir) / handoff_config.output_subdir / "outcomes.jsonl"
    )
    if existing_outcome_path.exists():
        outcome_path, existing_outcome = _verify_controlled_outcome(
            trial, handoff_config
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
    if trial.runtime.pi05_checkpoint_path is not None:
        os.environ["PI05_CHECKPOINT_PATH"] = trial.runtime.pi05_checkpoint_path
    if trial.runtime.sam3_checkpoint_path is not None:
        os.environ["SAM3_CHECKPOINT_PATH"] = trial.runtime.sam3_checkpoint_path

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
            trial, handoff_config
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
    "Gate0JobSpec",
    "SetupJsonlSink",
    "build_controlled_child_plan",
    "execute_controlled_child_plan",
    "instantiate_gate0_adapter",
    "load_gate0_job",
    "load_object",
    "run_controlled_trial",
    "write_controlled_plans",
    "write_resolved_handoff_config",
]
