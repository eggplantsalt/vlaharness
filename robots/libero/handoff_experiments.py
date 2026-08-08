"""Server-only execution adapters for controller-handoff experiments.

This module is intentionally outside the pure research core.  Its Gate-0
adapter may inspect one explicitly configured privileged observation key for
controlled setup, but the inherited online governor observation still passes
through the deployment-key whitelist in :mod:`robots.libero.handoff`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from pydantic import Field, field_validator, model_validator

from robots.libero.handoff import (
    LiberoGovernorAdapter,
    RuntimeInstrumentation,
    TargetRequest,
    build_target_provider,
    instrument_primitives_kwargs,
)
from robots.libero.handoff_runtime import (
    CompositeResearchSink,
    CoreRuntimeAPI,
    HandoffConfigurationError,
    RuntimeEventSink,
    load_handoff_runtime_config,
)
from rpent.research.handoff.dataset import (
    DatasetResearchSink,
    scan_decision_jsonl,
    scan_outcome_jsonl,
)
from rpent.research.handoff.experiments.config import RuntimeConfig, load_strict_json
from rpent.research.handoff.experiments.gate0 import (
    Gate0Collector,
    Gate0Config,
    Gate0RunIdentity,
    Gate0Setup,
)
from rpent.research.handoff.experiments.sampling import (
    Gate0Sample,
    sample_world_position,
)
from rpent.research.handoff.experiments.setup_data import SetupJsonlWriter
from rpent.research.handoff.experiments.runtime_identity import (
    attest_runtime_checkpoint_clients,
    write_runtime_attestation,
)
from rpent.research.handoff.privileged import ExperimentSetupRecord, SetupValue
from rpent.research.handoff.types import (
    CandidateGeometry,
    ControllerIdentity,
    HandoffRecord,
    OutcomeLabels,
    SkillIdentity,
    TrialIdentity,
)

GATE0_SERVER_SCHEMA_VERSION = "rpent.libero-gate0/v1"


class Gate0TaskConfig(HandoffRecord):
    suite: str
    task: int = Field(ge=0)
    seed: int = Field(ge=0)
    target_id: str
    target_description: str
    skill_name: str
    skill_prompt: str
    vla_method: Literal["pi0_pick", "pi0_doubled"] = "pi0_pick"
    training_target_label: Literal[
        "primitive_success", "skill_success", "task_success", "episode_truncated"
    ] = "primitive_success"

    @field_validator(
        "suite", "target_id", "target_description", "skill_name", "skill_prompt"
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value


class SetupTargetConfig(HandoffRecord):
    """Explicit simulator/setup-only target extraction contract."""

    raw_observation_key: str
    flat_indices: tuple[int, int, int] = (0, 1, 2)
    frame: str = "libero_world"
    unit: str = "m"
    provider_id: str = "libero_raw_setup/v1"
    reset_info_key: str | None = None

    @field_validator("raw_observation_key", "frame", "unit", "provider_id")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("flat_indices")
    @classmethod
    def validate_indices(
        cls, value: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        if any(index < 0 for index in value) or len(set(value)) != 3:
            raise ValueError("flat_indices must contain three unique non-negative indices")
        return value


class Gate0ServerConfig(HandoffRecord):
    schema_version: Literal[GATE0_SERVER_SCHEMA_VERSION] = GATE0_SERVER_SCHEMA_VERSION
    run_id: str
    output_dir: str
    handoff_config: str
    task: Gate0TaskConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    setup_target: SetupTargetConfig
    gate0: Gate0Config
    vla_kwargs: dict[str, Any] = Field(default_factory=dict)
    tool_defaults: dict[str, Any] = Field(default_factory=dict)
    labeler_factory: str | None = None
    source_revision: str | None = None
    fsync: bool = True

    @field_validator("run_id", "output_dir", "handoff_config")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("labeler_factory")
    @classmethod
    def validate_factory(cls, value: str | None) -> str | None:
        if value is not None and (not value or ":" not in value):
            raise ValueError("labeler_factory must use module:callable syntax")
        return value

    @field_validator("vla_kwargs")
    @classmethod
    def validate_vla_kwargs(cls, value: dict[str, Any]) -> dict[str, Any]:
        max_chunks = value.get("max_chunks")
        if max_chunks is not None and (
            isinstance(max_chunks, bool)
            or not isinstance(max_chunks, int)
            or max_chunks < 1
        ):
            raise ValueError("vla_kwargs.max_chunks must be a positive integer")
        return value

    @model_validator(mode="after")
    def validate_target_label(self) -> Self:
        if self.task.training_target_label == "skill_success" and not self.labeler_factory:
            raise ValueError(
                "skill_success collection requires an explicit labeler_factory; "
                "labels are never inferred from primitive/task success"
            )
        _runtime_checkpoint_environment(self.runtime)
        return self


class LiberoGate0FactoryConfig(HandoffRecord):
    """Adapter-specific portion of the generic Gate0JobSpec."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    handoff_config: str
    vla_method: Literal["pi0_pick", "pi0_doubled"] = "pi0_pick"
    setup_target: SetupTargetConfig
    tool_defaults: dict[str, Any] = Field(default_factory=dict)
    labeler_factory: str | None = None
    training_target_label: Literal[
        "primitive_success", "skill_success", "task_success", "episode_truncated"
    ] = "primitive_success"

    @field_validator("handoff_config")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_labeler(self) -> Self:
        if self.training_target_label == "skill_success" and not self.labeler_factory:
            raise ValueError(
                "skill_success Gate-0 data requires an explicit labeler_factory"
            )
        _runtime_checkpoint_environment(self.runtime)
        return self


def _runtime_checkpoint_environment(runtime: RuntimeConfig) -> dict[str, str]:
    values = {
        "PI05_CHECKPOINT_PATH": runtime.pi05_checkpoint_path,
        "SAM3_CHECKPOINT_PATH": runtime.sam3_checkpoint_path,
        "RPENT_PI05_CHECKPOINT_ID": runtime.pi05_checkpoint_id,
        "RPENT_SAM3_CHECKPOINT_ID": runtime.sam3_checkpoint_id,
    }
    missing = sorted(name for name, value in values.items() if value is None)
    if missing:
        raise ValueError(
            "Gate-0 runtime requires four checkpoint path/ID bindings: "
            + ", ".join(missing)
        )
    return {name: str(value) for name, value in values.items()}


@contextmanager
def _injected_checkpoint_environment(runtime: RuntimeConfig):
    values = _runtime_checkpoint_environment(runtime)
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield values
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _assert_checkpoint_environment(runtime: RuntimeConfig) -> None:
    expected = _runtime_checkpoint_environment(runtime)
    actual = {name: os.environ.get(name) for name in expected}
    if actual != expected:
        raise ValueError(
            "Gate-0 child checkpoint environment disagrees with its job: "
            f"expected={expected!r}, actual={actual!r}"
        )


def _resolve_config_paths(config: Gate0ServerConfig, source: Path) -> Gate0ServerConfig:
    def resolved(value: str) -> str:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = source.parent / path
        return str(path.resolve())

    runtime_updates: dict[str, str] = {}
    for name in ("pi05_checkpoint_path", "sam3_checkpoint_path"):
        value = getattr(config.runtime, name)
        if value is not None:
            runtime_updates[name] = resolved(value)
    runtime = (
        config.runtime.model_copy(update=runtime_updates)
        if runtime_updates
        else config.runtime
    )
    return config.model_copy(
        update={
            "output_dir": resolved(config.output_dir),
            "handoff_config": resolved(config.handoff_config),
            "runtime": runtime,
        }
    )


def load_gate0_server_config(path: str | Path) -> Gate0ServerConfig:
    source = Path(path).expanduser().resolve()
    def expand(value: Any) -> Any:
        if isinstance(value, str):
            resolved = os.path.expandvars(value)
            if "$" in resolved or ("%" in resolved and resolved.count("%") >= 2):
                raise ValueError(f"unresolved environment variable in {value!r}")
            return resolved
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    config = Gate0ServerConfig.model_validate(expand(load_strict_json(source)))
    return _resolve_config_paths(config, source)


def _load_factory(path: str) -> Callable[..., Any]:
    module_name, _, attribute = path.partition(":")
    value = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(value):
        raise TypeError(f"configured factory is not callable: {path}")
    return value


def _setup_record_id(identity: TrialIdentity, sample: Gate0Sample) -> str:
    payload = json.dumps(
        [identity.model_dump(mode="json"), sample.model_dump(mode="json")],
        sort_keys=True,
        separators=(",", ":"),
    )
    return "setup-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


class LiberoGate0Adapter(LiberoGovernorAdapter):
    """Gate-0 setup wrapper over the deployment-realistic governor adapter."""

    def __init__(
        self,
        *,
        setup_target: SetupTargetConfig,
        sampler_config,
        outcome_labeler: Callable[[Any], OutcomeLabels] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._setup_target = setup_target
        self._sampler_config = sampler_config
        self._outcome_labeler = outcome_labeler
        self._env_runtime_meta: dict[str, Any] = {}
        runtime_probe = getattr(self.primitives.env, "runtime_probe", None)
        if callable(runtime_probe):
            try:
                payload = runtime_probe()
            except Exception as exc:
                raise HandoffConfigurationError(
                    "Gate-0 env runtime capability probe failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise HandoffConfigurationError(
                    "Gate-0 env runtime probe returned a non-object"
                )
            if payload.get("schema_version") != "rpent.runtime-probe/v1":
                raise HandoffConfigurationError(
                    "Gate-0 env runtime probe has an unsupported schema_version: "
                    f"{payload.get('schema_version')!r}"
                )
            if payload.get("component") != "libero_env":
                raise HandoffConfigurationError(
                    "Gate-0 env runtime probe component must be 'libero_env', "
                    f"got {payload.get('component')!r}"
                )
            runtime_meta = payload.get("runtime_meta", {})
            if not isinstance(runtime_meta, Mapping):
                raise HandoffConfigurationError(
                    "Gate-0 env runtime probe runtime_meta is not an object"
                )
            self._env_runtime_meta = dict(runtime_meta)

    def reset_for_trial(
        self,
        identity: TrialIdentity,
        skill: SkillIdentity,
        sample: Gate0Sample,
    ) -> Gate0Setup:
        del skill
        _, info = self.primitives.reset()
        self._observation_sequence = 0
        self._started_monotonic = time.monotonic()
        self._invocation_id = identity.invocation_id
        raw = self.primitives.env.raw_obs()
        if not isinstance(raw, Mapping):
            raise RuntimeError("LIBERO raw_obs did not return a setup mapping")
        raw_value = raw.get(self._setup_target.raw_observation_key)
        if raw_value is None:
            raise RuntimeError(
                "configured setup target key is absent: "
                f"{self._setup_target.raw_observation_key}"
            )
        flattened = np.asarray(raw_value, dtype=np.float64).reshape(-1)
        indices = self._setup_target.flat_indices
        if max(indices) >= flattened.size:
            raise RuntimeError(
                f"setup target indices {indices} exceed flattened key shape "
                f"{np.asarray(raw_value).shape}"
            )
        target = tuple(float(flattened[index]) for index in indices)
        if not np.isfinite(target).all():
            raise RuntimeError("setup target contains NaN/Inf")

        runtime_reset = self._env_runtime_meta.get("reset_id")
        info_reset = None
        if self._setup_target.reset_info_key and isinstance(info, Mapping):
            info_reset = info.get(self._setup_target.reset_info_key)
        if (
            runtime_reset is not None
            and info_reset is not None
            and str(runtime_reset) != str(info_reset)
        ):
            raise RuntimeError(
                "live LIBERO runtime reset_id disagrees with configured reset "
                f"info: {runtime_reset!r} != {info_reset!r}"
            )
        if runtime_reset is not None:
            reset_id = str(runtime_reset)
            reset_source = "verified env runtime probe"
        elif info_reset is not None:
            reset_id = str(info_reset)
            reset_source = (
                f"reset info field {self._setup_target.reset_info_key!r}"
            )
        else:
            raise RuntimeError(
                "Gate-0 requires a source-verified reset_id from the env runtime "
                "probe or configured reset info; logical per-trial reset IDs are "
                "forbidden"
            )

        resolved_identity = identity.model_copy(update={"reset_id": reset_id})
        desired_position = sample_world_position(
            sample,
            target_position_m=target,
            approach_axis_world=self._sampler_config.approach_axis_world,
        )
        requested = CandidateGeometry(
            # Geometry identity is repeat-invariant. ``sample_id`` identifies
            # this execution attempt and belongs in the trial/outcome key, not
            # in the requested candidate stored in privileged setup data.
            candidate_id=sample.candidate_id,
            kind="perturbation",
            eef_position_m=desired_position,
            target_relative_position_m=tuple(
                float(desired - origin)
                for desired, origin in zip(desired_position, target)
            ),
            wrist_yaw_rad=sample.wrist_yaw_rad,
            wrist_pitch_rad=sample.wrist_pitch_rad,
            requested_standoff_m=sample.standoff_m,
        )
        record = ExperimentSetupRecord(
            record_id=_setup_record_id(resolved_identity, sample),
            identity=resolved_identity,
            setup_provider=self._setup_target.provider_id,
            requested_candidate=requested,
            values=(
                SetupValue(
                    name="target_position_m",
                    values=target,
                    unit=self._setup_target.unit,
                    frame=self._setup_target.frame,
                    source=(
                        "raw_obs."
                        f"{self._setup_target.raw_observation_key}"
                        f"[flat_indices={list(indices)}]"
                    ),
                ),
            ),
            notes=(
                "setup-only privileged value; forbidden as online policy input",
                f"reset identity came from {reset_source}",
            ),
        )
        return Gate0Setup(
            record=record,
            target_position_m=target,
            reset_id=reset_id,
        )

    def current_eef_position_m(self) -> tuple[float, float, float]:
        value = np.asarray(self.primitives._last_obs_eef_pos, dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise RuntimeError("current EEF position is unavailable or invalid")
        return tuple(float(item) for item in value)

    def current_eef_quaternion_xyzw(self) -> tuple[float, float, float, float]:
        raw = self.primitives.env.raw_obs()
        if not isinstance(raw, Mapping):
            raise RuntimeError("LIBERO raw_obs did not return a state mapping")
        value = np.asarray(raw.get("robot0_eef_quat"), dtype=np.float64)
        if value.shape != (4,) or not np.isfinite(value).all():
            raise RuntimeError("current EEF quaternion is unavailable or invalid")
        norm = float(np.linalg.norm(value))
        if norm < 1e-12:
            raise RuntimeError("current EEF quaternion has zero norm")
        return tuple(float(item / norm) for item in value)

    def label_outcome(self, result: Any) -> OutcomeLabels:
        if self._outcome_labeler is not None:
            labels = self._outcome_labeler(result)
            if not isinstance(labels, OutcomeLabels):
                labels = OutcomeLabels.model_validate(labels)
            return labels
        return super().label_outcome(result)


def _runtime_namespace(config: Gate0ServerConfig) -> argparse.Namespace:
    runtime = config.runtime
    return argparse.Namespace(
        suite=config.task.suite,
        task=config.task.task,
        seed=config.task.seed,
        max_episode_steps=runtime.max_episode_steps,
        libero_type=runtime.libero_type,
        env_endpoint=runtime.env_endpoint,
        vla_endpoint=runtime.vla_endpoint,
        sam3_endpoint=runtime.sam3_endpoint,
        cuda_device=runtime.cuda_device,
        handoff_config=config.handoff_config,
        output_dir=config.output_dir,
    )


def build_gate0_adapter_bundle(
    *,
    job: Any,
    adapter_config: Mapping[str, Any],
    gate0_config: Gate0Config,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Concrete factory for the generic ``collect-gate0`` child contract."""
    from robots.libero import get_env_spec
    from robots.libero.tools import LiberoPrimitives
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.research.handoff.experiments.runtime import (
        verify_gate0_job_external_bindings,
    )

    config = LiberoGate0FactoryConfig.model_validate(adapter_config)
    verify_gate0_job_external_bindings(job)
    handoff_path = Path(config.handoff_config).expanduser()
    if not handoff_path.is_absolute():
        handoff_path = Path.cwd() / handoff_path
    handoff = load_handoff_runtime_config(handoff_path.resolve())
    if not handoff.enabled:
        raise ValueError("Gate-0 adapter requires enabled=true handoff config")
    runtime = config.runtime
    if job.external_bindings is None:
        raise ValueError("Gate-0 adapter job lacks external bindings")
    if handoff_path.resolve() != Path(
        job.external_bindings.handoff_config_path
    ):
        raise ValueError("Gate-0 adapter handoff path disagrees with job binding")
    if handoff.configuration_id != job.external_bindings.handoff_configuration_id:
        raise ValueError("Gate-0 parsed handoff identity disagrees with job binding")
    if handoff.controller_method != job.controller_method:
        raise ValueError("Gate-0 job controller method disagrees with handoff config")
    if handoff.checkpoint_id not in (None, job.checkpoint_id):
        raise ValueError("Gate-0 handoff checkpoint ID disagrees with job")
    if runtime.pi05_checkpoint_id != job.checkpoint_id:
        raise ValueError("Gate-0 runtime Pi0.5 ID disagrees with job checkpoint")
    if handoff.metadata.get("pi05_checkpoint_id") not in (
        None,
        runtime.pi05_checkpoint_id,
    ):
        raise ValueError("Gate-0 handoff metadata Pi0.5 ID disagrees with runtime")
    if handoff.metadata.get("sam3_checkpoint_id") not in (
        None,
        runtime.sam3_checkpoint_id,
    ):
        raise ValueError("Gate-0 handoff metadata SAM3 ID disagrees with runtime")
    _assert_checkpoint_environment(runtime)
    plan_id = job.metadata.get("execution_plan_id")
    if not isinstance(plan_id, str) or not plan_id:
        raise ValueError("Gate-0 execution plan identity is missing")
    args = argparse.Namespace(
        suite=job.suite,
        task=int(job.task_id),
        seed=job.seed,
        max_episode_steps=runtime.max_episode_steps,
        libero_type=runtime.libero_type,
        env_endpoint=runtime.env_endpoint,
        vla_endpoint=runtime.vla_endpoint,
        sam3_endpoint=runtime.sam3_endpoint,
        cuda_device=runtime.cuda_device,
        handoff_config=str(handoff_path.resolve()),
        output_dir=str(output_dir),
    )
    daemons, primitive_kwargs = get_env_spec().init_runtime(
        args, output_dir, NullDashboardEventSink()
    )
    try:
        attestation = attest_runtime_checkpoint_clients(
            primitive_kwargs,
            runtime,
            trial_id=f"{job.run_id}/gate0-runtime",
            plan_id=plan_id,
            source_revision=job.source_revision,
        )
        attestation_path = write_runtime_attestation(
            attestation,
            output_dir / "runtime_identity" / f"{plan_id}.json",
        )
        attestation_sha256 = hashlib.sha256(
            attestation_path.read_bytes()
        ).hexdigest()
    except Exception:
        for daemon in reversed(daemons):
            daemon.stop()
        raise
    try:
        primitive_kwargs.pop("_rpent_handoff_runtime_config", None)
        telemetry = RuntimeEventSink(
            output_dir / "telemetry",
            configuration_id=handoff.configuration_id,
            controller_configuration_id=handoff.controller_configuration_id,
        )
        instrumentation = RuntimeInstrumentation(
            telemetry, enabled=handoff.instrumentation
        )
        prepared = instrument_primitives_kwargs(
            primitive_kwargs, instrumentation
        )
        primitives = LiberoPrimitives(
            check_cancelled=lambda: None,
            **prepared,
        )
    except Exception:
        for daemon in reversed(daemons):
            daemon.stop()
        raise
    try:
        provider = build_target_provider(handoff, primitives=primitives)
        labeler = None
        if config.labeler_factory is not None:
            labeler = _load_factory(config.labeler_factory)(
                primitives=primitives,
                task=job,
                configuration=config,
            )
            if not callable(labeler):
                raise TypeError("labeler_factory must return a callable")
        adapter = LiberoGate0Adapter(
            primitives=primitives,
            target_provider=provider,
            target_request=TargetRequest(
                target_id=job.target_id,
                description=job.target_description,
            ),
            core_api=CoreRuntimeAPI.load(handoff.governor_api_module),
            instrumentation=instrumentation,
            check_cancelled=lambda: None,
            vla_method=config.vla_method,
            invocation_id=f"{job.run_id}/gate0-pending",
            tool_defaults=config.tool_defaults,
            setup_target=config.setup_target,
            sampler_config=gate0_config.sampler,
            outcome_labeler=labeler,
        )
    except Exception:
        for daemon in reversed(daemons):
            daemon.stop()
        raise

    def cleanup() -> None:
        for daemon in reversed(daemons):
            daemon.stop()

    return {
        "adapter": adapter,
        "runtime_attestation": attestation,
        "runtime_attestation_sha256": attestation_sha256,
        "cleanup": cleanup,
    }


def run_gate0_server(
    config: Gate0ServerConfig,
    *,
    limit: int | None = None,
) -> tuple[Any, ...]:
    """Run/resume Gate-0 in the Linux LIBERO runtime.

    Calling this function starts or connects to the configured services and
    therefore belongs only in the explicit runtime child command.
    """
    from robots.libero import get_env_spec
    from robots.libero.tools import LiberoPrimitives
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.utils.logging import init_output_dir

    output_dir = init_output_dir(config.output_dir)
    handoff_config = load_handoff_runtime_config(config.handoff_config)
    if not handoff_config.enabled:
        raise ValueError("Gate-0 requires enabled=true handoff runtime config")
    if handoff_config.checkpoint_id not in (
        None,
        config.runtime.pi05_checkpoint_id,
    ):
        raise ValueError(
            "legacy Gate-0 handoff checkpoint ID disagrees with runtime Pi0.5 ID"
        )
    if handoff_config.metadata.get("pi05_checkpoint_id") not in (
        None,
        config.runtime.pi05_checkpoint_id,
    ) or handoff_config.metadata.get("sam3_checkpoint_id") not in (
        None,
        config.runtime.sam3_checkpoint_id,
    ):
        raise ValueError(
            "legacy Gate-0 handoff metadata checkpoint IDs disagree with runtime"
        )
    if config.source_revision is not None:
        from rpent.research.handoff.experiments.manifest import (
            compute_source_revision,
        )

        current_source = compute_source_revision(Path.cwd())
        if current_source != config.source_revision:
            raise ValueError(
                "current source bytes disagree with legacy Gate-0 config: "
                f"expected={config.source_revision!r}, actual={current_source!r}"
            )

    events = NullDashboardEventSink()
    env_spec = get_env_spec()
    with _injected_checkpoint_environment(config.runtime):
        daemons, primitives_kwargs = env_spec.init_runtime(
            _runtime_namespace(config), output_dir, events
        )
    try:
        attestation = attest_runtime_checkpoint_clients(
            primitives_kwargs,
            config.runtime,
            trial_id=f"{config.run_id}/legacy-gate0-runtime",
            source_revision=config.source_revision,
        )
        attestation_path = write_runtime_attestation(
            attestation,
            output_dir / "runtime_identity.json",
        )
        attestation_sha256 = hashlib.sha256(
            attestation_path.read_bytes()
        ).hexdigest()
    except Exception:
        for daemon in reversed(daemons):
            daemon.stop()
        raise
    try:
        primitives_kwargs.pop("_rpent_handoff_runtime_config", None)
        record_dir = output_dir / "records"
        outcome_path = record_dir / "outcomes.jsonl"
        existing_outcomes = ()
        if outcome_path.exists() and outcome_path.stat().st_size:
            outcome_scan = scan_outcome_jsonl(
                outcome_path,
                allow_partial_final_line=False,
            )
            if outcome_scan.needs_trailing_newline:
                raise ValueError(
                    "legacy Gate-0 outcome shard lacks a trailing newline"
                )
            existing_outcomes = tuple(
                envelope.payload for envelope in outcome_scan.envelopes
            )
        decision_path = record_dir / "decisions.jsonl"
        if decision_path.exists() and decision_path.stat().st_size:
            decision_scan = scan_decision_jsonl(
                decision_path,
                allow_partial_final_line=False,
            )
            if decision_scan.needs_trailing_newline:
                raise ValueError(
                    "legacy Gate-0 decision shard lacks a trailing newline"
                )
        dataset_sink = DatasetResearchSink(record_dir, fsync=config.fsync)
        telemetry_sink = RuntimeEventSink(
            output_dir / "telemetry",
            configuration_id=handoff_config.configuration_id,
            controller_configuration_id=(
                handoff_config.controller_configuration_id
            ),
        )
        sink = CompositeResearchSink([dataset_sink, telemetry_sink])
        instrumentation = RuntimeInstrumentation(
            sink, enabled=handoff_config.instrumentation
        )
        prepared = instrument_primitives_kwargs(
            primitives_kwargs, instrumentation
        )
        primitives = LiberoPrimitives(
            check_cancelled=lambda: None,
            **prepared,
        )
    except Exception:
        for daemon in reversed(daemons):
            daemon.stop()
        raise
    try:
        target_provider = build_target_provider(handoff_config, primitives=primitives)
        core_api = CoreRuntimeAPI.load(handoff_config.governor_api_module)
        labeler = None
        if config.labeler_factory is not None:
            labeler = _load_factory(config.labeler_factory)(
                primitives=primitives,
                task=config.task,
                configuration=config,
            )
            if not callable(labeler):
                raise TypeError("labeler_factory must return a callable")
        adapter = LiberoGate0Adapter(
            primitives=primitives,
            target_provider=target_provider,
            target_request=TargetRequest(
                target_id=config.task.target_id,
                description=config.task.target_description,
            ),
            core_api=core_api,
            instrumentation=instrumentation,
            check_cancelled=lambda: None,
            vla_method=config.task.vla_method,
            invocation_id=f"{config.run_id}/gate0-pending",
            tool_defaults=config.tool_defaults,
            setup_target=config.setup_target,
            sampler_config=config.gate0.sampler,
            outcome_labeler=labeler,
        )
        completed = ()
        if existing_outcomes:
            invalid_existing = [
                record.record_id
                for record in existing_outcomes
                if (
                    record.source_revision != config.source_revision
                    or record.controller.checkpoint_id
                    != config.runtime.pi05_checkpoint_id
                    or record.metadata.get("gate0_configuration_id")
                    != handoff_config.configuration_id
                    or record.metadata.get("runtime_attestation_id")
                    != attestation.attestation_id
                    or record.metadata.get("runtime_attestation_sha256")
                    != attestation_sha256
                )
            ]
            if invalid_existing:
                raise ValueError(
                    "legacy Gate-0 outcomes disagree with the live runtime "
                    f"identity: {invalid_existing[:10]}"
                )
            completed = tuple(
                record.identity.trial_id for record in existing_outcomes
            )
        collector = Gate0Collector(
            adapter=adapter,
            config=config.gate0,
            skill=SkillIdentity(
                name=config.task.skill_name,
                semantic_target=config.task.target_description,
                learned_controller="pi0.5",
            ),
            controller=ControllerIdentity(
                method="gate0_direct_frozen_pi0",
                implementation_version="rpent-libero-gate0/v1",
                checkpoint_id=config.runtime.pi05_checkpoint_id,
                configuration_id=handoff_config.controller_configuration_id,
            ),
            run_identity=Gate0RunIdentity(
                run_id=config.run_id,
                suite=config.task.suite,
                task_id=config.task.task,
                seed=config.task.seed,
                source_revision=config.source_revision,
                configuration_id=handoff_config.configuration_id,
                runtime_attestation_id=attestation.attestation_id,
                runtime_attestation_sha256=attestation_sha256,
            ),
            outcome_sink=sink,
            setup_sink=SetupJsonlWriter(
                record_dir / "setups.jsonl", fsync=config.fsync
            ),
            completed_trial_ids=completed,
            vla_kwargs={"prompt": config.task.skill_prompt, **config.vla_kwargs},
        )
        return collector.collect(limit=limit)
    finally:
        for daemon in reversed(daemons):
            daemon.stop()


def run_gate0_server_config(
    path: str | Path,
    *,
    limit: int | None = None,
) -> tuple[Any, ...]:
    return run_gate0_server(load_gate0_server_config(path), limit=limit)


__all__ = [
    "GATE0_SERVER_SCHEMA_VERSION",
    "Gate0ServerConfig",
    "Gate0TaskConfig",
    "LiberoGate0FactoryConfig",
    "LiberoGate0Adapter",
    "SetupTargetConfig",
    "build_gate0_adapter_bundle",
    "load_gate0_server_config",
    "run_gate0_server",
    "run_gate0_server_config",
]
