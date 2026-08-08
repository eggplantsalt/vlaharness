"""Thin, opt-in LIBERO adapter for the pure controller-handoff governor.

Dependency direction is intentionally one way: this module adapts existing
``LiberoPrimitives`` clients to the pure research governor.  It does not alter
the behavior of the original primitive methods, simulator server, or Pi0.5
inference path.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from robots.libero.handoff_runtime import (
    CoreRuntimeAPI,
    HandoffConfigurationError,
    HandoffRuntimeConfig,
    build_governor,
)
from robots.libero.tools import LiberoPrimitives, _metric_depth, _world_from_depth
from rpent.research.handoff.types import (
    ControllerIdentity,
    FeatureAvailability,
    FeatureProvenance,
    HandoffState,
    LabelSource,
    OutcomeLabels,
    OutcomeSignal,
    SkillIdentity,
    TargetContext,
    TargetEstimate,
    TrialIdentity,
    VisualGeometry,
    unavailable_signal,
)
from rpent.tools.toolkit import ToolCancelled
from rpent.utils.sam3_client import Sam3Client


HANDOFF_PICK_TOOL_NAME = "handoff_pi0_pick"
HANDOFF_DOUBLED_TOOL_NAME = "handoff_pi0_doubled"
HANDOFF_TOOL_NAMES = (HANDOFF_PICK_TOOL_NAME, HANDOFF_DOUBLED_TOOL_NAME)


HANDOFF_TOOLS_SPEC: tuple[dict[str, Any], ...] = (
    {
        "name": HANDOFF_PICK_TOOL_NAME,
        "description": (
            "Opt-in research composite pick. A local controller-handoff governor "
            "repeatedly observes and performs at most one bounded analytic "
            "adjustment per decision, then invokes the unchanged pi0_pick "
            "primitive. Use target_description for deployment perception."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The unchanged Pi0.5 pick instruction.",
                },
                "target_description": {
                    "type": ["string", "null"],
                    "description": (
                        "Perception target phrase; defaults to prompt. It is not "
                        "a simulator object-state key."
                    ),
                },
                "target_id": {
                    "type": ["string", "null"],
                    "description": "Stable semantic target id; defaults to target description.",
                },
                "target_xyz": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": (
                        "Injected/precomputed world xyz. Accepted only when the "
                        "run config explicitly selects the injected provider."
                    ),
                },
                "max_chunks": {"type": "integer", "minimum": 1},
                "lift_thresh": {"type": "number", "minimum": 0.0},
                "gripper_closed_thresh": {"type": "number", "minimum": 0.0},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": HANDOFF_DOUBLED_TOOL_NAME,
        "description": (
            "Opt-in research composite contact skill. A local handoff governor "
            "stages with bounded analytic steps and then invokes the unchanged "
            "pi0_doubled primitive. Its success still mirrors official LIBERO "
            "termination; intermediate contact success may remain unknown."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The unchanged Pi0.5 contact-skill instruction.",
                },
                "target_description": {"type": ["string", "null"]},
                "target_id": {"type": ["string", "null"]},
                "target_xyz": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "max_chunks": {"type": "integer", "minimum": 1},
            },
            "required": ["prompt"],
        },
    },
)


class GovernorProtocol(Protocol):
    def run(self, adapter: Any, invocation: Any) -> Any: ...


class ResearchSinkProtocol(Protocol):
    def append_decision(self, decision: Any) -> None: ...

    def append_outcome(self, outcome: Any) -> None: ...


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class RuntimeInstrumentation:
    """Research-only call tracing shared by transparent client wrappers."""

    def __init__(self, sink: Any, *, enabled: bool = True) -> None:
        self._sink = sink
        self.enabled = bool(enabled)
        self._phase = "research_idle"
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {
            "env_steps": 0,
            "env_chunks": 0,
            "env_actions": 0,
            "env_actions_submitted": 0,
            "vla_invocations": 0,
            "vla_failures": 0,
        }

    @property
    def phase_name(self) -> str:
        with self._lock:
            return self._phase

    @contextmanager
    def phase(self, name: str):
        with self._lock:
            previous = self._phase
            self._phase = name
        try:
            yield
        finally:
            with self._lock:
                self._phase = previous

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + int(amount)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        append = getattr(self._sink, "append_event", None)
        if append is not None:
            append(
                event_type,
                {
                    "phase": self.phase_name,
                    "monotonic_s": time.monotonic(),
                    **_json_safe(payload),
                },
            )


class InstrumentedEnvClient:
    """Transparent research-only wrapper around ``LiberoEnvClient``."""

    def __init__(self, client: Any, instrumentation: RuntimeInstrumentation) -> None:
        self._client = client
        self._instrumentation = instrumentation

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    @property
    def episode_terminated(self) -> bool:
        return bool(self._client.episode_terminated)

    @property
    def episode_truncated(self) -> bool:
        return bool(self._client.episode_truncated)

    @property
    def return_all_frames(self) -> bool:
        return bool(self._client.return_all_frames)

    @return_all_frames.setter
    def return_all_frames(self, value: bool) -> None:
        self._client.return_all_frames = value

    def reset(self):
        started = time.monotonic()
        try:
            result = self._client.reset()
        except Exception as exc:
            self._instrumentation.emit(
                "env_reset", {"elapsed_s": time.monotonic() - started, "error": str(exc)}
            )
            raise
        self._instrumentation.emit(
            "env_reset", {"elapsed_s": time.monotonic() - started, "error": None}
        )
        return result

    def step(self, action):
        started = time.monotonic()
        self._instrumentation.increment("env_actions_submitted")
        try:
            result = self._client.step(action)
        except Exception as exc:
            self._instrumentation.emit(
                "env_step",
                {
                    "elapsed_s": time.monotonic() - started,
                    "action": action,
                    "error": str(exc),
                },
            )
            raise
        self._instrumentation.increment("env_steps")
        self._instrumentation.increment("env_actions")
        self._instrumentation.emit(
            "env_step",
            {
                "elapsed_s": time.monotonic() - started,
                "action": action,
                "terminated": self.episode_terminated,
                "truncated": self.episode_truncated,
                "error": None,
            },
        )
        return result

    def chunk_step(self, actions, *, return_all_frames: bool | None = None):
        array = np.asarray(actions)
        chunk_size = int(array.shape[0]) if array.ndim >= 1 else 0
        started = time.monotonic()
        self._instrumentation.increment("env_actions_submitted", chunk_size)
        try:
            result = self._client.chunk_step(
                actions, return_all_frames=return_all_frames
            )
        except Exception as exc:
            self._instrumentation.emit(
                "env_chunk",
                {
                    "elapsed_s": time.monotonic() - started,
                    "action_shape": list(array.shape),
                    "error": str(exc),
                },
            )
            raise
        self._instrumentation.increment("env_chunks")
        _, _, term, trunc, _ = result
        self._instrumentation.emit(
            "env_chunk",
            {
                "elapsed_s": time.monotonic() - started,
                "action_shape": list(array.shape),
                "terminated_signals": term,
                "truncated_signals": trunc,
                "error": None,
            },
        )
        return result


class InstrumentedVLAClient:
    """Transparent research-only wrapper around the frozen VLA client."""

    def __init__(self, client: Any, instrumentation: RuntimeInstrumentation) -> None:
        self._client = client
        self._instrumentation = instrumentation

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def predict_action_batch(self, env_obs: dict[str, Any], mode: str = "eval", **kwargs):
        started = time.monotonic()
        self._instrumentation.increment("vla_invocations")
        try:
            result = self._client.predict_action_batch(env_obs, mode=mode, **kwargs)
        except Exception as exc:
            self._instrumentation.increment("vla_failures")
            self._instrumentation.emit(
                "vla_inference",
                {"elapsed_s": time.monotonic() - started, "error": str(exc)},
            )
            raise
        actions = np.asarray(result[0])
        self._instrumentation.emit(
            "vla_inference",
            {
                "elapsed_s": time.monotonic() - started,
                "action_shape": list(actions.shape),
                "action_dtype": str(actions.dtype),
                "finite": bool(np.isfinite(actions).all()),
                "error": None,
            },
        )
        return result


def instrument_primitives_kwargs(
    primitives_kwargs: Mapping[str, Any],
    instrumentation: RuntimeInstrumentation,
) -> dict[str, Any]:
    """Copy and wrap clients only for an explicitly enabled research run."""
    result = dict(primitives_kwargs)
    result["env"] = InstrumentedEnvClient(result["env"], instrumentation)
    result["model"] = InstrumentedVLAClient(result["model"], instrumentation)
    return result


@dataclass(frozen=True, slots=True)
class TargetRequest:
    target_id: str
    description: str
    injected_position_m: tuple[float, float, float] | None = None


class TargetProvider(Protocol):
    def estimate(
        self,
        *,
        request: TargetRequest,
        observation_sequence: int,
        raw_observation: Mapping[str, Any],
        policy_observation: Mapping[str, Any],
    ) -> TargetContext: ...


class CurrentObservationTargetProvider:
    """Deployment-realistic RGB-D + SAM3 target estimate for the current state."""

    _CAMERAS = {
        "agentview": (
            "agentview_image",
            "agentview_depth",
            "agentview",
        ),
        "wrist": (
            "robot0_eye_in_hand_image",
            "robot0_eye_in_hand_depth",
            "robot0_eye_in_hand",
        ),
    }

    def __init__(
        self,
        *,
        env: Any,
        sam3_client: Sam3Client,
        camera: str = "agentview",
        min_score: float = 0.2,
        provider_version: str = "rpent-current-rgbd-sam3/v1",
    ) -> None:
        if camera not in self._CAMERAS:
            raise HandoffConfigurationError(
                "perception target provider camera must be agentview or wrist"
            )
        if not 0.0 <= float(min_score) <= 1.0:
            raise HandoffConfigurationError("perception min_score must be in [0, 1]")
        self._env = env
        self._sam3 = sam3_client
        self._camera = camera
        self._min_score = float(min_score)
        self._provider_version = provider_version

    def estimate(
        self,
        *,
        request: TargetRequest,
        observation_sequence: int,
        raw_observation: Mapping[str, Any],
        policy_observation: Mapping[str, Any],
    ) -> TargetContext:
        del policy_observation
        captured_monotonic = time.monotonic()
        image_key, depth_key, rpc_camera = self._CAMERAS[self._camera]
        image_value = raw_observation.get(image_key)
        depth_value = raw_observation.get(depth_key)
        if image_value is None or depth_value is None:
            missing = [
                key
                for key, value in ((image_key, image_value), (depth_key, depth_value))
                if value is None
            ]
            return self._unavailable(
                request,
                observation_sequence,
                f"current observation missing {', '.join(missing)}",
            )

        image = np.asarray(image_value)
        if image.ndim != 3 or image.shape[-1] != 3:
            return self._unavailable(
                request,
                observation_sequence,
                f"{image_key} has invalid shape {image.shape}",
            )
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        # Match dump_state's calibration frame: both raw RGB and metric depth
        # are vertically flipped before applying get_camera_meta().
        calibrated_image = np.ascontiguousarray(image[::-1])
        try:
            result = self._sam3.segment_image(
                calibrated_image,
                text_prompt=request.description,
                min_score=self._min_score,
            )
        except Exception as exc:
            return self._unavailable(
                request,
                observation_sequence,
                f"SAM3 current-observation call failed: {exc}",
            )
        if not result.found or result.mask is None:
            return self._unavailable(
                request,
                observation_sequence,
                result.reason or "SAM3 found no target mask",
                confidence=result.score,
            )

        mask = np.asarray(result.mask, dtype=bool)
        if mask.shape != calibrated_image.shape[:2]:
            return self._unavailable(
                request,
                observation_sequence,
                f"SAM3 mask/image shape mismatch: {mask.shape} vs "
                f"{calibrated_image.shape[:2]}",
                confidence=result.score,
            )
        camera_meta = self._env.get_camera_meta(
            rpc_camera,
            int(mask.shape[0]),
            int(mask.shape[1]),
        )
        if not camera_meta:
            return self._unavailable(
                request,
                observation_sequence,
                f"camera metadata unavailable for {rpc_camera}",
                confidence=result.score,
            )
        try:
            depth = _metric_depth(depth_value, camera_meta)[::-1]
            if depth.shape != mask.shape:
                raise ValueError(
                    f"depth/mask shape mismatch: {depth.shape} vs {mask.shape}"
                )
            world = _world_from_depth(depth, camera_meta)
        except Exception as exc:
            return self._unavailable(
                request,
                observation_sequence,
                f"RGB-D back-projection failed: {exc}",
                confidence=result.score,
            )

        mask_count = int(mask.sum())
        if mask_count <= 0:
            return self._unavailable(
                request,
                observation_sequence,
                "SAM3 returned an empty mask",
                confidence=result.score,
            )
        selected = world[mask]
        valid = np.isfinite(selected).all(axis=1) & np.isfinite(depth[mask])
        valid &= depth[mask] > 0.0
        valid_count = int(valid.sum())
        if valid_count <= 0:
            return self._unavailable(
                request,
                observation_sequence,
                "target mask contains no finite positive-depth pixels",
                confidence=result.score,
            )
        position = np.median(selected[valid], axis=0)
        rows, cols = np.nonzero(mask)
        height, width = mask.shape
        row_norm = float(np.mean(rows) / max(height - 1, 1))
        col_norm = float(np.mean(cols) / max(width - 1, 1))
        visual = VisualGeometry(
            mask_area_fraction=float(mask_count / mask.size),
            valid_depth_fraction=float(valid_count / mask_count),
            image_centroid_rc_normalized=(row_norm, col_norm),
            camera_name=self._camera,
        )
        estimate = TargetEstimate(
            estimate_id=f"perception-{observation_sequence}-{request.target_id}",
            position_m=tuple(float(value) for value in position),
            frame="libero_world",
            provider=self._provider_version,
            availability=FeatureAvailability.DEPLOYMENT_PERCEPTION,
            confidence=result.score,
            observation_sequence=observation_sequence,
            age_s=max(0.0, time.monotonic() - captured_monotonic),
            visual_geometry=visual,
        )
        return TargetContext(
            target_id=request.target_id,
            description=request.description,
            estimate=estimate,
        )

    def _unavailable(
        self,
        request: TargetRequest,
        observation_sequence: int,
        reason: str,
        *,
        confidence: float | None = None,
    ) -> TargetContext:
        return TargetContext(
            target_id=request.target_id,
            description=request.description,
            estimate=TargetEstimate(
                estimate_id=f"perception-{observation_sequence}-{request.target_id}",
                position_m=None,
                frame="libero_world",
                provider=self._provider_version,
                availability=FeatureAvailability.DEPLOYMENT_PERCEPTION,
                confidence=confidence,
                observation_sequence=observation_sequence,
                age_s=0.0,
                unavailable_reason=reason,
            ),
        )


class InjectedTargetProvider:
    """Explicit precomputed target provider; never reads simulator state."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        configured = config.get("position_m")
        self._configured_position = (
            _position_tuple(configured, "target_provider.position_m")
            if configured is not None
            else None
        )
        availability_value = config.get("availability", "deployment_perception")
        try:
            self._availability = FeatureAvailability(availability_value)
        except ValueError as exc:
            raise HandoffConfigurationError(
                f"invalid injected target availability: {availability_value!r}"
            ) from exc
        if not self._availability.online_allowed:
            raise HandoffConfigurationError(
                "injected target provenance must be deployment-available; use "
                "the separate oracle provider for simulator-privileged state and "
                "never route experiment-setup values into an online policy"
            )
        self._provider = str(config.get("provider", "injected_precomputed/v1"))
        self._confidence = config.get("confidence", 1.0)

    def estimate(
        self,
        *,
        request: TargetRequest,
        observation_sequence: int,
        raw_observation: Mapping[str, Any],
        policy_observation: Mapping[str, Any],
    ) -> TargetContext:
        del raw_observation, policy_observation
        position = request.injected_position_m or self._configured_position
        reason = None if position is not None else "no injected target_xyz supplied"
        return TargetContext(
            target_id=request.target_id,
            description=request.description,
            estimate=TargetEstimate(
                estimate_id=f"injected-{observation_sequence}-{request.target_id}",
                position_m=position,
                frame="libero_world",
                provider=self._provider,
                availability=self._availability,
                confidence=float(self._confidence) if position is not None else None,
                observation_sequence=observation_sequence,
                age_s=0.0,
                unavailable_reason=reason,
            ),
        )


class OracleTargetProvider:
    """Privileged simulator target provider isolated behind oracle config."""

    def __init__(self, config: Mapping[str, Any], *, oracle_ablation: bool) -> None:
        if not oracle_ablation:
            raise HandoffConfigurationError(
                "OracleTargetProvider requires an explicitly labeled oracle ablation"
            )
        key = config.get("position_key")
        if not isinstance(key, str) or not key:
            raise HandoffConfigurationError(
                "oracle target_provider.position_key must name one raw observation key"
            )
        self._position_key = key

    def estimate(
        self,
        *,
        request: TargetRequest,
        observation_sequence: int,
        raw_observation: Mapping[str, Any],
        policy_observation: Mapping[str, Any],
    ) -> TargetContext:
        del policy_observation
        value = raw_observation.get(self._position_key)
        if value is None:
            position = None
            reason = f"privileged raw key missing: {self._position_key}"
        else:
            position = _position_tuple(value, self._position_key)
            reason = None
        return TargetContext(
            target_id=request.target_id,
            description=request.description,
            estimate=TargetEstimate(
                estimate_id=f"oracle-{observation_sequence}-{request.target_id}",
                position_m=position,
                frame="libero_world",
                provider=f"libero_raw_obs:{self._position_key}",
                availability=FeatureAvailability.SIMULATOR_PRIVILEGED,
                confidence=1.0 if position is not None else None,
                observation_sequence=observation_sequence,
                age_s=0.0,
                unavailable_reason=reason,
            ),
        )


def _position_tuple(value: Any, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain exactly three finite numbers")
    return tuple(float(item) for item in array)


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_nonnegative_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def build_target_provider(
    config: HandoffRuntimeConfig,
    *,
    primitives: LiberoPrimitives,
) -> TargetProvider:
    provider_config = config.target_provider
    kind = provider_config["kind"]
    if kind == "perception":
        return CurrentObservationTargetProvider(
            env=primitives.env,
            sam3_client=primitives._sam3_client,
            camera=str(provider_config.get("camera", "agentview")),
            min_score=float(provider_config.get("min_score", 0.2)),
            provider_version=str(
                provider_config.get("provider_version", "rpent-current-rgbd-sam3/v1")
            ),
        )
    if kind == "injected":
        return InjectedTargetProvider(provider_config)
    if kind == "oracle":
        return OracleTargetProvider(
            provider_config, oracle_ablation=config.oracle_ablation
        )
    raise AssertionError(f"validated target provider kind is unknown: {kind}")


_DEPLOYMENT_RAW_KEYS = frozenset(
    {
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "agentview_image",
        "agentview_depth",
        "robot0_eye_in_hand_image",
        "robot0_eye_in_hand_depth",
    }
)
_DEPLOYMENT_POLICY_KEYS = frozenset(
    {
        "states",
        "main_images",
        "wrist_images",
        "extra_view_images",
        "task_descriptions",
    }
)


def _runtime_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


class LiberoGovernorAdapter:
    """Adapter implementing the pure governor's environment-facing protocol."""

    def __init__(
        self,
        *,
        primitives: LiberoPrimitives,
        target_provider: TargetProvider,
        target_request: TargetRequest,
        core_api: CoreRuntimeAPI,
        instrumentation: RuntimeInstrumentation,
        check_cancelled: Callable[[], None],
        vla_method: str,
        invocation_id: str,
        tool_defaults: Mapping[str, Any],
    ) -> None:
        if vla_method not in {"pi0_pick", "pi0_doubled"}:
            raise ValueError(f"unsupported VLA method: {vla_method}")
        self.primitives = primitives
        self._target_provider = target_provider
        self._target_request = target_request
        self._core_api = core_api
        self._instrumentation = instrumentation
        self._check_cancelled = check_cancelled
        self._vla_method = vla_method
        self._invocation_id = invocation_id
        self._tool_defaults = dict(tool_defaults)
        self._observation_sequence = 0
        self._started_monotonic = time.monotonic()

    def observe(self, skill: SkillIdentity) -> HandoffState:
        self.raise_if_cancelled()
        if not isinstance(skill, SkillIdentity):
            skill = SkillIdentity.model_validate(skill)
        policy_obs = self.primitives._last_obs
        if not isinstance(policy_obs, Mapping):
            raise RuntimeError("LIBERO primitive has no current policy observation")
        states = np.asarray(policy_obs.get("states"), dtype=np.float64)
        if states.ndim != 1 or states.shape[0] < 8:
            raise RuntimeError(
                f"LIBERO states must be one-dimensional with at least 8 values; "
                f"got {states.shape}"
            )
        if not np.isfinite(states[:8]).all():
            raise RuntimeError("LIBERO proprioceptive state contains NaN/Inf")

        raw = self.primitives.env.raw_obs()
        if not isinstance(raw, Mapping):
            raise RuntimeError("LIBERO raw_obs did not return a mapping")
        # Deployment providers never receive simulator object-pose, contact, or
        # task-predicate fields.  The oracle class is the only separate path
        # allowed to inspect its explicitly configured privileged key.
        if isinstance(self._target_provider, OracleTargetProvider):
            provider_raw = raw
        else:
            provider_raw = {
                key: raw[key] for key in _DEPLOYMENT_RAW_KEYS if key in raw
            }
        provider_policy = {
            key: policy_obs[key]
            for key in _DEPLOYMENT_POLICY_KEYS
            if key in policy_obs
        }

        quaternion = np.asarray(raw.get("robot0_eef_quat"), dtype=np.float64)
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise RuntimeError(
                "robot0_eef_quat must contain four finite XYZW values"
            )
        sequence = self._observation_sequence
        self._observation_sequence += 1
        target = self._target_provider.estimate(
            request=self._target_request,
            observation_sequence=sequence,
            raw_observation=provider_raw,
            policy_observation=provider_policy,
        )
        provenance_items = [
            FeatureProvenance(
                feature_name="eef_position_m",
                availability=FeatureAvailability.DEPLOYMENT_SENSOR,
                source="libero_policy_state[0:3]",
                unit="m",
                frame="libero_world",
                provider_version="rpent-libero-state/v1",
            ),
            FeatureProvenance(
                feature_name="eef_quaternion_xyzw",
                availability=FeatureAvailability.DEPLOYMENT_SENSOR,
                source="raw_obs.robot0_eef_quat",
                unit="unit_quaternion",
                frame="libero_world",
                provider_version="rpent-libero-state/v1",
            ),
            FeatureProvenance(
                feature_name="gripper_opening_m",
                availability=FeatureAvailability.DERIVED_DEPLOYMENT,
                source="abs(libero_policy_state[6])+abs(state[7])",
                unit="m",
                frame="gripper",
                derivation="two-finger absolute-qpos separation proxy",
                provider_version="rpent-libero-state/v1",
            ),
            FeatureProvenance(
                feature_name="skill",
                availability=FeatureAvailability.DERIVED_DEPLOYMENT,
                source="semantic composite-tool invocation",
                unit="categorical",
                frame="semantic",
                provider_version="rpent-libero-handoff/v1",
            ),
            FeatureProvenance(
                feature_name="target_position_m",
                availability=target.estimate.availability,
                source=target.estimate.provider,
                unit="m",
                frame=target.estimate.frame,
                provider_version="rpent-libero-target/v1",
            ),
        ]
        if target.estimate.confidence is not None:
            provenance_items.append(
                FeatureProvenance(
                    feature_name="target_confidence",
                    availability=target.estimate.availability,
                    source=f"{target.estimate.provider}.confidence",
                    unit="probability",
                    frame="semantic",
                    provider_version="rpent-libero-target/v1",
                )
            )
        if target.estimate.visual_geometry is not None:
            provenance_items.extend(
                (
                    FeatureProvenance(
                        feature_name="mask_area_fraction",
                        availability=target.estimate.availability,
                        source=f"{target.estimate.provider}.visual_geometry",
                        unit="fraction",
                        frame="image",
                        provider_version="rpent-libero-target/v1",
                    ),
                    FeatureProvenance(
                        feature_name="valid_depth_fraction",
                        availability=target.estimate.availability,
                        source=f"{target.estimate.provider}.visual_geometry",
                        unit="fraction",
                        frame="image",
                        provider_version="rpent-libero-target/v1",
                    ),
                    FeatureProvenance(
                        feature_name="image_centroid_rc_normalized",
                        availability=target.estimate.availability,
                        source=f"{target.estimate.provider}.visual_geometry",
                        unit="normalized_image_coordinate",
                        frame="image",
                        provider_version="rpent-libero-target/v1",
                    ),
                )
            )
        provenance = tuple(provenance_items)
        state = HandoffState(
            state_id=f"{self._invocation_id}/observation-{sequence:04d}",
            observation_sequence=sequence,
            observed_elapsed_s=max(0.0, time.monotonic() - self._started_monotonic),
            eef_position_m=tuple(float(value) for value in states[:3]),
            eef_quaternion_xyzw=tuple(float(value) for value in quaternion),
            gripper_opening_m=float(abs(states[6]) + abs(states[7])),
            skill=skill,
            target=target,
            provenance=provenance,
        )
        self._instrumentation.emit("handoff_observation", {"state": state})
        return state

    def episode_status(self) -> Any:
        return self._core_api.EpisodeStatus(
            terminated=bool(self.primitives.env.episode_terminated),
            truncated=bool(self.primitives.env.episode_truncated),
        )

    def raise_if_cancelled(self) -> None:
        self._check_cancelled()

    def stage(self, candidate: Any, max_step_m: float) -> Any:
        self.raise_if_cancelled()
        max_step = float(max_step_m)
        if not math.isfinite(max_step) or max_step <= 0.0:
            raise ValueError("max_step_m must be a finite positive value")
        target = _position_tuple(
            _runtime_value(candidate, "eef_position_m"),
            "candidate.eef_position_m",
        )
        before = np.asarray(self.primitives._last_obs_eef_pos, dtype=np.float64)
        if before.shape != (3,) or not np.isfinite(before).all():
            raise RuntimeError("current LIBERO EEF position is not three finite values")
        requested_delta = np.asarray(target, dtype=np.float64) - before
        requested_distance = float(np.linalg.norm(requested_delta))
        if requested_distance > max_step:
            stage_target = before + requested_delta * (max_step / requested_distance)
        else:
            stage_target = np.asarray(target, dtype=np.float64)
        stage_target_tuple = tuple(float(value) for value in stage_target)
        started = time.monotonic()
        result: dict[str, Any] | None = None
        error: str | None = None
        counters_before = self._instrumentation.snapshot()
        try:
            with self._instrumentation.phase("analytic_stage"):
                yaw = _runtime_value(candidate, "wrist_yaw_rad")
                pitch = _runtime_value(candidate, "wrist_pitch_rad")
                gripper = float(self._tool_defaults.get("stage_gripper", -1.0))
                action_scale = float(self._tool_defaults.get("action_scale", 0.05))
                bounded_step = min(max_step, action_scale)
                tolerance = float(self._tool_defaults.get("stage_tolerance_m", 1e-6))
                if pitch is None and yaw is None:
                    result = self.primitives.move_to(
                        stage_target_tuple,
                        max_steps=1,
                        gripper=gripper,
                        step_clip=bounded_step,
                        tol=tolerance,
                        action_scale=action_scale,
                    )
                else:
                    result = self.primitives.move_pose(
                        stage_target_tuple,
                        target_pitch=float(pitch) if pitch is not None else None,
                        target_yaw=float(yaw) if yaw is not None else None,
                        gripper=gripper,
                        step_clip=bounded_step,
                        pitch_step=float(
                            self._tool_defaults.get("stage_pitch_step_rad", 0.08)
                        ),
                        yaw_step=float(
                            self._tool_defaults.get("stage_yaw_step_rad", 0.08)
                        ),
                        tol=tolerance,
                        ori_tol=float(
                            self._tool_defaults.get("stage_orientation_tolerance_rad", 1e-6)
                        ),
                        action_scale=action_scale,
                        max_steps=1,
                    )
        except ToolCancelled:
            raise
        except Exception as exc:
            error = str(exc)
        counters_after = self._instrumentation.snapshot()
        steps = max(
            0,
            counters_after.get("env_steps", 0)
            - counters_before.get("env_steps", 0),
        )
        after = np.asarray(self.primitives._last_obs_eef_pos, dtype=np.float64)
        distance = float(np.linalg.norm(after - before))
        elapsed = max(0.0, time.monotonic() - started)
        success = error is None and not (
            isinstance(result, Mapping) and result.get("error")
        )
        if not success and error is None and isinstance(result, Mapping):
            error = str(result.get("error"))
        if distance > max_step + 1e-9:
            success = False
            error = (
                f"analytic stage moved {distance:.9f} m, exceeding the "
                f"{max_step:.9f} m bound"
            )
        self._instrumentation.emit(
            "handoff_stage",
            {
                "candidate_id": _runtime_value(candidate, "candidate_id"),
                "candidate_distance_m": requested_distance,
                "requested_max_step_m": max_step,
                "actual_distance_m": distance,
                "steps": steps,
                "elapsed_s": elapsed,
                "success": success,
                "error": error,
                "primitive_result": result,
            },
        )
        return self._core_api.StageResult(
            success=success,
            steps=steps,
            distance_m=distance,
            elapsed_s=elapsed,
            achieved_state=None,
            error=error,
        )

    def execute_vla(self, invocation: Any) -> Any:
        self.raise_if_cancelled()
        kwargs = _runtime_value(invocation, "vla_kwargs", {})
        if not isinstance(kwargs, Mapping):
            raise TypeError("GovernorInvocation.vla_kwargs must be a mapping")
        started = time.monotonic()
        counters_before = self._instrumentation.snapshot()
        primitive_result: Mapping[str, Any] | None = None
        exception: str | None = None
        try:
            with self._instrumentation.phase("vla_execute"):
                primitive_result = getattr(self.primitives, self._vla_method)(**dict(kwargs))
        except ToolCancelled:
            raise
        except Exception as exc:
            exception = f"{type(exc).__name__}: {exc}"
        elapsed = max(0.0, time.monotonic() - started)
        counters_after = self._instrumentation.snapshot()
        invocations = max(
            0,
            counters_after.get("vla_invocations", 0)
            - counters_before.get("vla_invocations", 0),
        )
        chunks = max(
            0,
            counters_after.get("env_chunks", 0)
            - counters_before.get("env_chunks", 0),
        )
        source_verified_single_actions = max(
            0,
            counters_after.get("env_actions", 0)
            - counters_before.get("env_actions", 0),
        )
        submitted_actions = max(
            0,
            counters_after.get("env_actions_submitted", 0)
            - counters_before.get("env_actions_submitted", 0),
        )
        # External chunk_step does not expose a source-verified executed-step
        # count, especially when termination occurs inside a chunk.  Preserve
        # the submitted count in telemetry but leave the authoritative cost
        # unavailable until the server diagnostic establishes that contract.
        has_unverified_submissions = (
            submitted_actions > source_verified_single_actions
        )
        env_actions = (
            None
            if chunks > 0 or has_unverified_submissions
            else source_verified_single_actions
        )
        result_dict = dict(primitive_result or {})
        self._instrumentation.emit(
            "handoff_vla_execution",
            {
                "method": self._vla_method,
                "elapsed_s": elapsed,
                "result": result_dict,
                "exception": exception,
                "inference_invocations": invocations,
                "chunks": chunks,
                "env_actions": env_actions,
                "submitted_env_actions": submitted_actions,
            },
        )
        return self._core_api.VLAExecutionResult(
            result=result_dict,
            elapsed_s=elapsed,
            exception=exception,
            invocations=invocations,
            chunks=chunks,
            env_actions=env_actions,
        )

    def label_outcome(self, result: Any) -> OutcomeLabels:
        result_mapping = _runtime_value(result, "result", {})
        exception = _runtime_value(result, "exception")
        if not isinstance(result_mapping, Mapping):
            result_mapping = {}
        primitive_value = result_mapping.get("success")
        if self._vla_method == "pi0_pick" and exception is None:
            diagnostics = result_mapping.get("diagnostics")
            heuristic_value: bool | None = None
            if isinstance(diagnostics, Mapping):
                try:
                    descent_done = diagnostics["descent_done"]
                    ascent = float(diagnostics["post_min_ascent_m"])
                    lift_threshold = float(diagnostics["lift_thresh"])
                    final_gripper = float(result_mapping["final_gripper_opening"])
                    gripper_threshold = float(
                        diagnostics["gripper_closed_thresh"]
                    )
                    if (
                        isinstance(descent_done, bool)
                        and all(
                            math.isfinite(value)
                            for value in (
                                ascent,
                                lift_threshold,
                                final_gripper,
                                gripper_threshold,
                            )
                        )
                    ):
                        heuristic_value = bool(
                            descent_done
                            and ascent >= lift_threshold
                            and final_gripper < gripper_threshold
                        )
                except (KeyError, TypeError, ValueError):
                    heuristic_value = None
            if (
                heuristic_value is None
                and isinstance(primitive_value, bool)
                and result_mapping.get("libero_terminated") is not True
            ):
                heuristic_value = primitive_value
            primitive = (
                OutcomeSignal(
                    value=heuristic_value,
                    source=LabelSource.PRIMITIVE_HEURISTIC,
                    definition="existing pi0_pick EEF/gripper heuristic",
                    evaluator_id=self._vla_method,
                )
                if heuristic_value is not None
                else unavailable_signal(
                    "pi0_pick primitive heuristic cannot be separated from its "
                    "official-termination success mirror"
                )
            )
        elif self._vla_method == "pi0_doubled" and exception is None:
            primitive = unavailable_signal(
                "pi0_doubled exposes no primitive-level success independent "
                "of official LIBERO task termination"
            )
        else:
            primitive = unavailable_signal(
                "primitive result unavailable because VLA execution did not return success"
            )
        return OutcomeLabels(
            primitive_success=primitive,
            skill_success=unavailable_signal(
                "no independent skill-specific evaluator configured"
            ),
            task_success=OutcomeSignal(
                value=bool(self.primitives.env.episode_terminated),
                source=LabelSource.OFFICIAL_TERMINATION,
                definition="latched official LIBERO termination",
                evaluator_id="LiberoEnvClient.episode_terminated",
            ),
            episode_truncated=OutcomeSignal(
                value=bool(self.primitives.env.episode_truncated),
                source=LabelSource.RUNTIME,
                definition="latched LIBERO episode truncation",
                evaluator_id="LiberoEnvClient.episode_truncated",
            ),
            llm_finish=unavailable_signal(
                "planner finish is not available inside the local composite primitive"
            ),
        )


class LiberoHandoffComposite:
    """Planner-facing composite methods delegated to one pure governor."""

    def __init__(
        self,
        *,
        primitives: LiberoPrimitives,
        config: HandoffRuntimeConfig,
        sink: ResearchSinkProtocol,
        instrumentation: RuntimeInstrumentation,
        check_cancelled: Callable[[], None],
        run_output_dir: Path,
    ) -> None:
        self._primitives = primitives
        self._config = config
        self._sink = sink
        self._instrumentation = instrumentation
        self._check_cancelled = check_cancelled
        self._run_output_dir = run_output_dir
        self._core_api = CoreRuntimeAPI.load(config.governor_api_module)
        self._governor: GovernorProtocol = build_governor(config, sink=sink)
        self._target_provider = build_target_provider(config, primitives=primitives)
        self._env_runtime_meta: dict[str, Any] = {}
        runtime_probe = getattr(primitives.env, "runtime_probe", None)
        if callable(runtime_probe):
            try:
                payload = runtime_probe()
            except Exception as exc:
                raise HandoffConfigurationError(
                    "handoff-enabled env runtime capability probe failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise HandoffConfigurationError(
                    "handoff-enabled env runtime probe returned a non-object"
                )
            if payload.get("schema_version") != "rpent.runtime-probe/v1":
                raise HandoffConfigurationError(
                    "handoff-enabled env runtime probe has an unsupported "
                    f"schema_version: {payload.get('schema_version')!r}"
                )
            if payload.get("component") != "libero_env":
                raise HandoffConfigurationError(
                    "handoff-enabled env runtime probe component must be "
                    f"'libero_env', got {payload.get('component')!r}"
                )
            runtime_meta = payload.get("runtime_meta", {})
            if not isinstance(runtime_meta, Mapping):
                raise HandoffConfigurationError(
                    "handoff-enabled env runtime probe runtime_meta is not an object"
                )
            self._env_runtime_meta = dict(runtime_meta)
            self._instrumentation.emit(
                "handoff_env_capability",
                {
                    "probe_schema_version": payload.get("schema_version"),
                    "component": payload.get("component"),
                    "runtime_meta": self._env_runtime_meta,
                },
            )
        self._invocation_count = 0

    def handoff_pi0_pick(
        self,
        prompt: str,
        *,
        target_description: str | None = None,
        target_id: str | None = None,
        target_xyz: list[float] | None = None,
        max_chunks: int | None = None,
        lift_thresh: float | None = None,
        gripper_closed_thresh: float | None = None,
    ) -> dict[str, Any]:
        supplied = {
            "max_chunks": _optional_positive_int(max_chunks, "max_chunks"),
            "lift_thresh": _optional_nonnegative_float(
                lift_thresh, "lift_thresh"
            ),
            "gripper_closed_thresh": _optional_nonnegative_float(
                gripper_closed_thresh, "gripper_closed_thresh"
            ),
        }
        return self._run(
            tool_name=HANDOFF_PICK_TOOL_NAME,
            vla_method="pi0_pick",
            skill_name="pick",
            prompt=prompt,
            target_description=target_description,
            target_id=target_id,
            target_xyz=target_xyz,
            supplied_vla_kwargs=supplied,
        )

    def handoff_pi0_doubled(
        self,
        prompt: str,
        *,
        target_description: str | None = None,
        target_id: str | None = None,
        target_xyz: list[float] | None = None,
        max_chunks: int | None = None,
    ) -> dict[str, Any]:
        return self._run(
            tool_name=HANDOFF_DOUBLED_TOOL_NAME,
            vla_method="pi0_doubled",
            skill_name="contact",
            prompt=prompt,
            target_description=target_description,
            target_id=target_id,
            target_xyz=target_xyz,
            supplied_vla_kwargs={
                "max_chunks": _optional_positive_int(max_chunks, "max_chunks")
            },
        )

    def _run(
        self,
        *,
        tool_name: str,
        vla_method: str,
        skill_name: str,
        prompt: str,
        target_description: str | None,
        target_id: str | None,
        target_xyz: list[float] | None,
        supplied_vla_kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("prompt must be non-empty")
        description = str(target_description or prompt).strip()
        stable_target_id = str(target_id or description).strip()
        if not description or not stable_target_id:
            raise ValueError("target description/id must be non-empty")
        injected = (
            _position_tuple(target_xyz, "target_xyz")
            if target_xyz is not None
            else None
        )
        if injected is not None and not isinstance(
            self._target_provider, InjectedTargetProvider
        ):
            raise ValueError(
                "target_xyz is accepted only when target_provider.kind='injected'"
            )

        self._invocation_count += 1
        invocation_number = self._invocation_count
        server_meta = getattr(self._primitives.env, "server_meta", {})
        if not isinstance(server_meta, Mapping):
            server_meta = {}
        for identity_key in ("suite", "task", "seed"):
            configured_value = self._config.metadata.get(identity_key)
            live_value = server_meta.get(identity_key)
            if (
                configured_value is not None
                and live_value is not None
                and str(configured_value) != str(live_value)
            ):
                raise HandoffConfigurationError(
                    f"configured {identity_key} disagrees with the live LIBERO "
                    f"server identity: {configured_value!r} != {live_value!r}"
                )
        suite = str(server_meta.get("suite", self._config.metadata.get("suite", "unknown")))
        task = server_meta.get("task", self._config.metadata.get("task", "unknown"))
        seed = int(server_meta.get("seed", self._config.metadata.get("seed", 0)))
        run_id = str(
            self._config.metadata.get("run_id", self._run_output_dir.name)
        )
        episode_id = str(
            self._config.metadata.get(
                "episode_id", f"{suite}-task-{task}-seed-{seed}"
            )
        )
        invocation_id = f"{episode_id}/handoff-{invocation_number:04d}"
        configured_reset_id = self._config.metadata.get("reset_id")
        runtime_reset_id = self._env_runtime_meta.get("reset_id")
        if (
            configured_reset_id is not None
            and runtime_reset_id is not None
            and str(configured_reset_id) != str(runtime_reset_id)
        ):
            raise HandoffConfigurationError(
                "configured reset_id disagrees with the live LIBERO runtime probe: "
                f"{configured_reset_id!r} != {runtime_reset_id!r}"
            )
        resolved_reset_id = (
            configured_reset_id
            if configured_reset_id is not None
            else runtime_reset_id
        )
        identity = TrialIdentity(
            run_id=run_id,
            episode_id=episode_id,
            trial_id=str(self._config.metadata.get("trial_id", episode_id)),
            invocation_id=invocation_id,
            suite=suite,
            task_id=task,
            seed=seed,
            reset_id=(str(resolved_reset_id) if resolved_reset_id is not None else None),
            repeat_index=int(self._config.metadata.get("repeat_index", 0)),
        )
        skill = SkillIdentity(
            name=skill_name,
            semantic_target=description,
            learned_controller="pi0.5",
        )
        controller = ControllerIdentity(
            method=self._config.controller_method,
            implementation_version=self._config.controller_implementation_version,
            checkpoint_id=self._config.checkpoint_id,
            configuration_id=self._config.controller_configuration_id,
        )
        defaults = self._config.tool_defaults.get(tool_name, {})
        if defaults and not isinstance(defaults, Mapping):
            raise HandoffConfigurationError(
                f"tool_defaults.{tool_name} must be an object"
            )
        vla_kwargs = dict(defaults)
        for internal_key in (
            "stage_gripper",
            "action_scale",
            "stage_tolerance_m",
            "stage_pitch_step_rad",
            "stage_yaw_step_rad",
            "stage_orientation_tolerance_rad",
        ):
            vla_kwargs.pop(internal_key, None)
        vla_kwargs["prompt"] = prompt
        vla_kwargs.update(
            {key: value for key, value in supplied_vla_kwargs.items() if value is not None}
        )
        target_request = TargetRequest(
            target_id=stable_target_id,
            description=description,
            injected_position_m=injected,
        )
        adapter = LiberoGovernorAdapter(
            primitives=self._primitives,
            target_provider=self._target_provider,
            target_request=target_request,
            core_api=self._core_api,
            instrumentation=self._instrumentation,
            check_cancelled=self._check_cancelled,
            vla_method=vla_method,
            invocation_id=invocation_id,
            tool_defaults=dict(defaults),
        )
        invocation_metadata = dict(self._config.metadata)
        invocation_metadata.update(
            {
                "tool_name": tool_name,
                "target_id": stable_target_id,
                "target_description": description,
                "target_provider_kind": self._config.target_provider["kind"],
                "oracle_ablation": self._config.oracle_ablation,
                "handoff_configuration_id": self._config.configuration_id,
                "controller_configuration_id": (
                    self._config.controller_configuration_id
                ),
            }
        )
        if runtime_reset_id is not None:
            invocation_metadata["runtime_reset_id"] = str(runtime_reset_id)
        if self._env_runtime_meta.get("libero_type") is not None:
            invocation_metadata["libero_type"] = str(
                self._env_runtime_meta["libero_type"]
            )
        invocation = self._core_api.GovernorInvocation(
            identity=identity,
            skill=skill,
            controller=controller,
            vla_kwargs=vla_kwargs,
            metadata=invocation_metadata,
        )
        self._instrumentation.emit(
            "handoff_invocation_started",
            {
                "identity": identity,
                "skill": skill,
                "controller": controller,
                "tool_name": tool_name,
                "handoff_configuration_id": self._config.configuration_id,
                "controller_configuration_id": (
                    self._config.controller_configuration_id
                ),
            },
        )
        try:
            with self._instrumentation.phase("governor"):
                run_result = self._governor.run(adapter, invocation)
        except Exception as exc:
            self._instrumentation.emit(
                "handoff_invocation_failed",
                {
                    "invocation_id": invocation_id,
                    "handoff_configuration_id": self._config.configuration_id,
                    "controller_configuration_id": (
                        self._config.controller_configuration_id
                    ),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            raise
        planner_result = _normalise_governor_result(run_result)
        planner_result.setdefault("name", tool_name)
        planner_result.setdefault("handoff_configuration_id", self._config.configuration_id)
        planner_result.setdefault(
            "controller_configuration_id",
            self._config.controller_configuration_id,
        )
        planner_result.setdefault("invocation_id", invocation_id)
        planner_result.setdefault("oracle_ablation", self._config.oracle_ablation)
        self._instrumentation.emit(
            "handoff_invocation_completed",
            {"invocation_id": invocation_id, "result": planner_result},
        )
        return planner_result


def _normalise_governor_result(value: Any) -> dict[str, Any]:
    planner_result = _runtime_value(value, "planner_result")
    if isinstance(planner_result, Mapping):
        return _json_safe(planner_result)
    if isinstance(value, Mapping):
        return _json_safe(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json", exclude_none=False)
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return {"value": str(value)}


def build_handoff_composite(
    *,
    primitives: LiberoPrimitives,
    config: HandoffRuntimeConfig,
    sink: ResearchSinkProtocol,
    instrumentation: RuntimeInstrumentation,
    check_cancelled: Callable[[], None],
    run_output_dir: Path,
) -> LiberoHandoffComposite:
    if not config.enabled:
        raise HandoffConfigurationError(
            "cannot build handoff composite from enabled=false config"
        )
    return LiberoHandoffComposite(
        primitives=primitives,
        config=config,
        sink=sink,
        instrumentation=instrumentation,
        check_cancelled=check_cancelled,
        run_output_dir=run_output_dir,
    )


__all__ = [
    "HANDOFF_DOUBLED_TOOL_NAME",
    "HANDOFF_PICK_TOOL_NAME",
    "HANDOFF_TOOL_NAMES",
    "HANDOFF_TOOLS_SPEC",
    "CurrentObservationTargetProvider",
    "InjectedTargetProvider",
    "InstrumentedEnvClient",
    "InstrumentedVLAClient",
    "LiberoGovernorAdapter",
    "LiberoHandoffComposite",
    "OracleTargetProvider",
    "RuntimeInstrumentation",
    "TargetRequest",
    "build_handoff_composite",
    "build_target_provider",
    "instrument_primitives_kwargs",
]
