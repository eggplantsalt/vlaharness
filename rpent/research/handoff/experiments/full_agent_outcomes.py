"""Strict post-run episode summaries for full RPent/Harness trials.

The local governor writes invocation outcomes before the planner finishes. This
module joins those immutable records with the final RPent transcript and state
trace into a separate episode-scoped OutcomeRecord. Original Harness trials are
summarized through the same surface without enabling handoff behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rpent.research.handoff.evaluation.aggregate import read_outcome_jsonl
from rpent.research.handoff.experiments.config import ExecutionLayer, stable_identifier
from rpent.research.handoff.experiments.manifest import TrialManifest
from rpent.research.handoff.types import (
    ControllerIdentity,
    CostRecord,
    FeatureAvailability,
    FeatureProvenance,
    FailureMode,
    GovernorState,
    HandoffState,
    LabelSource,
    OutcomeLabels,
    OutcomeRecord,
    OutcomeSignal,
    SkillIdentity,
    TerminationReason,
    TerminationRecord,
    TimingRecord,
    TrialIdentity,
    unavailable_signal,
)

EPISODE_SUMMARY_VERSION = "rpent.handoff-full-agent-episode-summary/v1"

_DIRECT_VLA_ACTIONS = frozenset({"pi0_pick", "pi0_doubled"})
_COMPOSITE_VLA_ACTIONS = frozenset(
    {"handoff_pi0_pick", "handoff_pi0_doubled"}
)


class FullAgentSummaryError(ValueError):
    """A full-agent artifact set is missing, ambiguous, or contradictory."""


def _strict_json(path: Path) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant {token!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                parse_constant=reject_constant,
                object_pairs_hook=reject_duplicates,
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FullAgentSummaryError(f"invalid JSON artifact {path}: {exc}") from exc


def _single_artifact(root: Path, pattern: str, name: str) -> Path:
    matches = sorted(path for path in root.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise FullAgentSummaryError(
            f"expected exactly one {name} in {root}, found {len(matches)}"
        )
    return matches[0]


def _finite_nonnegative(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FullAgentSummaryError(f"{name} must be numeric when available")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise FullAgentSummaryError(f"{name} must be finite and non-negative")
    return number


def _nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FullAgentSummaryError(f"{name} must be a non-negative integer")
    return value


def _required_vector(
    value: Any,
    *,
    length: int,
    name: str,
) -> tuple[float, ...]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise FullAgentSummaryError(
            f"{name} must contain exactly {length} finite numeric values"
        )
    return tuple(float(item) for item in value)


def _physical_action(entry: Mapping[str, Any]) -> str | None:
    command = entry.get("command")
    if command is None:
        return None
    if not isinstance(command, Mapping):
        raise FullAgentSummaryError("states.json command must be an object")
    action = command.get("action")
    if not isinstance(action, str) or not action:
        raise FullAgentSummaryError("states.json command.action must be non-empty")
    return action


def _direct_vla_events(
    states: Sequence[Mapping[str, Any]],
) -> tuple[tuple[int, Mapping[str, Any], Mapping[str, Any]], ...]:
    """Return (state index, post-state, pre-state) for ungoverned Pi0 calls."""
    result: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]] = []
    for index, entry in enumerate(states):
        if _physical_action(entry) not in _DIRECT_VLA_ACTIONS:
            continue
        if index == 0:
            raise FullAgentSummaryError("a VLA action has no preceding state")
        result.append((index, entry, states[index - 1]))
    return tuple(result)


def _composite_event_indices(states: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    return tuple(
        index
        for index, entry in enumerate(states)
        if _physical_action(entry) in _COMPOSITE_VLA_ACTIONS
    )


def _direct_vla_costs(
    events: Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any]]],
) -> tuple[int, int | None, int | None, float]:
    chunks: list[int] = []
    elapsed_total = 0.0
    for index, entry, _pre_state in events:
        result = entry.get("result")
        if not isinstance(result, Mapping):
            raise FullAgentSummaryError(
                f"direct VLA state {index} has no structured primitive result"
            )
        chunks_used = _nonnegative_int(
            result.get("chunks_used"), f"states[{index}].result.chunks_used"
        )
        if chunks_used is not None:
            chunks.append(chunks_used)
        elapsed = _finite_nonnegative(
            entry.get("elapsed_s"), f"states[{index}].elapsed_s"
        )
        if elapsed is None:
            raise FullAgentSummaryError(
                f"direct VLA state {index} has no elapsed_s telemetry"
            )
        elapsed_total += elapsed
    chunk_total = sum(chunks) if len(chunks) == len(events) else None
    # LiberoPrimitives performs exactly one model inference per completed
    # `_vlm_chunk`. A cancelled/failed primitive may omit chunks_used after a
    # partially issued inference, so that case remains unknown.
    inference_total = chunk_total
    return len(events), inference_total, chunk_total, elapsed_total


def _direct_pre_handoff_state(
    trial: TrialManifest,
    states: Sequence[Mapping[str, Any]],
    event: tuple[int, Mapping[str, Any], Mapping[str, Any]],
) -> HandoffState:
    """Reconstruct deployment proprioception immediately before a direct call."""
    index, _post_state, pre_entry = event
    command = states[index].get("command")
    if not isinstance(command, Mapping):
        raise FullAgentSummaryError(
            f"states[{index}].command is missing for a direct VLA call"
        )
    action = command.get("action")
    prompt = command.get("prompt")
    if action not in _DIRECT_VLA_ACTIONS:
        raise FullAgentSummaryError(f"unexpected direct VLA action: {action!r}")
    semantic_target = (
        prompt if isinstance(prompt, str) and prompt else trial.task.target_description
    )
    raw_state = pre_entry.get("state")
    if not isinstance(raw_state, Mapping):
        raise FullAgentSummaryError(
            f"states[{index - 1}].state is missing before a direct VLA call"
        )
    position = _required_vector(
        raw_state.get("robot0_eef_pos"),
        length=3,
        name=f"states[{index - 1}].state.robot0_eef_pos",
    )
    quaternion = _required_vector(
        raw_state.get("robot0_eef_quat"),
        length=4,
        name=f"states[{index - 1}].state.robot0_eef_quat",
    )
    gripper = _required_vector(
        raw_state.get("robot0_gripper_qpos"),
        length=2,
        name=f"states[{index - 1}].state.robot0_gripper_qpos",
    )
    provenance = (
        FeatureProvenance(
            feature_name="eef_position_m",
            availability=FeatureAvailability.DEPLOYMENT_SENSOR,
            source="states.json.state.robot0_eef_pos",
            unit="m",
            frame="libero_world",
            provider_version="rpent-full-agent-summary/v1",
        ),
        FeatureProvenance(
            feature_name="eef_quaternion_xyzw",
            availability=FeatureAvailability.DEPLOYMENT_SENSOR,
            source="states.json.state.robot0_eef_quat",
            unit="unit_quaternion",
            frame="libero_world",
            provider_version="rpent-full-agent-summary/v1",
        ),
        FeatureProvenance(
            feature_name="gripper_opening_m",
            availability=FeatureAvailability.DERIVED_DEPLOYMENT,
            source="states.json.state.robot0_gripper_qpos",
            unit="m",
            frame="gripper",
            derivation="sum of absolute two-finger qpos values",
            provider_version="rpent-full-agent-summary/v1",
        ),
        FeatureProvenance(
            feature_name="skill",
            availability=FeatureAvailability.DERIVED_DEPLOYMENT,
            source="manifest semantic full-agent skill",
            unit="categorical",
            frame="semantic",
            provider_version="rpent-full-agent-summary/v1",
        ),
    )
    return HandoffState(
        state_id=f"{trial.trial_id}/direct-vla-pre-state-{index:04d}",
        observation_sequence=0,
        observed_elapsed_s=0.0,
        eef_position_m=position,
        eef_quaternion_xyzw=quaternion,
        gripper_opening_m=sum(abs(value) for value in gripper),
        skill=SkillIdentity(
            name=str(action),
            semantic_target=semantic_target,
            learned_controller="pi0.5",
        ),
        target=None,
        provenance=provenance,
    )


def load_probe_reset_map(
    paths: Sequence[str | Path],
) -> dict[tuple[str, int, int], str]:
    """Load exact suite/task/seed reset identities from observed probe reports."""
    from rpent.research.handoff.experiments.probes import (
        ProbeSafety,
        ProbeStatus,
        RuntimeProbeReport,
    )

    result: dict[tuple[str, int, int], str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        try:
            report = RuntimeProbeReport.model_validate(_strict_json(path))
        except Exception as exc:
            raise FullAgentSummaryError(
                f"runtime probe fails its complete schema: {path}: {exc}"
            ) from exc
        fact = report.fact("env.reset_identity")
        if (
            fact.component != "env"
            or fact.status is not ProbeStatus.OBSERVED
            or fact.safety is not ProbeSafety.READ_ONLY
        ):
            raise FullAgentSummaryError(
                f"runtime probe reset identity is not observed/read-only env evidence: {path}"
            )
        value = fact.value
        if not isinstance(value, Mapping):
            raise FullAgentSummaryError(
                f"runtime probe reset identity is not an object: {path}"
            )
        reset_raw = value.get("reset_id")
        if isinstance(reset_raw, bool) or not isinstance(reset_raw, (int, str)):
            raise FullAgentSummaryError(
                f"runtime probe reset_id must be an integer or string: {path}"
            )
        if isinstance(reset_raw, int) and reset_raw < 0:
            raise FullAgentSummaryError(
                f"runtime probe reset_id must be non-negative: {path}"
            )
        if isinstance(reset_raw, str) and not reset_raw.strip():
            raise FullAgentSummaryError(
                f"runtime probe reset_id must be non-empty: {path}"
            )
        context = value.get("context")
        if not isinstance(context, Mapping):
            raise FullAgentSummaryError(
                f"runtime probe reset identity has no context: {path}"
            )
        suite = context.get("suite")
        task = context.get("task")
        seed = context.get("seed")
        if not isinstance(suite, str) or not suite:
            raise FullAgentSummaryError(
                f"runtime probe reset context suite is invalid: {path}"
            )
        for name, item in (("task", task), ("seed", seed)):
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise FullAgentSummaryError(
                    f"runtime probe reset context {name} is invalid: {path}"
                )
        key = (suite, task, seed)
        reset_id = str(reset_raw)
        existing = result.get(key)
        if existing is not None and existing != reset_id:
            raise FullAgentSummaryError(
                f"contradictory reset probes for {key}: {existing!r} != {reset_id!r}"
            )
        result[key] = reset_id
    return result


def _load_run_reset_identity(trial: TrialManifest) -> tuple[str, Path, str]:
    """Load reset evidence captured by this exact child immediately after reset."""
    path = Path(trial.output_dir) / "reset_identity.json"
    payload = _strict_json(path)
    if not isinstance(payload, Mapping):
        raise FullAgentSummaryError(f"run-local reset sidecar is not an object: {path}")
    expected = {
        "schema_version": "rpent.research-reset-identity/v1",
        "trial_id": trial.trial_id,
        "suite": trial.task.suite,
        "task": trial.task.task,
        "seed": trial.task.seed,
        "observed_after_reset": True,
        "source": "live_env_runtime_probe",
        "probe_schema_version": "rpent.runtime-probe/v1",
        "probe_component": "libero_env",
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise FullAgentSummaryError(
            f"run-local reset sidecar disagrees with trial/runtime: {mismatches}"
        )
    reset_raw = payload.get("reset_id")
    if not isinstance(reset_raw, str) or not reset_raw:
        raise FullAgentSummaryError("run-local reset sidecar has invalid reset_id")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return reset_raw, path, digest


def _load_completion_identity(
    trial: TrialManifest,
    *,
    transcript_path: Path,
) -> tuple[Mapping[str, Any], Path, str]:
    """Load research-only completion evidence, including planner exceptions."""
    path = Path(trial.output_dir) / "completion.json"
    payload = _strict_json(path)
    if not isinstance(payload, Mapping):
        raise FullAgentSummaryError(f"completion sidecar is not an object: {path}")
    expected_digest = hashlib.sha256(transcript_path.read_bytes()).hexdigest()
    if (
        payload.get("schema_version") != "rpent.research-completion/v1"
        or payload.get("trial_id") != trial.trial_id
        or payload.get("transcript_path") != str(transcript_path.resolve())
        or payload.get("transcript_sha256") != expected_digest
    ):
        raise FullAgentSummaryError(
            f"completion sidecar does not bind this trial/transcript: {path}"
        )
    status = payload.get("status")
    error = payload.get("agent_error")
    if status not in {
        "planner_error",
        "finish_declared",
        "planner_returned_without_finish",
    }:
        raise FullAgentSummaryError(f"completion sidecar has invalid status: {status!r}")
    if status == "planner_error":
        if not isinstance(error, str) or not error:
            raise FullAgentSummaryError("planner_error completion lacks agent_error")
    elif error is not None:
        raise FullAgentSummaryError("non-error completion unexpectedly has agent_error")
    _finite_nonnegative(payload.get("elapsed_s"), "completion.elapsed_s")
    return payload, path, hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runtime_binding(trial: TrialManifest) -> tuple[Any, Path]:
    from robots.libero.handoff_runtime import (
        load_handoff_runtime_config,
        resolve_handoff_output_dir,
    )

    resolved_path = Path(trial.output_dir) / "resolved_handoff_runtime.json"
    if not resolved_path.is_file():
        raise FullAgentSummaryError(
            f"handoff-enabled trial has no resolved runtime config: {resolved_path}"
        )
    config = load_handoff_runtime_config(resolved_path)
    if not config.enabled:
        raise FullAgentSummaryError("resolved full-agent handoff config is disabled")
    expected_metadata = {
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
        "pi05_checkpoint_id": config.checkpoint_id,
        "model_artifact_id": config.model_artifact_id,
    }
    mismatches = {
        key: {"expected": expected, "actual": config.metadata.get(key)}
        for key, expected in expected_metadata.items()
        if config.metadata.get(key) != expected
    }
    if mismatches:
        raise FullAgentSummaryError(
            f"resolved runtime metadata disagrees with trial: {mismatches}"
        )
    expected_checkpoint = (
        trial.runtime.pi05_checkpoint_id or trial.condition.checkpoint_id
    )
    if config.checkpoint_id != expected_checkpoint:
        raise FullAgentSummaryError(
            "resolved controller checkpoint identity disagrees with trial runtime"
        )
    if config.controller_method != trial.condition.method:
        raise FullAgentSummaryError(
            "resolved controller method disagrees with trial condition"
        )
    output_root = resolve_handoff_output_dir(
        config, run_output_dir=trial.output_dir
    )
    sink_manifest_path = output_root / "manifest.json"
    sink_manifest = _strict_json(sink_manifest_path)
    if not isinstance(sink_manifest, Mapping) or (
        sink_manifest.get("configuration_id") != config.configuration_id
        or sink_manifest.get("controller_configuration_id")
        != config.controller_configuration_id
        or sink_manifest.get("configuration") != config.canonical_config
    ):
        raise FullAgentSummaryError(
            f"handoff sink manifest disagrees with resolved runtime: {sink_manifest_path}"
        )
    return config, output_root


def _load_detailed_outcomes(
    trial: TrialManifest, *, output_root: Path
) -> tuple[OutcomeRecord, ...]:
    if not trial.condition.handoff_enabled:
        return ()
    outcomes_path = output_root / "outcomes.jsonl"
    if not outcomes_path.exists():
        # A full agent may legitimately never call the composite tool.
        return ()
    if outcomes_path.stat().st_size == 0:
        raise FullAgentSummaryError(
            f"detailed outcome JSONL exists but is empty: {outcomes_path}"
        )
    records = read_outcome_jsonl(outcomes_path)
    wrong = [
        record.record_id
        for record in records
        if (
            record.identity.trial_id != trial.trial_id
            or record.identity.run_id != trial.experiment_id
            or record.identity.episode_id != trial.trial_id
            or record.identity.suite != trial.task.suite
            or str(record.identity.task_id) != str(trial.task.task)
            or record.identity.seed != trial.task.seed
            or record.identity.repeat_index != trial.repeat_index
            or record.identity.reset_id is None
            or record.skill.name != trial.task.skill_name
            or record.skill.semantic_target != trial.task.target_description
            or record.source_revision != trial.source_revision
        )
    ]
    if wrong:
        raise FullAgentSummaryError(
            f"detailed outcomes disagree with trial {trial.trial_id}: {wrong[:10]}"
        )
    return records


def _resolved_controller(
    trial: TrialManifest,
    records: Sequence[OutcomeRecord],
    runtime_config: Any | None,
) -> ControllerIdentity:
    if runtime_config is not None:
        expected = ControllerIdentity(
            method=runtime_config.controller_method,
            implementation_version=runtime_config.controller_implementation_version,
            checkpoint_id=runtime_config.checkpoint_id,
            configuration_id=runtime_config.controller_configuration_id,
        )
        wrong = [
            record.record_id
            for record in records
            if record.controller != expected
        ]
        if wrong:
            raise FullAgentSummaryError(
                f"detailed controllers disagree with resolved runtime: {wrong[:10]}"
            )
        return expected
    if records:
        controllers = {record.controller.canonical_json(): record.controller for record in records}
        if len(controllers) != 1:
            raise FullAgentSummaryError(
                f"trial {trial.trial_id} contains multiple controller identities"
            )
        controller = next(iter(controllers.values()))
        if controller.method != trial.condition.method:
            raise FullAgentSummaryError(
                f"outcome controller method {controller.method!r} disagrees with "
                f"condition {trial.condition.method!r}"
            )
        return controller
    scientific = {
        "method": trial.condition.method,
        "condition": trial.condition.model_dump(mode="json", exclude_none=False),
        "checkpoint_id": trial.condition.checkpoint_id,
        "pi05_checkpoint_id": trial.runtime.pi05_checkpoint_id,
        "model_artifact_path": trial.model_artifact_path,
        "pi05_checkpoint_path": trial.runtime.pi05_checkpoint_path,
    }
    return ControllerIdentity(
        method=trial.condition.method,
        implementation_version=(
            "rpent-original-harness/v1"
            if trial.condition.method == "original_harness"
            else "rpent-handoff-disabled-condition/v1"
        ),
        checkpoint_id=(
            trial.runtime.pi05_checkpoint_id or trial.condition.checkpoint_id
        ),
        configuration_id=stable_identifier("controller", scientific),
    )


def summarize_full_agent_trial(
    trial: TrialManifest,
    *,
    probe_resets: Mapping[tuple[str, int, int], str] | None = None,
) -> OutcomeRecord:
    """Create one episode-scoped record from immutable full-agent artifacts."""
    if trial.execution_layer is not ExecutionLayer.FULL_AGENT:
        raise FullAgentSummaryError(f"trial is not full_agent: {trial.trial_id}")
    output_dir = Path(trial.output_dir)
    transcript_path = _single_artifact(
        output_dir, "transcript_*.json", "RPent transcript"
    )
    transcript = _strict_json(transcript_path)
    if not isinstance(transcript, Mapping):
        raise FullAgentSummaryError("RPent transcript must be a JSON object")
    completion, completion_path, completion_sha256 = _load_completion_identity(
        trial,
        transcript_path=transcript_path,
    )
    states_path = output_dir / "states.json"
    states_payload = _strict_json(states_path)
    if (
        not isinstance(states_payload, list)
        or not states_payload
        or any(not isinstance(entry, Mapping) for entry in states_payload)
    ):
        raise FullAgentSummaryError(
            f"states trace must be a non-empty array of objects: {states_path}"
        )
    states = tuple(states_payload)
    for expected_index, entry in enumerate(states):
        if entry.get("step_idx") != expected_index:
            raise FullAgentSummaryError(
                f"states trace index mismatch at {expected_index}: "
                f"{entry.get('step_idx')!r}"
            )
    final_state = states[-1]
    previous_terminated = False
    previous_truncated = False
    for state_index, entry in enumerate(states):
        entry_terminated = entry.get("libero_terminated")
        entry_truncated = entry.get("episode_truncated")
        if not isinstance(entry_terminated, bool) or not isinstance(
            entry_truncated, bool
        ):
            raise FullAgentSummaryError(
                f"states[{state_index}] lacks boolean term/trunc flags"
            )
        if previous_terminated and not entry_terminated:
            raise FullAgentSummaryError("termination flag is not latched")
        if previous_truncated and not entry_truncated:
            raise FullAgentSummaryError("truncation flag is not latched")
        previous_terminated = entry_terminated
        previous_truncated = entry_truncated
    terminated = final_state.get("libero_terminated")
    truncated = final_state.get("episode_truncated")
    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        raise FullAgentSummaryError(
            "final state must contain boolean termination and truncation flags"
        )

    transcript_identity = {
        "suite": transcript.get("suite"),
        "task": transcript.get("task"),
        "seed": transcript.get("seed"),
    }
    expected_transcript_identity = {
        "suite": trial.task.suite,
        "task": trial.task.task,
        "seed": trial.task.seed,
    }
    if transcript_identity != expected_transcript_identity:
        raise FullAgentSummaryError(
            f"transcript identity disagrees with trial {trial.trial_id}: "
            f"{transcript_identity!r} != {expected_transcript_identity!r}"
        )
    if transcript.get("model") != trial.planner.model:
        raise FullAgentSummaryError(
            "transcript planner model disagrees with the trial manifest"
        )

    runtime_config = None
    detailed_output_root = None
    if trial.condition.handoff_enabled:
        runtime_config, detailed_output_root = _load_runtime_binding(trial)
    records = (
        _load_detailed_outcomes(trial, output_root=detailed_output_root)
        if detailed_output_root is not None
        else ()
    )
    run_reset, reset_sidecar_path, reset_sidecar_sha256 = (
        _load_run_reset_identity(trial)
    )
    configured_reset = trial.task.reset_id
    detailed_resets = {
        str(record.identity.reset_id)
        for record in records
        if record.identity.reset_id is not None
    }
    if len(detailed_resets) > 1:
        raise FullAgentSummaryError(
            f"trial {trial.trial_id} contains multiple live reset identities"
        )
    detailed_reset = next(iter(detailed_resets), None)
    probe_key = (trial.task.suite, trial.task.task, trial.task.seed)
    probed_reset = (probe_resets or {}).get(probe_key)
    candidates = {
        str(value)
        for value in (configured_reset, detailed_reset, probed_reset, run_reset)
        if value is not None
    }
    if len(candidates) > 1:
        raise FullAgentSummaryError(
            f"configured/detailed/probed/run-local reset identity contradiction for "
            f"{trial.trial_id}: {sorted(candidates)}"
        )
    # The run-local sidecar is definitive episode evidence. Detached probes are
    # useful only as expectations/cross-checks because they may describe a
    # different process or attempt.
    reset_id = run_reset

    stats = transcript.get("stats", {})
    if not isinstance(stats, Mapping):
        raise FullAgentSummaryError("transcript stats must be an object")
    elapsed = _finite_nonnegative(transcript.get("elapsed_s"), "transcript.elapsed_s")
    if elapsed is None:
        raise FullAgentSummaryError("transcript.elapsed_s is required")
    planner_time = _finite_nonnegative(stats.get("elapsed_s"), "stats.elapsed_s")
    planner_time_source = (
        "transcript.stats.elapsed_s" if planner_time is not None else None
    )
    llm_turns = _nonnegative_int(stats.get("turns_used"), "stats.turns_used")
    input_tokens = _nonnegative_int(
        stats.get("total_input_tokens"), "stats.total_input_tokens"
    )
    output_tokens = _nonnegative_int(
        stats.get("total_output_tokens"), "stats.total_output_tokens"
    )
    tool_calls = _nonnegative_int(stats.get("tool_calls"), "stats.tool_calls")

    finish = transcript.get("finish")
    if finish is not None and not isinstance(finish, Mapping):
        raise FullAgentSummaryError(
            "transcript finish must be an object or null"
        )
    finish_status = finish.get("status") if isinstance(finish, Mapping) else None
    if finish_status is not None and (
        not isinstance(finish_status, str)
        or finish_status not in {"success", "failure", "stuck"}
    ):
        raise FullAgentSummaryError(
            "transcript finish.status must be success, failure, or stuck"
        )
    llm_finish = unavailable_signal("planner did not provide a finish declaration")
    if finish_status is not None:
        llm_finish = OutcomeSignal(
            value=finish_status == "success",
            source=LabelSource.PLANNER_DECLARATION,
            definition="planner-declared finish status; not task success",
            evaluator_id="RPent transcript finish.status",
        )
    completion_status = completion["status"]
    if completion_status == "finish_declared" and finish_status is None:
        raise FullAgentSummaryError(
            "completion says finish_declared but transcript has no finish status"
        )
    if (
        completion_status == "planner_returned_without_finish"
        and finish is not None
    ):
        raise FullAgentSummaryError(
            "completion says no finish but transcript contains a finish object"
        )

    direct_events = _direct_vla_events(states)
    (
        direct_tool_calls,
        direct_inference_invocations,
        direct_chunks,
        direct_vla_time,
    ) = _direct_vla_costs(direct_events)
    composite_indices = _composite_event_indices(states)
    record_by_id = {record.record_id: record for record in records}
    composite_bindings: list[tuple[int, OutcomeRecord]] = []
    seen_composite_records: set[str] = set()
    for invocation_number, state_index in enumerate(composite_indices, start=1):
        result = states[state_index].get("result")
        if not isinstance(result, Mapping):
            raise FullAgentSummaryError(
                f"composite states[{state_index}].result is not an object"
            )
        record_id = result.get("handoff_record_id")
        invocation_id = result.get("invocation_id")
        expected_invocation = (
            f"{trial.trial_id}/handoff-{invocation_number:04d}"
        )
        if not isinstance(record_id, str) or record_id not in record_by_id:
            raise FullAgentSummaryError(
                f"composite state {state_index} does not bind a detailed outcome"
            )
        if record_id in seen_composite_records:
            raise FullAgentSummaryError(
                f"composite state reuses detailed outcome {record_id!r}"
            )
        record = record_by_id[record_id]
        if (
            invocation_id != expected_invocation
            or record.identity.invocation_id != expected_invocation
            or result.get("controller_configuration_id")
            != record.controller.configuration_id
        ):
            raise FullAgentSummaryError(
                f"composite invocation binding mismatch at state {state_index}"
            )
        seen_composite_records.add(record_id)
        composite_bindings.append((state_index, record))
    unbound_records = sorted(set(record_by_id).difference(seen_composite_records))
    if unbound_records:
        raise FullAgentSummaryError(
            f"detailed outcomes have no composite state binding: {unbound_records[:10]}"
        )
    detailed_handoff_states = {
        state_index: record.pre_handoff_state
        for state_index, record in composite_bindings
        if record.handoff_occurred and record.pre_handoff_state is not None
    }
    direct_handoff_states = {
        state_index: _direct_pre_handoff_state(trial, states, event)
        for event in direct_events
        for state_index in (event[0],)
    }
    all_handoff_states = {**detailed_handoff_states, **direct_handoff_states}
    handoff_occurred = bool(all_handoff_states)
    last_handoff_state = (
        all_handoff_states[max(all_handoff_states)]
        if all_handoff_states
        else None
    )
    decision_trace = tuple(
        decision for record in records for decision in record.decision_trace
    )

    def optional_detailed_sum(field: str) -> int | None:
        if not records:
            return 0
        values = [getattr(record.costs, field) for record in records]
        if any(value is None for value in values):
            return None
        return sum(int(value) for value in values)

    detailed_chunks = optional_detailed_sum("vla_chunks")
    combined_chunks = (
        detailed_chunks + direct_chunks
        if detailed_chunks is not None and direct_chunks is not None
        else None
    )
    detailed_vla_actions = optional_detailed_sum("vla_env_actions")
    protocol_adherent = not (
        trial.condition.handoff_enabled and direct_tool_calls > 0
    )
    detailed_inference_invocations = optional_detailed_sum("vla_invocations")
    combined_inference_invocations = (
        detailed_inference_invocations + direct_inference_invocations
        if detailed_inference_invocations is not None
        and direct_inference_invocations is not None
        else None
    )
    detailed_failure_counts: dict[str, int] = {}
    for detailed_record in records:
        failure = detailed_record.termination.failure_mode.value
        detailed_failure_counts[failure] = (
            detailed_failure_counts.get(failure, 0) + 1
        )
    direct_tool_errors = [
        {
            "state_index": state_index,
            "action": _physical_action(post_state),
            "error": post_state.get("result", {}).get("error"),
        }
        for state_index, post_state, _pre_state in direct_events
        if isinstance(post_state.get("result"), Mapping)
        and post_state.get("result", {}).get("error") is not None
    ]
    episode_failure = (
        FailureMode.TRUNCATION
        if truncated
        else (
            FailureMode.UNKNOWN
            if completion_status == "planner_error"
            or (
                not terminated
                and (finish_status is None or bool(direct_tool_errors))
            )
            else FailureMode.NONE
        )
    )
    episode_reason = (
        TerminationReason.UNKNOWN
        if episode_failure is FailureMode.UNKNOWN
        else TerminationReason.COMPLETED
    )

    controller = _resolved_controller(trial, records, runtime_config)
    record_payload = {
        "version": EPISODE_SUMMARY_VERSION,
        "trial_id": trial.trial_id,
        "controller": controller.model_dump(mode="json", exclude_none=False),
        "transcript_sha256": hashlib.sha256(transcript_path.read_bytes()).hexdigest(),
        "states_sha256": hashlib.sha256(states_path.read_bytes()).hexdigest(),
        "reset_identity_sha256": reset_sidecar_sha256,
        "completion_sha256": completion_sha256,
    }
    return OutcomeRecord(
        record_id=stable_identifier("full-agent-episode", record_payload),
        identity=TrialIdentity(
            run_id=trial.experiment_id,
            episode_id=trial.trial_id,
            trial_id=trial.trial_id,
            invocation_id=f"{trial.trial_id}/episode-summary",
            suite=trial.task.suite,
            task_id=trial.task.task,
            seed=trial.task.seed,
            reset_id=reset_id,
            repeat_index=trial.repeat_index,
        ),
        skill=SkillIdentity(
            name=trial.task.skill_name,
            semantic_target=trial.task.target_description,
            learned_controller="pi0.5",
        ),
        controller=controller,
        pre_handoff_state=last_handoff_state,
        handoff_occurred=handoff_occurred,
        decision_trace=decision_trace,
        labels=OutcomeLabels(
            primitive_success=unavailable_signal(
                "episode summary does not substitute a primitive-level result"
            ),
            skill_success=unavailable_signal(
                "episode summary has no independent skill evaluator"
            ),
            task_success=OutcomeSignal(
                value=terminated,
                source=LabelSource.OFFICIAL_TERMINATION,
                definition="final latched official LIBERO termination",
                evaluator_id="states.json.libero_terminated",
            ),
            episode_truncated=OutcomeSignal(
                value=truncated,
                source=LabelSource.RUNTIME,
                definition="final latched LIBERO truncation",
                evaluator_id="states.json.episode_truncated",
            ),
            llm_finish=llm_finish,
        ),
        costs=CostRecord(
            analytic_steps=sum(record.costs.analytic_steps for record in records),
            analytic_distance_m=sum(
                record.costs.analytic_distance_m for record in records
            ),
            analytic_time_s=sum(record.costs.analytic_time_s for record in records),
            vla_invocations=combined_inference_invocations,
            vla_chunks=combined_chunks,
            # Direct Pi0 states expose chunk counts but not a source-verified
            # number of executed actions, so a mixed/direct episode is unknown.
            vla_env_actions=(
                detailed_vla_actions if direct_tool_calls == 0 else None
            ),
            vla_time_s=(
                sum(record.costs.vla_time_s for record in records)
                + direct_vla_time
            ),
            # The outer states trace does not expose source-verified executed
            # env-step counts for all planner-selected analytic primitives.
            total_env_actions=None,
            total_elapsed_s=elapsed,
            planner_time_s=planner_time,
            llm_turns=llm_turns,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
        timing=TimingRecord(
            started_monotonic_s=0.0,
            ended_monotonic_s=elapsed,
            record_time_s=None,
        ),
        termination=TerminationRecord(
            reason=episode_reason,
            failure_mode=episode_failure,
            final_governor_state=(
                GovernorState.TRUNCATED
                if truncated
                else (
                    GovernorState.ABORT
                    if episode_failure is FailureMode.UNKNOWN
                    else GovernorState.DONE
                )
            ),
            episode_terminated=terminated,
            episode_truncated=truncated,
            message="post-run full-agent episode summary",
        ),
        source_revision=trial.source_revision,
        metadata={
            "data_status": "observed",
            "record_scope": "full_agent_episode",
            "execution_layer": trial.execution_layer.value,
            "condition": trial.condition.name,
            "representation": trial.condition.feature_set.value,
            "evidence_mode": trial.condition.evidence.value,
            "uncertainty_mode": trial.condition.uncertainty.value,
            "hierarchy_mode": trial.condition.hierarchy.value,
            "detailed_outcome_record_ids": [record.record_id for record in records],
            "direct_vla_tool_calls": direct_tool_calls,
            "direct_vla_inference_invocations": direct_inference_invocations,
            "composite_tool_invocations": len(composite_indices),
            "protocol_adherent": protocol_adherent,
            "protocol_violation": (
                "handoff-enabled trial invoked ungoverned pi0_pick/pi0_doubled"
                if not protocol_adherent
                else None
            ),
            "detailed_failure_counts": detailed_failure_counts,
            "direct_tool_errors": direct_tool_errors,
            "transcript_path": str(transcript_path.resolve()),
            "states_path": str(states_path.resolve()),
            "reset_identity_path": str(reset_sidecar_path.resolve()),
            "reset_identity_sha256": reset_sidecar_sha256,
            "reset_identity_evidence": "run_local_post_reset_runtime_probe",
            "detached_probe_reset_id": probed_reset,
            "completion_path": str(completion_path.resolve()),
            "completion_sha256": completion_sha256,
            "completion_status": completion_status,
            "planner_error": completion.get("agent_error"),
            "episode_relative_timing_origin": True,
            "analytic_cost_scope": "local_governor_staging_only",
            "total_env_actions_unavailable_reason": (
                "full states trace lacks executed-step counts for every tool"
            ),
            "planner_time_source": planner_time_source,
            "tool_calls": tool_calls,
            "planner_finish_status": finish_status,
            "pi05_checkpoint_path": trial.runtime.pi05_checkpoint_path,
        },
    )


__all__ = [
    "EPISODE_SUMMARY_VERSION",
    "FullAgentSummaryError",
    "load_probe_reset_map",
    "summarize_full_agent_trial",
]
