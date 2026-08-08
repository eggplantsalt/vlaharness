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
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.evaluation.aggregate import read_outcome_jsonl
from rpent.research.handoff.experiments.config import (
    ExecutionLayer,
    stable_digest,
    stable_identifier,
)
from rpent.research.handoff.experiments.full_agent import FullAgentChildPlan
from rpent.research.handoff.experiments.lifecycle import (
    TrialEventType,
    TrialLifecycleEvent,
)
from rpent.research.handoff.experiments.manifest import TrialManifest
from rpent.research.handoff.types import (
    ControllerIdentity,
    CostRecord,
    FeatureAvailability,
    FeatureProvenance,
    FailureMode,
    GovernorState,
    HandoffRecord,
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
    outcome_record_id,
    unavailable_signal,
)

EPISODE_SUMMARY_VERSION = "rpent.handoff-full-agent-episode-summary/v1"
FULL_AGENT_ATTEMPT_VERSION = "rpent.handoff-full-agent-attempt/v1"
DIRECT_VLA_ATTEMPT_VERSION = "rpent.research-direct-vla-attempt/v1"

_DIRECT_VLA_ACTIONS = frozenset({"pi0_pick", "pi0_doubled"})
_COMPOSITE_VLA_ACTIONS = frozenset(
    {"handoff_pi0_pick", "handoff_pi0_doubled"}
)


class FullAgentSummaryError(ValueError):
    """A full-agent artifact set is missing, ambiguous, or contradictory."""


class FullAgentAttemptRecord(HandoffRecord):
    """Wrapper evidence written before constructing the runtime command."""

    schema_version: Literal[FULL_AGENT_ATTEMPT_VERSION] = (
        FULL_AGENT_ATTEMPT_VERSION
    )
    trial_id: str
    manifest_id: str
    plan_id: str
    source_revision: str
    cwd: str
    resolved_inner_command_sha256: str

    @field_validator(
        "trial_id",
        "manifest_id",
        "plan_id",
        "source_revision",
        "cwd",
        "resolved_inner_command_sha256",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value


class DirectVlaAttemptEvent(HandoffRecord):
    """One fsynced event in the Original-Harness direct-VLA hash chain."""

    schema_version: Literal[DIRECT_VLA_ATTEMPT_VERSION] = (
        DIRECT_VLA_ATTEMPT_VERSION
    )
    event_sequence: int = Field(ge=1)
    previous_event_sha256: str | None
    attempt_index: int = Field(ge=1)
    step_index: int = Field(ge=1)
    trial_id: str
    manifest_id: str
    plan_id: str
    source_revision: str
    reset_id: str
    reset_identity_sha256: str
    runtime_attestation_id: str
    runtime_attestation_sha256: str
    tool_name: Literal["pi0_pick", "pi0_doubled"]
    phase: Literal[
        "started",
        "completed",
        "returned_error",
        "cancelled",
        "error",
    ]
    vla_attempted: Literal[True]
    attempt_unit: Literal["planner_visible_vla_tool_invocation"]
    elapsed_s: float | None = Field(ge=0.0)
    error_type: str | None
    error: str | None
    recorded_before_state_dump: Literal[True]
    event_sha256: str

    @field_validator(
        "trial_id",
        "manifest_id",
        "plan_id",
        "source_revision",
        "reset_id",
        "reset_identity_sha256",
        "runtime_attestation_id",
        "runtime_attestation_sha256",
        "event_sha256",
    )
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("previous_event_sha256", "error_type")
    @classmethod
    def validate_optional_text(
        cls, value: str | None, info
    ) -> str | None:
        if value is not None and not value:
            raise ValueError(f"{info.field_name} cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_phase_and_hash(self) -> Self:
        if self.phase == "started":
            if any(
                value is not None
                for value in (self.elapsed_s, self.error_type, self.error)
            ):
                raise ValueError("started direct-VLA event cannot be terminal")
        elif self.elapsed_s is None:
            raise ValueError("terminal direct-VLA event requires elapsed_s")
        if self.phase == "completed" and any(
            value is not None for value in (self.error_type, self.error)
        ):
            raise ValueError("completed direct-VLA event cannot contain an error")
        if self.phase in {"returned_error", "cancelled", "error"} and (
            self.error_type is None or self.error is None
        ):
            raise ValueError(f"{self.phase} direct-VLA event requires error evidence")
        payload = self.model_dump(
            mode="json", exclude={"event_sha256"}, exclude_none=False
        )
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        expected = hashlib.sha256(canonical).hexdigest()
        if self.event_sha256 != expected:
            raise ValueError("direct-VLA event hash does not bind its payload")
        return self


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


def _validate_plan_binding(
    trial: TrialManifest,
    plan: FullAgentChildPlan,
) -> None:
    from rpent.research.handoff.experiments.full_agent import build_child_plan
    from rpent.research.handoff.experiments.manifest import load_manifest

    expected_original = trial.condition.method == "original_harness"
    mismatches: dict[str, dict[str, Any]] = {}
    expected = {
        "trial_id": trial.trial_id,
        "output_dir": str(Path(trial.output_dir)),
        "original_harness": expected_original,
    }
    actual = {
        "trial_id": plan.trial_id,
        "output_dir": plan.output_dir,
        "original_harness": plan.original_harness,
    }
    for name, value in expected.items():
        observed = actual[name]
        if name == "output_dir":
            equal = Path(str(observed)).resolve() == Path(str(value)).resolve()
        else:
            equal = observed == value
        if not equal:
            mismatches[name] = {"expected": value, "actual": observed}
    if trial.execution_layer is not ExecutionLayer.FULL_AGENT:
        mismatches["execution_layer"] = {
            "expected": ExecutionLayer.FULL_AGENT.value,
            "actual": trial.execution_layer.value,
        }
    manifest_path = Path(plan.manifest_path).expanduser().resolve()
    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:
        raise FullAgentSummaryError(
            f"reviewed full-agent plan manifest is invalid: {manifest_path}: {exc}"
        ) from exc
    bound_trials = tuple(
        item for item in manifest.trials if item.trial_id == trial.trial_id
    )
    if manifest.manifest_id != plan.manifest_id:
        mismatches["manifest_id"] = {
            "expected": manifest.manifest_id,
            "actual": plan.manifest_id,
        }
    if len(bound_trials) != 1 or bound_trials[0] != trial:
        mismatches["manifest_trial"] = {
            "expected": trial.trial_id,
            "actual": [item.trial_id for item in bound_trials],
        }
    try:
        rebuilt = build_child_plan(
            trial,
            manifest_path=manifest_path,
            repo_root=plan.cwd,
            python_executable=plan.wrapper_command[0],
        )
    except Exception as exc:
        raise FullAgentSummaryError(
            f"cannot reconstruct reviewed full-agent plan: {exc}"
        ) from exc
    if rebuilt != plan:
        mismatches["exact_execution_plan"] = {
            "expected_plan_id": rebuilt.plan_id,
            "actual_plan_id": plan.plan_id,
        }
    if mismatches:
        raise FullAgentSummaryError(
            f"full-agent plan disagrees with trial: {mismatches}"
        )


def _validate_lifecycle_binding(
    trial: TrialManifest,
    plan: FullAgentChildPlan,
    events: Sequence[TrialLifecycleEvent],
) -> tuple[tuple[TrialLifecycleEvent, ...], TrialLifecycleEvent | None]:
    from rpent.research.handoff.experiments.lifecycle import _validate_transition

    relevant_raw = tuple(
        event for event in events if event.trial_id == trial.trial_id
    )
    relevant_list: list[TrialLifecycleEvent] = []
    previous: TrialLifecycleEvent | None = None
    for event in relevant_raw:
        try:
            validated = TrialLifecycleEvent.model_validate(
                event.model_dump(mode="python", exclude_none=False)
            )
        except Exception as exc:
            raise FullAgentSummaryError(
                f"invalid lifecycle evidence for {trial.trial_id}: {exc}"
            ) from exc
        expected_sequence = 0 if previous is None else previous.sequence + 1
        if validated.sequence != expected_sequence:
            raise FullAgentSummaryError(
                f"lifecycle sequence is not contiguous for {trial.trial_id}: "
                f"expected {expected_sequence}, got {validated.sequence}"
            )
        try:
            _validate_transition(
                previous,
                event=validated.event,
                attempt=validated.attempt,
            )
        except ValueError as exc:
            raise FullAgentSummaryError(
                f"invalid lifecycle transition for {trial.trial_id}: {exc}"
            ) from exc
        relevant_list.append(validated)
        previous = validated
    relevant = tuple(relevant_list)
    if not relevant:
        raise FullAgentSummaryError(
            f"trial {trial.trial_id} has no execution lifecycle evidence"
        )
    starts = [event for event in relevant if event.event is TrialEventType.STARTED]
    if len(starts) != 1:
        raise FullAgentSummaryError(
            f"trial {trial.trial_id} must have exactly one execution start; "
            f"found {len(starts)}"
        )
    start = starts[0]
    if start.details.get("plan_id") != plan.plan_id:
        raise FullAgentSummaryError("lifecycle start does not bind the reviewed plan")
    if start.artifact_path is None or Path(start.artifact_path).resolve() != Path(
        plan.output_dir
    ).resolve():
        raise FullAgentSummaryError("lifecycle start output path disagrees with plan")
    if any(event.attempt != start.attempt for event in relevant):
        raise FullAgentSummaryError(
            "full-agent summary refuses multiple lifecycle attempts for one trial ID"
        )
    terminals = [event for event in relevant if event.event.terminal]
    if len(terminals) > 1:
        raise FullAgentSummaryError("trial lifecycle has multiple terminal events")
    terminal = terminals[0] if terminals else None
    if terminal is not None:
        if terminal is not relevant[-1]:
            raise FullAgentSummaryError("terminal lifecycle event is not final")
        if terminal.artifact_path is None or Path(
            terminal.artifact_path
        ).resolve() != Path(plan.output_dir).resolve():
            raise FullAgentSummaryError(
                "terminal lifecycle output path disagrees with plan"
            )
        terminal_plan_id = terminal.details.get("plan_id")
        if terminal_plan_id is not None and terminal_plan_id != plan.plan_id:
            raise FullAgentSummaryError(
                "terminal lifecycle event disagrees with reviewed plan"
            )
    return relevant, terminal


def _load_attempt_identity(
    trial: TrialManifest,
    plan: FullAgentChildPlan,
    *,
    required: bool,
) -> tuple[FullAgentAttemptRecord | None, Path, str | None]:
    path = Path(trial.output_dir) / "attempt.json"
    if not path.is_file():
        if required:
            raise FullAgentSummaryError(f"full-agent attempt evidence is missing: {path}")
        return None, path, None
    try:
        record = FullAgentAttemptRecord.model_validate(_strict_json(path))
    except Exception as exc:
        raise FullAgentSummaryError(f"invalid full-agent attempt evidence: {exc}") from exc
    expected = {
        "trial_id": trial.trial_id,
        "manifest_id": plan.manifest_id,
        "plan_id": plan.plan_id,
        "source_revision": trial.source_revision,
        "cwd": str(Path(plan.cwd).resolve()),
        "resolved_inner_command_sha256": stable_digest(
            plan.resolved_inner_command
        ),
    }
    mismatches = {
        name: {"expected": value, "actual": getattr(record, name)}
        for name, value in expected.items()
        if (
            Path(getattr(record, name)).resolve() != Path(value).resolve()
            if name == "cwd"
            else getattr(record, name) != value
        )
    }
    if mismatches:
        raise FullAgentSummaryError(
            f"full-agent attempt disagrees with plan/trial: {mismatches}"
        )
    return record, path, hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runtime_attestation_identity(
    trial: TrialManifest,
    plan: FullAgentChildPlan,
    *,
    required: bool,
) -> tuple[Any | None, Path, str | None]:
    from rpent.research.handoff.experiments.runtime_identity import (
        verify_runtime_attestation_binding,
    )

    path = Path(trial.output_dir) / "runtime_identity.json"
    if not path.is_file():
        if required:
            raise FullAgentSummaryError(
                f"full-agent runtime attestation is missing: {path}"
            )
        return None, path, None
    try:
        attestation, digest = verify_runtime_attestation_binding(
            path,
            trial_id=trial.trial_id,
            manifest_id=plan.manifest_id,
            plan_id=plan.plan_id,
            source_revision=trial.source_revision,
        )
    except Exception as exc:
        raise FullAgentSummaryError(
            f"invalid full-agent runtime attestation: {exc}"
        ) from exc
    expected = (
        (
            "pi0.5_vla",
            trial.runtime.pi05_checkpoint_id,
            trial.runtime.pi05_checkpoint_path,
        ),
        (
            "sam3",
            trial.runtime.sam3_checkpoint_id,
            trial.runtime.sam3_checkpoint_path,
        ),
    )
    mismatches: list[dict[str, Any]] = []
    for observation, (component, checkpoint_id, checkpoint_path) in zip(
        attestation.observations, expected, strict=True
    ):
        expected_external = (
            trial.runtime.vla_endpoint is not None
            if component == "pi0.5_vla"
            else trial.runtime.sam3_endpoint is not None
        )
        expected_path = (
            str(Path(checkpoint_path).expanduser().resolve())
            if checkpoint_path is not None
            else None
        )
        observed_expected_path = (
            str(Path(observation.expected_checkpoint_path).expanduser().resolve())
            if observation.expected_checkpoint_path is not None
            else None
        )
        probe_payload = observation.probe_payload
        checkpoint_payload = probe_payload.get("checkpoint")
        try:
            canonical_probe = json.dumps(
                probe_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise FullAgentSummaryError(
                f"runtime attestation {component} probe payload is invalid: {exc}"
            ) from exc
        probe_sha256 = hashlib.sha256(canonical_probe).hexdigest()
        observed_path = str(observation.observed_checkpoint_path)
        payload_path = (
            checkpoint_payload.get("path")
            if isinstance(checkpoint_payload, Mapping)
            else None
        )
        local_path_matches = True
        if not expected_external and expected_path is not None:
            local_path_matches = (
                str(Path(observed_path).expanduser().resolve()) == expected_path
            )
        if (
            observation.component != component
            or observation.expected_checkpoint_id != checkpoint_id
            or observation.observed_checkpoint_id != checkpoint_id
            or observed_expected_path != expected_path
            or observation.external_endpoint is not expected_external
            or observation.probe_sha256 != probe_sha256
            or probe_payload.get("schema_version") != "rpent.runtime-probe/v1"
            or probe_payload.get("component") != component
            or not isinstance(checkpoint_payload, Mapping)
            or checkpoint_payload.get("configured_id") != checkpoint_id
            or checkpoint_payload.get("path") != observed_path
            or checkpoint_payload.get("exists") is not True
            or payload_path != observed_path
            or not local_path_matches
        ):
            mismatches.append(
                {
                    "component": component,
                    "expected_checkpoint_id": checkpoint_id,
                    "observed": observation.model_dump(
                        mode="json", exclude_none=False
                    ),
                }
            )
    if mismatches:
        raise FullAgentSummaryError(
            f"runtime attestation disagrees with trial checkpoints: {mismatches}"
        )
    return attestation, path, digest


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


def _load_direct_vla_attempts(
    trial: TrialManifest,
    plan: FullAgentChildPlan,
    *,
    reset_id: str,
    reset_identity_sha256: str,
    runtime_attestation: Any,
    runtime_attestation_sha256: str,
    completion: Mapping[str, Any] | None,
) -> tuple[tuple[DirectVlaAttemptEvent, ...], Path, str | None]:
    path = (Path(trial.output_dir) / "direct_vla_attempts.jsonl").resolve()
    completion_path = (
        completion.get("direct_vla_attempts_path")
        if completion is not None
        else None
    )
    completion_sha256 = (
        completion.get("direct_vla_attempts_sha256")
        if completion is not None
        else None
    )
    if not path.is_file():
        if completion_path is not None or completion_sha256 is not None:
            raise FullAgentSummaryError(
                "completion binds a missing direct-VLA attempt journal"
            )
        return (), path, None
    if completion is not None and (
        completion_path is None or completion_sha256 is None
    ):
        raise FullAgentSummaryError(
            "direct-VLA attempt journal exists but completion does not bind it"
        )

    records: list[DirectVlaAttemptEvent] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.endswith("\n"):
                raise FullAgentSummaryError(
                    f"direct-VLA journal has a torn final line at {line_number}"
                )
            line = raw_line[:-1]
            if not line:
                raise FullAgentSummaryError(
                    f"direct-VLA journal has a blank line at {line_number}"
                )

            def reject_constant(token: str) -> None:
                raise ValueError(f"non-finite JSON constant {token!r}")

            def reject_duplicates(
                pairs: list[tuple[str, Any]],
            ) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"duplicate JSON key {key!r}")
                    result[key] = value
                return result

            try:
                payload = json.loads(
                    line,
                    parse_constant=reject_constant,
                    object_pairs_hook=reject_duplicates,
                )
                event = DirectVlaAttemptEvent.model_validate(payload)
            except Exception as exc:
                raise FullAgentSummaryError(
                    f"invalid direct-VLA journal line {line_number}: {exc}"
                ) from exc
            records.append(event)
    if not records:
        raise FullAgentSummaryError("direct-VLA attempt journal is empty")

    previous_sha256: str | None = None
    attempts: dict[int, list[DirectVlaAttemptEvent]] = {}
    open_attempt: DirectVlaAttemptEvent | None = None
    next_attempt_index = 1
    previous_start_step = 0
    previous_terminal_phase: str | None = None
    expected_identity = {
        "trial_id": trial.trial_id,
        "manifest_id": plan.manifest_id,
        "plan_id": plan.plan_id,
        "source_revision": trial.source_revision,
        "reset_id": reset_id,
        "reset_identity_sha256": reset_identity_sha256,
        "runtime_attestation_id": runtime_attestation.attestation_id,
        "runtime_attestation_sha256": runtime_attestation_sha256,
    }
    for sequence, event in enumerate(records, start=1):
        if event.event_sequence != sequence:
            raise FullAgentSummaryError(
                "direct-VLA journal event_sequence is not contiguous"
            )
        if event.previous_event_sha256 != previous_sha256:
            raise FullAgentSummaryError(
                "direct-VLA journal previous-event hash chain is broken"
            )
        mismatches = {
            key: {"expected": value, "actual": getattr(event, key)}
            for key, value in expected_identity.items()
            if getattr(event, key) != value
        }
        if mismatches:
            raise FullAgentSummaryError(
                f"direct-VLA event identity mismatch: {mismatches}"
            )
        if event.phase == "started":
            if open_attempt is not None:
                raise FullAgentSummaryError(
                    "direct-VLA attempts overlap or interleave in the journal"
                )
            if event.attempt_index != next_attempt_index:
                raise FullAgentSummaryError(
                    "direct-VLA attempt indices are not contiguous from one"
                )
            if event.step_index < previous_start_step or (
                event.step_index == previous_start_step
                and previous_terminal_phase != "error"
            ):
                raise FullAgentSummaryError(
                    "direct-VLA attempt step index did not advance after a "
                    "state-producing attempt"
                )
            open_attempt = event
            previous_start_step = event.step_index
        else:
            if open_attempt is None:
                raise FullAgentSummaryError(
                    "direct-VLA terminal event has no open attempt"
                )
            if event.attempt_index != open_attempt.attempt_index:
                raise FullAgentSummaryError(
                    "direct-VLA terminal event interleaves attempt identities"
                )
            open_attempt = None
            previous_terminal_phase = event.phase
            next_attempt_index += 1
        attempts.setdefault(event.attempt_index, []).append(event)
        previous_sha256 = event.event_sha256

    if sorted(attempts) != list(range(1, len(attempts) + 1)):
        raise FullAgentSummaryError(
            "direct-VLA attempt indices are not contiguous from one"
        )
    for attempt_index, attempt_events in attempts.items():
        if attempt_events[0].phase != "started":
            raise FullAgentSummaryError(
                f"direct-VLA attempt {attempt_index} does not start with started"
            )
        if len(attempt_events) not in {1, 2}:
            raise FullAgentSummaryError(
                f"direct-VLA attempt {attempt_index} has invalid event count"
            )
        if len(attempt_events) == 1 and attempt_index != len(attempts):
            raise FullAgentSummaryError(
                "only the final direct-VLA attempt may lack a terminal event"
            )
        if len(attempt_events) == 2:
            started, terminal = attempt_events
            if terminal.phase == "started":
                raise FullAgentSummaryError(
                    f"direct-VLA attempt {attempt_index} has two starts"
                )
            if (
                terminal.step_index != started.step_index
                or terminal.tool_name != started.tool_name
            ):
                raise FullAgentSummaryError(
                    f"direct-VLA attempt {attempt_index} terminal identity changed"
                )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if completion_sha256 is not None and completion_sha256 != digest:
        raise FullAgentSummaryError(
            "completion direct-VLA journal hash disagrees with file bytes"
        )
    if completion_path is not None and Path(str(completion_path)).resolve() != path:
        raise FullAgentSummaryError(
            "completion direct-VLA journal path disagrees with file"
        )
    return tuple(records), path, digest


def _bind_direct_states_to_attempts(
    states_events: Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any]]],
    journal_events: Sequence[DirectVlaAttemptEvent],
) -> None:
    states_by_key: dict[
        tuple[int, str], tuple[Mapping[str, Any], Mapping[str, Any]]
    ] = {}
    for state_index, post_state, _pre_state in states_events:
        action = _physical_action(post_state)
        key = (state_index, str(action))
        if key in states_by_key:
            raise FullAgentSummaryError(
                f"states trace repeats direct VLA step/tool identity {key!r}"
            )
        states_by_key[key] = (post_state, _pre_state)

    attempts: dict[int, list[DirectVlaAttemptEvent]] = {}
    for event in journal_events:
        attempts.setdefault(event.attempt_index, []).append(event)
    consumed_states: set[tuple[int, str]] = set()
    for attempt_index in sorted(attempts):
        attempt_events = attempts[attempt_index]
        start = attempt_events[0]
        terminal = attempt_events[1] if len(attempt_events) == 2 else None
        key = (start.step_index, start.tool_name)
        if terminal is None or terminal.phase == "error":
            continue
        if key not in states_by_key:
            raise FullAgentSummaryError(
                "direct-VLA terminal evidence has no corresponding states entry: "
                f"attempt={attempt_index}, step={key[0]}, tool={key[1]}"
            )
        if key in consumed_states:
            raise FullAgentSummaryError(
                f"multiple direct-VLA attempts consume states entry {key!r}"
            )
        post_state, _pre_state = states_by_key[key]
        elapsed = _finite_nonnegative(
            post_state.get("elapsed_s"), f"states[{key[0]}].elapsed_s"
        )
        if elapsed != terminal.elapsed_s:
            raise FullAgentSummaryError(
                f"direct VLA state {key[0]} elapsed time disagrees with journal"
            )
        result = post_state.get("result")
        if not isinstance(result, Mapping):
            raise FullAgentSummaryError(
                f"direct VLA state {key[0]} has no structured result"
            )
        if terminal.phase == "completed" and result.get("error") is not None:
            raise FullAgentSummaryError(
                f"direct VLA state {key[0]} error disagrees with completion phase"
            )
        if terminal.phase in {"returned_error", "cancelled"} and (
            str(result.get("error")) != terminal.error
        ):
            raise FullAgentSummaryError(
                f"direct VLA state {key[0]} error disagrees with journal"
            )
        consumed_states.add(key)
    unbound_states = sorted(set(states_by_key).difference(consumed_states))
    if unbound_states:
        raise FullAgentSummaryError(
            f"direct VLA states have no terminal journal binding: {unbound_states}"
        )


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
        RuntimeProbeArtifact,
    )

    result: dict[tuple[str, int, int], str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        try:
            artifact = RuntimeProbeArtifact.model_validate(_strict_json(path))
        except Exception as exc:
            raise FullAgentSummaryError(
                f"runtime probe fails its complete schema: {path}: {exc}"
            ) from exc
        if not artifact.readiness_ok or not artifact.probe_calls_ok:
            raise FullAgentSummaryError(
                f"runtime probe is not ready/complete reset evidence: {path}"
            )
        report = artifact.report
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


def _system_analytic_time_s(
    states: Sequence[Mapping[str, Any]],
    records: Sequence[OutcomeRecord],
) -> float:
    """Return observed analytic-controller wall time across both hierarchies.

    Planner-mediated physical primitives are represented directly in
    ``states.json``.  Composite handoff entries include both local staging and
    VLA execution, so their outer elapsed values are deliberately excluded and
    replaced by the detailed governor staging telemetry.
    """
    outer_analytic_time = 0.0
    for index, entry in enumerate(states[1:], start=1):
        action = _physical_action(entry)
        if action in _DIRECT_VLA_ACTIONS or action in _COMPOSITE_VLA_ACTIONS:
            continue
        elapsed = _finite_nonnegative(
            entry.get("elapsed_s"), f"states[{index}].elapsed_s"
        )
        if elapsed is None:
            raise FullAgentSummaryError(
                f"analytic physical state {index} has no elapsed_s telemetry"
            )
        outer_analytic_time += elapsed
    detailed = [record.costs.analytic_time_s for record in records]
    if any(value is None for value in detailed):
        raise FullAgentSummaryError(
            "detailed governor outcome lacks analytic-time telemetry"
        )
    return outer_analytic_time + sum(float(value) for value in detailed)


def _load_run_reset_identity(
    trial: TrialManifest,
    plan: FullAgentChildPlan,
    *,
    runtime_attestation: Any,
    runtime_attestation_sha256: str,
    required: bool,
) -> tuple[str | None, Path, str | None]:
    """Load reset evidence captured by this exact child immediately after reset."""
    path = Path(trial.output_dir) / "reset_identity.json"
    if not path.is_file():
        if required:
            raise FullAgentSummaryError(
                f"full-agent run-local reset sidecar is missing: {path}"
            )
        return None, path, None
    payload = _strict_json(path)
    if not isinstance(payload, Mapping):
        raise FullAgentSummaryError(f"run-local reset sidecar is not an object: {path}")
    expected = {
        "schema_version": "rpent.research-reset-identity/v1",
        "trial_id": trial.trial_id,
        "manifest_id": plan.manifest_id,
        "plan_id": plan.plan_id,
        "source_revision": trial.source_revision,
        "suite": trial.task.suite,
        "task": trial.task.task,
        "seed": trial.task.seed,
        "max_episode_steps": trial.runtime.max_episode_steps,
        "observed_after_reset": True,
        "source": "live_env_runtime_probe",
        "probe_schema_version": "rpent.runtime-probe/v1",
        "probe_component": "libero_env",
        "runtime_attestation_id": runtime_attestation.attestation_id,
        "runtime_attestation_sha256": runtime_attestation_sha256,
    }
    expected_keys = set(expected).union({"reset_id"})
    if set(payload) != expected_keys:
        raise FullAgentSummaryError(
            "run-local reset sidecar fields disagree with the strict schema: "
            f"missing={sorted(expected_keys.difference(payload))}, "
            f"extra={sorted(set(payload).difference(expected_keys))}"
        )
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
    plan: FullAgentChildPlan,
    transcript_path: Path,
    states_path: Path,
    runtime_attestation: Any,
    runtime_attestation_path: Path,
    runtime_attestation_sha256: str,
    reset_id: str,
    reset_identity_path: Path,
    reset_identity_sha256: str,
) -> tuple[Mapping[str, Any], Path, str]:
    """Load research-only completion evidence, including planner exceptions."""
    path = Path(trial.output_dir) / "completion.json"
    payload = _strict_json(path)
    if not isinstance(payload, Mapping):
        raise FullAgentSummaryError(f"completion sidecar is not an object: {path}")
    expected_digest = hashlib.sha256(transcript_path.read_bytes()).hexdigest()
    expected = {
        "schema_version": "rpent.research-completion/v2",
        "trial_id": trial.trial_id,
        "manifest_id": plan.manifest_id,
        "plan_id": plan.plan_id,
        "source_revision": trial.source_revision,
        "transcript_path": str(transcript_path.resolve()),
        "transcript_sha256": expected_digest,
        "states_path": str(states_path.resolve()),
        "states_sha256": hashlib.sha256(states_path.read_bytes()).hexdigest(),
        "planner_backend": trial.planner.backend,
        "planner_model": trial.planner.model,
        "planner_base_url": trial.planner.base_url,
        "runtime_attestation_id": runtime_attestation.attestation_id,
        "runtime_attestation_path": str(runtime_attestation_path.resolve()),
        "runtime_attestation_sha256": runtime_attestation_sha256,
        "reset_id": reset_id,
        "reset_identity_path": str(reset_identity_path.resolve()),
        "reset_identity_sha256": reset_identity_sha256,
    }
    expected_keys = set(expected).union(
        {
            "status",
            "agent_error",
            "elapsed_s",
            "direct_vla_attempts_path",
            "direct_vla_attempts_sha256",
        }
    )
    if set(payload) != expected_keys:
        raise FullAgentSummaryError(
            "completion sidecar fields disagree with the strict schema: "
            f"missing={sorted(expected_keys.difference(payload))}, "
            f"extra={sorted(set(payload).difference(expected_keys))}"
        )
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise FullAgentSummaryError(
            "completion sidecar does not bind this trial/plan/runtime/reset/"
            f"transcript: {mismatches}"
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
        if not isinstance(error, str):
            raise FullAgentSummaryError("planner_error completion lacks agent_error")
    elif error is not None:
        raise FullAgentSummaryError("non-error completion unexpectedly has agent_error")
    if _finite_nonnegative(
        payload.get("elapsed_s"), "completion.elapsed_s"
    ) is None:
        raise FullAgentSummaryError("completion.elapsed_s is required")
    direct_path = payload.get("direct_vla_attempts_path")
    direct_sha256 = payload.get("direct_vla_attempts_sha256")
    if (direct_path is None) != (direct_sha256 is None):
        raise FullAgentSummaryError(
            "completion direct-VLA path/hash must be jointly present or absent"
        )
    if direct_path is not None:
        expected_direct_path = (
            Path(trial.output_dir) / "direct_vla_attempts.jsonl"
        ).resolve()
        if Path(str(direct_path)).resolve() != expected_direct_path:
            raise FullAgentSummaryError(
                "completion direct-VLA journal path is not output-local"
            )
        if not isinstance(direct_sha256, str) or not direct_sha256:
            raise FullAgentSummaryError(
                "completion direct-VLA journal hash is invalid"
            )
    return payload, path, hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runtime_binding(
    trial: TrialManifest,
    plan: FullAgentChildPlan,
    *,
    require_sink_manifest: bool = True,
) -> tuple[Any, Path]:
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
        "manifest_id": plan.manifest_id,
        "execution_plan_id": plan.plan_id,
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
    if not require_sink_manifest:
        return config, output_root
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
    trial: TrialManifest,
    *,
    plan: FullAgentChildPlan,
    runtime_config: Any,
    output_root: Path,
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
            or record.metadata.get("manifest_id") != plan.manifest_id
            or record.metadata.get("execution_plan_id") != plan.plan_id
            or record.metadata.get("handoff_configuration_id")
            != runtime_config.configuration_id
            or record.metadata.get("controller_configuration_id")
            != runtime_config.controller_configuration_id
        )
    ]
    if wrong:
        raise FullAgentSummaryError(
            f"detailed outcomes disagree with trial {trial.trial_id}: {wrong[:10]}"
        )
    return records


def _analysis_configuration_id(trial: TrialManifest) -> str:
    """Stable intention-to-treat identity for one manifest-assigned condition."""
    return stable_identifier(
        "assigned-controller-condition",
        {
            "condition": trial.condition.model_dump(
                mode="json", exclude_none=False
            ),
            "checkpoint_id": (
                trial.runtime.pi05_checkpoint_id
                or trial.condition.checkpoint_id
            ),
            "model_artifact_path": trial.model_artifact_path,
            "artifact_bindings": dict(sorted(trial.artifact_bindings.items())),
        },
    )


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
    return ControllerIdentity(
        method=trial.condition.method,
        implementation_version=(
            "rpent-original-harness/v1"
            if trial.condition.method == "original_harness"
            else "rpent-assigned-controller-not-instantiated/v1"
        ),
        checkpoint_id=(
            trial.runtime.pi05_checkpoint_id or trial.condition.checkpoint_id
        ),
        configuration_id=_analysis_configuration_id(trial),
    )


def _incomplete_failure_classification(
    terminal: TrialLifecycleEvent | None,
    direct_events: Sequence[DirectVlaAttemptEvent],
) -> tuple[TerminationReason, FailureMode, GovernorState]:
    if terminal is not None and terminal.event is TrialEventType.CANCELLED:
        return (
            TerminationReason.CANCELLED,
            FailureMode.CANCELLATION,
            GovernorState.CANCELLED,
        )
    lifecycle_text = " ".join(
        str(value)
        for value in (
            terminal.message if terminal is not None else None,
            terminal.details.get("error_type") if terminal is not None else None,
        )
        if value is not None
    ).lower()
    if "timeout" in lifecycle_text or "timed out" in lifecycle_text:
        return TerminationReason.TIMEOUT, FailureMode.TIMEOUT, GovernorState.ABORT
    if any(
        event.phase in {"returned_error", "error"}
        for event in direct_events
    ):
        return TerminationReason.VLA_FAILURE, FailureMode.VLA, GovernorState.VLA_FAILURE
    return TerminationReason.UNKNOWN, FailureMode.UNKNOWN, GovernorState.ABORT


def _summarize_incomplete_full_agent_trial(
    trial: TrialManifest,
    *,
    plan: FullAgentChildPlan,
    lifecycle_events: Sequence[TrialLifecycleEvent],
    terminal: TrialLifecycleEvent,
    probe_resets: Mapping[tuple[str, int, int], str] | None,
) -> OutcomeRecord:
    """Materialize a denominator record for a source-verified interrupted run."""
    attempt, attempt_path, attempt_sha256 = _load_attempt_identity(
        trial, plan, required=False
    )
    runtime_path = Path(trial.output_dir) / "runtime_identity.json"
    if runtime_path.is_file() and attempt is None:
        raise FullAgentSummaryError(
            "runtime attestation exists without wrapper attempt evidence"
        )
    runtime_attestation, runtime_path, runtime_sha256 = (
        _load_runtime_attestation_identity(trial, plan, required=False)
    )
    reset_path = Path(trial.output_dir) / "reset_identity.json"
    if reset_path.is_file() and runtime_attestation is None:
        raise FullAgentSummaryError(
            "reset identity exists without runtime attestation"
        )
    reset_id: str | None = None
    reset_sha256: str | None = None
    if runtime_attestation is not None and runtime_sha256 is not None:
        reset_id, reset_path, reset_sha256 = _load_run_reset_identity(
            trial,
            plan,
            runtime_attestation=runtime_attestation,
            runtime_attestation_sha256=runtime_sha256,
            required=False,
        )

    direct_path = Path(trial.output_dir) / "direct_vla_attempts.jsonl"
    if direct_path.is_file() and (
        runtime_attestation is None
        or runtime_sha256 is None
        or reset_id is None
        or reset_sha256 is None
    ):
        raise FullAgentSummaryError(
            "direct-VLA journal exists without runtime/reset identity evidence"
        )
    direct_events: tuple[DirectVlaAttemptEvent, ...] = ()
    direct_sha256: str | None = None
    if direct_path.is_file():
        direct_events, direct_path, direct_sha256 = _load_direct_vla_attempts(
            trial,
            plan,
            reset_id=str(reset_id),
            reset_identity_sha256=str(reset_sha256),
            runtime_attestation=runtime_attestation,
            runtime_attestation_sha256=str(runtime_sha256),
            completion=None,
        )
    starts = tuple(event for event in direct_events if event.phase == "started")
    terminal_direct = tuple(
        event for event in direct_events if event.phase != "started"
    )
    reason, failure_mode, governor_state = _incomplete_failure_classification(
        terminal, direct_events
    )

    runtime_config = None
    resolved_runtime_path = (
        Path(trial.output_dir) / "resolved_handoff_runtime.json"
    )
    if trial.condition.handoff_enabled and resolved_runtime_path.is_file():
        runtime_config, _unused_output_root = _load_runtime_binding(
            trial,
            plan,
            require_sink_manifest=False,
        )

    if attempt is None:
        incomplete_stage = "before_wrapper_attempt"
    elif runtime_attestation is None:
        incomplete_stage = "before_runtime_attestation"
    elif reset_id is None:
        incomplete_stage = "before_reset_identity"
    elif starts and len(terminal_direct) < len(starts):
        incomplete_stage = "during_direct_vla_tool"
    else:
        incomplete_stage = "after_reset_before_completion"

    probe_key = (trial.task.suite, trial.task.task, trial.task.seed)
    probed_reset = (probe_resets or {}).get(probe_key)
    reset_candidates = {
        str(value)
        for value in (trial.task.reset_id, reset_id, probed_reset)
        if value is not None
    }
    if len(reset_candidates) > 1:
        raise FullAgentSummaryError(
            "configured/probed/run-local reset identity contradiction in "
            f"incomplete trial: {sorted(reset_candidates)}"
        )

    identity = TrialIdentity(
        run_id=trial.experiment_id,
        episode_id=trial.trial_id,
        trial_id=trial.trial_id,
        invocation_id=f"{trial.trial_id}/episode-summary",
        suite=trial.task.suite,
        task_id=trial.task.task,
        seed=trial.task.seed,
        reset_id=reset_id,
        repeat_index=trial.repeat_index,
    )
    def unavailable(detail: str) -> OutcomeSignal:
        return unavailable_signal(
            f"incomplete full-agent execution: {detail}"
        )
    known_paths = tuple(
        path
        for path in (
            attempt_path,
            runtime_path,
            reset_path,
            direct_path,
            Path(trial.output_dir) / "resolved_handoff_runtime.json",
            Path(trial.output_dir) / "states.json",
        )
        if path.is_file()
    )
    return OutcomeRecord(
        record_id=outcome_record_id(identity),
        identity=identity,
        skill=SkillIdentity(
            name=trial.task.skill_name,
            semantic_target=trial.task.target_description,
            learned_controller="pi0.5",
        ),
        controller=_resolved_controller(trial, (), runtime_config),
        pre_handoff_state=None,
        handoff_occurred=False,
        decision_trace=(),
        labels=OutcomeLabels(
            primitive_success=unavailable("primitive result unavailable"),
            skill_success=unavailable("skill result unavailable"),
            task_success=unavailable("official termination unavailable"),
            episode_truncated=unavailable("truncation unavailable"),
            llm_finish=unavailable("planner finish unavailable"),
        ),
        costs=CostRecord(
            analytic_steps=None,
            analytic_distance_m=None,
            analytic_time_s=None,
            vla_invocations=None,
            vla_chunks=None,
            vla_env_actions=None,
            vla_time_s=None,
            total_env_actions=None,
            total_elapsed_s=None,
            planner_time_s=None,
            llm_turns=None,
            input_tokens=None,
            output_tokens=None,
            system_analytic_time_s=None,
            intervention_count=0,
            recovery_retry_cost=(
                float(max(0, len(starts) - 1)) if starts else 0.0
            ),
        ),
        timing=TimingRecord(
            started_monotonic_s=0.0,
            ended_monotonic_s=0.0,
            record_time_s=None,
        ),
        termination=TerminationRecord(
            reason=reason,
            failure_mode=failure_mode,
            final_governor_state=governor_state,
            episode_terminated=False,
            episode_truncated=False,
            message="full-agent execution ended before a complete episode record",
        ),
        source_revision=trial.source_revision,
        metadata={
            "data_status": "observed",
            "incomplete_execution": True,
            "denominator_eligible": True,
            "system_attempt_success": False,
            "incomplete_stage": incomplete_stage,
            "record_scope": "full_agent_episode",
            "execution_layer": trial.execution_layer.value,
            "condition": trial.condition.name,
            "representation": trial.condition.feature_set.value,
            "evidence_mode": trial.condition.evidence.value,
            "decision_mode": trial.condition.decision.value,
            "uncertainty_mode": trial.condition.uncertainty.value,
            "hierarchy_mode": trial.condition.hierarchy.value,
            "manifest_id": plan.manifest_id,
            "execution_plan_id": plan.plan_id,
            "analysis_configuration_id": _analysis_configuration_id(trial),
            "assigned_controller_runtime_config_resolved": (
                runtime_config is not None
            ),
            "attempt_path": str(attempt_path.resolve()) if attempt is not None else None,
            "attempt_sha256": attempt_sha256,
            "runtime_attestation_id": (
                runtime_attestation.attestation_id
                if runtime_attestation is not None
                else None
            ),
            "runtime_attestation_path": (
                str(runtime_path.resolve()) if runtime_attestation is not None else None
            ),
            "runtime_attestation_sha256": runtime_sha256,
            "reset_identity_path": (
                str(reset_path.resolve()) if reset_id is not None else None
            ),
            "reset_identity_sha256": reset_sha256,
            "detached_probe_reset_id": probed_reset,
            "direct_vla_attempts_path": (
                str(direct_path.resolve()) if direct_events else None
            ),
            "direct_vla_attempts_sha256": direct_sha256,
            "direct_vla_tool_attempts": len(starts),
            "direct_vla_tool_calls": len(starts),
            "direct_vla_terminal_events": len(terminal_direct),
            "direct_vla_attempt_unit": "planner_visible_vla_tool_invocation",
            "protocol_adherent": (
                False if trial.condition.handoff_enabled and starts else None
            ),
            "lifecycle_event_ids": [event.event_id for event in lifecycle_events],
            "lifecycle_terminal": terminal.event.value if terminal is not None else None,
            "lifecycle_message": terminal.message if terminal is not None else None,
            "observed_incomplete_artifacts": [
                str(path.resolve()) for path in known_paths
            ],
            "tool_calls": None,
        },
    )


def summarize_full_agent_trial(
    trial: TrialManifest,
    *,
    plan: FullAgentChildPlan,
    lifecycle_events: Sequence[TrialLifecycleEvent],
    probe_resets: Mapping[tuple[str, int, int], str] | None = None,
) -> OutcomeRecord:
    """Create one episode-scoped record from immutable full-agent artifacts."""
    if trial.execution_layer is not ExecutionLayer.FULL_AGENT:
        raise FullAgentSummaryError(f"trial is not full_agent: {trial.trial_id}")
    _validate_plan_binding(trial, plan)
    relevant_lifecycle, terminal_lifecycle = _validate_lifecycle_binding(
        trial,
        plan,
        lifecycle_events,
    )
    if terminal_lifecycle is None:
        raise FullAgentSummaryError(
            "full-agent lifecycle has no terminal event; the trial may still be running"
        )
    output_dir = Path(trial.output_dir)
    completion_sidecar = output_dir / "completion.json"
    if not completion_sidecar.is_file():
        if terminal_lifecycle.event is TrialEventType.COMPLETED:
            raise FullAgentSummaryError(
                "completed lifecycle is missing its completion sidecar"
            )
        return _summarize_incomplete_full_agent_trial(
            trial,
            plan=plan,
            lifecycle_events=relevant_lifecycle,
            terminal=terminal_lifecycle,
            probe_resets=probe_resets,
        )

    attempt, attempt_path, attempt_sha256 = _load_attempt_identity(
        trial,
        plan,
        required=True,
    )
    if attempt is None or attempt_sha256 is None:
        raise FullAgentSummaryError("required full-agent attempt evidence vanished")
    runtime_attestation, runtime_path, runtime_sha256 = (
        _load_runtime_attestation_identity(trial, plan, required=True)
    )
    if runtime_attestation is None or runtime_sha256 is None:
        raise FullAgentSummaryError("required runtime attestation vanished")
    run_reset, reset_sidecar_path, reset_sidecar_sha256 = (
        _load_run_reset_identity(
            trial,
            plan,
            runtime_attestation=runtime_attestation,
            runtime_attestation_sha256=runtime_sha256,
            required=True,
        )
    )
    if run_reset is None or reset_sidecar_sha256 is None:
        raise FullAgentSummaryError("required reset identity vanished")
    transcript_path = _single_artifact(
        output_dir, "transcript_*.json", "RPent transcript"
    )
    states_path = output_dir / "states.json"
    if not states_path.is_file():
        raise FullAgentSummaryError(
            f"completion sidecar has no states trace: {states_path}"
        )
    transcript = _strict_json(transcript_path)
    if not isinstance(transcript, Mapping):
        raise FullAgentSummaryError("RPent transcript must be a JSON object")
    completion, completion_path, completion_sha256 = _load_completion_identity(
        trial,
        plan=plan,
        transcript_path=transcript_path,
        states_path=states_path,
        runtime_attestation=runtime_attestation,
        runtime_attestation_path=runtime_path,
        runtime_attestation_sha256=runtime_sha256,
        reset_id=run_reset,
        reset_identity_path=reset_sidecar_path,
        reset_identity_sha256=reset_sidecar_sha256,
    )
    completion_status = completion["status"]
    expected_terminal = (
        TrialEventType.FAILED
        if completion_status == "planner_error"
        else TrialEventType.COMPLETED
    )
    if terminal_lifecycle.event is not expected_terminal:
        raise FullAgentSummaryError(
            "completion status disagrees with lifecycle terminal: "
            f"status={completion_status!r}, terminal={terminal_lifecycle.event.value!r}"
        )
    expected_returncode = 1 if completion_status == "planner_error" else 0
    lifecycle_returncode = terminal_lifecycle.details.get("returncode")
    if (
        lifecycle_returncode is not None
        and lifecycle_returncode != expected_returncode
    ):
        raise FullAgentSummaryError(
            "completion status disagrees with lifecycle return code"
        )
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
        if state_index > 0 and not isinstance(entry.get("command"), Mapping):
            raise FullAgentSummaryError(
                f"states[{state_index}] lacks a physical command object"
            )
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
        runtime_config, detailed_output_root = _load_runtime_binding(
            trial,
            plan,
        )
    records = (
        _load_detailed_outcomes(
            trial,
            plan=plan,
            runtime_config=runtime_config,
            output_root=detailed_output_root,
        )
        if detailed_output_root is not None
        else ()
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
    if completion.get("elapsed_s") != transcript.get("elapsed_s"):
        raise FullAgentSummaryError(
            "completion elapsed_s disagrees with transcript.elapsed_s"
        )
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
    if isinstance(finish, Mapping) and finish_status is None:
        raise FullAgentSummaryError(
            "non-null transcript finish must contain a status"
        )
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

    journal_events, direct_journal_path, direct_journal_sha256 = (
        _load_direct_vla_attempts(
            trial,
            plan,
            reset_id=run_reset,
            reset_identity_sha256=reset_sidecar_sha256,
            runtime_attestation=runtime_attestation,
            runtime_attestation_sha256=runtime_sha256,
            completion=completion,
        )
    )
    journal_starts = tuple(
        event for event in journal_events if event.phase == "started"
    )
    journal_terminals = tuple(
        event for event in journal_events if event.phase != "started"
    )
    direct_events = _direct_vla_events(states)
    _bind_direct_states_to_attempts(direct_events, journal_events)
    (
        direct_state_calls,
        direct_inference_invocations,
        direct_chunks,
        _state_direct_vla_time,
    ) = _direct_vla_costs(direct_events)
    direct_tool_calls = len(journal_starts)
    if direct_state_calls != direct_tool_calls:
        # A started call with no post-state may have issued zero or more model
        # inferences; completed state telemetry cannot close that uncertainty.
        direct_inference_invocations = None
        direct_chunks = None
    direct_vla_time = (
        sum(float(event.elapsed_s) for event in journal_terminals)
        if len(journal_terminals) == direct_tool_calls
        else None
    )
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

    def required_detailed_sum(field: str) -> float:
        values = [getattr(record.costs, field) for record in records]
        if any(value is None for value in values):
            raise FullAgentSummaryError(
                f"detailed governor outcome lacks required {field} telemetry"
            )
        return sum(float(value) for value in values)

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
            "attempt_index": event.attempt_index,
            "state_index": event.step_index,
            "action": event.tool_name,
            "phase": event.phase,
            "error_type": event.error_type,
            "error": event.error,
        }
        for event in journal_terminals
        if event.phase in {"returned_error", "cancelled", "error"}
    ]
    episode_failure = (
        FailureMode.TRUNCATION
        if truncated
        else (FailureMode.NONE if terminated else FailureMode.UNKNOWN)
    )
    episode_reason = (
        TerminationReason.UNKNOWN
        if episode_failure is FailureMode.UNKNOWN
        else TerminationReason.COMPLETED
    )

    controller = _resolved_controller(trial, records, runtime_config)
    identity = TrialIdentity(
        run_id=trial.experiment_id,
        episode_id=trial.trial_id,
        trial_id=trial.trial_id,
        invocation_id=f"{trial.trial_id}/episode-summary",
        suite=trial.task.suite,
        task_id=trial.task.task,
        seed=trial.task.seed,
        reset_id=reset_id,
        repeat_index=trial.repeat_index,
    )
    system_analytic_time = _system_analytic_time_s(states, records)
    learned_controller_tool_attempts = direct_tool_calls + len(composite_indices)
    # This benchmark configures one semantic learned skill per episode.  The
    # observable retry cost is therefore the number of additional learned-
    # controller tool attempts beyond the first; it does not infer planner
    # intent or count failed analytic staging as a successful VLA invocation.
    recovery_retry_cost = float(max(0, learned_controller_tool_attempts - 1))
    return OutcomeRecord(
        record_id=outcome_record_id(identity),
        identity=identity,
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
            analytic_steps=int(required_detailed_sum("analytic_steps")),
            analytic_distance_m=required_detailed_sum("analytic_distance_m"),
            analytic_time_s=required_detailed_sum("analytic_time_s"),
            vla_invocations=combined_inference_invocations,
            vla_chunks=combined_chunks,
            # Direct Pi0 states expose chunk counts but not a source-verified
            # number of executed actions, so a mixed/direct episode is unknown.
            vla_env_actions=(
                detailed_vla_actions if direct_tool_calls == 0 else None
            ),
            vla_time_s=(
                required_detailed_sum("vla_time_s")
                + direct_vla_time
                if direct_vla_time is not None
                else None
            ),
            # The outer states trace does not expose source-verified executed
            # env-step counts for all planner-selected analytic primitives.
            total_env_actions=None,
            total_elapsed_s=elapsed,
            planner_time_s=planner_time,
            llm_turns=llm_turns,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            system_analytic_time_s=system_analytic_time,
            intervention_count=0,
            recovery_retry_cost=recovery_retry_cost,
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
            "incomplete_execution": False,
            "denominator_eligible": True,
            "system_attempt_success": terminated,
            "record_scope": "full_agent_episode",
            "execution_layer": trial.execution_layer.value,
            "condition": trial.condition.name,
            "representation": trial.condition.feature_set.value,
            "evidence_mode": trial.condition.evidence.value,
            "decision_mode": trial.condition.decision.value,
            "uncertainty_mode": trial.condition.uncertainty.value,
            "hierarchy_mode": trial.condition.hierarchy.value,
            "manifest_id": plan.manifest_id,
            "execution_plan_id": plan.plan_id,
            "analysis_configuration_id": _analysis_configuration_id(trial),
            "attempt_path": str(attempt_path.resolve()),
            "attempt_sha256": attempt_sha256,
            "runtime_attestation_id": runtime_attestation.attestation_id,
            "runtime_attestation_path": str(runtime_path.resolve()),
            "runtime_attestation_sha256": runtime_sha256,
            "detailed_outcome_record_ids": [record.record_id for record in records],
            "direct_vla_tool_calls": direct_tool_calls,
            "direct_vla_tool_attempts": direct_tool_calls,
            "direct_vla_state_records": direct_state_calls,
            "direct_vla_terminal_events": len(journal_terminals),
            "direct_vla_attempt_unit": "planner_visible_vla_tool_invocation",
            "direct_vla_inference_invocations": direct_inference_invocations,
            "direct_vla_attempts_path": (
                str(direct_journal_path.resolve()) if journal_events else None
            ),
            "direct_vla_attempts_sha256": direct_journal_sha256,
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
            "lifecycle_event_ids": [
                event.event_id for event in relevant_lifecycle
            ],
            "lifecycle_terminal": terminal_lifecycle.event.value,
            "episode_relative_timing_origin": True,
            "analytic_cost_scope": "local_governor_staging_only",
            "total_env_actions_unavailable_reason": (
                "full states trace lacks executed-step counts for every tool"
            ),
            "planner_time_source": planner_time_source,
            "system_analytic_time_definition": (
                "sum of planner-mediated non-VLA physical-tool elapsed_s plus "
                "local-governor staging time; composite outer elapsed excluded"
            ),
            "intervention_count_definition": (
                "automated research episode; no human intervention interface"
            ),
            "recovery_retry_cost_definition": (
                "additional direct/composite learned-controller tool attempts "
                "beyond the first in the configured single-skill episode"
            ),
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
