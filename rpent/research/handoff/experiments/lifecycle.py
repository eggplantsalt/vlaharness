"""Append-only trial lifecycle journal and deterministic resume selection."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.experiments.config import stable_identifier
from rpent.research.handoff.experiments.manifest import (
    ExperimentManifest,
    TrialManifest,
)
from rpent.research.handoff.types import HandoffRecord

LIFECYCLE_SCHEMA_VERSION = "rpent.handoff-trial-lifecycle/v1"


class TrialEventType(str, Enum):
    PLANNED = "planned"
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

    @property
    def terminal(self) -> bool:
        return self in {
            self.COMPLETED,
            self.FAILED,
            self.CANCELLED,
            self.SKIPPED,
        }


class TrialLifecycleEvent(HandoffRecord):
    """One durable lifecycle transition for a resolved trial."""

    schema_version: Literal[LIFECYCLE_SCHEMA_VERSION] = LIFECYCLE_SCHEMA_VERSION
    event_id: str
    trial_id: str
    sequence: int = Field(ge=0)
    attempt: int = Field(ge=1)
    event: TrialEventType
    timestamp_utc: str
    message: str | None = None
    artifact_path: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id", "trial_id", "timestamp_utc")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("message", "artifact_path")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is not None and not value:
            raise ValueError(f"{info.field_name} cannot be empty")
        return value

    @field_validator("details")
    @classmethod
    def validate_details(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("details must contain finite JSON values only") from exc
        return value

    @field_validator("timestamp_utc")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp_utc must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ValueError("timestamp_utc must include a timezone")
        return value


class TrialResumeState(HandoffRecord):
    """Derived current lifecycle state for one manifest trial."""

    trial_id: str
    latest_event: TrialEventType | None = None
    latest_attempt: int = 0
    latest_sequence: int = -1
    should_run: bool
    next_attempt: int = Field(ge=1)
    reason: str

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        if self.next_attempt < max(1, self.latest_attempt):
            raise ValueError("next attempt cannot precede latest attempt")
        return self


def utc_now() -> str:
    """Return a Windows-safe, timezone-explicit UTC event timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _strict_json_object(raw: str, *, source: Path, line_number: int) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite constant {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"invalid lifecycle JSON in {source} at line {line_number}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(
            f"lifecycle record must be an object in {source} at line {line_number}"
        )
    return value


def read_lifecycle_events(
    path: str | os.PathLike[str],
    *,
    missing_ok: bool = True,
) -> tuple[TrialLifecycleEvent, ...]:
    """Read a journal strictly; a malformed final line is never ignored."""
    source = Path(path)
    if not source.exists():
        if missing_ok:
            return ()
        raise FileNotFoundError(source)
    events: list[TrialLifecycleEvent] = []
    seen_event_ids: set[str] = set()
    last_sequence: dict[str, int] = {}
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(
                    f"blank lifecycle line in {source} at line {line_number}"
                )
            value = _strict_json_object(
                line,
                source=source,
                line_number=line_number,
            )
            try:
                event = TrialLifecycleEvent.model_validate(value)
            except Exception as exc:
                raise ValueError(
                    f"invalid lifecycle event in {source} at line {line_number}: {exc}"
                ) from exc
            if event.event_id in seen_event_ids:
                raise ValueError(f"duplicate lifecycle event ID: {event.event_id}")
            expected = last_sequence.get(event.trial_id, -1) + 1
            if event.sequence != expected:
                raise ValueError(
                    f"non-contiguous lifecycle sequence for {event.trial_id}: "
                    f"expected {expected}, got {event.sequence}"
                )
            last_sequence[event.trial_id] = event.sequence
            seen_event_ids.add(event.event_id)
            events.append(event)
    return tuple(events)


def _validate_transition(
    previous: TrialLifecycleEvent | None,
    *,
    event: TrialEventType,
    attempt: int,
) -> None:
    if previous is None:
        if event not in {TrialEventType.PLANNED, TrialEventType.STARTED}:
            raise ValueError("first lifecycle event must be planned or started")
        if attempt != 1:
            raise ValueError("first lifecycle event must use attempt 1")
        return

    if previous.event.terminal:
        if event is not TrialEventType.STARTED:
            raise ValueError("a terminal trial can only transition to a retry start")
        if attempt != previous.attempt + 1:
            raise ValueError("retry start must increment the attempt by one")
        return

    if (
        event is TrialEventType.STARTED
        and previous.event in {TrialEventType.STARTED, TrialEventType.PROGRESS}
        and attempt == previous.attempt + 1
    ):
        # The prior process disappeared without being able to append a terminal
        # event.  Preserve that incomplete attempt and start a new one rather
        # than rewriting history.
        return

    if attempt != previous.attempt:
        raise ValueError("non-terminal transition must remain in the same attempt")
    allowed = {
        TrialEventType.PLANNED: {
            TrialEventType.STARTED,
            TrialEventType.SKIPPED,
            TrialEventType.CANCELLED,
            TrialEventType.FAILED,
        },
        TrialEventType.STARTED: {
            TrialEventType.PROGRESS,
            TrialEventType.COMPLETED,
            TrialEventType.FAILED,
            TrialEventType.CANCELLED,
        },
        TrialEventType.PROGRESS: {
            TrialEventType.PROGRESS,
            TrialEventType.COMPLETED,
            TrialEventType.FAILED,
            TrialEventType.CANCELLED,
        },
    }
    if event not in allowed.get(previous.event, set()):
        raise ValueError(f"invalid lifecycle transition {previous.event} -> {event}")


class LifecycleJournal:
    """Thread-safe append-only JSONL journal for one experiment process.

    Multi-process launchers should write one shard per worker and merge only
    after validation; this class intentionally does not pretend a portable file
    lock exists across all supported platforms.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        allowed_trial_ids: set[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.allowed_trial_ids = allowed_trial_ids
        self._lock = threading.Lock()
        self._events_cache: list[TrialLifecycleEvent] | None = None

    def read(self) -> tuple[TrialLifecycleEvent, ...]:
        if self._events_cache is None:
            self._events_cache = list(read_lifecycle_events(self.path))
        return tuple(self._events_cache)

    def refresh(self) -> tuple[TrialLifecycleEvent, ...]:
        """Reload a journal shard after an explicitly coordinated external write."""
        with self._lock:
            self._events_cache = list(read_lifecycle_events(self.path))
            return tuple(self._events_cache)

    def append(
        self,
        trial_id: str,
        event: TrialEventType,
        *,
        attempt: int | None = None,
        timestamp_utc: str | None = None,
        message: str | None = None,
        artifact_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> TrialLifecycleEvent:
        """Validate, append, flush, and fsync one lifecycle transition."""
        if self.allowed_trial_ids is not None and trial_id not in self.allowed_trial_ids:
            raise ValueError(f"trial is not present in the manifest: {trial_id}")
        with self._lock:
            events = self.read()
            previous = next(
                (item for item in reversed(events) if item.trial_id == trial_id),
                None,
            )
            resolved_attempt = attempt
            if resolved_attempt is None:
                resolved_attempt = previous.attempt if previous is not None else 1
                if previous is not None and event is TrialEventType.STARTED:
                    if previous.event.terminal or previous.event in {
                        TrialEventType.STARTED,
                        TrialEventType.PROGRESS,
                    }:
                        resolved_attempt += 1
            _validate_transition(
                previous,
                event=event,
                attempt=resolved_attempt,
            )
            sequence = previous.sequence + 1 if previous is not None else 0
            timestamp = timestamp_utc or utc_now()
            event_payload = {
                "trial_id": trial_id,
                "sequence": sequence,
                "attempt": resolved_attempt,
                "event": event.value,
                "timestamp_utc": timestamp,
                "message": message,
                "artifact_path": artifact_path,
                "details": details or {},
            }
            record = TrialLifecycleEvent(
                event_id=stable_identifier("event", event_payload),
                **event_payload,
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(record.canonical_json())
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            assert self._events_cache is not None
            self._events_cache.append(record)
            return record


def derive_resume_states(
    manifest: ExperimentManifest,
    events: tuple[TrialLifecycleEvent, ...] | list[TrialLifecycleEvent],
    *,
    retry_failed: bool = False,
    retry_cancelled: bool = True,
) -> tuple[TrialResumeState, ...]:
    """Derive which manifest trials should run after an interruption."""
    manifest_ids = {trial.trial_id for trial in manifest.trials}
    unknown = sorted({event.trial_id for event in events}.difference(manifest_ids))
    if unknown:
        raise ValueError(f"lifecycle journal contains unknown trial IDs: {unknown}")
    latest: dict[str, TrialLifecycleEvent] = {}
    for event in events:
        previous = latest.get(event.trial_id)
        if previous is not None and event.sequence <= previous.sequence:
            raise ValueError("events are not sequence ordered")
        latest[event.trial_id] = event

    result: list[TrialResumeState] = []
    for trial in manifest.trials:
        event = latest.get(trial.trial_id)
        if event is None:
            result.append(
                TrialResumeState(
                    trial_id=trial.trial_id,
                    should_run=True,
                    next_attempt=1,
                    reason="not_started",
                )
            )
            continue
        if event.event in {TrialEventType.COMPLETED, TrialEventType.SKIPPED}:
            should_run = False
            reason = event.event.value
            next_attempt = event.attempt
        elif event.event is TrialEventType.FAILED:
            should_run = retry_failed
            reason = "retry_failed" if retry_failed else "failed_not_retryable"
            next_attempt = event.attempt + 1
        elif event.event is TrialEventType.CANCELLED:
            should_run = retry_cancelled
            reason = "retry_cancelled" if retry_cancelled else "cancelled_not_retryable"
            next_attempt = event.attempt + 1
        else:
            should_run = True
            reason = "interrupted_incomplete_attempt"
            next_attempt = event.attempt + 1
        result.append(
            TrialResumeState(
                trial_id=trial.trial_id,
                latest_event=event.event,
                latest_attempt=event.attempt,
                latest_sequence=event.sequence,
                should_run=should_run,
                next_attempt=next_attempt,
                reason=reason,
            )
        )
    return tuple(result)


def resumable_trials(
    manifest: ExperimentManifest,
    events: tuple[TrialLifecycleEvent, ...] | list[TrialLifecycleEvent],
    *,
    retry_failed: bool = False,
    retry_cancelled: bool = True,
) -> tuple[TrialManifest, ...]:
    """Return manifest-ordered trials selected by resume policy."""
    states = derive_resume_states(
        manifest,
        events,
        retry_failed=retry_failed,
        retry_cancelled=retry_cancelled,
    )
    selected = {state.trial_id for state in states if state.should_run}
    return tuple(trial for trial in manifest.trials if trial.trial_id in selected)


__all__ = [
    "LifecycleJournal",
    "TrialEventType",
    "TrialLifecycleEvent",
    "TrialResumeState",
    "derive_resume_states",
    "read_lifecycle_events",
    "resumable_trials",
    "utc_now",
]
