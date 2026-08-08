"""Checksummed append-only outcome datasets and strict training extraction."""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Self, Sequence

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.types import (
    FailureMode,
    HandoffDecision,
    HandoffRecord,
    HandoffState,
    OutcomeRecord,
    TerminationReason,
)

DATASET_ENVELOPE_SCHEMA_VERSION = "rpent.handoff-outcome-jsonl/v1"
DECISION_ENVELOPE_SCHEMA_VERSION = "rpent.handoff-decision-jsonl/v1"
DATASET_FINGERPRINT_VERSION = "rpent.handoff-dataset-fingerprint/v1"


class DatasetError(ValueError):
    """Base class for dataset validation errors."""


class DatasetCorruptionError(DatasetError):
    """A complete JSONL entry is malformed, reordered, or inconsistent."""


class DatasetConflictError(DatasetError):
    """A stable record ID was reused for different content."""


def outcome_checksum(record: OutcomeRecord) -> str:
    """Return the checksum stored alongside an authoritative outcome record."""
    return hashlib.sha256(record.canonical_json().encode("utf-8")).hexdigest()


class OutcomeEnvelope(HandoffRecord):
    """One independently checksummed JSONL entry."""

    schema_version: Literal[DATASET_ENVELOPE_SCHEMA_VERSION] = (
        DATASET_ENVELOPE_SCHEMA_VERSION
    )
    record_type: Literal["outcome"] = "outcome"
    sequence: int = Field(ge=0)
    record_id: str
    payload_sha256: str
    payload: OutcomeRecord

    @field_validator("record_id", "payload_sha256")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        if self.record_id != self.payload.record_id:
            raise ValueError("envelope and payload record IDs disagree")
        expected = outcome_checksum(self.payload)
        if self.payload_sha256 != expected:
            raise ValueError("outcome payload checksum mismatch")
        return self


def decision_checksum(record: HandoffDecision) -> str:
    """Return the checksum stored alongside a decision trace record."""
    return hashlib.sha256(record.canonical_json().encode("utf-8")).hexdigest()


class DecisionEnvelope(HandoffRecord):
    """One independently checksummed handoff-decision JSONL entry."""

    schema_version: Literal[DECISION_ENVELOPE_SCHEMA_VERSION] = (
        DECISION_ENVELOPE_SCHEMA_VERSION
    )
    record_type: Literal["decision"] = "decision"
    sequence: int = Field(ge=0)
    record_id: str
    payload_sha256: str
    payload: HandoffDecision

    @field_validator("record_id", "payload_sha256")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        if self.record_id != self.payload.decision_id:
            raise ValueError("envelope record ID and payload decision ID disagree")
        expected = decision_checksum(self.payload)
        if self.payload_sha256 != expected:
            raise ValueError("decision payload checksum mismatch")
        return self


@dataclass(frozen=True, slots=True)
class DatasetScan:
    envelopes: tuple[OutcomeEnvelope, ...]
    partial_tail_bytes: int = 0
    valid_bytes: int = 0
    needs_trailing_newline: bool = False


@dataclass(frozen=True, slots=True)
class DecisionDatasetScan:
    envelopes: tuple[DecisionEnvelope, ...]
    partial_tail_bytes: int = 0
    valid_bytes: int = 0
    needs_trailing_newline: bool = False


@dataclass(frozen=True, slots=True)
class OutcomeDataset:
    """Strict in-memory view of one outcome JSONL shard."""

    records: tuple[OutcomeRecord, ...]
    fingerprint: str
    source_path: Path | None = None
    ignored_partial_tail_bytes: int = 0

    @classmethod
    def from_records(cls, records: Sequence[OutcomeRecord]) -> OutcomeDataset:
        values = tuple(records)
        _validate_unique_record_ids(values)
        return cls(records=values, fingerprint=dataset_fingerprint(values))

    @classmethod
    def from_jsonl(
        cls,
        path: str | os.PathLike[str],
        *,
        allow_partial_final_line: bool = True,
    ) -> OutcomeDataset:
        source = Path(path)
        scan = scan_outcome_jsonl(
            source, allow_partial_final_line=allow_partial_final_line
        )
        records = tuple(envelope.payload for envelope in scan.envelopes)
        return cls(
            records=records,
            fingerprint=dataset_fingerprint(records),
            source_path=source,
            ignored_partial_tail_bytes=scan.partial_tail_bytes,
        )


class AppendStatus(str, Enum):
    APPENDED = "appended"
    ALREADY_PRESENT = "already_present"


@dataclass(frozen=True, slots=True)
class AppendResult:
    status: AppendStatus
    sequence: int
    record_id: str


class OutcomeJsonlWriter:
    """Single-writer append/resume logger with torn-tail recovery.

    Rollout workers should write separate shards.  Within one process this
    object serializes appends with a lock.  On resume, an incomplete final line
    is truncated to the last valid newline; malformed complete lines and
    checksum conflicts always fail.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        fsync: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._fsync = fsync
        self._lock = threading.Lock()
        scan = scan_outcome_jsonl(self.path, allow_partial_final_line=True)
        if scan.partial_tail_bytes:
            with self.path.open("r+b") as stream:
                stream.truncate(scan.valid_bytes)
                stream.flush()
                if fsync:
                    os.fsync(stream.fileno())
        self._needs_trailing_newline = scan.needs_trailing_newline
        self._checksums: dict[str, str] = {}
        self._sequences: dict[str, int] = {}
        for envelope in scan.envelopes:
            previous = self._checksums.get(envelope.record_id)
            if previous is not None:
                raise DatasetCorruptionError(
                    f"duplicate record ID in dataset: {envelope.record_id!r}"
                )
            self._checksums[envelope.record_id] = envelope.payload_sha256
            self._sequences[envelope.record_id] = envelope.sequence
        self._next_sequence = len(scan.envelopes)

    def append(self, record: OutcomeRecord) -> AppendResult:
        """Append once, no-op on an identical retry, reject conflicting retry."""
        checksum = outcome_checksum(record)
        with self._lock:
            previous = self._checksums.get(record.record_id)
            if previous is not None:
                if previous != checksum:
                    raise DatasetConflictError(
                        f"record ID {record.record_id!r} already has different content"
                    )
                return AppendResult(
                    status=AppendStatus.ALREADY_PRESENT,
                    sequence=self._sequences[record.record_id],
                    record_id=record.record_id,
                )

            sequence = self._next_sequence
            envelope = OutcomeEnvelope(
                sequence=sequence,
                record_id=record.record_id,
                payload_sha256=checksum,
                payload=record,
            )
            prefix = b"\n" if self._needs_trailing_newline else b""
            encoded = prefix + envelope.canonical_json().encode("utf-8") + b"\n"
            descriptor = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0),
            )
            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("append-only dataset write made no progress")
                    view = view[written:]
                if self._fsync:
                    os.fsync(descriptor)
            finally:
                os.close(descriptor)

            self._needs_trailing_newline = False
            self._checksums[record.record_id] = checksum
            self._sequences[record.record_id] = sequence
            self._next_sequence += 1
            return AppendResult(
                status=AppendStatus.APPENDED,
                sequence=sequence,
                record_id=record.record_id,
            )


class DecisionJsonlWriter:
    """Decision counterpart to :class:`OutcomeJsonlWriter`."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        fsync: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._fsync = fsync
        self._lock = threading.Lock()
        scan = scan_decision_jsonl(self.path, allow_partial_final_line=True)
        if scan.partial_tail_bytes:
            with self.path.open("r+b") as stream:
                stream.truncate(scan.valid_bytes)
                stream.flush()
                if fsync:
                    os.fsync(stream.fileno())
        self._needs_trailing_newline = scan.needs_trailing_newline
        self._checksums: dict[str, str] = {}
        self._sequences: dict[str, int] = {}
        for envelope in scan.envelopes:
            previous = self._checksums.get(envelope.record_id)
            if previous is not None:
                raise DatasetCorruptionError(
                    f"duplicate decision ID in dataset: {envelope.record_id!r}"
                )
            self._checksums[envelope.record_id] = envelope.payload_sha256
            self._sequences[envelope.record_id] = envelope.sequence
        self._next_sequence = len(scan.envelopes)

    def append(self, record: HandoffDecision) -> AppendResult:
        checksum = decision_checksum(record)
        with self._lock:
            previous = self._checksums.get(record.decision_id)
            if previous is not None:
                if previous != checksum:
                    raise DatasetConflictError(
                        f"decision ID {record.decision_id!r} already has different content"
                    )
                return AppendResult(
                    status=AppendStatus.ALREADY_PRESENT,
                    sequence=self._sequences[record.decision_id],
                    record_id=record.decision_id,
                )

            sequence = self._next_sequence
            envelope = DecisionEnvelope(
                sequence=sequence,
                record_id=record.decision_id,
                payload_sha256=checksum,
                payload=record,
            )
            prefix = b"\n" if self._needs_trailing_newline else b""
            encoded = prefix + envelope.canonical_json().encode("utf-8") + b"\n"
            _append_bytes(self.path, encoded, fsync=self._fsync)
            self._needs_trailing_newline = False
            self._checksums[record.decision_id] = checksum
            self._sequences[record.decision_id] = sequence
            self._next_sequence += 1
            return AppendResult(
                status=AppendStatus.APPENDED,
                sequence=sequence,
                record_id=record.decision_id,
            )


class DatasetResearchSink:
    """Governor-facing sink with separate decision and outcome shards."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        fsync: bool = False,
    ) -> None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.decision_path = root / "decisions.jsonl"
        self.outcome_path = root / "outcomes.jsonl"
        self._decisions = DecisionJsonlWriter(self.decision_path, fsync=fsync)
        self._outcomes = OutcomeJsonlWriter(self.outcome_path, fsync=fsync)

    def append_decision(self, decision: HandoffDecision) -> None:
        self._decisions.append(decision)

    def append_outcome(self, outcome: OutcomeRecord) -> None:
        self._outcomes.append(outcome)


def _append_bytes(path: Path, encoded: bytes, *, fsync: bool) -> None:
    descriptor = os.open(
        path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0),
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("append-only dataset write made no progress")
            view = view[written:]
        if fsync:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def scan_outcome_jsonl(
    path: str | os.PathLike[str],
    *,
    allow_partial_final_line: bool = True,
) -> DatasetScan:
    """Strictly validate a shard, optionally ignoring one torn final write."""
    source = Path(path)
    if not source.exists():
        return DatasetScan(envelopes=())
    data = source.read_bytes()
    if not data:
        return DatasetScan(envelopes=())

    raw_lines = data.splitlines(keepends=True)
    envelopes: list[OutcomeEnvelope] = []
    offset = 0
    valid_bytes = 0
    partial_tail_bytes = 0
    needs_trailing_newline = False
    for index, raw_line in enumerate(raw_lines):
        is_last = index == len(raw_lines) - 1
        complete = raw_line.endswith((b"\n", b"\r"))
        content = raw_line.rstrip(b"\r\n")
        if not content:
            raise DatasetCorruptionError(f"blank JSONL line at index {index}")
        try:
            text = content.decode("utf-8")
            envelope = OutcomeEnvelope.from_json(text)
        except Exception as exc:
            if is_last and not complete and allow_partial_final_line:
                partial_tail_bytes = len(raw_line)
                valid_bytes = offset
                break
            raise DatasetCorruptionError(
                f"invalid outcome JSONL line {index}: {exc}"
            ) from exc

        if envelope.sequence != len(envelopes):
            raise DatasetCorruptionError(
                f"non-contiguous sequence at line {index}: "
                f"expected {len(envelopes)}, got {envelope.sequence}"
            )
        envelopes.append(envelope)
        offset += len(raw_line)
        valid_bytes = offset
        if is_last and not complete:
            needs_trailing_newline = True

    records = tuple(envelope.payload for envelope in envelopes)
    _validate_unique_record_ids(records)
    return DatasetScan(
        envelopes=tuple(envelopes),
        partial_tail_bytes=partial_tail_bytes,
        valid_bytes=valid_bytes,
        needs_trailing_newline=needs_trailing_newline,
    )


def scan_decision_jsonl(
    path: str | os.PathLike[str],
    *,
    allow_partial_final_line: bool = True,
) -> DecisionDatasetScan:
    """Strictly validate a decision shard with torn-final-write handling."""
    source = Path(path)
    if not source.exists():
        return DecisionDatasetScan(envelopes=())
    data = source.read_bytes()
    if not data:
        return DecisionDatasetScan(envelopes=())

    raw_lines = data.splitlines(keepends=True)
    envelopes: list[DecisionEnvelope] = []
    offset = 0
    valid_bytes = 0
    partial_tail_bytes = 0
    needs_trailing_newline = False
    for index, raw_line in enumerate(raw_lines):
        is_last = index == len(raw_lines) - 1
        complete = raw_line.endswith((b"\n", b"\r"))
        content = raw_line.rstrip(b"\r\n")
        if not content:
            raise DatasetCorruptionError(f"blank decision JSONL line at index {index}")
        try:
            text = content.decode("utf-8")
            envelope = DecisionEnvelope.from_json(text)
        except Exception as exc:
            if is_last and not complete and allow_partial_final_line:
                partial_tail_bytes = len(raw_line)
                valid_bytes = offset
                break
            raise DatasetCorruptionError(
                f"invalid decision JSONL line {index}: {exc}"
            ) from exc

        if envelope.sequence != len(envelopes):
            raise DatasetCorruptionError(
                f"non-contiguous decision sequence at line {index}: "
                f"expected {len(envelopes)}, got {envelope.sequence}"
            )
        envelopes.append(envelope)
        offset += len(raw_line)
        valid_bytes = offset
        if is_last and not complete:
            needs_trailing_newline = True

    decision_ids = tuple(envelope.payload.decision_id for envelope in envelopes)
    if len(decision_ids) != len(set(decision_ids)):
        raise DatasetCorruptionError("duplicate decision ID in dataset")
    return DecisionDatasetScan(
        envelopes=tuple(envelopes),
        partial_tail_bytes=partial_tail_bytes,
        valid_bytes=valid_bytes,
        needs_trailing_newline=needs_trailing_newline,
    )


def read_outcome_records(
    path: str | os.PathLike[str],
    *,
    allow_partial_final_line: bool = True,
) -> tuple[OutcomeRecord, ...]:
    """Read only fully validated, checksummed ``OutcomeRecord`` payloads."""
    return OutcomeDataset.from_jsonl(
        path, allow_partial_final_line=allow_partial_final_line
    ).records


def read_decision_records(
    path: str | os.PathLike[str],
    *,
    allow_partial_final_line: bool = True,
) -> tuple[HandoffDecision, ...]:
    """Read only fully validated, checksummed decision payloads."""
    scan = scan_decision_jsonl(
        path, allow_partial_final_line=allow_partial_final_line
    )
    return tuple(envelope.payload for envelope in scan.envelopes)


def dataset_fingerprint(records: Sequence[OutcomeRecord]) -> str:
    """Return an order-independent fingerprint of strict outcome contents."""
    values = tuple(records)
    _validate_unique_record_ids(values)
    digest = hashlib.sha256()
    digest.update(DATASET_FINGERPRINT_VERSION.encode("ascii"))
    digest.update(b"\0")
    for record in sorted(values, key=lambda item: item.record_id):
        encoded = record.canonical_json().encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_unique_record_ids(records: Sequence[OutcomeRecord]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.record_id in seen:
            raise DatasetCorruptionError(
                f"duplicate outcome record ID: {record.record_id!r}"
            )
        seen.add(record.record_id)


class TrainingTarget(str, Enum):
    PRIMITIVE_SUCCESS = "primitive_success"
    SKILL_SUCCESS = "skill_success"
    TASK_SUCCESS = "task_success"
    EPISODE_TRUNCATED = "episode_truncated"
    LLM_FINISH = "llm_finish"


class ExclusionReason(str, Enum):
    NON_INVOCATION_SCOPE = "non_invocation_scope"
    STAGING_FAILURE = "staging_failure"
    PERCEPTION_FAILURE = "perception_failure"
    NO_HANDOFF = "no_handoff"
    NO_VLA_INVOCATION = "no_vla_invocation"
    UNKNOWN_VLA_INVOCATION_COUNT = "unknown_vla_invocation_count"
    OUTCOME_LABEL_FAILURE = "outcome_label_failure"
    MISSING_PRE_HANDOFF_STATE = "missing_pre_handoff_state"
    TARGET_LABEL_UNAVAILABLE = "target_label_unavailable"


@dataclass(frozen=True, slots=True)
class LabeledOutcome:
    record: OutcomeRecord
    state: HandoffState
    target: TrainingTarget
    value: bool


@dataclass(frozen=True, slots=True)
class ExcludedOutcome:
    record_id: str
    reason: ExclusionReason


@dataclass(frozen=True, slots=True)
class LabelExtractionResult:
    target: TrainingTarget
    included: tuple[LabeledOutcome, ...]
    excluded: tuple[ExcludedOutcome, ...]


def extract_labeled_outcomes(
    records: Sequence[OutcomeRecord],
    *,
    target: TrainingTarget | str,
) -> LabelExtractionResult:
    """Select exactly one explicit target without semantic fallback.

    Analytic staging and perception failures are not VLA negatives.  Neither
    are trials where handoff never occurred or no VLA invocation was made.
    Unknown requested labels remain excluded rather than being substituted by a
    different success signal.
    """
    selected_target = TrainingTarget(target)
    included: list[LabeledOutcome] = []
    excluded: list[ExcludedOutcome] = []
    for record in records:
        reason = _training_exclusion(record)
        if reason is not None:
            excluded.append(ExcludedOutcome(record.record_id, reason))
            continue
        value = record.labels.target_value(selected_target.value)
        if value is None:
            excluded.append(
                ExcludedOutcome(
                    record.record_id, ExclusionReason.TARGET_LABEL_UNAVAILABLE
                )
            )
            continue
        state = record.pre_handoff_state
        if state is None:
            # Kept explicit despite OutcomeRecord's handoff validator so this
            # function remains fail-safe if a future schema relaxes that rule.
            excluded.append(
                ExcludedOutcome(
                    record.record_id, ExclusionReason.MISSING_PRE_HANDOFF_STATE
                )
            )
            continue
        included.append(
            LabeledOutcome(
                record=record,
                state=state,
                target=selected_target,
                value=value,
            )
        )
    return LabelExtractionResult(
        target=selected_target,
        included=tuple(included),
        excluded=tuple(excluded),
    )


def _training_exclusion(record: OutcomeRecord) -> ExclusionReason | None:
    record_scope = record.metadata.get("record_scope")
    if record_scope not in (None, "handoff_invocation"):
        return ExclusionReason.NON_INVOCATION_SCOPE
    failure = record.termination.failure_mode
    if failure is FailureMode.OUTCOME_LABEL:
        return ExclusionReason.OUTCOME_LABEL_FAILURE
    reason = record.termination.reason
    if failure is FailureMode.STAGING or reason is TerminationReason.STAGING_FAILURE:
        return ExclusionReason.STAGING_FAILURE
    if (
        failure is FailureMode.PERCEPTION
        or reason is TerminationReason.PERCEPTION_FAILURE
    ):
        return ExclusionReason.PERCEPTION_FAILURE
    if not record.handoff_occurred:
        return ExclusionReason.NO_HANDOFF
    if record.costs.vla_invocations is None:
        return ExclusionReason.UNKNOWN_VLA_INVOCATION_COUNT
    if record.costs.vla_invocations < 1:
        return ExclusionReason.NO_VLA_INVOCATION
    if record.pre_handoff_state is None:
        return ExclusionReason.MISSING_PRE_HANDOFF_STATE
    return None
