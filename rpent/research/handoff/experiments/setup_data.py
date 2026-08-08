"""Append-only label/setup-side records for controlled Gate-0 collection."""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from rpent.research.handoff.privileged import ExperimentSetupRecord
from rpent.research.handoff.types import HandoffRecord

SETUP_ENVELOPE_SCHEMA_VERSION = "rpent.handoff-setup-jsonl/v1"


def setup_checksum(record: ExperimentSetupRecord) -> str:
    return hashlib.sha256(record.canonical_json().encode("utf-8")).hexdigest()


class SetupEnvelope(HandoffRecord):
    schema_version: Literal[SETUP_ENVELOPE_SCHEMA_VERSION] = (
        SETUP_ENVELOPE_SCHEMA_VERSION
    )
    sequence: int = Field(ge=0)
    record_id: str
    payload_sha256: str
    payload: ExperimentSetupRecord

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        if self.record_id != self.payload.record_id:
            raise ValueError("setup envelope and payload IDs disagree")
        if self.payload_sha256 != setup_checksum(self.payload):
            raise ValueError("setup payload checksum mismatch")
        return self


@dataclass(frozen=True, slots=True)
class SetupScan:
    envelopes: tuple[SetupEnvelope, ...]
    valid_bytes: int
    partial_tail_bytes: int = 0
    needs_trailing_newline: bool = False


def scan_setup_jsonl(
    path: str | os.PathLike[str],
    *,
    allow_partial_final_line: bool = True,
) -> SetupScan:
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return SetupScan(envelopes=(), valid_bytes=0)
    data = source.read_bytes()
    envelopes: list[SetupEnvelope] = []
    offset = 0
    valid_bytes = 0
    partial = 0
    needs_newline = False
    seen: set[str] = set()
    for index, raw in enumerate(data.splitlines(keepends=True)):
        last = offset + len(raw) == len(data)
        complete = raw.endswith((b"\n", b"\r"))
        content = raw.rstrip(b"\r\n")
        if not content:
            raise ValueError(f"blank setup JSONL line at index {index}")
        try:
            envelope = SetupEnvelope.from_json(content.decode("utf-8"))
        except Exception as exc:
            if last and not complete and allow_partial_final_line:
                partial = len(raw)
                valid_bytes = offset
                break
            raise ValueError(f"invalid setup JSONL line {index}: {exc}") from exc
        if envelope.sequence != len(envelopes):
            raise ValueError(
                f"non-contiguous setup sequence: expected {len(envelopes)}, "
                f"got {envelope.sequence}"
            )
        if envelope.record_id in seen:
            raise ValueError(f"duplicate setup record ID: {envelope.record_id}")
        seen.add(envelope.record_id)
        envelopes.append(envelope)
        offset += len(raw)
        valid_bytes = offset
        needs_newline = last and not complete
    return SetupScan(
        envelopes=tuple(envelopes),
        valid_bytes=valid_bytes,
        partial_tail_bytes=partial,
        needs_trailing_newline=needs_newline,
    )


def read_setup_records(
    path: str | os.PathLike[str],
) -> tuple[ExperimentSetupRecord, ...]:
    return tuple(item.payload for item in scan_setup_jsonl(path).envelopes)


class SetupJsonlWriter:
    """Idempotent single-shard writer implementing the Gate-0 setup sink."""

    def __init__(self, path: str | os.PathLike[str], *, fsync: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._fsync = fsync
        self._lock = threading.Lock()
        scan = scan_setup_jsonl(self.path)
        if scan.partial_tail_bytes:
            with self.path.open("r+b") as stream:
                stream.truncate(scan.valid_bytes)
                stream.flush()
                if fsync:
                    os.fsync(stream.fileno())
        self._needs_newline = scan.needs_trailing_newline
        self._checksums = {
            item.record_id: item.payload_sha256 for item in scan.envelopes
        }
        self._next_sequence = len(scan.envelopes)

    def append_setup(self, setup: ExperimentSetupRecord) -> None:
        checksum = setup_checksum(setup)
        with self._lock:
            previous = self._checksums.get(setup.record_id)
            if previous is not None:
                if previous != checksum:
                    raise ValueError(
                        f"conflicting setup retry for record ID {setup.record_id}"
                    )
                return
            envelope = SetupEnvelope(
                sequence=self._next_sequence,
                record_id=setup.record_id,
                payload_sha256=checksum,
                payload=setup,
            )
            encoded = (
                (b"\n" if self._needs_newline else b"")
                + envelope.canonical_json().encode("utf-8")
                + b"\n"
            )
            descriptor = os.open(
                self.path,
                os.O_APPEND
                | os.O_CREAT
                | os.O_WRONLY
                | getattr(os, "O_BINARY", 0),
            )
            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("setup append made no progress")
                    view = view[written:]
                if self._fsync:
                    os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._needs_newline = False
            self._checksums[setup.record_id] = checksum
            self._next_sequence += 1


__all__ = [
    "SETUP_ENVELOPE_SCHEMA_VERSION",
    "SetupEnvelope",
    "SetupJsonlWriter",
    "SetupScan",
    "read_setup_records",
    "scan_setup_jsonl",
    "setup_checksum",
]
