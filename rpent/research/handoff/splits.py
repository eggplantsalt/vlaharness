"""Deterministic connected-component group splits for outcome records."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Self, Sequence

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.dataset import dataset_fingerprint
from rpent.research.handoff.types import HandoffRecord, OutcomeRecord

SPLIT_SCHEMA_VERSION = "rpent.handoff-group-split/v1"

_ALLOWED_GROUP_FIELDS = {
    "record_id",
    "identity.run_id",
    "identity.episode_id",
    "identity.trial_id",
    "identity.invocation_id",
    "identity.candidate_id",
    "identity.suite",
    "identity.task_id",
    "identity.seed",
    "identity.reset_id",
    "identity.repeat_index",
    "skill.name",
    "controller.method",
    "pre_handoff_state.state_id",
}


class SplitError(ValueError):
    """Base class for invalid or impossible group splits."""


class GroupKeyError(SplitError):
    """A configured group identity is invalid or unavailable."""


class GroupLeakageError(SplitError):
    """At least one configured group crosses split boundaries."""


class UnsplittableGroupsError(SplitError):
    """Connected grouping leaves too few independent components."""


class SplitName(str, Enum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"


class GroupConstraint(HandoffRecord):
    """One equality constraint; sharing its composite value joins two rows."""

    name: str
    fields: tuple[str, ...]
    allow_missing: bool = False

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value:
            raise ValueError("group constraint name must be non-empty")
        return value

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("group constraint needs at least one field")
        if len(value) != len(set(value)):
            raise ValueError("group constraint fields must be unique")
        unsupported = sorted(set(value).difference(_ALLOWED_GROUP_FIELDS))
        if unsupported:
            raise ValueError(f"unsupported group fields: {unsupported}")
        return value


def default_group_constraints() -> tuple[GroupConstraint, ...]:
    """Leakage barriers for episodes, resets, and repeated candidates."""
    return (
        GroupConstraint(
            name="episode",
            fields=("identity.run_id", "identity.episode_id"),
        ),
        GroupConstraint(
            name="reset",
            fields=(
                "identity.suite",
                "identity.task_id",
                "identity.reset_id",
            ),
        ),
        GroupConstraint(
            name="candidate",
            fields=(
                "identity.suite",
                "identity.task_id",
                "identity.candidate_id",
            ),
        ),
    )


class GroupSplitConfig(HandoffRecord):
    """Serializable deterministic train/calibration/test split config."""

    schema_version: Literal[SPLIT_SCHEMA_VERSION] = SPLIT_SCHEMA_VERSION
    train_fraction: float = Field(default=0.7, gt=0.0, lt=1.0)
    calibration_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    seed: int = 0
    constraints: tuple[GroupConstraint, ...] = Field(
        default_factory=default_group_constraints
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        total = self.train_fraction + self.calibration_fraction + self.test_fraction
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("split fractions must sum to 1")
        if not self.constraints:
            raise ValueError("at least one group constraint is required")
        names = tuple(item.name for item in self.constraints)
        if len(names) != len(set(names)):
            raise ValueError("group constraint names must be unique")
        return self

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class SplitEntry(HandoffRecord):
    record_id: str
    split: SplitName

    @field_validator("record_id")
    @classmethod
    def validate_record_id(cls, value: str) -> str:
        if not value:
            raise ValueError("record_id must be non-empty")
        return value


class SplitAssignment(HandoffRecord):
    schema_version: Literal[SPLIT_SCHEMA_VERSION] = SPLIT_SCHEMA_VERSION
    dataset_fingerprint: str
    config_fingerprint: str
    entries: tuple[SplitEntry, ...]

    @field_validator("dataset_fingerprint", "config_fingerprint")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_entries(self) -> Self:
        ids = tuple(entry.record_id for entry in self.entries)
        if len(ids) != len(set(ids)):
            raise ValueError("split assignment record IDs must be unique")
        if ids != tuple(sorted(ids)):
            raise ValueError("split assignment entries must be sorted by record_id")
        return self

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, SplitName]:
        return {entry.record_id: entry.split for entry in self.entries}


@dataclass(frozen=True, slots=True)
class SplitResult:
    train: tuple[OutcomeRecord, ...]
    calibration: tuple[OutcomeRecord, ...]
    test: tuple[OutcomeRecord, ...]
    assignment: SplitAssignment


def apply_split_assignment(
    records: Sequence[OutcomeRecord],
    assignment: SplitAssignment,
) -> dict[SplitName, tuple[OutcomeRecord, ...]]:
    """Materialize an existing assignment without silently changing its cohort.

    The input dataset may contain records excluded before training, but every
    assignment id must exist exactly once and the selected eligible cohort must
    reproduce the assignment fingerprint. This makes held-out evaluation use
    the exact split that produced the model artifact.
    """
    by_id = {record.record_id: record for record in records}
    if len(by_id) != len(tuple(records)):
        # Reuse the dataset-level error and its precise duplicate-id message.
        dataset_fingerprint(records)
    assigned_ids = {entry.record_id for entry in assignment.entries}
    missing = sorted(assigned_ids.difference(by_id))
    if missing:
        raise SplitError(
            "split assignment references records absent from the dataset: "
            + ", ".join(missing[:10])
        )
    eligible = tuple(by_id[entry.record_id] for entry in assignment.entries)
    actual_fingerprint = dataset_fingerprint(eligible)
    if actual_fingerprint != assignment.dataset_fingerprint:
        raise SplitError(
            "split assignment dataset fingerprint mismatch: "
            f"expected {assignment.dataset_fingerprint}, got {actual_fingerprint}"
        )
    partitions: dict[SplitName, list[OutcomeRecord]] = {
        split: [] for split in SplitName
    }
    for entry in assignment.entries:
        partitions[entry.split].append(by_id[entry.record_id])
    return {
        split: tuple(partitions[split])
        for split in SplitName
    }


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def connected_group_ids(
    records: Sequence[OutcomeRecord],
    constraints: Sequence[GroupConstraint] | None = None,
) -> dict[str, str]:
    """Return deterministic union-group IDs for grouped bootstrapping.

    Two records receive the same ID when they share any configured group key,
    including transitively through different episode/reset/candidate keys.
    """
    ordered = tuple(sorted(records, key=lambda record: record.record_id))
    if len({record.record_id for record in ordered}) != len(ordered):
        raise SplitError("outcome record IDs must be unique")
    selected_constraints = tuple(constraints or default_group_constraints())
    if not selected_constraints:
        raise GroupKeyError("at least one group constraint is required")
    components = _connected_components(ordered, selected_constraints)
    result: dict[str, str] = {}
    for component in components:
        digest = hashlib.sha256()
        digest.update(b"rpent.handoff-connected-group/v1\0")
        for record in component:
            digest.update(record.record_id.encode("utf-8"))
            digest.update(b"\0")
        group_id = f"group:{digest.hexdigest()}"
        for record in component:
            result[record.record_id] = group_id
    return result


def split_outcomes(
    records: Sequence[OutcomeRecord],
    config: GroupSplitConfig | None = None,
) -> SplitResult:
    """Split connected groups without leaking any configured identity."""
    config = config or GroupSplitConfig()
    ordered = tuple(sorted(records, key=lambda record: record.record_id))
    if len({record.record_id for record in ordered}) != len(ordered):
        raise SplitError("outcome record IDs must be unique")
    if not ordered:
        raise UnsplittableGroupsError("cannot split an empty dataset")

    components = _connected_components(ordered, config.constraints)
    if len(components) < 3:
        raise UnsplittableGroupsError(
            "group constraints leave fewer than three independent components"
        )
    component_splits = _allocate_components(components, len(ordered), config)
    split_by_record: dict[str, SplitName] = {}
    for component_index, component in enumerate(components):
        split = component_splits[component_index]
        for record in component:
            split_by_record[record.record_id] = split

    entries = tuple(
        SplitEntry(record_id=record.record_id, split=split_by_record[record.record_id])
        for record in ordered
    )
    assignment = SplitAssignment(
        dataset_fingerprint=dataset_fingerprint(ordered),
        config_fingerprint=config.fingerprint,
        entries=entries,
    )
    verify_no_group_leakage(ordered, assignment, config.constraints)
    buckets = {
        SplitName.TRAIN: [],
        SplitName.CALIBRATION: [],
        SplitName.TEST: [],
    }
    for record in ordered:
        buckets[split_by_record[record.record_id]].append(record)
    if any(not values for values in buckets.values()):
        raise UnsplittableGroupsError("split allocation produced an empty partition")
    return SplitResult(
        train=tuple(buckets[SplitName.TRAIN]),
        calibration=tuple(buckets[SplitName.CALIBRATION]),
        test=tuple(buckets[SplitName.TEST]),
        assignment=assignment,
    )


def verify_no_group_leakage(
    records: Sequence[OutcomeRecord],
    assignment: SplitAssignment,
    constraints: Sequence[GroupConstraint] | None = None,
) -> None:
    """Fail if any episode/reset/candidate group appears in two partitions."""
    constraints = tuple(constraints or default_group_constraints())
    split_by_record = assignment.as_dict()
    record_ids = {record.record_id for record in records}
    if set(split_by_record) != record_ids:
        missing = sorted(record_ids.difference(split_by_record))
        extra = sorted(set(split_by_record).difference(record_ids))
        raise GroupLeakageError(
            f"split assignment membership mismatch; missing={missing}, extra={extra}"
        )
    if assignment.dataset_fingerprint != dataset_fingerprint(records):
        raise GroupLeakageError("split assignment dataset fingerprint mismatch")

    for constraint in constraints:
        owners: dict[tuple[str, ...], SplitName] = {}
        for record in records:
            token = _group_token(record, constraint)
            split = split_by_record[record.record_id]
            previous = owners.setdefault(token, split)
            if previous is not split:
                raise GroupLeakageError(
                    f"{constraint.name} group {token!r} leaks across "
                    f"{previous.value} and {split.value}"
                )


def _connected_components(
    records: tuple[OutcomeRecord, ...],
    constraints: Sequence[GroupConstraint],
) -> list[tuple[OutcomeRecord, ...]]:
    disjoint = _DisjointSet(len(records))
    for constraint in constraints:
        first_by_token: dict[tuple[str, ...], int] = {}
        for index, record in enumerate(records):
            token = _group_token(record, constraint)
            first = first_by_token.setdefault(token, index)
            disjoint.union(first, index)

    members: dict[int, list[OutcomeRecord]] = {}
    for index, record in enumerate(records):
        members.setdefault(disjoint.find(index), []).append(record)
    components = [tuple(values) for values in members.values()]
    components.sort(key=lambda values: tuple(record.record_id for record in values))
    return components


def _allocate_components(
    components: Sequence[tuple[OutcomeRecord, ...]],
    record_count: int,
    config: GroupSplitConfig,
) -> dict[int, SplitName]:
    fractions = {
        SplitName.TRAIN: config.train_fraction,
        SplitName.CALIBRATION: config.calibration_fraction,
        SplitName.TEST: config.test_fraction,
    }
    targets = {split: fraction * record_count for split, fraction in fractions.items()}
    counts = {split: 0 for split in SplitName}
    assignments: dict[int, SplitName] = {}

    keyed = [
        (
            index,
            len(component),
            _component_tiebreak(component, config.seed),
        )
        for index, component in enumerate(components)
    ]
    by_size = sorted(keyed, key=lambda item: (item[1], item[2]))
    largest_split = max(SplitName, key=lambda split: fractions[split])
    smaller_splits = sorted(
        (split for split in SplitName if split is not largest_split),
        key=lambda split: (fractions[split], split.value),
    )
    for split in smaller_splits:
        index, size, _ = by_size.pop(0)
        assignments[index] = split
        counts[split] += size
    index, size, _ = by_size.pop(-1)
    assignments[index] = largest_split
    counts[largest_split] += size

    for index, size, _ in sorted(by_size, key=lambda item: (-item[1], item[2])):
        split = max(
            SplitName,
            key=lambda candidate: (
                (targets[candidate] - counts[candidate]) / targets[candidate],
                -counts[candidate],
                candidate.value,
            ),
        )
        assignments[index] = split
        counts[split] += size
    return assignments


def _component_tiebreak(
    component: Sequence[OutcomeRecord],
    seed: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    for record in component:
        digest.update(record.record_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _group_token(
    record: OutcomeRecord,
    constraint: GroupConstraint,
) -> tuple[str, ...]:
    parts: list[str] = []
    missing = False
    for path in constraint.fields:
        value = _resolve_field(record, path)
        if value is None:
            missing = True
        parts.append(_stable_scalar(value))
    if missing:
        if not constraint.allow_missing:
            raise GroupKeyError(
                f"record {record.record_id!r} lacks required {constraint.name} key "
                f"from fields {constraint.fields}"
            )
        parts.append(f"missing-record:{record.record_id}")
    return tuple(parts)


def _resolve_field(record: OutcomeRecord, path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if value is None:
            return None
        if not hasattr(value, part):
            raise GroupKeyError(f"group field {path!r} is unavailable")
        value = getattr(value, part)
    return value


def _stable_scalar(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise GroupKeyError(f"group value is not a finite JSON scalar: {value!r}") from exc
    return f"{type(value).__name__}:{encoded}"
