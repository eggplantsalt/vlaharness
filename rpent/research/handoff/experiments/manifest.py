"""Deterministic expansion and persistence of experiment trial manifests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.experiments.config import (
    ConditionSpec,
    ExecutionLayer,
    ExperimentConfig,
    PlannerConfig,
    RuntimeConfig,
    TaskSpec,
    load_strict_json,
    resolve_reference,
    stable_identifier,
)
from rpent.research.handoff.types import HandoffRecord, LabelSource

TRIAL_MANIFEST_SCHEMA_VERSION = "rpent.handoff-trial-manifest/v1"
EXPERIMENT_MANIFEST_SCHEMA_VERSION = "rpent.handoff-manifest/v1"


class ResolvedTaskSpec(HandoffRecord):
    """One concrete task/reset cell after matrix expansion."""

    suite: str
    task: int = Field(ge=0)
    seed: int = Field(ge=0)
    reset_id: str | None = None
    target_id: str
    target_description: str
    skill_name: str
    skill_prompt: str
    training_target_label: str
    label_source: LabelSource
    task_config_path: str | None = None

    @field_validator(
        "suite",
        "target_id",
        "target_description",
        "skill_name",
        "skill_prompt",
        "training_target_label",
    )
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value


class TrialManifest(HandoffRecord):
    """One immutable, executable experiment trial."""

    schema_version: Literal[TRIAL_MANIFEST_SCHEMA_VERSION] = (
        TRIAL_MANIFEST_SCHEMA_VERSION
    )
    trial_id: str
    experiment_id: str
    configuration_id: str
    execution_layer: ExecutionLayer
    condition: ConditionSpec
    task: ResolvedTaskSpec
    repeat_index: int = Field(ge=0)
    output_dir: str
    runtime: RuntimeConfig
    planner: PlannerConfig
    handoff_config_path: str | None = None
    model_artifact_path: str | None = None
    source_revision: str | None = None

    @field_validator(
        "trial_id", "experiment_id", "configuration_id", "output_dir"
    )
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_resolved_condition(self) -> Self:
        if self.execution_layer is not self.condition.execution_layer:
            raise ValueError("trial execution_layer disagrees with condition")
        if self.condition.handoff_enabled and self.condition.handoff_config is not None:
            if self.handoff_config_path is None:
                raise ValueError("enabled handoff config was not resolved")
        elif self.handoff_config_path is not None:
            raise ValueError("unexpected resolved handoff config path")
        if self.condition.model_artifact is not None and self.model_artifact_path is None:
            raise ValueError("model artifact was not resolved")
        expected_trial_id = stable_identifier(
            "trial", _resolved_trial_identity_payload(self)
        )
        if self.trial_id != expected_trial_id:
            raise ValueError(
                f"trial_id does not bind the resolved scientific payload: "
                f"expected {expected_trial_id!r}"
            )
        return self


class ExperimentManifest(HandoffRecord):
    """Complete deterministic manifest for an expanded matrix."""

    schema_version: Literal[EXPERIMENT_MANIFEST_SCHEMA_VERSION] = (
        EXPERIMENT_MANIFEST_SCHEMA_VERSION
    )
    manifest_id: str
    experiment_id: str
    configuration_id: str
    source_config_path: str | None = None
    trials: tuple[TrialManifest, ...]

    @field_validator("manifest_id", "experiment_id", "configuration_id")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_trials(self) -> Self:
        if not self.trials:
            raise ValueError("manifest must contain at least one trial")
        ids = [trial.trial_id for trial in self.trials]
        if len(ids) != len(set(ids)):
            raise ValueError("manifest contains duplicate trial IDs")
        outputs = [str(Path(trial.output_dir)) for trial in self.trials]
        if len(outputs) != len(set(outputs)):
            raise ValueError("manifest contains duplicate trial output directories")
        for trial in self.trials:
            if trial.experiment_id != self.experiment_id:
                raise ValueError("trial experiment_id disagrees with manifest")
            if trial.configuration_id != self.configuration_id:
                raise ValueError("trial configuration_id disagrees with manifest")
        expected_manifest_id = stable_identifier(
            "manifest", _manifest_identity_payload(self)
        )
        if self.manifest_id != expected_manifest_id:
            raise ValueError(
                f"manifest_id does not bind the complete resolved manifest: "
                f"expected {expected_manifest_id!r}"
            )
        return self


def _resolved_trial_identity_payload(trial: TrialManifest) -> dict[str, Any]:
    """Rebuild the exact scientific identity from a resolved trial."""
    return {
        "schema_version": TRIAL_MANIFEST_SCHEMA_VERSION,
        "experiment_id": trial.experiment_id,
        "execution_layer": trial.execution_layer.value,
        "condition": trial.condition.model_dump(mode="json", exclude_none=False),
        "task": {
            "suite": trial.task.suite,
            "task": trial.task.task,
            "seed": trial.task.seed,
            "reset_id": trial.task.reset_id,
            "target_id": trial.task.target_id,
            "target_description": trial.task.target_description,
            "skill_name": trial.task.skill_name,
            "skill_prompt": trial.task.skill_prompt,
            "training_target_label": trial.task.training_target_label,
            "label_source": trial.task.label_source.value,
        },
        "repeat_index": trial.repeat_index,
        "runtime": {
            "env_name": trial.runtime.env_name,
            "libero_type": trial.runtime.libero_type,
            "max_episode_steps": trial.runtime.max_episode_steps,
        },
        "planner": (
            trial.planner.model_dump(mode="json", exclude_none=False)
            if trial.execution_layer is ExecutionLayer.FULL_AGENT
            else None
        ),
        "source_revision": trial.source_revision,
    }


def _manifest_identity_payload(manifest: ExperimentManifest) -> dict[str, Any]:
    """Bind the ID to all resolved trials, paths, and source-config identity."""
    return {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "experiment_id": manifest.experiment_id,
        "configuration_id": manifest.configuration_id,
        "source_config_path": manifest.source_config_path,
        "trials": [
            trial.model_dump(mode="json", exclude_none=False)
            for trial in manifest.trials
        ],
    }


def _resolved_reset_id(task: TaskSpec, *, seed: int, repeat_index: int) -> str | None:
    if task.reset_id_template is None:
        return None
    try:
        value = task.reset_id_template.format(
            suite=task.suite,
            task=task.task,
            seed=seed,
            repeat=repeat_index,
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"invalid reset_id_template for {task.suite}/task {task.task}: {exc}"
        ) from exc
    if not value:
        raise ValueError("reset_id_template produced an empty reset ID")
    return value


def _trial_identity_payload(
    config: ExperimentConfig,
    condition: ConditionSpec,
    task: TaskSpec,
    *,
    seed: int,
    repeat_index: int,
) -> dict[str, Any]:
    """Scientific identity fields; intentionally excludes output locations."""
    return {
        "schema_version": TRIAL_MANIFEST_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "execution_layer": condition.execution_layer.value,
        "condition": condition.model_dump(mode="json", exclude_none=False),
        "task": {
            "suite": task.suite,
            "task": task.task,
            "seed": seed,
            "reset_id": _resolved_reset_id(
                task, seed=seed, repeat_index=repeat_index
            ),
            "target_id": task.target_id,
            "target_description": task.target_description,
            "skill_name": task.skill_name,
            "skill_prompt": task.skill_prompt,
            "training_target_label": task.training_target_label,
            "label_source": task.label_source.value,
        },
        "repeat_index": repeat_index,
        "runtime": {
            "env_name": config.runtime.env_name,
            "libero_type": config.runtime.libero_type,
            "max_episode_steps": config.runtime.max_episode_steps,
        },
        "planner": (
            config.planner.model_dump(mode="json", exclude_none=False)
            if condition.execution_layer is ExecutionLayer.FULL_AGENT
            else None
        ),
        "source_revision": config.source_revision,
    }


def _resolve_output_root(
    output_root: str,
    *,
    config_path: str | os.PathLike[str] | None,
) -> Path:
    root = Path(output_root).expanduser()
    if not root.is_absolute() and config_path is not None:
        root = Path(config_path).resolve().parent / root
    return root.resolve()


def _resolved_runtime(
    runtime: RuntimeConfig,
    *,
    config_path: str | os.PathLike[str] | None,
) -> RuntimeConfig:
    updates: dict[str, str] = {}
    for field_name in ("pi05_checkpoint_path", "sam3_checkpoint_path"):
        value = getattr(runtime, field_name)
        if value is not None:
            updates[field_name] = str(
                resolve_reference(value, config_path=config_path)
            )
    return runtime.model_copy(update=updates) if updates else runtime


def expand_manifest(
    config: ExperimentConfig,
    *,
    config_path: str | os.PathLike[str] | None = None,
) -> ExperimentManifest:
    """Expand tasks × seeds × conditions × repeats in deterministic order."""
    config_id = config.configuration_id
    output_root = _resolve_output_root(config.output_root, config_path=config_path)
    runtime = _resolved_runtime(config.runtime, config_path=config_path)
    trials: list[TrialManifest] = []

    for task in config.tasks:
        for seed in task.seeds:
            for condition in config.conditions:
                for repeat_index in range(config.repeats):
                    identity_payload = _trial_identity_payload(
                        config,
                        condition,
                        task,
                        seed=seed,
                        repeat_index=repeat_index,
                    )
                    trial_id = stable_identifier("trial", identity_payload)
                    task_config_path = (
                        str(resolve_reference(task.task_config, config_path=config_path))
                        if task.task_config is not None
                        else None
                    )
                    handoff_config_path = (
                        str(resolve_reference(condition.handoff_config, config_path=config_path))
                        if condition.handoff_config is not None
                        else None
                    )
                    model_artifact_path = (
                        str(resolve_reference(condition.model_artifact, config_path=config_path))
                        if condition.model_artifact is not None
                        else None
                    )
                    resolved_task = ResolvedTaskSpec(
                        suite=task.suite,
                        task=task.task,
                        seed=seed,
                        reset_id=_resolved_reset_id(
                            task,
                            seed=seed,
                            repeat_index=repeat_index,
                        ),
                        target_id=task.target_id,
                        target_description=task.target_description,
                        skill_name=task.skill_name,
                        skill_prompt=task.skill_prompt,
                        training_target_label=task.training_target_label,
                        label_source=task.label_source,
                        task_config_path=task_config_path,
                    )
                    trials.append(
                        TrialManifest(
                            trial_id=trial_id,
                            experiment_id=config.experiment_id,
                            configuration_id=config_id,
                            execution_layer=condition.execution_layer,
                            condition=condition,
                            task=resolved_task,
                            repeat_index=repeat_index,
                            output_dir=str(
                                output_root
                                / config.experiment_id
                                / "trials"
                                / trial_id
                            ),
                            runtime=runtime,
                            planner=config.planner,
                            handoff_config_path=handoff_config_path,
                            model_artifact_path=model_artifact_path,
                            source_revision=config.source_revision,
                        )
                    )

    provisional = ExperimentManifest.model_construct(
        manifest_id="pending",
        experiment_id=config.experiment_id,
        configuration_id=config_id,
        source_config_path=(
            str(Path(config_path).resolve()) if config_path is not None else None
        ),
        trials=tuple(trials),
    )
    return ExperimentManifest(
        **provisional.model_dump(mode="python", exclude={"manifest_id"}),
        manifest_id=stable_identifier(
            "manifest", _manifest_identity_payload(provisional)
        ),
    )


def write_manifest(
    manifest: ExperimentManifest,
    path: str | os.PathLike[str],
) -> Path:
    """Atomically write the complete manifest as strict normalized JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(
            manifest.model_dump(mode="json", exclude_none=False),
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


def load_manifest(path: str | os.PathLike[str]) -> ExperimentManifest:
    """Strictly load a complete manifest JSON file."""
    return ExperimentManifest.model_validate(load_strict_json(path))


def write_trial_jsonl(
    manifest: ExperimentManifest,
    path: str | os.PathLike[str],
) -> Path:
    """Atomically write resolved trials as one canonical record per line."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for trial in manifest.trials:
            stream.write(trial.canonical_json())
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return destination


def load_trial_jsonl(path: str | os.PathLike[str]) -> tuple[TrialManifest, ...]:
    """Load strict trial JSONL and fail closed on blank/corrupt/duplicate rows."""
    source = Path(path)
    trials: list[TrialManifest] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(f"blank line in {source} at line {line_number}")
            try:
                trial = TrialManifest.from_json(line)
            except Exception as exc:
                raise ValueError(
                    f"invalid trial manifest in {source} at line {line_number}: {exc}"
                ) from exc
            if trial.trial_id in seen:
                raise ValueError(f"duplicate trial ID in {source}: {trial.trial_id}")
            seen.add(trial.trial_id)
            trials.append(trial)
    if not trials:
        raise ValueError(f"trial manifest is empty: {source}")
    return tuple(trials)


def select_trials(
    manifest: ExperimentManifest,
    *,
    execution_layer: ExecutionLayer | None = None,
    condition_names: set[str] | None = None,
    trial_ids: set[str] | None = None,
) -> tuple[TrialManifest, ...]:
    """Select a deterministic manifest subset without changing trial identity."""
    return tuple(
        trial
        for trial in manifest.trials
        if (execution_layer is None or trial.execution_layer is execution_layer)
        and (condition_names is None or trial.condition.name in condition_names)
        and (trial_ids is None or trial.trial_id in trial_ids)
    )


__all__ = [
    "ExperimentManifest",
    "ResolvedTaskSpec",
    "TrialManifest",
    "expand_manifest",
    "load_manifest",
    "load_trial_jsonl",
    "select_trials",
    "write_manifest",
    "write_trial_jsonl",
]
