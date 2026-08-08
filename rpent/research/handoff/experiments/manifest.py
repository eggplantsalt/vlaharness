"""Deterministic expansion and persistence of experiment trial manifests."""

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.experiments.config import (
    ConditionSpec,
    ExecutionLayer,
    ExperimentConfig,
    PlannerConfig,
    RuntimeConfig,
    RuntimeProbeReference,
    TaskSpec,
    load_strict_json,
    resolve_reference,
    stable_identifier,
)
from rpent.research.handoff.experiments.probes import (
    ProbeStatus,
    RuntimeProbeArtifact,
)
from rpent.research.handoff.types import HandoffRecord, LabelSource

TRIAL_MANIFEST_SCHEMA_VERSION = "rpent.handoff-trial-manifest/v1"
EXPERIMENT_MANIFEST_SCHEMA_VERSION = "rpent.handoff-manifest/v1"


class RuntimeProbeBinding(HandoffRecord):
    """A complete probe artifact bound by path and content digest."""

    name: str
    resolved_path: str
    sha256: str
    artifact: RuntimeProbeArtifact

    @field_validator("name", "resolved_path", "sha256")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value



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
    artifact_bindings: dict[str, str] = Field(default_factory=dict)
    source_revision: str

    @field_validator(
        "trial_id",
        "experiment_id",
        "configuration_id",
        "output_dir",
        "source_revision",
    )
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("artifact_bindings")
    @classmethod
    def validate_artifact_bindings(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not key or not item for key, item in value.items()):
            raise ValueError("artifact binding keys and values must be non-empty")
        return dict(sorted(value.items()))

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
    source_config_sha256: str | None = None
    runtime_probe_bindings: tuple[RuntimeProbeBinding, ...] = ()
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
        probe_names = [item.name for item in self.runtime_probe_bindings]
        if len(probe_names) != len(set(probe_names)):
            raise ValueError("manifest contains duplicate runtime probe names")
        if (self.source_config_path is None) != (self.source_config_sha256 is None):
            raise ValueError(
                "source config path and checksum must be jointly present or absent"
            )
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
        "runtime": trial.runtime.model_dump(mode="json", exclude_none=False),
        "planner": (
            trial.planner.model_dump(mode="json", exclude_none=False)
            if trial.execution_layer is ExecutionLayer.FULL_AGENT
            else None
        ),
        "source_revision": trial.source_revision,
        "artifact_bindings": dict(sorted(trial.artifact_bindings.items())),
    }


def _manifest_identity_payload(manifest: ExperimentManifest) -> dict[str, Any]:
    """Bind the ID to all resolved trials, paths, and source-config identity."""
    return {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "experiment_id": manifest.experiment_id,
        "configuration_id": manifest.configuration_id,
        "source_config_path": manifest.source_config_path,
        "source_config_sha256": manifest.source_config_sha256,
        "runtime_probe_bindings": [
            item.model_dump(mode="json", exclude_none=False)
            for item in manifest.runtime_probe_bindings
        ],
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_embedded_path(value: str, *, base_dir: Path, name: str) -> Path:
    expanded = os.path.expandvars(value)
    if "$" in expanded or re.search(r"%[A-Za-z_][A-Za-z0-9_]*%", expanded):
        raise ValueError(f"{name} contains an unresolved environment variable")
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_runtime_probe_bindings(
    references: tuple[RuntimeProbeReference, ...],
    *,
    config_path: str | os.PathLike[str] | None,
    runtime: RuntimeConfig,
    require_artifacts: bool,
) -> tuple[RuntimeProbeBinding, ...]:
    bindings: list[RuntimeProbeBinding] = []
    for reference in references:
        path = resolve_reference(reference.path, config_path=config_path)
        if not path.is_file():
            if require_artifacts:
                raise FileNotFoundError(f"runtime probe artifact not found: {path}")
            continue
        artifact = RuntimeProbeArtifact.model_validate(load_strict_json(path))
        if not artifact.readiness_ok:
            raise ValueError(
                f"runtime probe artifact is not ready: {reference.name} ({path})"
            )
        if artifact.checkpoint_identity_mismatches:
            raise ValueError(
                f"runtime probe contains checkpoint mismatches: {reference.name}"
            )
        facts = {fact.name: fact for fact in artifact.report.facts}
        unknown = sorted(set(reference.required_observed_facts).difference(facts))
        if unknown:
            raise ValueError(
                f"runtime probe reference {reference.name!r} names unknown facts: {unknown}"
            )
        not_observed = sorted(
            name
            for name in reference.required_observed_facts
            if facts[name].status is not ProbeStatus.OBSERVED
        )
        if not_observed:
            raise ValueError(
                f"runtime probe reference {reference.name!r} lacks observed facts: "
                f"{not_observed}"
            )
        bindings.append(
            RuntimeProbeBinding(
                name=reference.name,
                resolved_path=str(path.resolve()),
                sha256=_sha256(path),
                artifact=artifact,
            )
        )

    if bindings:
        expected = {
            "vla": runtime.pi05_checkpoint_id,
            "sam3": runtime.sam3_checkpoint_id,
        }
        observed: dict[str, set[str]] = {"vla": set(), "sam3": set()}
        for binding in bindings:
            for component in ("vla", "sam3"):
                fact = binding.artifact.report.fact(
                    f"{component}.model_checkpoint_identity"
                )
                if fact.status is not ProbeStatus.OBSERVED:
                    continue
                if not isinstance(fact.value, Mapping):
                    raise ValueError(
                        f"{component} checkpoint fact is not an object"
                    )
                checkpoint = fact.value.get("checkpoint")
                if not isinstance(checkpoint, Mapping):
                    raise ValueError(
                        f"{component} checkpoint fact lacks checkpoint object"
                    )
                configured_id = checkpoint.get("configured_id")
                if not isinstance(configured_id, str) or not configured_id:
                    raise ValueError(
                        f"{component} checkpoint fact lacks configured_id"
                    )
                if configured_id != expected[component]:
                    raise ValueError(
                        f"runtime probe {binding.name!r} observes {component} "
                        f"checkpoint {configured_id!r}, expected {expected[component]!r}"
                    )
                observed[component].add(configured_id)
        missing_components = [
            component for component, values in observed.items() if not values
        ]
        if missing_components:
            raise ValueError(
                "runtime probes do not contain observed checkpoint evidence for: "
                + ", ".join(missing_components)
            )
    return tuple(bindings)


def _resolve_artifact_bindings(
    *,
    condition: ConditionSpec,
    handoff_config_path: str | None,
    model_artifact_path: str | None,
    task_config_path: str | None,
    require_artifacts: bool,
    runtime_probe_sha256s: set[str] | None = None,
) -> tuple[ConditionSpec, dict[str, str]]:
    """Resolve and checksum every behavior-affecting external artifact."""
    bindings: dict[str, str] = {}
    positive_artifacts: list[Any] = []
    if task_config_path is not None:
        task_path = Path(task_config_path)
        if not task_path.is_file() and require_artifacts:
            raise FileNotFoundError(f"task config not found: {task_path}")
        if task_path.is_file():
            bindings["task_config_sha256"] = _sha256(task_path)

    if handoff_config_path is not None:
        handoff_path = Path(handoff_config_path)
        if not handoff_path.is_file() and require_artifacts:
            raise FileNotFoundError(f"handoff config not found: {handoff_path}")
        if not handoff_path.is_file():
            return condition, dict(sorted(bindings.items()))
        bindings["handoff_config_sha256"] = _sha256(handoff_path)
        raw = load_strict_json(handoff_path)
        core = raw.get("core")
        if not isinstance(core, dict):
            raise ValueError(f"handoff config core is not an object: {handoff_path}")
        for policy_name in ("policy", "fallback_policy"):
            policy = core.get(policy_name)
            if not isinstance(policy, dict):
                continue
            reference_value = policy.get("positive_references_file")
            if reference_value is None:
                continue
            if not isinstance(reference_value, str) or not reference_value:
                raise ValueError(
                    f"core.{policy_name}.positive_references_file is invalid"
                )
            reference_path = _resolved_embedded_path(
                reference_value,
                base_dir=handoff_path.parent,
                name=f"core.{policy_name}.positive_references_file",
            )
            if not reference_path.is_file() and require_artifacts:
                raise FileNotFoundError(
                    f"positive-reference artifact not found: {reference_path}"
                )
            if not reference_path.is_file():
                continue
            from rpent.research.handoff.baseline_data import (
                load_positive_reference_artifact,
            )

            reference = load_positive_reference_artifact(reference_path)
            positive_artifacts.append(reference)
            prefix = f"{policy_name}_positive_reference"
            bindings[f"{prefix}_artifact_id"] = reference.artifact_id
            bindings[f"{prefix}_sha256"] = _sha256(reference_path)
            bindings[f"{prefix}_source_dataset_fingerprint"] = (
                reference.source_dataset_fingerprint
            )
            bindings[f"{prefix}_split_assignment_fingerprint"] = (
                reference.split_assignment_fingerprint
            )

    resolved_condition = condition
    if model_artifact_path is not None:
        from rpent.research.handoff.artifacts import (
            ModelArtifactManifest,
            _artifact_id,
        )

        model_root = Path(model_artifact_path)
        manifest_path = model_root / "manifest.json"
        if not manifest_path.is_file() and require_artifacts:
            raise FileNotFoundError(f"model manifest not found: {manifest_path}")
        if not manifest_path.is_file():
            return resolved_condition, dict(sorted(bindings.items()))
        model_manifest = ModelArtifactManifest.model_validate(
            load_strict_json(manifest_path)
        )
        if model_manifest.artifact_id != _artifact_id(model_manifest):
            raise ValueError(f"model manifest identity mismatch: {manifest_path}")
        estimator_path = model_root / model_manifest.estimator_file
        if not estimator_path.is_file():
            raise FileNotFoundError(f"model estimator not found: {estimator_path}")
        actual_estimator_sha = _sha256(estimator_path)
        if actual_estimator_sha != model_manifest.estimator_sha256:
            raise ValueError(f"model estimator checksum mismatch: {estimator_path}")
        if (
            condition.model_artifact_id is not None
            and condition.model_artifact_id != model_manifest.artifact_id
        ):
            raise ValueError(
                "condition model_artifact_id disagrees with resolved artifact"
            )
        runtime_identity_sha = (
            model_manifest.source_identity.external_runtime_identity_sha256
        )
        if runtime_probe_sha256s is not None:
            if runtime_identity_sha is None:
                raise ValueError(
                    "model artifact lacks a content-bound external runtime identity"
                )
            if runtime_identity_sha not in runtime_probe_sha256s:
                raise ValueError(
                    "model artifact runtime identity is not one of the manifest-bound "
                    "probe artifacts"
                )
        for reference in positive_artifacts:
            if reference.source_partition != "train":
                raise ValueError("positive references are not train-partition-only")
            if reference.source_dataset_fingerprint != model_manifest.dataset_fingerprint:
                raise ValueError(
                    "positive references and model artifact use different source datasets"
                )
            if (
                reference.split_assignment_fingerprint
                != model_manifest.split_assignment_fingerprint
            ):
                raise ValueError(
                    "positive references and model artifact use different split assignments"
                )
            if not set(reference.source_record_ids).issubset(
                model_manifest.training_record_ids
            ):
                raise ValueError(
                    "positive-reference source IDs are not a subset of model training IDs"
                )
            if reference.target_label.value != model_manifest.training_target_label:
                raise ValueError(
                    "positive references and model artifact use different target labels"
                )
        resolved_condition = condition.model_copy(
            update={"model_artifact_id": model_manifest.artifact_id}
        )
        bindings.update(
            {
                "model_artifact_id": model_manifest.artifact_id,
                "model_manifest_sha256": _sha256(manifest_path),
                "model_estimator_sha256": actual_estimator_sha,
                "model_runtime_identity_sha256": runtime_identity_sha or "unbound",
                "model_source_revision": (
                    model_manifest.source_identity.source_revision or "unbound"
                ),
            }
        )
    elif condition.model_artifact_id is not None:
        raise ValueError("model_artifact_id configured without model_artifact")
    return resolved_condition, dict(sorted(bindings.items()))


def expand_manifest(
    config: ExperimentConfig,
    *,
    config_path: str | os.PathLike[str] | None = None,
    require_artifacts: bool = True,
) -> ExperimentManifest:
    """Expand tasks × seeds × conditions × repeats in deterministic order."""
    config_id = config.configuration_id
    output_root = _resolve_output_root(config.output_root, config_path=config_path)
    runtime = _resolved_runtime(config.runtime, config_path=config_path)
    source_config_path = (
        str(Path(config_path).resolve()) if config_path is not None else None
    )
    source_config_sha256: str | None = None
    if source_config_path is not None:
        source_path = Path(source_config_path)
        if not source_path.is_file():
            if require_artifacts:
                raise FileNotFoundError(f"source experiment config not found: {source_path}")
        else:
            source_config_sha256 = _sha256(source_path)
    runtime_probe_bindings = _resolve_runtime_probe_bindings(
        config.runtime_probes,
        config_path=config_path,
        runtime=runtime,
        require_artifacts=require_artifacts,
    )
    manifest_bindings: dict[str, str] = {}
    if source_config_sha256 is not None:
        manifest_bindings["source_config_sha256"] = source_config_sha256
    for binding in runtime_probe_bindings:
        prefix = f"runtime_probe_{binding.name}"
        manifest_bindings[f"{prefix}_sha256"] = binding.sha256
    trials: list[TrialManifest] = []

    for task in config.tasks:
        for seed in task.seeds:
            for condition in config.conditions:
                for repeat_index in range(config.repeats):
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
                    resolved_condition, artifact_bindings = (
                        _resolve_artifact_bindings(
                            condition=condition,
                            handoff_config_path=handoff_config_path,
                            model_artifact_path=model_artifact_path,
                            task_config_path=task_config_path,
                            require_artifacts=require_artifacts,
                            runtime_probe_sha256s={
                                binding.sha256 for binding in runtime_probe_bindings
                            }
                            if runtime_probe_bindings
                            else None,
                        )
                    )
                    artifact_bindings = dict(
                        sorted({**manifest_bindings, **artifact_bindings}.items())
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
                    provisional_trial = TrialManifest.model_construct(
                        trial_id="pending",
                        experiment_id=config.experiment_id,
                        configuration_id=config_id,
                        execution_layer=resolved_condition.execution_layer,
                        condition=resolved_condition,
                        task=resolved_task,
                        repeat_index=repeat_index,
                        output_dir="pending",
                        runtime=runtime,
                        planner=config.planner,
                        handoff_config_path=handoff_config_path,
                        model_artifact_path=model_artifact_path,
                        artifact_bindings=artifact_bindings,
                        source_revision=config.source_revision,
                    )
                    trial_id = stable_identifier(
                        "trial", _resolved_trial_identity_payload(provisional_trial)
                    )
                    trials.append(
                        TrialManifest(
                            trial_id=trial_id,
                            experiment_id=config.experiment_id,
                            configuration_id=config_id,
                            execution_layer=resolved_condition.execution_layer,
                            condition=resolved_condition,
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
                            artifact_bindings=artifact_bindings,
                            source_revision=config.source_revision,
                        )
                    )

    provisional = ExperimentManifest.model_construct(
        manifest_id="pending",
        experiment_id=config.experiment_id,
        configuration_id=config_id,
        source_config_path=source_config_path,
        source_config_sha256=source_config_sha256,
        runtime_probe_bindings=runtime_probe_bindings,
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


def compute_source_revision(repo_root: str | os.PathLike[str]) -> str:
    """Recompute the runbook's exact commit + dirty/untracked byte identity."""
    root = Path(repo_root).expanduser().resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root
    ).decode("utf-8").strip()
    digest = hashlib.sha256(
        subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=root
        )
    )
    untracked_raw = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
    )
    untracked = sorted(
        item.decode("utf-8")
        for item in untracked_raw.split(b"\0")
        if item
    )
    for name in untracked:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
    return f"git:{revision};worktree-sha256:{digest.hexdigest()}"


def verify_manifest_external_bindings(
    manifest: ExperimentManifest,
    *,
    repo_root: str | os.PathLike[str] | None = None,
    require_runtime_probes: bool = True,
) -> None:
    """Revalidate every mutable external input immediately before execution."""
    base_bindings: dict[str, str] = {}
    if manifest.source_config_path is not None:
        source = Path(manifest.source_config_path)
        if not source.is_file():
            raise FileNotFoundError(f"bound source config is missing: {source}")
        actual = _sha256(source)
        if actual != manifest.source_config_sha256:
            raise ValueError("bound source experiment config checksum changed")
        base_bindings["source_config_sha256"] = actual
    if require_runtime_probes and not manifest.runtime_probe_bindings:
        raise ValueError(
            "server execution requires content-bound runtime probe artifacts"
        )
    for binding in manifest.runtime_probe_bindings:
        path = Path(binding.resolved_path)
        if not path.is_file():
            raise FileNotFoundError(f"bound runtime probe is missing: {path}")
        actual = _sha256(path)
        if actual != binding.sha256:
            raise ValueError(
                f"bound runtime probe checksum changed: {binding.name}"
            )
        parsed = RuntimeProbeArtifact.model_validate(load_strict_json(path))
        if parsed != binding.artifact:
            raise ValueError(
                f"bound runtime probe content changed: {binding.name}"
            )
        base_bindings[f"runtime_probe_{binding.name}_sha256"] = actual

    for trial in manifest.trials:
        resolved_condition, current = _resolve_artifact_bindings(
            condition=trial.condition,
            handoff_config_path=trial.handoff_config_path,
            model_artifact_path=trial.model_artifact_path,
            task_config_path=trial.task.task_config_path,
            require_artifacts=True,
            runtime_probe_sha256s={
                binding.sha256 for binding in manifest.runtime_probe_bindings
            }
            if manifest.runtime_probe_bindings
            else None,
        )
        if resolved_condition != trial.condition:
            raise ValueError(
                f"trial condition/artifact identity changed: {trial.trial_id}"
            )
        expected = dict(sorted({**base_bindings, **current}.items()))
        if expected != trial.artifact_bindings:
            raise ValueError(
                f"trial external artifact bindings changed: {trial.trial_id}"
            )
    if repo_root is not None:
        actual_source = compute_source_revision(repo_root)
        expected_sources = {trial.source_revision for trial in manifest.trials}
        if expected_sources != {actual_source}:
            raise ValueError(
                "current repository bytes disagree with manifest source_revision: "
                f"expected={sorted(expected_sources)!r}, actual={actual_source!r}"
            )


def verify_trial_artifact_bindings(trial: TrialManifest) -> None:
    """Recheck one trial's behavior-affecting files, excluding manifest-wide inputs."""
    resolved_condition, current = _resolve_artifact_bindings(
        condition=trial.condition,
        handoff_config_path=trial.handoff_config_path,
        model_artifact_path=trial.model_artifact_path,
        task_config_path=trial.task.task_config_path,
        require_artifacts=True,
    )
    if resolved_condition != trial.condition:
        raise ValueError("trial condition/model identity changed")
    mismatches = {
        key: {
            "expected": trial.artifact_bindings.get(key),
            "actual": value,
        }
        for key, value in current.items()
        if trial.artifact_bindings.get(key) != value
    }
    if mismatches:
        raise ValueError(f"trial artifact bindings changed: {mismatches}")


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
    "RuntimeProbeBinding",
    "TrialManifest",
    "compute_source_revision",
    "expand_manifest",
    "load_manifest",
    "load_trial_jsonl",
    "select_trials",
    "verify_manifest_external_bindings",
    "verify_trial_artifact_bindings",
    "write_manifest",
    "write_trial_jsonl",
]
