"""Offline, non-mutating preflight checks for experiment manifests."""

from __future__ import annotations

import hashlib
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from rpent.research.handoff.experiments.config import (
    ExecutionLayer,
    ExperimentConfig,
    load_strict_json,
    resolve_reference,
)
from rpent.research.handoff.experiments.manifest import (
    ExperimentManifest,
    expand_manifest,
)
from rpent.research.handoff.types import HandoffRecord

PREFLIGHT_SCHEMA_VERSION = "rpent.handoff-offline-preflight/v1"


class CheckStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class PreflightCheck(HandoffRecord):
    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "message")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("details")
    @classmethod
    def validate_details(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("preflight details must contain finite JSON values") from exc
        return value


class PreflightReport(HandoffRecord):
    schema_version: Literal[PREFLIGHT_SCHEMA_VERSION] = PREFLIGHT_SCHEMA_VERSION
    configuration_id: str
    manifest_id: str
    checks: tuple[PreflightCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.status is not CheckStatus.FAIL for check in self.checks)

    @property
    def failures(self) -> tuple[PreflightCheck, ...]:
        return tuple(check for check in self.checks if check.status is CheckStatus.FAIL)

    def raise_for_failure(self) -> None:
        if self.ok:
            return
        messages = "; ".join(f"{check.name}: {check.message}" for check in self.failures)
        raise RuntimeError(f"offline preflight failed: {messages}")


def _path_check(
    name: str,
    path: Path,
    *,
    expected: str = "any",
) -> PreflightCheck:
    if not path.exists():
        return PreflightCheck(
            name=name,
            status=CheckStatus.FAIL,
            message="required path does not exist",
            details={"path": str(path), "expected": expected},
        )
    if expected == "file" and not path.is_file():
        return PreflightCheck(
            name=name,
            status=CheckStatus.FAIL,
            message="required path is not a file",
            details={"path": str(path)},
        )
    if expected == "directory" and not path.is_dir():
        return PreflightCheck(
            name=name,
            status=CheckStatus.FAIL,
            message="required path is not a directory",
            details={"path": str(path)},
        )
    return PreflightCheck(
        name=name,
        status=CheckStatus.PASS,
        message="required path is present",
        details={"path": str(path)},
    )


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_artifact_checks(
    path: Path,
    *,
    condition_index: int,
    condition,
    config: ExperimentConfig,
) -> list[PreflightCheck]:
    """Validate artifact JSON/checksum without importing or executing joblib."""
    name = f"model_artifact[{condition_index}]"
    present = _path_check(name, path, expected="directory")
    if present.status is CheckStatus.FAIL:
        return [present]
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return [
            present,
            PreflightCheck(
                name=f"{name}.manifest",
                status=CheckStatus.FAIL,
                message="model artifact directory has no manifest.json",
                details={"path": str(manifest_path)},
            ),
        ]
    try:
        from rpent.research.handoff.artifacts import ModelArtifactManifest

        artifact = ModelArtifactManifest.model_validate(load_strict_json(manifest_path))
    except Exception as exc:
        return [
            present,
            PreflightCheck(
                name=f"{name}.manifest",
                status=CheckStatus.FAIL,
                message="model artifact manifest is invalid or incompatible",
                details={"path": str(manifest_path), "error": str(exc)},
            ),
        ]
    checks = [
        present,
        PreflightCheck(
            name=f"{name}.manifest",
            status=CheckStatus.PASS,
            message="model artifact manifest schema is compatible",
            details={"artifact_id": artifact.artifact_id},
        ),
    ]
    estimator_path = path / artifact.estimator_file
    if not estimator_path.is_file():
        checks.append(
            PreflightCheck(
                name=f"{name}.checksum",
                status=CheckStatus.FAIL,
                message="model estimator file is missing",
                details={"path": str(estimator_path)},
            )
        )
    else:
        actual_digest = _sha256(estimator_path)
        checksum_ok = actual_digest == artifact.estimator_sha256
        checks.append(
            PreflightCheck(
                name=f"{name}.checksum",
                status=CheckStatus.PASS if checksum_ok else CheckStatus.FAIL,
                message=(
                    "model estimator checksum matches manifest"
                    if checksum_ok
                    else "model estimator checksum mismatch"
                ),
                details={
                    "expected": artifact.estimator_sha256,
                    "actual": actual_digest,
                },
            )
        )
    expected_targets = sorted({task.training_target_label for task in config.tasks})
    target_ok = expected_targets == [artifact.training_target_label]
    checks.append(
        PreflightCheck(
            name=f"{name}.target_label",
            status=CheckStatus.PASS if target_ok else CheckStatus.FAIL,
            message=(
                "model training label matches every task"
                if target_ok
                else "model training label differs from experiment target label"
            ),
            details={
                "artifact": artifact.training_target_label,
                "experiment": expected_targets,
            },
        )
    )
    preset_ok = artifact.feature_spec.preset.value == condition.feature_set.value
    checks.append(
        PreflightCheck(
            name=f"{name}.feature_spec",
            status=CheckStatus.PASS if preset_ok else CheckStatus.FAIL,
            message=(
                "model feature preset matches the condition"
                if preset_ok
                else "model feature preset differs from the condition"
            ),
            details={
                "artifact": artifact.feature_spec.preset.value,
                "condition": condition.feature_set.value,
            },
        )
    )
    expected_skills = sorted({task.skill_name for task in config.tasks})
    missing_skills = sorted(
        set(expected_skills).difference(artifact.feature_spec.skill_vocabulary)
    )
    checks.append(
        PreflightCheck(
            name=f"{name}.skill_vocabulary",
            status=CheckStatus.FAIL if missing_skills else CheckStatus.PASS,
            message=(
                "model skill vocabulary is missing configured skills"
                if missing_skills
                else "model skill vocabulary covers configured skills"
            ),
            details={"missing": missing_skills},
        )
    )
    if condition.model_artifact_id is not None:
        identity_ok = condition.model_artifact_id == artifact.artifact_id
        checks.append(
            PreflightCheck(
                name=f"{name}.identity",
                status=CheckStatus.PASS if identity_ok else CheckStatus.FAIL,
                message=(
                    "configured model identity matches artifact"
                    if identity_ok
                    else "configured model identity differs from artifact"
                ),
                details={
                    "configured": condition.model_artifact_id,
                    "artifact": artifact.artifact_id,
                },
            )
        )
    return checks


def _handoff_config_checks(
    path: Path,
    *,
    condition_index: int,
    condition,
) -> list[PreflightCheck]:
    name = f"handoff_config[{condition_index}]"
    present = _path_check(name, path, expected="file")
    if present.status is CheckStatus.FAIL:
        return [present]
    try:
        from robots.libero.handoff_runtime import load_handoff_runtime_config

        load_strict_json(path)
        runtime_config = load_handoff_runtime_config(path)
    except Exception as exc:
        return [
            present,
            PreflightCheck(
                name=f"{name}.schema",
                status=CheckStatus.FAIL,
                message="handoff runtime config is invalid",
                details={"path": str(path), "error": str(exc)},
            ),
        ]
    checks = [
        present,
        PreflightCheck(
            name=f"{name}.schema",
            status=CheckStatus.PASS,
            message="handoff runtime config is strict and valid",
            details={"configuration_id": runtime_config.configuration_id},
        ),
    ]
    enabled_ok = runtime_config.enabled and condition.handoff_enabled
    checks.append(
        PreflightCheck(
            name=f"{name}.enabled",
            status=CheckStatus.PASS if enabled_ok else CheckStatus.FAIL,
            message=(
                "handoff is explicitly enabled in both matrix and runtime config"
                if enabled_ok
                else "matrix/runtime handoff enablement disagrees"
            ),
        )
    )
    method_ok = runtime_config.controller_method == condition.method
    checks.append(
        PreflightCheck(
            name=f"{name}.method",
            status=CheckStatus.PASS if method_ok else CheckStatus.FAIL,
            message=(
                "controller method matches condition"
                if method_ok
                else "runtime controller method differs from matrix condition"
            ),
            details={
                "runtime": runtime_config.controller_method,
                "condition": condition.method,
            },
        )
    )
    if condition.checkpoint_id is not None:
        checkpoint_ok = runtime_config.checkpoint_id == condition.checkpoint_id
        checks.append(
            PreflightCheck(
                name=f"{name}.checkpoint",
                status=CheckStatus.PASS if checkpoint_ok else CheckStatus.FAIL,
                message=(
                    "runtime checkpoint identity matches condition"
                    if checkpoint_ok
                    else "runtime checkpoint identity differs from condition"
                ),
                details={
                    "runtime": runtime_config.checkpoint_id,
                    "condition": condition.checkpoint_id,
                },
            )
        )
    return checks


def _output_root_check(root: Path) -> PreflightCheck:
    resolved = root.resolve()
    if resolved == Path(resolved.anchor):
        return PreflightCheck(
            name="output_root",
            status=CheckStatus.FAIL,
            message="output root cannot be a filesystem root",
            details={"path": str(resolved)},
        )
    if resolved.exists() and not resolved.is_dir():
        return PreflightCheck(
            name="output_root",
            status=CheckStatus.FAIL,
            message="output root exists but is not a directory",
            details={"path": str(resolved)},
        )
    parent = _nearest_existing_parent(resolved)
    if not parent.exists() or not parent.is_dir():
        return PreflightCheck(
            name="output_root",
            status=CheckStatus.FAIL,
            message="output root has no existing directory ancestor",
            details={"path": str(resolved)},
        )
    if not os.access(parent, os.W_OK):
        return PreflightCheck(
            name="output_root",
            status=CheckStatus.FAIL,
            message="nearest existing output ancestor is not writable",
            details={"path": str(resolved), "ancestor": str(parent)},
        )
    return PreflightCheck(
        name="output_root",
        status=CheckStatus.PASS,
        message="output root is structurally safe and has a writable ancestor",
        details={"path": str(resolved), "ancestor": str(parent)},
    )


def run_offline_preflight(
    config: ExperimentConfig,
    manifest: ExperimentManifest | None = None,
    *,
    config_path: str | os.PathLike[str] | None = None,
    require_referenced_paths: bool = True,
) -> PreflightReport:
    """Run deterministic checks that require no server, GPU, or rollout.

    This function performs no writes. Runtime capability checks belong to the
    separate server probe surface.
    """
    expected_manifest = expand_manifest(config, config_path=config_path)
    resolved_manifest = manifest or expected_manifest
    checks: list[PreflightCheck] = []

    if resolved_manifest.configuration_id == config.configuration_id:
        checks.append(
            PreflightCheck(
                name="configuration_identity",
                status=CheckStatus.PASS,
                message="manifest matches normalized configuration identity",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="configuration_identity",
                status=CheckStatus.FAIL,
                message="manifest configuration identity is stale or mismatched",
                details={
                    "expected": config.configuration_id,
                    "actual": resolved_manifest.configuration_id,
                },
            )
        )

    expected_trials = [
        trial.model_dump(mode="json", exclude_none=False)
        for trial in expected_manifest.trials
    ]
    actual_trials = [
        trial.model_dump(mode="json", exclude_none=False)
        for trial in resolved_manifest.trials
    ]
    expansion_matches = (
        resolved_manifest.manifest_id == expected_manifest.manifest_id
        and resolved_manifest.experiment_id == expected_manifest.experiment_id
        and resolved_manifest.source_config_path
        == expected_manifest.source_config_path
        and actual_trials == expected_trials
    )
    checks.append(
        PreflightCheck(
            name="manifest_expansion",
            status=(CheckStatus.PASS if expansion_matches else CheckStatus.FAIL),
            message=(
                "complete resolved manifest is deterministic and current"
                if expansion_matches
                else "manifest payload differs from deterministic config expansion"
            ),
            details={
                "expected_manifest_id": expected_manifest.manifest_id,
                "actual_manifest_id": resolved_manifest.manifest_id,
                "expected_trials": len(expected_trials),
                "actual_trials": len(actual_trials),
            },
        )
    )

    task_bound_cells = {
        (trial.task.suite, trial.task.task, trial.task.seed)
        for trial in resolved_manifest.trials
    }
    if config.runtime.env_endpoint is not None and len(task_bound_cells) > 1:
        checks.append(
            PreflightCheck(
                name="task_bound_env_endpoint",
                status=CheckStatus.FAIL,
                message=(
                    "one external LIBERO env endpoint cannot satisfy multiple "
                    "suite/task/seed identities without an explicit external rotator"
                ),
                details={"task_cells": len(task_bound_cells)},
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="task_bound_env_endpoint",
                status=CheckStatus.PASS,
                message="env ownership is compatible with resolved task cells",
                details={"task_cells": len(task_bound_cells)},
            )
        )

    output_root = Path(config.output_root).expanduser()
    if not output_root.is_absolute() and config_path is not None:
        output_root = Path(config_path).resolve().parent / output_root
    output_root = output_root.resolve()
    checks.append(_output_root_check(output_root))
    escaped_outputs = [
        trial.trial_id
        for trial in resolved_manifest.trials
        if not Path(trial.output_dir).resolve().is_relative_to(output_root)
    ]
    checks.append(
        PreflightCheck(
            name="trial_output_isolation",
            status=(CheckStatus.FAIL if escaped_outputs else CheckStatus.PASS),
            message=(
                "one or more trial outputs escape output_root"
                if escaped_outputs
                else "all trial outputs are unique and contained by output_root"
            ),
            details={"escaped_trial_ids": escaped_outputs},
        )
    )

    privileged_conditions = [
        condition.name
        for condition in config.conditions
        if any(not item.online_allowed for item in condition.policy_feature_availability)
    ]
    checks.append(
        PreflightCheck(
            name="privileged_feature_firewall",
            status=(CheckStatus.FAIL if privileged_conditions else CheckStatus.PASS),
            message=(
                "deployment policy feature sets include privileged/setup provenance"
                if privileged_conditions
                else "all configured policy provenance is deployment-allowed"
            ),
            details={"conditions": privileged_conditions},
        )
    )

    baseline_conditions = [
        condition
        for condition in config.conditions
        if condition.method == "original_harness"
    ]
    bad_baselines = [
        condition.name
        for condition in baseline_conditions
        if condition.execution_layer is not ExecutionLayer.FULL_AGENT
        or condition.handoff_enabled
        or condition.handoff_config is not None
    ]
    if bad_baselines:
        baseline_status = CheckStatus.FAIL
        baseline_message = "Original Harness baseline is contaminated by handoff settings"
    elif baseline_conditions:
        baseline_status = CheckStatus.PASS
        baseline_message = "Original Harness conditions have no handoff configuration"
    else:
        baseline_status = CheckStatus.WARNING
        baseline_message = "matrix does not include an Original Harness condition"
    checks.append(
        PreflightCheck(
            name="original_harness_isolation",
            status=baseline_status,
            message=baseline_message,
            details={"invalid_conditions": bad_baselines},
        )
    )

    if require_referenced_paths:
        for task_index, task in enumerate(config.tasks):
            if task.task_config is not None:
                checks.append(
                    _path_check(
                        f"task_config[{task_index}]",
                        resolve_reference(task.task_config, config_path=config_path),
                        expected="file",
                    )
                )
        for condition_index, condition in enumerate(config.conditions):
            if condition.handoff_config is not None:
                checks.extend(
                    _handoff_config_checks(
                        resolve_reference(
                            condition.handoff_config,
                            config_path=config_path,
                        ),
                        condition_index=condition_index,
                        condition=condition,
                    )
                )
            if condition.model_artifact is not None:
                checks.extend(
                    _model_artifact_checks(
                        resolve_reference(
                            condition.model_artifact,
                            config_path=config_path,
                        ),
                        condition_index=condition_index,
                        condition=condition,
                        config=config,
                    )
                )

    for variable, endpoint, configured_path in (
        (
            "PI05_CHECKPOINT_PATH",
            config.runtime.vla_endpoint,
            config.runtime.pi05_checkpoint_path,
        ),
        (
            "SAM3_CHECKPOINT_PATH",
            config.runtime.sam3_endpoint,
            config.runtime.sam3_checkpoint_path,
        ),
    ):
        component = "pi05_runtime" if variable.startswith("PI05") else "sam3_runtime"
        if endpoint is not None:
            checks.append(
                PreflightCheck(
                    name=component,
                    status=CheckStatus.PASS,
                    message="external endpoint configured; checkpoint is server-owned",
                )
            )
            continue
        environment_path = os.environ.get(variable)
        path_value = configured_path or environment_path
        if not path_value:
            checks.append(
                PreflightCheck(
                    name=component,
                    status=CheckStatus.FAIL,
                    message=f"local service needs runtime.{component.split('_')[0]} checkpoint or {variable}",
                )
            )
            continue
        checkpoint = (
            resolve_reference(path_value, config_path=config_path)
            if configured_path is not None
            else Path(path_value).expanduser().resolve()
        )
        checks.append(_path_check(component, checkpoint))

    for component, checkpoint_id in (
        ("pi05_checkpoint_identity", config.runtime.pi05_checkpoint_id),
        ("sam3_checkpoint_identity", config.runtime.sam3_checkpoint_id),
    ):
        checks.append(
            PreflightCheck(
                name=component,
                status=(
                    CheckStatus.PASS
                    if checkpoint_id is not None
                    else CheckStatus.FAIL
                ),
                message=(
                    "expected checkpoint identity is configuration-bound"
                    if checkpoint_id is not None
                    else "runtime checkpoint identity must be configured and probe-verified"
                ),
                details={"checkpoint_id": checkpoint_id},
            )
        )

    return PreflightReport(
        configuration_id=config.configuration_id,
        manifest_id=resolved_manifest.manifest_id,
        checks=tuple(checks),
    )


__all__ = [
    "CheckStatus",
    "PreflightCheck",
    "PreflightReport",
    "run_offline_preflight",
]
