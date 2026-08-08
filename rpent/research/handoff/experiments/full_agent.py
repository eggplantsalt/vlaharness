"""Deterministic child-process plans for full RPent/Harness experiments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal, Sequence

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.experiments.config import ExecutionLayer, stable_identifier
from rpent.research.handoff.experiments.manifest import TrialManifest
from rpent.research.handoff.types import HandoffRecord

CHILD_PLAN_SCHEMA_VERSION = "rpent.handoff-full-agent-plan/v1"


class FullAgentChildPlan(HandoffRecord):
    """One shell-free subprocess invocation; construction performs no execution."""

    schema_version: Literal[CHILD_PLAN_SCHEMA_VERSION] = CHILD_PLAN_SCHEMA_VERSION
    plan_id: str
    trial_id: str
    command: tuple[str, ...]
    cwd: str
    env_overrides: dict[str, str] = Field(default_factory=dict)
    output_dir: str
    original_harness: bool

    @field_validator("plan_id", "trial_id", "cwd", "output_dir")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not token or "\x00" in token for token in value):
            raise ValueError("command must contain non-empty safe argument tokens")
        return value

    @model_validator(mode="after")
    def validate_baseline(self):
        if self.original_harness and "--handoff-config" in self.command:
            raise ValueError("Original Harness command contains handoff configuration")
        return self


class FullAgentChildResult(HandoffRecord):
    """Serializable result of an explicitly authorized child run."""

    plan_id: str
    trial_id: str
    returncode: int
    stdout: str | None = None
    stderr: str | None = None


def _append_option(command: list[str], name: str, value: object | None) -> None:
    if value is not None:
        command.extend((name, str(value)))


def build_full_agent_command(
    trial: TrialManifest,
    *,
    python_executable: str | None = None,
) -> tuple[str, ...]:
    """Build the existing RPent CLI command for one full-agent trial.

    Original Harness is represented by the complete absence of
    ``--handoff-config``. Research behavior is opt-in and receives exactly one
    resolved configuration path.
    """
    if trial.execution_layer is not ExecutionLayer.FULL_AGENT:
        raise ValueError(f"trial is not full_agent: {trial.trial_id}")
    executable = python_executable or sys.executable
    command = [
        executable,
        "-m",
        "rpent.cli.main",
        "--env",
        trial.runtime.env_name,
        "--suite",
        trial.task.suite,
        "--task",
        str(trial.task.task),
        "--seed",
        str(trial.task.seed),
        "--libero-type",
        trial.runtime.libero_type,
        "--max-episode-steps",
        str(trial.runtime.max_episode_steps),
        "--output-dir",
        trial.output_dir,
        "--research-trial-id",
        trial.trial_id,
        "--research-reset-identity-output",
        str(Path(trial.output_dir) / "reset_identity.json"),
        "--research-completion-output",
        str(Path(trial.output_dir) / "completion.json"),
        "--planner",
        trial.planner.backend,
        "--max-turns",
        str(trial.planner.max_turns),
        "--max-tokens",
        str(trial.planner.max_tokens),
    ]
    _append_option(command, "--model", trial.planner.model)
    _append_option(command, "--base-url", trial.planner.base_url)
    _append_option(command, "--planner-timeout-s", trial.planner.planner_timeout_s)
    _append_option(
        command,
        "--claude-code-max-budget-usd",
        trial.planner.claude_code_max_budget_usd,
    )
    if trial.planner.no_images:
        command.append("--no-images")
    _append_option(command, "--env-endpoint", trial.runtime.env_endpoint)
    _append_option(command, "--vla-endpoint", trial.runtime.vla_endpoint)
    _append_option(command, "--sam3-endpoint", trial.runtime.sam3_endpoint)
    _append_option(command, "--cuda-device", trial.runtime.cuda_device)
    command.extend(trial.condition.extra_cli_args)
    if trial.condition.handoff_enabled:
        if trial.handoff_config_path is None:
            raise ValueError("handoff-enabled full-agent trial lacks resolved config")
        command.extend(("--handoff-config", trial.handoff_config_path))
    elif "--handoff-config" in command:
        raise ValueError("disabled condition unexpectedly contains handoff config")
    return tuple(command)


def build_child_plan(
    trial: TrialManifest,
    *,
    repo_root: str | os.PathLike[str],
    python_executable: str | None = None,
) -> FullAgentChildPlan:
    """Create a serializable dry-run plan without touching the filesystem."""
    command = build_full_agent_command(
        trial,
        python_executable=python_executable,
    )
    env_overrides: dict[str, str] = {}
    if trial.runtime.pi05_checkpoint_path is not None:
        env_overrides["PI05_CHECKPOINT_PATH"] = trial.runtime.pi05_checkpoint_path
    if trial.runtime.sam3_checkpoint_path is not None:
        env_overrides["SAM3_CHECKPOINT_PATH"] = trial.runtime.sam3_checkpoint_path
    plan_payload = {
        "trial_id": trial.trial_id,
        "command": command,
        "cwd": str(Path(repo_root).resolve()),
        "env_overrides": env_overrides,
    }
    return FullAgentChildPlan(
        plan_id=stable_identifier("child", plan_payload),
        trial_id=trial.trial_id,
        command=command,
        cwd=str(Path(repo_root).resolve()),
        env_overrides=env_overrides,
        output_dir=trial.output_dir,
        original_harness=trial.condition.method == "original_harness",
    )


def plan_full_agent_trials(
    trials: Sequence[TrialManifest],
    *,
    repo_root: str | os.PathLike[str],
    python_executable: str | None = None,
) -> tuple[FullAgentChildPlan, ...]:
    """Create manifest-ordered child plans; non-full-agent trials fail closed."""
    return tuple(
        build_child_plan(
            trial,
            repo_root=repo_root,
            python_executable=python_executable,
        )
        for trial in trials
    )


def write_child_plans(
    plans: Sequence[FullAgentChildPlan],
    path: str | os.PathLike[str],
) -> Path:
    """Atomically persist dry-run plans for review before authorization."""
    if not plans:
        raise ValueError("refusing to write an empty full-agent plan")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(
            [plan.model_dump(mode="json", exclude_none=False) for plan in plans],
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


def execute_child_plan(
    plan: FullAgentChildPlan,
    *,
    allow_execution: bool = False,
    capture_output: bool = False,
    timeout_s: float | None = None,
) -> FullAgentChildResult:
    """Execute only after an explicit opt-in; never invokes a shell."""
    if not allow_execution:
        raise PermissionError(
            "full-agent child execution is disabled by default; "
            "pass allow_execution=True after reviewing the persisted plan"
        )
    environment = os.environ.copy()
    environment.update(plan.env_overrides)
    completed = subprocess.run(
        list(plan.command),
        cwd=plan.cwd,
        env=environment,
        shell=False,
        check=False,
        capture_output=capture_output,
        text=capture_output,
        timeout=timeout_s,
    )
    return FullAgentChildResult(
        plan_id=plan.plan_id,
        trial_id=plan.trial_id,
        returncode=completed.returncode,
        stdout=completed.stdout if capture_output else None,
        stderr=completed.stderr if capture_output else None,
    )


__all__ = [
    "FullAgentChildPlan",
    "FullAgentChildResult",
    "build_child_plan",
    "build_full_agent_command",
    "execute_child_plan",
    "plan_full_agent_trials",
    "write_child_plans",
]
