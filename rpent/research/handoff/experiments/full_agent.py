"""Deterministic child-process plans for full RPent/Harness experiments."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, Sequence

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.experiments.config import (
    ExecutionLayer,
    stable_digest,
    stable_identifier,
)
from rpent.research.handoff.experiments.manifest import TrialManifest
from rpent.research.handoff.types import HandoffRecord

CHILD_PLAN_SCHEMA_VERSION = "rpent.handoff-full-agent-plan/v2"
FULL_AGENT_PYTHON_EXECUTABLE_ENV = "RPENT_FULL_AGENT_PYTHON_EXECUTABLE"
_BOUND_PLAN_ID_TOKEN = "__RPENT_BOUND_PLAN_ID__"


def _option_value(command: Sequence[str], option: str) -> str:
    positions = [index for index, token in enumerate(command) if token == option]
    if len(positions) != 1:
        raise ValueError(f"command must contain {option!r} exactly once")
    position = positions[0]
    if position + 1 >= len(command):
        raise ValueError(f"command option {option!r} lacks a value")
    return command[position + 1]


def _replace_option_value(
    command: Sequence[str],
    option: str,
    value: str,
) -> tuple[str, ...]:
    position = tuple(command).index(option)
    if tuple(command).count(option) != 1 or position + 1 >= len(command):
        raise ValueError(f"command must contain valued option {option!r} exactly once")
    updated = list(command)
    updated[position + 1] = value
    return tuple(updated)


def _child_plan_identity_payload(
    *,
    trial_id: str,
    manifest_id: str,
    manifest_path: str,
    wrapper_command: Sequence[str],
    resolved_inner_command: Sequence[str],
    cwd: str,
    env_overrides: dict[str, str],
    output_dir: str,
    original_harness: bool,
) -> dict[str, object]:
    """Normalize the plan's self-referential ID tokens before hashing."""
    return {
        "schema_version": CHILD_PLAN_SCHEMA_VERSION,
        "trial_id": trial_id,
        "manifest_id": manifest_id,
        "manifest_path": manifest_path,
        "wrapper_command": _replace_option_value(
            wrapper_command, "--plan-id", _BOUND_PLAN_ID_TOKEN
        ),
        "resolved_inner_command": _replace_option_value(
            resolved_inner_command,
            "--research-plan-id",
            _BOUND_PLAN_ID_TOKEN,
        ),
        "cwd": cwd,
        "env_overrides": dict(sorted(env_overrides.items())),
        "output_dir": output_dir,
        "original_harness": original_harness,
    }


def _resolve_python_executable(value: str | None) -> str:
    requested = value or sys.executable
    discovered = shutil.which(requested)
    if discovered is None:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            raise ValueError(f"python executable cannot be resolved: {requested!r}")
        discovered = str(path)
    return os.path.abspath(os.path.expanduser(discovered))


class FullAgentChildPlan(HandoffRecord):
    """One wrapper invocation plus its exact, identity-bound RPent command."""

    schema_version: Literal[CHILD_PLAN_SCHEMA_VERSION] = CHILD_PLAN_SCHEMA_VERSION
    plan_id: str
    trial_id: str
    manifest_id: str
    manifest_path: str
    wrapper_command: tuple[str, ...]
    resolved_inner_command: tuple[str, ...]
    cwd: str
    env_overrides: dict[str, str] = Field(default_factory=dict)
    output_dir: str
    original_harness: bool

    @field_validator(
        "plan_id",
        "trial_id",
        "manifest_id",
        "manifest_path",
        "cwd",
        "output_dir",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("wrapper_command", "resolved_inner_command")
    @classmethod
    def validate_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not token or "\x00" in token for token in value):
            raise ValueError("command must contain non-empty safe argument tokens")
        return value

    @model_validator(mode="after")
    def validate_baseline(self):
        if self.wrapper_command == self.resolved_inner_command:
            raise ValueError("wrapper and resolved inner commands must be distinct")
        expected_wrapper_tail = (
            "-m",
            "rpent.research.handoff",
            "--traceback",
            "_full-agent-child",
            "--manifest",
            self.manifest_path,
            "--trial-id",
            self.trial_id,
            "--plan-id",
            self.plan_id,
        )
        if self.wrapper_command[1:] != expected_wrapper_tail:
            raise ValueError("wrapper command is not the exact research child entrypoint")
        if self.resolved_inner_command[:3] != (
            self.wrapper_command[0],
            "-m",
            "rpent.cli.main",
        ):
            raise ValueError("inner command is not the bound RPent CLI entrypoint")
        if (
            self.env_overrides.get(FULL_AGENT_PYTHON_EXECUTABLE_ENV)
            != self.wrapper_command[0]
        ):
            raise ValueError("plan environment does not bind its Python executable")
        if _option_value(self.wrapper_command, "--plan-id") != self.plan_id:
            raise ValueError("wrapper command plan ID disagrees with plan record")
        if _option_value(self.wrapper_command, "--trial-id") != self.trial_id:
            raise ValueError("wrapper command trial ID disagrees with plan record")
        if Path(_option_value(self.wrapper_command, "--manifest")).resolve() != Path(
            self.manifest_path
        ).resolve():
            raise ValueError("wrapper command manifest path disagrees with plan record")
        if _option_value(
            self.resolved_inner_command, "--research-plan-id"
        ) != self.plan_id:
            raise ValueError("inner command plan ID disagrees with plan record")
        if _option_value(
            self.resolved_inner_command, "--research-trial-id"
        ) != self.trial_id:
            raise ValueError("inner command trial ID disagrees with plan record")
        if _option_value(
            self.resolved_inner_command, "--research-manifest-id"
        ) != self.manifest_id:
            raise ValueError("inner command manifest ID disagrees with plan record")
        if Path(
            _option_value(
                self.resolved_inner_command, "--research-manifest-path"
            )
        ).resolve() != Path(self.manifest_path).resolve():
            raise ValueError("inner command manifest path disagrees with plan record")
        if _option_value(
            self.resolved_inner_command, "--output-dir"
        ) != self.output_dir:
            raise ValueError("inner command output directory disagrees with plan record")
        output_path = Path(self.output_dir)
        for option, filename in (
            ("--research-reset-identity-output", "reset_identity.json"),
            ("--research-completion-output", "completion.json"),
            ("--research-runtime-identity-output", "runtime_identity.json"),
        ):
            if _option_value(self.resolved_inner_command, option) != str(
                output_path / filename
            ):
                raise ValueError(f"inner command {option} is not output-local")
        if self.original_harness and "--handoff-config" in self.resolved_inner_command:
            raise ValueError("Original Harness inner command contains handoff configuration")
        expected_plan_id = stable_identifier(
            "child",
            _child_plan_identity_payload(
                trial_id=self.trial_id,
                manifest_id=self.manifest_id,
                manifest_path=self.manifest_path,
                wrapper_command=self.wrapper_command,
                resolved_inner_command=self.resolved_inner_command,
                cwd=self.cwd,
                env_overrides=self.env_overrides,
                output_dir=self.output_dir,
                original_harness=self.original_harness,
            ),
        )
        if self.plan_id != expected_plan_id:
            raise ValueError("plan ID does not bind the wrapper and inner commands")
        return self

    @property
    def command(self) -> tuple[str, ...]:
        """Compatibility view; persisted plans name this wrapper_command."""
        return self.wrapper_command


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
    research_manifest_path: str | os.PathLike[str] | None = None,
    research_manifest_id: str | None = None,
    research_plan_id: str | None = None,
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
        "--research-runtime-identity-output",
        str(Path(trial.output_dir) / "runtime_identity.json"),
        "--planner",
        trial.planner.backend,
        "--max-turns",
        str(trial.planner.max_turns),
        "--max-tokens",
        str(trial.planner.max_tokens),
    ]
    research_values = (
        research_manifest_path,
        research_manifest_id,
        research_plan_id,
    )
    if any(value is not None for value in research_values):
        if not all(value is not None for value in research_values):
            raise ValueError(
                "research manifest path, manifest ID, and plan ID are jointly required"
            )
        command.extend(
            (
                "--research-manifest-path",
                str(Path(str(research_manifest_path)).expanduser().resolve()),
                "--research-manifest-id",
                str(research_manifest_id),
                "--research-plan-id",
                str(research_plan_id),
            )
        )
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
    manifest_path: str | os.PathLike[str],
    repo_root: str | os.PathLike[str],
    python_executable: str | None = None,
) -> FullAgentChildPlan:
    """Create a plan binding both the validator wrapper and exact RPent child."""
    from rpent.research.handoff.experiments.manifest import load_manifest

    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = load_manifest(resolved_manifest_path)
    matches = [item for item in manifest.trials if item.trial_id == trial.trial_id]
    if len(matches) != 1 or matches[0] != trial:
        raise ValueError("trial does not exactly match its bound manifest entry")

    executable = _resolve_python_executable(python_executable)
    wrapper_template = (
        executable,
        "-m",
        "rpent.research.handoff",
        "--traceback",
        "_full-agent-child",
        "--manifest",
        str(resolved_manifest_path),
        "--trial-id",
        trial.trial_id,
        "--plan-id",
        _BOUND_PLAN_ID_TOKEN,
    )
    env_overrides: dict[str, str] = {
        FULL_AGENT_PYTHON_EXECUTABLE_ENV: executable,
    }
    if trial.runtime.pi05_checkpoint_path is not None:
        env_overrides["PI05_CHECKPOINT_PATH"] = trial.runtime.pi05_checkpoint_path
    if trial.runtime.sam3_checkpoint_path is not None:
        env_overrides["SAM3_CHECKPOINT_PATH"] = trial.runtime.sam3_checkpoint_path
    if trial.runtime.pi05_checkpoint_id is not None:
        env_overrides["RPENT_PI05_CHECKPOINT_ID"] = trial.runtime.pi05_checkpoint_id
    if trial.runtime.sam3_checkpoint_id is not None:
        env_overrides["RPENT_SAM3_CHECKPOINT_ID"] = trial.runtime.sam3_checkpoint_id
    cwd = str(Path(repo_root).expanduser().resolve())
    execution_trial = trial
    if trial.condition.handoff_enabled:
        output_path = Path(trial.output_dir)
        if not output_path.is_absolute():
            output_path = Path(cwd) / output_path
        execution_trial = trial.model_copy(
            update={
                "handoff_config_path": str(
                    (output_path / "resolved_handoff_runtime.json").resolve()
                )
            }
        )
    inner_template = build_full_agent_command(
        execution_trial,
        python_executable=executable,
        research_manifest_path=resolved_manifest_path,
        research_manifest_id=manifest.manifest_id,
        research_plan_id=_BOUND_PLAN_ID_TOKEN,
    )
    plan_payload = _child_plan_identity_payload(
        trial_id=trial.trial_id,
        manifest_id=manifest.manifest_id,
        manifest_path=str(resolved_manifest_path),
        wrapper_command=wrapper_template,
        resolved_inner_command=inner_template,
        cwd=cwd,
        env_overrides=env_overrides,
        output_dir=trial.output_dir,
        original_harness=trial.condition.method == "original_harness",
    )
    plan_id = stable_identifier("child", plan_payload)
    wrapper_command = _replace_option_value(
        wrapper_template, "--plan-id", plan_id
    )
    resolved_inner_command = _replace_option_value(
        inner_template, "--research-plan-id", plan_id
    )
    return FullAgentChildPlan(
        plan_id=plan_id,
        trial_id=trial.trial_id,
        manifest_id=manifest.manifest_id,
        manifest_path=str(resolved_manifest_path),
        wrapper_command=wrapper_command,
        resolved_inner_command=resolved_inner_command,
        cwd=cwd,
        env_overrides=env_overrides,
        output_dir=trial.output_dir,
        original_harness=trial.condition.method == "original_harness",
    )


def plan_full_agent_trials(
    trials: Sequence[TrialManifest],
    *,
    manifest_path: str | os.PathLike[str],
    repo_root: str | os.PathLike[str],
    python_executable: str | None = None,
) -> tuple[FullAgentChildPlan, ...]:
    """Create manifest-ordered child plans; non-full-agent trials fail closed."""
    return tuple(
        build_child_plan(
            trial,
            manifest_path=manifest_path,
            repo_root=repo_root,
            python_executable=python_executable,
        )
        for trial in trials
    )


def write_child_plans(
    plans: Sequence[FullAgentChildPlan],
    path: str | os.PathLike[str],
) -> Path:
    """Immutably persist dry-run plans for review before authorization."""
    if not plans:
        raise ValueError("refusing to write an empty full-agent plan")
    plans = tuple(
        FullAgentChildPlan.model_validate(
            plan.model_dump(mode="python", exclude_none=False)
        )
        for plan in plans
    )
    plan_ids = [plan.plan_id for plan in plans]
    trial_ids = [plan.trial_id for plan in plans]
    if len(plan_ids) != len(set(plan_ids)):
        raise ValueError("refusing to write duplicate full-agent plan IDs")
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("refusing to write duplicate full-agent trial IDs")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(
        [plan.model_dump(mode="json", exclude_none=False) for plan in plans],
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != canonical:
            raise FileExistsError(
                f"immutable full-agent plan artifact differs: {destination}"
            )
        return destination
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical)
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def load_child_plans(
    path: str | os.PathLike[str],
) -> tuple[FullAgentChildPlan, ...]:
    """Strictly load identity-validated plans, rejecting ambiguous JSON."""
    source = Path(path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {source}: {key!r}")
            result[key] = value
        return result

    try:
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(
                stream,
                parse_constant=reject_constant,
                object_pairs_hook=object_pairs,
            )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid child-plan JSON in {source}: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("child-plan artifact must be a non-empty JSON array")
    plans = tuple(FullAgentChildPlan.model_validate(item) for item in payload)
    plan_ids = [plan.plan_id for plan in plans]
    trial_ids = [plan.trial_id for plan in plans]
    if len(plan_ids) != len(set(plan_ids)):
        raise ValueError("child-plan artifact contains duplicate plan IDs")
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("child-plan artifact contains duplicate trial IDs")
    return plans


def _write_immutable_json(path: Path, payload: dict[str, object]) -> Path:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != canonical:
            raise FileExistsError(f"immutable execution artifact differs: {path}")
        return path
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def run_full_agent_trial(
    manifest_path: str | os.PathLike[str],
    trial_id: str,
    plan_id: str,
) -> int:
    """Validate and launch one manifest-derived RPent command in this wrapper."""
    from rpent.research.handoff.experiments.manifest import (
        load_manifest,
        verify_manifest_external_bindings,
    )
    from rpent.research.handoff.experiments.runtime import (
        write_resolved_handoff_config,
    )

    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = load_manifest(resolved_manifest_path)
    verify_manifest_external_bindings(
        manifest,
        repo_root=Path.cwd(),
        require_runtime_probes=True,
    )
    matches = [trial for trial in manifest.trials if trial.trial_id == trial_id]
    if len(matches) != 1:
        raise ValueError(
            f"manifest must contain exactly one full-agent trial {trial_id!r}"
        )
    trial = matches[0]
    if trial.execution_layer is not ExecutionLayer.FULL_AGENT:
        raise ValueError(f"trial is not full_agent: {trial_id}")
    bound_python = os.environ.get(FULL_AGENT_PYTHON_EXECUTABLE_ENV)
    if not bound_python:
        raise ValueError("full-agent wrapper lacks its bound Python executable")
    try:
        same_python = os.path.samefile(bound_python, sys.executable)
    except OSError as exc:
        raise ValueError("bound Python executable cannot be verified") from exc
    if not same_python:
        raise ValueError("wrapper is not running under its bound Python executable")
    expected_plan = build_child_plan(
        trial,
        manifest_path=resolved_manifest_path,
        repo_root=Path.cwd(),
        python_executable=bound_python,
    )
    if expected_plan.plan_id != plan_id:
        raise ValueError("full-agent plan ID does not bind this invocation")
    if expected_plan.manifest_id != manifest.manifest_id:
        raise ValueError("full-agent plan does not bind the loaded manifest")
    if Path.cwd().resolve() != Path(expected_plan.cwd).resolve():
        raise ValueError("full-agent wrapper cwd disagrees with its plan")

    output_dir = Path(trial.output_dir)
    stale_files = (
        tuple(path for path in output_dir.rglob("*") if path.is_file())
        if output_dir.exists()
        else ()
    )
    if stale_files:
        raise FileExistsError(
            "full-agent output already contains an attempt; preserve it and "
            "allocate a new trial identity: "
            + ", ".join(str(path) for path in stale_files[:20])
        )
    _write_immutable_json(
        output_dir / "attempt.json",
        {
            "schema_version": "rpent.handoff-full-agent-attempt/v1",
            "trial_id": trial.trial_id,
            "manifest_id": manifest.manifest_id,
            "plan_id": plan_id,
            "source_revision": trial.source_revision,
            "cwd": str(Path.cwd().resolve()),
            "resolved_inner_command_sha256": stable_digest(
                expected_plan.resolved_inner_command
            ),
        },
    )
    execution_trial = trial
    if trial.condition.handoff_enabled:
        resolved_handoff_path = write_resolved_handoff_config(
            trial,
            execution_plan_id=plan_id,
            manifest_id=manifest.manifest_id,
        )
        execution_trial = trial.model_copy(
            update={"handoff_config_path": str(resolved_handoff_path.resolve())}
        )
    inner_command = build_full_agent_command(
        execution_trial,
        python_executable=expected_plan.resolved_inner_command[0],
        research_manifest_path=resolved_manifest_path,
        research_manifest_id=manifest.manifest_id,
        research_plan_id=plan_id,
    )
    if inner_command != expected_plan.resolved_inner_command:
        raise RuntimeError(
            "execution-time RPent command differs from the identity-bound plan"
        )
    environment = os.environ.copy()
    environment.update(expected_plan.env_overrides)
    os.chdir(expected_plan.cwd)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvpe(inner_command[0], list(inner_command), environment)
    raise RuntimeError("os.execvpe unexpectedly returned")


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
    plan = FullAgentChildPlan.model_validate(
        plan.model_dump(mode="python", exclude_none=False)
    )
    environment = os.environ.copy()
    environment.update(plan.env_overrides)
    completed = subprocess.run(
        list(plan.wrapper_command),
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
    "FULL_AGENT_PYTHON_EXECUTABLE_ENV",
    "build_child_plan",
    "build_full_agent_command",
    "execute_child_plan",
    "load_child_plans",
    "plan_full_agent_trials",
    "run_full_agent_trial",
    "write_child_plans",
]
