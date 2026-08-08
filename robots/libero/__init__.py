"""LIBERO environment extension."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.libero.prompt_bundle import system_prompt, user_prompt
from robots.libero.spec import LIBERO_DASHBOARD_SPEC
from rpent.dashboard.events import DashboardEventSink, RuntimeStatusEvent
from rpent.envs.env_spec import EnvSpec, RunConfig
from rpent.envs.prompt_bundle import PromptBundle
from rpent.utils.config import get_repo_root

if TYPE_CHECKING:
    from robots.libero.handoff_runtime import HandoffRuntimeConfig
    from rpent.utils.daemon import ProcessDaemon
    from rpent.utils.rpc import RpcClient


_HANDOFF_CONFIG_ATTR = "_rpent_handoff_runtime_config"
_HANDOFF_CONFIG_PATH_ATTR = "_rpent_handoff_runtime_config_path"
_RESEARCH_RESET_IDENTITY_ATTR = "_rpent_research_reset_identity"
_RESEARCH_RUNTIME_ATTESTATION_ATTR = "_rpent_research_runtime_attestation"
_RESEARCH_RUNTIME_ATTESTATION_SHA256_ATTR = (
    "_rpent_research_runtime_attestation_sha256"
)


def _ensure_handoff_config(args: argparse.Namespace) -> HandoffRuntimeConfig | None:
    """Load/validate opt-in research config before any heavyweight startup."""
    configured_path = getattr(args, "handoff_config", None)
    if configured_path is None:
        setattr(args, _HANDOFF_CONFIG_ATTR, None)
        setattr(args, _HANDOFF_CONFIG_PATH_ATTR, None)
        return None
    resolved = str(Path(configured_path).expanduser().resolve())
    cached_path = getattr(args, _HANDOFF_CONFIG_PATH_ATTR, None)
    if cached_path == resolved:
        return getattr(args, _HANDOFF_CONFIG_ATTR)

    from robots.libero.handoff_runtime import (
        load_handoff_runtime_config,
        validate_handoff_runtime_bindings,
    )

    config = load_handoff_runtime_config(resolved)
    validate_handoff_runtime_bindings(config)
    setattr(args, _HANDOFF_CONFIG_ATTR, config)
    setattr(args, _HANDOFF_CONFIG_PATH_ATTR, resolved)
    return config


def _validate_handoff_output(
    config: HandoffRuntimeConfig | None,
    output_dir: Path,
) -> None:
    if config is None or not config.enabled:
        return
    from robots.libero.handoff_runtime import resolve_handoff_output_dir

    resolve_handoff_output_dir(config, run_output_dir=output_dir)


def _attest_research_runtime(
    args: argparse.Namespace,
    primitives_kwargs: dict[str, Any],
) -> None:
    """Probe and persist live Pi/SAM identity before reset or model action."""
    trial_id = getattr(args, "research_trial_id", None)
    if trial_id is None:
        return
    trial = getattr(args, "_rpent_research_trial_manifest", None)
    if trial is None or trial.trial_id != trial_id:
        raise RuntimeError("research runtime lacks its parser-validated trial")

    from rpent.research.handoff.experiments.runtime_identity import (
        attest_runtime_checkpoint_clients,
        verify_runtime_attestation_binding,
        write_runtime_attestation,
    )

    attestation = attest_runtime_checkpoint_clients(
        primitives_kwargs,
        trial.runtime,
        trial_id=trial.trial_id,
        manifest_id=str(args.research_manifest_id),
        plan_id=str(args.research_plan_id),
        source_revision=trial.source_revision,
    )
    destination = write_runtime_attestation(
        attestation,
        args.research_runtime_identity_output,
    )
    verified, digest = verify_runtime_attestation_binding(
        destination,
        trial_id=trial.trial_id,
        manifest_id=str(args.research_manifest_id),
        plan_id=str(args.research_plan_id),
        source_revision=trial.source_revision,
        expected_attestation_id=attestation.attestation_id,
    )
    setattr(args, _RESEARCH_RUNTIME_ATTESTATION_ATTR, verified)
    setattr(args, _RESEARCH_RUNTIME_ATTESTATION_SHA256_ATTR, digest)


def _research_reset_identity_request(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    trial_id = getattr(args, "research_trial_id", None)
    if trial_id is None:
        return None
    trial = getattr(args, "_rpent_research_trial_manifest", None)
    attestation = getattr(args, _RESEARCH_RUNTIME_ATTESTATION_ATTR, None)
    digest = getattr(args, _RESEARCH_RUNTIME_ATTESTATION_SHA256_ATTR, None)
    if trial is None or attestation is None or not isinstance(digest, str):
        raise RuntimeError(
            "research reset cannot precede live Pi/SAM runtime attestation"
        )
    return {
        "path": str(Path(args.research_reset_identity_output).resolve()),
        "trial_id": str(trial_id),
        "manifest_id": str(args.research_manifest_id),
        "plan_id": str(args.research_plan_id),
        "source_revision": trial.source_revision,
        "suite": str(args.suite),
        "task": int(args.task),
        "seed": int(args.seed),
        "max_episode_steps": int(args.max_episode_steps),
        "runtime_attestation_path": str(
            Path(args.research_runtime_identity_output).resolve()
        ),
        "runtime_attestation_id": attestation.attestation_id,
        "runtime_attestation_sha256": digest,
    }


def get_env_spec() -> EnvSpec:
    """Return the LIBERO env identity, prompt bundle, and runner hooks.

    Tool schemas, handlers, server lifecycle, and the MCP allowlist live on
    the LIBERO toolkit (see :func:`get_toolkit`).
    """
    return EnvSpec(
        name="libero",
        prompts=PromptBundle(
            system=system_prompt,
            user=user_prompt,
        ),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_shared_runtime=init_shared_runtime,
        init_task_runtime=init_task_runtime,
        init_runtime=_init_runtime,
        dashboard=LIBERO_DASHBOARD_SPEC,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    video_path: str | None = None,
):
    """Return the LIBERO toolkit (common tools + LIBERO primitives)."""
    from robots.libero.toolkit import LiberoToolkit

    reset_identity_request = primitives_kwargs.get(
        _RESEARCH_RESET_IDENTITY_ATTR
    )
    if _HANDOFF_CONFIG_ATTR in primitives_kwargs or reset_identity_request is not None:
        primitive_args = dict(primitives_kwargs)
        handoff_config = primitive_args.pop(_HANDOFF_CONFIG_ATTR, None)
        primitive_args.pop(_RESEARCH_RESET_IDENTITY_ATTR, None)
    else:
        # Preserve the exact baseline kwargs object when research is absent.
        primitive_args = primitives_kwargs
        handoff_config = None
    return LiberoToolkit(
        primitives_kwargs=primitive_args,
        dashboard_events=dashboard_events,
        video_path=video_path,
        handoff_config=handoff_config,
        reset_identity_request=reset_identity_request,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    """Register LIBERO CLI flags on the shared ``parser``.

    When ``use_dashboard`` is True, ``--suite`` / ``--task`` are made optional
    because the dashboard launcher will fill them in before ``_parse_config``
    validates. Under CLI-only, they are required — argparse errors out early
    with the usual usage message.
    """
    required = not use_dashboard
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--libero-type", default=None,
                        choices=["standard", "pro", "plus"],
                        help="LIBERO variant (auto-routed from suite suffix if not set).")
    parser.add_argument("--suite", default=None, required=required,
                        help="e.g. libero_object_task, libero_spatial_swap")
    parser.add_argument("--task", type=int, default=None, required=required)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-endpoint", default=None,
                        help="[protocol://]host:port of an existing env_server "
                             "(protocol=http|socket, defaults to http). "
                             "If unset, a local env_server is spawned.")
    parser.add_argument("--vla-endpoint", default=None,
                        help="[protocol://]host:port of an existing vla_server "
                             "(protocol=http|socket, defaults to http). "
                             "If unset, a local vla_server is spawned.")
    parser.add_argument("--sam3-endpoint", default=None,
                        help="[protocol://]host:port of an existing SAM3 server "
                             "(protocol=http|socket, defaults to http). "
                             "If unset, a local SAM3 server is spawned.")
    parser.add_argument("--cuda-device", type=int, default=None,
                        help="GPU device to expose via CUDA_VISIBLE_DEVICES.")
    parser.add_argument(
        "--handoff-config",
        default=None,
        help=(
            "Path to a validated opt-in governor config that routes the original "
            "Pi0 tool schemas through an instance-local research handler. Omit "
            "for Original Harness tool routing."
        ),
    )
    parser.add_argument("--research-trial-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--research-reset-identity-output",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--research-completion-output",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--research-runtime-identity-output",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--research-manifest-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--research-manifest-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--research-plan-id", default=None, help=argparse.SUPPRESS)


def _parse_config(args: argparse.Namespace) -> RunConfig:
    """Validate final ``args`` and derive per-run identifiers.

    Under ``--dashboard``, ``_add_cli_args`` left ``--suite`` / ``--task``
    optional so the dashboard could fill them; this is where we enforce
    they're set now that any overrides have been applied.
    """
    _ensure_handoff_config(args)
    if not args.suite:
        raise ValueError("--suite is required")
    if args.task is None:
        raise ValueError("--task is required")
    research_trial_id = getattr(args, "research_trial_id", None)
    reset_identity_output = getattr(
        args, "research_reset_identity_output", None
    )
    completion_output = getattr(args, "research_completion_output", None)
    runtime_identity_output = getattr(
        args, "research_runtime_identity_output", None
    )
    research_manifest_path = getattr(args, "research_manifest_path", None)
    research_manifest_id = getattr(args, "research_manifest_id", None)
    research_plan_id = getattr(args, "research_plan_id", None)
    if len(
        {
            research_trial_id is None,
            reset_identity_output is None,
            completion_output is None,
            runtime_identity_output is None,
            research_manifest_path is None,
            research_manifest_id is None,
            research_plan_id is None,
        }
    ) != 1:
        raise ValueError(
            "research telemetry requires the complete hidden manifest/plan argument set"
        )

    recipe_tag = f"{args.suite.replace('libero_', '')}_t{args.task}_s{args.seed}"
    prompt_vars = {
        "suite": args.suite,
        "task": args.task,
        "seed": args.seed,
        "recipe_tag": recipe_tag,
    }

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = get_repo_root() / "logs" / f"{timestamp}_{args.suite}_t{args.task}_s{args.seed}"
    output_dir = Path(output_dir)
    if reset_identity_output is not None:
        resolved_output = Path(reset_identity_output).expanduser().resolve()
        expected_output = (output_dir.resolve() / "reset_identity.json")
        if resolved_output != expected_output:
            raise ValueError(
                "research reset identity output must be reset_identity.json "
                "inside this run output directory"
            )
        resolved_completion = Path(completion_output).expanduser().resolve()
        expected_completion = output_dir.resolve() / "completion.json"
        if resolved_completion != expected_completion:
            raise ValueError(
                "research completion output must be completion.json inside "
                "this run output directory"
            )
        resolved_runtime_identity = Path(runtime_identity_output).expanduser().resolve()
        expected_runtime_identity = output_dir.resolve() / "runtime_identity.json"
        if resolved_runtime_identity != expected_runtime_identity:
            raise ValueError(
                "research runtime identity output must be runtime_identity.json "
                "inside this run output directory"
            )
        occupied = [
            path
            for path in (
                resolved_output,
                resolved_completion,
                resolved_runtime_identity,
                output_dir.resolve() / "states.json",
            )
            if path.exists()
        ]
        occupied.extend(output_dir.resolve().glob("transcript_*.json"))
        if occupied:
            raise FileExistsError(
                "research child refuses stale authoritative artifacts: "
                + ", ".join(str(path) for path in occupied)
            )

        from rpent.research.handoff.experiments.manifest import (
            load_manifest,
            verify_manifest_external_bindings,
        )
        from rpent.research.handoff.experiments.config import (
            load_strict_json,
            stable_digest,
        )
        from rpent.research.handoff.experiments.full_agent import (
            FULL_AGENT_PYTHON_EXECUTABLE_ENV,
            build_child_plan,
        )

        manifest = load_manifest(research_manifest_path)
        if manifest.manifest_id != research_manifest_id:
            raise ValueError("research manifest ID does not match manifest bytes")
        verify_manifest_external_bindings(
            manifest,
            repo_root=get_repo_root(),
            require_runtime_probes=True,
        )
        matches = [
            trial
            for trial in manifest.trials
            if trial.trial_id == research_trial_id
        ]
        if len(matches) != 1:
            raise ValueError("research manifest does not contain exactly one trial")
        trial = matches[0]
        bound_python = os.environ.get(FULL_AGENT_PYTHON_EXECUTABLE_ENV)
        if not bound_python:
            raise ValueError("research child lacks its bound Python executable")
        try:
            same_python = os.path.samefile(bound_python, sys.executable)
        except OSError as exc:
            raise ValueError("research child Python executable is unverifiable") from exc
        if not same_python:
            raise ValueError("research child Python executable disagrees with its plan")
        expected_plan = build_child_plan(
            trial,
            manifest_path=research_manifest_path,
            repo_root=get_repo_root(),
            python_executable=bound_python,
        )
        if expected_plan.plan_id != research_plan_id:
            raise ValueError("research plan ID does not bind this RPent child")
        if expected_plan.manifest_id != manifest.manifest_id:
            raise ValueError("research plan does not bind this manifest")
        expected_values = {
            "env_name": trial.runtime.env_name,
            "suite": trial.task.suite,
            "task": trial.task.task,
            "seed": trial.task.seed,
            "libero_type": trial.runtime.libero_type,
            "max_episode_steps": trial.runtime.max_episode_steps,
            "output_dir": str(Path(trial.output_dir)),
            "planner": trial.planner.backend,
            "model": trial.planner.model,
            "base_url": trial.planner.base_url,
            "max_turns": trial.planner.max_turns,
            "max_tokens": trial.planner.max_tokens,
            "planner_timeout_s": trial.planner.planner_timeout_s,
            "claude_code_max_budget_usd": trial.planner.claude_code_max_budget_usd,
            "no_images": trial.planner.no_images,
            "env_endpoint": trial.runtime.env_endpoint,
            "vla_endpoint": trial.runtime.vla_endpoint,
            "sam3_endpoint": trial.runtime.sam3_endpoint,
            "cuda_device": trial.runtime.cuda_device,
        }
        actual_values = {
            key: (str(Path(value)) if key == "output_dir" else value)
            for key, value in ((key, getattr(args, key)) for key in expected_values)
        }
        if actual_values != expected_values:
            raise ValueError(
                "actual RPent arguments disagree with manifest trial: "
                f"expected={expected_values!r}, actual={actual_values!r}"
            )
        expected_handoff = (
            output_dir.resolve() / "resolved_handoff_runtime.json"
            if trial.condition.handoff_enabled
            else None
        )
        actual_handoff = (
            Path(args.handoff_config).expanduser().resolve()
            if args.handoff_config is not None
            else None
        )
        if actual_handoff != expected_handoff:
            raise ValueError("actual handoff config path disagrees with manifest trial")
        allowed_preexisting = {
            output_dir.resolve() / "attempt.json",
        }
        if expected_handoff is not None:
            allowed_preexisting.add(expected_handoff)
        unexpected = sorted(
            (
                path.resolve()
                for path in output_dir.resolve().rglob("*")
                if path.is_file() and path.resolve() not in allowed_preexisting
            ),
            key=str,
        )
        if unexpected:
            raise FileExistsError(
                "research child found stale files before service startup: "
                + ", ".join(str(path) for path in unexpected[:20])
            )
        for variable, expected in (
            ("PI05_CHECKPOINT_PATH", trial.runtime.pi05_checkpoint_path),
            ("SAM3_CHECKPOINT_PATH", trial.runtime.sam3_checkpoint_path),
            ("RPENT_PI05_CHECKPOINT_ID", trial.runtime.pi05_checkpoint_id),
            ("RPENT_SAM3_CHECKPOINT_ID", trial.runtime.sam3_checkpoint_id),
        ):
            if os.environ.get(variable) != expected:
                raise ValueError(
                    f"research child environment {variable} disagrees with manifest"
                )
        attempt = load_strict_json(output_dir.resolve() / "attempt.json")
        if (
            attempt.get("trial_id") != trial.trial_id
            or attempt.get("manifest_id") != manifest.manifest_id
            or attempt.get("plan_id") != research_plan_id
            or attempt.get("source_revision") != trial.source_revision
            or attempt.get("cwd") != str(get_repo_root().resolve())
            or attempt.get("resolved_inner_command_sha256")
            != stable_digest(expected_plan.resolved_inner_command)
        ):
            raise ValueError("research attempt marker disagrees with manifest/plan")
        setattr(args, "_rpent_research_trial_manifest", trial)
        setattr(args, "_rpent_research_manifest_id", manifest.manifest_id)
        setattr(args, "_rpent_research_full_agent_plan", expected_plan)

    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars=prompt_vars,
        task_desc={"suite": args.suite, "task": args.task, "seed": args.seed},
    )


def _subprocess_env(**extra: str) -> dict[str, str]:
    """Build the env dict for a subprocess: inherit from parent, layer extras on top.

    CUDA device selection is passed via ``--cuda-device`` on the server command
    line — the server itself handles ``CUDA_VISIBLE_DEVICES`` and EGL alignment.
    """
    env = os.environ.copy()
    env.update(extra)
    return env


def init_task_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Initialize one TaskRun-owned LIBERO environment.

    A local env server is fresh for every call. When ``--env-endpoint`` is
    supplied, the returned daemon list is empty so the external service stays
    running.

    Heavy runtime dependencies stay lazy so importing :mod:`robots.libero`
    for its descriptor or toolkit does not load RPC/model packages.
    """
    handoff_config = _ensure_handoff_config(args)
    _validate_handoff_output(handoff_config, output_dir)

    from robots.libero.env_client import LiberoEnvClient
    from rpent.utils.config import get_libero_type
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.rpc import parse_endpoint, wait_for_ready
    from rpent.utils.socket_rpc import SocketRpcClient

    owned_daemons: list[ProcessDaemon] = []
    libero_type = args.libero_type or get_libero_type()
    cuda_args = ["--cuda-device", str(args.cuda_device)] if args.cuda_device is not None else []

    dashboard_events.emit(RuntimeStatusEvent("env", "starting"))
    try:
        env_daemon: ProcessDaemon | None = None
        if args.env_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            env_daemon = ProcessDaemon(
                name="env_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "libero" / "env_server.py"),
                    "--suite", args.suite,
                    "--task", str(args.task),
                    "--seed", str(args.seed),
                    "--max-episode-steps", str(args.max_episode_steps),
                    "--transport", "http",
                    "--host", host,
                    "--port", str(port),
                    "--parent-watch",
                    *cuda_args,
                ],
                env=_subprocess_env(
                    LIBERO_TYPE=libero_type,
                    MUJOCO_GL="egl",
                    ROBOT_PLATFORM="LIBERO",
                ),
                log_path=str(Path(output_dir) / "env_server.log"),
            )
            env_daemon.start()
            owned_daemons.append(env_daemon)
            env_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        else:
            protocol, host, port = parse_endpoint(args.env_endpoint)
            if protocol == "socket":
                env_rpc = SocketRpcClient(host, port)
            elif protocol == "http":
                env_rpc = HttpRpcClient(f"http://{host}:{port}")
            else:
                raise ValueError(
                    f"--env-endpoint protocol must be socket or http, got {protocol!r}"
                )
        wait_for_ready(env_rpc, daemon=env_daemon)
        env = LiberoEnvClient(
            env_rpc,
            expected_meta={
                "suite": args.suite,
                "task": args.task,
                "seed": args.seed,
                "max_episode_steps": args.max_episode_steps,
            },
        )
    except Exception as exc:
        _stop_owned_daemons(owned_daemons)
        dashboard_events.emit(RuntimeStatusEvent("env", "failed", error=exc))
        raise
    dashboard_events.emit(RuntimeStatusEvent("env", "ready"))
    runtime_values: dict[str, Any] = {"env": env}
    if handoff_config is not None and handoff_config.enabled:
        runtime_values[_HANDOFF_CONFIG_ATTR] = handoff_config
    try:
        reset_identity_request = _research_reset_identity_request(args)
    except Exception:
        _stop_owned_daemons(owned_daemons)
        raise
    if reset_identity_request is not None:
        runtime_values[_RESEARCH_RESET_IDENTITY_ATTR] = reset_identity_request
    return owned_daemons, runtime_values


def init_shared_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Initialize Session-owned VLA and SAM3 services.

    The returned list contains only locally started services. External
    endpoints are connected to but never become owned.
    """
    handoff_config = _ensure_handoff_config(args)
    _validate_handoff_output(handoff_config, output_dir)

    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.rpc import parse_endpoint, wait_for_ready
    from rpent.utils.sam3_client import Sam3Client
    from rpent.utils.socket_rpc import SocketRpcClient
    from rpent.utils.vla_client import VLAClient

    owned_daemons: list[ProcessDaemon] = []
    cuda_args = (
        ["--cuda-device", str(args.cuda_device)]
        if args.cuda_device is not None
        else []
    )

    # --- vla_server --------------------------------------------------------
    dashboard_events.emit(RuntimeStatusEvent("vla", "starting"))
    try:
        vla_daemon: ProcessDaemon | None = None
        if args.vla_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            vla_daemon = ProcessDaemon(
                name="vla_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "libero" / "vla_server.py"),
                    "--transport", "http",
                    "--host", host,
                    "--port", str(port),
                    "--parent-watch",
                    *cuda_args,
                ],
                env=_subprocess_env(),
                log_path=str(Path(output_dir) / "vla_server.log"),
            )
            vla_daemon.start()
            owned_daemons.append(vla_daemon)
            vla_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        else:
            protocol, host, port = parse_endpoint(args.vla_endpoint)
            if protocol == "socket":
                vla_rpc = SocketRpcClient(host, port)
            elif protocol == "http":
                vla_rpc = HttpRpcClient(f"http://{host}:{port}")
            else:
                raise ValueError(
                    f"--vla-endpoint protocol must be socket or http, got {protocol!r}"
                )
    except Exception as exc:
        _stop_owned_daemons(owned_daemons)
        dashboard_events.emit(RuntimeStatusEvent("vla", "failed", error=exc))
        raise

    # --- sam3_server -------------------------------------------------------
    dashboard_events.emit(RuntimeStatusEvent("sam3", "starting"))
    try:
        sam3_daemon: ProcessDaemon | None = None
        if args.sam3_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            sam3_daemon = ProcessDaemon(
                name="sam3_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "libero" / "sam3_server.py"),
                    "--transport", "http",
                    "--host", host,
                    "--port", str(port),
                    "--parent-watch",
                    *cuda_args,
                ],
                env=_subprocess_env(),
                log_path=str(Path(output_dir) / "sam3_server.log"),
            )
            sam3_daemon.start()
            owned_daemons.append(sam3_daemon)
            sam3_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        else:
            protocol, host, port = parse_endpoint(args.sam3_endpoint)
            if protocol == "socket":
                sam3_rpc = SocketRpcClient(host, port)
            elif protocol == "http":
                sam3_rpc = HttpRpcClient(f"http://{host}:{port}")
            else:
                raise ValueError(
                    f"--sam3-endpoint protocol must be socket or http, got {protocol!r}"
                )
    except Exception as exc:
        _stop_owned_daemons(owned_daemons)
        dashboard_events.emit(RuntimeStatusEvent("sam3", "failed", error=exc))
        raise

    # Start both local services before waiting so heavyweight initialization
    # continues concurrently, matching the one-shot runtime behavior.
    for component, client, daemon in (
        ("sam3", sam3_rpc, sam3_daemon),
        ("vla", vla_rpc, vla_daemon),
    ):
        try:
            wait_for_ready(client, daemon=daemon)
        except Exception as exc:
            _stop_owned_daemons(owned_daemons)
            dashboard_events.emit(RuntimeStatusEvent(component, "failed", error=exc))
            raise
        dashboard_events.emit(RuntimeStatusEvent(component, "ready"))

    model = VLAClient(vla_rpc)
    sam3_client = Sam3Client(sam3_rpc)

    primitives_kwargs = {
        "model": model,
        "sam3_client": sam3_client,
    }
    try:
        _attest_research_runtime(args, primitives_kwargs)
    except Exception:
        _stop_owned_daemons(owned_daemons)
        raise
    return owned_daemons, primitives_kwargs


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Spawn env + vla + SAM3 daemons and build clients for LIBERO.

    Each server can be spawned or attached-to independently: pass an
    endpoint to attach, or leave it unset to spawn a local subprocess.

    Heavy deps (rpc / vla / daemon / env_client) are imported lazily so
    that a bare ``import robots.libero`` (for ``get_env_spec`` /
    ``get_toolkit``) doesn't drag them in.
    """
    handoff_config = _ensure_handoff_config(args)
    _validate_handoff_output(handoff_config, output_dir)

    from robots.libero.env_client import LiberoEnvClient
    from rpent.utils.config import get_libero_type
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.rpc import parse_endpoint, wait_for_ready
    from rpent.utils.sam3_client import Sam3Client
    from rpent.utils.socket_rpc import SocketRpcClient
    from rpent.utils.vla_client import VLAClient

    daemons: list[ProcessDaemon] = []
    libero_type = args.libero_type or get_libero_type()
    cuda_args = ["--cuda-device", str(args.cuda_device)] if args.cuda_device is not None else []

    # --- env_server --------------------------------------------------------
    dashboard_events.emit(RuntimeStatusEvent("env", "starting"))
    try:
        env_daemon: ProcessDaemon | None = None
        if args.env_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            env_daemon = ProcessDaemon(
                name="env_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "libero" / "env_server.py"),
                    "--suite", args.suite,
                    "--task", str(args.task),
                    "--seed", str(args.seed),
                    "--max-episode-steps", str(args.max_episode_steps),
                    "--transport", "http",
                    "--host", host,
                    "--port", str(port),
                    "--parent-watch",
                    *cuda_args,
                ],
                env=_subprocess_env(
                    LIBERO_TYPE=libero_type,
                    MUJOCO_GL="egl",
                    ROBOT_PLATFORM="LIBERO",
                ),
                log_path=str(Path(output_dir) / "env_server.log"),
            )
            env_daemon.start()
            daemons.append(env_daemon)
            env_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        else:
            protocol, host, port = parse_endpoint(args.env_endpoint)
            if protocol == "socket":
                env_rpc = SocketRpcClient(host, port)
            elif protocol == "http":
                env_rpc = HttpRpcClient(f"http://{host}:{port}")
            else:
                raise ValueError(
                    f"--env-endpoint protocol must be socket or http, got {protocol!r}"
                )
    except Exception as exc:
        dashboard_events.emit(RuntimeStatusEvent("env", "failed", error=exc))
        raise

    # --- vla_server --------------------------------------------------------
    dashboard_events.emit(RuntimeStatusEvent("vla", "starting"))
    try:
        vla_daemon: ProcessDaemon | None = None
        if args.vla_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            vla_daemon = ProcessDaemon(
                name="vla_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "libero" / "vla_server.py"),
                    "--transport", "http",
                    "--host", host,
                    "--port", str(port),
                    "--parent-watch",
                    *cuda_args,
                ],
                env=_subprocess_env(),
                log_path=str(Path(output_dir) / "vla_server.log"),
            )
            vla_daemon.start()
            daemons.append(vla_daemon)
            vla_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        else:
            protocol, host, port = parse_endpoint(args.vla_endpoint)
            if protocol == "socket":
                vla_rpc = SocketRpcClient(host, port)
            elif protocol == "http":
                vla_rpc = HttpRpcClient(f"http://{host}:{port}")
            else:
                raise ValueError(
                    f"--vla-endpoint protocol must be socket or http, got {protocol!r}"
                )
    except Exception as exc:
        dashboard_events.emit(RuntimeStatusEvent("vla", "failed", error=exc))
        raise

    # --- sam3_server -------------------------------------------------------
    dashboard_events.emit(RuntimeStatusEvent("sam3", "starting"))
    try:
        sam3_daemon: ProcessDaemon | None = None
        if args.sam3_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            sam3_daemon = ProcessDaemon(
                name="sam3_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "libero" / "sam3_server.py"),
                    "--transport", "http",
                    "--host", host,
                    "--port", str(port),
                    "--parent-watch",
                    *cuda_args,
                ],
                env=_subprocess_env(),
                log_path=str(Path(output_dir) / "sam3_server.log"),
            )
            sam3_daemon.start()
            daemons.append(sam3_daemon)
            sam3_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        else:
            protocol, host, port = parse_endpoint(args.sam3_endpoint)
            if protocol == "socket":
                sam3_rpc = SocketRpcClient(host, port)
            elif protocol == "http":
                sam3_rpc = HttpRpcClient(f"http://{host}:{port}")
            else:
                raise ValueError(
                    f"--sam3-endpoint protocol must be socket or http, got {protocol!r}"
                )
    except Exception as exc:
        dashboard_events.emit(RuntimeStatusEvent("sam3", "failed", error=exc))
        raise

    # All local daemons are running now, so they initialize concurrently while
    # readiness is checked in a deterministic order.
    for component, client, daemon in (
        ("env", env_rpc, env_daemon),
        ("sam3", sam3_rpc, sam3_daemon),
        ("vla", vla_rpc, vla_daemon),
    ):
        try:
            wait_for_ready(client, daemon=daemon)
        except Exception as exc:
            for started_daemon in reversed(daemons):
                started_daemon.stop()
            dashboard_events.emit(RuntimeStatusEvent(component, "failed", error=exc))
            raise
        dashboard_events.emit(RuntimeStatusEvent(component, "ready"))

    primitives_kwargs = {
        "env": LiberoEnvClient(
            env_rpc,
            expected_meta={
                "suite": args.suite,
                "task": args.task,
                "seed": args.seed,
                "max_episode_steps": args.max_episode_steps,
            },
        ),
        "model": VLAClient(vla_rpc),
        "sam3_client": Sam3Client(sam3_rpc),
    }
    try:
        _attest_research_runtime(args, primitives_kwargs)
    except Exception:
        _stop_owned_daemons(daemons)
        raise
    if handoff_config is not None and handoff_config.enabled:
        primitives_kwargs[_HANDOFF_CONFIG_ATTR] = handoff_config
    try:
        reset_identity_request = _research_reset_identity_request(args)
    except Exception:
        _stop_owned_daemons(daemons)
        raise
    if reset_identity_request is not None:
        primitives_kwargs[_RESEARCH_RESET_IDENTITY_ATTR] = reset_identity_request
    return daemons, primitives_kwargs


def _stop_owned_daemons(daemons: list[ProcessDaemon]) -> None:
    """Stop owned daemons in reverse order without masking startup errors."""
    for daemon in reversed(daemons):
        try:
            daemon.stop()
        except Exception:
            pass
