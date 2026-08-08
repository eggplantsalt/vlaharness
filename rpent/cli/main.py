"""Physical agent main CLI entrypoint."""
# `rpent/cli/`
#
# CLI entrypoints for RPent (currently just `main.py`).
#
# ## Run
#
# `main()` is exposed as the `rpent` console script (see `[project.scripts]`
# in `pyproject.toml`):
#
# ```bash
# rpent --env libero --suite libero_object_task --task 0 --seed 0 [...]
# ```
#
# ## Note
#
# Do not import `rpent.cli` from other `rpent` modules. `main.py` pulls in
# `rpent.planner`, `rpent.envs`, `rpent.utils`, `rpent.dashboard`, and
# `rpent.tools`, so importing the CLI back into any of them would create an
# import cycle. Nothing else should depend on this package.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shlex
import sys
import time
from collections.abc import Callable
from pathlib import Path

from rpent.cli.tui import (
    start_first_prompt_resolver,
    start_interactive_reader,
)
from rpent.dashboard.events import (
    NullDashboardEventSink,
    RunStartedEvent,
)
from rpent.envs import get_env_spec, get_toolkit
from rpent.planner.base import build_planner
from rpent.utils.logging import get_logger, init_output_dir
from rpent.utils.resources import ensure_resources

logger = get_logger("agent")


# ---------------------------------------------------------------------------
# API agent transcript serialization
# ---------------------------------------------------------------------------


def _strip_images(value):
    """Return a copy of ``value`` with inline image payloads omitted.

    SDK objects are left untouched; ``json.dump(..., default=str)`` handles
    them at write time. Only the bulky base64 image blocks are replaced.
    """
    if isinstance(value, list):
        return [_strip_images(v) for v in value]
    if isinstance(value, dict):
        if value.get("type") == "image":
            return {"type": "image", "source": {"_omitted_for_transcript": True}}
        if value.get("type") == "image_url":
            return {"type": "image_url", "image_url": {"_omitted_for_transcript": True}}
        return {k: _strip_images(v) for k, v in value.items()}
    return value


def _serialize_messages(messages: list[dict]) -> list[dict]:
    """Strip inline image payloads from messages before writing the transcript."""
    return [
        {**{k: v for k, v in m.items() if k != "content"},
         "content": _strip_images(m.get("content"))}
        for m in messages
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Standalone hybrid LLM-in-the-loop physical agent",
    )

    ap.add_argument("--env", dest="env_name", required=True, choices=["libero"],
                    help="Environment backend: libero.")

    # models
    ap.add_argument("--planner", default="api",
                    choices=["api", "claude_code", "codex"],
                    help="LLM backend: api | claude_code | codex.")
    ap.add_argument("--model", default=None,
                    help="Model id. For the 'api' planner, prefix the provider "
                         "(e.g. anthropic:claude-opus-4-8, openai:gpt-5.5, "
                         "openai-chat:glm-5.2). For claude_code/codex this "
                         "overrides the backend default model.")
    ap.add_argument("--base-url", default=None,
                    help="API base URL. Defaults to the selected backend's base URL env var.")
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--no-images", action="store_true",
                    help="Never send image bytes to the model (api planner only). "
                         "Use for text-only models that reject image input "
                         "(e.g. 400 \"message type 'image_url' is not supported\"); "
                         "read_image then returns the file path with a notice.")
    ap.add_argument("--planner-timeout-s", type=int, default=None,
                    help="Wall-clock cap for api/claude_code/codex planner runs. "
                         "Terminal interactive API/Claude sessions are exempt. "
                         "Defaults to CODEX_TIMEOUT_S (codex only), "
                         "CELL_TIMEOUT_S, or 1200.")
    ap.add_argument("--claude-code-max-budget-usd", type=float, default=None,
                    help="Budget passed to claude -p --max-budget-usd. "
                         "Defaults to MAX_BUDGET_USD env or 10.")

    # other config
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dashboard", action="store_true",
                    help="Start a local dashboard server for this single run.")
    ap.add_argument("--dashboard-host", default="127.0.0.1",
                    help="Dashboard bind host. Defaults to 127.0.0.1.")
    ap.add_argument("--dashboard-port", type=int, default=0,
                    help="Dashboard port. 0 asks the OS for a free port.")
    ap.add_argument("--dashboard-language", choices=["en", "zh-cn"], default="en",
                    help="Dashboard UI language. 'zh-cn' serves the Chinese "
                         "translation; defaults to English.")
    ap.add_argument("--verbose", action="store_true",
                    help="Enable DEBUG-level logging for stdout and the run.log "
                         "file. Defaults to INFO when not set.")
    ap.add_argument("--interactive", "-i", action="store_true",
                    help="Interactive mode: opens an interactive cli session.")

    return ap


def main() -> int:
    parser = _build_argparser()
    # Two-phase argparse: first grab --env / --dashboard so we know which
    # env's flags to add and whether to make its required flags optional.
    early, _ = parser.parse_known_args()

    env_spec = get_env_spec(early.env_name)
    env_spec.add_cli_args(parser, use_dashboard=early.dashboard)
    args = parser.parse_args()
    if args.dashboard and args.interactive:
        parser.error("--dashboard and --interactive cannot be used together")
    if args.dashboard:
        from rpent.cli.dashboard import run_dashboard_session

        return run_dashboard_session(args, env_spec, parser=parser)

    run_config = env_spec.parse_config(args)
    recipe_tag = run_config.recipe_tag
    output_dir = run_config.output_dir
    prompt_vars = run_config.prompt_vars
    task_desc = run_config.task_desc

    env_name = args.env_name

    # mkdir + logging wiring (env-side already picked the path).
    output_dir = init_output_dir(output_dir, verbose=args.verbose)
    logger.info("physical agent cmd: %s", shlex.join([sys.executable, *sys.argv]))

    ensure_resources(env_name)

    dashboard_events = NullDashboardEventSink()

    planner = build_planner(
        args.planner,
        output_dir=output_dir,
        recipe_tag=recipe_tag,
        env_name=env_name,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        planner_timeout_s=args.planner_timeout_s,
        claude_code_max_budget_usd=args.claude_code_max_budget_usd,
        dashboard_events=dashboard_events,
        no_images=args.no_images,
    )
    prompt_bundle = env_spec.prompts
    prompt_vars = {**prompt_vars, "output_dir": output_dir}
    system_prompt = prompt_bundle.render(
        "system",
        variables=prompt_vars,
    )
    user_msg = prompt_bundle.render(
        "user",
        variables=prompt_vars,
    )

    input_queue: "queue.Queue[str | None] | None" = None
    await_first_prompt: "Callable[[], str | None] | None" = None
    if args.interactive:
        input_queue = queue.Queue()
        # Pre-fill the first prompt with the rendered default task (editable
        # preset);
        start_interactive_reader(input_queue, first_prompt_default=user_msg)
        logger.info(
            "interactive mode on: the built-in task is pre-filled — "
            "edit it and press Enter, submit it as-is, or clear it to "
            "type your own. Once running, type to steer the agent. "
            "/help for commands."
        )
        # Resolve the opening prompt on a background thread so the user can type
        # it while the (slow) env/VLA servers boot below.
        await_first_prompt = start_first_prompt_resolver(input_queue)

    # --- initialise environment --------------------------------------------
    daemons, primitives_kwargs = env_spec.init_runtime(
        args,
        output_dir,
        dashboard_events,
    )

    # --- toolkit -----------------------------------------------------------
    toolkit = get_toolkit(
        env_name,
        primitives_kwargs=primitives_kwargs,
        video_path=str(Path(output_dir) / "episode.mp4"),
        dashboard_events=dashboard_events,
    )

    # --- agent loop --------------------------------------------------------
    t0 = time.time()
    finish_result, messages, agent_error = None, [], None
    stats: dict = {}
    first_user_msg: str | None = user_msg
    if await_first_prompt is not None:
        # Block until the opening prompt typed during startup is ready.
        first_user_msg = await_first_prompt()
        if first_user_msg is None:
            logger.info("no task entered; ending session before start.")
    try:
        if first_user_msg is not None:
            dashboard_events.emit(RunStartedEvent())
            result = planner.solve(
                system_prompt=system_prompt,
                user_message=first_user_msg,
                toolkit=toolkit,
                max_turns=args.max_turns,
                input_queue=input_queue,
            )
            finish_result = result.finish_result
            messages = result.messages
            stats = result.stats
            agent_error = result.error
    except Exception as exc:
        logger.error("EXCEPTION in agent loop: %s", exc)
        agent_error = str(exc)
    finally:
        # Agent-side: flush the episode video before the env+model
        recipe_path = toolkit.write_recipe(recipe_tag)
        logger.info("recipe: %s", recipe_path)

        toolkit.close()
        for d in daemons:
            d.stop()

    elapsed = time.time() - t0

    transcript_path = Path(output_dir) / f"transcript_{recipe_tag}.json"
    record = {
        **task_desc,
        "model": args.model,
        "elapsed_s": round(elapsed, 1),
        "finish": finish_result,
        "stats": stats,
        "messages": _serialize_messages(messages),
    }
    with open(transcript_path, "a") as f:
        json.dump(record, f, indent=2, default=str)

    # Research manifests pass hidden, output-local telemetry arguments.  The
    # normal Original Harness CLI does not, so its files and control behavior
    # remain unchanged.  This sidecar preserves planner exceptions that the
    # historical transcript schema intentionally did not contain.
    completion_output = getattr(args, "research_completion_output", None)
    research_trial_id = getattr(args, "research_trial_id", None)
    if completion_output is not None:
        from rpent.research.handoff.experiments.config import load_strict_json
        from rpent.research.handoff.experiments.runtime_identity import (
            verify_runtime_attestation_binding,
        )

        destination = Path(completion_output).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(
                f"research completion sidecar already exists: {destination}"
            )
        trial = getattr(args, "_rpent_research_trial_manifest", None)
        if trial is None or trial.trial_id != research_trial_id:
            raise RuntimeError("research completion lacks its validated manifest trial")
        runtime_identity_path = Path(
            args.research_runtime_identity_output
        ).expanduser().resolve()
        attestation, runtime_attestation_sha256 = (
            verify_runtime_attestation_binding(
                runtime_identity_path,
                trial_id=trial.trial_id,
                manifest_id=str(args.research_manifest_id),
                plan_id=str(args.research_plan_id),
                source_revision=trial.source_revision,
            )
        )
        reset_identity_path = Path(
            args.research_reset_identity_output
        ).expanduser().resolve()
        reset_identity = load_strict_json(reset_identity_path)
        expected_reset_fields = {
            "schema_version": "rpent.research-reset-identity/v1",
            "trial_id": trial.trial_id,
            "manifest_id": str(args.research_manifest_id),
            "plan_id": str(args.research_plan_id),
            "source_revision": trial.source_revision,
            "suite": trial.task.suite,
            "task": trial.task.task,
            "seed": trial.task.seed,
            "max_episode_steps": trial.runtime.max_episode_steps,
            "observed_after_reset": True,
            "source": "live_env_runtime_probe",
            "probe_schema_version": "rpent.runtime-probe/v1",
            "probe_component": "libero_env",
            "runtime_attestation_id": attestation.attestation_id,
            "runtime_attestation_sha256": runtime_attestation_sha256,
        }
        reset_mismatches = {
            field: {"expected": expected, "actual": reset_identity.get(field)}
            for field, expected in expected_reset_fields.items()
            if reset_identity.get(field) != expected
        }
        reset_id = reset_identity.get("reset_id")
        if reset_mismatches or not isinstance(reset_id, str) or not reset_id:
            raise RuntimeError(
                "research reset sidecar disagrees with completion identity: "
                f"mismatches={reset_mismatches!r}, reset_id={reset_id!r}"
            )
        reset_identity_sha256 = hashlib.sha256(
            reset_identity_path.read_bytes()
        ).hexdigest()
        direct_vla_attempts_path = (
            Path(output_dir) / "direct_vla_attempts.jsonl"
        ).resolve()
        states_path = (Path(output_dir) / "states.json").resolve()
        if not states_path.is_file():
            raise RuntimeError(
                f"research completion cannot bind missing states trace: {states_path}"
            )
        direct_vla_attempts_sha256 = (
            hashlib.sha256(direct_vla_attempts_path.read_bytes()).hexdigest()
            if direct_vla_attempts_path.is_file()
            else None
        )
        completion_record = {
            "schema_version": "rpent.research-completion/v2",
            "trial_id": str(research_trial_id),
            "manifest_id": str(args.research_manifest_id),
            "plan_id": str(args.research_plan_id),
            "source_revision": trial.source_revision,
            "status": (
                "planner_error"
                if agent_error is not None
                else (
                    "finish_declared"
                    if finish_result is not None
                    else "planner_returned_without_finish"
                )
            ),
            "agent_error": (
                str(agent_error) if agent_error is not None else None
            ),
            "transcript_path": str(transcript_path.resolve()),
            "transcript_sha256": hashlib.sha256(
                transcript_path.read_bytes()
            ).hexdigest(),
            "states_path": str(states_path),
            "states_sha256": hashlib.sha256(states_path.read_bytes()).hexdigest(),
            "elapsed_s": round(elapsed, 1),
            "planner_backend": args.planner,
            "planner_model": args.model,
            "planner_base_url": args.base_url,
            "runtime_attestation_id": attestation.attestation_id,
            "runtime_attestation_path": str(runtime_identity_path),
            "runtime_attestation_sha256": runtime_attestation_sha256,
            "reset_id": reset_id,
            "reset_identity_path": str(reset_identity_path),
            "reset_identity_sha256": reset_identity_sha256,
            "direct_vla_attempts_path": (
                str(direct_vla_attempts_path)
                if direct_vla_attempts_sha256 is not None
                else None
            ),
            "direct_vla_attempts_sha256": direct_vla_attempts_sha256,
        }
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                completion_record,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    logger.info("elapsed: %.1fs", elapsed)
    logger.info("usage: in=%s out=%s tool_calls=%s",
                 stats.get('total_input_tokens', '?'),
                 stats.get('total_output_tokens', '?'),
                 stats.get('tool_calls', '?'))
    logger.info("transcript: %s", transcript_path)
    if agent_error is not None:
        logger.error("error: %s", agent_error)

    if completion_output is not None and agent_error is not None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
