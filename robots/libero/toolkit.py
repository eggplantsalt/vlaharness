"""LIBERO toolkit: common tools + LIBERO primitives.

Inherits the common file/IO tools from :class:`Toolkit` and registers the
LIBERO primitives (``move_to``, ``pi0_pick``, ``release``, ...) on top.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.libero import tools as libero_tools
from rpent.dashboard.events import DashboardEventSink, ToolResultEvent
from rpent.tools.toolkit import ToolCancelled, Toolkit
from rpent.utils.logging import get_logger, get_output_dir

if TYPE_CHECKING:
    from robots.libero.handoff_runtime import HandoffRuntimeConfig


class LiberoToolkit(Toolkit):
    """Toolkit for the LIBERO environment."""

    # Tool schemas keyed by name (built once from the canonical ordered list
    # in libero_tools.TOOLS_SPEC) so each tool registers with its own spec.
    _SPECS = {spec["name"]: spec for spec in libero_tools.TOOLS_SPEC}

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        video_path: str | None = None,
        handoff_config: HandoffRuntimeConfig | None = None,
        reset_identity_request: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(dashboard_events=dashboard_events)
        self._next_step: int = 0
        self._video_path: str | None = video_path
        self._reset_identity_request = (
            dict(reset_identity_request)
            if reset_identity_request is not None
            else None
        )
        self._handoff = None
        self._handoff_config = (
            handoff_config
            if handoff_config is not None and handoff_config.enabled
            else None
        )
        if self._handoff_config is None:
            # Baseline path: exact original kwargs and exact LiberoPrimitives.
            prepared_primitives_kwargs = primitives_kwargs
            handoff_sink = None
            instrumentation = None
        else:
            from robots.libero.handoff import (
                RuntimeInstrumentation,
                instrument_primitives_kwargs,
            )
            from robots.libero.handoff_runtime import build_research_sink

            handoff_sink = build_research_sink(
                self._handoff_config,
                run_output_dir=get_output_dir(),
            )
            instrumentation = RuntimeInstrumentation(
                handoff_sink,
                enabled=self._handoff_config.instrumentation,
            )
            prepared_primitives_kwargs = instrument_primitives_kwargs(
                primitives_kwargs,
                instrumentation,
            )
        self.init_primitives_clean(primitives_kwargs=prepared_primitives_kwargs)
        if self._handoff_config is not None:
            from robots.libero.handoff import build_handoff_composite

            assert handoff_sink is not None
            assert instrumentation is not None
            self._handoff = build_handoff_composite(
                primitives=self._primitives,
                config=self._handoff_config,
                sink=handoff_sink,
                instrumentation=instrumentation,
                check_cancelled=self.raise_if_cancelled,
                run_output_dir=get_output_dir(),
            )
        self._register_libero_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def _register_libero_tools(self) -> None:
        specs = self._SPECS
        # Inspection tools do not advance environment state. Most are stateless
        # module functions; segment is bound to the primitives-owned SAM3 client.
        inspection_handlers = {
            "view_driver_state": libero_tools.view_driver_state,
            "view_camera_meta": libero_tools.view_camera_meta,
            "back_project": libero_tools.back_project,
            "segment": self._primitives.segment,
        }
        for name, handler in inspection_handlers.items():
            self.add_tool(name, specs[name], handler)
        # In an opt-in handoff run, bind the two existing semantic Pi0 tool
        # schemas to the local composite.  The planner still makes exactly the
        # same one-shot tool commitment and sees no extra choice; the composite
        # eventually calls the unchanged primitive method.  With research
        # disabled, handlers and schema order remain the Original Harness path.
        handoff_overrides = {
            "pi0_pick": "handoff_pi0_pick",
            "pi0_doubled": "handoff_pi0_doubled",
        }
        for name in libero_tools.PRIMITIVE_TOOL_NAMES:
            composite_name = handoff_overrides.get(name)
            if self._handoff is not None and composite_name is not None:
                handler = partial(self._handoff_step, composite_name)
            else:
                handler = partial(self._step, name)
            self.add_tool(name, specs[name], handler)

    def _handoff_step(self, name: str, **kwargs) -> dict:
        assert self._handoff is not None
        return self._run_physical_step(
            name,
            getattr(self._handoff, name),
            kwargs,
        )

    def _step(self, name: str, **kwargs) -> dict:
        """Run ``self._primitives.<name>(**kwargs)``, dump the new step, and
        return the rendered state view + log.
        """
        return self._run_physical_step(
            name,
            getattr(self._primitives, name),
            kwargs,
        )

    def _run_physical_step(
        self,
        name: str,
        handler,
        kwargs: dict[str, Any],
    ) -> dict:
        """Shared outer lifecycle for baseline and opt-in composite tools."""
        command = {"action": name, **kwargs}
        t0 = time.time()
        start_frame = self._primitives.recorded_frame_count()
        try:
            result = handler(**kwargs)
            self.raise_if_cancelled()
        except ToolCancelled as exc:
            result = {
                "error": str(exc),
                "code": "tool_cancelled",
                "interrupted": True,
            }
        elapsed = round(time.time() - t0, 2)

        if isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"value": result}

        self._next_step += 1
        step_idx = self._next_step
        output_dir = get_output_dir()
        if self._dashboard_events.enabled:
            video_dir = libero_tools.artifact_path(output_dir, "action_videos")
            video_path = video_dir / f"step_{step_idx:02d}_{name}.mp4"
            try:
                self._primitives.save_frame_slice(start_frame, str(video_path), fps=20)
            except Exception as e:
                get_logger("libero_toolkit").warning(
                    f"failed to save action clip to {video_path}: {e}"
                )
        libero_tools.dump_state(
            self._primitives,
            str(output_dir),
            step_idx=step_idx,
            log={"command": command, "result": result_dict, "elapsed_s": elapsed},
        )
        out = libero_tools.view_driver_state(step_idx)
        out["agent_elapsed_s"] = elapsed
        if result_dict.get("interrupted"):
            out.update(result_dict)
        return out

    def init_primitives_clean(
        self,
        *,
        primitives_kwargs: dict[str, Any],
    ) -> None:
        """Wipe stale run artifacts, build the LiberoPrimitives, dump step 0."""
        out_dir = get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        for sub in libero_tools.ARTIFACT_DIRECTORIES:
            target = out_dir / sub
            if target.exists():
                shutil.rmtree(target)
        for target in (
            libero_tools.artifact_path(out_dir, "states"),
            libero_tools.artifact_path(out_dir, "metadata", camera="agentview", resolution="low"),
            libero_tools.artifact_path(out_dir, "episode_video"),
        ):
            if target.exists():
                target.unlink()

        primitives = libero_tools.LiberoPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **primitives_kwargs,
        )
        primitives.reset()
        self._write_research_reset_identity(primitives)
        primitives.start_recording()
        libero_tools.dump_state(primitives, str(out_dir), step_idx=0, log=None)
        self._dashboard_events.emit(
            ToolResultEvent(
                name="view_driver_state",
                result=libero_tools.view_driver_state(0),
            )
        )

        self._primitives = primitives

    def _write_research_reset_identity(self, primitives) -> None:
        """Persist run-local reset evidence for an explicitly research-tagged run.

        This hook is absent from the normal Original Harness call path.  It
        records only deployment-safe identities returned by the live server;
        raw observations and privileged simulator state are deliberately not
        copied into the sidecar.
        """
        request = self._reset_identity_request
        if request is None:
            return
        required = {"path", "trial_id", "suite", "task", "seed"}
        if set(request) != required:
            raise ValueError(
                "research reset identity request must contain exactly "
                f"{sorted(required)}"
            )
        if not isinstance(request["trial_id"], str) or not request["trial_id"]:
            raise ValueError("research reset trial_id must be non-empty")
        if not isinstance(request["suite"], str) or not request["suite"]:
            raise ValueError("research reset suite must be non-empty")
        for field in ("task", "seed"):
            value = request[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"research reset {field} must be a non-negative integer"
                )

        payload = primitives.env.runtime_probe()
        if not isinstance(payload, Mapping):
            raise RuntimeError("live env runtime_probe did not return an object")
        server_meta = payload.get("server_meta")
        runtime_meta = payload.get("runtime_meta")
        if not isinstance(server_meta, Mapping) or not isinstance(
            runtime_meta, Mapping
        ):
            raise RuntimeError(
                "live env runtime_probe lacks server_meta/runtime_meta objects"
            )
        for field in ("suite", "task", "seed"):
            if server_meta.get(field) != request[field]:
                raise RuntimeError(
                    "live env identity disagrees with research trial: "
                    f"{field}={server_meta.get(field)!r}, "
                    f"expected {request[field]!r}"
                )
        reset_id = runtime_meta.get("reset_id")
        if isinstance(reset_id, bool) or not isinstance(reset_id, (int, str)):
            raise RuntimeError(
                "live env runtime_probe reset_id must be an integer or string"
            )
        if isinstance(reset_id, int) and reset_id < 0:
            raise RuntimeError("live env runtime_probe reset_id must be non-negative")
        if isinstance(reset_id, str) and not reset_id.strip():
            raise RuntimeError("live env runtime_probe reset_id must be non-empty")

        destination = Path(str(request["path"])).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(
                f"research reset identity sidecar already exists: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        sidecar = {
            "schema_version": "rpent.research-reset-identity/v1",
            "trial_id": request["trial_id"],
            "suite": request["suite"],
            "task": request["task"],
            "seed": request["seed"],
            "reset_id": str(reset_id),
            "observed_after_reset": True,
            "source": "live_env_runtime_probe",
            "probe_schema_version": payload.get("schema_version"),
            "probe_component": payload.get("component"),
        }
        temporary = destination.with_name(f".{destination.name}.tmp")
        if temporary.exists():
            raise FileExistsError(
                f"research reset identity temporary already exists: {temporary}"
            )
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(
                    sidecar,
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise

    def close(self) -> None:
        """Flush the agent-side video buffer to disk (end-of-run).
        """
        if self._video_path is None:
            return
        try:
            self._primitives.stop_recording_and_save(self._video_path)
        except Exception as e:
            # The runner is in the cleanup path; never let a video save
            # abort it.
            get_logger("libero_toolkit").warning(
                f"failed to save video to {self._video_path}: {e}"
            )

    def write_recipe(self, recipe_tag: str) -> str:
        """Write the LIBERO recipe JSONL from the dumped state trace."""
        return libero_tools.write_recipe_from_states(str(get_output_dir()), recipe_tag)
