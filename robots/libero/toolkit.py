"""LIBERO toolkit: common tools + LIBERO primitives.

Inherits the common file/IO tools from :class:`Toolkit` and registers the
LIBERO primitives (``move_to``, ``pi0_pick``, ``release``, ...) on top.
"""
from __future__ import annotations

import hashlib
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
        self._research_direct_vla_attempt_count = 0
        self._research_direct_vla_event_count = 0
        self._research_direct_vla_previous_event_sha256: str | None = None
        self._research_reset_id: str | None = None
        self._research_reset_identity_sha256: str | None = None
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
        direct_vla_attempt_index: int | None = None
        terminal_phase = "completed"
        terminal_error_type: str | None = None
        terminal_error: str | None = None
        if (
            self._reset_identity_request is not None
            and self._handoff is None
            and name in {"pi0_pick", "pi0_doubled"}
        ):
            self._research_direct_vla_attempt_count += 1
            direct_vla_attempt_index = self._research_direct_vla_attempt_count
            self._append_direct_vla_attempt_event(
                attempt_index=direct_vla_attempt_index,
                tool_name=name,
                phase="started",
                elapsed_s=None,
                error_type=None,
                error=None,
            )
        try:
            result = handler(**kwargs)
            self.raise_if_cancelled()
        except ToolCancelled as exc:
            terminal_phase = "cancelled"
            terminal_error_type = type(exc).__name__
            terminal_error = str(exc)
            result = {
                "error": str(exc),
                "code": "tool_cancelled",
                "interrupted": True,
            }
        except Exception as exc:
            if direct_vla_attempt_index is not None:
                try:
                    self._append_direct_vla_attempt_event(
                        attempt_index=direct_vla_attempt_index,
                        tool_name=name,
                        phase="error",
                        elapsed_s=max(0.0, time.time() - t0),
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                except Exception as evidence_error:
                    if hasattr(exc, "add_note"):
                        exc.add_note(
                            "failed to append research direct-VLA error evidence: "
                            f"{type(evidence_error).__name__}: {evidence_error}"
                        )
            raise
        elapsed = round(time.time() - t0, 2)

        if direct_vla_attempt_index is not None:
            if (
                terminal_phase == "completed"
                and isinstance(result, Mapping)
                and result.get("error") is not None
            ):
                terminal_phase = "returned_error"
                terminal_error_type = str(result.get("code") or "returned_error")
                terminal_error = str(result.get("error"))
            self._append_direct_vla_attempt_event(
                attempt_index=direct_vla_attempt_index,
                tool_name=name,
                phase=terminal_phase,
                elapsed_s=elapsed,
                error_type=terminal_error_type,
                error=terminal_error,
            )

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

    def _append_direct_vla_attempt_event(
        self,
        *,
        attempt_index: int,
        tool_name: str,
        phase: str,
        elapsed_s: float | None,
        error_type: str | None,
        error: str | None,
    ) -> None:
        """Append fsynced evidence before/after each research Original VLA call."""
        request = self._reset_identity_request
        if request is None:
            raise RuntimeError("direct-VLA attempt evidence requires research identity")
        if (
            self._research_reset_id is None
            or self._research_reset_identity_sha256 is None
        ):
            raise RuntimeError("direct-VLA attempt cannot precede reset identity")
        self._research_direct_vla_event_count += 1
        payload = {
            "schema_version": "rpent.research-direct-vla-attempt/v1",
            "event_sequence": self._research_direct_vla_event_count,
            "previous_event_sha256": (
                self._research_direct_vla_previous_event_sha256
            ),
            "attempt_index": attempt_index,
            "step_index": self._next_step + 1,
            "trial_id": request["trial_id"],
            "manifest_id": request["manifest_id"],
            "plan_id": request["plan_id"],
            "source_revision": request["source_revision"],
            "reset_id": self._research_reset_id,
            "reset_identity_sha256": self._research_reset_identity_sha256,
            "runtime_attestation_id": request["runtime_attestation_id"],
            "runtime_attestation_sha256": request[
                "runtime_attestation_sha256"
            ],
            "tool_name": tool_name,
            "phase": phase,
            "vla_attempted": True,
            "attempt_unit": "planner_visible_vla_tool_invocation",
            "elapsed_s": elapsed_s,
            "error_type": error_type,
            "error": error,
            "recorded_before_state_dump": True,
        }
        canonical_payload = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        event_sha256 = hashlib.sha256(canonical_payload).hexdigest()
        record = {**payload, "event_sha256": event_sha256}
        line = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"
        destination = get_output_dir() / "direct_vla_attempts.jsonl"
        if self._research_direct_vla_event_count == 1:
            mode = "x"
        else:
            if not destination.is_file():
                raise RuntimeError("direct-VLA attempt journal disappeared during run")
            mode = "a"
        with destination.open(mode, encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        self._research_direct_vla_previous_event_sha256 = event_sha256

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
        required = {
            "path",
            "trial_id",
            "manifest_id",
            "plan_id",
            "source_revision",
            "suite",
            "task",
            "seed",
            "max_episode_steps",
            "runtime_attestation_path",
            "runtime_attestation_id",
            "runtime_attestation_sha256",
        }
        if set(request) != required:
            raise ValueError(
                "research reset identity request must contain exactly "
                f"{sorted(required)}"
            )
        for field in (
            "trial_id",
            "manifest_id",
            "plan_id",
            "source_revision",
            "suite",
            "runtime_attestation_id",
            "runtime_attestation_sha256",
        ):
            if not isinstance(request[field], str) or not request[field]:
                raise ValueError(f"research reset {field} must be non-empty")
        for field in ("task", "seed", "max_episode_steps"):
            value = request[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"research reset {field} must be a non-negative integer"
                )

        from rpent.research.handoff.experiments.runtime_identity import (
            verify_runtime_attestation_binding,
        )

        attestation, attestation_sha256 = verify_runtime_attestation_binding(
            request["runtime_attestation_path"],
            trial_id=request["trial_id"],
            manifest_id=request["manifest_id"],
            plan_id=request["plan_id"],
            source_revision=request["source_revision"],
            expected_attestation_id=request["runtime_attestation_id"],
            expected_sha256=request["runtime_attestation_sha256"],
        )

        payload = primitives.env.runtime_probe()
        if not isinstance(payload, Mapping):
            raise RuntimeError("live env runtime_probe did not return an object")
        if payload.get("schema_version") != "rpent.runtime-probe/v1":
            raise RuntimeError("live env runtime_probe schema mismatch")
        if payload.get("component") != "libero_env":
            raise RuntimeError("live env runtime_probe component mismatch")
        server_meta = payload.get("server_meta")
        runtime_meta = payload.get("runtime_meta")
        if not isinstance(server_meta, Mapping) or not isinstance(
            runtime_meta, Mapping
        ):
            raise RuntimeError(
                "live env runtime_probe lacks server_meta/runtime_meta objects"
            )
        for field in ("suite", "task", "seed", "max_episode_steps"):
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
            "manifest_id": request["manifest_id"],
            "plan_id": request["plan_id"],
            "source_revision": request["source_revision"],
            "suite": request["suite"],
            "task": request["task"],
            "seed": request["seed"],
            "max_episode_steps": request["max_episode_steps"],
            "reset_id": str(reset_id),
            "observed_after_reset": True,
            "source": "live_env_runtime_probe",
            "probe_schema_version": payload.get("schema_version"),
            "probe_component": payload.get("component"),
            "runtime_attestation_id": attestation.attestation_id,
            "runtime_attestation_sha256": attestation_sha256,
        }
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
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
        self._research_reset_id = str(reset_id)
        self._research_reset_identity_sha256 = hashlib.sha256(
            destination.read_bytes()
        ).hexdigest()

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
