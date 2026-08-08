from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rpent.research.handoff.experiments.config import (
    ExperimentConfig,
    stable_digest,
)
from rpent.research.handoff.experiments.full_agent import build_child_plan
from rpent.research.handoff.experiments.full_agent_outcomes import (
    FullAgentSummaryError,
    summarize_full_agent_trial,
)
from rpent.research.handoff.experiments.manifest import (
    expand_manifest,
    write_manifest,
)
from rpent.research.handoff.experiments.runtime_identity import (
    attest_runtime_checkpoint_clients,
    write_runtime_attestation,
)


def _baseline_context(tmp_path):
    config = ExperimentConfig.model_validate(
        {
            "experiment_id": "full-summary-test",
            "output_root": str(tmp_path / "outputs"),
            "tasks": [
                {
                    "suite": "libero_object",
                    "task": 2,
                    "seeds": [7],
                    "target_id": "cup",
                    "target_description": "the cup",
                    "skill_name": "pick",
                    "skill_prompt": "pick the cup",
                    "label_source": "official_termination",
                }
            ],
            "conditions": [
                {
                    "name": "original",
                    "execution_layer": "full_agent",
                    "method": "original_harness",
                    "decision": "direct",
                    "hierarchy": "planner_mediated",
                }
            ],
            "runtime": {
                "vla_endpoint": "http://vla.example:8011",
                "sam3_endpoint": "http://sam.example:8012",
                "pi05_checkpoint_id": "sha256:pi05-test",
                "sam3_checkpoint_id": "sha256:sam3-test",
            },
            "planner": {
                "backend": "api",
                "model": "planner-test-model",
                "base_url": "http://planner.example:8000/v1",
            },
            "source_revision": "git:test-source",
        }
    )
    manifest = expand_manifest(config)
    manifest_path = write_manifest(manifest, tmp_path / "manifest.json")
    trial = manifest.trials[0]
    plan = build_child_plan(
        trial,
        manifest_path=manifest_path,
        repo_root=tmp_path,
    )
    return manifest, trial, plan


class _CheckpointClient:
    def __init__(self, component: str, checkpoint_id: str, path: str) -> None:
        self.component = component
        self.checkpoint_id = checkpoint_id
        self.path = path

    def runtime_probe(self):
        return {
            "schema_version": "rpent.runtime-probe/v1",
            "component": self.component,
            "checkpoint": {
                "configured_id": self.checkpoint_id,
                "path": self.path,
                "exists": True,
            },
            "model_class": f"fixture.{self.component}",
        }


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _direct_attempt_record(payload: dict) -> dict:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {**payload, "event_sha256": hashlib.sha256(canonical).hexdigest()}


def _write_direct_attempts(
    trial,
    plan,
    *,
    reset_id: str,
    reset_sha256: str,
    attestation_id: str,
    attestation_sha256: str,
) -> Path:
    root = Path(trial.output_dir)
    common = {
        "schema_version": "rpent.research-direct-vla-attempt/v1",
        "attempt_index": 1,
        "step_index": 1,
        "trial_id": trial.trial_id,
        "manifest_id": plan.manifest_id,
        "plan_id": plan.plan_id,
        "source_revision": trial.source_revision,
        "reset_id": reset_id,
        "reset_identity_sha256": reset_sha256,
        "runtime_attestation_id": attestation_id,
        "runtime_attestation_sha256": attestation_sha256,
        "tool_name": "pi0_pick",
        "vla_attempted": True,
        "attempt_unit": "planner_visible_vla_tool_invocation",
        "error_type": None,
        "error": None,
        "recorded_before_state_dump": True,
    }
    started = _direct_attempt_record(
        {
            **common,
            "event_sequence": 1,
            "previous_event_sha256": None,
            "phase": "started",
            "elapsed_s": None,
        }
    )
    completed = _direct_attempt_record(
        {
            **common,
            "event_sequence": 2,
            "previous_event_sha256": started["event_sha256"],
            "phase": "completed",
            "elapsed_s": 0.4,
        }
    )
    path = root / "direct_vla_attempts.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for record in (started, completed)
        ),
        encoding="utf-8",
    )
    return path


def _write_full_agent_artifacts(trial, plan) -> None:
    output = Path(trial.output_dir)
    output.mkdir(parents=True)
    _write_json(
        output / "attempt.json",
        {
            "schema_version": "rpent.handoff-full-agent-attempt/v1",
            "trial_id": trial.trial_id,
            "manifest_id": plan.manifest_id,
            "plan_id": plan.plan_id,
            "source_revision": trial.source_revision,
            "cwd": plan.cwd,
            "resolved_inner_command_sha256": stable_digest(
                plan.resolved_inner_command
            ),
        },
    )
    states = [
        {
            "step_idx": 0,
            "command": None,
            "state": {
                "robot0_eef_pos": [0.1, 0.2, 0.3],
                "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0],
                "robot0_gripper_qpos": [0.02, -0.02],
            },
            "libero_terminated": False,
            "episode_truncated": False,
        },
        {
            "step_idx": 1,
            "command": {"action": "pi0_pick", "prompt": "pick the cup"},
            "result": {"chunks_used": 2},
            "elapsed_s": 0.4,
            "state": {
                "robot0_eef_pos": [0.1, 0.2, 0.08],
                "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0],
                "robot0_gripper_qpos": [0.0, 0.0],
            },
            "libero_terminated": True,
            "episode_truncated": False,
        },
    ]
    _write_json(output / "states.json", states)
    transcript = {
        "suite": trial.task.suite,
        "task": trial.task.task,
        "seed": trial.task.seed,
        "model": trial.planner.model,
        "elapsed_s": 1.5,
        "finish": {"status": "success", "summary": "fixture"},
        "stats": {
            "elapsed_s": 1.2,
            "turns_used": 2,
            "total_input_tokens": 10,
            "total_output_tokens": 4,
            "tool_calls": 1,
        },
        "messages": [],
    }
    transcript_path = output / "transcript_fixture.json"
    _write_json(transcript_path, transcript)
    attestation = attest_runtime_checkpoint_clients(
        {
            "model": _CheckpointClient(
                "pi0.5_vla",
                trial.runtime.pi05_checkpoint_id,
                "/models/pi05",
            ),
            "sam3_client": _CheckpointClient(
                "sam3",
                trial.runtime.sam3_checkpoint_id,
                "/models/sam3.pt",
            ),
        },
        trial.runtime,
        trial_id=trial.trial_id,
        manifest_id=plan.manifest_id,
        plan_id=plan.plan_id,
        source_revision=trial.source_revision,
    )
    runtime_identity_path = write_runtime_attestation(
        attestation,
        output / "runtime_identity.json",
    )
    runtime_identity_sha = hashlib.sha256(
        runtime_identity_path.read_bytes()
    ).hexdigest()
    reset = {
        "schema_version": "rpent.research-reset-identity/v1",
        "trial_id": trial.trial_id,
        "manifest_id": plan.manifest_id,
        "plan_id": plan.plan_id,
        "source_revision": trial.source_revision,
        "suite": trial.task.suite,
        "task": trial.task.task,
        "seed": trial.task.seed,
        "max_episode_steps": trial.runtime.max_episode_steps,
        "reset_id": "live-reset-7",
        "observed_after_reset": True,
        "source": "live_env_runtime_probe",
        "probe_schema_version": "rpent.runtime-probe/v1",
        "probe_component": "libero_env",
        "runtime_attestation_id": attestation.attestation_id,
        "runtime_attestation_sha256": runtime_identity_sha,
    }
    reset_path = output / "reset_identity.json"
    _write_json(reset_path, reset)
    reset_sha = hashlib.sha256(reset_path.read_bytes()).hexdigest()
    direct_attempts_path = _write_direct_attempts(
        trial,
        plan,
        reset_id=reset["reset_id"],
        reset_sha256=reset_sha,
        attestation_id=attestation.attestation_id,
        attestation_sha256=runtime_identity_sha,
    )
    completion = {
        "schema_version": "rpent.research-completion/v1",
        "trial_id": trial.trial_id,
        "manifest_id": plan.manifest_id,
        "plan_id": plan.plan_id,
        "source_revision": trial.source_revision,
        "status": "finish_declared",
        "agent_error": None,
        "transcript_path": str(transcript_path.resolve()),
        "transcript_sha256": hashlib.sha256(
            transcript_path.read_bytes()
        ).hexdigest(),
        "elapsed_s": 1.5,
        "planner_backend": trial.planner.backend,
        "planner_model": trial.planner.model,
        "planner_base_url": trial.planner.base_url,
        "runtime_attestation_id": attestation.attestation_id,
        "runtime_attestation_path": str(runtime_identity_path.resolve()),
        "runtime_attestation_sha256": runtime_identity_sha,
        "reset_id": reset["reset_id"],
        "reset_identity_path": str(reset_path.resolve()),
        "reset_identity_sha256": reset_sha,
        "direct_vla_attempts_path": str(direct_attempts_path.resolve()),
        "direct_vla_attempts_sha256": hashlib.sha256(
            direct_attempts_path.read_bytes()
        ).hexdigest(),
    }
    _write_json(output / "completion.json", completion)


def test_full_agent_summary_binds_run_local_identity_and_direct_vla_costs(
    tmp_path,
) -> None:
    _manifest, trial, plan = _baseline_context(tmp_path)
    _write_full_agent_artifacts(trial, plan)

    outcome = summarize_full_agent_trial(trial, plan=plan)

    assert outcome.identity.reset_id == "live-reset-7"
    assert outcome.labels.task_success.value is True
    assert outcome.labels.llm_finish.value is True
    assert outcome.handoff_occurred
    assert outcome.costs.vla_invocations == 2
    assert outcome.costs.vla_chunks == 2
    assert outcome.costs.vla_env_actions is None
    assert outcome.costs.system_analytic_time_s == 0.0
    assert outcome.costs.intervention_count == 0
    assert outcome.costs.recovery_retry_cost == 0.0
    assert outcome.metadata["record_scope"] == "full_agent_episode"
    assert outcome.metadata["protocol_adherent"] is True
    assert outcome.metadata["manifest_id"] == plan.manifest_id
    assert outcome.metadata["execution_plan_id"] == plan.plan_id
    assert outcome.metadata["reset_identity_evidence"] == (
        "run_local_post_reset_runtime_probe"
    )


def test_full_agent_summary_rejects_completion_not_bound_to_transcript(tmp_path) -> None:
    _manifest, trial, plan = _baseline_context(tmp_path)
    _write_full_agent_artifacts(trial, plan)
    completion_path = Path(trial.output_dir) / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["transcript_sha256"] = "0" * 64
    completion_path.write_text(
        json.dumps(completion, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FullAgentSummaryError, match="does not bind"):
        summarize_full_agent_trial(trial, plan=plan)


def test_full_agent_summary_rejects_tampered_direct_vla_hash_chain(tmp_path) -> None:
    _manifest, trial, plan = _baseline_context(tmp_path)
    _write_full_agent_artifacts(trial, plan)
    attempts_path = Path(trial.output_dir) / "direct_vla_attempts.jsonl"
    lines = attempts_path.read_text(encoding="utf-8").splitlines()
    terminal = json.loads(lines[1])
    terminal["previous_event_sha256"] = "0" * 64
    lines[1] = json.dumps(
        terminal,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    attempts_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    completion_path = Path(trial.output_dir) / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["direct_vla_attempts_sha256"] = hashlib.sha256(
        attempts_path.read_bytes()
    ).hexdigest()
    _write_json(completion_path, completion)

    with pytest.raises(FullAgentSummaryError, match="direct-VLA"):
        summarize_full_agent_trial(trial, plan=plan)
