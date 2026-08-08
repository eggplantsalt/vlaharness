from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rpent.research.handoff.evaluation.metrics import evaluate_system_records
from rpent.research.handoff.experiments.config import (
    ExperimentConfig,
    stable_digest,
)
from rpent.research.handoff.experiments.full_agent import build_child_plan
from rpent.research.handoff.experiments.full_agent_outcomes import (
    FullAgentSummaryError,
    summarize_full_agent_trial,
)
from rpent.research.handoff.experiments.lifecycle import (
    LifecycleJournal,
    TrialEventType,
)
from rpent.research.handoff.experiments.manifest import (
    expand_manifest,
    write_manifest,
)
from rpent.research.handoff.experiments.probes import (
    ProbeStatus,
    RuntimeProbeArtifact,
    run_runtime_probes,
)
from rpent.research.handoff.experiments.runtime_identity import (
    attest_runtime_checkpoint_clients,
    write_runtime_attestation,
)


def _baseline_context(tmp_path):
    probe_report = run_runtime_probes(
        vla_client=_CheckpointClient(
            "pi0.5_vla",
            "sha256:pi05-test",
            "/models/pi05",
        ),
        sam3_client=_CheckpointClient(
            "sam3",
            "sha256:sam3-test",
            "/models/sam3.pt",
        ),
        local_version_probe=lambda: {
            "python": "3.11.9",
            "implementation": "CPython",
            "platform": "fixture-linux",
            "packages": {"rpent": "fixture"},
        },
    )
    required_probe_facts = (
        "vla.model_checkpoint_identity",
        "sam3.model_checkpoint_identity",
    )
    assert all(
        probe_report.fact(name).status is ProbeStatus.OBSERVED
        for name in required_probe_facts
    )
    probe_artifact = RuntimeProbeArtifact(
        report=probe_report,
        timestamp_utc="2026-08-09T00:00:00Z",
        ok=True,
        probe_calls_ok=True,
        readiness_ok=True,
        required_observed_facts=required_probe_facts,
    )
    probe_path = tmp_path / "runtime-probe.json"
    probe_path.write_text(
        probe_artifact.canonical_json() + "\n",
        encoding="utf-8",
    )
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
            "runtime_probes": [
                {
                    "name": "full-agent-fixture",
                    "path": str(probe_path),
                    "required_observed_facts": list(required_probe_facts),
                }
            ],
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


def _refresh_completion_states_binding(output: Path) -> None:
    states_path = output / "states.json"
    completion_path = output / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["states_path"] = str(states_path.resolve())
    completion["states_sha256"] = hashlib.sha256(states_path.read_bytes()).hexdigest()
    _write_json(completion_path, completion)


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


def _rewrite_direct_attempts_with_same_step_retry(path: Path) -> None:
    seed = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    base = {
        key: value
        for key, value in seed.items()
        if key
        not in {
            "event_sequence",
            "previous_event_sha256",
            "attempt_index",
            "phase",
            "elapsed_s",
            "error_type",
            "error",
            "event_sha256",
        }
    }
    specifications = (
        (1, "started", None, None, None),
        (
            1,
            "error",
            0.1,
            "RuntimeError",
            "first attempt failed before state dump",
        ),
        (2, "started", None, None, None),
        (2, "completed", 0.4, None, None),
    )
    records = []
    previous_sha256 = None
    for sequence, (attempt, phase, elapsed, error_type, error) in enumerate(
        specifications,
        start=1,
    ):
        record = _direct_attempt_record(
            {
                **base,
                "event_sequence": sequence,
                "previous_event_sha256": previous_sha256,
                "attempt_index": attempt,
                # Both planner-visible attempts belong to the same RPent step
                # and tool; ordering is carried by attempt/event sequence.
                "step_index": 1,
                "phase": phase,
                "elapsed_s": elapsed,
                "error_type": error_type,
                "error": error,
            }
        )
        records.append(record)
        previous_sha256 = record["event_sha256"]
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
            for record in records
        ),
        encoding="utf-8",
    )


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
        "schema_version": "rpent.research-completion/v2",
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
        "states_path": str((output / "states.json").resolve()),
        "states_sha256": hashlib.sha256(
            (output / "states.json").read_bytes()
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


def _completed_lifecycle(trial, plan):
    journal = LifecycleJournal(
        Path(trial.output_dir).parent / f"{trial.trial_id}-lifecycle.jsonl",
        allowed_trial_ids={trial.trial_id},
    )
    journal.append(
        trial.trial_id,
        TrialEventType.STARTED,
        timestamp_utc="2026-08-09T00:00:00Z",
        artifact_path=plan.output_dir,
        details={"plan_id": plan.plan_id},
    )
    journal.append(
        trial.trial_id,
        TrialEventType.COMPLETED,
        timestamp_utc="2026-08-09T00:00:02Z",
        artifact_path=plan.output_dir,
        details={"plan_id": plan.plan_id, "returncode": 0},
    )
    return journal.read()


def _failed_lifecycle(trial, plan):
    journal = LifecycleJournal(
        Path(trial.output_dir).parent / f"{trial.trial_id}-failed.jsonl",
        allowed_trial_ids={trial.trial_id},
    )
    journal.append(
        trial.trial_id,
        TrialEventType.STARTED,
        timestamp_utc="2026-08-09T00:00:00Z",
        artifact_path=plan.output_dir,
        details={"plan_id": plan.plan_id},
    )
    journal.append(
        trial.trial_id,
        TrialEventType.FAILED,
        timestamp_utc="2026-08-09T00:00:01Z",
        message="fixture child failed before completion",
        artifact_path=plan.output_dir,
        details={"plan_id": plan.plan_id, "returncode": 1},
    )
    return journal.read()


def _started_lifecycle(trial, plan):
    journal = LifecycleJournal(
        Path(trial.output_dir).parent / f"{trial.trial_id}-started.jsonl",
        allowed_trial_ids={trial.trial_id},
    )
    journal.append(
        trial.trial_id,
        TrialEventType.STARTED,
        timestamp_utc="2026-08-09T00:00:00Z",
        artifact_path=plan.output_dir,
        details={"plan_id": plan.plan_id},
    )
    return journal.read()


def test_full_agent_summary_binds_run_local_identity_and_direct_vla_costs(
    tmp_path,
) -> None:
    _manifest, trial, plan = _baseline_context(tmp_path)
    _write_full_agent_artifacts(trial, plan)
    lifecycle_events = _completed_lifecycle(trial, plan)

    outcome = summarize_full_agent_trial(
        trial,
        plan=plan,
        lifecycle_events=lifecycle_events,
    )

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
    assert outcome.metadata["direct_vla_tool_calls"] == 1
    assert outcome.metadata["direct_vla_tool_attempts"] == 1
    assert outcome.metadata["manifest_id"] == plan.manifest_id
    assert outcome.metadata["execution_plan_id"] == plan.plan_id
    assert outcome.metadata["system_attempt_success"] is True
    assert evaluate_system_records((outcome,))[
        "end_to_end_attempt_success_rate"
    ].value == pytest.approx(1.0)
    assert outcome.metadata["reset_identity_evidence"] == (
        "run_local_post_reset_runtime_probe"
    )


def test_full_agent_summary_rejects_completion_not_bound_to_transcript(tmp_path) -> None:
    _manifest, trial, plan = _baseline_context(tmp_path)
    _write_full_agent_artifacts(trial, plan)
    lifecycle_events = _completed_lifecycle(trial, plan)
    completion_path = Path(trial.output_dir) / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["transcript_sha256"] = "0" * 64
    completion_path.write_text(
        json.dumps(completion, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FullAgentSummaryError, match="does not bind"):
        summarize_full_agent_trial(
            trial,
            plan=plan,
            lifecycle_events=lifecycle_events,
        )


def test_full_agent_summary_rejects_tampered_direct_vla_hash_chain(tmp_path) -> None:
    _manifest, trial, plan = _baseline_context(tmp_path)
    _write_full_agent_artifacts(trial, plan)
    lifecycle_events = _completed_lifecycle(trial, plan)
    attempts_path = Path(trial.output_dir) / "direct_vla_attempts.jsonl"
    lines = attempts_path.read_text(encoding="utf-8").splitlines()
    terminal = json.loads(lines[1])
    terminal["previous_event_sha256"] = "0" * 64
    terminal.pop("event_sha256")
    terminal = _direct_attempt_record(terminal)
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
        summarize_full_agent_trial(
            trial,
            plan=plan,
            lifecycle_events=lifecycle_events,
        )


def test_full_agent_summary_rejects_states_bytes_not_bound_to_completion(
    tmp_path,
) -> None:
    _manifest, trial, plan = _baseline_context(tmp_path)
    _write_full_agent_artifacts(trial, plan)
    lifecycle_events = _completed_lifecycle(trial, plan)
    states_path = Path(trial.output_dir) / "states.json"
    states_path.write_text(
        states_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    with pytest.raises(FullAgentSummaryError, match="states"):
        summarize_full_agent_trial(
            trial,
            plan=plan,
            lifecycle_events=lifecycle_events,
        )


def test_completed_direct_vla_journal_requires_matching_state_trace(tmp_path) -> None:
    _manifest, trial, plan = _baseline_context(tmp_path)
    _write_full_agent_artifacts(trial, plan)
    lifecycle_events = _completed_lifecycle(trial, plan)
    states_path = Path(trial.output_dir) / "states.json"
    states = json.loads(states_path.read_text(encoding="utf-8"))
    states[1]["command"]["action"] = "move_to"
    _write_json(states_path, states)
    _refresh_completion_states_binding(Path(trial.output_dir))

    with pytest.raises(FullAgentSummaryError, match="direct-VLA"):
        summarize_full_agent_trial(
            trial,
            plan=plan,
            lifecycle_events=lifecycle_events,
        )


def test_completed_episode_preserves_open_direct_vla_attempt_as_unknown_cost(
    tmp_path,
) -> None:
    _manifest, trial, plan = _baseline_context(tmp_path)
    _write_full_agent_artifacts(trial, plan)
    lifecycle_events = _completed_lifecycle(trial, plan)
    attempts_path = Path(trial.output_dir) / "direct_vla_attempts.jsonl"
    started = attempts_path.read_text(encoding="utf-8").splitlines()[0]
    attempts_path.write_text(started + "\n", encoding="utf-8")
    states_path = Path(trial.output_dir) / "states.json"
    states = json.loads(states_path.read_text(encoding="utf-8"))
    # A terminal journal event is written before the corresponding state dump.
    # Therefore a final open start is retainable only when that state was never
    # persisted; an open start plus a direct-action state is contradictory.
    _write_json(states_path, states[:1])
    _refresh_completion_states_binding(Path(trial.output_dir))
    completion_path = Path(trial.output_dir) / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["direct_vla_attempts_sha256"] = hashlib.sha256(
        attempts_path.read_bytes()
    ).hexdigest()
    _write_json(completion_path, completion)

    outcome = summarize_full_agent_trial(
        trial,
        plan=plan,
        lifecycle_events=lifecycle_events,
    )

    assert outcome.metadata["direct_vla_tool_attempts"] == 1
    assert outcome.metadata["direct_vla_tool_calls"] == 1
    assert outcome.handoff_occurred is False
    assert outcome.labels.task_success.value is False
    assert outcome.metadata["system_attempt_success"] is False
    assert outcome.costs.vla_invocations is None
    assert outcome.costs.vla_chunks is None
    assert outcome.costs.vla_time_s is None


def test_open_direct_vla_attempt_with_persisted_state_is_rejected(tmp_path) -> None:
    _manifest, trial, plan = _baseline_context(tmp_path)
    _write_full_agent_artifacts(trial, plan)
    lifecycle_events = _completed_lifecycle(trial, plan)
    attempts_path = Path(trial.output_dir) / "direct_vla_attempts.jsonl"
    started = attempts_path.read_text(encoding="utf-8").splitlines()[0]
    attempts_path.write_text(started + "\n", encoding="utf-8")
    completion_path = Path(trial.output_dir) / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["direct_vla_attempts_sha256"] = hashlib.sha256(
        attempts_path.read_bytes()
    ).hexdigest()
    _write_json(completion_path, completion)

    with pytest.raises(FullAgentSummaryError, match="terminal"):
        summarize_full_agent_trial(
            trial,
            plan=plan,
            lifecycle_events=lifecycle_events,
        )


def test_direct_vla_retry_may_reuse_step_and_tool_after_error(tmp_path) -> None:
    _manifest, trial, plan = _baseline_context(tmp_path)
    _write_full_agent_artifacts(trial, plan)
    lifecycle_events = _completed_lifecycle(trial, plan)
    attempts_path = Path(trial.output_dir) / "direct_vla_attempts.jsonl"
    _rewrite_direct_attempts_with_same_step_retry(attempts_path)
    completion_path = Path(trial.output_dir) / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["direct_vla_attempts_sha256"] = hashlib.sha256(
        attempts_path.read_bytes()
    ).hexdigest()
    _write_json(completion_path, completion)

    outcome = summarize_full_agent_trial(
        trial,
        plan=plan,
        lifecycle_events=lifecycle_events,
    )

    assert outcome.metadata["direct_vla_tool_attempts"] == 2
    assert outcome.metadata["direct_vla_state_records"] == 1
    assert outcome.costs.vla_invocations is None
    assert outcome.costs.vla_chunks is None
    assert outcome.costs.vla_time_s == pytest.approx(0.5)
    assert outcome.costs.recovery_retry_cost == pytest.approx(1.0)


def test_full_agent_summary_rejects_completion_attestation_mismatch(tmp_path) -> None:
    _manifest, trial, plan = _baseline_context(tmp_path)
    _write_full_agent_artifacts(trial, plan)
    lifecycle_events = _completed_lifecycle(trial, plan)
    completion_path = Path(trial.output_dir) / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["runtime_attestation_sha256"] = "0" * 64
    _write_json(completion_path, completion)

    with pytest.raises(FullAgentSummaryError, match="runtime"):
        summarize_full_agent_trial(
            trial,
            plan=plan,
            lifecycle_events=lifecycle_events,
        )


def test_full_agent_failed_lifecycle_without_completion_stays_in_denominator(
    tmp_path,
) -> None:
    _manifest, trial, plan = _baseline_context(tmp_path)
    lifecycle_events = _failed_lifecycle(trial, plan)

    outcome = summarize_full_agent_trial(
        trial,
        plan=plan,
        lifecycle_events=lifecycle_events,
    )

    assert outcome.identity.trial_id == trial.trial_id
    assert outcome.handoff_occurred is False
    assert outcome.labels.task_success.value is None
    assert outcome.costs.vla_invocations is None
    assert outcome.termination.failure_mode.value == "unknown"
    assert outcome.metadata["data_status"] == "observed"
    assert outcome.metadata["incomplete_execution"] is True
    assert outcome.metadata["denominator_eligible"] is True
    assert outcome.metadata["system_attempt_success"] is False
    assert outcome.metadata["manifest_id"] == plan.manifest_id
    assert outcome.metadata["execution_plan_id"] == plan.plan_id
    assert evaluate_system_records((outcome,))[
        "end_to_end_attempt_success_rate"
    ].value == pytest.approx(0.0)


def test_completed_lifecycle_without_completion_sidecar_is_rejected(tmp_path) -> None:
    _manifest, trial, plan = _baseline_context(tmp_path)
    lifecycle_events = _completed_lifecycle(trial, plan)

    with pytest.raises(FullAgentSummaryError, match="completion"):
        summarize_full_agent_trial(
            trial,
            plan=plan,
            lifecycle_events=lifecycle_events,
        )


def test_started_lifecycle_without_terminal_is_not_materialized(tmp_path) -> None:
    _manifest, trial, plan = _baseline_context(tmp_path)
    lifecycle_events = _started_lifecycle(trial, plan)

    with pytest.raises(FullAgentSummaryError, match="lifecycle"):
        summarize_full_agent_trial(
            trial,
            plan=plan,
            lifecycle_events=lifecycle_events,
        )


def test_failed_direct_vla_start_without_state_preserves_attempt_evidence(
    tmp_path,
) -> None:
    _manifest, trial, plan = _baseline_context(tmp_path)
    _write_full_agent_artifacts(trial, plan)
    output = Path(trial.output_dir)
    attempts_path = output / "direct_vla_attempts.jsonl"
    started_line = attempts_path.read_text(encoding="utf-8").splitlines()[0]
    attempts_path.write_text(started_line + "\n", encoding="utf-8")
    for path in (
        output / "completion.json",
        output / "states.json",
        output / "transcript_fixture.json",
    ):
        path.unlink()
    lifecycle_events = _failed_lifecycle(trial, plan)

    outcome = summarize_full_agent_trial(
        trial,
        plan=plan,
        lifecycle_events=lifecycle_events,
    )

    assert outcome.handoff_occurred is False
    assert outcome.costs.vla_invocations is None
    assert outcome.metadata["direct_vla_tool_calls"] == 1
    assert outcome.metadata["direct_vla_tool_attempts"] == 1
    assert outcome.metadata["incomplete_execution"] is True
