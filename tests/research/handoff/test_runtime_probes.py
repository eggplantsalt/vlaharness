from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rpent.research.handoff.artifacts import (
    ModelArtifactManifest,
    SourceIdentity,
    _artifact_id as model_artifact_id,
)
from rpent.research.handoff.experiments.probes import (
    HiddenStateConclusion,
    ProbeSafety,
    ProbeStatus,
    REQUIRED_RUNTIME_FACTS,
    RuntimeProbeArtifact,
    RuntimeProbeOptions,
    probe_nvidia_smi,
    run_runtime_probes,
)
from rpent.research.handoff.experiments.config import ExperimentConfig
from rpent.research.handoff.experiments.full_agent_outcomes import (
    load_probe_reset_map,
)
from rpent.research.handoff.experiments.manifest import (
    expand_manifest,
    verify_manifest_external_bindings,
)
from rpent.research.handoff.experiments.runtime_identity import (
    attest_runtime_checkpoint_clients,
)
from rpent.research.handoff.experiments.runtime import (
    load_gate0_job,
    verify_gate0_job_external_bindings,
)
from rpent.research.handoff.features import FeaturePreset, make_feature_spec


class _EnvClient:
    def __init__(self) -> None:
        self.runtime_calls = 0
        self.diagnostic_calls = 0

    def runtime_probe(self):
        self.runtime_calls += 1
        return {
            "python": "3.11.9",
            "packages": {"mujoco": "3.2.0", "rlinf-libero": "fixture"},
            "environment_class": "fixture.LiberoEnv",
            "server_meta": {
                "suite": "libero_object",
                "task": 2,
                "seed": 7,
                "max_episode_steps": 100,
            },
            "runtime_meta": {"reset_id": "reset-004", "libero_type": "pro"},
            "policy_observation_fields": {
                "states": {"available": True, "shape": [8], "dtype": "float32"},
                "main_images": {
                    "available": True,
                    "shape": [256, 256, 3],
                    "dtype": "uint8",
                },
                "wrist_images": {"available": False},
            },
            "raw_observation_fields": {
                "robot0_eef_pos": {
                    "available": True,
                    "shape": [3],
                    "dtype": "float32",
                },
                "secret_object_pos": {
                    "available": True,
                    "shape": [3],
                    "dtype": "float32",
                },
            },
            "state_sample": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04, 0.04],
            "state_sample_provenance": (
                "experiment_only_runtime_diagnostic; not a policy feature source"
            ),
            "cuda": {
                "available": True,
                "device_count": 1,
                "memory_reserved_bytes": 1024,
            },
        }

    def diagnostic_chunk_step(self, actions):
        self.diagnostic_calls += 1
        return (
            [{"states": [0.0]} for _ in range(3)],
            [0.0, 1.0, 0.0],
            [False, True, True],
            [False, False, False],
            {},
        )


class _VlaClient:
    def __init__(self, checkpoint_id: str = "sha256:pi05-test") -> None:
        self.runtime_arguments = []
        self.checkpoint_id = checkpoint_id

    def healthz(self):
        return {"ok": True, "component": "vla"}

    def runtime_probe(self, observation=None):
        self.runtime_arguments.append(observation)
        payload = {
            "schema_version": "rpent.runtime-probe/v1",
            "component": "pi0.5_vla",
            "python": "3.11.9",
            "packages": {"torch": "2.5.0", "rlinf-openpi": "fixture"},
            "checkpoint": {
                "path": "/models/pi05",
                "exists": True,
                "mtime_ns": 123,
                "configured_id": self.checkpoint_id,
            },
            "model_class": "fixture.Pi05",
            "model_config": {"num_action_chunks": 5, "action_dim": 7},
            "configured_action_shape": {"batch": 1, "chunk": 5, "action_dim": 7},
            "actual_action_shape": None,
            "cuda": {
                "available": True,
                "device_count": 1,
                "memory_allocated_bytes": 2048,
            },
        }
        if observation is not None:
            payload["inference_probe"] = {
                "action_shape": [5, 7],
                "action_dtype": "float32",
                "finite": True,
                "executed_in_environment": False,
            }
        return payload


class _Sam3Client:
    def __init__(self, checkpoint_id: str = "sha256:sam3-test") -> None:
        self.segment_calls = 0
        self.checkpoint_id = checkpoint_id

    def healthz(self):
        return {"ok": True, "component": "sam3"}

    def runtime_probe(self):
        return {
            "schema_version": "rpent.runtime-probe/v1",
            "component": "sam3",
            "python": "3.11.9",
            "packages": {"sam3": "0.1.4", "torch": "2.5.0"},
            "checkpoint": {
                "path": "/models/sam3.pt",
                "exists": True,
                "configured_id": self.checkpoint_id,
            },
            "model_class": "fixture.Sam3",
            "image_contract": {
                "wire_encoding": "base64 encoded image bytes",
                "in_memory_client_method": "Sam3Client.segment_image",
                "point_convention": "[row, col]",
                "exactly_one_prompt": True,
            },
            "cuda": {"available": True, "device_count": 1},
        }

    def segment_image(self, image, *, text_prompt):
        self.segment_calls += 1
        assert image == "current-rgb-frame"
        assert text_prompt == "target mug"
        return SimpleNamespace(
            found=True,
            score=0.91,
            mask_shape=(256, 256),
            reason=None,
        )


def _versions():
    return {
        "python": "3.11.9",
        "implementation": "CPython",
        "platform": "fixture-linux",
        "packages": {"rpent": "0.0.0"},
    }


def _ready_probe_artifact(timestamp: str = "2026-08-09T00:00:00Z"):
    report = run_runtime_probes(
        env_client=_EnvClient(),
        vla_client=_VlaClient(),
        sam3_client=_Sam3Client(),
        local_version_probe=_versions,
    )
    required = (
        "vla.model_checkpoint_identity",
        "sam3.model_checkpoint_identity",
    )
    assert all(report.fact(name).status is ProbeStatus.OBSERVED for name in required)
    return RuntimeProbeArtifact(
        report=report,
        timestamp_utc=timestamp,
        ok=True,
        probe_calls_ok=True,
        readiness_ok=True,
        required_observed_facts=required,
    )


def test_gate0_job_binds_handoff_and_probe_bytes_and_rechecks_them(tmp_path) -> None:
    required = (
        "env.endpoint_health",
        "env.reset_identity",
        "vla.endpoint_health",
        "vla.model_checkpoint_identity",
        "sam3.endpoint_health",
        "sam3.model_checkpoint_identity",
    )
    base_artifact = _ready_probe_artifact()
    artifact = RuntimeProbeArtifact(
        **{
            **base_artifact.model_dump(mode="json", exclude_none=False),
            "required_observed_facts": required,
        }
    )
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(artifact.canonical_json() + "\n", encoding="utf-8")
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": "rpent.libero-handoff-runtime/v1",
                "enabled": True,
                "core": {},
                "controller_method": "gate0_direct_frozen_pi0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    job_path = tmp_path / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "schema_version": "rpent.handoff-gate0-job/v2",
                "output_dir": "outputs",
                "adapter_factory": "robots.fake:factory",
                "adapter_config": {
                    "runtime": {
                        "pi05_checkpoint_path": "/models/pi05",
                        "sam3_checkpoint_path": "/models/sam3.pt",
                        "pi05_checkpoint_id": "sha256:pi05-test",
                        "sam3_checkpoint_id": "sha256:sam3-test",
                    },
                    "handoff_config": "handoff.json",
                },
                "runtime_probes": [
                    {
                        "name": "read-only",
                        "path": "probe.json",
                        "required_observed_facts": list(required),
                    }
                ],
                "gate0": {},
                "run_id": "gate0-run",
                "suite": "libero_object",
                "task_id": 2,
                "seed": 7,
                "target_id": "cup",
                "target_description": "the cup",
                "skill_name": "pick",
                "skill_prompt": "pick the cup",
                "controller_method": "gate0_direct_frozen_pi0",
                "checkpoint_id": "sha256:pi05-test",
                "source_revision": "git:test-source",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    mismatched_payload = artifact.model_dump(mode="json", exclude_none=False)
    reset_fact = next(
        fact
        for fact in mismatched_payload["report"]["facts"]
        if fact["name"] == "env.reset_identity"
    )
    reset_fact["value"]["context"]["seed"] = 8
    probe_path.write_text(
        RuntimeProbeArtifact.model_validate(mismatched_payload).canonical_json()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reset context"):
        load_gate0_job(job_path)
    probe_path.write_text(artifact.canonical_json() + "\n", encoding="utf-8")

    job = load_gate0_job(job_path)

    assert job.external_bindings is not None
    assert job.external_bindings.handoff_config_sha256 == hashlib.sha256(
        handoff_path.read_bytes()
    ).hexdigest()
    assert job.external_bindings.runtime_probes[0].sha256 == hashlib.sha256(
        probe_path.read_bytes()
    ).hexdigest()
    probe_path.write_text(
        probe_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="probe bytes changed"):
        verify_gate0_job_external_bindings(job)


def test_read_only_probe_covers_unknowns_without_running_diagnostics() -> None:
    env = _EnvClient()
    vla = _VlaClient()
    sam3 = _Sam3Client()

    report = run_runtime_probes(
        env_client=env,
        vla_client=vla,
        sam3_client=sam3,
        local_version_probe=_versions,
    )

    assert tuple(fact.name for fact in report.facts) == REQUIRED_RUNTIME_FACTS
    assert env.runtime_calls == 1
    assert env.diagnostic_calls == 0
    assert vla.runtime_arguments == [None]
    assert sam3.segment_calls == 0
    assert report.fact("env.observation_keys").status is ProbeStatus.OBSERVED
    assert report.fact("env.observation_keys").value["raw_values_included"] is False
    assert report.fact("env.state_field_values").policy_eligible is False
    assert "experiment_only" in report.fact("env.state_field_values").value["provenance"]
    assert report.fact("vla.actual_action_shape").status is ProbeStatus.REQUIRES_DIAGNOSTIC
    assert report.fact("sam3.current_observation_acceptance").status is ProbeStatus.REQUIRES_DIAGNOSTIC
    assert report.fact("env.termination_truncation_arrays").safety is ProbeSafety.DESTRUCTIVE_ENVIRONMENT
    assert report.fact("diagnostic.vla_hidden_episode_state").status is ProbeStatus.REQUIRES_DIAGNOSTIC


def test_explicit_diagnostic_gates_collect_runtime_shapes_and_behavior() -> None:
    env = _EnvClient()
    vla = _VlaClient()
    sam3 = _Sam3Client()
    options = RuntimeProbeOptions(
        run_host_gpu_discovery=True,
        run_vla_inference=True,
        run_sam3_inference=True,
        run_destructive_chunk_diagnostic=True,
        run_hidden_state_diagnostic=True,
        fresh_env_reset_confirmed=True,
        isolated_env_trial_confirmed=True,
        isolated_model_session_confirmed=True,
    )

    def hidden_state_diagnostic(client):
        assert client is vla
        return {
            "protocol": "controlled reset/interleaving fixture",
            "conclusion": HiddenStateConclusion.NO_STATE_DEPENDENCE_OBSERVED,
            "repetitions": 4,
            "isolated_model_session": True,
            "reset_controlled": True,
            "stochasticity_controlled": True,
            "evidence": {"sequence_pairs": 2},
        }

    report = run_runtime_probes(
        env_client=env,
        vla_client=vla,
        sam3_client=sam3,
        options=options,
        vla_probe_observation={"states": [0.0] * 8, "main_images": "fixture"},
        sam3_probe_image="current-rgb-frame",
        sam3_text_prompt="target mug",
        chunk_diagnostic_actions=[[0.0] * 7 for _ in range(3)],
        host_gpu_probe=lambda: {
            "device_count": 1,
            "devices": [{"index": 0, "memory_total_mib": 24_576}],
        },
        hidden_state_diagnostic=hidden_state_diagnostic,
        local_version_probe=_versions,
    )

    assert env.diagnostic_calls == 1
    assert sam3.segment_calls == 1
    assert vla.runtime_arguments[0] is not None
    assert report.fact("host.cuda_gpu").status is ProbeStatus.OBSERVED
    assert report.fact("vla.actual_action_shape").value["shape"] == [5, 7]
    assert report.fact("vla.actual_chunk_size").value["chunk_size"] == 5
    assert report.fact("sam3.current_observation_acceptance").value[
        "accepted_current_observation"
    ] is True
    arrays = report.fact("env.termination_truncation_arrays")
    assert arrays.value["terminated"]["shape"] == [3]
    behavior = report.fact("diagnostic.chunk_done_mid_chunk_behavior")
    assert behavior.status is ProbeStatus.REQUIRES_DIAGNOSTIC
    assert behavior.value["first_done_index"] == 1
    assert behavior.value["rpc_entries_returned_after_first_done"] == 1
    assert behavior.value["physical_step_continuation_established"] is False
    hidden = report.fact("diagnostic.vla_hidden_episode_state")
    assert hidden.status is ProbeStatus.OBSERVED
    assert hidden.value["conclusion"] == "no_state_dependence_observed"


def test_destructive_and_hidden_state_diagnostics_need_strong_authorization() -> None:
    with pytest.raises(ValidationError, match="fresh reset"):
        RuntimeProbeOptions(run_destructive_chunk_diagnostic=True)

    with pytest.raises(ValidationError, match="isolated model session"):
        RuntimeProbeOptions(run_hidden_state_diagnostic=True)

    with pytest.raises(ValidationError, match="episode-local model state"):
        RuntimeProbeOptions(run_vla_inference=True)


def test_inconclusive_diagnostics_and_endpoint_errors_are_not_promoted() -> None:
    class BrokenEnv:
        def runtime_probe(self):
            raise ConnectionError("env endpoint refused connection")

    report = run_runtime_probes(
        env_client=BrokenEnv(),
        local_version_probe=_versions,
    )

    assert report.fact("env.endpoint_health").status is ProbeStatus.ERROR
    assert report.fact("env.observation_keys").status is ProbeStatus.ERROR
    assert report.fact("vla.endpoint_health").status is ProbeStatus.UNAVAILABLE
    assert report.errors

    options = RuntimeProbeOptions(
        run_hidden_state_diagnostic=True,
        isolated_model_session_confirmed=True,
    )
    inconclusive = run_runtime_probes(
        vla_client=_VlaClient(),
        options=options,
        hidden_state_diagnostic=lambda _client: {
            "protocol": "repeat-only fixture",
            "conclusion": "inconclusive",
            "repetitions": 2,
            "isolated_model_session": True,
            "reset_controlled": False,
            "stochasticity_controlled": False,
            "evidence": {"reason": "no reset API"},
        },
        local_version_probe=_versions,
    )
    hidden = inconclusive.fact("diagnostic.vla_hidden_episode_state")
    assert hidden.status is ProbeStatus.REQUIRES_DIAGNOSTIC
    assert hidden.value["conclusion"] == "inconclusive"


def test_nvidia_smi_helper_uses_fixed_non_shell_command_and_parses_memory() -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "0, NVIDIA RTX 6000 Ada, GPU-fixture, 49140, 1024, 550.54.15\n"
            ),
            stderr="",
        )

    result = probe_nvidia_smi(runner=runner, timeout_s=3.0)

    command, kwargs = calls[0]
    assert command[0] == "nvidia-smi"
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == 3.0
    assert result["device_count"] == 1
    assert result["devices"][0]["memory_total_mib"] == 49_140
    assert result["devices"][0]["memory_used_mib"] == 1024


def test_runtime_probe_artifact_envelope_round_trips_strictly() -> None:
    artifact = _ready_probe_artifact()

    encoded = artifact.canonical_json()
    decoded = RuntimeProbeArtifact.model_validate_json(encoded)

    assert decoded == artifact
    assert decoded.schema_version == "rpent.runtime-probe-artifact/v1"
    assert decoded.report.schema_version == "rpent.handoff-runtime-probe/v1"
    assert decoded.report.fact("vla.model_checkpoint_identity").value[
        "checkpoint"
    ]["configured_id"] == "sha256:pi05-test"


def test_full_agent_probe_reset_consumer_reads_typed_artifact_envelope(
    tmp_path,
) -> None:
    artifact = _ready_probe_artifact()
    path = tmp_path / "runtime-probe.json"
    path.write_text(artifact.canonical_json() + "\n", encoding="utf-8")

    reset_map = load_probe_reset_map((path,))

    assert reset_map == {("libero_object", 2, 7): "reset-004"}


def test_manifest_identity_binds_runtime_probe_bytes_and_detects_mutation(
    tmp_path,
) -> None:
    probe_path = tmp_path / "runtime-probe.json"
    first_artifact = _ready_probe_artifact("2026-08-09T00:00:00Z")
    probe_path.write_text(first_artifact.canonical_json() + "\n", encoding="utf-8")
    config = ExperimentConfig.model_validate(
        {
            "experiment_id": "probe-binding-test",
            "output_root": str(tmp_path / "outputs"),
            "tasks": [
                {
                    "suite": "libero_object",
                    "task": 0,
                    "seeds": [0],
                    "target_id": "mug",
                    "target_description": "the mug",
                    "skill_name": "pick",
                    "skill_prompt": "pick the mug",
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
                    "name": "read_only",
                    "path": str(probe_path),
                    "required_observed_facts": [
                        "vla.model_checkpoint_identity",
                        "sam3.model_checkpoint_identity",
                    ],
                }
            ],
            "source_revision": "git:test-source",
        }
    )

    first = expand_manifest(config)
    expected_sha = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    assert first.runtime_probe_bindings[0].sha256 == expected_sha
    assert first.trials[0].artifact_bindings[
        "runtime_probe_read_only_sha256"
    ] == expected_sha

    second_artifact = _ready_probe_artifact("2026-08-09T00:00:01Z")
    probe_path.write_text(second_artifact.canonical_json() + "\n", encoding="utf-8")
    second = expand_manifest(config)

    assert second.manifest_id != first.manifest_id
    assert second.trials[0].trial_id != first.trials[0].trial_id
    with pytest.raises(ValueError, match="runtime probe checksum changed"):
        verify_manifest_external_bindings(first)


def test_live_checkpoint_attestation_accepts_exact_ids_and_rejects_mismatch() -> None:
    runtime = SimpleNamespace(
        pi05_checkpoint_id="sha256:pi05-test",
        pi05_checkpoint_path=None,
        vla_endpoint="http://vla.example:8011",
        sam3_checkpoint_id="sha256:sam3-test",
        sam3_checkpoint_path=None,
        sam3_endpoint="http://sam.example:8012",
    )
    clients = {"model": _VlaClient(), "sam3_client": _Sam3Client()}

    attestation = attest_runtime_checkpoint_clients(
        clients,
        runtime,
        trial_id="trial-fixture",
        manifest_id="manifest-fixture",
        plan_id="plan-fixture",
        source_revision="git:test-source",
    )

    assert tuple(item.component for item in attestation.observations) == (
        "pi0.5_vla",
        "sam3",
    )
    assert all(item.external_endpoint for item in attestation.observations)
    assert attestation.observations[0].observed_checkpoint_id == (
        "sha256:pi05-test"
    )

    mismatched = {**clients, "model": _VlaClient("sha256:wrong-pi05")}
    with pytest.raises(RuntimeError, match="live checkpoint ID mismatch"):
        attest_runtime_checkpoint_clients(
            mismatched,
            runtime,
            trial_id="trial-fixture",
        )


def test_model_source_identity_must_bind_one_manifest_runtime_probe(tmp_path) -> None:
    probe_path = tmp_path / "runtime-probe.json"
    probe = _ready_probe_artifact()
    probe_path.write_text(probe.canonical_json() + "\n", encoding="utf-8")
    probe_sha = hashlib.sha256(probe_path.read_bytes()).hexdigest()

    model_root = tmp_path / "model"
    model_root.mkdir()
    estimator = model_root / "estimator.joblib"
    estimator.write_bytes(b"fixture estimator bytes")
    provisional = ModelArtifactManifest(
        artifact_id="pending",
        model_kind="StaticFixtureModel",
        estimator_sha256=hashlib.sha256(estimator.read_bytes()).hexdigest(),
        feature_spec=make_feature_spec(
            FeaturePreset.ABSOLUTE,
            skill_vocabulary=("pick",),
        ),
        training_target_label="primitive_success",
        calibration_method="none",
        dataset_fingerprint="dataset-fixture",
        split_assignment_fingerprint="split-fixture",
        training_record_ids=("train-1",),
        calibration_record_ids=("calibration-1",),
        held_out_record_ids=("test-1",),
        training_configuration={"fixture": True},
        source_identity=SourceIdentity(
            source_revision="git:test-source",
            external_runtime_identity=str(probe_path.resolve()),
            external_runtime_identity_sha256=probe_sha,
        ),
    )
    model_manifest = provisional.model_copy(
        update={"artifact_id": model_artifact_id(provisional)}
    )
    (model_root / "manifest.json").write_text(
        model_manifest.canonical_json() + "\n",
        encoding="utf-8",
    )
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(
        """{
  "schema_version": "rpent.libero-handoff-runtime/v1",
  "enabled": true,
  "controller_method": "outcome_calibrated_switching",
  "core": {"model_artifact": "${GENERIC_MODEL}"},
  "metadata": {}
}
""",
        encoding="utf-8",
    )
    config = ExperimentConfig.model_validate(
        {
            "experiment_id": "model-probe-binding-test",
            "output_root": str(tmp_path / "outputs"),
            "tasks": [
                {
                    "suite": "libero_object",
                    "task": 0,
                    "seeds": [0],
                    "target_id": "mug",
                    "target_description": "the mug",
                    "skill_name": "pick",
                    "skill_prompt": "pick the mug",
                    "label_source": "primitive_heuristic",
                }
            ],
            "conditions": [
                {
                    "name": "ours",
                    "execution_layer": "controlled",
                    "method": "outcome_calibrated_switching",
                    "handoff_enabled": True,
                    "handoff_config": str(handoff_path),
                    "model_artifact": str(model_root),
                    "model_artifact_id": model_manifest.artifact_id,
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
                    "name": "read_only",
                    "path": str(probe_path),
                    "required_observed_facts": [
                        "vla.model_checkpoint_identity",
                        "sam3.model_checkpoint_identity",
                    ],
                }
            ],
            "source_revision": "git:test-source",
        }
    )

    trial = expand_manifest(config).trials[0]

    assert trial.artifact_bindings["model_runtime_identity_sha256"] == probe_sha
    assert trial.artifact_bindings["model_source_revision"] == "git:test-source"

    wrong_source = model_manifest.source_identity.model_copy(
        update={"external_runtime_identity_sha256": "0" * 64}
    )
    wrong_provisional = model_manifest.model_copy(
        update={"artifact_id": "pending", "source_identity": wrong_source}
    )
    wrong_manifest = wrong_provisional.model_copy(
        update={"artifact_id": model_artifact_id(wrong_provisional)}
    )
    (model_root / "manifest.json").write_text(
        wrong_manifest.canonical_json() + "\n",
        encoding="utf-8",
    )
    wrong_condition = config.conditions[0].model_copy(
        update={"model_artifact_id": wrong_manifest.artifact_id}
    )
    with pytest.raises(ValueError, match="not one of the manifest-bound"):
        expand_manifest(config.model_copy(update={"conditions": (wrong_condition,)}))
