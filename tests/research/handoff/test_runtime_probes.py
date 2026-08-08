from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rpent.research.handoff.experiments.probes import (
    HiddenStateConclusion,
    ProbeSafety,
    ProbeStatus,
    REQUIRED_RUNTIME_FACTS,
    RuntimeProbeOptions,
    probe_nvidia_smi,
    run_runtime_probes,
)


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
    def __init__(self) -> None:
        self.runtime_arguments = []

    def healthz(self):
        return {"ok": True, "component": "vla"}

    def runtime_probe(self, observation=None):
        self.runtime_arguments.append(observation)
        payload = {
            "python": "3.11.9",
            "packages": {"torch": "2.5.0", "rlinf-openpi": "fixture"},
            "checkpoint": {
                "path": "/models/pi05",
                "exists": True,
                "mtime_ns": 123,
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
    def __init__(self) -> None:
        self.segment_calls = 0

    def healthz(self):
        return {"ok": True, "component": "sam3"}

    def runtime_probe(self):
        return {
            "python": "3.11.9",
            "packages": {"sam3": "0.1.4", "torch": "2.5.0"},
            "checkpoint": {"path": "/models/sam3.pt", "exists": True},
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
