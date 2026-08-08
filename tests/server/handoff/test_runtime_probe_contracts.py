from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("omegaconf", reason="server probe contracts need server extras")

from robots.libero.env_client import LiberoEnvClient
from robots.libero.env_server import LiberoEnvFacade
from robots.libero.sam3_server import Sam3Facade
from robots.libero.vla_server import VLAFacade, build_model_cfg
from rpent.utils.rpc import RpcFacade
from rpent.utils.sam3_client import Sam3Client
from rpent.utils.vla_client import VLAClient


pytestmark = pytest.mark.server


class _Rpc:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, args, kwargs, timeout_s))
        value = self.responses[method]
        return value() if callable(value) else value


class _FakeExternalEnv:
    def __init__(self):
        self.current_raw_obs = [
            {
                "robot0_eef_pos": np.array([0.0, 0.0, 0.3]),
                "secret_object_pos": np.array([1.0, 2.0, 3.0]),
            }
        ]


def test_env_runtime_probe_describes_fields_without_raw_values():
    facade = LiberoEnvFacade(
        _FakeExternalEnv(),
        meta={"suite": "suite", "task": 0, "seed": 0, "max_episode_steps": 10},
        runtime_meta={"reset_id": 3, "libero_type": "pro"},
    )
    facade._last_obs = {
        "states": np.arange(8, dtype=np.float32),
        "main_images": np.zeros((4, 4, 3), dtype=np.uint8),
    }
    payload = facade._dispatch("env.runtime_probe", (), {})
    assert payload["schema_version"] == "rpent.runtime-probe/v1"
    assert payload["runtime_meta"]["reset_id"] == 3
    assert payload["raw_observation_fields"]["secret_object_pos"]["shape"] == [3]
    assert "value" not in payload["raw_observation_fields"]["secret_object_pos"]
    assert payload["policy_observation_fields"]["states"]["shape"] == [8]
    assert payload["state_sample_provenance"].startswith("experiment_only")


def test_env_runtime_probe_never_serializes_nonfinite_state_values():
    facade = LiberoEnvFacade(
        _FakeExternalEnv(),
        meta={"suite": "suite", "task": 0, "seed": 0, "max_episode_steps": 10},
        runtime_meta={"reset_id": 3},
    )
    facade._last_obs = {"states": np.array([0.0, np.nan], dtype=np.float32)}
    payload = facade.runtime_probe()
    assert payload["state_sample"] is None
    assert payload["state_sample_error"] == "state contains NaN/Inf"


def test_env_client_exposes_probe_and_explicit_destructive_diagnostic():
    meta = {"suite": "suite", "task": 0, "seed": 0, "max_episode_steps": 10}
    reset_obs = ({"states": np.zeros(8)}, {})
    diagnostic = (
        [{"states": np.zeros(8)}],
        np.array([0.0]),
        np.array([False]),
        np.array([False]),
        {},
    )
    rpc = _Rpc(
        {
            "env.get_env_meta": meta,
            "env.reset": reset_obs,
            "env.runtime_probe": {"component": "libero_env"},
            "env.diagnostic_chunk_step": diagnostic,
        }
    )
    client = LiberoEnvClient(rpc, expected_meta=meta)
    assert client.runtime_probe()["component"] == "libero_env"
    returned = client.diagnostic_chunk_step(np.zeros((1, 7), dtype=np.float32))
    assert returned is diagnostic
    assert any(call[0] == "env.diagnostic_chunk_step" for call in rpc.calls)


def test_vla_runtime_probe_contract_without_model_inference(tmp_path):
    checkpoint = tmp_path / "pi05.ckpt"
    checkpoint.write_bytes(b"probe-fixture")
    facade = object.__new__(VLAFacade)
    RpcFacade.__init__(facade)
    facade._model_path = str(checkpoint)
    facade._model_cfg = build_model_cfg(str(checkpoint))
    facade._model = SimpleNamespace()
    payload = facade._dispatch("vla.runtime_probe", (), {})
    assert payload["component"] == "pi0.5_vla"
    assert payload["checkpoint"]["exists"] is True
    assert payload["configured_action_shape"] == {
        "batch": 1,
        "chunk": 5,
        "action_dim": 7,
    }
    assert payload["actual_action_shape"] is None


def test_vla_client_can_add_nonexecuted_actual_shape_probe():
    rpc = _Rpc(
        {
            "vla.runtime_probe": {"component": "pi0.5_vla"},
            "predict": {
                "actions": np.zeros((1, 5, 7), dtype=np.float32).tolist(),
                "shape": [1, 5, 7],
                "dtype": "float32",
            },
        }
    )
    client = VLAClient(rpc)
    payload = client.runtime_probe(
        {
            "main_images": np.zeros((4, 4, 3), dtype=np.uint8),
            "states": np.zeros(8, dtype=np.float32),
            "task_descriptions": "probe",
        }
    )
    assert payload["inference_probe"]["action_shape"] == [5, 7]
    assert payload["inference_probe"]["executed_in_environment"] is False


def test_sam_runtime_probe_and_in_memory_client_contract():
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)
    )
    engine = SimpleNamespace(
        _checkpoint_path=None,
        _torch=torch_stub,
        _device="cpu",
        _model=SimpleNamespace(),
    )
    facade = Sam3Facade(engine)
    payload = facade._dispatch("sam3.runtime_probe", (), {})
    assert payload["component"] == "sam3"
    assert payload["image_contract"]["in_memory_client_method"] == (
        "Sam3Client.segment_image"
    )

    rpc = _Rpc(
        {
            "segment": {
                "found": False,
                "reason": "fixture miss",
            },
            "sam3.runtime_probe": payload,
        }
    )
    client = Sam3Client(rpc)
    result = client.segment_image(
        np.zeros((4, 4, 3), dtype=np.uint8),
        text_prompt="fixture",
    )
    assert result.found is False
    assert client.runtime_probe()["component"] == "sam3"
