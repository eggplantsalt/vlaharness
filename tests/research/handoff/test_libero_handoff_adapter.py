from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero import tools as libero_tools
from robots.libero.handoff import (
    HANDOFF_TOOL_NAMES,
    CurrentObservationTargetProvider,
    InstrumentedEnvClient,
    LiberoGovernorAdapter,
    OracleTargetProvider,
    RuntimeInstrumentation,
    TargetRequest,
    build_handoff_composite,
)
from robots.libero.handoff_runtime import (
    CoreRuntimeAPI,
    HandoffConfigurationError,
    RuntimeEventSink,
    build_research_sink,
    load_handoff_runtime_config,
)
from robots.libero.toolkit import LiberoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.research.handoff.governor import InMemoryResearchSink
from rpent.research.handoff.features import FeatureBuilder, make_feature_spec
from rpent.research.handoff.types import (
    FeatureAvailability,
    SkillIdentity,
    TargetContext,
    TargetEstimate,
)
from rpent.tools import common
from rpent.tools.toolkit import Toolkit


class _FakeEnv:
    def __init__(self) -> None:
        self.episode_terminated = False
        self.episode_truncated = False
        self.return_all_frames = False
        self.server_meta = {
            "suite": "libero_spatial",
            "task": 1,
            "seed": 2,
            "max_episode_steps": 100,
        }
        self.raw = {
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_eef_pos": np.array([0.0, 0.0, 0.3]),
            "robot0_gripper_qpos": np.array([0.04, -0.04]),
            "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "agentview_depth": np.ones((4, 4, 1), dtype=np.float32),
            "secret_object_pos": np.array([9.0, 9.0, 9.0]),
            "task_success_predicate": True,
        }

    def raw_obs(self):
        return dict(self.raw)

    def get_camera_meta(self, camera_name, height, width):
        del camera_name
        return {
            "intrinsic_K": [
                [100.0, 0.0, (width - 1) / 2.0],
                [0.0, 100.0, (height - 1) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            "extrinsic_cam2world": np.eye(4).tolist(),
        }


class _FakePrimitives:
    def __init__(self) -> None:
        self.env = _FakeEnv()
        self._sam3_client = None
        self._last_obs = {
            "states": np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.04, -0.04]),
            "main_images": np.zeros((4, 4, 3), dtype=np.uint8),
            "task_descriptions": "task",
        }
        self._last_obs_eef_pos = np.array([0.0, 0.0, 0.3], dtype=np.float32)
        self.calls: list[tuple[str, dict]] = []

    def pi0_pick(self, **kwargs):
        self.calls.append(("pi0_pick", dict(kwargs)))
        return {"name": "pick", "success": True, "chunks_used": 1}

    def pi0_doubled(self, **kwargs):
        self.calls.append(("pi0_doubled", dict(kwargs)))
        return {"name": "pi0_doubled", "success": False, "chunks_used": 1}

    def segment(self, **kwargs):
        return {"segment": kwargs}

    def __getattr__(self, name):
        if name in libero_tools.PRIMITIVE_TOOL_NAMES:
            return lambda **kwargs: {"name": name, **kwargs}
        raise AttributeError(name)


class _CaptureProvider:
    def __init__(self) -> None:
        self.raw_keys: set[str] | None = None

    def estimate(
        self,
        *,
        request,
        observation_sequence,
        raw_observation,
        policy_observation,
    ):
        del policy_observation
        self.raw_keys = set(raw_observation)
        return TargetContext(
            target_id=request.target_id,
            description=request.description,
            estimate=TargetEstimate(
                estimate_id="capture",
                position_m=(0.0, 0.0, 0.0),
                frame="libero_world",
                provider="capture",
                availability=FeatureAvailability.DEPLOYMENT_PERCEPTION,
                confidence=1.0,
                observation_sequence=observation_sequence,
            ),
        )


def _write_config(tmp_path, payload):
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_handoff_runtime_config(path)


def test_observe_whitelists_deployment_raw_fields(tmp_path):
    del tmp_path
    primitives = _FakePrimitives()
    provider = _CaptureProvider()
    sink = InMemoryResearchSink()
    adapter = LiberoGovernorAdapter(
        primitives=primitives,
        target_provider=provider,
        target_request=TargetRequest("target", "target"),
        core_api=CoreRuntimeAPI.load("rpent.research.handoff.governor"),
        instrumentation=RuntimeInstrumentation(sink),
        check_cancelled=lambda: None,
        vla_method="pi0_pick",
        invocation_id="invocation",
        tool_defaults={},
    )
    state = adapter.observe(SkillIdentity(name="pick"))
    assert state.target is not None
    assert provider.raw_keys is not None
    assert "robot0_eef_quat" in provider.raw_keys
    assert "secret_object_pos" not in provider.raw_keys
    assert "task_success_predicate" not in provider.raw_keys
    vector = FeatureBuilder(
        make_feature_spec("deployment_full", skill_vocabulary=["pick", "contact"])
    ).build(state)
    assert vector.state_id == state.state_id


def test_current_observation_provider_uses_in_memory_sam_and_rgbd():
    class _Sam:
        def __init__(self):
            self.image = None

        def segment_image(self, image, **kwargs):
            self.image = np.asarray(image)
            assert kwargs["text_prompt"] == "black bowl"
            mask = np.zeros((4, 4), dtype=bool)
            mask[1:3, 1:3] = True
            return SimpleNamespace(
                found=True,
                mask=mask,
                score=0.9,
                reason=None,
            )

    env = _FakeEnv()
    sam = _Sam()
    provider = CurrentObservationTargetProvider(env=env, sam3_client=sam)
    context = provider.estimate(
        request=TargetRequest("bowl", "black bowl"),
        observation_sequence=3,
        raw_observation=env.raw,
        policy_observation={},
    )
    assert sam.image is not None
    assert context.estimate.position_m is not None
    assert context.estimate.availability is FeatureAvailability.DEPLOYMENT_PERCEPTION
    assert context.estimate.visual_geometry.mask_area_fraction == pytest.approx(0.25)
    assert context.estimate.visual_geometry.valid_depth_fraction == pytest.approx(1.0)


def test_oracle_provider_requires_explicit_oracle_ablation():
    with pytest.raises(HandoffConfigurationError, match="oracle"):
        OracleTargetProvider(
            {"position_key": "secret_object_pos"},
            oracle_ablation=False,
        )


def test_doubled_outcome_does_not_relabel_task_success_as_primitive_success():
    primitives = _FakePrimitives()
    adapter = LiberoGovernorAdapter(
        primitives=primitives,
        target_provider=_CaptureProvider(),
        target_request=TargetRequest("target", "target"),
        core_api=CoreRuntimeAPI.load("rpent.research.handoff.governor"),
        instrumentation=RuntimeInstrumentation(InMemoryResearchSink()),
        check_cancelled=lambda: None,
        vla_method="pi0_doubled",
        invocation_id="invocation",
        tool_defaults={},
    )

    labels = adapter.label_outcome(
        SimpleNamespace(result={"success": False}, exception=None)
    )

    assert labels.primitive_success.value is None
    assert labels.primitive_success.source is LabelSource.UNAVAILABLE
    assert labels.task_success.value is False
    assert labels.task_success.source is LabelSource.OFFICIAL_TERMINATION


def test_pick_outcome_separates_heuristic_from_official_termination():
    primitives = _FakePrimitives()
    primitives.env.episode_terminated = True
    adapter = LiberoGovernorAdapter(
        primitives=primitives,
        target_provider=_CaptureProvider(),
        target_request=TargetRequest("target", "target"),
        core_api=CoreRuntimeAPI.load("rpent.research.handoff.governor"),
        instrumentation=RuntimeInstrumentation(InMemoryResearchSink()),
        check_cancelled=lambda: None,
        vla_method="pi0_pick",
        invocation_id="invocation",
        tool_defaults={},
    )
    labels = adapter.label_outcome(
        SimpleNamespace(
            exception=None,
            result={
                "success": True,
                "libero_terminated": True,
                "final_gripper_opening": 0.08,
                "diagnostics": {
                    "descent_done": False,
                    "post_min_ascent_m": 0.0,
                    "lift_thresh": 0.05,
                    "gripper_closed_thresh": 0.06,
                },
            },
        )
    )

    assert labels.primitive_success.value is False
    assert labels.primitive_success.source is LabelSource.PRIMITIVE_HEURISTIC
    assert labels.task_success.value is True


def test_disabled_toolkit_keeps_exact_original_tool_order(monkeypatch, tmp_path):
    fake = _FakePrimitives()

    def _fake_init(self, *, primitives_kwargs):
        assert primitives_kwargs == {"sentinel": object_value}
        self._primitives = fake

    object_value = object()
    monkeypatch.setattr(LiberoToolkit, "init_primitives_clean", _fake_init)
    monkeypatch.setattr("robots.libero.toolkit.get_output_dir", lambda: tmp_path)
    toolkit = LiberoToolkit(
        primitives_kwargs={"sentinel": object_value},
        dashboard_events=NullDashboardEventSink(),
        handoff_config=None,
    )
    names = [spec["name"] for spec in toolkit.get_tools_spec()]
    expected = [spec["name"] for spec in common.TOOLS_SPEC]
    expected += ["view_driver_state", "view_camera_meta", "back_project", "segment"]
    expected += list(libero_tools.PRIMITIVE_TOOL_NAMES)
    assert names == expected
    assert all(name not in names for name in HANDOFF_TOOL_NAMES)
    assert type(toolkit._primitives) is _FakePrimitives
    assert not (tmp_path / "handoff").exists()


def test_enabled_toolkit_keeps_planner_schema_and_routes_pi0_to_composite():
    fake = _FakePrimitives()
    toolkit = object.__new__(LiberoToolkit)
    Toolkit.__init__(toolkit, dashboard_events=NullDashboardEventSink())
    toolkit._primitives = fake
    toolkit._handoff = object()
    routed = []

    def route(name, **kwargs):
        routed.append((name, kwargs))
        return {"routed": name}

    toolkit._handoff_step = route
    toolkit._register_libero_tools()

    names = [spec["name"] for spec in toolkit.get_tools_spec()]
    expected = [spec["name"] for spec in common.TOOLS_SPEC]
    expected += ["view_driver_state", "view_camera_meta", "back_project", "segment"]
    expected += list(libero_tools.PRIMITIVE_TOOL_NAMES)
    assert names == expected
    assert all(name not in names for name in HANDOFF_TOOL_NAMES)

    result = toolkit.execute_tool("pi0_pick", {"prompt": "pick bowl"})
    assert result.result == {"routed": "handoff_pi0_pick"}
    assert routed == [("handoff_pi0_pick", {"prompt": "pick bowl"})]


def test_handoff_physical_step_preserves_result_in_standard_log_envelope(
    monkeypatch, tmp_path
):
    toolkit = object.__new__(LiberoToolkit)
    Toolkit.__init__(toolkit, dashboard_events=NullDashboardEventSink())
    toolkit._next_step = 0
    toolkit._video_path = None
    toolkit._primitives = SimpleNamespace(recorded_frame_count=lambda: 0)
    monkeypatch.setattr("robots.libero.toolkit.get_output_dir", lambda: tmp_path)
    monkeypatch.setattr("robots.libero.tools.get_output_dir", lambda: tmp_path)

    def _dump_state(_primitives, output_dir, step_idx, log):
        libero_tools._append_state(
            output_dir,
            {
                "step_idx": step_idx,
                "state": {},
                "libero_terminated": False,
                "episode_truncated": False,
                **log,
            },
        )

    monkeypatch.setattr(libero_tools, "dump_state", _dump_state)
    composite_result = {
        "handoff_occurred": True,
        "handoff_termination_reason": "handoff_completed",
        "controller_configuration_id": "controller-id",
    }

    returned = toolkit._run_physical_step(
        "handoff_pi0_pick",
        lambda **_kwargs: composite_result,
        {"prompt": "pick bowl"},
    )

    assert returned["log"]["result"] == composite_result
    assert "handoff_occurred" not in returned


def test_direct_composite_calls_unchanged_pick_executor(tmp_path):
    config = _write_config(
        tmp_path,
        {
            "target_provider": {
                "kind": "injected",
                "position_m": [0.1, 0.2, 0.3],
                "availability": "deployment_perception",
            },
            "core": {
                "policy": {"name": "direct_frozen_pi0"},
                "feature_spec": {
                    "preset": "deployment_full",
                    "skill_vocabulary": ["pick", "contact"],
                },
            },
        },
    )
    primitives = _FakePrimitives()
    sink = InMemoryResearchSink()
    instrumentation = RuntimeInstrumentation(sink)
    composite = build_handoff_composite(
        primitives=primitives,
        config=config,
        sink=sink,
        instrumentation=instrumentation,
        check_cancelled=lambda: None,
        run_output_dir=tmp_path,
    )
    result = composite.handoff_pi0_pick(
        "pick up the bowl",
        max_chunks=7,
        lift_thresh=0.04,
    )
    assert primitives.calls == [
        (
            "pi0_pick",
            {"prompt": "pick up the bowl", "max_chunks": 7, "lift_thresh": 0.04},
        )
    ]
    assert result["handoff_occurred"] is True
    assert result["handoff_configuration_id"] == config.configuration_id
    assert (
        result["controller_configuration_id"]
        == config.controller_configuration_id
    )
    assert len(sink.outcomes) == 1
    assert (
        sink.outcomes[0].controller.configuration_id
        == config.controller_configuration_id
    )
    with pytest.raises(ValueError, match="positive integer"):
        composite.handoff_pi0_pick("pick up the bowl", max_chunks=0)
    assert len(primitives.calls) == 1


def test_config_resolves_core_artifacts_from_config_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("RPENT_TEST_HANDOFF_MODEL", "artifacts/outcome-model")
    config = _write_config(
        tmp_path,
        {
            "core": {
                "model_artifact": "${RPENT_TEST_HANDOFF_MODEL}",
                "policy": {
                    "name": "positive_nearest_success",
                    "positive_references_file": "references/train.json",
                },
                "fallback_policy": {
                    "name": "positive_support_region",
                    "positive_references_file": "references/fallback.json",
                },
                "feature_spec": {
                    "preset": "deployment_full",
                    "skill_vocabulary": ["pick", "contact"],
                },
            }
        },
    )
    assert config.governor_config["model_artifact"] == str(
        (tmp_path / "artifacts" / "outcome-model").resolve()
    )
    assert config.governor_config["policy"]["positive_references_file"] == str(
        (tmp_path / "references" / "train.json").resolve()
    )
    assert config.governor_config["fallback_policy"][
        "positive_references_file"
    ] == str((tmp_path / "references" / "fallback.json").resolve())
    assert config.canonical_config["core"] == config.governor_config


def test_enabled_sink_writes_canonical_manifest(tmp_path):
    config = _write_config(tmp_path, {})
    assert config.controller_method == "direct_frozen_pi0"
    with pytest.raises(TypeError, match="immutable"):
        config.governor_config["policy"]["name"] = "mutated"
    sink = build_research_sink(config, run_output_dir=tmp_path / "run")
    manifest_path = sink.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["configuration_id"] == config.configuration_id
    assert (
        manifest["controller_configuration_id"]
        == config.controller_configuration_id
    )
    assert manifest["configuration"] == config.canonical_config


def test_runtime_event_sink_recovers_only_torn_final_jsonl_record(tmp_path):
    output = tmp_path / "handoff-events"
    output.mkdir()
    event_path = output / "runtime_events.jsonl"
    event_path.write_bytes(b'{"prior":true}\n{"torn"')

    sink = RuntimeEventSink(
        output,
        configuration_id="full-id",
        controller_configuration_id="controller-id",
    )
    sink.append_event("resumed", {"ok": True})
    records = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert records[0] == {"prior": True}
    assert records[1]["event_type"] == "resumed"

    corrupt = tmp_path / "corrupt-events"
    corrupt.mkdir()
    (corrupt / "outcomes.jsonl").write_text("not-json\n", encoding="utf-8")
    with pytest.raises(HandoffConfigurationError, match="invalid existing"):
        RuntimeEventSink(corrupt, configuration_id="full-id")


def test_controller_configuration_id_excludes_run_only_configuration(tmp_path):
    base = {
        "target_provider": {
            "kind": "injected",
            "position_m": [0.1, 0.2, 0.3],
        },
        "tool_defaults": {"handoff_pi0_pick": {"max_chunks": 3}},
    }
    first = _write_config(
        tmp_path,
        {
            **base,
            "output_subdir": "handoff-first",
            "instrumentation": True,
            "sink": {"run_label": "first"},
            "metadata": {"run_id": "run-first", "seed": 1},
        },
    )
    second = _write_config(
        tmp_path,
        {
            **base,
            "output_subdir": "handoff-second",
            "instrumentation": False,
            "sink": {"run_label": "second"},
            "metadata": {"run_id": "run-second", "seed": 2},
        },
    )
    changed_controller = _write_config(
        tmp_path,
        {
            **base,
            "tool_defaults": {"handoff_pi0_pick": {"max_chunks": 4}},
        },
    )

    assert first.configuration_id != second.configuration_id
    assert (
        first.controller_configuration_id
        == second.controller_configuration_id
    )
    assert (
        first.controller_configuration_id
        != changed_controller.controller_configuration_id
    )


def test_config_rejects_unresolved_artifact_environment_variable(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("RPENT_MISSING_TEST_PATH", raising=False)
    path = tmp_path / "handoff.json"
    path.write_text(
        json.dumps({"core": {"model_artifact": "${RPENT_MISSING_TEST_PATH}"}}),
        encoding="utf-8",
    )
    with pytest.raises(HandoffConfigurationError, match="unresolved"):
        load_handoff_runtime_config(path)


def test_config_rejects_duplicate_keys_and_canonicalizes_output_path(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"enabled":true,"core":{"policy":{"name":"direct_frozen_pi0"},'
        '"policy":{"name":"fixed_distance"}}}',
        encoding="utf-8",
    )
    with pytest.raises(HandoffConfigurationError, match="duplicate JSON key"):
        load_handoff_runtime_config(path)

    config = _write_config(tmp_path, {"output_subdir": "nested\\handoff"})
    assert config.output_subdir == "nested/handoff"
    for invalid in ("/handoff", "env_server.log", "handoff\u0000escape"):
        with pytest.raises(HandoffConfigurationError, match="output_subdir"):
            _write_config(tmp_path, {"output_subdir": invalid})


def test_stage_clamps_candidate_to_one_euclidean_step():
    primitives = _FakePrimitives()
    sink = InMemoryResearchSink()
    instrumentation = RuntimeInstrumentation(sink)
    captured = {}

    def _move_to(target, **kwargs):
        captured["target"] = tuple(target)
        captured["kwargs"] = kwargs
        primitives._last_obs_eef_pos = np.asarray(target, dtype=np.float64)
        instrumentation.increment("env_steps")
        return {"steps_used": 1}

    primitives.move_to = _move_to
    adapter = LiberoGovernorAdapter(
        primitives=primitives,
        target_provider=_CaptureProvider(),
        target_request=TargetRequest("target", "target"),
        core_api=CoreRuntimeAPI.load("rpent.research.handoff.governor"),
        instrumentation=instrumentation,
        check_cancelled=lambda: None,
        vla_method="pi0_pick",
        invocation_id="invocation",
        tool_defaults={},
    )
    start = np.asarray(primitives._last_obs_eef_pos, dtype=np.float64)
    result = adapter.stage(
        SimpleNamespace(candidate_id="future", eef_position_m=(1.0, 1.0, 1.0)),
        0.03,
    )
    commanded = np.asarray(captured["target"], dtype=np.float64)
    assert np.linalg.norm(commanded - start) == pytest.approx(0.03)
    assert result.steps == 1
    assert result.distance_m == pytest.approx(0.03)


def test_failed_env_rpc_is_submitted_but_not_counted_as_executed():
    class _FailingClient:
        episode_terminated = False
        episode_truncated = False
        return_all_frames = False

        def step(self, _action):
            raise RuntimeError("RPC failed")

    instrumentation = RuntimeInstrumentation(InMemoryResearchSink())
    client = InstrumentedEnvClient(_FailingClient(), instrumentation)
    with pytest.raises(RuntimeError, match="RPC failed"):
        client.step(np.zeros(7, dtype=np.float32))

    counters = instrumentation.snapshot()
    assert counters["env_actions_submitted"] == 1
    assert counters["env_steps"] == 0
    assert counters["env_actions"] == 0
