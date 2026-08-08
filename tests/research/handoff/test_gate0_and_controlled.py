from __future__ import annotations

import json
import math
from dataclasses import dataclass

import pytest

from rpent.research.handoff.candidates import (
    CandidateGeneratorConfig,
    ObjectRelativeCandidateGenerator,
    wrist_yaw_pitch,
)
from rpent.research.handoff.experiments.config import ExperimentConfig
from rpent.research.handoff.experiments.controlled import (
    ControlledReset,
    ControlledRunner,
)
from rpent.research.handoff.experiments.gate0 import (
    Gate0Collector,
    Gate0Config,
    Gate0RunIdentity,
    Gate0Setup,
)
from rpent.research.handoff.experiments.manifest import expand_manifest
from rpent.research.handoff.experiments.sampling import (
    Gate0SamplerConfig,
    SampleRange,
    SamplingMode,
    generate_gate0_samples,
)
from rpent.research.handoff.governor import (
    EpisodeStatus,
    GovernorConfig,
    HandoffGovernor,
    InMemoryResearchSink,
    StageResult,
    VLAExecutionResult,
)
from rpent.research.handoff.policies import DirectHandoffPolicy
from rpent.research.handoff.privileged import ExperimentSetupRecord, SetupValue
from rpent.research.handoff.types import (
    ControllerIdentity,
    FeatureAvailability,
    FeatureProvenance,
    HandoffState,
    LabelSource,
    OutcomeLabels,
    OutcomeSignal,
    SkillIdentity,
    TargetContext,
    TargetEstimate,
)


def _quat_for_yaw_pitch(
    yaw: float,
    pitch: float,
) -> tuple[float, float, float, float]:
    yaw_half = yaw / 2.0
    down_pitch_half = (math.pi + pitch) / 2.0
    return (
        math.cos(yaw_half) * math.sin(down_pitch_half),
        math.sin(yaw_half) * math.sin(down_pitch_half),
        math.sin(yaw_half) * math.cos(down_pitch_half),
        math.cos(yaw_half) * math.cos(down_pitch_half),
    )


def _state(
    z: float,
    sequence: int = 0,
    *,
    yaw: float = 0.0,
    pitch: float = 0.0,
) -> HandoffState:
    skill = SkillIdentity(name="pick", semantic_target="mug")
    return HandoffState(
        state_id=f"state-{sequence}-{z}",
        observation_sequence=sequence,
        observed_elapsed_s=float(sequence),
        eef_position_m=(0.0, 0.0, z),
        eef_quaternion_xyzw=_quat_for_yaw_pitch(yaw, pitch),
        gripper_opening_m=0.08,
        skill=skill,
        target=TargetContext(
            target_id="mug",
            description="mug",
            estimate=TargetEstimate(
                estimate_id=f"target-{sequence}",
                position_m=(0.0, 0.0, 0.0),
                frame="world",
                provider="fake-perception",
                availability=FeatureAvailability.DEPLOYMENT_PERCEPTION,
                observation_sequence=sequence,
            ),
        ),
        provenance=(
            FeatureProvenance(feature_name="eef_position_m", availability=FeatureAvailability.DEPLOYMENT_SENSOR, source="fake", unit="m", frame="world"),
            FeatureProvenance(feature_name="eef_quaternion_xyzw", availability=FeatureAvailability.DEPLOYMENT_SENSOR, source="fake", unit="unit", frame="world"),
            FeatureProvenance(feature_name="gripper_opening_m", availability=FeatureAvailability.DEPLOYMENT_SENSOR, source="fake", unit="m", frame="gripper"),
            FeatureProvenance(feature_name="skill", availability=FeatureAvailability.DERIVED_DEPLOYMENT, source="fake", unit="categorical", frame="semantic"),
            FeatureProvenance(feature_name="target_position_m", availability=FeatureAvailability.DEPLOYMENT_PERCEPTION, source="fake", unit="m", frame="world"),
        ),
    )


@dataclass
class _Gate0Adapter:
    z: float = 0.3
    yaw: float = 0.0
    pitch: float = 0.0
    fail_stage: bool = False
    vla_calls: int = 0
    reset_calls: int = 0

    def reset_for_trial(self, identity, skill, sample):
        del skill
        self.reset_calls += 1
        self.z = 0.3
        self.yaw = 0.0
        self.pitch = 0.0
        resolved = identity.model_copy(update={"reset_id": f"reset-{sample.sample_id}"})
        record = ExperimentSetupRecord(
            record_id=f"setup-{sample.sample_id}",
            identity=resolved,
            setup_provider="fake-privileged/v1",
            values=(SetupValue(name="target_position_m", values=(0.0, 0.0, 0.0), unit="m", frame="world", source="fake simulator"),),
        )
        return Gate0Setup(record=record, target_position_m=(0.0, 0.0, 0.0), reset_id=resolved.reset_id)

    def current_eef_position_m(self):
        return (0.0, 0.0, self.z)

    def current_eef_quaternion_xyzw(self):
        return _quat_for_yaw_pitch(self.yaw, self.pitch)

    def observe(self, skill):
        del skill
        return _state(self.z, yaw=self.yaw, pitch=self.pitch)

    def episode_status(self):
        return EpisodeStatus()

    def raise_if_cancelled(self):
        return None

    def stage(self, candidate, max_step_m):
        if self.fail_stage:
            return StageResult(success=False, steps=0, distance_m=0.0, elapsed_s=0.01, error="unreachable")
        before = self.z
        requested = candidate.eef_position_m[2]
        delta = max(-max_step_m, min(max_step_m, requested - before))
        self.z += delta
        self.yaw = candidate.wrist_yaw_rad
        self.pitch = candidate.wrist_pitch_rad
        return StageResult(success=True, steps=1, distance_m=abs(delta), elapsed_s=0.01)

    def execute_vla(self, invocation):
        del invocation
        self.vla_calls += 1
        return VLAExecutionResult(result={"success": True}, elapsed_s=0.1, chunks=1, env_actions=5)

    def label_outcome(self, result):
        del result
        return OutcomeLabels(primitive_success=OutcomeSignal(value=True, source=LabelSource.PRIMITIVE_HEURISTIC, definition="fake"))


def _gate0_config() -> Gate0Config:
    def constant(value: float) -> SampleRange:
        return SampleRange(minimum=value, maximum=value, grid_count=1)

    return Gate0Config(
        sampler=Gate0SamplerConfig(
            mode=SamplingMode.GRID,
            relative_x_m=constant(0.0),
            relative_y_m=constant(0.0),
            relative_z_m=constant(0.08),
            standoff_m=constant(0.0),
            wrist_yaw_rad=constant(0.0),
            wrist_pitch_rad=constant(0.0),
            repeats=1,
        ),
        staging_step_m=0.1,
        max_staging_steps=5,
        max_staging_distance_m=0.5,
    )


def _collector(adapter, sink, completed=(), *, config=None, run_identity=None):
    return Gate0Collector(
        adapter=adapter,
        config=config or _gate0_config(),
        skill=SkillIdentity(name="pick", semantic_target="mug"),
        controller=ControllerIdentity(method="gate0", implementation_version="v1", configuration_id="cfg"),
        run_identity=run_identity
        or Gate0RunIdentity(run_id="run", suite="suite", task_id=0, seed=0),
        outcome_sink=sink,
        completed_trial_ids=completed,
        vla_kwargs={"prompt": "pick mug", "max_chunks": 5},
    )


def test_gate0_persists_actually_achieved_pre_handoff_state_and_resumes() -> None:
    sink = InMemoryResearchSink()
    adapter = _Gate0Adapter()
    outcome = _collector(adapter, sink).collect()[0]
    assert outcome.pre_handoff_state is not None
    assert abs(outcome.pre_handoff_state.eef_position_m[2] - 0.08) < 1e-9
    assert outcome.identity.reset_id is not None
    assert outcome.costs.analytic_steps > 0
    assert adapter.vla_calls == 1
    resumed = _collector(adapter, sink, completed=(outcome.identity.trial_id,)).collect()
    assert resumed == ()


def test_gate0_outcome_binds_plan_runtime_identity_and_matched_cohort() -> None:
    outcome = _collector(
        _Gate0Adapter(),
        InMemoryResearchSink(),
        run_identity=Gate0RunIdentity(
            run_id="run",
            suite="suite",
            task_id=0,
            seed=0,
            source_revision="git:source",
            configuration_id="gate0-config",
            execution_plan_id="gate0-plan",
            runtime_attestation_id="runtime-attestation",
            runtime_attestation_sha256="a" * 64,
        ),
    ).collect()[0]

    assert outcome.metadata["gate0_configuration_id"] == "gate0-config"
    assert outcome.metadata["execution_plan_id"] == "gate0-plan"
    assert outcome.metadata["runtime_attestation_id"] == "runtime-attestation"
    assert outcome.metadata["runtime_attestation_sha256"] == "a" * 64
    assert outcome.metadata["gate0_candidate_id"] == outcome.identity.candidate_id
    assert outcome.metadata["gate0_matched_cohort_id"].startswith(
        "gate0-cohort-"
    )


def test_gate0_staging_failure_never_invokes_vla() -> None:
    sink = InMemoryResearchSink()
    adapter = _Gate0Adapter(fail_stage=True)
    outcome = _collector(adapter, sink).collect()[0]
    assert not outcome.handoff_occurred
    assert outcome.termination.failure_mode.value == "staging"
    assert adapter.vla_calls == 0


def test_gate0_verifies_sampled_orientation_and_records_achieved_quaternion() -> None:
    base = _gate0_config()
    sampler = base.sampler.model_copy(
        update={
            "wrist_yaw_rad": SampleRange(
                minimum=0.3,
                maximum=0.3,
                grid_count=1,
            ),
            "wrist_pitch_rad": SampleRange(
                minimum=-0.2,
                maximum=-0.2,
                grid_count=1,
            ),
        }
    )
    config = base.model_copy(update={"sampler": sampler})
    outcome = _collector(
        _Gate0Adapter(),
        InMemoryResearchSink(),
        config=config,
    ).collect()[0]

    assert outcome.pre_handoff_state is not None
    yaw, pitch = wrist_yaw_pitch(
        outcome.pre_handoff_state.eef_quaternion_xyzw
    )
    assert yaw == pytest.approx(0.3)
    assert pitch == pytest.approx(-0.2)


class _OrientationStuckAdapter(_Gate0Adapter):
    def stage(self, candidate, max_step_m):
        result = super().stage(candidate, max_step_m)
        self.yaw = 0.0
        self.pitch = 0.0
        return result


def test_gate0_orientation_not_reached_is_staging_failure() -> None:
    base = _gate0_config()
    sampler = base.sampler.model_copy(
        update={
            "wrist_yaw_rad": SampleRange(
                minimum=0.4,
                maximum=0.4,
                grid_count=1,
            )
        }
    )
    config = base.model_copy(
        update={"sampler": sampler, "max_staging_steps": 2}
    )
    adapter = _OrientationStuckAdapter()
    outcome = _collector(
        adapter,
        InMemoryResearchSink(),
        config=config,
    ).collect()[0]

    assert outcome.termination.reason.value == "staging_failure"
    assert adapter.vla_calls == 0


class _ControlledAdapter(_Gate0Adapter):
    def reset_for_trial(self, trial):
        self.reset_calls += 1
        self.z = 0.3
        return ControlledReset(
            reset_id=trial.task.reset_id,
            episode_id=f"controlled-{trial.trial_id}",
        )


def test_gate0_candidate_identity_is_stable_across_execution_repeats() -> None:
    sampler = _gate0_config().sampler.model_copy(update={"repeats": 2})
    first, second = generate_gate0_samples(sampler)

    assert first.candidate_id == second.candidate_id
    assert first.sample_id != second.sample_id
    assert (first.repeat_index, second.repeat_index) == (0, 1)


def _stochastic_sampler(mode: str, *, seed: int = 11) -> Gate0SamplerConfig:
    return Gate0SamplerConfig(
        mode=mode,
        relative_x_m=SampleRange(minimum=-0.2, maximum=0.2),
        relative_y_m=SampleRange(minimum=-0.1, maximum=0.1),
        relative_z_m=SampleRange(minimum=0.02, maximum=0.12),
        standoff_m=SampleRange(minimum=0.0, maximum=0.05),
        wrist_yaw_rad=SampleRange(minimum=-0.4, maximum=0.4),
        wrist_pitch_rad=SampleRange(minimum=-0.2, maximum=0.2),
        random_samples=8,
        seed=seed,
    )


def test_random_sampler_is_seeded_bounded_and_identity_stable() -> None:
    config = _stochastic_sampler("random")
    first = generate_gate0_samples(config)
    second = generate_gate0_samples(config)
    different = generate_gate0_samples(
        _stochastic_sampler("random", seed=config.seed + 1)
    )

    assert first == second
    assert tuple(item.candidate_id for item in first) != tuple(
        item.candidate_id for item in different
    )
    assert len({item.candidate_id for item in first}) == config.random_samples
    for item in first:
        assert config.relative_x_m.minimum <= item.relative_xyz_m[0] <= config.relative_x_m.maximum
        assert config.relative_y_m.minimum <= item.relative_xyz_m[1] <= config.relative_y_m.maximum
        assert config.relative_z_m.minimum <= item.relative_xyz_m[2] <= config.relative_z_m.maximum


def test_latin_hypercube_sampler_uses_every_stratum_per_dimension() -> None:
    config = _stochastic_sampler("latin_hypercube")
    samples = generate_gate0_samples(config)
    dimensions = (
        (config.relative_x_m, [sample.relative_xyz_m[0] for sample in samples]),
        (config.relative_y_m, [sample.relative_xyz_m[1] for sample in samples]),
        (config.relative_z_m, [sample.relative_xyz_m[2] for sample in samples]),
        (config.standoff_m, [sample.standoff_m for sample in samples]),
        (config.wrist_yaw_rad, [sample.wrist_yaw_rad for sample in samples]),
        (config.wrist_pitch_rad, [sample.wrist_pitch_rad for sample in samples]),
    )
    for sample_range, values in dimensions:
        width = sample_range.maximum - sample_range.minimum
        strata = {
            min(
                config.random_samples - 1,
                int((value - sample_range.minimum) / width * config.random_samples),
            )
            for value in values
        }
        assert strata == set(range(config.random_samples))


def test_stochastic_sampler_enforces_explicit_trial_cap() -> None:
    config = _stochastic_sampler("random").model_copy(
        update={"random_samples": 4, "repeats": 3, "maximum_total_trials": 11}
    )
    with pytest.raises(ValueError, match="maximum_total_trials"):
        generate_gate0_samples(config)


def test_controlled_runner_has_no_planner_and_varies_policy_via_manifest(tmp_path) -> None:
    handoff_path = tmp_path / "direct.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": "rpent.libero-handoff-runtime/v1",
                "enabled": True,
                "controller_method": "direct_frozen_pi0",
                "core": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = ExperimentConfig.model_validate(
        {
            "experiment_id": "controlled-test",
            "output_root": str(tmp_path / "outputs"),
            "tasks": [{"suite": "suite", "task": 0, "seeds": [0], "target_id": "mug", "target_description": "mug", "skill_name": "pick", "skill_prompt": "pick mug", "label_source": "primitive_heuristic", "reset_id_template": "reset-{seed}-{repeat}"}],
            "conditions": [{"name": "direct", "execution_layer": "controlled", "method": "direct_frozen_pi0", "handoff_enabled": True, "handoff_config": str(handoff_path), "decision": "direct"}],
            "runtime": {
                "pi05_checkpoint_id": "sha256:pi05-test",
                "sam3_checkpoint_id": "sha256:sam3-test"
            },
            "source_revision": "git:test-source",
        }
    )
    trial = expand_manifest(config).trials[0]
    sink = InMemoryResearchSink()
    adapter = _ControlledAdapter()
    runner = ControlledRunner(
        governor_factory=lambda _: HandoffGovernor(
            policy=DirectHandoffPolicy(),
            candidate_generator=ObjectRelativeCandidateGenerator(CandidateGeneratorConfig(standoff_distances_m=(0.08,), max_candidates=1)),
            config=GovernorConfig(max_analytic_steps=0, max_staging_distance_m=0.0),
            sink=sink,
        ),
        adapter_factory=lambda _: adapter,
        run_id="controlled-run",
    )
    result = runner.run((trial,))[0]
    assert result.outcome.handoff_occurred
    assert adapter.reset_calls == 1
    assert adapter.vla_calls == 1
