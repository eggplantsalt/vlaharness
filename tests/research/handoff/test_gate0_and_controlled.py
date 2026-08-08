from __future__ import annotations

from dataclasses import dataclass

from rpent.research.handoff.candidates import (
    CandidateGeneratorConfig,
    ObjectRelativeCandidateGenerator,
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


def _state(z: float, sequence: int = 0) -> HandoffState:
    skill = SkillIdentity(name="pick", semantic_target="mug")
    return HandoffState(
        state_id=f"state-{sequence}-{z}",
        observation_sequence=sequence,
        observed_elapsed_s=float(sequence),
        eef_position_m=(0.0, 0.0, z),
        eef_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
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
    fail_stage: bool = False
    vla_calls: int = 0
    reset_calls: int = 0

    def reset_for_trial(self, identity, skill, sample):
        del skill
        self.reset_calls += 1
        self.z = 0.3
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

    def observe(self, skill):
        del skill
        return _state(self.z)

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


def _collector(adapter, sink, completed=()):
    return Gate0Collector(
        adapter=adapter,
        config=_gate0_config(),
        skill=SkillIdentity(name="pick", semantic_target="mug"),
        controller=ControllerIdentity(method="gate0", implementation_version="v1", configuration_id="cfg"),
        run_identity=Gate0RunIdentity(run_id="run", suite="suite", task_id=0, seed=0),
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


def test_gate0_staging_failure_never_invokes_vla() -> None:
    sink = InMemoryResearchSink()
    adapter = _Gate0Adapter(fail_stage=True)
    outcome = _collector(adapter, sink).collect()[0]
    assert not outcome.handoff_occurred
    assert outcome.termination.failure_mode.value == "staging"
    assert adapter.vla_calls == 0


class _ControlledAdapter(_Gate0Adapter):
    def reset_for_trial(self, trial):
        self.reset_calls += 1
        self.z = 0.3
        return ControlledReset(
            reset_id=trial.task.reset_id,
            episode_id=f"controlled-{trial.trial_id}",
        )


def test_controlled_runner_has_no_planner_and_varies_policy_via_manifest() -> None:
    config = ExperimentConfig.model_validate(
        {
            "experiment_id": "controlled-test",
            "output_root": "outputs",
            "tasks": [{"suite": "suite", "task": 0, "seeds": [0], "target_id": "mug", "target_description": "mug", "skill_name": "pick", "skill_prompt": "pick mug", "label_source": "primitive_heuristic", "reset_id_template": "reset-{seed}-{repeat}"}],
            "conditions": [{"name": "direct", "execution_layer": "controlled", "method": "direct_frozen_pi0", "handoff_enabled": True, "handoff_config": "direct.json", "decision": "direct"}],
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
