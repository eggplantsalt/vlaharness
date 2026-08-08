from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rpent.research.handoff.artifacts import (
    ModelArtifactManifest,
    SourceIdentity,
    _artifact_id as model_artifact_id,
)
from rpent.research.handoff.baseline_data import (
    PositiveReferenceArtifact,
    PositiveReferenceBuildSettings,
    _artifact_id as positive_artifact_id,
    write_positive_reference_artifact,
)
from rpent.research.handoff.experiments.config import (
    ConditionSpec,
    DecisionMode,
    ExecutionLayer,
    ExperimentConfig,
    HierarchyMode,
    PlannerConfig,
    RuntimeConfig,
    TaskSpec,
    load_experiment_config,
)
from rpent.research.handoff.features import FeaturePreset, make_feature_spec
from rpent.research.handoff.policies import PositiveReference
from rpent.research.handoff.experiments.full_agent import (
    build_child_plan,
    build_full_agent_command,
    execute_child_plan,
    load_child_plans,
    write_child_plans,
)
from rpent.research.handoff.experiments.lifecycle import (
    LifecycleJournal,
    TrialEventType,
    derive_resume_states,
    read_lifecycle_events,
)
from rpent.research.handoff.experiments.manifest import (
    expand_manifest,
    write_manifest,
)
from rpent.research.handoff.experiments.preflight import run_offline_preflight
from rpent.research.handoff.experiments.runtime import (
    Gate0JobSpec,
    gate0_resume_anchor,
    write_resolved_handoff_config,
)
from rpent.research.handoff.types import LabelSource


def _config(tmp_path) -> ExperimentConfig:
    handoff_config = tmp_path / "handoff.json"
    handoff_config.write_text(
        json.dumps(
            {
                "schema_version": "rpent.libero-handoff-runtime/v1",
                "enabled": True,
                "controller_method": "outcome_calibrated_switching",
                "core": {},
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return ExperimentConfig(
        experiment_id="surface-test",
        output_root=str(tmp_path / "outputs"),
        repeats=1,
        tasks=(
            TaskSpec(
                suite="libero_object",
                task=2,
                seeds=(0, 3),
                target_id="target-cup",
                target_description="the target cup",
                skill_name="pick",
                skill_prompt="pick up the target cup",
                label_source=LabelSource.SKILL_EVALUATOR,
            ),
        ),
        conditions=(
            ConditionSpec(
                name="original",
                execution_layer=ExecutionLayer.FULL_AGENT,
                method="original_harness",
                decision=DecisionMode.DIRECT,
                hierarchy=HierarchyMode.PLANNER_MEDIATED,
            ),
            ConditionSpec(
                name="ours",
                execution_layer=ExecutionLayer.FULL_AGENT,
                method="outcome_calibrated_switching",
                handoff_enabled=True,
                handoff_config=str(handoff_config),
            ),
        ),
        runtime=RuntimeConfig(
            vla_endpoint="http://vla.example:8011",
            sam3_endpoint="http://sam.example:8012",
            pi05_checkpoint_id="sha256:pi05-test",
            sam3_checkpoint_id="sha256:sam3-test",
        ),
        planner=PlannerConfig(
            backend="api",
            model="planner-test-model",
            base_url="http://planner.example:8000/v1",
        ),
        source_revision="git:test-source",
    )


def _write_fake_model_artifact(tmp_path):
    root = tmp_path / "models" / "condition"
    root.mkdir(parents=True)
    estimator = root / "estimator.joblib"
    estimator.write_bytes(b"static manifest-expansion fixture")
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
        calibration_record_ids=("cal-1",),
        held_out_record_ids=("test-1",),
        training_configuration={"fixture": True},
        source_identity=SourceIdentity(
            git_revision="test-source",
            source_revision="git:test-source;worktree-sha256:fixture",
        ),
    )
    manifest = provisional.model_copy(
        update={"artifact_id": model_artifact_id(provisional)}
    )
    (root / "manifest.json").write_text(
        manifest.canonical_json() + "\n",
        encoding="utf-8",
    )
    return root, manifest


def _write_fake_positive_artifact(path):
    provisional = PositiveReferenceArtifact(
        artifact_id="pending",
        dataset_fingerprint="train-only-dataset",
        source_dataset_fingerprint="full-source-dataset",
        split_assignment_fingerprint="split-fixture",
        source_partition="train",
        target_label="primitive_success",
        deployment_provenance_verified=True,
        build_settings=PositiveReferenceBuildSettings(),
        source_record_ids=("record-1",),
        references=(
            PositiveReference(
                reference_id="positive-record-1",
                target_relative_position_m=(0.0, 0.0, 0.08),
                wrist_yaw_rad=0.0,
                wrist_pitch_rad=0.0,
            ),
        ),
    )
    artifact = provisional.model_copy(
        update={"artifact_id": positive_artifact_id(provisional)}
    )
    write_positive_reference_artifact(artifact, path)
    return artifact


def test_config_is_strict_and_configuration_id_ignores_json_key_order(tmp_path) -> None:
    config = _config(tmp_path)
    payload = config.model_dump(mode="json", exclude_none=False)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(payload), encoding="utf-8")
    second.write_text(
        json.dumps(dict(reversed(list(payload.items())))),
        encoding="utf-8",
    )

    assert load_experiment_config(first).configuration_id == config.configuration_id
    assert load_experiment_config(second).configuration_id == config.configuration_id

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({**payload, "unexpected": True})


def test_gate0_job_rejects_zero_chunk_vla_execution(tmp_path) -> None:
    payload = {
        "output_dir": str(tmp_path / "gate0"),
        "adapter_factory": "package.module:factory",
        "gate0": {},
        "vla_kwargs": {"max_chunks": 0},
        "run_id": "run",
        "suite": "libero_object",
        "task_id": 0,
        "seed": 0,
        "target_id": "cup",
        "target_description": "the cup",
        "skill_name": "pick",
        "skill_prompt": "pick the cup",
        "controller_method": "direct_frozen_pi0",
        "checkpoint_id": "sha256:pi05-test",
        "source_revision": "git:test-source",
    }
    with pytest.raises(ValidationError, match="positive integer"):
        Gate0JobSpec.model_validate(payload)


def test_gate0_job_rejects_checkpoint_identity_disagreement(tmp_path) -> None:
    payload = {
        "output_dir": str(tmp_path / "gate0"),
        "adapter_factory": "package.module:factory",
        "adapter_config": {
            "runtime": {
                "pi05_checkpoint_path": str(tmp_path / "pi05"),
                "sam3_checkpoint_path": str(tmp_path / "sam3.pt"),
                "pi05_checkpoint_id": "sha256:runtime-pi05",
                "sam3_checkpoint_id": "sha256:runtime-sam3",
            },
            "handoff_config": str(tmp_path / "handoff.json"),
        },
        "gate0": {},
        "run_id": "run",
        "suite": "libero_object",
        "task_id": 0,
        "seed": 0,
        "target_id": "cup",
        "target_description": "the cup",
        "skill_name": "pick",
        "skill_prompt": "pick the cup",
        "controller_method": "direct_frozen_pi0",
        "checkpoint_id": "sha256:different-pi05",
        "source_revision": "git:test-source",
    }

    with pytest.raises(ValidationError, match="runtime Pi0.5 checkpoint ID"):
        Gate0JobSpec.model_validate(payload)


def test_gate0_resume_anchor_detects_orphan_setup_and_attempt(tmp_path) -> None:
    output = tmp_path / "gate0"
    setup = output / "privileged" / "setups.jsonl"
    attempt = output / "attempts" / "plan-a.json"
    setup.parent.mkdir(parents=True)
    attempt.parent.mkdir(parents=True)
    setup.write_bytes(b"orphan-setup\n")
    attempt.write_bytes(b'{"plan_id":"plan-a"}\n')

    anchor = gate0_resume_anchor(output)

    assert set(anchor) == {
        "attempts/plan-a.json",
        "privileged/setups.jsonl",
    }
    assert all(len(value) == 64 for value in anchor.values())


def test_manifest_ids_are_deterministic_and_output_location_independent(tmp_path) -> None:
    config = _config(tmp_path)
    first = expand_manifest(config)
    second = expand_manifest(config)
    moved = expand_manifest(
        config.model_copy(update={"output_root": str(tmp_path / "elsewhere")})
    )

    assert first == second
    assert len(first.trials) == 4
    assert [trial.trial_id for trial in first.trials] == [
        trial.trial_id for trial in moved.trials
    ]
    assert len({trial.output_dir for trial in first.trials}) == 4


def test_lifecycle_resume_skips_complete_and_retries_interrupted(tmp_path) -> None:
    manifest = expand_manifest(_config(tmp_path))
    journal = LifecycleJournal(
        tmp_path / "status.jsonl",
        allowed_trial_ids={trial.trial_id for trial in manifest.trials},
    )
    complete = manifest.trials[0]
    interrupted = manifest.trials[1]
    journal.append(complete.trial_id, TrialEventType.PLANNED)
    journal.append(complete.trial_id, TrialEventType.STARTED)
    journal.append(complete.trial_id, TrialEventType.COMPLETED)
    journal.append(interrupted.trial_id, TrialEventType.STARTED)

    states = {state.trial_id: state for state in derive_resume_states(manifest, journal.read())}

    assert states[complete.trial_id].should_run is False
    assert states[interrupted.trial_id].should_run is True
    assert states[interrupted.trial_id].next_attempt == 2
    assert states[manifest.trials[2].trial_id].reason == "not_started"


def test_lifecycle_event_identity_rejects_payload_mutation(tmp_path) -> None:
    trial_id = "trial-fixture"
    path = tmp_path / "status.jsonl"
    journal = LifecycleJournal(path, allowed_trial_ids={trial_id})
    journal.append(
        trial_id,
        TrialEventType.STARTED,
        timestamp_utc="2026-08-09T00:00:00Z",
        artifact_path=str(tmp_path / "output"),
        details={"plan_id": "plan-fixture"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["details"]["plan_id"] = "mutated-plan"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="event_id does not bind"):
        read_lifecycle_events(path, missing_ok=False)


def test_preflight_and_full_agent_baseline_isolation(tmp_path) -> None:
    config = _config(tmp_path)
    manifest = expand_manifest(config)
    manifest_path = write_manifest(manifest, tmp_path / "manifest.json")
    report = run_offline_preflight(config, manifest)
    assert report.ok

    baseline = next(
        trial for trial in manifest.trials if trial.condition.method == "original_harness"
    )
    ours = next(
        trial
        for trial in manifest.trials
        if trial.condition.method == "outcome_calibrated_switching"
    )
    baseline_command = build_full_agent_command(
        baseline, python_executable="python"
    )
    ours_command = build_full_agent_command(ours, python_executable="python")

    assert "--handoff-config" not in baseline_command
    assert ours_command[-2:] == ("--handoff-config", ours.handoff_config_path)
    for command in (baseline_command, ours_command):
        assert "--research-reset-identity-output" in command
        assert "--research-completion-output" in command
        assert "--research-runtime-identity-output" in command
        assert command[command.index("--model") + 1] == "planner-test-model"
    assert baseline_command[baseline_command.index("--research-trial-id") + 1] == baseline.trial_id

    plan = build_child_plan(
        baseline,
        manifest_path=manifest_path,
        repo_root=tmp_path,
        python_executable="python",
    )
    assert plan.schema_version == "rpent.handoff-full-agent-plan/v2"
    assert plan.manifest_id == manifest.manifest_id
    assert plan.manifest_path == str(manifest_path.resolve())
    assert plan.command == plan.wrapper_command
    assert plan.wrapper_command != plan.resolved_inner_command
    assert plan.wrapper_command[plan.wrapper_command.index("--plan-id") + 1] == (
        plan.plan_id
    )
    assert plan.resolved_inner_command[
        plan.resolved_inner_command.index("--research-plan-id") + 1
    ] == plan.plan_id
    assert plan.resolved_inner_command[
        plan.resolved_inner_command.index("--research-manifest-id") + 1
    ] == manifest.manifest_id
    assert Path(
        plan.resolved_inner_command[
            plan.resolved_inner_command.index("--research-manifest-path") + 1
        ]
    ).resolve() == manifest_path.resolve()
    for option, filename in (
        ("--research-reset-identity-output", "reset_identity.json"),
        ("--research-completion-output", "completion.json"),
        ("--research-runtime-identity-output", "runtime_identity.json"),
    ):
        assert plan.resolved_inner_command[
            plan.resolved_inner_command.index(option) + 1
        ] == str(Path(plan.output_dir) / filename)
    assert "--handoff-config" not in plan.resolved_inner_command
    assert plan.env_overrides["RPENT_FULL_AGENT_PYTHON_EXECUTABLE"] == (
        plan.wrapper_command[0]
    )
    plans_path = write_child_plans((plan,), tmp_path / "full-agent-plans.json")
    assert load_child_plans(plans_path) == (plan,)
    ours_plan = build_child_plan(
        ours,
        manifest_path=manifest_path,
        repo_root=tmp_path,
        python_executable="python",
    )
    resolved_handoff = ours_plan.resolved_inner_command[
        ours_plan.resolved_inner_command.index("--handoff-config") + 1
    ]
    assert resolved_handoff == str(
        (Path(ours.output_dir) / "resolved_handoff_runtime.json").resolve()
    )
    assert ours_plan.original_harness is False
    with pytest.raises(PermissionError, match="disabled by default"):
        execute_child_plan(plan)


def test_resolved_handoff_config_binds_manifest_model_and_layer(tmp_path) -> None:
    source = tmp_path / "handoff-with-model.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "rpent.libero-handoff-runtime/v1",
                "enabled": True,
                "core": {"model_artifact": "${GENERIC_MODEL}"},
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    model, artifact = _write_fake_model_artifact(tmp_path)
    condition = ConditionSpec(
        name="ours-bound",
        execution_layer=ExecutionLayer.FULL_AGENT,
        method="outcome_calibrated_switching",
        handoff_enabled=True,
        handoff_config=str(source),
        model_artifact=str(model),
        model_artifact_id=artifact.artifact_id,
    )
    config = _config(tmp_path).model_copy(update={"conditions": (condition,)})
    trial = expand_manifest(config).trials[0]

    resolved_path = write_resolved_handoff_config(trial)
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))

    assert resolved["core"]["model_artifact"] == str(model.resolve())
    assert resolved["model_artifact_id"] == artifact.artifact_id
    assert resolved["metadata"]["execution_layer"] == "full_agent"
    assert resolved["metadata"]["trial_id"] == trial.trial_id


def test_resolved_handoff_config_preserves_source_relative_reference(tmp_path) -> None:
    source_dir = tmp_path / "policies"
    source_dir.mkdir()
    reference_path = tmp_path / "artifacts" / "positive.json"
    reference_path.parent.mkdir()
    artifact = _write_fake_positive_artifact(reference_path)
    source = source_dir / "positive.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "rpent.libero-handoff-runtime/v1",
                "enabled": True,
                "core": {
                    "policy": {
                        "name": "positive_nearest_success",
                        "positive_references_file": "../artifacts/positive.json",
                    }
                },
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    condition = ConditionSpec(
        name="positive",
        execution_layer=ExecutionLayer.FULL_AGENT,
        method="positive_nearest_success",
        handoff_enabled=True,
        handoff_config=str(source),
    )
    trial = expand_manifest(
        _config(tmp_path).model_copy(update={"conditions": (condition,)})
    ).trials[0]

    resolved = json.loads(
        write_resolved_handoff_config(trial).read_text(encoding="utf-8")
    )

    assert resolved["core"]["policy"]["positive_references_file"] == str(
        reference_path.resolve()
    )
    assert trial.artifact_bindings[
        "policy_positive_reference_artifact_id"
    ] == artifact.artifact_id
