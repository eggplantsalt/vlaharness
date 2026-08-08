from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from rpent.research.handoff.experiments.config import (
    ConditionSpec,
    DecisionMode,
    ExecutionLayer,
    ExperimentConfig,
    HierarchyMode,
    RuntimeConfig,
    TaskSpec,
    load_experiment_config,
)
from rpent.research.handoff.experiments.full_agent import (
    build_child_plan,
    build_full_agent_command,
    execute_child_plan,
)
from rpent.research.handoff.experiments.lifecycle import (
    LifecycleJournal,
    TrialEventType,
    derive_resume_states,
)
from rpent.research.handoff.experiments.manifest import expand_manifest
from rpent.research.handoff.experiments.preflight import run_offline_preflight
from rpent.research.handoff.experiments.runtime import (
    Gate0JobSpec,
    write_resolved_handoff_config,
)
from rpent.research.handoff.types import LabelSource


def _config(tmp_path) -> ExperimentConfig:
    handoff_config = tmp_path / "handoff.json"
    handoff_config.write_text("{}\n", encoding="utf-8")
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
        ),
    )


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
    }
    with pytest.raises(ValidationError, match="positive integer"):
        Gate0JobSpec.model_validate(payload)


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


def test_preflight_and_full_agent_baseline_isolation(tmp_path) -> None:
    config = _config(tmp_path)
    manifest = expand_manifest(config)
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

    plan = build_child_plan(
        baseline,
        repo_root=tmp_path,
        python_executable="python",
    )
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
    model = tmp_path / "models" / "condition.joblib"
    condition = ConditionSpec(
        name="ours-bound",
        execution_layer=ExecutionLayer.FULL_AGENT,
        method="outcome_calibrated_switching",
        handoff_enabled=True,
        handoff_config=str(source),
        model_artifact=str(model),
    )
    config = _config(tmp_path).model_copy(update={"conditions": (condition,)})
    trial = expand_manifest(config).trials[0]

    resolved_path = write_resolved_handoff_config(trial)
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))

    assert resolved["core"]["model_artifact"] == str(model.resolve())
    assert resolved["metadata"]["execution_layer"] == "full_agent"
    assert resolved["metadata"]["trial_id"] == trial.trial_id


def test_resolved_handoff_config_preserves_source_relative_reference(tmp_path) -> None:
    source_dir = tmp_path / "policies"
    source_dir.mkdir()
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
        (tmp_path / "artifacts" / "positive.json").resolve()
    )
