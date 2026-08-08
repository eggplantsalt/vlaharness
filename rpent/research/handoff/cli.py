"""Command-line orchestration for opt-in controller-handoff research.

All server/GPU execution is dry-run by default.  Commands that can start a
LIBERO, Pi0.5, or SAM3 runtime require both ``--execute`` and the exact
confirmation token printed in their plan.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


def _strict_json_object(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {source}")
            result[key] = value
        return result

    with source.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            parse_constant=reject_constant,
            object_pairs_hook=pairs_hook,
        )
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {source}")
    return value


def _atomic_json(path: str | os.PathLike[str], value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def _emit(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=False)
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("runtime probe returned NaN or infinity")
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json", exclude_none=False))
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


def _require_server_execution(args: argparse.Namespace) -> None:
    from rpent.research.handoff.experiments.runtime import EXECUTION_CONFIRMATION

    if not args.execute:
        return
    if args.execution_token != EXECUTION_CONFIRMATION:
        raise PermissionError(
            "server execution needs --execution-token " + EXECUTION_CONFIRMATION
        )


def _load_outcome_dataset(paths: Sequence[str]):
    """Load either checksummed dataset envelopes or raw runtime records."""
    from rpent.research.handoff.dataset import OutcomeDataset
    from rpent.research.handoff.evaluation.aggregate import read_outcome_jsonl

    records = []
    for path in paths:
        source = Path(path)
        first: dict[str, Any] | None = None
        with source.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, start=1):
                line = raw.strip()
                if not line:
                    raise ValueError(f"blank outcome line in {source} at {line_number}")
                first = json.loads(line)
                break
        if first is None:
            raise ValueError(f"outcome JSONL is empty: {source}")
        if {
            "payload",
            "payload_sha256",
            "sequence",
            "schema_version",
        }.issubset(first):
            records.extend(
                OutcomeDataset.from_jsonl(
                    source, allow_partial_final_line=False
                ).records
            )
        else:
            records.extend(read_outcome_jsonl(source))
    return OutcomeDataset.from_records(records)


def _cmd_manifest(args: argparse.Namespace) -> int:
    from rpent.research.handoff.experiments.config import load_experiment_config
    from rpent.research.handoff.experiments.manifest import (
        expand_manifest,
        write_manifest,
        write_trial_jsonl,
    )

    config = load_experiment_config(args.config)
    manifest = expand_manifest(config, config_path=args.config)
    destination = write_manifest(manifest, args.output)
    trials_path = None
    if args.trials_jsonl is not None:
        trials_path = write_trial_jsonl(manifest, args.trials_jsonl)
    _emit(
        {
            "manifest_id": manifest.manifest_id,
            "configuration_id": manifest.configuration_id,
            "trials": len(manifest.trials),
            "manifest_path": str(destination.resolve()),
            "trials_jsonl": str(trials_path.resolve()) if trials_path else None,
        }
    )
    return 0


def _cmd_offline_preflight(args: argparse.Namespace) -> int:
    from rpent.research.handoff.experiments.config import load_experiment_config
    from rpent.research.handoff.experiments.manifest import load_manifest
    from rpent.research.handoff.experiments.preflight import run_offline_preflight

    config = load_experiment_config(args.config)
    manifest = load_manifest(args.manifest) if args.manifest else None
    report = run_offline_preflight(
        config,
        manifest,
        config_path=args.config,
        require_referenced_paths=not args.allow_missing_references,
    )
    payload = report.model_dump(mode="json", exclude_none=False)
    if args.output:
        _atomic_json(args.output, payload)
    _emit(payload)
    return 0 if report.ok else 2


def _rpc_client(endpoint: str):
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.rpc import parse_endpoint
    from rpent.utils.socket_rpc import SocketRpcClient

    protocol, host, port = parse_endpoint(endpoint)
    if protocol == "http":
        return HttpRpcClient(f"http://{host}:{port}")
    if protocol == "socket":
        return SocketRpcClient(host, port)
    raise ValueError(f"unsupported RPC protocol in endpoint: {endpoint}")


def _probe_component(
    name: str,
    endpoint: str,
    method: str,
    *,
    timeout_s: float,
) -> tuple[Any | None, dict[str, Any]]:
    client = _rpc_client(endpoint)
    try:
        health = client.call("healthz", timeout_s=timeout_s)
        payload = client.call(method, timeout_s=timeout_s)
        return client, {
            "status": "pass",
            "endpoint": endpoint,
            "healthz": _json_safe(health),
            "probe": _json_safe(payload),
        }
    except Exception as exc:
        return None, {
            "status": "fail",
            "endpoint": endpoint,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "component": name,
        }


def _cmd_probe_runtime(args: argparse.Namespace) -> int:
    from rpent.research.handoff.experiments.probes import (
        ProbeStatus,
        RuntimeProbeOptions,
        run_runtime_probes,
    )

    runtime = None
    if args.config:
        from rpent.research.handoff.experiments.config import load_experiment_config

        runtime = load_experiment_config(args.config).runtime
    endpoints = {
        "env": args.env_endpoint or (runtime.env_endpoint if runtime else None),
        "vla": args.vla_endpoint or (runtime.vla_endpoint if runtime else None),
        "sam3": args.sam3_endpoint or (runtime.sam3_endpoint if runtime else None),
    }
    if not any(endpoints.values()):
        raise ValueError(
            "probe-runtime needs configured external endpoints; it never silently "
            "starts heavyweight servers"
        )
    raw_clients = {
        name: (_rpc_client(endpoint) if endpoint is not None else None)
        for name, endpoint in endpoints.items()
    }

    class EnvProbeClient:
        def __init__(self, client) -> None:
            self.client = client

        def runtime_probe(self):
            return self.client.call("env.runtime_probe", timeout_s=args.timeout_s)

        def diagnostic_chunk_step(self, actions):
            return self.client.call(
                "env.diagnostic_chunk_step",
                args=(actions,),
                timeout_s=max(args.timeout_s, 120.0),
            )

    env_client = (
        EnvProbeClient(raw_clients["env"])
        if raw_clients["env"] is not None
        else None
    )
    if raw_clients["vla"] is not None:
        from rpent.utils.vla_client import VLAClient

        vla_client = VLAClient(raw_clients["vla"])
    else:
        vla_client = None
    if raw_clients["sam3"] is not None:
        from rpent.utils.sam3_client import Sam3Client

        sam3_client = Sam3Client(raw_clients["sam3"], timeout_s=args.timeout_s)
    else:
        sam3_client = None

    if (
        args.capture_vla_observation_npz is not None
        and args.vla_observation_npz is not None
    ):
        raise ValueError(
            "use either --vla-observation-npz or "
            "--capture-vla-observation-npz, not both"
        )
    observation = None
    sam_image = None
    chunk_actions = None
    captured_observation_path = None
    if any(
        path is not None
        for path in (
            args.vla_observation_npz,
            args.capture_vla_observation_npz,
            args.sam3_image_npy,
            args.chunk_actions_npy,
        )
    ):
        import numpy as np

        if args.vla_observation_npz is not None:
            with np.load(args.vla_observation_npz, allow_pickle=False) as archive:
                required = {"main_images", "states"}
                missing = sorted(required.difference(archive.files))
                if missing:
                    raise ValueError(
                        "VLA observation NPZ is missing arrays: " + ", ".join(missing)
                    )
                observation = {
                    "main_images": archive["main_images"],
                    "states": archive["states"],
                    "task_descriptions": args.inference_instruction or "",
                }
                for optional in ("wrist_images", "extra_view_images"):
                    if optional in archive.files:
                        observation[optional] = archive[optional]
        if args.capture_vla_observation_npz is not None:
            if raw_clients["env"] is None:
                raise ValueError(
                    "--capture-vla-observation-npz requires --env-endpoint"
                )
            if not (
                args.fresh_env_reset_confirmed
                and args.isolated_env_trial_confirmed
            ):
                raise PermissionError(
                    "capturing a VLA observation resets the env; pass both "
                    "--fresh-env-reset-confirmed and "
                    "--isolated-env-trial-confirmed for a disposable trial"
                )
            reset_payload = raw_clients["env"].call(
                "env.reset", timeout_s=max(args.timeout_s, 120.0)
            )
            if (
                not isinstance(reset_payload, (list, tuple))
                or len(reset_payload) != 2
                or not isinstance(reset_payload[0], dict)
            ):
                raise ValueError(
                    "env.reset did not return the expected (observation, info) pair"
                )
            reset_observation = reset_payload[0]
            required = {"main_images", "states"}
            missing = sorted(required.difference(reset_observation))
            if missing:
                raise ValueError(
                    "captured env observation is missing VLA arrays: "
                    + ", ".join(missing)
                )
            arrays = {
                key: np.asarray(reset_observation[key])
                for key in (
                    "main_images",
                    "states",
                    "wrist_images",
                    "extra_view_images",
                )
                if key in reset_observation
            }
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("captured VLA observation contains NaN or infinity")
            capture_path = Path(args.capture_vla_observation_npz)
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            if capture_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite captured observation: {capture_path}"
                )
            np.savez(capture_path, **arrays)
            captured_observation_path = str(capture_path.resolve())
            observation = {
                **arrays,
                "task_descriptions": args.inference_instruction or "",
            }
        if args.sam3_image_npy is not None:
            sam_image = np.load(args.sam3_image_npy, allow_pickle=False)
        if args.chunk_actions_npy is not None:
            chunk_actions = np.load(args.chunk_actions_npy, allow_pickle=False)

    hidden_diagnostic = None
    if args.hidden_state_diagnostic is not None:
        from rpent.research.handoff.experiments.runtime import load_object

        hidden_diagnostic = load_object(args.hidden_state_diagnostic)
        if not callable(hidden_diagnostic):
            raise TypeError("hidden-state diagnostic object must be callable")

    options = RuntimeProbeOptions(
        run_host_gpu_discovery=args.discover_host_gpu,
        run_vla_inference=args.allow_model_inference,
        run_sam3_inference=args.allow_sam3_inference,
        run_destructive_chunk_diagnostic=args.allow_destructive_chunk_diagnostic,
        run_hidden_state_diagnostic=args.allow_hidden_state_diagnostic,
        fresh_env_reset_confirmed=args.fresh_env_reset_confirmed,
        isolated_env_trial_confirmed=args.isolated_env_trial_confirmed,
        isolated_model_session_confirmed=args.isolated_model_session_confirmed,
    )
    health_checks = {
        name: (lambda client=client: client.call("healthz", timeout_s=args.timeout_s))
        for name, client in raw_clients.items()
        if client is not None
    }
    probe = run_runtime_probes(
        env_client=env_client,
        vla_client=vla_client,
        sam3_client=sam3_client,
        options=options,
        vla_probe_observation=observation,
        sam3_probe_image=sam_image,
        sam3_text_prompt=args.sam3_text_prompt,
        chunk_diagnostic_actions=chunk_actions,
        health_checks=health_checks,
        hidden_state_diagnostic=hidden_diagnostic,
    )
    ok = not any(fact.status is ProbeStatus.ERROR for fact in probe.facts)
    report = {
        **probe.model_dump(mode="json", exclude_none=False),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "captured_vla_observation_npz": captured_observation_path,
        "pending_diagnostics": [fact.name for fact in probe.pending_diagnostics],
    }
    if args.output:
        _atomic_json(args.output, report)
    _emit(report)
    return 0 if ok else 2


def _gate0_plan(args: argparse.Namespace) -> dict[str, Any]:
    from rpent.research.handoff.experiments.runtime import (
        EXECUTION_CONFIRMATION,
        load_gate0_job,
    )

    job = load_gate0_job(args.job)
    command = [
        args.python_executable or sys.executable,
        "-m",
        "rpent.research.handoff",
        "--traceback",
        "_gate0-child",
        "--job",
        str(Path(args.job).resolve()),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    command.append("--resume" if args.resume else "--no-resume")
    return {
        "schema_version": "rpent.handoff-gate0-child-plan/v1",
        "job_configuration_id": job.stable_configuration_id,
        "command": command,
        "cwd": str(Path(args.repo_root).resolve()),
        "output_dir": job.output_dir,
        "execution_confirmation": EXECUTION_CONFIRMATION,
    }


def _cmd_collect_gate0(args: argparse.Namespace) -> int:
    plan = _gate0_plan(args)
    plan_path = args.plan_output or str(
        Path(plan["output_dir"]) / "gate0_child_plan.json"
    )
    _atomic_json(plan_path, plan)
    if not args.execute:
        _emit({**plan, "dry_run": True, "plan_path": str(Path(plan_path).resolve())})
        return 0
    _require_server_execution(args)
    completed = subprocess.run(
        plan["command"],
        cwd=plan["cwd"],
        shell=False,
        check=False,
        timeout=args.timeout_s,
    )
    _emit(
        {
            "dry_run": False,
            "returncode": completed.returncode,
            "plan_path": str(Path(plan_path).resolve()),
        }
    )
    return completed.returncode


class _CompositeSetupSink:
    def __init__(self, sinks: Sequence[Any]) -> None:
        self._sinks = tuple(sinks)

    def append_setup(self, setup: Any) -> None:
        for sink in self._sinks:
            sink.append_setup(setup)


def _cmd_gate0_child(args: argparse.Namespace) -> int:
    from rpent.research.handoff.dataset import DatasetResearchSink, OutcomeDataset
    from rpent.research.handoff.experiments.gate0 import (
        Gate0Adapter,
        Gate0Collector,
        Gate0Config,
        Gate0RunIdentity,
    )
    from rpent.research.handoff.experiments.runtime import (
        SetupJsonlSink,
        instantiate_gate0_adapter,
        load_gate0_job,
    )
    from rpent.research.handoff.types import ControllerIdentity, SkillIdentity

    job = load_gate0_job(args.job)
    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gate0_config = Gate0Config.model_validate(job.gate0)
    run_manifest = {
        "schema_version": "rpent.handoff-gate0-run-manifest/v1",
        "configuration_id": job.stable_configuration_id,
        "source_job_path": str(Path(args.job).resolve()),
        "job": job.model_dump(mode="json", exclude_none=False),
    }
    run_manifest_path = output_dir / "gate0_run_manifest.json"
    if run_manifest_path.exists():
        existing_manifest = _strict_json_object(run_manifest_path)
        if existing_manifest != run_manifest:
            raise ValueError(
                "existing Gate-0 run manifest disagrees with this job; "
                "choose a new output_dir/run_id"
            )
    else:
        _atomic_json(run_manifest_path, run_manifest)
    bundle = instantiate_gate0_adapter(
        job,
        gate0_config=gate0_config,
        output_dir=output_dir,
    )
    try:
        if not isinstance(bundle.adapter, Gate0Adapter):
            raise TypeError(
                "configured Gate-0 adapter does not implement reset/current EEF/"
                "governor adapter methods"
            )
        dataset_dir = output_dir / "online"
        outcome_sink = DatasetResearchSink(dataset_dir, fsync=True)
        durable_setup = SetupJsonlSink(output_dir / "privileged" / "setups.jsonl")
        setup_sink = (
            _CompositeSetupSink((durable_setup, bundle.setup_sink))
            if bundle.setup_sink is not None
            else durable_setup
        )
        completed_trial_ids: set[str] = set()
        outcome_path = dataset_dir / "outcomes.jsonl"
        if args.resume and outcome_path.exists():
            existing_records = OutcomeDataset.from_jsonl(outcome_path).records
            expected_samples = {
                f"{job.episode_prefix}-trial-{sample.sample_id}"
                for sample in Gate0Collector(
                    adapter=bundle.adapter,
                    config=gate0_config,
                    skill=SkillIdentity(
                        name=job.skill_name,
                        semantic_target=job.target_description,
                        learned_controller="pi0.5",
                    ),
                    controller=ControllerIdentity(
                        method=job.controller_method,
                        implementation_version=job.controller_implementation_version,
                        checkpoint_id=job.checkpoint_id,
                        configuration_id=job.stable_configuration_id,
                    ),
                    run_identity=Gate0RunIdentity(
                        run_id=job.run_id,
                        suite=job.suite,
                        task_id=job.task_id,
                        seed=job.seed,
                        episode_prefix=job.episode_prefix,
                        source_revision=job.source_revision,
                    ),
                    outcome_sink=outcome_sink,
                    setup_sink=setup_sink,
                    vla_kwargs={"prompt": job.skill_prompt, **job.vla_kwargs},
                ).samples()
            }
            invalid = [
                record.record_id
                for record in existing_records
                if (
                    record.identity.trial_id not in expected_samples
                    or record.identity.run_id != job.run_id
                    or record.identity.suite != job.suite
                    or str(record.identity.task_id) != str(job.task_id)
                    or record.identity.seed != job.seed
                    or record.identity.reset_id is None
                    or record.skill.name != job.skill_name
                    or record.skill.semantic_target != job.target_description
                    or record.controller.method != job.controller_method
                    or record.controller.implementation_version
                    != job.controller_implementation_version
                    or record.controller.checkpoint_id != job.checkpoint_id
                    or record.controller.configuration_id
                    != job.stable_configuration_id
                    or record.source_revision != job.source_revision
                )
            ]
            if invalid:
                raise ValueError(
                    "existing Gate-0 outcomes disagree with the immutable run "
                    f"manifest: {invalid[:10]}"
                )
            completed_trial_ids = {
                record.identity.trial_id for record in existing_records
            }
        collector = Gate0Collector(
            adapter=bundle.adapter,
            config=gate0_config,
            skill=SkillIdentity(
                name=job.skill_name,
                semantic_target=job.target_description,
                learned_controller="pi0.5",
            ),
            controller=ControllerIdentity(
                method=job.controller_method,
                implementation_version=job.controller_implementation_version,
                checkpoint_id=job.checkpoint_id,
                configuration_id=job.stable_configuration_id,
            ),
            run_identity=Gate0RunIdentity(
                run_id=job.run_id,
                suite=job.suite,
                task_id=job.task_id,
                seed=job.seed,
                episode_prefix=job.episode_prefix,
                source_revision=job.source_revision,
            ),
            outcome_sink=outcome_sink,
            setup_sink=setup_sink,
            completed_trial_ids=completed_trial_ids,
            vla_kwargs={"prompt": job.skill_prompt, **job.vla_kwargs},
        )
        outcomes = collector.collect(limit=args.limit)
    finally:
        if bundle.cleanup is not None:
            bundle.cleanup()
    summary = {
        "schema_version": "rpent.handoff-gate0-collection-summary/v1",
        "configuration_id": job.stable_configuration_id,
        "collected": len(outcomes),
        "resumed_completed": len(completed_trial_ids),
        "outcome_jsonl": str(outcome_sink.outcome_path.resolve()),
        "decision_jsonl": str(outcome_sink.decision_path.resolve()),
        "setup_jsonl": str(durable_setup.path.resolve()),
        "run_manifest": str(run_manifest_path.resolve()),
        "record_ids": [record.record_id for record in outcomes],
    }
    summary_path = _atomic_json(output_dir / "collection_summary.json", summary)
    _emit({**summary, "summary_path": str(summary_path.resolve())})
    return 0


def _load_feature_spec(path: str):
    from rpent.research.handoff.features import FeatureSpec, make_feature_spec

    value = _strict_json_object(path)
    if "fields" in value or "spec_id" in value:
        return FeatureSpec.model_validate(value)
    allowed = {"preset", "skill_vocabulary"}
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"unknown compact feature spec keys: {unknown}")
    if "preset" not in value or "skill_vocabulary" not in value:
        raise ValueError(
            "compact feature spec needs preset and skill_vocabulary"
        )
    return make_feature_spec(
        value["preset"],
        skill_vocabulary=value["skill_vocabulary"],
    )


def _git_source_identity(
    repo_root: str | None,
    *,
    external_runtime_identity: str | None,
):
    from rpent.research.handoff.artifacts import SourceIdentity

    revision = None
    dirty = None
    if repo_root is not None:
        root = Path(repo_root).resolve()
        revision_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
        )
        if revision_result.returncode == 0:
            revision = revision_result.stdout.strip() or None
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
        )
        if status_result.returncode == 0:
            dirty = bool(status_result.stdout.strip())
    try:
        package_version = importlib.metadata.version("rpent")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    return SourceIdentity(
        git_revision=revision,
        dirty=dirty,
        package_version=package_version,
        external_runtime_identity=external_runtime_identity,
    )


def _cmd_train(args: argparse.Namespace) -> int:
    from rpent.research.handoff.artifacts import save_model_artifact
    from rpent.research.handoff.training import (
        OutcomeTrainingConfig,
        train_outcome_model,
    )

    dataset = _load_outcome_dataset(args.outcomes)
    feature_spec = _load_feature_spec(args.feature_spec)
    training_config = OutcomeTrainingConfig.model_validate(
        _strict_json_object(args.training_config)
    )
    result = train_outcome_model(
        dataset,
        feature_spec=feature_spec,
        config=training_config,
    )
    artifact_dir = Path(args.artifact_dir)
    report_path = Path(args.report_output) if args.report_output else (
        artifact_dir / "training_report.json"
    )
    assignment_path = (
        Path(args.assignment_output)
        if args.assignment_output
        else artifact_dir / "split_assignment.json"
    )
    if not args.overwrite:
        existing = [path for path in (report_path, assignment_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "training outputs already exist; pass --overwrite explicitly: "
                + ", ".join(str(path) for path in existing)
            )
    source_identity = _git_source_identity(
        args.repo_root,
        external_runtime_identity=args.external_runtime_identity,
    )
    manifest = save_model_artifact(
        artifact_dir,
        model=result.model,
        feature_spec=result.feature_spec,
        training_target_label=training_config.target_label.value,
        dataset_fingerprint=result.report.eligible_dataset_fingerprint,
        training_configuration=training_config.model_dump(mode="json"),
        source_identity=source_identity,
        calibration_method=training_config.calibration_method,
        split_assignment_fingerprint=result.assignment.fingerprint,
        training_record_ids=tuple(
            record.record_id for record in result.train.records
        ),
        calibration_record_ids=tuple(
            record.record_id for record in result.calibration.records
        ),
        held_out_record_ids=tuple(
            record.record_id for record in result.test.records
        ),
        overwrite=args.overwrite,
    )
    _atomic_json(
        report_path,
        result.report.model_dump(mode="json", exclude_none=False),
    )
    _atomic_json(
        assignment_path,
        result.assignment.model_dump(mode="json", exclude_none=False),
    )
    _emit(
        {
            "artifact_id": manifest.artifact_id,
            "artifact_dir": str(artifact_dir.resolve()),
            "training_report": str(report_path.resolve()),
            "split_assignment": str(assignment_path.resolve()),
            "partitions": {
                "train": result.report.train.model_dump(mode="json"),
                "calibration": result.report.calibration.model_dump(mode="json"),
                "test": result.report.test.model_dump(mode="json"),
            },
        }
    )
    return 0


def _cmd_positive_references(args: argparse.Namespace) -> int:
    from rpent.research.handoff.baseline_data import (
        build_positive_reference_artifact,
        write_positive_reference_artifact,
    )

    destination = Path(args.output)
    if destination.exists() and not args.overwrite:
        raise FileExistsError(
            f"positive-reference artifact exists: {destination}; pass --overwrite"
        )
    dataset = _load_outcome_dataset(args.outcomes)
    artifact = build_positive_reference_artifact(
        dataset,
        target=args.target_label,
        maximum_references=args.maximum_references,
    )
    path = write_positive_reference_artifact(artifact, destination)
    _emit(
        {
            "artifact_id": artifact.artifact_id,
            "dataset_fingerprint": artifact.dataset_fingerprint,
            "target_label": artifact.target_label.value,
            "reference_count": len(artifact.references),
            "output": str(path.resolve()),
        }
    )
    return 0


def _cmd_materialize_splits(args: argparse.Namespace) -> int:
    """Write the exact train/calibration/test cohorts from a saved assignment."""
    from rpent.research.handoff.dataset import (
        OutcomeJsonlWriter,
        dataset_fingerprint,
    )
    from rpent.research.handoff.splits import (
        SplitAssignment,
        SplitName,
        apply_split_assignment,
    )

    dataset = _load_outcome_dataset(args.outcomes)
    assignment = SplitAssignment.model_validate(
        _strict_json_object(args.assignment)
    )
    partitions = apply_split_assignment(dataset.records, assignment)
    empty = [split.value for split, records in partitions.items() if not records]
    if empty:
        raise ValueError(
            "refusing to materialize empty split partitions: " + ", ".join(empty)
        )

    destination = Path(args.output_dir)
    final_paths = {
        split: destination / f"{split.value}.jsonl" for split in SplitName
    }
    report_path = destination / "split_materialization.json"
    occupied = [
        path for path in (*final_paths.values(), report_path) if path.exists()
    ]
    if occupied:
        raise FileExistsError(
            "split output already exists; choose a new output directory: "
            + ", ".join(str(path) for path in occupied)
        )
    destination.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        split: path.with_name(f"{path.name}.tmp")
        for split, path in final_paths.items()
    }
    stale = [path for path in temporary_paths.values() if path.exists()]
    if stale:
        raise FileExistsError(
            "stale split temporary files require manual inspection: "
            + ", ".join(str(path) for path in stale)
        )

    try:
        for split in SplitName:
            writer = OutcomeJsonlWriter(temporary_paths[split])
            for record in partitions[split]:
                writer.append(record)
        for split in SplitName:
            os.replace(temporary_paths[split], final_paths[split])
    except BaseException:
        # Preserve any temporary shards for diagnosis; never pretend an
        # interrupted materialization is complete.
        raise

    assigned_records = tuple(
        record for split in SplitName for record in partitions[split]
    )
    report = {
        "schema_version": "rpent.handoff-split-materialization/v1",
        "raw_dataset_fingerprint": dataset.fingerprint,
        "eligible_dataset_fingerprint": dataset_fingerprint(assigned_records),
        "assignment_fingerprint": assignment.fingerprint,
        "unassigned_record_count": len(dataset.records) - len(assigned_records),
        "partitions": {
            split.value: {
                "records": len(partitions[split]),
                "path": str(final_paths[split].resolve()),
            }
            for split in SplitName
        },
    }
    _atomic_json(report_path, report)
    _emit({**report, "report": str(report_path.resolve())})
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from rpent.research.handoff.artifacts import load_model_artifact
    from rpent.research.handoff.dataset import (
        TrainingTarget,
        extract_labeled_outcomes,
    )
    from rpent.research.handoff.evaluation.metrics import (
        evaluate_binary_predictions,
    )
    from rpent.research.handoff.features import FeatureBuilder

    dataset = _load_outcome_dataset(args.outcomes)
    model, manifest = load_model_artifact(
        args.artifact_dir,
        trusted=args.trust_artifact,
    )
    target = TrainingTarget(args.target_label or manifest.training_target_label)
    extracted = extract_labeled_outcomes(dataset.records, target=target)
    if not extracted.included:
        raise ValueError(f"no eligible evaluation labels for {target.value!r}")
    from rpent.research.handoff.dataset import dataset_fingerprint

    eligible_records = tuple(item.record for item in extracted.included)
    eligible_fingerprint = dataset_fingerprint(eligible_records)
    evaluation_ids = {record.record_id for record in eligible_records}
    fitted_ids = set(manifest.training_record_ids).union(
        manifest.calibration_record_ids
    )
    overlap = sorted(evaluation_ids.intersection(fitted_ids))
    if overlap and not args.allow_training_data_evaluation:
        raise ValueError(
            "evaluation records overlap the artifact training/calibration "
            f"cohort ({len(overlap)} IDs; first={overlap[:10]}); supply the "
            "materialized held-out split or explicitly pass "
            "--allow-training-data-evaluation"
        )
    builder = FeatureBuilder(manifest.feature_spec)
    predictions = []
    labels = []
    probabilities = []
    uncertainties = []
    for item in extracted.included:
        vector = builder.build(item.record.pre_handoff_state)
        estimate = model.predict_one(vector)
        label = int(item.value)
        labels.append(label)
        probabilities.append(estimate.mean_success_probability)
        uncertainties.append(estimate.epistemic_std)
        predictions.append(
            {
                "record_id": item.record.record_id,
                "label": label,
                "estimate": estimate.model_dump(mode="json", exclude_none=False),
            }
        )
    metrics = evaluate_binary_predictions(
        labels,
        probabilities,
        uncertainties=uncertainties,
        threshold=args.threshold,
        n_bins=args.calibration_bins,
    )
    payload = {
        "schema_version": "rpent.handoff-heldout-evaluation/v1",
        "artifact_id": manifest.artifact_id,
        "raw_evaluation_dataset_fingerprint": dataset.fingerprint,
        "eligible_evaluation_dataset_fingerprint": eligible_fingerprint,
        "training_dataset_fingerprint": manifest.dataset_fingerprint,
        "target_label": target.value,
        "caller_asserted_training_data_evaluation": args.allow_training_data_evaluation,
        "overlap_check": (
            "exact dataset fingerprint only; artifact does not contain training record IDs"
        ),
        "included": len(predictions),
        "excluded": len(extracted.excluded),
        "metrics": metrics.model_dump(mode="json", exclude_none=False),
        "predictions": predictions,
    }
    _atomic_json(args.output, payload)
    _emit({**payload, "predictions": f"{len(predictions)} rows", "output": str(Path(args.output).resolve())})
    return 0


def _cmd_aggregate(args: argparse.Namespace) -> int:
    from rpent.research.handoff.evaluation.aggregate import (
        aggregate_outcomes,
        write_aggregation,
    )

    dataset = _load_outcome_dataset(args.outcomes)
    result = aggregate_outcomes(
        dataset.records,
        target_label=args.target_label,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    artifacts = write_aggregation(result, dataset.records, args.output_dir)
    _emit(
        {
            "aggregation": result.model_dump(mode="json", exclude_none=False),
            "artifacts": artifacts.model_dump(mode="json", exclude_none=False),
        }
    )
    return 0


def _coerce_csv_value(value: str | None) -> Any:
    if value is None or value == "":
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        try:
            number = float(value)
        except ValueError:
            return value
        if not math.isfinite(number):
            raise ValueError("CSV plot input contains NaN or infinity")
        return number


def _read_csv_rows(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _cmd_plot(args: argparse.Namespace) -> int:
    from rpent.research.handoff.evaluation.aggregate import AggregationResult
    from rpent.research.handoff.evaluation.plotting import (
        plot_ablation,
        plot_calibration_curve,
        plot_gate0_landscape,
        plot_handoff_regret,
        plot_method_success_cost,
    )

    if args.kind in {"calibration", "success-cost", "regret"}:
        if args.summary is None:
            raise ValueError(f"plot kind {args.kind!r} requires --summary")
        result = AggregationResult.model_validate(
            _strict_json_object(args.summary)
        )
        functions: dict[str, Callable[..., Path]] = {
            "calibration": plot_calibration_curve,
            "success-cost": plot_method_success_cost,
            "regret": plot_handoff_regret,
        }
        path = functions[args.kind](result, args.output)
    else:
        if args.rows_csv is None:
            raise ValueError(f"plot kind {args.kind!r} requires --rows-csv")
        rows = _read_csv_rows(args.rows_csv)
        if args.kind == "gate0":
            path = plot_gate0_landscape(
                rows,
                args.output,
                x_key=args.x_key,
                y_key=args.y_key,
                value_key=args.value_key,
            )
        else:
            if args.factor_key is None:
                raise ValueError("ablation plot requires --factor-key")
            path = plot_ablation(
                rows,
                args.output,
                factor_key=args.factor_key,
                value_key=args.value_key,
            )
    _emit({"kind": args.kind, "output": str(path.resolve())})
    return 0


def _selected_trials(
    args: argparse.Namespace,
    *,
    layer: str,
):
    from rpent.research.handoff.experiments.config import ExecutionLayer
    from rpent.research.handoff.experiments.lifecycle import resumable_trials
    from rpent.research.handoff.experiments.manifest import (
        load_manifest,
        select_trials,
    )

    manifest = load_manifest(args.manifest)
    layer_enum = ExecutionLayer(layer)
    requested_ids = set(args.trial_id) if args.trial_id else None
    requested_conditions = set(args.condition) if args.condition else None
    known_ids = {trial.trial_id for trial in manifest.trials}
    if requested_ids is not None:
        unknown = sorted(requested_ids.difference(known_ids))
        if unknown:
            raise ValueError(f"unknown requested trial IDs: {unknown}")
    known_conditions = {trial.condition.name for trial in manifest.trials}
    if requested_conditions is not None:
        unknown = sorted(requested_conditions.difference(known_conditions))
        if unknown:
            raise ValueError(f"unknown requested conditions: {unknown}")

    selected = select_trials(
        manifest,
        execution_layer=layer_enum,
        condition_names=requested_conditions,
        trial_ids=requested_ids,
    )
    journal_path = Path(args.journal) if args.journal else (
        Path(args.manifest).resolve().parent / f"{layer}_lifecycle.jsonl"
    )
    from rpent.research.handoff.experiments.lifecycle import LifecycleJournal

    journal = LifecycleJournal(
        journal_path,
        allowed_trial_ids={trial.trial_id for trial in manifest.trials},
    )
    events = journal.read()
    if args.execute and events and not args.resume:
        raise RuntimeError(
            f"lifecycle journal already contains events ({journal_path}); "
            "use --resume to avoid duplicate execution"
        )
    if args.resume:
        resumable = {
            trial.trial_id
            for trial in resumable_trials(
                manifest,
                events,
                retry_failed=args.retry_failed,
                retry_cancelled=args.retry_cancelled,
            )
        }
        selected = tuple(
            trial for trial in selected if trial.trial_id in resumable
        )
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        selected = selected[: args.limit]
    return manifest, selected, journal, journal_path


def _execute_plans(
    plans: Sequence[Any],
    *,
    journal: Any,
    execute: Callable[[Any], tuple[int, str | None, str | None]],
    continue_on_error: bool,
) -> tuple[list[dict[str, Any]], bool]:
    from rpent.research.handoff.experiments.lifecycle import TrialEventType

    results: list[dict[str, Any]] = []
    failed = False
    for plan in plans:
        journal.append(
            plan.trial_id,
            TrialEventType.STARTED,
            artifact_path=plan.output_dir,
            details={"plan_id": plan.plan_id},
        )
        try:
            returncode, stdout, stderr = execute(plan)
        except KeyboardInterrupt:
            journal.append(
                plan.trial_id,
                TrialEventType.CANCELLED,
                message="launcher interrupted by user",
                artifact_path=plan.output_dir,
            )
            raise
        except Exception as exc:
            journal.append(
                plan.trial_id,
                TrialEventType.FAILED,
                message=str(exc),
                artifact_path=plan.output_dir,
                details={"error_type": type(exc).__name__},
            )
            results.append(
                {
                    "trial_id": plan.trial_id,
                    "plan_id": plan.plan_id,
                    "returncode": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            failed = True
            if not continue_on_error:
                break
            continue
        result = {
            "trial_id": plan.trial_id,
            "plan_id": plan.plan_id,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
        results.append(result)
        if returncode == 0:
            journal.append(
                plan.trial_id,
                TrialEventType.COMPLETED,
                artifact_path=plan.output_dir,
                details={"plan_id": plan.plan_id, "returncode": returncode},
            )
        elif returncode == 130:
            failed = True
            journal.append(
                plan.trial_id,
                TrialEventType.CANCELLED,
                message="child reported cancellation",
                artifact_path=plan.output_dir,
                details={"plan_id": plan.plan_id, "returncode": returncode},
            )
            if not continue_on_error:
                break
        else:
            failed = True
            journal.append(
                plan.trial_id,
                TrialEventType.FAILED,
                message=f"child exited with code {returncode}",
                artifact_path=plan.output_dir,
                details={"plan_id": plan.plan_id, "returncode": returncode},
            )
            if not continue_on_error:
                break
    return results, failed


def _cmd_run_controlled(args: argparse.Namespace) -> int:
    from rpent.research.handoff.experiments.runtime import (
        build_controlled_child_plan,
        execute_controlled_child_plan,
        write_controlled_plans,
    )

    _require_server_execution(args)
    _manifest, trials, journal, journal_path = _selected_trials(
        args,
        layer="controlled",
    )
    if not trials:
        _emit({"dry_run": not args.execute, "trials": 0, "reason": "no selected pending trials"})
        return 0
    plans = tuple(
        build_controlled_child_plan(
            trial,
            manifest_path=args.manifest,
            repo_root=args.repo_root,
            python_executable=args.python_executable,
        )
        for trial in trials
    )
    plans_path = Path(args.plans_output) if args.plans_output else (
        Path(args.manifest).resolve().parent / "controlled_child_plans.json"
    )
    write_controlled_plans(plans, plans_path)
    if not args.execute:
        _emit(
            {
                "dry_run": True,
                "trials": len(plans),
                "plans": str(plans_path.resolve()),
                "journal": str(journal_path.resolve()),
                "execution_confirmation": (
                    "I_UNDERSTAND_SERVER_EXECUTION"
                ),
            }
        )
        return 0

    def execute(plan: Any) -> tuple[int, str | None, str | None]:
        completed = execute_controlled_child_plan(
            plan,
            allow_execution=True,
            capture_output=args.capture_output,
            timeout_s=args.timeout_s,
        )
        return completed.returncode, completed.stdout, completed.stderr

    results, failed = _execute_plans(
        plans,
        journal=journal,
        execute=execute,
        continue_on_error=args.continue_on_error,
    )
    results_path = Path(args.results_output) if args.results_output else (
        plans_path.with_name("controlled_child_results.json")
    )
    _atomic_json(results_path, results)
    _emit(
        {
            "dry_run": False,
            "planned": len(plans),
            "executed": len(results),
            "failed": failed,
            "results": str(results_path.resolve()),
            "journal": str(journal_path.resolve()),
        }
    )
    return 1 if failed else 0


def _cmd_controlled_child(args: argparse.Namespace) -> int:
    from rpent.research.handoff.experiments.runtime import run_controlled_trial

    summary = run_controlled_trial(args.manifest, args.trial_id)
    _emit(summary)
    return 130 if summary.get("cancelled") else 0


def _cmd_run_full_agent(args: argparse.Namespace) -> int:
    from rpent.research.handoff.experiments.full_agent import (
        execute_child_plan,
        plan_full_agent_trials,
        write_child_plans,
    )
    from rpent.research.handoff.experiments.runtime import (
        write_resolved_handoff_config,
    )

    _require_server_execution(args)
    _manifest, trials, journal, journal_path = _selected_trials(
        args,
        layer="full_agent",
    )
    if not trials:
        _emit({"dry_run": not args.execute, "trials": 0, "reason": "no selected pending trials"})
        return 0
    # Materialize the exact controller configuration that each opt-in child
    # will consume.  The Original Harness trials are deliberately left byte-for-
    # byte on their existing command path and never receive --handoff-config.
    planned_trials = []
    for trial in trials:
        if trial.condition.handoff_enabled:
            resolved_path = write_resolved_handoff_config(trial)
            trial = trial.model_copy(
                update={"handoff_config_path": str(resolved_path.resolve())}
            )
        planned_trials.append(trial)
    plans = plan_full_agent_trials(
        planned_trials,
        repo_root=args.repo_root,
        python_executable=args.python_executable,
    )
    plans_path = Path(args.plans_output) if args.plans_output else (
        Path(args.manifest).resolve().parent / "full_agent_child_plans.json"
    )
    write_child_plans(plans, plans_path)
    if not args.execute:
        _emit(
            {
                "dry_run": True,
                "trials": len(plans),
                "plans": str(plans_path.resolve()),
                "journal": str(journal_path.resolve()),
                "execution_confirmation": (
                    "I_UNDERSTAND_SERVER_EXECUTION"
                ),
            }
        )
        return 0

    stale_episode_artifacts = {
        plan.trial_id: [
            str(path)
            for path in (
                *Path(plan.output_dir).glob("transcript_*.json"),
                Path(plan.output_dir) / "states.json",
            )
            if path.exists()
        ]
        for plan in plans
    }
    stale_episode_artifacts = {
        trial_id: paths
        for trial_id, paths in stale_episode_artifacts.items()
        if paths
    }
    if stale_episode_artifacts:
        raise FileExistsError(
            "full-agent retry would mix deterministic episode artifacts; "
            "preserve this attempt and generate a new experiment/trial identity: "
            + json.dumps(stale_episode_artifacts, sort_keys=True)
        )

    def execute(plan: Any) -> tuple[int, str | None, str | None]:
        result = execute_child_plan(
            plan,
            allow_execution=True,
            capture_output=args.capture_output,
            timeout_s=args.timeout_s,
        )
        return result.returncode, result.stdout, result.stderr

    results, failed = _execute_plans(
        plans,
        journal=journal,
        execute=execute,
        continue_on_error=args.continue_on_error,
    )
    results_path = Path(args.results_output) if args.results_output else (
        plans_path.with_name("full_agent_child_results.json")
    )
    _atomic_json(results_path, results)
    _emit(
        {
            "dry_run": False,
            "planned": len(plans),
            "executed": len(results),
            "failed": failed,
            "results": str(results_path.resolve()),
            "journal": str(journal_path.resolve()),
        }
    )
    return 1 if failed else 0


def _cmd_summarize_full_agent(args: argparse.Namespace) -> int:
    """Join completed full-agent artifacts into one outcome per episode."""
    from rpent.research.handoff.dataset import (
        OutcomeJsonlWriter,
        dataset_fingerprint,
    )
    from rpent.research.handoff.experiments.config import ExecutionLayer
    from rpent.research.handoff.experiments.full_agent_outcomes import (
        load_probe_reset_map,
        summarize_full_agent_trial,
    )
    from rpent.research.handoff.experiments.manifest import load_manifest

    manifest = load_manifest(args.manifest)
    trial_ids = set(args.trial_id or ())
    conditions = set(args.condition or ())
    known_trial_ids = {trial.trial_id for trial in manifest.trials}
    known_conditions = {trial.condition.name for trial in manifest.trials}
    missing_trials = sorted(trial_ids.difference(known_trial_ids))
    missing_conditions = sorted(conditions.difference(known_conditions))
    if missing_trials or missing_conditions:
        raise ValueError(
            f"unknown summary filters: trial_ids={missing_trials}, "
            f"conditions={missing_conditions}"
        )
    selected = tuple(
        trial
        for trial in manifest.trials
        if trial.execution_layer is ExecutionLayer.FULL_AGENT
        and (not trial_ids or trial.trial_id in trial_ids)
        and (not conditions or trial.condition.name in conditions)
    )
    if not selected:
        raise ValueError("no full-agent trials match the requested summary filters")

    probe_resets = load_probe_reset_map(args.runtime_probe or ())
    records = tuple(
        summarize_full_agent_trial(trial, probe_resets=probe_resets)
        for trial in selected
    )
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite episode summaries: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale summary temporary file exists: {temporary}")
    try:
        writer = OutcomeJsonlWriter(temporary, fsync=True)
        for record in records:
            writer.append(record)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _emit(
        {
            "manifest_id": manifest.manifest_id,
            "records": len(records),
            "dataset_fingerprint": dataset_fingerprint(records),
            "output": str(destination.resolve()),
            "trial_ids": [record.identity.trial_id for record in records],
        }
    )
    return 0


def _add_execution_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--execute",
        action="store_true",
        help="Execute persisted child plans (server-heavy).",
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Persist and print plans without executing (default).",
    )
    parser.add_argument(
        "--execution-token",
        help="Exact confirmation token printed in dry-run output.",
    )


def _add_trial_runner_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument("--journal")
    parser.add_argument("--plans-output")
    parser.add_argument("--results-output")
    parser.add_argument("--trial-id", action="append")
    parser.add_argument("--condition", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--retry-cancelled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--capture-output", action="store_true")
    parser.add_argument("--timeout-s", type=float)
    parser.add_argument("--python-executable")
    _add_execution_options(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rpent-handoff",
        description="Configuration-driven RPent controller-handoff research.",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Re-raise command errors with a full traceback.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="Expand a strict trial manifest.")
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--trials-jsonl")
    manifest.set_defaults(handler=_cmd_manifest)

    preflight = subparsers.add_parser(
        "offline-preflight",
        help="Validate configuration/artifacts without starting services.",
    )
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--manifest")
    preflight.add_argument("--output")
    preflight.add_argument("--allow-missing-references", action="store_true")
    preflight.set_defaults(handler=_cmd_offline_preflight)

    probe = subparsers.add_parser(
        "probe-runtime",
        help="Probe already-running env/VLA/SAM3 endpoints.",
    )
    probe.add_argument("--config")
    probe.add_argument("--env-endpoint")
    probe.add_argument("--vla-endpoint")
    probe.add_argument("--sam3-endpoint")
    probe.add_argument("--timeout-s", type=float, default=30.0)
    probe.add_argument("--output")
    probe.add_argument("--vla-observation-npz")
    probe.add_argument(
        "--capture-vla-observation-npz",
        help=(
            "Reset an explicitly isolated env trial and save its deployment VLA "
            "arrays; requires both reset/isolation confirmations."
        ),
    )
    probe.add_argument("--inference-instruction")
    probe.add_argument("--allow-model-inference", action="store_true")
    probe.add_argument("--isolated-model-session-confirmed", action="store_true")
    probe.add_argument("--sam3-image-npy")
    probe.add_argument("--sam3-text-prompt")
    probe.add_argument("--allow-sam3-inference", action="store_true")
    probe.add_argument("--discover-host-gpu", action="store_true")
    probe.add_argument("--chunk-actions-npy")
    probe.add_argument(
        "--allow-destructive-chunk-diagnostic", action="store_true"
    )
    probe.add_argument("--fresh-env-reset-confirmed", action="store_true")
    probe.add_argument("--isolated-env-trial-confirmed", action="store_true")
    probe.add_argument("--hidden-state-diagnostic")
    probe.add_argument("--allow-hidden-state-diagnostic", action="store_true")
    probe.set_defaults(handler=_cmd_probe_runtime)

    gate0 = subparsers.add_parser(
        "collect-gate0",
        help="Plan or execute a process-isolated Gate-0 collection job.",
    )
    gate0.add_argument("--job", required=True)
    gate0.add_argument("--repo-root", default=str(Path.cwd()))
    gate0.add_argument("--plan-output")
    gate0.add_argument("--limit", type=int)
    gate0.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    gate0.add_argument("--timeout-s", type=float)
    gate0.add_argument("--python-executable")
    _add_execution_options(gate0)
    gate0.set_defaults(handler=_cmd_collect_gate0)

    train = subparsers.add_parser("train", help="Train and save a checksummed model artifact.")
    train.add_argument("--outcomes", action="append", required=True)
    train.add_argument("--feature-spec", required=True)
    train.add_argument("--training-config", required=True)
    train.add_argument("--artifact-dir", required=True)
    train.add_argument("--report-output")
    train.add_argument("--assignment-output")
    train.add_argument("--repo-root")
    train.add_argument("--external-runtime-identity")
    train.add_argument("--overwrite", action="store_true")
    train.set_defaults(handler=_cmd_train)

    positive = subparsers.add_parser(
        "build-positive-references",
        help="Build a versioned success-only retrieval/support artifact.",
    )
    positive.add_argument("--outcomes", action="append", required=True)
    positive.add_argument("--target-label", required=True)
    positive.add_argument("--maximum-references", type=int)
    positive.add_argument("--output", required=True)
    positive.add_argument("--overwrite", action="store_true")
    positive.set_defaults(handler=_cmd_positive_references)

    materialize = subparsers.add_parser(
        "materialize-splits",
        help="Materialize exact train/calibration/test JSONL from an assignment.",
    )
    materialize.add_argument("--outcomes", action="append", required=True)
    materialize.add_argument("--assignment", required=True)
    materialize.add_argument("--output-dir", required=True)
    materialize.set_defaults(handler=_cmd_materialize_splits)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trusted artifact on caller-supplied held-out outcomes.",
    )
    evaluate.add_argument("--artifact-dir", required=True)
    evaluate.add_argument("--outcomes", action="append", required=True)
    evaluate.add_argument("--target-label")
    evaluate.add_argument("--threshold", type=float, default=0.5)
    evaluate.add_argument("--calibration-bins", type=int, default=10)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--trust-artifact", action="store_true")
    evaluate.add_argument("--allow-training-data-evaluation", action="store_true")
    evaluate.set_defaults(handler=_cmd_evaluate)

    aggregate = subparsers.add_parser(
        "aggregate",
        help="Aggregate observed OutcomeRecords into paper-oriented artifacts.",
    )
    aggregate.add_argument("--outcomes", action="append", required=True)
    aggregate.add_argument("--output-dir", required=True)
    aggregate.add_argument("--target-label")
    aggregate.add_argument("--bootstrap-iterations", type=int, default=2000)
    aggregate.add_argument("--bootstrap-seed", type=int, default=0)
    aggregate.set_defaults(handler=_cmd_aggregate)

    plot = subparsers.add_parser("plot", help="Plot observed aggregation artifacts.")
    plot.add_argument(
        "--kind",
        required=True,
        choices=("calibration", "success-cost", "regret", "gate0", "ablation"),
    )
    plot.add_argument("--summary")
    plot.add_argument("--rows-csv")
    plot.add_argument("--output", required=True)
    plot.add_argument("--x-key", default="target_relative_x_m")
    plot.add_argument("--y-key", default="target_relative_y_m")
    plot.add_argument("--value-key", default="skill_success")
    plot.add_argument("--factor-key")
    plot.set_defaults(handler=_cmd_plot)

    controlled = subparsers.add_parser(
        "run-controlled",
        help="Plan or execute controller-only trials in isolated children.",
    )
    _add_trial_runner_options(controlled)
    controlled.set_defaults(handler=_cmd_run_controlled)

    full_agent = subparsers.add_parser(
        "run-full-agent",
        help="Plan or execute full RPent/Harness trials in isolated children.",
    )
    _add_trial_runner_options(full_agent)
    full_agent.set_defaults(handler=_cmd_run_full_agent)

    summarize_full_agent = subparsers.add_parser(
        "summarize-full-agent",
        help="Create one checksummed OutcomeRecord per completed full-agent episode.",
    )
    summarize_full_agent.add_argument("--manifest", required=True)
    summarize_full_agent.add_argument(
        "--runtime-probe",
        action="append",
        help="Probe report supplying an observed exact reset identity; repeatable.",
    )
    summarize_full_agent.add_argument("--trial-id", action="append")
    summarize_full_agent.add_argument("--condition", action="append")
    summarize_full_agent.add_argument("--output", required=True)
    summarize_full_agent.set_defaults(handler=_cmd_summarize_full_agent)

    controlled_child = subparsers.add_parser(
        "_controlled-child",
        help=argparse.SUPPRESS,
    )
    controlled_child.add_argument("--manifest", required=True)
    controlled_child.add_argument("--trial-id", required=True)
    controlled_child.set_defaults(handler=_cmd_controlled_child)

    gate0_child = subparsers.add_parser("_gate0-child", help=argparse.SUPPRESS)
    gate0_child.add_argument("--job", required=True)
    gate0_child.add_argument("--limit", type=int)
    gate0_child.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    gate0_child.set_defaults(handler=_cmd_gate0_child)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("rpent-handoff: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.traceback:
            raise
        print(f"rpent-handoff: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
