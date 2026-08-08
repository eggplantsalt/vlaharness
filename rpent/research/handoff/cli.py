"""Command-line orchestration for opt-in controller-handoff research.

All server/GPU execution is dry-run by default.  Commands that can start a
LIBERO, Pi0.5, or SAM3 runtime require both ``--execute`` and the exact
confirmation token printed in their plan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
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


def _immutable_json(path: str | os.PathLike[str], value: Any) -> Path:
    """Write canonical JSON once, accepting only an identical prior value."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canonical = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    if destination.exists():
        if destination.read_text(encoding="utf-8") != canonical:
            raise FileExistsError(
                f"immutable JSON artifact already differs: {destination}"
            )
        return destination
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical)
        stream.flush()
        os.fsync(stream.fileno())
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
        RuntimeProbeArtifact,
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
    requested_observed = tuple(dict.fromkeys(args.require_observed or ()))
    fact_by_name = {fact.name: fact for fact in probe.facts}
    unknown_required = sorted(set(requested_observed).difference(fact_by_name))
    if unknown_required:
        raise ValueError(
            "unknown --require-observed runtime facts: "
            + ", ".join(unknown_required)
        )
    missing_required = [
        name
        for name in requested_observed
        if fact_by_name[name].status is not ProbeStatus.OBSERVED
    ]
    checkpoint_identity_mismatches: list[dict[str, Any]] = []
    if runtime is not None:
        expected_checkpoint_ids = {
            "vla": runtime.pi05_checkpoint_id,
            "sam3": runtime.sam3_checkpoint_id,
        }
        for component, expected_id in expected_checkpoint_ids.items():
            if endpoints[component] is None:
                continue
            if expected_id is None:
                continue
            fact = fact_by_name[f"{component}.model_checkpoint_identity"]
            actual_id = None
            if isinstance(fact.value, Mapping):
                checkpoint_value = fact.value.get("checkpoint")
                if isinstance(checkpoint_value, Mapping):
                    actual_id = checkpoint_value.get("configured_id")
            if actual_id != expected_id:
                checkpoint_identity_mismatches.append(
                    {
                        "component": component,
                        "expected": expected_id,
                        "actual": actual_id,
                    }
                )
    probe_calls_ok = not any(
        fact.status is ProbeStatus.ERROR for fact in probe.facts
    )
    readiness_ok = (
        probe_calls_ok
        and not missing_required
        and not checkpoint_identity_mismatches
    )
    ok = readiness_ok
    artifact = RuntimeProbeArtifact(
        report=probe,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        ok=ok,
        probe_calls_ok=probe_calls_ok,
        readiness_ok=readiness_ok,
        required_observed_facts=tuple(requested_observed),
        required_facts_not_observed=tuple(missing_required),
        checkpoint_identity_mismatches=tuple(checkpoint_identity_mismatches),
        captured_vla_observation_npz=captured_observation_path,
        pending_diagnostics=tuple(
            fact.name for fact in probe.pending_diagnostics
        ),
    )
    report = artifact.model_dump(mode="json", exclude_none=False)
    if args.output:
        _atomic_json(args.output, report)
    _emit(report)
    return 0 if ok else 2


def _gate0_plan(args: argparse.Namespace) -> dict[str, Any]:
    from rpent.research.handoff.experiments.runtime import (
        EXECUTION_CONFIRMATION,
        gate0_plan_id,
        gate0_resume_anchor,
        gate0_runtime_environment,
        load_gate0_job,
    )

    job = load_gate0_job(args.job)
    repo_root = str(Path(args.repo_root).resolve())
    resume_anchor = gate0_resume_anchor(job.output_dir)
    plan_id = gate0_plan_id(
        job,
        repo_root=repo_root,
        limit=args.limit,
        resume=args.resume,
        resume_anchor=resume_anchor,
    )
    command = [
        args.python_executable or sys.executable,
        "-m",
        "rpent.research.handoff",
        "--traceback",
        "_gate0-child",
        "--job",
        str(Path(args.job).resolve()),
        "--plan-id",
        plan_id,
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    command.append("--resume" if args.resume else "--no-resume")
    return {
        "schema_version": "rpent.handoff-gate0-child-plan/v2",
        "plan_id": plan_id,
        "job_configuration_id": job.stable_configuration_id,
        "command": command,
        "cwd": repo_root,
        "output_dir": job.output_dir,
        "resume_anchor": resume_anchor,
        "env_overrides": gate0_runtime_environment(job),
        "execution_confirmation": EXECUTION_CONFIRMATION,
    }


def _cmd_collect_gate0(args: argparse.Namespace) -> int:
    plan = _gate0_plan(args)
    plan_path = args.plan_output or str(
        Path(plan["output_dir"]) / "plans" / f"{plan['plan_id']}.json"
    )
    _immutable_json(plan_path, plan)
    if not args.execute:
        _emit({**plan, "dry_run": True, "plan_path": str(Path(plan_path).resolve())})
        return 0
    _require_server_execution(args)
    environment = os.environ.copy()
    environment.update(plan["env_overrides"])
    completed = subprocess.run(
        plan["command"],
        cwd=plan["cwd"],
        env=environment,
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
    from rpent.research.handoff.dataset import (
        DatasetResearchSink,
        OutcomeDataset,
        dataset_fingerprint,
        scan_decision_jsonl,
    )
    from rpent.research.handoff.experiments.gate0 import (
        Gate0Adapter,
        Gate0Collector,
        Gate0Config,
        Gate0RunIdentity,
    )
    from rpent.research.handoff.experiments.sampling import (
        generate_gate0_samples,
        sample_world_position,
    )
    from rpent.research.handoff.experiments.runtime import (
        SetupJsonlSink,
        gate0_plan_id,
        gate0_resume_anchor,
        gate0_runtime_environment,
        instantiate_gate0_adapter,
        load_gate0_job,
        verify_gate0_job_external_bindings,
    )
    from rpent.research.handoff.experiments.runtime_identity import (
        load_runtime_attestation,
    )
    from rpent.research.handoff.types import ControllerIdentity, SkillIdentity

    job = load_gate0_job(args.job)
    verify_gate0_job_external_bindings(job, repo_root=Path.cwd())
    resume_anchor = gate0_resume_anchor(job.output_dir)
    expected_plan_id = gate0_plan_id(
        job,
        repo_root=Path.cwd(),
        limit=args.limit,
        resume=args.resume,
        resume_anchor=resume_anchor,
    )
    if args.plan_id != expected_plan_id:
        raise ValueError(
            "Gate-0 child plan ID disagrees with the job/current output state: "
            f"expected={expected_plan_id!r}, actual={args.plan_id!r}"
        )
    expected_environment = gate0_runtime_environment(job)
    actual_environment = {
        name: os.environ.get(name) for name in expected_environment
    }
    if actual_environment != expected_environment:
        raise ValueError(
            "Gate-0 child checkpoint environment disagrees with its plan: "
            f"expected={expected_environment!r}, actual={actual_environment!r}"
        )
    if resume_anchor and not args.resume:
        raise FileExistsError(
            "Gate-0 authoritative artifacts already exist; choose --resume "
            "and generate a new plan, or use a new output directory: "
            + ", ".join(sorted(resume_anchor))
        )
    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gate0_config = Gate0Config.model_validate(job.gate0)
    dataset_dir = output_dir / "online"
    outcome_path = dataset_dir / "outcomes.jsonl"
    decision_path = dataset_dir / "decisions.jsonl"
    setup_path = output_dir / "privileged" / "setups.jsonl"
    summary_path = output_dir / "collection_summary.json"
    attempt_path = output_dir / "attempts" / f"{args.plan_id}.json"
    attempt_summary_path = (
        output_dir / "collection_attempts" / f"{args.plan_id}.json"
    )
    expected_samples = {
        f"{job.episode_prefix}-trial-{sample.sample_id}": sample
        for sample in generate_gate0_samples(gate0_config.sampler)
    }
    run_manifest = {
        "schema_version": "rpent.handoff-gate0-run-manifest/v2",
        "configuration_id": job.stable_configuration_id,
        "source_revision": job.source_revision,
        "source_job_path": job.external_bindings.source_job_path,
        "source_job_sha256": job.external_bindings.source_job_sha256,
        "handoff_config_sha256": (
            job.external_bindings.handoff_config_sha256
        ),
        "runtime_probe_sha256": {
            binding.name: binding.sha256
            for binding in job.external_bindings.runtime_probes
        },
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
        _immutable_json(run_manifest_path, run_manifest)
    attempt = {
        "schema_version": "rpent.handoff-gate0-attempt/v1",
        "plan_id": args.plan_id,
        "configuration_id": job.stable_configuration_id,
        "source_revision": job.source_revision,
        "resume": args.resume,
        "limit": args.limit,
        "resume_anchor": dict(sorted(resume_anchor.items())),
        "checkpoint_environment": expected_environment,
    }
    completed_trial_ids: set[str] = set()
    existing_records = ()
    if args.resume and outcome_path.exists():
        # Validate the complete durable shard before constructing an adapter or
        # a writer. A torn tail is evidence requiring inspection, not something
        # a new physical process may silently truncate and continue past.
        existing_records = OutcomeDataset.from_jsonl(
            outcome_path,
            allow_partial_final_line=False,
        ).records
        trial_keys = [record.identity.trial_id for record in existing_records]
        invocation_keys = [
            (record.identity.run_id, record.identity.invocation_id)
            for record in existing_records
        ]
        if len(trial_keys) != len(set(trial_keys)):
            raise ValueError(
                "existing Gate-0 outcomes contain duplicate trial identities"
            )
        if len(invocation_keys) != len(set(invocation_keys)):
            raise ValueError(
                "existing Gate-0 outcomes contain duplicate invocation identities"
            )
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
                or record.identity.candidate_id
                != expected_samples[record.identity.trial_id].candidate_id
                or record.identity.repeat_index
                != expected_samples[record.identity.trial_id].repeat_index
                or record.identity.episode_id
                != (
                    f"{job.episode_prefix}-episode-"
                    f"{expected_samples[record.identity.trial_id].sample_id}"
                )
                or record.identity.invocation_id
                != (
                    f"{job.episode_prefix}-vla-"
                    f"{expected_samples[record.identity.trial_id].sample_id}"
                )
                or record.skill.name != job.skill_name
                or record.skill.semantic_target != job.target_description
                or record.controller.method != job.controller_method
                or record.controller.implementation_version
                != job.controller_implementation_version
                or record.controller.checkpoint_id != job.checkpoint_id
                or record.controller.configuration_id
                != job.stable_configuration_id
                or record.source_revision != job.source_revision
                or record.metadata.get("gate0_configuration_id")
                != job.stable_configuration_id
                or not isinstance(
                    record.metadata.get("execution_plan_id"), str
                )
                or not isinstance(
                    record.metadata.get("runtime_attestation_id"), str
                )
                or not isinstance(
                    record.metadata.get("runtime_attestation_sha256"), str
                )
                or record.metadata.get("gate0_candidate_id")
                != record.identity.candidate_id
                or record.metadata.get("gate0_repeat_index")
                != record.identity.repeat_index
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
    decision_scan = None
    if args.resume and decision_path.exists():
        decision_scan = scan_decision_jsonl(
            decision_path,
            allow_partial_final_line=False,
        )
    attempts = {
        path.stem: _strict_json_object(path)
        for path in sorted((output_dir / "attempts").glob("*.json"))
    }
    attempt_summaries = {
        path.stem: _strict_json_object(path)
        for path in sorted((output_dir / "collection_attempts").glob("*.json"))
    }
    prior_attempt_ids = set(attempts).difference({args.plan_id})
    if set(attempt_summaries) != prior_attempt_ids:
        raise ValueError(
            "Gate-0 resume found an unsealed attempt or orphan attempt summary; "
            "inspect the run and choose a new output directory"
        )
    if not prior_attempt_ids:
        orphan_shards = [
            path
            for path in (outcome_path, decision_path, setup_path)
            if path.exists()
        ]
        telemetry_root = output_dir / "telemetry"
        if telemetry_root.is_dir():
            orphan_shards.extend(
                path for path in telemetry_root.rglob("*") if path.is_file()
            )
        if orphan_shards:
            raise ValueError(
                "Gate-0 resume found shards with no sealed attempt provenance: "
                + ", ".join(str(path) for path in orphan_shards[:10])
            )
    runtime_identity_paths = {
        path.stem: path
        for path in sorted((output_dir / "runtime_identity").glob("*.json"))
    }
    expected_runtime_attempts: set[str] = set()
    outcome_owner: dict[str, str] = {}
    for plan_id in sorted(prior_attempt_ids):
        marker = attempts[plan_id]
        summary = attempt_summaries[plan_id]
        marker_anchor = marker.get("resume_anchor")
        if (
            marker.get("schema_version") != "rpent.handoff-gate0-attempt/v1"
            or marker.get("plan_id") != plan_id
            or marker.get("configuration_id") != job.stable_configuration_id
            or marker.get("source_revision") != job.source_revision
            or marker.get("checkpoint_environment") != expected_environment
            or not isinstance(marker_anchor, dict)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in marker_anchor.items()
            )
            or not isinstance(marker.get("resume"), bool)
            or (
                marker.get("limit") is not None
                and (
                    isinstance(marker.get("limit"), bool)
                    or not isinstance(marker.get("limit"), int)
                    or marker.get("limit") < 0
                )
            )
        ):
            raise ValueError(f"Gate-0 attempt marker is invalid: {plan_id}")
        if gate0_plan_id(
            job,
            repo_root=Path.cwd(),
            limit=marker["limit"],
            resume=marker["resume"],
            resume_anchor=marker_anchor,
        ) != plan_id:
            raise ValueError(
                f"Gate-0 attempt plan ID does not bind its marker: {plan_id}"
            )
        if (
            summary.get("schema_version")
            != "rpent.handoff-gate0-attempt-summary/v1"
            or summary.get("plan_id") != plan_id
            or summary.get("configuration_id") != job.stable_configuration_id
            or summary.get("source_revision") != job.source_revision
        ):
            raise ValueError(f"Gate-0 attempt summary is invalid: {plan_id}")
        record_ids = summary.get("record_ids")
        if not isinstance(record_ids, list) or any(
            not isinstance(record_id, str) or not record_id
            for record_id in record_ids
        ):
            raise ValueError(f"Gate-0 attempt record IDs are invalid: {plan_id}")
        if (
            summary.get("collected") != len(record_ids)
            or not isinstance(summary.get("resumed_completed"), int)
            or not isinstance(summary.get("completed_after"), int)
            or summary.get("completed_after")
            != summary.get("resumed_completed") + len(record_ids)
            or not isinstance(summary.get("dataset_fingerprint_before"), str)
            or not isinstance(summary.get("dataset_fingerprint_after"), str)
        ):
            raise ValueError(
                f"Gate-0 attempt counts/fingerprints are invalid: {plan_id}"
            )
        for record_id in record_ids:
            if record_id in outcome_owner:
                raise ValueError(
                    "Gate-0 attempt summaries claim the same outcome twice: "
                    f"{record_id}"
                )
            outcome_owner[record_id] = plan_id
        attestation_id = summary.get("runtime_attestation_id")
        attestation_sha256 = summary.get("runtime_attestation_sha256")
        if record_ids and (
            not isinstance(attestation_id, str)
            or not isinstance(attestation_sha256, str)
        ):
            raise ValueError(
                f"Gate-0 outcome-bearing attempt lacks runtime identity: {plan_id}"
            )
        if attestation_id is not None or attestation_sha256 is not None:
            if not isinstance(attestation_id, str) or not isinstance(
                attestation_sha256, str
            ):
                raise ValueError(
                    f"Gate-0 attempt runtime identity is only partially bound: {plan_id}"
                )
            expected_runtime_attempts.add(plan_id)
            attestation_path = runtime_identity_paths.get(plan_id)
            if attestation_path is None:
                raise ValueError(
                    f"Gate-0 runtime attestation is missing: {plan_id}"
                )
            actual_sha256 = hashlib.sha256(
                attestation_path.read_bytes()
            ).hexdigest()
            attestation = load_runtime_attestation(attestation_path)
            if (
                actual_sha256 != attestation_sha256
                or attestation.attestation_id != attestation_id
                or attestation.plan_id != plan_id
                or attestation.source_revision != job.source_revision
                or tuple(
                    item.observed_checkpoint_id
                    for item in attestation.observations
                )
                != (
                    expected_environment["RPENT_PI05_CHECKPOINT_ID"],
                    expected_environment["RPENT_SAM3_CHECKPOINT_ID"],
                )
            ):
                raise ValueError(
                    f"Gate-0 runtime attestation is invalid: {plan_id}"
                )
    if set(runtime_identity_paths) != expected_runtime_attempts:
        raise ValueError(
            "Gate-0 resume found an orphan runtime identity sidecar"
        )
    existing_by_id = {record.record_id: record for record in existing_records}
    if set(outcome_owner) != set(existing_by_id):
        raise ValueError(
            "Gate-0 attempt summaries do not partition the durable outcomes"
        )
    cumulative_records: list[Any] = []
    expected_before_count = 0
    chronological_attempts = sorted(
        prior_attempt_ids,
        key=lambda plan_id: attempt_summaries[plan_id]["resumed_completed"],
    )
    if len(
        {
            attempt_summaries[plan_id]["resumed_completed"]
            for plan_id in chronological_attempts
        }
    ) != len(chronological_attempts):
        raise ValueError("Gate-0 sealed attempts have an ambiguous resume order")
    for plan_id in chronological_attempts:
        summary = attempt_summaries[plan_id]
        if summary["resumed_completed"] != expected_before_count:
            raise ValueError("Gate-0 sealed attempt counts do not form a chain")
        if summary["dataset_fingerprint_before"] != dataset_fingerprint(
            cumulative_records
        ):
            raise ValueError(
                "Gate-0 sealed attempt input fingerprints do not form a chain"
            )
        try:
            cumulative_records.extend(
                existing_by_id[record_id]
                for record_id in summary["record_ids"]
            )
        except KeyError as exc:
            raise ValueError(
                f"Gate-0 attempt references a missing outcome: {exc.args[0]}"
            ) from exc
        expected_before_count = len(cumulative_records)
        if (
            summary["completed_after"] != expected_before_count
            or summary["dataset_fingerprint_after"]
            != dataset_fingerprint(cumulative_records)
        ):
            raise ValueError(
                "Gate-0 sealed attempt output fingerprints do not form a chain"
            )
    for record_id, plan_id in outcome_owner.items():
        record = existing_by_id[record_id]
        summary = attempt_summaries[plan_id]
        if (
            record.metadata.get("execution_plan_id") != plan_id
            or record.metadata.get("runtime_attestation_id")
            != summary.get("runtime_attestation_id")
            or record.metadata.get("runtime_attestation_sha256")
            != summary.get("runtime_attestation_sha256")
        ):
            raise ValueError(
                f"Gate-0 outcome provenance disagrees with its attempt: {record_id}"
            )
    decision_count = (
        len(decision_scan.envelopes) if decision_scan is not None else 0
    )
    handoff_outcome_count = sum(
        1 for record in existing_records if record.handoff_occurred
    )
    if decision_count != handoff_outcome_count:
        raise ValueError(
            "Gate-0 decision and handoff-outcome shards are not a sealed pair"
        )
    if decision_scan is not None and any(
        envelope.payload.action.value != "handoff_now"
        for envelope in decision_scan.envelopes
    ):
        raise ValueError("Gate-0 direct collection contains a non-handoff decision")
    durable_setup = SetupJsonlSink(setup_path)
    if args.resume and durable_setup.records:
        def setup_is_valid(record: Any) -> bool:
            sample = expected_samples.get(record.identity.trial_id)
            if sample is None or record.requested_candidate is None:
                return False
            expected_episode = (
                f"{job.episode_prefix}-episode-{sample.sample_id}"
            )
            expected_invocation = (
                f"{job.episode_prefix}-vla-{sample.sample_id}"
            )
            identity_ok = (
                record.identity.run_id == job.run_id
                and record.identity.episode_id == expected_episode
                and record.identity.invocation_id == expected_invocation
                and record.identity.suite == job.suite
                and str(record.identity.task_id) == str(job.task_id)
                and record.identity.seed == job.seed
                and record.identity.reset_id is not None
                and record.identity.repeat_index == sample.repeat_index
                and record.identity.candidate_id == sample.candidate_id
            )
            candidate = record.requested_candidate
            axis = gate0_config.sampler.approach_axis_world
            expected_relative = sample_world_position(
                sample,
                target_position_m=(0.0, 0.0, 0.0),
                approach_axis_world=axis,
            )
            target_values = [
                value.values
                for value in record.values
                if value.name == "target_position_m"
            ]
            if len(target_values) != 1 or len(target_values[0]) != 3:
                return False
            expected_world = tuple(
                float(target + relative)
                for target, relative in zip(
                    target_values[0], expected_relative, strict=True
                )
            )
            close = lambda left, right: math.isclose(
                float(left), float(right), rel_tol=0.0, abs_tol=1e-12
            )
            geometry_ok = (
                candidate.candidate_id == sample.candidate_id
                and candidate.kind == "perturbation"
                and candidate.target_relative_position_m is not None
                and all(
                    close(left, right)
                    for left, right in zip(
                        candidate.target_relative_position_m,
                        expected_relative,
                        strict=True,
                    )
                )
                and all(
                    close(left, right)
                    for left, right in zip(
                        candidate.eef_position_m,
                        expected_world,
                        strict=True,
                    )
                )
                and candidate.wrist_yaw_rad is not None
                and candidate.wrist_pitch_rad is not None
                and candidate.requested_standoff_m is not None
                and close(candidate.wrist_yaw_rad, sample.wrist_yaw_rad)
                and close(candidate.wrist_pitch_rad, sample.wrist_pitch_rad)
                and close(candidate.requested_standoff_m, sample.standoff_m)
            )
            return identity_ok and geometry_ok

        invalid_setups = [
            record.record_id
            for record in durable_setup.records
            if not setup_is_valid(record)
        ]
        if invalid_setups:
            raise ValueError(
                "existing Gate-0 setup records disagree with the immutable "
                f"job/sample geometry: {invalid_setups[:10]}"
            )
        referenced_setup_ids = [
            record.setup_record_id
            for record in existing_records
            if record.setup_record_id is not None
        ]
        if len(referenced_setup_ids) != len(set(referenced_setup_ids)):
            raise ValueError(
                "existing Gate-0 outcomes reuse a privileged setup record"
            )
        durable_setup_ids = {record.record_id for record in durable_setup.records}
        if durable_setup_ids != set(referenced_setup_ids):
            raise ValueError(
                "existing Gate-0 setups and outcome setup_record_id bindings "
                "differ; an orphan/stale setup requires inspection and a new run"
            )
        outcome_by_setup = {
            record.setup_record_id: record
            for record in existing_records
            if record.setup_record_id is not None
        }
        mismatched_setup_outcomes = [
            setup.record_id
            for setup in durable_setup.records
            if setup.identity != outcome_by_setup[setup.record_id].identity
        ]
        if mismatched_setup_outcomes:
            raise ValueError(
                "existing Gate-0 setup/outcome identities disagree: "
                f"{mismatched_setup_outcomes[:10]}"
            )
    # This equality is intentionally checked even when either side is empty.
    # An outcome without its setup, or a setup whose outcome append never
    # completed, is an interrupted physical trial and must not be resumed.
    referenced_setup_ids = [
        record.setup_record_id
        for record in existing_records
        if record.setup_record_id is not None
    ]
    if len(referenced_setup_ids) != len(set(referenced_setup_ids)):
        raise ValueError("existing Gate-0 outcomes reuse a privileged setup record")
    durable_setup_ids = {record.record_id for record in durable_setup.records}
    if durable_setup_ids != set(referenced_setup_ids):
        raise ValueError(
            "existing Gate-0 setup/outcome shards are not a sealed pair; "
            "an orphan or missing setup requires a new run"
        )
    pending_trial_ids = set(expected_samples).difference(completed_trial_ids)
    if not pending_trial_ids:
        if not summary_path.is_file():
            raise ValueError(
                "Gate-0 outcomes are complete but the immutable final summary "
                "is missing; this is an unsealed prior attempt"
            )
        final_summary = _strict_json_object(summary_path)
        if (
            final_summary.get("schema_version")
            != "rpent.handoff-gate0-collection-summary/v2"
            or final_summary.get("configuration_id")
            != job.stable_configuration_id
            or final_summary.get("source_revision") != job.source_revision
            or final_summary.get("completed") != len(expected_samples)
            or final_summary.get("dataset_fingerprint")
            != dataset_fingerprint(existing_records)
            or final_summary.get("attempt_plan_ids")
            != sorted(prior_attempt_ids)
            or final_summary.get("record_ids")
            != [record.record_id for record in existing_records]
            or final_summary.get("run_manifest")
            != str(run_manifest_path.resolve())
        ):
            raise ValueError("Gate-0 final summary disagrees with durable outcomes")
        _emit({**final_summary, "summary_path": str(summary_path.resolve())})
        return 0
    if summary_path.exists():
        raise ValueError(
            "Gate-0 final summary exists while expected trials are missing"
        )
    if args.limit == 0:
        _emit(
            {
                "schema_version": "rpent.handoff-gate0-noop/v1",
                "plan_id": args.plan_id,
                "configuration_id": job.stable_configuration_id,
                "pending": len(pending_trial_ids),
            }
        )
        return 0

    _immutable_json(attempt_path, attempt)
    execution_job = job.model_copy(
        update={
            "metadata": {
                **job.metadata,
                "execution_plan_id": args.plan_id,
            }
        }
    )
    bundle = instantiate_gate0_adapter(
        execution_job,
        gate0_config=gate0_config,
        output_dir=output_dir,
    )
    try:
        if not isinstance(bundle.adapter, Gate0Adapter):
            raise TypeError(
                "configured Gate-0 adapter does not implement reset/current EEF/"
                "governor adapter methods"
            )
        if (
            bundle.runtime_attestation is None
            or not isinstance(bundle.runtime_attestation_sha256, str)
            or bundle.runtime_attestation.plan_id != args.plan_id
            or bundle.runtime_attestation.source_revision != job.source_revision
            or tuple(
                item.observed_checkpoint_id
                for item in bundle.runtime_attestation.observations
            )
            != (
                expected_environment["RPENT_PI05_CHECKPOINT_ID"],
                expected_environment["RPENT_SAM3_CHECKPOINT_ID"],
            )
        ):
            raise RuntimeError(
                "Gate-0 adapter did not return the required live runtime attestation"
            )
        outcome_sink = DatasetResearchSink(dataset_dir, fsync=True)
        setup_sink = (
            _CompositeSetupSink((durable_setup, bundle.setup_sink))
            if bundle.setup_sink is not None
            else durable_setup
        )
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
                configuration_id=job.stable_configuration_id,
                execution_plan_id=args.plan_id,
                runtime_attestation_id=(
                    bundle.runtime_attestation.attestation_id
                ),
                runtime_attestation_sha256=(
                    bundle.runtime_attestation_sha256
                ),
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

    all_records = OutcomeDataset.from_jsonl(
        outcome_path,
        allow_partial_final_line=False,
    ).records
    all_decisions = scan_decision_jsonl(
        decision_path,
        allow_partial_final_line=False,
    )
    if len(all_decisions.envelopes) != sum(
        1 for record in all_records if record.handoff_occurred
    ):
        raise RuntimeError(
            "Gate-0 collection ended with unpaired decision/outcome shards"
        )
    final_setup_ids = {record.record_id for record in durable_setup.records}
    final_referenced_setup_ids = {
        record.setup_record_id
        for record in all_records
        if record.setup_record_id is not None
    }
    if final_setup_ids != final_referenced_setup_ids:
        raise RuntimeError("Gate-0 collection ended with unpaired setup/outcome shards")
    attempt_summary = {
        "schema_version": "rpent.handoff-gate0-attempt-summary/v1",
        "plan_id": args.plan_id,
        "configuration_id": job.stable_configuration_id,
        "source_revision": job.source_revision,
        "collected": len(outcomes),
        "resumed_completed": len(completed_trial_ids),
        "completed_after": len(all_records),
        "dataset_fingerprint_before": dataset_fingerprint(existing_records),
        "dataset_fingerprint_after": dataset_fingerprint(all_records),
        "runtime_attestation_id": bundle.runtime_attestation.attestation_id,
        "runtime_attestation_sha256": bundle.runtime_attestation_sha256,
        "outcome_jsonl": str(outcome_sink.outcome_path.resolve()),
        "decision_jsonl": str(outcome_sink.decision_path.resolve()),
        "setup_jsonl": str(durable_setup.path.resolve()),
        "run_manifest": str(run_manifest_path.resolve()),
        "record_ids": [record.record_id for record in outcomes],
    }
    _immutable_json(attempt_summary_path, attempt_summary)
    final_summary_path: str | None = None
    if len(all_records) == len(expected_samples):
        final_summary = {
            "schema_version": "rpent.handoff-gate0-collection-summary/v2",
            "configuration_id": job.stable_configuration_id,
            "source_revision": job.source_revision,
            "completed": len(all_records),
            "expected": len(expected_samples),
            "dataset_fingerprint": dataset_fingerprint(all_records),
            "run_manifest": str(run_manifest_path.resolve()),
            "attempt_plan_ids": sorted((*prior_attempt_ids, args.plan_id)),
            "record_ids": [record.record_id for record in all_records],
        }
        _immutable_json(summary_path, final_summary)
        final_summary_path = str(summary_path.resolve())
    _emit(
        {
            **attempt_summary,
            "attempt_summary_path": str(attempt_summary_path.resolve()),
            "final_summary_path": final_summary_path,
        }
    )
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
    source_revision = None
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
        from rpent.research.handoff.experiments.manifest import (
            compute_source_revision,
        )

        source_revision = compute_source_revision(root)
    try:
        package_version = importlib.metadata.version("rpent")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    runtime_path: str | None = None
    runtime_sha256: str | None = None
    if external_runtime_identity is not None:
        candidate = Path(external_runtime_identity).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"external runtime identity artifact not found: {candidate}"
            )
        from rpent.research.handoff.experiments.probes import RuntimeProbeArtifact

        RuntimeProbeArtifact.model_validate(_strict_json_object(candidate))
        runtime_path = str(candidate)
        runtime_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    return SourceIdentity(
        git_revision=revision,
        dirty=dirty,
        package_version=package_version,
        source_revision=source_revision,
        external_runtime_identity=runtime_path,
        external_runtime_identity_sha256=runtime_sha256,
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
    from rpent.research.handoff.splits import SplitAssignment, SplitName

    assignment = SplitAssignment.model_validate(
        _strict_json_object(args.split_assignment)
    )
    expected_train_ids = tuple(
        entry.record_id
        for entry in assignment.entries
        if entry.split is SplitName.TRAIN
    )
    actual_ids = tuple(sorted(record.record_id for record in dataset.records))
    if actual_ids != expected_train_ids:
        raise ValueError(
            "positive-reference input must be exactly the train partition from "
            "the supplied split assignment"
        )
    artifact = build_positive_reference_artifact(
        dataset,
        target=args.target_label,
        source_dataset_fingerprint=assignment.dataset_fingerprint,
        split_assignment_fingerprint=assignment.fingerprint,
        maximum_references=args.maximum_references,
    )
    path = write_positive_reference_artifact(artifact, destination)
    _emit(
        {
            "artifact_id": artifact.artifact_id,
            "dataset_fingerprint": artifact.dataset_fingerprint,
            "source_dataset_fingerprint": artifact.source_dataset_fingerprint,
            "split_assignment_fingerprint": artifact.split_assignment_fingerprint,
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


def _cmd_materialize_oracle(args: argparse.Namespace) -> int:
    """Annotate Gate-0 landscapes and optional matched policy choices."""
    from rpent.research.handoff.dataset import (
        OutcomeJsonlWriter,
        dataset_fingerprint,
    )
    from rpent.research.handoff.evaluation.oracle import (
        OracleCostConfig,
        annotate_matched_oracle_costs,
    )

    dataset = _load_outcome_dataset(args.outcomes)
    policy_dataset = (
        _load_outcome_dataset(args.policy_outcomes)
        if args.policy_outcomes
        else None
    )
    if (policy_dataset is None) != (args.policy_output is None):
        raise ValueError(
            "--policy-outcomes and --policy-output must be supplied together"
        )
    config = OracleCostConfig.model_validate(_strict_json_object(args.config))
    result = annotate_matched_oracle_costs(
        dataset.records,
        config,
        policy_records=(policy_dataset.records if policy_dataset else ()),
    )
    if policy_dataset is not None and result.annotated_policy_records == 0:
        raise ValueError(
            "--policy-outcomes was supplied, but no eligible policy record "
            "matched a Gate-0 candidate landscape"
        )
    destination = Path(args.output)
    policy_destination = Path(args.policy_output) if args.policy_output else None
    report_path = Path(args.report_output) if args.report_output else (
        destination.with_suffix(destination.suffix + ".report.json")
    )
    occupied = [
        path
        for path in (destination, policy_destination, report_path)
        if path is not None and path.exists()
    ]
    if occupied:
        raise FileExistsError(
            "oracle outputs already exist: "
            + ", ".join(str(path) for path in occupied)
        )
    staged_outputs: list[tuple[Path, Path]] = []

    def stage_records(path: Path, records: Sequence[Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        if temporary.exists():
            raise FileExistsError(f"stale oracle temporary exists: {temporary}")
        try:
            writer = OutcomeJsonlWriter(temporary, fsync=True)
            for record in records:
                writer.append(record)
            staged_outputs.append((temporary, path))
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    report = {
        "schema_version": "rpent.handoff-oracle-materialization/v1",
        "oracle_cost_configuration_id": result.configuration_id,
        "oracle_cost_config": config.model_dump(mode="json", exclude_none=False),
        "source_dataset_fingerprint": dataset.fingerprint,
        "output_dataset_fingerprint": dataset_fingerprint(result.records),
        "records": len(result.records),
        "matched_groups": result.matched_groups,
        "annotated_records": result.annotated_records,
        "annotated_policy_records": result.annotated_policy_records,
        "eligible_unmatched_records": result.eligible_unmatched_records,
        "eligible_unmatched_policy_records": (
            result.eligible_unmatched_policy_records
        ),
        "ineligible_records": result.ineligible_records,
        "ineligible_policy_records": result.ineligible_policy_records,
        "policy_source_dataset_fingerprint": (
            policy_dataset.fingerprint if policy_dataset else None
        ),
        "policy_output_dataset_fingerprint": (
            dataset_fingerprint(result.policy_records)
            if result.policy_records
            else None
        ),
        "policy_output": (
            str(policy_destination.resolve())
            if policy_destination is not None
            else None
        ),
        "output": str(destination.resolve()),
    }
    report_temporary = report_path.with_name(f"{report_path.name}.tmp")
    try:
        stage_records(destination, result.records)
        if policy_destination is not None:
            stage_records(policy_destination, result.policy_records)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_temporary.exists():
            raise FileExistsError(
                f"stale oracle report temporary exists: {report_temporary}"
            )
        staged_outputs.append((report_temporary, report_path))
    except BaseException:
        for temporary, _final in staged_outputs:
            temporary.unlink(missing_ok=True)
        raise
    created_outputs: list[Path] = []
    try:
        report_temporary.write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        with report_temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        for temporary, final in staged_outputs:
            os.replace(temporary, final)
            created_outputs.append(final)
    except BaseException:
        for temporary, _final in staged_outputs:
            temporary.unlink(missing_ok=True)
        for final in created_outputs:
            final.unlink(missing_ok=True)
        raise
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
        filters: dict[str, str] = {}
        for expression in args.where or ():
            key, separator, value = expression.partition("=")
            if not separator or not key or not value:
                raise ValueError("--where must use non-empty KEY=VALUE syntax")
            if key in filters and filters[key] != value:
                raise ValueError(f"contradictory --where filters for {key!r}")
            filters[key] = value
        if filters:
            rows = [
                row
                for row in rows
                if all(
                    str(row.get(key)) == value
                    for key, value in filters.items()
                )
            ]
            if not rows:
                raise ValueError(
                    f"--where filters selected no plot rows: {filters}"
                )
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

    summary = run_controlled_trial(args.manifest, args.trial_id, args.plan_id)
    _emit(summary)
    return 130 if summary.get("cancelled") else 0


def _cmd_full_agent_child(args: argparse.Namespace) -> int:
    from rpent.research.handoff.experiments.full_agent import run_full_agent_trial

    return run_full_agent_trial(args.manifest, args.trial_id, args.plan_id)


def _cmd_run_full_agent(args: argparse.Namespace) -> int:
    from rpent.research.handoff.experiments.full_agent import (
        execute_child_plan,
        plan_full_agent_trials,
        write_child_plans,
    )
    _require_server_execution(args)
    _manifest, trials, journal, journal_path = _selected_trials(
        args,
        layer="full_agent",
    )
    if not trials:
        _emit({"dry_run": not args.execute, "trials": 0, "reason": "no selected pending trials"})
        return 0
    plans = plan_full_agent_trials(
        trials,
        manifest_path=args.manifest,
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
                Path(plan.output_dir) / "reset_identity.json",
                Path(plan.output_dir) / "completion.json",
                Path(plan.output_dir) / "handoff" / "outcomes.jsonl",
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
    probe.add_argument(
        "--require-observed",
        action="append",
        help=(
            "Require this named fact to be observed for readiness (repeatable); "
            "without it, exit status only reflects probe errors."
        ),
    )
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
    positive.add_argument("--split-assignment", required=True)
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

    oracle = subparsers.add_parser(
        "materialize-oracle",
        help=(
            "Annotate Gate-0 landscapes and optional matched controlled-policy "
            "choices with post-hoc oracle costs."
        ),
    )
    oracle.add_argument("--outcomes", action="append", required=True)
    oracle.add_argument("--policy-outcomes", action="append")
    oracle.add_argument("--config", required=True)
    oracle.add_argument("--output", required=True)
    oracle.add_argument("--policy-output")
    oracle.add_argument("--report-output")
    oracle.set_defaults(handler=_cmd_materialize_oracle)

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
    plot.add_argument(
        "--where",
        action="append",
        help="Filter tidy CSV rows by exact KEY=VALUE before plotting (repeatable).",
    )
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
    controlled_child.add_argument("--plan-id", required=True)
    controlled_child.set_defaults(handler=_cmd_controlled_child)

    full_agent_child = subparsers.add_parser(
        "_full-agent-child", help=argparse.SUPPRESS
    )
    full_agent_child.add_argument("--manifest", required=True)
    full_agent_child.add_argument("--trial-id", required=True)
    full_agent_child.add_argument("--plan-id", required=True)
    full_agent_child.set_defaults(handler=_cmd_full_agent_child)

    gate0_child = subparsers.add_parser("_gate0-child", help=argparse.SUPPRESS)
    gate0_child.add_argument("--job", required=True)
    gate0_child.add_argument("--plan-id", required=True)
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
