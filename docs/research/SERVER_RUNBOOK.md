# Linux GPU Server Runbook

This is the first runtime-verification path for the Phase-A implementation.
The code is implementation-complete but has not been executed against Linux,
CUDA, MuJoCo, LIBERO, Pi0.5, or SAM3 in this worktree. Stop on every failed
check. `unavailable`, `requires_diagnostic`, an identity mismatch, and a null
required field are not passes.

Run all commands from the RPent repository root. Use a new output root. Never
reuse a trial/run ID after changing code, configuration, a model/reference
artifact, checkpoint identity, target, sampler, or planner setting.

## 0. Install, identify, and pin the runtime

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[full,handoff]"
liberopro-download-assets --skip-existing

export PI05_CHECKPOINT_PATH=<absolute-pi05-checkpoint>
export SAM3_CHECKPOINT_PATH=<absolute-sam3-checkpoint-or-file>
export LIBERO_TYPE=pro
export RPENT_HANDOFF_OUTPUT_ROOT=<absolute-new-output-root>
mkdir -p "$RPENT_HANDOFF_OUTPUT_ROOT"

# Future artifact paths during structural preflight; immutable content
# bindings after the probes below succeed.
export RPENT_RUNTIME_PROBE_READONLY="$RPENT_HANDOFF_OUTPUT_ROOT/runtime_probe_readonly.json"
export RPENT_RUNTIME_PROBE_INFERENCE="$RPENT_HANDOFF_OUTPUT_ROOT/runtime_probe_inference.json"
export RPENT_RUNTIME_PROBE_SAM3="$RPENT_HANDOFF_OUTPUT_ROOT/runtime_probe_sam3.json"

# Clean upstream commit used only for parity. This is distinct from the
# composite dirty-worktree identity in RPENT_SOURCE_REVISION.
export RPENT_UPSTREAM_BASE_REF=<exact-clean-upstream-commit>
```

Compute content identities. This hashes every byte and relative filename; it can
take time for a directory checkpoint, but avoids treating a mutable path or
mtime as a scientific identity.

```bash
tree_sha256 () {
  python - "$1" <<'PY'
import hashlib, sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
if not root.exists():
    raise SystemExit(f"missing checkpoint: {root}")
digest = hashlib.sha256()
paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
for path in paths:
    relative = "." if root.is_file() else path.relative_to(root).as_posix()
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
exported = digest.hexdigest()
print(exported)
PY
}

export RPENT_PI05_CHECKPOINT_ID="sha256-tree:$(tree_sha256 "$PI05_CHECKPOINT_PATH")"
export RPENT_SAM3_CHECKPOINT_ID="sha256-tree:$(tree_sha256 "$SAM3_CHECKPOINT_PATH")"
```

Bind the exact repository state, including dirty and untracked content. A clean
commit hash alone is insufficient for a dirty research tree.

```bash
export RPENT_SOURCE_REVISION="$(python - <<'PY'
import hashlib, subprocess
from pathlib import Path

revision = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
digest = hashlib.sha256(subprocess.check_output(["git", "diff", "--binary", "HEAD"]))
untracked = subprocess.check_output(
    ["git", "ls-files", "--others", "--exclude-standard", "-z"]
).decode().split("\0")
for name in sorted(item for item in untracked if item):
    digest.update(name.encode("utf-8")); digest.update(b"\0")
    digest.update(Path(name).read_bytes())
print(f"git:{revision};worktree-sha256:{digest.hexdigest()}")
PY
)"
```

Set every matrix identity explicitly. No planner or provider default is allowed.

```bash
export RPENT_HANDOFF_EXPERIMENT_ID=<path-safe-experiment-id>
export RPENT_HANDOFF_REPEATS=3
export RPENT_HANDOFF_SUITE=<libero-suite>
export RPENT_HANDOFF_TASK=<integer-task-id>
export RPENT_HANDOFF_TARGET_ID=<path-safe-target-id>
export RPENT_HANDOFF_TARGET_DESCRIPTION='<target description>'
export RPENT_HANDOFF_SKILL_PROMPT='<frozen Pi0.5 instruction>'
export RPENT_PLANNER_MODEL='<provider-prefixed-model-id>'
export RPENT_PLANNER_BASE_URL='<explicit-api-base-url>'

export RPENT_HANDOFF_MODEL_FULL="$RPENT_HANDOFF_OUTPUT_ROOT/models/full"
export RPENT_HANDOFF_MODEL_ABSOLUTE="$RPENT_HANDOFF_OUTPUT_ROOT/models/absolute"
export RPENT_HANDOFF_MODEL_RELATIVE="$RPENT_HANDOFF_OUTPUT_ROOT/models/relative"
export RPENT_HANDOFF_MODEL_RELATIVE_VISUAL="$RPENT_HANDOFF_OUTPUT_ROOT/models/relative_visual"
export RPENT_HANDOFF_POSITIVE_REFERENCES="$RPENT_HANDOFF_OUTPUT_ROOT/models/positive_references.json"
```

Before data exists, only a structural preflight is meaningful. Do not create an
executable manifest yet: manifest generation now requires and content-binds all
declared model and positive-reference artifacts.

```bash
rpent-handoff offline-preflight \
  --config configs/research/handoff/matrix/main_and_ablations.json \
  --allow-missing-references \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/preflight_structure_only.json"
```

## 1. Probe disposable runtime services

Start one disposable task cell in three supervised terminals. These processes
must not be reused as a measured environment episode.

```bash
# Terminal A
python -m robots.libero.env_server --transport http --host 127.0.0.1 \
  --port 8112 --suite "$RPENT_HANDOFF_SUITE" --task "$RPENT_HANDOFF_TASK" \
  --seed 0 --max-episode-steps 10000 --cuda-device 0

# Terminal B
python -m robots.libero.vla_server --transport http --host 127.0.0.1 \
  --port 8113 --model-path "$PI05_CHECKPOINT_PATH" \
  --checkpoint-id "$RPENT_PI05_CHECKPOINT_ID" --cuda-device 0

# Terminal C
python -m robots.libero.sam3_server --transport http --host 127.0.0.1 \
  --port 8114 --checkpoint-id "$RPENT_SAM3_CHECKPOINT_ID" --cuda-device 0
```

Run the read-only readiness profile. The repeated requirements make a pending
fact fail the command instead of merely appearing in a report.

```bash
rpent-handoff probe-runtime \
  --config configs/research/handoff/matrix/main_and_ablations.json \
  --env-endpoint http://127.0.0.1:8112 \
  --vla-endpoint http://127.0.0.1:8113 \
  --sam3-endpoint http://127.0.0.1:8114 \
  --discover-host-gpu \
  --require-observed host.versions \
  --require-observed host.cuda_gpu \
  --require-observed env.endpoint_health \
  --require-observed env.versions \
  --require-observed env.observation_keys \
  --require-observed env.state_vector_shape \
  --require-observed env.camera_views \
  --require-observed env.controller_environment_config \
  --require-observed env.reset_identity \
  --require-observed vla.endpoint_health \
  --require-observed vla.versions \
  --require-observed vla.configured_action_shape \
  --require-observed vla.controller_config \
  --require-observed vla.model_checkpoint_identity \
  --require-observed sam3.endpoint_health \
  --require-observed sam3.versions \
  --require-observed sam3.current_observation_contract \
  --require-observed sam3.model_checkpoint_identity \
  --output "$RPENT_RUNTIME_PROBE_READONLY"
```

Then reset an explicitly disposable env, capture only deployment VLA arrays,
and run one non-executed Pi0.5 inference. No predicted action is sent to LIBERO.

```bash
rpent-handoff probe-runtime \
  --config configs/research/handoff/matrix/main_and_ablations.json \
  --env-endpoint http://127.0.0.1:8112 \
  --vla-endpoint http://127.0.0.1:8113 \
  --sam3-endpoint http://127.0.0.1:8114 \
  --capture-vla-observation-npz "$RPENT_HANDOFF_OUTPUT_ROOT/probe_observation.npz" \
  --inference-instruction "$RPENT_HANDOFF_SKILL_PROMPT" \
  --allow-model-inference --isolated-model-session-confirmed \
  --fresh-env-reset-confirmed --isolated-env-trial-confirmed \
  --require-observed vla.actual_action_shape \
  --require-observed vla.actual_chunk_size \
  --require-observed env.termination_truncation_arrays \
  --output "$RPENT_RUNTIME_PROBE_INFERENCE"
```

Extract the same current RGB image and verify SAM3 acceptance:

```bash
python - <<'PY'
import os
from pathlib import Path
import numpy as np

root = Path(os.environ["RPENT_HANDOFF_OUTPUT_ROOT"])
with np.load(root / "probe_observation.npz", allow_pickle=False) as data:
    image = data["main_images"]
if image.ndim == 4 and image.shape[0] == 1:
    image = image[0]
if image.ndim != 3 or image.shape[-1] != 3:
    raise SystemExit(f"unexpected current RGB shape: {image.shape}")
np.save(root / "probe_sam3_image.npy", image)
PY

rpent-handoff probe-runtime \
  --config configs/research/handoff/matrix/main_and_ablations.json \
  --sam3-endpoint http://127.0.0.1:8114 \
  --sam3-image-npy "$RPENT_HANDOFF_OUTPUT_ROOT/probe_sam3_image.npy" \
  --sam3-text-prompt "$RPENT_HANDOFF_TARGET_DESCRIPTION" \
  --allow-sam3-inference \
  --require-observed sam3.current_observation_acceptance \
  --output "$RPENT_RUNTIME_PROBE_SAM3"
```

Preserve every report. A server-reported checkpoint ID is an operator-supplied
content identity cross-checked against the matrix; the digest commands above are
the evidence that the supplied ID actually names bytes. Stop the disposable
services before measured jobs.

Two facts cannot be established by read-only introspection. Run either
diagnostic only in a separate throwaway service session when its answer is
needed. For mid-chunk termination, first prepare and review a task-specific NPY
action sequence that is expected to reach `done`; the probe must remain
`requires_diagnostic` if the sequence does not trigger termination.

```bash
export RPENT_CHUNK_DIAGNOSTIC_ACTIONS=<absolute-reviewed-action-chunk.npy>
rpent-handoff probe-runtime \
  --env-endpoint http://127.0.0.1:8112 \
  --chunk-actions-npy "$RPENT_CHUNK_DIAGNOSTIC_ACTIONS" \
  --allow-destructive-chunk-diagnostic \
  --fresh-env-reset-confirmed --isolated-env-trial-confirmed \
  --require-observed diagnostic.chunk_done_mid_chunk_behavior \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/runtime_probe_chunk_diagnostic.json"
```

For VLA episode-local hidden state, supply a reviewed import path implementing
the typed `HiddenStateDiagnosticResult` protocol. It must control model-session
reset and stochasticity across at least two repetitions; repeated identical
requests alone are not conclusive.

```bash
export RPENT_HIDDEN_STATE_DIAGNOSTIC=<python.module:reviewed_callback>
rpent-handoff probe-runtime \
  --vla-endpoint http://127.0.0.1:8113 \
  --hidden-state-diagnostic "$RPENT_HIDDEN_STATE_DIAGNOSTIC" \
  --allow-hidden-state-diagnostic --isolated-model-session-confirmed \
  --require-observed diagnostic.vla_hidden_episode_state \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/runtime_probe_hidden_state.json"
```

Archive inconclusive/error reports too; never promote either diagnostic to a
known fact by editing its JSON. Restart all disposable services afterward.

## 2. Tiny smoke rollout, then Original Harness parity

This pre-Gate-0 manifest has only a direct controlled condition and Original
Harness. It needs no learned or positive-reference artifact, so it can be
strictly materialized before data collection.

```bash
rpent-handoff manifest \
  --config configs/research/handoff/matrix/smoke_parity.json \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/smoke_manifest.json" \
  --trials-jsonl "$RPENT_HANDOFF_OUTPUT_ROOT/smoke_trials.jsonl"

rpent-handoff offline-preflight \
  --config configs/research/handoff/matrix/smoke_parity.json \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/smoke_manifest.json" \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/preflight_smoke_strict.json"
```

Review and execute exactly one direct-controller rollout:

```bash
rpent-handoff run-controlled \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/smoke_manifest.json" \
  --condition smoke-controlled-direct-pi0 --limit 1 --dry-run \
  --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/smoke_direct.json"

rpent-handoff run-controlled \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/smoke_manifest.json" \
  --condition smoke-controlled-direct-pi0 --limit 1 --execute \
  --execution-token I_UNDERSTAND_SERVER_EXECUTION \
  --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/smoke_direct.json" \
  --journal "$RPENT_HANDOFF_OUTPUT_ROOT/journals/smoke_controlled.jsonl" \
  --capture-output
```

Stop on any wrong task/reset/checkpoint identity, missing live post-reset
sidecar, zero VLA entry, malformed state/result nesting, or unexpected analytic
staging. Task failure alone is an observed outcome, not infrastructure failure.

Next inspect and execute exactly one Original Harness trial. The reviewed child
command must have no `--handoff-config`; planner-visible tools must remain
`pi0_pick` and `pi0_doubled`.

```bash
rpent-handoff run-full-agent \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/smoke_manifest.json" \
  --condition smoke-original-harness --limit 1 --dry-run \
  --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/smoke_original.json"

rpent-handoff run-full-agent \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/smoke_manifest.json" \
  --condition smoke-original-harness --limit 1 --execute \
  --execution-token I_UNDERSTAND_SERVER_EXECUTION \
  --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/smoke_original.json" \
  --journal "$RPENT_HANDOFF_OUTPUT_ROOT/journals/smoke_full_agent.jsonl" \
  --capture-output

rpent-handoff summarize-full-agent \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/smoke_manifest.json" \
  --condition smoke-original-harness \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/smoke_original_episode.jsonl"
```

For parity, compare the same task, seed, exact reset, checkpoint bytes, planner,
prompt, and budgets with a separate clean upstream worktree checked out at
`RPENT_UPSTREAM_BASE_REF`; record its `git rev-parse HEAD` output.
`RPENT_SOURCE_REVISION` is the research checkout's composite
commit-plus-worktree byte identity and is deliberately not a checkout ref.
Compare public tool schemas, child command, state/result structure, VLA entry,
termination/truncation, and normal artifacts. Research-only reset/completion
sidecars are separately named observational evidence. Stochastic task success
alone is not a parity proof. Do not start Gate-0 until smoke and parity pass.

## 3. Tiny Gate-0, then full leakage-safe Gate-0

Set the privileged setup-only target key after verifying it from the runtime.
It may produce labels/setup geometry, but it must never enter online policy
features. The online target comes from current RGB-D/SAM3.

```bash
export RPENT_GATE0_SUITE="$RPENT_HANDOFF_SUITE"
export RPENT_GATE0_TASK="$RPENT_HANDOFF_TASK"
export RPENT_GATE0_TARGET_ID="$RPENT_HANDOFF_TARGET_ID"
export RPENT_GATE0_TARGET_DESCRIPTION="$RPENT_HANDOFF_TARGET_DESCRIPTION"
export RPENT_GATE0_SKILL_PROMPT="$RPENT_HANDOFF_SKILL_PROMPT"
export RPENT_GATE0_TARGET_POSITION_KEY=<verified-privileged-setup-key>
```

First collect two trials in cohort 0, inspect them, then resume the immutable
job. `--no-resume` fails before starting services if authoritative files exist.

```bash
export RPENT_GATE0_SEED=0
export RPENT_GATE0_SAMPLER_SEED=2026080900
export RPENT_GATE0_RUN_ID="${RPENT_HANDOFF_EXPERIMENT_ID}-gate0-0"

rpent-handoff collect-gate0 \
  --job configs/research/handoff/gate0/server_example.json \
  --limit 2 --dry-run \
  --plan-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/gate0_tiny.json"

rpent-handoff collect-gate0 \
  --job configs/research/handoff/gate0/server_example.json \
  --limit 2 --execute --execution-token I_UNDERSTAND_SERVER_EXECUTION \
  --plan-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/gate0_tiny.json"

rpent-handoff collect-gate0 \
  --job configs/research/handoff/gate0/server_example.json \
  --resume --execute --execution-token I_UNDERSTAND_SERVER_EXECUTION \
  --plan-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/gate0_full_0.json"
```

Require correct requested/reached geometry, exact live reset ID, deployment
provenance, positive and negative frozen-VLA outcomes, and explicit exclusion
of perception/staging/labeler failures from competence training. Then collect
two additional independent reset cohorts with different sampler seeds:

```bash
for cohort in 1 2; do
  export RPENT_GATE0_SEED="$cohort"
  export RPENT_GATE0_SAMPLER_SEED="$((2026080900 + cohort))"
  export RPENT_GATE0_RUN_ID="${RPENT_HANDOFF_EXPERIMENT_ID}-gate0-${cohort}"
  rpent-handoff collect-gate0 \
    --job configs/research/handoff/gate0/server_example.json \
    --dry-run \
    --plan-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/gate0_full_${cohort}.json"
  rpent-handoff collect-gate0 \
    --job configs/research/handoff/gate0/server_example.json \
    --execute --execution-token I_UNDERSTAND_SERVER_EXECUTION \
    --plan-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/gate0_full_${cohort}.json"
done
```

Different sampler seeds intentionally produce disjoint candidate IDs across
reset cohorts. Within a cohort, candidate IDs remain stable across repeats.
This avoids the transitive reset-by-candidate component that would make a
three-way leakage-safe split mathematically impossible.

```bash
export GATE0_OUTCOMES_0="$RPENT_HANDOFF_OUTPUT_ROOT/gate0/${RPENT_HANDOFF_EXPERIMENT_ID}-gate0-0/online/outcomes.jsonl"
export GATE0_OUTCOMES_1="$RPENT_HANDOFF_OUTPUT_ROOT/gate0/${RPENT_HANDOFF_EXPERIMENT_ID}-gate0-1/online/outcomes.jsonl"
export GATE0_OUTCOMES_2="$RPENT_HANDOFF_OUTPUT_ROOT/gate0/${RPENT_HANDOFF_EXPERIMENT_ID}-gate0-2/online/outcomes.jsonl"
GATE0_ARGS=(
  --outcomes "$GATE0_OUTCOMES_0"
  --outcomes "$GATE0_OUTCOMES_1"
  --outcomes "$GATE0_OUTCOMES_2"
)
```

## 4. Train, calibrate, offline-evaluate, and build baselines

Train every declared representation from the same eligible cohort. Each model
artifact binds estimator bytes, calibration, split membership, feature spec,
target, source/runtime identity, and training configuration.

```bash
for pair in \
  "full:deployment_full" \
  "absolute:absolute" \
  "relative:target_relative" \
  "relative_visual:target_relative_visual"; do
  name="${pair%%:*}"
  spec="${pair##*:}"
  artifact="$RPENT_HANDOFF_OUTPUT_ROOT/models/$name"
  rpent-handoff train "${GATE0_ARGS[@]}" \
    --feature-spec "configs/research/handoff/features/${spec}.json" \
    --training-config configs/research/handoff/training/hgb_bootstrap_isotonic.json \
    --artifact-dir "$artifact" --repo-root . \
    --external-runtime-identity "$RPENT_HANDOFF_OUTPUT_ROOT/runtime_probe_readonly.json"
done

rpent-handoff materialize-splits "${GATE0_ARGS[@]}" \
  --assignment "$RPENT_HANDOFF_MODEL_FULL/split_assignment.json" \
  --output-dir "$RPENT_HANDOFF_MODEL_FULL/splits"

rpent-handoff evaluate \
  --artifact-dir "$RPENT_HANDOFF_MODEL_FULL" \
  --outcomes "$RPENT_HANDOFF_MODEL_FULL/splits/test.jsonl" \
  --trust-artifact \
  --output "$RPENT_HANDOFF_MODEL_FULL/heldout_evaluation.json"

rpent-handoff build-positive-references \
  --outcomes "$RPENT_HANDOFF_MODEL_FULL/splits/train.jsonl" \
  --split-assignment "$RPENT_HANDOFF_MODEL_FULL/split_assignment.json" \
  --target-label primitive_success \
  --output "$RPENT_HANDOFF_POSITIVE_REFERENCES"
```

The positive-only artifact is built from the training split, never the complete
Gate-0 dataset. Training must stop if three components, both classes, requested
labels, or calibration support are absent; collect more independent data rather
than weakening grouping or changing the target after seeing results.

Materialize the realized matched-context oracle used only for evaluation:

```bash
rpent-handoff materialize-oracle "${GATE0_ARGS[@]}" \
  --config configs/research/handoff/oracle_cost.json \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/gate0_oracle_annotated.jsonl" \
  --report-output "$RPENT_HANDOFF_OUTPUT_ROOT/gate0_oracle_report.json"
```

The oracle selects the minimum realized configured cost only among candidates
executed under the exact same run/suite/task/seed/reset/repeat/skill/controller
context. It is marked post-hoc and is never policy-eligible.

## 5. Generate the immutable main experiment manifest

All behavior-affecting files now exist, so generate the executable manifest.
Trial IDs bind the full runtime/planner configuration, source revision,
checkpoint IDs, handoff-config bytes, model estimator/manifest bytes, and
positive-reference artifact bytes.

```bash
rpent-handoff manifest \
  --config configs/research/handoff/matrix/main_and_ablations.json \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  --trials-jsonl "$RPENT_HANDOFF_OUTPUT_ROOT/trials.jsonl"

rpent-handoff offline-preflight \
  --config configs/research/handoff/matrix/main_and_ablations.json \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/preflight_strict.json"
```

Do not execute unless every strict check passes.

## 6. Controlled online switching experiment

Run the controlled main comparison first, excluding conditions prefixed
`ablation-`. Review the complete plan before execution.

```bash
CONTROLLED_MAIN=(
  --condition controlled-direct-pi0
  --condition controlled-canonical
  --condition controlled-fixed-distance
  --condition controlled-positive-retrieval
  --condition controlled-positive-support
  --condition controlled-competence-threshold
  --condition controlled-competence-projection
  --condition controlled-ours-conservative
)

rpent-handoff run-controlled \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  "${CONTROLLED_MAIN[@]}" --dry-run \
  --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/controlled_main.json"

rpent-handoff run-controlled \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  "${CONTROLLED_MAIN[@]}" --execute \
  --execution-token I_UNDERSTAND_SERVER_EXECUTION \
  --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/controlled_main.json" \
  --journal "$RPENT_HANDOFF_OUTPUT_ROOT/journals/controlled_all.jsonl" \
  --continue-on-error
```

Positive/negative outcomes are both data. Infrastructure identity mismatch,
missing artifact/checksum, privileged provenance, zero VLA entry after a claimed
handoff, or protocol/config mismatch invalidate a trial. An ordinary task or
primitive failure remains an observed result.

## 7. Full Harness system experiment

Run Original Harness and the full local-governor condition from the same main
manifest. The Original command must still have no `--handoff-config`; the enabled
condition exposes the same `pi0_pick`/`pi0_doubled` schemas and routes their
handlers internally.

```bash
FULL_MAIN=(
  --condition full-original-harness
  --condition full-local-governor
)

rpent-handoff run-full-agent \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  "${FULL_MAIN[@]}" --dry-run \
  --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/full_agent_main.json"

rpent-handoff run-full-agent \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  "${FULL_MAIN[@]}" --execute \
  --execution-token I_UNDERSTAND_SERVER_EXECUTION \
  --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/full_agent_main.json" \
  --journal "$RPENT_HANDOFF_OUTPUT_ROOT/journals/full_agent_all.jsonl" \
  --capture-output --continue-on-error
```

A retry with a transcript, states trace, reset sidecar, completion sidecar, or
handoff outcome already present is refused; resume only untouched/interrupted
trial outputs. Then create episode-scoped summaries. The run-local post-reset
sidecar is definitive; detached probes are expectation cross-checks.

```bash
rpent-handoff summarize-full-agent \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  "${FULL_MAIN[@]}" \
  --runtime-probe "$RPENT_RUNTIME_PROBE_READONLY" \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/full_agent_episode_summaries.jsonl"
```

Planner errors, missing finishes, direct-tool errors, detailed governor
failures, unknown counts, exact reset contradictions, and protocol violations
remain explicit. Episode summaries are excluded from model training.

## 8. Ablations

Run ablations only after the main controlled and full-system comparisons have
finished. Reuse the same immutable manifest and lifecycle journal; `--resume`
prevents rerunning completed main trials.

```bash
ABLATIONS=(
  --condition ablation-evidence-outcome-projection-relative
  --condition ablation-evidence-positive-projection-relative
  --condition ablation-decision-fixed
  --condition ablation-decision-threshold
  --condition ablation-uncertainty-mean
  --condition ablation-representation-absolute
  --condition ablation-representation-relative
  --condition ablation-representation-relative-visual
)

rpent-handoff run-controlled \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  "${ABLATIONS[@]}" --dry-run \
  --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/controlled_ablations.json"

rpent-handoff run-controlled \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  "${ABLATIONS[@]}" --execute --resume \
  --execution-token I_UNDERSTAND_SERVER_EXECUTION \
  --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/controlled_ablations.json" \
  --journal "$RPENT_HANDOFF_OUTPUT_ROOT/journals/controlled_all.jsonl" \
  --continue-on-error
```

Do not reinterpret positive-only versus success+failure methods as a perfectly
isolated estimator ablation. For representation/uncertainty plots, filter all
other structural factors to one value.

## 9. Aggregate layers separately and plot observed data

Aggregation rejects mixed execution layers/scopes and per-protocol violations.
Build the list of controlled invocation shards from the manifest:

```bash
mapfile -t CONTROLLED_OUTCOMES < <(python - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["RPENT_HANDOFF_OUTPUT_ROOT"])
manifest = json.loads((root / "experiment_manifest.json").read_text())
for trial in manifest["trials"]:
    if trial["execution_layer"] != "controlled":
        continue
    path = Path(trial["output_dir"]) / "handoff" / "outcomes.jsonl"
    if path.is_file() and path.stat().st_size:
        print(path)
PY
)
CONTROLLED_ARGS=()
POLICY_ORACLE_ARGS=()
for path in "${CONTROLLED_OUTCOMES[@]}"; do
  CONTROLLED_ARGS+=(--outcomes "$path")
  POLICY_ORACLE_ARGS+=(--policy-outcomes "$path")
done

# Re-materialize the Gate-0 oracle together with the actually chosen controlled
# policy states. This produces policy-vs-matched-landscape regret; it is still
# post-hoc and never becomes training or online policy input.
rpent-handoff materialize-oracle "${GATE0_ARGS[@]}" \
  "${POLICY_ORACLE_ARGS[@]}" \
  --config configs/research/handoff/oracle_cost.json \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/matched_gate0_oracle_annotated.jsonl" \
  --policy-output "$RPENT_HANDOFF_OUTPUT_ROOT/controlled_policy_oracle_annotated.jsonl" \
  --report-output "$RPENT_HANDOFF_OUTPUT_ROOT/matched_policy_oracle_report.json"

rpent-handoff aggregate \
  --outcomes "$RPENT_HANDOFF_OUTPUT_ROOT/controlled_policy_oracle_annotated.jsonl" \
  --target-label primitive_success \
  --output-dir "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/controlled"

rpent-handoff aggregate \
  --outcomes "$RPENT_HANDOFF_OUTPUT_ROOT/full_agent_episode_summaries.jsonl" \
  --target-label task_success \
  --output-dir "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/full_agent"

rpent-handoff aggregate \
  --outcomes "$RPENT_HANDOFF_OUTPUT_ROOT/matched_gate0_oracle_annotated.jsonl" \
  --target-label primitive_success \
  --output-dir "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/gate0"

# Calibration is never pooled across predictive identities. Aggregate the
# reference condition alone before drawing its calibration curve.
mapfile -t OURS_OUTCOMES < <(python - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["RPENT_HANDOFF_OUTPUT_ROOT"])
manifest = json.loads((root / "experiment_manifest.json").read_text())
for trial in manifest["trials"]:
    if trial["condition"]["name"] != "controlled-ours-conservative":
        continue
    path = Path(trial["output_dir"]) / "handoff" / "outcomes.jsonl"
    if path.is_file() and path.stat().st_size:
        print(path)
PY
)
OURS_ARGS=()
for path in "${OURS_OUTCOMES[@]}"; do
  OURS_ARGS+=(--outcomes "$path")
done
rpent-handoff aggregate "${OURS_ARGS[@]}" \
  --target-label primitive_success \
  --output-dir "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/controlled_ours_calibration"
```

```bash
rpent-handoff plot --kind success-cost \
  --summary "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/controlled/summary.json" \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/controlled/success_cost.png"

rpent-handoff plot --kind calibration \
  --summary "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/controlled_ours_calibration/summary.json" \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/controlled_ours_calibration/calibration.png"

rpent-handoff plot --kind regret \
  --summary "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/controlled/summary.json" \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/controlled/policy_vs_gate0_regret.png"

rpent-handoff plot --kind gate0 \
  --rows-csv "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/gate0/results.csv" \
  --value-key primitive_success \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/gate0/landscape.png"

rpent-handoff plot --kind ablation \
  --rows-csv "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/controlled/results.csv" \
  --factor-key representation --value-key primitive_success \
  --where execution_layer=controlled \
  --where record_scope=handoff_invocation \
  --where method=outcome_calibrated_switching \
  --where evidence_mode=success_and_failure \
  --where decision_mode=online_switching \
  --where uncertainty_mode=conservative \
  --where hierarchy_mode=local_governor \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/controlled/representation_ablation.png"

rpent-handoff plot --kind ablation \
  --rows-csv "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/controlled/results.csv" \
  --factor-key uncertainty_mode --value-key primitive_success \
  --where execution_layer=controlled \
  --where record_scope=handoff_invocation \
  --where method=outcome_calibrated_switching \
  --where representation=deployment_full \
  --where evidence_mode=success_and_failure \
  --where decision_mode=online_switching \
  --where hierarchy_mode=local_governor \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/controlled/uncertainty_ablation.png"
```

The `overall` aggregation is labeled a pooled inventory, not a method
comparison. Per-method/task rows retain condition, controller configuration,
execution layer, and record scope. Ablation plotting rejects uncontrolled mixed
factors. Evidence and decision comparisons deliberately change method family,
and hierarchy changes the system layer; report those A/C/E axes from the
condition-preserving tables instead of presenting them as isolated estimator
plots. Missing telemetry stays null; never replace it with zero.

## 10. Stop conditions and archive

- Setup/staging/perception/outcome-label failures are not frozen-VLA negatives.
- A reset, checkpoint, model/reference checksum, source, planner, or controller
  identity mismatch invalidates resume/comparison.
- A single-class or fewer-than-three-component cohort cannot support the stated
  calibrated split.
- `requires_diagnostic` remains unknown until its explicitly gated diagnostic is
  run. Destructive chunk and hidden-state diagnostics require isolated sessions.
- A protocol-violating full-agent run is preserved but excluded from
  per-protocol aggregation.
- Code completion is not empirical evidence that handoff state matters, that the
  main method wins, or that residual novelty survives peer review.

Archive matrix and Gate-0 jobs, the exact source identity, checkpoint digest
procedure/output, manifests and child plans, resolved runtime configs, run-local
reset/completion sidecars, raw positive and negative outcome shards, privileged
setup labels in their separate namespace, model/reference manifests and bytes,
split assignments, lifecycle journals, probe reports, aggregate JSON/CSV, plots,
and all service/child logs.
