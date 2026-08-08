# Linux GPU Server Runbook

> This is the first runtime-verification path for the 2026-08-08 Phase-A
> implementation. Stop on every failed check; do not reinterpret a warning or
> missing probe as evidence that the runtime contract is satisfied.

All commands are run from the RPent repository root. Replace values in angle
brackets. Use a fresh output root and keep the generated manifests, plans, probe
reports, logs, and resolved configs together.

## 0. Install and pin the runtime

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
```

Installation and asset download were not attempted on Windows. Record `python
--version`, `pip freeze`, GPU driver/CUDA versions, checkpoint paths and digests,
repository revision, and dirty status in the run directory. A successful import
does not validate MuJoCo/EGL, external RLinf behavior, or checkpoint identity.

## 1. Prepare one immutable matrix identity

Choose the final artifact paths now, even though training will create them later.
Changing a path after manifest generation changes the configuration identity.

```bash
export RPENT_HANDOFF_EXPERIMENT_ID=<path-safe-run-id>
export RPENT_HANDOFF_REPEATS=3
export RPENT_HANDOFF_SUITE=<libero-suite>
export RPENT_HANDOFF_TASK=<integer-task-id>
export RPENT_HANDOFF_TARGET_ID=<path-safe-target-id>
export RPENT_HANDOFF_TARGET_DESCRIPTION='<target description>'
export RPENT_HANDOFF_SKILL_PROMPT='<frozen VLA skill instruction>'
export RPENT_HANDOFF_MODEL_FULL="$RPENT_HANDOFF_OUTPUT_ROOT/models/full"
export RPENT_HANDOFF_MODEL_ABSOLUTE="$RPENT_HANDOFF_OUTPUT_ROOT/models/absolute"
export RPENT_HANDOFF_MODEL_RELATIVE="$RPENT_HANDOFF_OUTPUT_ROOT/models/relative"
export RPENT_HANDOFF_MODEL_RELATIVE_VISUAL="$RPENT_HANDOFF_OUTPUT_ROOT/models/relative_visual"
export RPENT_HANDOFF_POSITIVE_REFERENCES="$RPENT_HANDOFF_OUTPUT_ROOT/models/positive_references.json"

rpent-handoff manifest \
  --config configs/research/handoff/matrix/main_and_ablations.json \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  --trials-jsonl "$RPENT_HANDOFF_OUTPUT_ROOT/trials.jsonl"

rpent-handoff offline-preflight \
  --config configs/research/handoff/matrix/main_and_ablations.json \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  --allow-missing-references \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/preflight_before_collection.json"
```

`--allow-missing-references` is only an early structural check. It is not
authorization for collection or evaluation. Before any model-based trial, rerun
strict preflight without this flag.

## 2. Start disposable probe services

Use three terminals (or an equivalent supervised process manager) and one task
cell. These services are for capability probing; do not share their mutable env
episode with a measured trial.

```bash
# Terminal A
python -m robots.libero.env_server --transport http --host 127.0.0.1 \
  --port 8112 --suite "$RPENT_HANDOFF_SUITE" --task "$RPENT_HANDOFF_TASK" \
  --seed 0 --max-episode-steps 10000 --cuda-device 0

# Terminal B
python -m robots.libero.vla_server --transport http --host 127.0.0.1 \
  --port 8113 --model-path "$PI05_CHECKPOINT_PATH" --cuda-device 0

# Terminal C
python -m robots.libero.sam3_server --transport http --host 127.0.0.1 \
  --port 8114 --cuda-device 0
```

If EGL selection, service health, checkpoint load, or GPU memory fails, fix the
environment before proceeding. Do not weaken the configuration or provenance
checks to make a service appear healthy.

## 3. Run read-only capability probes first

```bash
rpent-handoff probe-runtime \
  --env-endpoint http://127.0.0.1:8112 \
  --vla-endpoint http://127.0.0.1:8113 \
  --sam3-endpoint http://127.0.0.1:8114 \
  --discover-host-gpu \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/runtime_probe_readonly.json"
```

Review every fact. Required evidence includes endpoint health, package/runtime
identity, exact live reset id, safe raw/policy observation descriptors, state
shape and diagnostic values, camera availability, model/checkpoint identity,
configured action dimensions, CUDA devices and memory, and termination/truncation
contract descriptors. Raw object values are diagnostic/privileged and must never
be copied into an online feature config.

## 4. Probe actual VLA and SAM3 input/output contracts

The next command resets the env. It is intentionally gated to an isolated,
disposable trial. It captures only the deployment VLA arrays and performs one
non-executed model inference; no predicted action is sent to the environment.

```bash
rpent-handoff probe-runtime \
  --env-endpoint http://127.0.0.1:8112 \
  --vla-endpoint http://127.0.0.1:8113 \
  --sam3-endpoint http://127.0.0.1:8114 \
  --capture-vla-observation-npz "$RPENT_HANDOFF_OUTPUT_ROOT/probe_observation.npz" \
  --inference-instruction "$RPENT_HANDOFF_SKILL_PROMPT" \
  --allow-model-inference --isolated-model-session-confirmed \
  --fresh-env-reset-confirmed --isolated-env-trial-confirmed \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/runtime_probe_inference.json"
```

To exercise SAM3 on that same current main view, first inspect the captured image
layout; only extract it if it is `[H,W,3]` or `[1,H,W,3]` RGB:

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
  --sam3-endpoint http://127.0.0.1:8114 \
  --sam3-image-npy "$RPENT_HANDOFF_OUTPUT_ROOT/probe_sam3_image.npy" \
  --sam3-text-prompt "$RPENT_HANDOFF_TARGET_DESCRIPTION" \
  --allow-sam3-inference \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/runtime_probe_sam3.json"
```

An actual action shape other than the configured contract, non-finite output,
image/layout disagreement, or rejected current-observation SAM3 call is a hard
runtime blocker. Update an environment-specific config only after preserving the
probe evidence; never guess a shape in the online path.

### Destructive diagnostics

`--allow-destructive-chunk-diagnostic` additionally requires a fresh isolated
env, a caller-supplied action array, and both env confirmation flags. It can
describe returned frames/term/trunc arrays, but without an external executed-step
counter it cannot prove whether physics continued after the first done signal.
The hidden-state probe likewise requires a controlled callable supplied through
`--hidden-state-diagnostic module:callable`, `--allow-hidden-state-diagnostic`,
and an isolated model session. An inconclusive result remains inconclusive.

Stop the disposable services before measured runs, unless the experiment config
explicitly owns and records those exact external endpoints.

## 5. Tiny controller smoke rollout

First persist the child plan; inspect its command, output directory, exact trial,
resolved handoff config, and absence of unintended conditions. Heavy launchers
are dry-run by default.

```bash
rpent-handoff run-controlled \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  --condition controlled-direct-pi0 --limit 1 --dry-run \
  --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/smoke_direct.json"

rpent-handoff run-controlled \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  --condition controlled-direct-pi0 --limit 1 --execute \
  --execution-token I_UNDERSTAND_SERVER_EXECUTION \
  --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/smoke_direct.json" \
  --journal "$RPENT_HANDOFF_OUTPUT_ROOT/journals/smoke_direct.jsonl" \
  --capture-output --continue-on-error
```

Require one valid outcome for the exact trial, one real VLA invocation, explicit
labels/provenance, and normal RPent artifacts. A staging, target, RPC, label, or
identity error is a failed smoke test, not a negative competence example.

## 6. Original Harness baseline parity

```bash
rpent-handoff run-full-agent \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  --condition full-original-harness --limit 1 --dry-run \
  --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/original_harness.json"
```

The persisted command must contain no `--handoff-config`. Run it only after that
inspection:

```bash
rpent-handoff run-full-agent \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  --condition full-original-harness --limit 1 --execute \
  --execution-token I_UNDERSTAND_SERVER_EXECUTION \
  --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/original_harness.json" \
  --journal "$RPENT_HANDOFF_OUTPUT_ROOT/journals/original_harness.jsonl" \
  --capture-output
```

For empirical parity, compare the same task/reset/checkpoint/planner/budget with a
separate clean worktree at the recorded upstream revision. Check registered tool
schemas, command construction, state/result nesting, reset identity, VLA calls,
termination, and artifacts. Stochastic task success alone is not a parity proof.

## 7. Tiny Gate-0, then full Gate-0

Set every placeholder used by the Gate-0 template:

```bash
export RPENT_GATE0_RUN_ID=<path-safe-gate0-id>
export RPENT_GATE0_SUITE="$RPENT_HANDOFF_SUITE"
export RPENT_GATE0_TASK="$RPENT_HANDOFF_TASK"
export RPENT_GATE0_SEED=0
export RPENT_GATE0_TARGET_ID="$RPENT_HANDOFF_TARGET_ID"
export RPENT_GATE0_TARGET_DESCRIPTION="$RPENT_HANDOFF_TARGET_DESCRIPTION"
export RPENT_GATE0_SKILL_PROMPT="$RPENT_HANDOFF_SKILL_PROMPT"
export RPENT_GATE0_TARGET_POSITION_KEY=<verified-privileged-setup-key>

rpent-handoff collect-gate0 \
  --job configs/research/handoff/gate0/server_example.json \
  --limit 2 --dry-run \
  --plan-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/gate0_tiny.json"

rpent-handoff collect-gate0 \
  --job configs/research/handoff/gate0/server_example.json \
  --limit 2 --execute --execution-token I_UNDERSTAND_SERVER_EXECUTION \
  --plan-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/gate0_tiny.json"
```

Inspect both rows before scaling: requested versus reached staging state, exact
reset grouping, online SAM3 target provenance, VLA invocation count, labels,
positive/negative retention, and staging-failure exclusion. Then resume the same
job without a limit:

```bash
rpent-handoff collect-gate0 \
  --job configs/research/handoff/gate0/server_example.json \
  --execute --execution-token I_UNDERSTAND_SERVER_EXECUTION --resume \
  --plan-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/gate0_full.json"
```

Never change the sampler/config under an existing run id. A manifest/config
identity mismatch is evidence of an unsafe resume and must fail.

## 8. Train, calibrate, and materialize the held-out split

Assume the Gate-0 online outcome file is stored in `GATE0_OUTCOMES`:

```bash
export GATE0_OUTCOMES="$RPENT_HANDOFF_OUTPUT_ROOT/gate0/$RPENT_GATE0_RUN_ID/online/outcomes.jsonl"

rpent-handoff train \
  --outcomes "$GATE0_OUTCOMES" \
  --feature-spec configs/research/handoff/features/deployment_full.json \
  --training-config configs/research/handoff/training/hgb_bootstrap_isotonic.json \
  --artifact-dir "$RPENT_HANDOFF_MODEL_FULL" \
  --repo-root . \
  --external-runtime-identity "$RPENT_HANDOFF_OUTPUT_ROOT/runtime_probe_readonly.json"

rpent-handoff materialize-splits \
  --outcomes "$GATE0_OUTCOMES" \
  --assignment "$RPENT_HANDOFF_MODEL_FULL/split_assignment.json" \
  --output-dir "$RPENT_HANDOFF_MODEL_FULL/splits"

rpent-handoff evaluate \
  --artifact-dir "$RPENT_HANDOFF_MODEL_FULL" \
  --outcomes "$RPENT_HANDOFF_MODEL_FULL/splits/test.jsonl" \
  --trust-artifact \
  --output "$RPENT_HANDOFF_MODEL_FULL/heldout_evaluation.json"

rpent-handoff build-positive-references \
  --outcomes "$GATE0_OUTCOMES" \
  --target-label primitive_success \
  --output "$RPENT_HANDOFF_POSITIVE_REFERENCES"
```

Repeat training with the absolute, target-relative, and target-relative-visual
feature configs into their predeclared artifact paths. Training must fail if the
requested target is missing, groups cannot form three leakage-free partitions,
or train/calibration lacks both classes. Do not change the target, split, or
calibration method merely to suppress that failure; collect adequate independent
outcomes.

## 9. Strict preflight and controlled online experiment

Regenerate the manifest after all declared artifacts exist, then run strict
preflight without exceptions:

```bash
rpent-handoff manifest \
  --config configs/research/handoff/matrix/main_and_ablations.json \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  --trials-jsonl "$RPENT_HANDOFF_OUTPUT_ROOT/trials.jsonl"

rpent-handoff offline-preflight \
  --config configs/research/handoff/matrix/main_and_ablations.json \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/preflight_strict.json"

rpent-handoff run-controlled \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  --dry-run --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/controlled_all.json"

rpent-handoff run-controlled \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  --execute --execution-token I_UNDERSTAND_SERVER_EXECUTION --resume \
  --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/controlled_all.json" \
  --journal "$RPENT_HANDOFF_OUTPUT_ROOT/journals/controlled_all.jsonl" \
  --continue-on-error
```

Review failures before retrying. `--retry-failed` is an explicit scientific
choice; preserve the original failed lifecycle row and report retries.

## 10. Full RPent/Harness experiment

Configure the planner model/API settings in the matrix or environment, inspect
the persisted plan, then execute:

```bash
rpent-handoff run-full-agent \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  --dry-run --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/full_agent_all.json"

rpent-handoff run-full-agent \
  --manifest "$RPENT_HANDOFF_OUTPUT_ROOT/experiment_manifest.json" \
  --execute --execution-token I_UNDERSTAND_SERVER_EXECUTION --resume \
  --plans-output "$RPENT_HANDOFF_OUTPUT_ROOT/plans/full_agent_all.json" \
  --journal "$RPENT_HANDOFF_OUTPUT_ROOT/journals/full_agent_all.jsonl" \
  --capture-output --continue-on-error
```

Each trial runs in a child process because RPent runtime globals and service
ownership are process-scoped. The Original Harness condition remains a child
command with no research config; enabled conditions receive a per-trial resolved
config bound to that trial's model artifact and execution layer.

## 11. Aggregate and plot observed data

Pass every observed outcome shard explicitly (repeat `--outcomes`):

```bash
rpent-handoff aggregate \
  --outcomes <outcome-shard-1.jsonl> \
  --outcomes <outcome-shard-2.jsonl> \
  --target-label primitive_success \
  --output-dir "$RPENT_HANDOFF_OUTPUT_ROOT/analysis"

rpent-handoff plot --kind success-cost \
  --summary "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/summary.json" \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/success_cost.png"

rpent-handoff plot --kind calibration \
  --summary "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/summary.json" \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/calibration.png"

rpent-handoff plot --kind gate0 \
  --rows-csv "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/results.csv" \
  --output "$RPENT_HANDOFF_OUTPUT_ROOT/analysis/gate0.png"
```

Also generate `regret` and `ablation` plots only when their required matched
oracle or factor data are present. Missing telemetry remains null; do not replace
it with zero. Archive the matrix, manifest, resolved configs, probe reports,
model/reference manifests and checksums, split assignment, lifecycle journals,
raw outcome shards, aggregate JSON/CSV, plots, logs, and source/runtime identity.

## 12. Interpretation stop conditions

- A probe marked `requires_diagnostic`, `unavailable`, or `error` is not a pass.
- A setup/staging/perception failure is not a negative frozen-VLA training label.
- An exact-reset mismatch invalidates paired/grouped comparisons.
- A checkpoint, feature-spec, target-label, or controller-id mismatch invalidates
  resume/evaluation.
- A single-class split cannot support calibrated binary outcome modeling.
- Code completion is not evidence that handoff state matters, that the main
  method beats baselines, or that the method is novel.
