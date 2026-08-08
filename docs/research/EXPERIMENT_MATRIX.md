# Controller-Handoff Experiment Matrix

> Status: implementation-complete, server-runtime-unverified (2026-08-08).
> This document maps executable configuration to intended comparisons. It does
> not report results.

## 1. Common experimental unit

Every expanded trial binds a suite, task, seed, exact live reset identity,
repeat, controller condition, frozen-VLA checkpoint, feature specification,
training target, and isolated output directory. `configuration_id` identifies
the complete launch document; `controller_configuration_id` hashes the
scientific controller settings while excluding run/output/telemetry metadata.

`reset_id_template` is deliberately `null` in the shipped matrix. The logical
repeat number is not a physical reset. On the server, the LIBERO environment
must report its exact global reset id; Gate-0 cross-checks that live value and
fails closed when it is unavailable or contradictory. Repeated executions of
the same physical reset share the same reset group during splitting and grouped
bootstrap.

The deployment training target in the examples is `primitive_success`. Task
success, primitive success, truncation, planner-declared finish, staging
failure, and VLA/RPC failure remain distinct fields. A condition may choose a
different explicit target, but model artifacts and evaluation inputs must agree.

## 2. Gate-0 competence landscape

The job template is
`configs/research/handoff/gate0/server_example.json`. Its Latin-hypercube sampler
varies target-relative XYZ, standoff, wrist yaw, and wrist pitch, with configured
repeats and a deterministic sampler seed.

For each trial the collector performs:

```text
fresh exact reset
-> read privileged target only into an ExperimentSetupRecord
-> analytically stage to the requested candidate
-> measure the actual reached state
-> re-observe through the deployment SAM3/RGB-D path
-> invoke unchanged frozen Pi0.5
-> store positive or negative execution outcome
```

Staging failures are retained but excluded from VLA-success training because the
VLA was never invoked. Setup values live under `privileged/setups.jsonl`; online
decisions/outcomes live under `online/`. Resumability uses stable trial ids and
strict manifest/config identity checks.

Expected artifacts include `collection_summary.json`, setup records, decision
and outcome JSONL, runtime-event JSONL, run manifest, and normal RPent/LIBERO
state/video/log artifacts. No landscape is plotted until observed rows exist.

## 3. Main comparison

The configuration-driven matrix is
`configs/research/handoff/matrix/main_and_ablations.json`.

| Condition family | Execution layer | Method | Evidence / purpose | Config mapping |
|---|---|---|---|---|
| Direct frozen VLA | controlled | `direct_frozen_pi0` | no analytic staging | `policies/direct_pi0.json` |
| Original Harness | full agent | `original_harness` | untouched planner-mediated upstream behavior | no handoff config |
| Canonical staging | controlled | `fixed_canonical_precontact` | fixed target-relative pose | `policies/canonical_staging.json` |
| Fixed distance | controlled | `fixed_distance` | fixed standoff | `policies/fixed_distance.json` |
| Positive retrieval | controlled | `positive_nearest_success` | success-only nearest reference | `policies/positive_retrieval.json` plus a reference artifact |
| Positive support | controlled | `positive_support_region` | success-only support/bridge region | `policies/positive_support_region.json` plus a reference artifact |
| Competence threshold | controlled | `competence_probability_threshold` | success+failure probability gate | `policies/competence_threshold.json` plus model artifact |
| Competence projection | controlled | `competence_projection` | nearest candidate above probability threshold | `policies/competence_projection.json` plus model artifact |
| Main reference method | controlled | `outcome_calibrated_switching` | success+failure, cost- and uncertainty-aware one-step switching | `policies/outcome_switching_conservative.json` plus model artifact |
| Full local governor | full agent | `outcome_calibrated_switching` | local governor inside the complete RPent/Harness system | same policy plus full-agent launch |
| Oracle upper bound | post-hoc Gate-0 analysis | `posthoc_oracle_upper_bound` | matched realized candidate costs only | explicit privileged/post-hoc mode; never online |

The positive-only files contain placeholders for a path, not fake examples. Build
the versioned reference artifact from observed successful outcomes before
materializing those trials.

## 4. Ablations

| Axis | Levels represented | Interpretation |
|---|---|---|
| Evidence | positive-only retrieval/support; success+failure outcome model | asks whether negative outcomes add useful boundary information; method differences must be acknowledged rather than treated as a perfectly isolated estimator ablation |
| Representation | `absolute`, `target_relative`, `target_relative_visual`, `deployment_full` | all use deployment-allowed provenance; future visual features are explicitly unavailable or marked as approximated |
| Decision | fixed, threshold, projection, online switching | separates geometry rules, competence membership, projected staging, and repeated cost comparison |
| Uncertainty | mean probability; conservative bootstrap score | compares calibrated mean with uncertainty-penalized success estimate |
| Hierarchy | Original planner-mediated sequence; local governor | compares full-system orchestration while preserving the original baseline path |

Feature configs live in `configs/research/handoff/features/`; training configs
live in `configs/research/handoff/training/`. Model paths in the matrix are
environment variables and are rebound to absolute per-trial paths when resolved
runtime configs are materialized. Relative positive-reference paths are resolved
against the source policy config, not the process working directory.

## 5. Metrics and statistics

Outcome-model evaluation provides accuracy where defined, AUROC, AUPRC, Brier
score, log loss, ECE/calibration bins, and risk-coverage diagnostics. Single-class
metrics return unavailable values instead of misleading numbers.

Controller aggregation retains skill/task success, VLA invocation and failure
rates, analytic step/distance/time, VLA timing and available chunk/action counts,
fallback/abort/intervention rates, termination/failure breakdowns, and handoff
state. Full-agent metadata can add LLM turns/tokens/planner time; absent backend
telemetry stays null. Matched-context oracle costs produce handoff regret only
when the required matched candidate data exists.

Confidence intervals use grouped bootstrap rather than treating repeated rows
from the same task/reset as independent. The split configuration jointly groups
episode, exact reset, and candidate identities to prevent leakage across
train/calibration/test.

## 6. Paper-oriented outputs

`rpent-handoff aggregate` writes:

- `results.csv` (tidy observed records);
- `summary.json` (machine-readable aggregate);
- `per_method.csv` and `per_task.csv`;
- `failure_breakdown.csv`;
- `calibration.csv` when prediction records are available.

`rpent-handoff plot` supports `gate0`, `calibration`, `success-cost`, `regret`,
and `ablation`. Plotting is data-honest: the command requires observed JSON/CSV
inputs and does not synthesize paper numbers.

## 7. Validity rules

- Never compare conditions with different exact resets, checkpoints, task
  budgets, or label definitions without reporting the mismatch.
- Never use setup/object-pose/task-predicate data as online policy features.
- Treat staging failure as a controller outcome, not a failed VLA example.
- Treat the oracle as post-hoc only.
- Keep `full-original-harness` free of `--handoff-config` and research tools.
- Report implementation, runtime verification, and empirical evidence as three
  separate statuses.
