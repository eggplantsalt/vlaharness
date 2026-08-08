# Controller-Handoff Experiment Matrix

> Status (2026-08-09): static implementation and final static audit complete.
> No test, project import, or compile check has been run. Linux
> CUDA/MuJoCo/LIBERO runtime verification and all empirical results are absent.
> This document maps executable configuration to intended comparisons; it does
> not report results.

## 1. Common experimental unit

Every expanded trial binds suite, task, seed, repeat, controller condition,
frozen Pi0.5 and SAM3 checkpoint IDs, feature/label definitions, planner
backend/model/base URL for full-agent trials, source revision, resolved config
checksums, model/reference artifact identities, estimator checksum, and an
isolated output directory. `configuration_id` identifies the normalized launch
document. Each `trial_id` hashes the fully resolved scientific payload rather
than an output location. The runtime's `controller_configuration_id` hashes
controller behavior while excluding run/output/telemetry metadata.

A full-agent `trial_id` denotes exactly one execution attempt. A started,
failed, or interrupted identity is preserved and cannot be reused for a retry;
a retry requires a newly materialized trial identity.

`reset_id_template` is deliberately `null` in the shipped matrix. A repeat
number is not a physical reset. The live LIBERO server must report the exact
post-reset identity; Gate-0 and full-agent summarization fail closed when it is
missing or contradictory. Repeated executions of one physical reset remain in
the same split/bootstrap group.

The example deployment target is `primitive_success`. Primitive success, skill
success, official task termination, truncation, planner finish, staging failure,
label failure, and VLA/RPC failure remain separate signals. A condition may use
another explicit target only when its artifact and evaluation target agree.

## 2. Gate-0 competence landscape

The job template is
`configs/research/handoff/gate0/server_example.json`. Its Latin-hypercube sampler
varies target-relative XYZ, standoff, wrist yaw, and wrist pitch with configured
repeats and a deterministic sampler seed.

The server protocol collects at least three independently sampled cohorts with
distinct sampler seeds. This creates disjoint candidate components for the
joint episode/reset/candidate group split. `candidate_id` hashes requested
geometry plus the approach axis and is stable across repeats; `sample_id` also
binds the repeat and identifies one execution.

For every sample the collector performs:

```text
fresh exact reset
-> read privileged target only into ExperimentSetupRecord
-> analytically stage to the requested candidate
-> measure the reached state
-> re-observe through deployment SAM3/RGB-D
-> invoke unchanged frozen Pi0.5
-> retain positive or negative execution outcome
```

Staging failures are retained but excluded from VLA-success training because no
VLA was invoked. Labeler failures after VLA entry are explicit outcomes and are
also excluded rather than silently converted to negatives. Setup values live in
`privileged/setups.jsonl`; online decisions/outcomes live below `online/`.
Stable trial and outcome keys make resume idempotent; a contradictory retry is
rejected instead of appended as a second result.

Expected artifacts include `collection_summary.json`, strict run manifest,
setup/decision/outcome/runtime-event JSONL, and normal LIBERO state/video/log
artifacts. No landscape or paper number is synthesized before observed rows
exist.

## 3. Main comparison

The matrix is `configs/research/handoff/matrix/main_and_ablations.json`.

| Condition | Layer | Method | Purpose | Mapping |
|---|---|---|---|---|
| Direct frozen VLA | controlled | `direct_frozen_pi0` | no analytic staging | `policies/direct_pi0.json` |
| Original Harness | full agent | `original_harness` | original planner-mediated behavior | no handoff config |
| Canonical staging | controlled | `fixed_canonical_precontact` | fixed target-relative pose | `policies/canonical_staging.json` |
| Fixed distance | controlled | `fixed_distance` | fixed standoff | `policies/fixed_distance.json` |
| Positive retrieval | controlled | `positive_nearest_success` | success-only nearest reference | `policies/positive_retrieval.json` + reference artifact |
| Positive support | controlled | `positive_support_region` | success-only support/bridge region | `policies/positive_support_region.json` + reference artifact |
| Competence threshold | controlled | `competence_probability_threshold` | success+failure probability gate | `policies/competence_threshold.json` + model artifact |
| Competence projection | controlled | `competence_projection` | nearest candidate above threshold | `policies/competence_projection.json` + model artifact |
| Reference method | controlled | `outcome_calibrated_switching` | cost/uncertainty-aware one-step switching | `policies/outcome_switching_conservative.json` + model artifact |
| Full local governor | full agent | `outcome_calibrated_switching` | same local mechanism in complete RPent | same policy + full-agent launch |
| Oracle upper bound | post-hoc Gate-0 | `materialize-oracle` output | matched realized candidate costs | `oracle_cost.json`; never online |

Positive-only artifacts are built solely from the materialized training split.
Their full references, build settings, exclusions, and source record IDs are
identity-bound. A model artifact ID binds its complete manifest and estimator
checksum. Manifest expansion rejects stale IDs or bytes.

Full-agent handoff does not add planner choices: the planner still sees the
original `pi0_pick` and `pi0_doubled` schemas. Only when a handoff config is
enabled do those handlers route internally through the local governor and then
delegate to the unchanged primitive. With research disabled, the original
handlers and normal CLI outputs remain unchanged. Research-launched Original
Harness trials add hidden, observational reset/completion sidecars only.

## 4. Ablations

| Axis | Levels | Manifest mapping | Interpretation |
|---|---|---|---|
| A — Evidence | positive-only; success+failure | `ablation-evidence-positive-projection-relative` vs `ablation-evidence-outcome-projection-relative` | holds target-relative representation, projection decision, mean score, and local hierarchy fixed; estimator family necessarily differs and is reported |
| B — Representation | `absolute`, `target_relative`, `target_relative_visual`, `deployment_full` | three `ablation-representation-*` conditions vs `controlled-ours-conservative` | deployment-allowed provenance only; future visual features are unavailable or explicitly approximated |
| C — Decision | fixed, threshold, projection, online switching | `ablation-decision-fixed`, `ablation-decision-threshold`, `controlled-competence-projection`, `ablation-uncertainty-mean` | holds `deployment_full`, success+failure evidence, mean score, and local hierarchy fixed |
| D — Uncertainty | mean; conservative bootstrap score | `ablation-uncertainty-mean` vs `controlled-ours-conservative` | empirical uncertainty, not a formal safety guarantee |
| E — Hierarchy | Original planner sequence; local governor | `full-original-harness` vs `full-local-governor` | full-system comparison; method/decision differences are explicit confounders, not an isolated estimator effect |

Feature configs live in `configs/research/handoff/features/`; training configs
live in `configs/research/handoff/training/`. Environment paths are resolved and
rebound per trial. A relative positive-reference path is resolved against its
source policy file, never the process working directory.

## 5. Metrics and statistics

Outcome-model evaluation includes accuracy where defined, AUROC, AUPRC, Brier,
log loss, calibration/ECE, and risk-coverage diagnostics. Single-class metrics
are explicitly unavailable. Controller summaries retain all label levels,
VLA attempts/failures, analytic step/distance/time, VLA time and source-verified
chunk/action counts, fallbacks/aborts/interventions, failure modes, and handoff
state. Full-agent summaries additionally retain available LLM turns/tokens and
planner time; absent backend telemetry stays null.

Confidence intervals use grouped bootstrap. Split assignment jointly groups
episode, exact reset, and candidate identities. The post-hoc oracle annotates
regret only for unique candidates in an exact observed Gate-0 context matching
run/suite/task/seed/reset/repeat/skill/controller and the configured minimum
group size.

Controlled condition comparisons are reset-matched: every compared condition
must have exactly one record for the same task/seed/repeat and exact live
`reset_id`. Aggregation rejects duplicate, incomplete, or nonidentical condition
sets instead of treating them as paired evidence.

Aggregation is layer- and scope-specific. One aggregate call rejects mixtures
of Gate-0 invocation, controlled invocation, and full-agent episode records.
Method/task tables retain execution layer, scope, condition, method, controller
configuration, suite, and task. The overall row is labeled
`pooled_inventory_not_method_comparison`. Ablation plots reject uncontrolled
factor mixtures unless the caller filters them.

## 6. Paper-oriented outputs

`rpent-handoff summarize-full-agent` binds the reviewed child plan and lifecycle
journal to each trial's attempt marker, live runtime attestation, transcript,
states trace, run-local reset/completion sidecars, direct-VLA attempt hash chain,
and detailed governor outcomes. A terminal early failure remains an incomplete
episode in the fixed `end_to_end_attempt_success_rate` denominator while
`task_success` remains unavailable. Planner exceptions, reset conflicts,
unbound outcomes, and protocol violations are not inferred away.

`rpent-handoff materialize-oracle` writes Gate-0 landscape copies and, when
`--policy-outcomes` is supplied, controlled-policy copies annotated with their
realized chosen cost and the unique matching Gate-0 landscape minimum. Policy
matching uses suite/task/seed/exact reset/repeat/skill/checkpoint and refuses an
ambiguous landscape; supplying policy outcomes with zero eligible matches is an
error. All annotations are post-hoc and policy-ineligible.

`rpent-handoff aggregate` writes `results.csv`, `summary.json`,
`per_method.csv`, `per_task.csv`, `failure_breakdown.csv`, and `calibration.csv`
when predictions exist. `rpent-handoff plot` supports `gate0`, `calibration`,
`success-cost`, `regret`, and `ablation`, with repeatable `--where KEY=VALUE`
filters. Both surfaces require observed inputs and never invent rows.

## 7. Validity rules

- Do not compare different exact resets, checkpoints, task budgets, label
  definitions, execution layers, or record scopes as though matched.
- Never expose setup/object pose/task predicate data to an online policy.
- Treat staging and label failures as controller/data outcomes, not failed VLA
  training examples.
- Treat the oracle as post-hoc evaluation only.
- Require exact matched resets for controlled policy comparisons, and use only
  policy-vs-Gate-0 oracle annotations produced from an unambiguous observed
  landscape.
- Keep `full-original-harness` free of `--handoff-config`; its planner-visible
  schemas and physical handlers remain the Original Harness path.
- Require source revision and content-derived Pi0.5/SAM3 IDs, then verify
  server-reported configured IDs before execution.
- Report implementation status, runtime verification, and empirical evidence
  separately.
