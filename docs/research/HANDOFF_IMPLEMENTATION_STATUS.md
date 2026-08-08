# Controller-Handoff Implementation Ledger

> Persistent work ledger for the Phase A + Phase B goal started on 2026-08-08.
> This is an implementation record, not an empirical-results document.

## Repository baseline

- Starting commit: `97ad4ff6e922c9bfa258711b4be558a4cd7f6ecd` (`main`).
- Starting worktree: untracked `AGENTS.md`, `docs/OVERVIEW.md`,
  `docs/research/HANDOFF_PROJECT_CONTEXT.md`, and
  `docs/research/RPENT_CODEBASE_GUIDE.md`.
- `docs/OVERVIEW.md` is a pre-existing user file and is excluded from this work.
- No commit, push, pull, branch change, model download, dependency installation,
  simulator rollout, Gate-0 run, or pytest run is authorized for this goal.

## Source-verified execution boundary

- The agent process owns `LiberoPrimitives`, the env/VLA/SAM3 clients, and local
  analytic loops.
- `LiberoPrimitives._step_env()` provides observation and cancellation feedback
  after each analytic action.
- Handoff can occur before the first `_vlm_chunk()` call; the existing learned
  primitives then retain their current Pi0.5 execution semantics.
- A learned chunk is statically configured as five 7-D actions, but actual model
  shape and within-chunk termination behavior remain server-runtime-unverified.
- Simulator raw observations can contain privileged object coordinates. The
  online handoff path must use an explicit deployment-feature whitelist and a
  perception-derived target estimate.

## Phase A implementation checklist

- [ ] Versioned domain records and deterministic JSON serialization.
- [ ] Feature specifications, builders, ablations, and fail-closed provenance
  firewall.
- [ ] Separate privileged setup/label/oracle namespace.
- [ ] Append-only resumable outcome dataset and explicit outcome labeler.
- [ ] Group-aware splits, train/calibration/test pipeline, model artifacts, and
  feature/schema compatibility checks.
- [ ] Constant, logistic, nonlinear tabular, calibrated, and bootstrap outcome
  models with non-placeholder uncertainty.
- [ ] Candidate generation and prediction with explicit approximation metadata.
- [ ] Unified direct/fixed/retrieval/threshold/projection/support/main/oracle
  policy surface.
- [ ] Bounded, recoverable, fully logged Handoff Governor.
- [ ] Current-observation perception, injected, and privileged target providers.
- [ ] Opt-in LIBERO composite tools delegating to unchanged Pi0.5 primitives.
- [ ] Transparent env/VLA instrumentation and baseline-disabled parity.
- [ ] Gate-0 collector and controller-level controlled runner.
- [ ] Full RPent/Harness launch path and configuration-driven experiment matrix.
- [ ] Runtime preflight and focused probes for every server-only unknown.
- [ ] Metrics, grouped statistics, regret, aggregation, and generic plots.
- [ ] Focused pure/fake-boundary tests (written but not executed in this turn).
- [ ] Experiment matrix, implementation docs, and Linux server runbook.
- [ ] Static completion, privileged-leakage, and baseline-preservation audits.

## Phase B checklist

- [ ] Critique the completed design from author and skeptical-reviewer views.
- [ ] Apply the problem-reality gate to research-level alternatives.
- [ ] Search current primary-source prior art before retaining any candidate.
- [ ] Record mechanism-level PASS/UNCERTAIN/FAIL judgments and residual risk.
- [ ] Implement clearly worthwhile in-scope engineering improvements and any
  surviving research variant without replacing the Phase-A reference method.
- [ ] Update experiment/config/test/runbook surfaces for retained changes.
- [ ] Write `AUTONOMOUS_RESEARCH_REVIEW.md` and repeat all final audits.

## Verification status

- Required project context: read completely.
- Source inspection: in progress, limited to implementation-relevant paths.
- Runtime/tests: intentionally not run.

