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
  simulator rollout, Gate-0 run, test run, project import, or compile check is
  authorized for this goal.

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

## Phase A static-implementation checklist

- [x] Versioned domain records and deterministic JSON serialization.
- [x] Feature specifications, builders, ablations, and fail-closed provenance
  firewall.
- [x] Separate privileged setup/label/oracle namespace.
- [x] Append-only resumable outcome dataset and explicit outcome labeler.
- [x] Group-aware splits, train/calibration/test pipeline, model artifacts, and
  feature/schema compatibility checks.
- [x] Constant, logistic, nonlinear tabular, calibrated, and bootstrap outcome
  models with non-placeholder uncertainty.
- [x] Candidate generation and prediction with explicit approximation metadata.
- [x] Unified direct/fixed/retrieval/threshold/projection/support/main/oracle
  policy surface.
- [x] Bounded, recoverable, fully logged Handoff Governor.
- [x] Current-observation perception, injected, and privileged target providers.
- [x] Opt-in LIBERO composite tools delegating to unchanged Pi0.5 primitives.
- [x] Transparent env/VLA instrumentation and baseline-disabled parity path.
- [x] Gate-0 collector and controller-level controlled runner.
- [x] Full RPent/Harness launch path and configuration-driven experiment matrix.
- [x] Runtime preflight and focused probes for every server-only unknown.
- [x] Metrics, grouped statistics, regret, aggregation, and generic plots.
- [x] Focused pure/fake-boundary tests written (not executed in this goal).
- [x] Experiment matrix, implementation docs, and Linux server runbook.
- [x] Static completion, privileged-leakage, and baseline-preservation audits.

## Phase B checklist

- [x] Critique the completed design from author and skeptical-reviewer views.
- [x] Apply the problem-reality gate to research-level alternatives.
- [x] Search current primary-source prior art before retaining any candidate.
- [x] Record mechanism-level PASS/UNCERTAIN/FAIL judgments and residual risk.
- [x] Implement clearly worthwhile in-scope engineering improvements; no
  speculative Phase-B research variant survived both gates.
- [x] Update experiment/config/test/runbook surfaces for retained changes.
- [x] Write `AUTONOMOUS_RESEARCH_REVIEW.md`.
- [x] Repeat and close every final static audit.

## Verification status

- Required project context: read completely.
- Source status: final static implementation audit complete on 2026-08-09.
- Static verification: `git diff --check`; text/privileged-leakage/baseline
  inspection; AST parsing of 55 Python files; strict parsing of 26 research JSON
  configs and `pyproject.toml`. No static parse errors or whitespace errors were
  found; Git reported only the checkout's expected LF-to-CRLF notices.
- Tests, project imports, and compile checks: intentionally not run.
- Linux CUDA/MuJoCo/LIBERO/Pi0.5/SAM3 runtime: unverified.
- Empirical Gate-0, calibration, controller, and full-agent results: absent.
