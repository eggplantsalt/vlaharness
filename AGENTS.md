# RPent controller-handoff research instructions

This repository is the upstream base for a long-running RPent/Harness VLA
controller-handoff research project. Before changing runtime code, read:

- `docs/research/HANDOFF_PROJECT_CONTEXT.md`
- `docs/research/RPENT_CODEBASE_GUIDE.md`

The research question is when an analytic controller should stop staging and
hand control to a frozen VLA, using continuous embodied state and real execution
outcomes. A learned competence set, nearest-success projection, positive-only
memory, or fewer LLM calls may be useful components/baselines, but they are not
the core novelty.

Project rules:

- Keep the Original Harness VLA behavior reproducible. New research behavior
  must be opt-in, separately configured, and separately logged.
- Prefer an isolated research module and small adapters over refactoring
  upstream planner, environment, primitive, or RPC code.
- Distinguish source-verified facts, documented-but-unverified claims, and
  working research hypotheses in code, tests, and documentation.
- Never use simulator-only object poses, contacts, task predicates, or other
  privileged state as deployment-time policy input unless the experiment is an
  explicitly labeled oracle ablation. Privileged data may be used for labels and
  evaluation.
- Preserve success and failure outcomes; do not silently reduce outcome data to
  successful trajectory retrieval.
- Add focused unit tests for pure research logic and integration verification
  for every touched execution boundary. Run server/GPU integration tests only in
  the appropriate Linux CUDA/MuJoCo environment.
- Inspect `git status` before editing, preserve user changes, and do not commit
  or push unless explicitly requested.

