# Controller Handoff Research Context

> Status: long-lived project context. This file records the research intent and
> working hypotheses, not facts implied by the current RPent implementation.
> Source-verified implementation facts live in
> [`RPENT_CODEBASE_GUIDE.md`](RPENT_CODEBASE_GUIDE.md).

## 1. Motivation

Harness VLA combines three capabilities:

1. a high-level LLM agent for semantic understanding, task decomposition,
   observation feedback, and primitive orchestration;
2. analytic primitives for relatively predictable, usually non-contact motion;
3. a frozen VLA (currently Pi0.5) for contact-rich skills such as grasping and
   articulation.

This decomposition is useful, but its current semantic tool boundary need not
match the learned controller's physical competence boundary. The same instruction
can have a very different outcome when the end effector, wrist, gripper, target
visibility, or surrounding geometry changes.

We call this mismatch the **controller handoff gap**:

> The high-level agent chooses a symbolic time to invoke a VLA, while whether the
> frozen VLA should physically take control depends on continuous,
> state-dependent embodied conditions.

The research question is therefore not merely “where can the VLA work?” It is:

> While an analytic controller is staging the robot, in which state and at which
> time should analytic control stop and control pass to the frozen VLA?

“Can hand off?” and “Should hand off now?” are different decisions.

## 2. Working problem formulation

> **Working hypothesis — not experimentally established.** The formulation in
> this section is expected to evolve.

Let the pre-handoff embodied state for skill \(\sigma\) be

\[
z_t = \phi(o_t, q_t, g, \sigma).
\]

The representation should preferentially be object-centric and task-relative,
rather than a bare absolute XYZ vector. Candidate fields include:

- end-effector pose relative to the target;
- wrist orientation;
- gripper state;
- target visibility and occlusion;
- camera-relative target geometry;
- reachability or joint margin;
- skill/task identity;
- any additional field that the real RPent interfaces actually support.

No candidate field is assumed to exist merely because it is desirable. The
codebase guide identifies current availability and simulator-only leakage risks.

From positive and negative VLA executions, we may learn

\[
p_\sigma(z) = P(\text{VLA succeeds}\mid z, \sigma),
\]

possibly together with execution cost, uncertainty, and failure mode. A switching
decision compares

\[
Q_{\mathrm{handoff}}(z)
\]

with

\[
Q_{\mathrm{continue}}(z,u),
\]

where \(u\) is an analytic staging action. The desired handoff region has the
conceptual form

\[
H_\sigma = \left\{z:\ Q_{\mathrm{handoff}}(z)
\leq \min_u Q_{\mathrm{continue}}(z,u)\right\}.
\]

The point is not simply to reach a learned competence region. Handoff occurs when
the expected value of further staging is no better than transferring control now.
An eventual extension may jointly optimize a staging path and handoff time,

\[
(r^*, \tau^*) = \arg\min_{r,\tau} J(r,\tau),
\]

but no particular objective or estimator is final yet.

## 3. Intended system architecture

The intended design has two decision timescales and a contact-rich executor:

```text
slow semantic layer:   LLM Agent chooses a subgoal, e.g. grasp(target)
fast physical layer:   local Handoff Governor stages, observes, and switches
contact-rich executor: frozen VLA executes after handoff
```

After semantic commitment to a skill, a local sequence should eventually be able
to run without sending every micro-adjustment back through the LLM:

```text
analytic move -> observe -> adjust -> observe -> handoff -> frozen VLA
```

Reduced LLM turns and latency are useful system effects, but are not an
independent core contribution.

## 4. Novelty guardrails

The central contribution must remain:

> Recast frozen-VLA invocation in agentic manipulation from an LLM tool-call
> heuristic into a continuous physical controller-switching problem calibrated by
> real execution outcomes, deciding online between continuing analytic control
> and handing off now.

The following are not sufficient as the core contribution:

1. using analytic primitives to help a frozen VLA — Harness VLA already does so;
2. learning a VLA competence/capability region and moving the robot into it —
   RoboHarness and related work are close to this framing;
3. retrieving a similar successful pose and motion-planning to it;
4. observing that skill handoff matters — Semantic Handoff and related work study
   handoff failure;
5. reducing LLM calls and therefore latency.

A competence set or threshold, projection to a successful state, and
positive-only memory remain valid components or baselines. They must not displace
the switching formulation as the central claim. In particular, the Outcome Model
must preserve both success and failure evidence instead of becoming a successful
trajectory retriever.

### Closest-neighbor boundary status

The neighbor descriptions above come from the project research brief. They are
**literature-positioning assumptions, not source-code facts and not yet a formal
related-work review**. Before publication claims are made, verify the exact scope,
dates, and wording against the original papers.

## 5. Data and evaluation boundaries

Deployment-time policy inputs must be separated from simulator-only labels.

Allowed in principle for a deployment-realistic policy, when actually provided by
the robot stack:

- calibrated RGB/RGB-D observations;
- camera intrinsics/extrinsics;
- robot proprioception, end-effector pose, and gripper state;
- task instruction and skill identity;
- target geometry derived from a deployment-available perception system.

Privileged by default and forbidden as deployment-time input unless an oracle
ablation is explicit:

- simulator ground-truth object poses;
- contacts/collision state unavailable on the target robot;
- benchmark goal predicates or task-success internals;
- BDDL initialization state;
- hidden simulator model/data fields.

Privileged information may be stored separately for labels, diagnostics, and
evaluation. Dataset schemas should make provenance explicit so a training loader
cannot accidentally include label-only columns as features.

## 6. Implemented logical components

As of 2026-08-08, the Phase-A implementation is complete at the
**statically inspected, server-runtime-unverified** checkpoint. It includes:

- a versioned, serializable handoff-state record with feature provenance;
- an outcome record containing skill, pre-handoff state, result, cost, termination
  reason, and metadata;
- an Outcome Model interface with uncertainty/calibration support;
- an explicit decision result (continue, handoff, abort/fallback, plus rationale);
- a Handoff Governor that owns the fast local loop;
- a LIBERO adapter for deployment-realistic state extraction and analytic staging;
- an opt-in local/composite primitive that invokes the existing VLA primitive
  without changing baseline behavior;
- append-only rollout logging and an offline dataset reader;
- evaluators for predictive calibration, switching quality, task success, cost,
  and baseline parity.

The pure core lives under `rpent/research/handoff/`; the LIBERO-specific adapter,
composite tools, and runtime configuration live under `robots/libero/`. The
implementation also includes Gate-0, training, controlled/full-agent launchers,
runtime probes, aggregation, plotting, strict configs, and focused tests. Those
tests and all simulator/model paths were deliberately not executed in this
Windows implementation pass. Code presence therefore does not establish any of
the experimental hypotheses below.

## 7. Explicit non-goals

At the current stage we are not:

- fine-tuning Pi0.5;
- replacing RPent's high-level planner;
- treating LLM semantic reasoning as the fast physical controller;
- claiming a final mathematical objective before experiments;
- using privileged simulator state in the deployment policy;
- refactoring RPent upstream for architectural elegance;
- replacing textual Harness VLA memory with the outcome dataset;
- claiming latency reduction as the paper's standalone novelty.

## 8. Hypotheses requiring experiments

Every item below is unproven:

1. VLA outcome probability varies sufficiently with pre-handoff embodied state to
   support a useful switching policy.
2. An object-centric/task-relative state transfers better than absolute pose.
3. Success-plus-failure data identifies the usable boundary better than
   positive-only retrieval.
4. Comparing `continue` with `handoff now` outperforms competence-threshold and
   nearest-success projection baselines.
5. A fast local loop improves reliability and cost without losing the semantic
   flexibility of the LLM layer.
6. Existing scripted servo primitives are adequate for initial staging, rather
   than requiring a new motion planner immediately.
7. Five-action Pi0.5 chunks are sufficiently fine for initial handoff experiments;
   intervention during learned-controller execution may later require a finer
   environment stepping interface.
8. A handoff model learned in simulation can retain useful calibration under
   camera, dynamics, or robot-domain shift.

## 9. Roadmap and verification state

### Completed locally: implementation, not empirical validation

- source-anchored architecture and privilege boundary;
- versioned pure-Python records, feature firewall, outcome models, calibration,
  bootstrap uncertainty, grouped splits, artifacts, policies, and governor;
- opt-in LIBERO integration that leaves `pi0_pick` and `pi0_doubled` intact;
- Gate-0, controller-only, and full-agent experiment launch surfaces;
- baselines, ablation configuration, evaluation, aggregation, and plotting;
- server preflight/probes and focused pure/fake-boundary tests.

### Next authoritative checkpoint: Linux GPU server

Follow `SERVER_RUNBOOK.md` in order: install the already-declared dependencies,
run offline preflight and capability probes, perform a tiny smoke rollout, prove
Original Harness baseline parity, then collect Gate-0 data. Runtime facts must be
written into manifests before full collection. Only after held-out training,
controlled trials, and full-agent trials may the hypotheses in Section 8 be
evaluated.

### Later, only if evidence supports it

- evaluate robustness across checkpoints, tasks, camera perturbations, and
  eventually real hardware;
- consider multi-step staging/handoff optimization only if the implemented
  one-step re-observation policy shows a real, reproducible limitation.

## 10. Documentation discipline

Future updates must label claims as one of:

- **source-verified fact** — traced to a repository path and symbol;
- **documented, not verified in source** — stated only by README/docs or an
  external dependency;
- **statically verified / runtime unverified** — path is clear but required
  simulator/model dependencies were unavailable;
- **working hypothesis** — research design awaiting experiments.
