# Milestones

This document records the completed M0–M7 development sequence. It is a compact
history of what was tested, the main result, and where detailed evidence lives;
it is not a roadmap commitment beyond the current repository state.

## M0 — Simulation Foundation ✅

Established the Franka Panda tabletop scene in MuJoCo / robosuite, Cartesian
OSC control, ground-truth object poses, and runnable motion demos. See
[architecture.md](architecture.md).

## M1 — Manipulation Primitives ✅

Added workspace-validated pose and straight-line motion, gripper primitives,
bounded waits, and structured primitive results. Randomized grasp validation
confirmed the controller and primitive boundary.

## M2 — PickSkill ✅

Added an explicit Pick finite-state machine, deterministic grasp candidates,
semantic success checks, and bounded skill-local recovery.

## M3 — PlaceSkill and SkillExecutor ✅

Added Place execution and release verification, deterministic multi-skill
sequencing, and layered `SkillResult` / `TaskResult` propagation.

## M4 — Closed-loop Semantic Agent ✅

Connected Goal → Plan → Validate → Execute → Observe → Compare → Replan using
serializable `WorldState`, machine-readable Skill effects,
`SemanticResidual`, bounded event-triggered replanning, and `AgentResult`. See
[architecture.md](architecture.md).

## M5 — Dynamic World ✅

Added multiple objects and targets, explicit external world modifications,
semantic-step inspection, structured capability gaps, and deterministic
disturbance scenarios. A bounded symbolic search also demonstrated occupied
target clearing by composing only registered Pick and Place skills. See
[dynamic_playground.md](dynamic_playground.md).

## M6 — Classical Push ✅

Added semantic `Push(object, target)` for the `right_side` region, deterministic
push geometry, a Cartesian finite-state implementation, semantic success
checks, and one bounded recovery attempt. The classical path reached 20/20 on
randomized valid placements. See [push_skill.md](push_skill.md).

## M7 — Learned Physical Backends ✅

M7 kept `PushSkill`, semantic verification, safety limits, controller, and
MuJoCo execution fixed while varying only the physical Push backend.

### M7.1 — Pluggable backend and demonstration pipeline

Introduced the `PushBackend` boundary and a pickle-free 20 Hz NPZ pipeline with
episode-level splits, training-only normalization, and absolute Cartesian
action targets. See [learned_push_backend.md](learned_push_backend.md).

### M7.2 — Naive BC baselines

One-step BC and progress-conditioned BC were evaluated as diagnostic imitation
baselines. Both reached 0/20 on the frozen matched states, showing that low
offline error did not establish physical task success. See
[learned_push_backend.md](learned_push_backend.md).

### M7.3 — Temporal aliasing and Chunk BC diagnosis

Dataset diagnostics found near-identical state observations with materially
different actions. K=10/20/40 Chunk BC and K=20 execution horizons H=1/5/20
were compared; the best physical result was 2/20 at K=20/H=20. See
[chunk_bc_experiment.md](chunk_bc_experiment.md).

### M7.4 — Standard LeRobot ACT

LeRobot 0.4.4 ACT was trained once on the unchanged 10D state / 3D absolute
waypoint contract. Queue execution at K=32/H=8 reached 0/20 with 20 timeouts
and no unsafe-action rejection. See [act_push_backend.md](act_push_backend.md).

### M7.5 — Native ACT Temporal Ensemble

The exact same ACT weights were reused with LeRobot's native Temporal Ensemble,
K=32/H=1, and coefficient 0.01. It reached 20/20 on the same frozen states,
with all trials reaching cube proximity and no timeout or unsafe rejection.
This is a fixed-distribution result, not a general manipulation claim. The
single requested evaluation completed the stopping rule; no sweep or further
ablation followed. See [act_push_backend.md](act_push_backend.md) and the
compact [experiments index](experiments.md).

M8 has not been started.
