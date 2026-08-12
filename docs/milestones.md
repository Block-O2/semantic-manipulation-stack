# Milestones

This roadmap records the development sequence. Completed milestones describe
validated repository capabilities. Later milestones are exploratory directions,
not delivery commitments.

## M0 — Simulation Foundation ✅

Panda and tabletop setup in MuJoCo / robosuite, Cartesian OSC pose control,
end-effector state, ground-truth object poses, and a runnable motion demo.

## M1 — Manipulation Primitives ✅

Workspace-validated move-to-pose and straight-line Cartesian motion, gripper
primitives, structured primitive results, and randomized contact-grasp
validation.

## M2 — PickSkill + Local Recovery ✅

Explicit Pick finite-state machine, deterministic grasp candidate generation,
semantic success checks, and bounded skill-local recovery.

## M3 — PlaceSkill + SkillExecutor ✅

Explicit Place finite-state machine, release and target-state verification,
deterministic ordered skill sequencing, and structured `TaskResult` propagation.

## M4 — Closed-Loop Semantic Agent ✅

Goal → Plan → Validate → Execute → Observe → Compare → Replan, with serializable
`WorldState`, machine-readable Skill contracts, expected semantic effects,
`SemanticResidual`, bounded event-triggered replanning, and `AgentResult`.

## M5 — Dynamic World Playground ✅

Three cubes, two target trays and a temporary area; configurable geometric
relations; explicit external world modifications; semantic-step inspection;
reachable and unreachable target changes; occupied-target handling; structured
capability gaps; and deterministic dynamic-world scenarios.

### Occupied-target skill-composition exploration ✅

Geometry-derived `occupied_by` state and a bounded symbolic search demonstrate
that the planner can clear a destination by composing only the existing Pick
and Place skills. No recovery policy or simulator access was added to
`AgentRuntime`.

## M6 — Classical PushSkill ✅

Semantic `Push(object, target)` planning and validation; a dedicated
`right_side` push region; deterministic push geometry; an explicit Cartesian
Push FSM; displacement, region, and tabletop success checks; and one bounded
local contact-recovery attempt. The full Agent path and 20 randomized valid
placements are verified without adding simulator access above the controller
boundary.

## M7 — Hybrid Learned Skill

Experiment with learned physical implementations behind the same trusted Skill
interface and validation boundaries. One-step, progress-conditioned,
deterministic action-chunk BC, and a standard LeRobot ACT reproduction now
exist.

### Pluggable backend and imitation-data preparation ✅

`PushSkill` now delegates physical attempts to a `PushBackend`, with classical
and deliberately simple one-step BC implementations. A 20 Hz state/action NPZ
pipeline, episode-level split, action-window helper, NumPy MLP checkpoint, and
matched physical comparison establish the baseline. This is preparation for
the temporal-policy experiment below; ACT itself is not implemented.

### Temporal aliasing diagnosis and Chunk BC ✅

Dataset motion/transition/nearest-neighbor diagnostics, stratified offline
metrics, a scalar progress diagnosis, K=10/20/40 deterministic Chunk BC, and a
matched K=20 execution-horizon comparison are recorded in
`docs/chunk_bc_experiment.md`. Full-chunk H=20 reached 2/20 while H=1 and H=5
remained 0/20; the result supports temporal ambiguity and commitment as real
factors but does not establish a robust learned Push backend. Temporal
ensembling and ACT were deliberately not implemented.

### Standard LeRobot ACT PushBackend ✅

Hugging Face LeRobot 0.4.4 ACTPolicy was trained once on the unchanged 10D
state / 3D absolute-waypoint dataset and injected as `ACTPushBackend`. Its
K=32, eight-action queue produced smooth, safe EE motion but finished 0/20 on
the frozen matched states: all runs timed out before cube contact. The negative
result, exact training config, and no-expert/no-fallback runtime boundary are
recorded in `docs/act_push_backend.md`. No follow-up ablation was started.

## M8 — Skill Composition / Code-as-Skill

Explore dynamic composition of existing trusted skills without allowing
arbitrary generated robot-control code.
