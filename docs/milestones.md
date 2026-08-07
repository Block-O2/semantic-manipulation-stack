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

## M5 — Dynamic World Playground

Explore interactive disturbances, richer spatial relations, explicit capability
gaps, and systematic disturbance evaluation.

## M6 — PushSkill

Explore a classical, deterministic push implementation behind the existing
Skill and primitive boundaries.

## M7 — Hybrid Learned Skill

Experiment with a learned policy, such as ACT, behind the same trusted Skill
interface and validation boundaries.

## M8 — Skill Composition / Code-as-Skill

Explore dynamic composition of existing trusted skills without allowing
arbitrary generated robot-control code.
