# Classical PushSkill

`Push(object, target)` is a semantic, nonprehensile manipulation capability.
The v1 implementation supports the dedicated `right_side` tabletop region. It
is classical Cartesian control; it does not use ACT, VLA, reinforcement
learning, force control, or a learned contact model.

## Execution path

```text
AgentRuntime
→ SkillExecutor
→ PushSkill
→ ManipulationPrimitives
→ PandaRobot
→ CartesianController
→ robosuite / MuJoCo
```

Only `CartesianController` constructs the seven-element robosuite action. The
planner selects a named region and never receives raw Cartesian push targets.

## Geometry and finite-state execution

`AxisAlignedCubePushPoseGenerator` observes the cube pose and region bounds,
then computes the push direction and pre-push, contact, end, and retreat poses.
The open gripper is offset laterally so one finger pad contacts the cube near
its centre of mass. The primary contact-to-end segment uses `move_linear`.

The nominal FSM is:

```text
CHECK_PRECONDITIONS
→ GENERATE_PUSH
→ MOVE_TO_SAFE_HEIGHT
→ OPEN_GRIPPER
→ MOVE_ABOVE_PREPUSH
→ DESCEND_TO_PUSH_HEIGHT
→ MOVE_TO_CONTACT
→ VERIFY_CONTACT
→ PUSH_LINEAR
→ RETREAT
→ VERIFY_SUCCESS
→ SUCCESS
```

Contact verification is a v1 geometric proximity check through `WorldModel`.
Success additionally requires at least 6 cm of XY object displacement, final
membership in the requested push region, and a final pose on the tabletop.

For `NO_PUSH_CONTACT`, `INSUFFICIENT_DISPLACEMENT`, or
`PUSH_TARGET_NOT_REACHED`, the Skill may perform one local recovery: retreat,
refresh the object pose, recompute geometry, and retry. Structural failures are
returned immediately as a structured `SkillResult`.

## Running and evaluating

```bash
python -m demos.push_test --no-render --inspect-seconds 0
python -m evaluation.push_trials --trials 20 --seed 127
```

On macOS, replace `python` with `mjpython` for the interactive viewer.

The randomized evaluator samples the full configured central range. Because
teleporting a cube into Panda's initialized link geometry creates an invalid
MuJoCo state, each sampled placement receives a short settling check and is
rejected if it moves by more than 3 mm or leaves the table before the Agent
runs. This validates 20 independent, physically valid initial placements
without narrowing the sampling bounds.

## Current limits

- only `right_side` is exposed as a public push destination;
- no obstacle or other-object reasoning;
- no curved, force-controlled, or closed-loop object-path pushing;
- contact uses proximity rather than raw MuJoCo contact pairs;
- goals are single predicates, so a Push-then-Pick/Place conjunction is not yet
  represented cleanly;
- no learned Push backend exists.

A future backend can implement the same semantic Skill contract and be selected
by the skill factory without changing planner validation or AgentRuntime.
