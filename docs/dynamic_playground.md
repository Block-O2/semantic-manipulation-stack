# Dynamic World Playground

Milestone 5 turns the fixed Pick-and-Place scene into an inspectable world in
which external changes are explicit and the Agent must respond from fresh
semantic state.

## Scene

The deterministic default scene contains:

- `red_cube`, `green_cube`, and `blue_cube`;
- `red_target` and `blue_target`;
- `temporary_area`, which accepts the same normal Place execution as a target.

Cube positions can be randomized with the existing environment reset option.
World state reports each entity's pose, existence, and reachability. Targets
also expose the geometry-derived `occupied_by` object list. It derives
`inside`, `occupied`, `left_of`, `right_of`, and symmetric `near` relations
from geometry. Relation tolerances live in `SemanticThresholds`.

## Interactive use

On macOS, use `mjpython` for a rendered viewer:

```bash
mjpython -m demos.dynamic_playground
```

For a headless terminal session:

```bash
python -m demos.dynamic_playground --no-render --inspect-seconds 0
```

The demo pauses immediately before every semantic Skill. At the prompt:

```text
continue
state
move object red_cube -0.05 -0.10
move target blue_target 0.20 0.10
drop
occupy blue_target green_cube
remove red_cube
restore red_cube
help
quit
```

Every command produces a `WorldUpdate`; there are no hidden corrections. After
the hook returns, `AgentRuntime` observes again. A moved but reachable target
preserves valid Place preconditions and does not cause a replan. An unreachable
target, removed object, dropped grasp, or newly occupied target invalidates the
next semantic step and produces a structured failure or planner response.

Local Skill recovery and Agent replanning remain different mechanisms. A Pick
or Place Skill may retry its own bounded grasp, release, or Cartesian phase
without asking the planner. A world update that invalidates a semantic effect or
the next Skill's preconditions returns control to `AgentRuntime`, which records
the event and requests a new validated Plan.

Optional LLM planning uses the same deterministic validator and registry:

```bash
mjpython -m demos.dynamic_playground --planner llm
```

Configure `SEMANTIC_AGENT_MODEL`, `SEMANTIC_AGENT_API_KEY`, and optionally
`SEMANTIC_AGENT_BASE_URL` before selecting this mode.

## Deterministic scenarios

Run the complete scripted set:

```bash
python -m demos.dynamic_scenarios --scenario all
```

The scenarios cover:

- nominal Pick and Place;
- a dropped object followed by Agent-level replanning;
- a reachable target move with no unnecessary replan;
- an unreachable target move and clean refusal;
- an initially occupied target solved by composing four Pick / Place steps;
- target occupancy introduced after Pick, followed by replanning and a
  five-step rearrangement plan;
- a `push_to_edge` goal reported as a missing `push` capability.

## Supported goals and capability gaps

`put_inside(object, target)` is executable through registered Pick and Place
skills. The goal schema can also represent `left_of`, `near`, and
`push_to_edge`, so planners and interfaces do not have to encode these ideas as
free text. They are deliberately not executable in Milestone 5.

For `inside` goals, the deterministic planner performs a bounded breadth-first
search over legal Pick and Place transitions. If another movable object occupies
the requested target, the search can stage it in an empty target—preferring a
target whose semantic role is `temporary`—and then finish the original goal.
The search uses entity state and Skill preconditions rather than literal cube or
destination names.

When a request is semantically valid but unsupported, a planner returns a
`CANNOT` plan with `missing_capabilities` and a human-readable detail. This is a
normal, inspectable Agent result—not an exception and not permission to edit
MuJoCo state directly.

## Boundary rule

Normal manipulation remains:

```text
AgentRuntime → SkillExecutor → Skill → Primitive → PandaRobot
→ CartesianController → robosuite
```

`WorldController` exists beside this chain solely to model external scene
changes in demos and evaluation. Only `CartesianController` submits robosuite
robot control vectors.
