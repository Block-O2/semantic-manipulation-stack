"""Nominal randomized evaluation of the closed-loop semantic AgentRuntime."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter

from planner import Goal, PlanValidator, RuleBasedPlanner, SkillRegistry
from primitives import ManipulationPrimitives
from robot import PandaRobot
from runtime import AgentRuntime, SemanticSkillFactory
from sim import make_environment
from world import WorldModel


GOAL = Goal.put_inside(
    "Put the red cube inside the blue target.",
    "red_cube",
    "blue_target",
)


def run_evaluation(*, trials: int, seed: int) -> dict[str, object]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    env = make_environment(render=False, randomize_cube=True, seed=seed)
    successes = 0
    planner_calls = 0
    replans = 0
    failure_reasons: Counter[str] = Counter()
    try:
        robot = PandaRobot(env)
        world = WorldModel(env)
        primitives = ManipulationPrimitives(robot)
        registry = SkillRegistry.standard()
        for trial in range(1, trials + 1):
            env.reset()
            initial = world.semantic_state()
            runtime = AgentRuntime(
                planner=RuleBasedPlanner(),
                registry=registry,
                validator=PlanValidator(registry),
                observer=world.semantic_state,
                skill_factory=SemanticSkillFactory(world, primitives),
            )
            result = runtime.run(GOAL)
            successes += int(result.success)
            planner_calls += result.planner_calls
            replans += result.replans
            if result.failure_reason:
                failure_reasons[result.failure_reason.value] += 1
            print(
                json.dumps(
                    {
                        "trial": trial,
                        "cube_initial_position": initial.objects[
                            "red_cube"
                        ].pose.position,
                        "success": result.success,
                        "planner_calls": result.planner_calls,
                        "replans": result.replans,
                        "executed_steps": len(result.executed_steps),
                        "goal_satisfied": result.final_world_state.relations[
                            "red_cube_inside_blue_target"
                        ],
                        "failure_reason": (
                            result.failure_reason.value if result.failure_reason else None
                        ),
                    },
                    sort_keys=True,
                )
            )
    finally:
        env.close()

    required_successes = math.ceil(0.9 * trials)
    summary: dict[str, object] = {
        "trials": trials,
        "successes": successes,
        "success_rate": successes / trials,
        "average_planner_calls": planner_calls / trials,
        "total_replans": replans,
        "average_replans": replans / trials,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "required_successes": required_successes,
        "target_met": successes >= required_successes,
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=59)
    args = parser.parse_args()
    summary = run_evaluation(trials=args.trials, seed=args.seed)
    if not summary["target_met"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
