"""Interactive closed-loop semantic-agent demo with optional LLM planning."""

from __future__ import annotations

import argparse
import json
import time

from evaluation.disturbances import ObjectLostAfterPick, TargetUnreachableAfterPick
from planner import (
    Goal,
    OpenAICompatiblePlanner,
    PlanValidator,
    RuleBasedPlanner,
    SkillRegistry,
)
from primitives import ManipulationPrimitives
from robot import PandaRobot
from runtime import AgentRuntime, SemanticSkillFactory
from sim import make_environment
from world import WorldModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", choices=("deterministic", "llm"), default="deterministic")
    parser.add_argument(
        "--disturbance",
        choices=("none", "object-lost", "target-unreachable"),
        default="none",
    )
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--randomize-cube", action="store_true")
    parser.add_argument("--inspect-seconds", type=float, default=8.0)
    args = parser.parse_args()

    env = make_environment(
        render=not args.no_render,
        randomize_cube=args.randomize_cube,
        seed=67,
    )
    try:
        env.reset()
        robot = PandaRobot(env)
        world = WorldModel(env)
        primitives = ManipulationPrimitives(robot)
        registry = SkillRegistry.standard()
        planner_backend = (
            OpenAICompatiblePlanner() if args.planner == "llm" else RuleBasedPlanner()
        )
        disturbance = None
        if args.disturbance == "object-lost":
            disturbance = ObjectLostAfterPick(env)
        elif args.disturbance == "target-unreachable":
            disturbance = TargetUnreachableAfterPick(env)

        runtime = AgentRuntime(
            planner=planner_backend,
            registry=registry,
            validator=PlanValidator(registry),
            observer=world.semantic_state,
            skill_factory=SemanticSkillFactory(world, primitives),
            logger=print,
        )
        goal = Goal.put_inside(
            "Put the red cube inside the blue target.",
            "red_cube",
            "blue_target",
        )
        result = runtime.run(goal, after_step=disturbance)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

        inspect_steps = round(max(0.0, args.inspect_seconds) * env.control_freq)
        for _ in range(inspect_steps):
            robot.hold(1)
            if not args.no_render:
                time.sleep(1.0 / env.control_freq)
        if not result.success:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
