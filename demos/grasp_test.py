"""Run one explicit primitive-based contact grasp of the red cube."""

from __future__ import annotations

import argparse
import json
import time

from evaluation.grasp_validation import run_grasp_trial
from primitives import ManipulationPrimitives
from robot import PandaRobot
from sim import make_environment
from world import WorldModel


def run_demo(*, render: bool, randomize_cube: bool, inspect_seconds: float) -> bool:
    env = make_environment(render=render, randomize_cube=randomize_cube, seed=11)
    try:
        env.reset()
        robot = PandaRobot(env)
        world = WorldModel(env)
        primitives = ManipulationPrimitives(robot)
        result = run_grasp_trial(robot, world, primitives)
        print(json.dumps(result.as_log_record(1), indent=2, sort_keys=True))

        inspect_steps = round(max(0.0, inspect_seconds) * env.control_freq)
        for _ in range(inspect_steps):
            robot.hold(1)
            if render:
                time.sleep(1.0 / env.control_freq)
        return result.success
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--randomize-cube", action="store_true")
    parser.add_argument("--inspect-seconds", type=float, default=8.0)
    args = parser.parse_args()
    success = run_demo(
        render=not args.no_render,
        randomize_cube=args.randomize_cube,
        inspect_seconds=args.inspect_seconds,
    )
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
