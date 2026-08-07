"""Execute one semantic PickSkill for the red cube."""

from __future__ import annotations

import argparse
import json
import time

from primitives import ManipulationPrimitives
from robot import PandaRobot
from sim import make_environment
from skills import PickSkill
from world import WorldModel


def _result_record(result) -> dict[str, object]:
    return {
        "success": result.success,
        "skill": result.skill,
        "object_name": result.object_name,
        "phase": result.phase.value,
        "reason": result.reason.value if result.reason else None,
        "attempts": result.attempts,
        "primitive_result": (
            {
                "success": result.primitive_result.success,
                "reason": (
                    result.primitive_result.reason.value
                    if result.primitive_result.reason
                    else None
                ),
                "steps": result.primitive_result.steps,
                "position_error": result.primitive_result.final_position_error,
                "orientation_error": result.primitive_result.final_orientation_error,
            }
            if result.primitive_result
            else None
        ),
        "trace": list(result.trace),
    }


def run_demo(*, render: bool, randomize_cube: bool, inspect_seconds: float) -> bool:
    env = make_environment(render=render, randomize_cube=randomize_cube, seed=23)
    try:
        env.reset()
        robot = PandaRobot(env)
        world = WorldModel(env)
        primitives = ManipulationPrimitives(robot)
        skill = PickSkill("red_cube", world, primitives, logger=print)
        result = skill.execute()
        print(json.dumps(_result_record(result), indent=2, sort_keys=True))

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
