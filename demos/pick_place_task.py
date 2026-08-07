"""Run one deterministic PickSkill -> PlaceSkill task."""

from __future__ import annotations

import argparse
import json
import time

from primitives import ManipulationPrimitives
from robot import PandaRobot
from runtime import SkillExecutor, TaskResult
from sim import make_environment
from skills import PickSkill, PlaceSkill, SkillResult
from world import WorldModel


def _skill_record(result: SkillResult) -> dict[str, object]:
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


def _task_record(result: TaskResult) -> dict[str, object]:
    return {
        "success": result.success,
        "completed_steps": result.completed_steps,
        "failed_step": result.failed_step,
        "failed_skill": result.failed_skill,
        "reason": result.reason.value if result.reason else None,
        "skill_results": [_skill_record(item) for item in result.skill_results],
        "trace": list(result.trace),
    }


def run_demo(*, render: bool, randomize_cube: bool, inspect_seconds: float) -> bool:
    env = make_environment(render=render, randomize_cube=randomize_cube, seed=41)
    try:
        env.reset()
        robot = PandaRobot(env)
        world = WorldModel(env)
        primitives = ManipulationPrimitives(robot)

        print(
            json.dumps(
                {
                    "event": "START_STATE",
                    "cube_position": world.pose("red_cube").position.round(5).tolist(),
                    "target_position": world.pose("blue_target").position.round(5).tolist(),
                },
                sort_keys=True,
            )
        )
        executor = SkillExecutor(logger=print)
        task_result = executor.execute(
            [
                PickSkill("red_cube", world, primitives, logger=print),
                PlaceSkill("red_cube", "blue_target", world, primitives, logger=print),
            ]
        )
        final_pose = world.pose("red_cube")
        output = _task_record(task_result)
        output["final_state"] = {
            "cube_position": final_pose.position.round(5).tolist(),
            "cube_speed": round(float((world.linear_velocity("red_cube") ** 2).sum() ** 0.5), 6),
            "is_grasped": world.is_grasped("red_cube"),
            "inside_target": world.is_inside_target("red_cube", "blue_target"),
            "stable": world.is_stable("red_cube"),
        }
        print(json.dumps(output, indent=2, sort_keys=True))

        inspect_steps = round(max(0.0, inspect_seconds) * env.control_freq)
        for _ in range(inspect_steps):
            robot.hold(1)
            if render:
                time.sleep(1.0 / env.control_freq)
        return task_result.success
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
