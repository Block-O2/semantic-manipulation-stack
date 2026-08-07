"""Evaluate the complete deterministic PickSkill -> PlaceSkill task."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import numpy as np

from primitives import ManipulationPrimitives
from robot import PandaRobot
from runtime import SkillExecutor
from sim import make_environment
from skills import PickSkill, PlaceSkill
from world import WorldModel


def run_evaluation(*, trials: int, seed: int) -> dict[str, object]:
    if trials <= 0:
        raise ValueError("trials must be positive")

    env = make_environment(render=False, randomize_cube=True, seed=seed)
    full_successes = 0
    pick_successes = 0
    place_successes = 0
    first_attempt_successes = 0
    successful_recoveries = 0
    total_recovery_attempts = 0
    total_skill_attempts = 0
    total_completed_steps = 0
    failure_reasons: Counter[str] = Counter()

    try:
        robot = PandaRobot(env)
        world = WorldModel(env)
        primitives = ManipulationPrimitives(robot)
        executor = SkillExecutor()

        for trial in range(1, trials + 1):
            env.reset()
            initial_cube = world.pose("red_cube")
            initial_target = world.pose("blue_target")
            task_result = executor.execute(
                [
                    PickSkill("red_cube", world, primitives),
                    PlaceSkill("red_cube", "blue_target", world, primitives),
                ]
            )
            final_cube = world.pose("red_cube")
            skill_results = task_result.skill_results
            pick_success = len(skill_results) >= 1 and skill_results[0].success
            place_success = len(skill_results) >= 2 and skill_results[1].success
            full_success = task_result.success
            first_attempt = full_success and all(item.attempts == 1 for item in skill_results)

            full_successes += int(full_success)
            pick_successes += int(pick_success)
            place_successes += int(place_success)
            first_attempt_successes += int(first_attempt)
            total_completed_steps += task_result.completed_steps
            total_skill_attempts += sum(item.attempts for item in skill_results)
            total_recovery_attempts += sum(max(0, item.attempts - 1) for item in skill_results)
            successful_recoveries += sum(
                int(item.success and item.attempts > 1) for item in skill_results
            )
            if not task_result.success and task_result.reason is not None:
                failure_reasons[task_result.reason.value] += 1

            speed = float(np.linalg.norm(world.linear_velocity("red_cube")))
            print(
                json.dumps(
                    {
                        "trial": trial,
                        "cube_initial_position": initial_cube.position.round(5).tolist(),
                        "target_position": initial_target.position.round(5).tolist(),
                        "task_success": full_success,
                        "pick_success": pick_success,
                        "place_success": place_success,
                        "first_attempt_success": first_attempt,
                        "completed_steps": task_result.completed_steps,
                        "failed_step": task_result.failed_step,
                        "failed_skill": task_result.failed_skill,
                        "failure_reason": (
                            task_result.reason.value if task_result.reason else None
                        ),
                        "skill_attempts": [item.attempts for item in skill_results],
                        "skill_phases": [item.phase.value for item in skill_results],
                        "cube_final_position": final_cube.position.round(5).tolist(),
                        "cube_final_height": round(float(final_cube.position[2]), 5),
                        "cube_final_speed": round(speed, 6),
                        "object_grasped": world.is_grasped("red_cube"),
                        "object_inside_target": world.is_inside_target(
                            "red_cube", "blue_target"
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
        "full_successes": full_successes,
        "full_success_rate": full_successes / trials,
        "pick_successes": pick_successes,
        "pick_success_rate": pick_successes / trials,
        "place_successes": place_successes,
        "place_success_rate": place_successes / trials,
        "first_attempt_successes": first_attempt_successes,
        "first_attempt_success_rate": first_attempt_successes / trials,
        "successful_recoveries": successful_recoveries,
        "total_recovery_attempts": total_recovery_attempts,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "average_completed_steps_per_task": total_completed_steps / trials,
        "average_skill_attempts_per_task": total_skill_attempts / trials,
        "required_successes": required_successes,
        "target_met": full_successes >= required_successes,
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    summary = run_evaluation(trials=args.trials, seed=args.seed)
    if not summary["target_met"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
