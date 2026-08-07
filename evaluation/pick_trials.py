"""Evaluate PickSkill over randomized red-cube poses."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from primitives import ManipulationPrimitives
from robot import PandaRobot
from sim import make_environment
from skills import PickSkill
from world import WorldModel


def run_evaluation(*, trials: int, seed: int) -> dict[str, object]:
    if trials <= 0:
        raise ValueError("trials must be positive")

    env = make_environment(render=False, randomize_cube=True, seed=seed)
    successes = 0
    first_attempt_successes = 0
    successful_recoveries = 0
    successful_attempts: list[int] = []
    failure_reasons: Counter[str] = Counter()
    try:
        robot = PandaRobot(env)
        world = WorldModel(env)
        primitives = ManipulationPrimitives(robot)
        for trial in range(1, trials + 1):
            env.reset()
            initial_cube = world.pose("red_cube")
            result = PickSkill("red_cube", world, primitives).execute()
            final_cube = world.pose("red_cube")

            successes += int(result.success)
            first_attempt_successes += int(result.success and result.attempts == 1)
            successful_recoveries += int(result.success and result.attempts > 1)
            if result.success:
                successful_attempts.append(result.attempts)
            elif result.reason is not None:
                failure_reasons[result.reason.value] += 1

            print(
                json.dumps(
                    {
                        "trial": trial,
                        "cube_initial_position": initial_cube.position.round(5).tolist(),
                        "success": result.success,
                        "attempts": result.attempts,
                        "phase": result.phase.value,
                        "reason": result.reason.value if result.reason else None,
                        "final_cube_height": round(float(final_cube.position[2]), 5),
                        "trace": list(result.trace),
                    },
                    sort_keys=True,
                )
            )
    finally:
        env.close()

    summary: dict[str, object] = {
        "trials": trials,
        "successes": successes,
        "success_rate": successes / trials,
        "first_attempt_successes": first_attempt_successes,
        "first_attempt_success_rate": first_attempt_successes / trials,
        "successful_recoveries": successful_recoveries,
        "failed_trials": trials - successes,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "average_attempts_per_success": (
            sum(successful_attempts) / len(successful_attempts)
            if successful_attempts
            else None
        ),
        "target_met": successes >= 0.9 * trials,
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    summary = run_evaluation(trials=args.trials, seed=args.seed)
    if not summary["target_met"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
