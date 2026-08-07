"""Run repeated randomized contact-grasp validation trials."""

from __future__ import annotations

import argparse
import json

from evaluation.grasp_validation import GraspParameters, run_grasp_trial
from primitives import ManipulationPrimitives
from robot import PandaRobot
from sim import make_environment
from world import WorldModel


def run_evaluation(*, trials: int, seed: int) -> tuple[int, int]:
    if trials <= 0:
        raise ValueError("trials must be positive")

    env = make_environment(render=False, randomize_cube=True, seed=seed)
    successes = 0
    try:
        robot = PandaRobot(env)
        world = WorldModel(env)
        primitives = ManipulationPrimitives(robot)
        parameters = GraspParameters()

        for trial in range(1, trials + 1):
            env.reset()
            result = run_grasp_trial(
                robot,
                world,
                primitives,
                parameters=parameters,
            )
            successes += int(result.success)
            print(json.dumps(result.as_log_record(trial), sort_keys=True))
    finally:
        env.close()

    print(
        json.dumps(
            {
                "summary": {
                    "successes": successes,
                    "trials": trials,
                    "success_rate": successes / trials,
                    "target_met": successes >= 0.9 * trials,
                }
            },
            sort_keys=True,
        )
    )
    return successes, trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    successes, trials = run_evaluation(trials=args.trials, seed=args.seed)
    if successes < 0.9 * trials:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
