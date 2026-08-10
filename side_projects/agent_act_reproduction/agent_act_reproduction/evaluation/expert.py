"""Reliability evaluation for the demonstration-only Bottle expert."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from agent_act_reproduction.experts import BottleExpert
from agent_act_reproduction.robot import PandaCartesianAdapter
from agent_act_reproduction.sim import make_bottle_env


def evaluate_expert(*, trials: int, seed: int, perturbation_m: float) -> dict[str, object]:
    env = make_bottle_env(render=False, seed=seed)
    successes = 0
    lengths: list[int] = []
    failures: Counter[str] = Counter()
    try:
        expert = BottleExpert(env, PandaCartesianAdapter(env))
        for index in range(trials):
            episode = expert.run(seed=seed + index, perturbation_m=perturbation_m)
            successes += int(episode.success)
            lengths.append(episode.episode_length)
            if not episode.success:
                for key in ("inside_shelf_xy", "released", "height_ok", "stable"):
                    if not bool(episode.success_details[key]):
                        failures[key] += 1
            print(
                json.dumps(
                    {
                        "trial": index + 1,
                        "seed": episode.seed,
                        "success": episode.success,
                        "episode_length": episode.episode_length,
                        "initial_bottle_pose": episode.initial_bottle_pose,
                        "final_bottle_pose": episode.final_bottle_pose,
                        "success_details": episode.success_details,
                    },
                    sort_keys=True,
                )
            )
    finally:
        env.close()

    summary = {
        "trials": trials,
        "successes": successes,
        "success_rate": successes / trials,
        "mean_episode_length": sum(lengths) / len(lengths),
        "min_episode_length": min(lengths),
        "max_episode_length": max(lengths),
        "failure_distribution": dict(sorted(failures.items())),
        "target_18_of_20_met": trials != 20 or successes >= 18,
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--perturbation-m", type=float, default=0.01)
    args = parser.parse_args()
    summary = evaluate_expert(
        trials=args.trials,
        seed=args.seed,
        perturbation_m=args.perturbation_m,
    )
    if not summary["target_18_of_20_met"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
