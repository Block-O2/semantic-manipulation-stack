"""Expert gates and ACT evaluations for Tissue and Draw."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_act_reproduction.experts import DrawExpert, TissueExpert
from agent_act_reproduction.robot import PandaCartesianAdapter
from agent_act_reproduction.runtime.execute_tasks import execute_command
from agent_act_reproduction.sim import make_draw_env, make_tissue_env


COMMANDS = {
    "tissue": "从纸巾盒抽出一张纸巾",
    "draw": "拿笔在纸上画一横",
}
SKILLS = {"tissue": "pull_tissue_from_box", "draw": "draw_horizontal_line"}


def evaluate_expert(*, task: str, trials: int, seed: int, perturbation_m: float) -> dict[str, object]:
    env_factory = make_tissue_env if task == "tissue" else make_draw_env
    expert_type = TissueExpert if task == "tissue" else DrawExpert
    env = env_factory(seed=seed)
    rows = []
    try:
        expert = expert_type(env, PandaCartesianAdapter(env))
        for index in range(trials):
            episode = expert.run(seed=seed + index, perturbation_m=perturbation_m)
            row = {"trial": index + 1, "seed": seed + index, "success": episode.success, "episode_length": episode.episode_length, "details": episode.success_details}
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
    finally:
        env.close()
    summary = {"task": task, "mode": "expert", "trials": trials, "successes": sum(int(row["success"]) for row in rows), "mean_episode_length": sum(int(row["episode_length"]) for row in rows) / trials, "perturbation_m": perturbation_m}
    print(json.dumps({"summary": summary}, sort_keys=True))
    return summary


def evaluate_act(*, task: str, checkpoint: Path, trials: int, seed: int, perturbation_m: float, device: str) -> dict[str, object]:
    rows = []
    for index in range(trials):
        result = execute_command(
            command=COMMANDS[task],
            checkpoints={SKILLS[task]: checkpoint},
            seed=seed + index,
            perturbation_m=perturbation_m,
            device=device,
            verbose=False,
        )
        row = {"trial": index + 1, "seed": seed + index, "success": result["success"], "episode_length": result["episode_length"], "details": result["success_details"]}
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    summary = {"task": task, "mode": "act", "trials": trials, "successes": sum(int(row["success"]) for row in rows), "mean_episode_length": sum(int(row["episode_length"]) for row in rows) / trials, "perturbation_m": perturbation_m, "checkpoint": str(checkpoint.resolve())}
    print(json.dumps({"summary": summary}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("tissue", "draw"))
    parser.add_argument("--mode", choices=("expert", "act"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--perturbation-m", type=float, default=0.006)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.mode == "expert":
        evaluate_expert(task=args.task, trials=args.trials, seed=args.seed, perturbation_m=args.perturbation_m)
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for --mode act")
        evaluate_act(task=args.task, checkpoint=args.checkpoint, trials=args.trials, seed=args.seed, perturbation_m=args.perturbation_m, device=args.device)


if __name__ == "__main__":
    main()
