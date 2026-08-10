"""Physical nominal, small-perturbation, and OOD ACT evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agent_act_reproduction.runtime.execute import execute_command


def evaluate(
    *,
    checkpoint: Path,
    trials: int,
    seed: int,
    perturbation_m: float,
    device: str,
    label: str,
) -> dict[str, object]:
    successes = 0
    lengths: list[int] = []
    failures: Counter[str] = Counter()
    for index in range(trials):
        result = execute_command(
            command="Put the bottle on the shelf.",
            checkpoint=checkpoint,
            seed=seed + index,
            perturbation_m=perturbation_m,
            device=device,
            render=False,
            verbose=False,
        )
        successes += int(bool(result["success"]))
        lengths.append(int(result["episode_length"]))
        if not result["success"]:
            details = result["success_details"]
            for key in ("inside_shelf_xy", "released", "height_ok", "stable"):
                if not details[key]:
                    failures[key] += 1
        print(
            json.dumps(
                {
                    "label": label,
                    "trial": index + 1,
                    "seed": seed + index,
                    "success": result["success"],
                    "episode_length": result["episode_length"],
                    "final_bottle_position": result["final_bottle_position"],
                    "success_details": result["success_details"],
                },
                sort_keys=True,
            )
        )
    summary = {
        "label": label,
        "perturbation_m": perturbation_m,
        "trials": trials,
        "successes": successes,
        "success_rate": successes / trials,
        "mean_episode_length": sum(lengths) / len(lengths),
        "failure_distribution": dict(sorted(failures.items())),
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/bottle_act.pt"))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--perturbation-m", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--label", default="small_perturbation")
    args = parser.parse_args()
    evaluate(
        checkpoint=args.checkpoint,
        trials=args.trials,
        seed=args.seed,
        perturbation_m=args.perturbation_m,
        device=args.device,
        label=args.label,
    )


if __name__ == "__main__":
    main()
