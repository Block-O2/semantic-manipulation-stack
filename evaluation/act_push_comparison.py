"""Evaluate LeRobot ACT on the frozen replay-stable Push initial states."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from datasets import load_push_dataset
from evaluation.push_backend_comparison import run_backend_trial
from runtime import create_push_backend


MATCHED_DATASET_INDICES = (
    0, 1, 2, 3, 4, 5, 6, 8, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 22, 23,
)


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    failures = Counter(
        str(record["failure_reason"])
        for record in records
        if not record["success"]
    )
    return {
        "trials": len(records),
        "successes": sum(bool(record["success"]) for record in records),
        "failure_reasons": dict(sorted(failures.items())),
        "mean_control_steps": float(
            np.mean([record["control_steps"] for record in records])
        ),
        "mean_cube_displacement_m": float(
            np.mean([record["displacement"] for record in records])
        ),
        "target_satisfied": sum(bool(record["target_satisfied"]) for record in records),
        "unsafe_action_count": failures.get("POLICY_ACTION_UNSAFE", 0),
        "timeout_count": failures.get("POLICY_TIMEOUT", 0),
        "mean_ee_path_length_m": float(
            np.mean([record["ee_path_length_m"] for record in records])
        ),
        "mean_act_inference_calls": float(
            np.mean([record["act_inference_calls"] for record in records])
        ),
        "mean_predicted_action_magnitude_m": float(
            np.mean([
                record["mean_predicted_action_magnitude_m"] for record in records
            ])
        ),
        "mean_max_predicted_action_magnitude_m": float(
            np.mean([
                record["max_predicted_action_magnitude_m"] for record in records
            ])
        ),
        "maximum_predicted_action_magnitude_m": float(
            np.max([
                record["max_predicted_action_magnitude_m"] for record in records
            ])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/push_demos_50_seed123.npz")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/push_act_seed17.pt")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/act_push/physical_comparison.json"),
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=Path("artifacts/act_push/trajectories"),
    )
    args = parser.parse_args()
    dataset = load_push_dataset(args.dataset)
    backend = create_push_backend(
        "act", checkpoint=str(args.checkpoint), device=args.device
    )
    records: list[dict[str, object]] = []
    args.trajectory_dir.mkdir(parents=True, exist_ok=True)
    for trial, dataset_index in enumerate(MATCHED_DATASET_INDICES, start=1):
        episode = dataset.episodes[dataset_index]
        initial_xy = np.asarray(episode.metadata["initial_cube_position"][:2])
        seed = int(episode.metadata["random_seed"])
        record = run_backend_trial(backend=backend, initial_xy=initial_xy, seed=seed)
        rollout = backend.last_rollout_log
        magnitudes = [float(row["predicted_step_m"]) for row in rollout]
        record.update({
            "trial": trial,
            "dataset_index": dataset_index,
            "seed": seed,
            "ee_path_length_m": (
                float(rollout[-1]["ee_path_length_m"]) if rollout else 0.0
            ),
            "mean_predicted_action_magnitude_m": (
                float(np.mean(magnitudes)) if magnitudes else 0.0
            ),
            "max_predicted_action_magnitude_m": (
                float(np.max(magnitudes)) if magnitudes else 0.0
            ),
            "act_inference_calls": int(backend.policy.inference_calls),
        })
        records.append(record)
        sampled = rollout[::5]
        if rollout and sampled[-1] is not rollout[-1]:
            sampled.append(rollout[-1])
        (args.trajectory_dir / f"trial_{trial:02d}.json").write_text(
            json.dumps(sampled, indent=2) + "\n"
        )
        print(json.dumps(record, sort_keys=True), flush=True)
    report = {
        "dataset_indices": list(MATCHED_DATASET_INDICES),
        "records": records,
        "summary": summarize(records),
        "reference_results": {
            "classical": "20/20",
            "one_step_bc": "0/20",
            "simple_chunk_bc_k20_h20": "2/20",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": report["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
