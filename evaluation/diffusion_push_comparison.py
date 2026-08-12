"""Evaluate state-based Diffusion Policy on the frozen 20 Push states once."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from datasets import load_push_dataset
from evaluation.act_push_comparison import MATCHED_DATASET_INDICES
from evaluation.push_backend_comparison import run_backend_trial
from runtime import create_push_backend


CONTACT_PROXIMITY_M = 0.07


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    failures = Counter(
        str(record["failure_reason"])
        for record in records
        if not record["success"]
    )
    latencies = [
        float(value)
        for record in records
        for value in record["diffusion_inference_latencies_s"]
    ]
    return {
        "trials": len(records),
        "successes": sum(bool(record["success"]) for record in records),
        "failure_reasons": dict(sorted(failures.items())),
        "target_satisfied": sum(bool(record["target_satisfied"]) for record in records),
        "unsafe_action_count": failures.get("POLICY_ACTION_UNSAFE", 0),
        "timeout_count": failures.get("POLICY_TIMEOUT", 0),
        "mean_control_steps": float(np.mean([record["control_steps"] for record in records])),
        "mean_cube_displacement_m": float(np.mean([record["displacement"] for record in records])),
        "mean_minimum_ee_cube_distance_m": float(
            np.mean([record["minimum_ee_cube_distance_m"] for record in records])
        ),
        "minimum_ee_cube_distance_m": float(
            np.min([record["minimum_ee_cube_distance_m"] for record in records])
        ),
        "proximity_reached_trials": sum(bool(record["proximity_reached"]) for record in records),
        "mean_ee_path_length_m": float(np.mean([record["ee_path_length_m"] for record in records])),
        "mean_diffusion_inference_calls": float(
            np.mean([record["diffusion_inference_calls"] for record in records])
        ),
        "mean_predicted_step_m": float(np.mean([record["mean_predicted_step_m"] for record in records])),
        "mean_executed_step_m": float(np.mean([record["mean_executed_step_m"] for record in records])),
        "mean_diffusion_inference_latency_s": float(np.mean(latencies)),
        "p95_diffusion_inference_latency_s": float(np.percentile(latencies, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/push_demos_50_seed123.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/push_diffusion_seed17.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/diffusion_push/physical_comparison.json"),
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=Path("artifacts/diffusion_push/representative_trajectory.json"),
    )
    args = parser.parse_args()
    checkpoint_hash_before = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    dataset = load_push_dataset(args.dataset)
    backend = create_push_backend(
        "diffusion", checkpoint=str(args.checkpoint), device=args.device
    )
    records: list[dict[str, object]] = []
    best_rollout: list[dict[str, object]] = []
    best_key = (-1.0, float("-inf"))
    for trial, dataset_index in enumerate(MATCHED_DATASET_INDICES, start=1):
        episode = dataset.episodes[dataset_index]
        initial_xy = np.asarray(episode.metadata["initial_cube_position"][:2])
        seed = int(episode.metadata["random_seed"])
        record = run_backend_trial(backend=backend, initial_xy=initial_xy, seed=seed)
        rollout = backend.last_rollout_log
        distances = [
            float(np.linalg.norm(np.asarray(row["ee_xyz"]) - np.asarray(row["cube_xyz"])))
            for row in rollout
        ]
        predicted_steps = [float(row["predicted_step_m"]) for row in rollout]
        executed_steps = [float(row["executed_step_m"]) for row in rollout]
        minimum_distance = min(distances) if distances else float("inf")
        record.update({
            "trial": trial,
            "dataset_index": dataset_index,
            "seed": seed,
            "minimum_ee_cube_distance_m": minimum_distance,
            "proximity_threshold_m": CONTACT_PROXIMITY_M,
            "proximity_reached": minimum_distance <= CONTACT_PROXIMITY_M,
            "ee_path_length_m": float(rollout[-1]["ee_path_length_m"]) if rollout else 0.0,
            "mean_predicted_step_m": float(np.mean(predicted_steps)) if predicted_steps else 0.0,
            "mean_executed_step_m": float(np.mean(executed_steps)) if executed_steps else 0.0,
            "diffusion_inference_calls": int(backend.policy.inference_calls),
            "diffusion_inference_latencies_s": [
                float(value) for value in backend.policy.inference_latencies
            ],
        })
        records.append(record)
        key = (float(record["displacement"]), -minimum_distance)
        if key > best_key:
            best_key = key
            best_rollout = list(rollout)
        print(json.dumps(record, sort_keys=True), flush=True)
    sampled = best_rollout[::5]
    if best_rollout and sampled[-1] is not best_rollout[-1]:
        sampled.append(best_rollout[-1])
    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    args.trajectory_output.write_text(json.dumps(sampled, indent=2) + "\n")
    checkpoint_hash_after = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report = {
        "configuration": {
            "implementation": "state-based Diffusion Policy conditional 1D U-Net",
            "architecture": backend.policy.architecture_dict(backend.policy.architecture),
            "checkpoint_sha256_before": checkpoint_hash_before,
            "checkpoint_sha256_after": checkpoint_hash_after,
            "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after,
            "training_metadata": backend.policy.metadata,
        },
        "dataset_indices": list(MATCHED_DATASET_INDICES),
        "records": records,
        "summary": summarize(records),
        "reference_results": {
            "classical": "20/20",
            "one_step_bc": "0/20",
            "simple_chunk_bc_k20_h20": "2/20",
            "act_queue_k32_h8": "0/20",
            "act_native_temporal_ensemble_k32_h1": "20/20",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": report["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
