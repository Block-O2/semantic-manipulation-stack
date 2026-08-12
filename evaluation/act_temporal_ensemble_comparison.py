"""Evaluate native LeRobot ACT temporal ensembling on frozen Push states."""

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
    return {
        "trials": len(records),
        "successes": sum(bool(record["success"]) for record in records),
        "failure_reasons": dict(sorted(failures.items())),
        "target_satisfied": sum(bool(record["target_satisfied"]) for record in records),
        "unsafe_action_count": failures.get("POLICY_ACTION_UNSAFE", 0),
        "timeout_count": failures.get("POLICY_TIMEOUT", 0),
        "mean_control_steps": float(
            np.mean([record["control_steps"] for record in records])
        ),
        "mean_cube_displacement_m": float(
            np.mean([record["displacement"] for record in records])
        ),
        "mean_minimum_ee_cube_distance_m": float(
            np.mean([record["minimum_ee_cube_distance_m"] for record in records])
        ),
        "minimum_ee_cube_distance_m": float(
            np.min([record["minimum_ee_cube_distance_m"] for record in records])
        ),
        "proximity_reached_trials": sum(
            bool(record["proximity_reached"]) for record in records
        ),
        "mean_ee_path_length_m": float(
            np.mean([record["ee_path_length_m"] for record in records])
        ),
        "mean_realized_ee_step_m": float(
            np.mean([
                float(record["ee_path_length_m"]) / int(record["act_inference_calls"])
                for record in records
            ])
        ),
        "mean_act_inference_calls": float(
            np.mean([record["act_inference_calls"] for record in records])
        ),
        "mean_predicted_step_m": float(
            np.mean([record["mean_predicted_step_m"] for record in records])
        ),
        "mean_executed_step_m": float(
            np.mean([record["mean_executed_step_m"] for record in records])
        ),
        "mean_overlap_count": float(
            np.mean([record["mean_overlap_count"] for record in records])
        ),
        "maximum_overlap_count": int(
            np.max([record["maximum_overlap_count"] for record in records])
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
        default=Path("artifacts/act_temporal_ensemble/physical_comparison.json"),
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=Path("artifacts/act_temporal_ensemble/representative_trajectory.json"),
    )
    args = parser.parse_args()
    checkpoint_hash_before = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    dataset = load_push_dataset(args.dataset)
    backend = create_push_backend(
        "act",
        checkpoint=str(args.checkpoint),
        device=args.device,
        act_execution_mode="temporal_ensemble",
        temporal_ensemble_coeff=0.01,
    )
    records: list[dict[str, object]] = []
    best_rollout: list[dict[str, object]] = []
    best_key = (-1.0, float("-inf"))
    mechanism_diagnostic: list[dict[str, object]] = []
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
        overlap_counts = [int(row["ensemble_overlap_count"]) for row in rollout]
        minimum_distance = min(distances) if distances else float("inf")
        ee_path_length = float(rollout[-1]["ee_path_length_m"]) if rollout else 0.0
        record.update({
            "trial": trial,
            "dataset_index": dataset_index,
            "seed": seed,
            "minimum_ee_cube_distance_m": minimum_distance,
            "proximity_threshold_m": CONTACT_PROXIMITY_M,
            "proximity_reached": minimum_distance <= CONTACT_PROXIMITY_M,
            "ee_path_length_m": ee_path_length,
            "mean_predicted_step_m": (
                float(np.mean(predicted_steps)) if predicted_steps else 0.0
            ),
            "mean_executed_step_m": (
                float(np.mean(executed_steps)) if executed_steps else 0.0
            ),
            "act_inference_calls": int(backend.policy.inference_calls),
            "mean_overlap_count": (
                float(np.mean(overlap_counts)) if overlap_counts else 0.0
            ),
            "maximum_overlap_count": max(overlap_counts, default=0),
        })
        records.append(record)
        if trial == 1:
            mechanism_diagnostic = list(backend.policy.ensemble_diagnostics)
        key = (float(record["displacement"]), -minimum_distance)
        if key > best_key:
            best_key = key
            best_rollout = rollout
        print(json.dumps(record, sort_keys=True), flush=True)
    sampled = best_rollout[::5]
    if best_rollout and sampled[-1] is not best_rollout[-1]:
        sampled.append(best_rollout[-1])
    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    args.trajectory_output.write_text(json.dumps(sampled, indent=2) + "\n")
    checkpoint_hash_after = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report = {
        "configuration": {
            "implementation": "LeRobot ACTTemporalEnsembler",
            "temporal_ensemble_coeff": 0.01,
            "chunk_size": 32,
            "n_action_steps": 1,
            "checkpoint_sha256_before": checkpoint_hash_before,
            "checkpoint_sha256_after": checkpoint_hash_after,
            "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after,
        },
        "dataset_indices": list(MATCHED_DATASET_INDICES),
        "mechanism_diagnostic": mechanism_diagnostic,
        "records": records,
        "summary": summarize(records),
        "queue_act_reference": {
            "successes": 0,
            "trials": 20,
            "mean_cube_displacement_m": 8.673617379884036e-20,
            "mean_ee_path_length_m": 0.08600969628778307,
            "mean_act_inference_calls": 82.0,
            "mean_predicted_step_m": 0.0008842529044604507,
            "unsafe_action_count": 0,
            "timeout_count": 20,
            "proximity_threshold_m": CONTACT_PROXIMITY_M,
            "proximity_reached_trials": 0,
            "sampled_minimum_ee_cube_distance_m": 0.33207988522827536,
            "sampled_mean_minimum_ee_cube_distance_m": 0.42773962555978207,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": report["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
