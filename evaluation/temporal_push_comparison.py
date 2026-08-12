"""Frozen diagnostic comparison for progress and simple Chunk BC variants."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from datasets import load_push_dataset
from evaluation.push_backend_comparison import run_backend_trial
from runtime import create_push_backend


def summarize(records: list[dict[str, object]], backend_name: str) -> dict[str, object]:
    selected = [record for record in records if record["backend"] == backend_name]
    failures = Counter(
        str(record["failure_reason"])
        for record in selected
        if not record["success"]
    )
    return {
        "trials": len(selected),
        "successes": sum(bool(record["success"]) for record in selected),
        "failure_reasons": dict(sorted(failures.items())),
        "mean_control_steps": float(np.mean([record["control_steps"] for record in selected])),
        "mean_cube_displacement_m": float(np.mean([record["displacement"] for record in selected])),
        "target_satisfied": sum(bool(record["target_satisfied"]) for record in selected),
        "unsafe_action_count": failures.get("POLICY_ACTION_UNSAFE", 0),
        "timeout_count": failures.get("POLICY_TIMEOUT", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--one-step-checkpoint", required=True)
    parser.add_argument("--progress-checkpoint", required=True)
    parser.add_argument("--chunk-checkpoint", required=True)
    parser.add_argument("--execution-horizons", type=int, nargs="+", required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_push_dataset(args.dataset)
    backends = [
        create_push_backend("classical"),
        create_push_backend("bc", checkpoint=args.one_step_checkpoint),
        create_push_backend("progress_bc", checkpoint=args.progress_checkpoint),
        *[
            create_push_backend(
                "chunk_bc",
                checkpoint=args.chunk_checkpoint,
                execution_horizon=horizon,
            )
            for horizon in args.execution_horizons
        ],
    ]
    if len({backend.name for backend in backends}) != len(backends):
        raise ValueError("backend names must be unique")
    records: list[dict[str, object]] = []
    skipped = 0
    args.trajectory_dir.mkdir(parents=True, exist_ok=True)
    completed_trials = 0
    for dataset_index, episode in enumerate(dataset.episodes):
        if completed_trials >= args.trials:
            break
        initial_xy = np.asarray(episode.metadata["initial_cube_position"][:2])
        seed = int(episode.metadata["random_seed"])
        trial_records = []
        try:
            for backend in backends:
                record = run_backend_trial(backend=backend, initial_xy=initial_xy, seed=seed)
                record.update({"trial": completed_trials + 1, "dataset_index": dataset_index, "seed": seed})
                trial_records.append(record)
                rollout_log = getattr(backend, "last_rollout_log", None)
                if rollout_log is not None and completed_trials < 3:
                    path = args.trajectory_dir / f"trial_{completed_trials + 1:02d}_{backend.name}.json"
                    sampled = rollout_log[::5]
                    if rollout_log and sampled[-1] is not rollout_log[-1]:
                        sampled.append(rollout_log[-1])
                    path.write_text(json.dumps(sampled, indent=2) + "\n")
        except RuntimeError as exc:
            skipped += 1
            print(json.dumps({"skipped_dataset_index": dataset_index, "reason": str(exc)}))
            continue
        records.extend(trial_records)
        completed_trials += 1
        for record in trial_records:
            print(json.dumps(record, sort_keys=True))
    if completed_trials != args.trials:
        raise RuntimeError("dataset did not contain enough replay-stable matched states")
    report = {
        "matched_trials": completed_trials,
        "skipped_initial_states": skipped,
        "dataset_indices": sorted({int(record["dataset_index"]) for record in records}),
        "records": records,
        "summary": {backend.name: summarize(records, backend.name) for backend in backends},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": report["summary"], "skipped_initial_states": skipped}, sort_keys=True))


if __name__ == "__main__":
    main()
