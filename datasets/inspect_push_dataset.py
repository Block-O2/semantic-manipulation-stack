"""Validate and inspect deterministic samples from a local Push dataset."""

from __future__ import annotations

import argparse
import json

import numpy as np

from datasets import load_push_dataset, validate_push_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    dataset = load_push_dataset(args.dataset)
    report = validate_push_dataset(dataset)
    rng = np.random.default_rng(args.seed)
    count = min(max(args.samples, 0), len(dataset.episodes))
    indices = rng.choice(len(dataset.episodes), size=count, replace=False)
    payload = {
        "valid": report.valid,
        "errors": list(report.errors),
        "statistics": report.statistics,
        "sampled_episodes": [
            {
                "index": int(index),
                "metadata": dataset.episodes[int(index)].metadata,
                "first_observation": dataset.episodes[int(index)]
                .observations[0]
                .tolist(),
                "first_action": dataset.episodes[int(index)].actions[0].tolist(),
                "last_observation": dataset.episodes[int(index)]
                .observations[-1]
                .tolist(),
                "last_action": dataset.episodes[int(index)].actions[-1].tolist(),
            }
            for index in indices
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not report.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
