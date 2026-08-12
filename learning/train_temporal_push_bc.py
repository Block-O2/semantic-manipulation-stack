"""Train Progress BC or deterministic Chunk BC on the Push dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_push_dataset, validate_push_dataset
from learning.temporal_push_bc import train_chunk_bc, train_progress_bc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("progress", "chunk"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prediction-horizon", type=int)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dimension", type=int)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    dataset = load_push_dataset(args.dataset)
    validation = validate_push_dataset(dataset)
    if not validation.valid:
        raise ValueError(f"dataset validation failed: {validation.errors}")
    started = time.perf_counter()
    common = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    if args.mode == "progress":
        policy, result = train_progress_bc(
            dataset, hidden_dimension=args.hidden_dimension or 64, **common
        )
        architecture = "MLP(11, 64, 64, 3) with ReLU"
    else:
        if args.prediction_horizon is None:
            parser.error("--prediction-horizon is required for chunk mode")
        policy, result = train_chunk_bc(
            dataset,
            prediction_horizon=args.prediction_horizon,
            hidden_dimension=args.hidden_dimension or 128,
            **common,
        )
        architecture = f"MLP(10, 128, 128, {3 * args.prediction_horizon}) with ReLU"
    config = {
        "mode": args.mode,
        "dataset": args.dataset,
        "architecture": architecture,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "prediction_horizon": args.prediction_horizon,
        "training_seconds": time.perf_counter() - started,
        "training_result": result.to_dict(),
    }
    policy.save(args.checkpoint, metadata=config)
    report = Path(args.checkpoint).with_suffix(".json")
    report.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
