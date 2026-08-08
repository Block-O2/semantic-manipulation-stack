"""Train and offline-evaluate the NumPy one-step Push BC baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_push_dataset, validate_push_dataset
from learning import train_one_step_bc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/push_bc.npz")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dimension", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    dataset = load_push_dataset(args.dataset)
    validation = validate_push_dataset(dataset)
    if not validation.valid:
        raise ValueError(f"dataset validation failed: {validation.errors}")
    policy, result = train_one_step_bc(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dimension=args.hidden_dimension,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    config = {
        "dataset": str(args.dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dimension": args.hidden_dimension,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "architecture": "MLP(10, hidden, hidden, 3) with ReLU",
        "action_convention": "absolute world-frame desired EE xyz in metres",
        "training_result": result.to_dict(),
    }
    policy.save(args.checkpoint, metadata=config)
    report_path = Path(args.checkpoint).with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
