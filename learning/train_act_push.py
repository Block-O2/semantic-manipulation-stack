"""Train one standard LeRobot ACT policy on the existing Push NPZ dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import load_push_dataset, split_episode_indices, validate_push_dataset
from learning.act_push import (
    ACTArchitecture,
    PushACTDataset,
    build_act_policy,
    compute_act_normalization,
    push_act_arrays,
    resolve_device,
)


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def validate(
    policy,
    loader: DataLoader,
    device: torch.device,
    batches: int = 24,
) -> dict[str, float]:
    policy.train()
    losses: list[float] = []
    l1_losses: list[float] = []
    kld_losses: list[float] = []
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        loss, metrics = policy(_move(batch, device))
        losses.append(float(loss.detach().cpu()))
        l1_losses.append(float(metrics["l1_loss"]))
        kld_losses.append(float(metrics.get("kld_loss", 0.0)))
    return {
        "loss": float(np.mean(losses)),
        "l1_loss": float(np.mean(l1_losses)),
        "kld_loss": float(np.mean(kld_losses)),
    }


@torch.no_grad()
def offline_action_metrics(
    policy,
    loader: DataLoader,
    device: torch.device,
    stats,
) -> dict[str, object]:
    policy.eval()
    absolute_errors: list[np.ndarray] = []
    horizon_errors: list[list[np.ndarray]] = [
        [] for _ in range(policy.config.chunk_size)
    ]
    action_mean = torch.from_numpy(stats["action"]["mean"]).to(device)
    action_std = torch.from_numpy(stats["action"]["std"]).to(device)
    for batch in loader:
        moved = _move(batch, device)
        predictions = policy.predict_action_chunk(
            {
                "observation.state": moved["observation.state"],
                "observation.environment_state": moved[
                    "observation.environment_state"
                ],
            }
        )
        predictions = predictions * action_std + action_mean
        targets = moved["action"] * action_std + action_mean
        errors = torch.abs(predictions - targets).cpu().numpy()
        valid = (~moved["action_is_pad"]).cpu().numpy()
        absolute_errors.append(errors[valid])
        for horizon in range(policy.config.chunk_size):
            horizon_errors[horizon].append(errors[valid[:, horizon], horizon])
    combined = np.concatenate(absolute_errors)
    per_horizon = [float(np.mean(np.concatenate(values))) for values in horizon_errors]
    return {
        "masked_chunk_l1_m": float(np.mean(combined)),
        "first_action_l1_m": per_horizon[0],
        "first_action_l1_per_axis_m": [
            float(value) for value in np.mean(np.concatenate(horizon_errors[0]), axis=0)
        ],
        "horizon_l1_m": per_horizon,
    }


def train(
    *,
    dataset_path: Path,
    output: Path,
    device_name: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device_name)
    dataset = load_push_dataset(dataset_path)
    validation = validate_push_dataset(dataset)
    if not validation.valid:
        raise ValueError(f"dataset validation failed: {validation.errors}")
    train_indices, validation_indices = split_episode_indices(
        len(dataset.episodes), validation_fraction=0.2, seed=seed
    )
    train_ids = set(int(value) for value in train_indices)
    validation_ids = set(int(value) for value in validation_indices)
    architecture = ACTArchitecture()
    arrays = push_act_arrays(dataset)
    normalization = compute_act_normalization(arrays, train_ids)
    train_dataset = PushACTDataset(
        arrays,
        episode_ids=train_ids,
        chunk_size=architecture.chunk_size,
        normalization=normalization,
    )
    validation_dataset = PushACTDataset(
        arrays,
        episode_ids=validation_ids,
        chunk_size=architecture.chunk_size,
        normalization=normalization,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    policy = build_act_policy(architecture).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    iterator = iter(train_loader)
    history: list[dict[str, object]] = []
    best_validation = float("inf")
    best_state = None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = policy(_move(batch, device))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite ACT loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            validation_metrics = validate(policy, validation_loader, device)
            record = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "train_l1_loss": float(metrics["l1_loss"]),
                "train_kld_loss": float(metrics.get("kld_loss", 0.0)),
                "validation": validation_metrics,
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if validation_metrics["loss"] < best_validation:
                best_validation = validation_metrics["loss"]
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in policy.state_dict().items()
                }
    assert best_state is not None
    policy.load_state_dict(best_state)
    offline = offline_action_metrics(policy, validation_loader, device, normalization)
    training_seconds = time.perf_counter() - started
    serializable_stats = {
        key: {name: values.tolist() for name, values in stat.items()}
        for key, stat in normalization.items()
    }
    metadata = {
        "act_implementation": "Hugging Face LeRobot ACTPolicy",
        "lerobot_version": "0.4.4",
        "torch_version": torch.__version__,
        "device": str(device),
        "dataset_path": str(dataset_path.resolve()),
        "episodes": len(dataset.episodes),
        "timesteps": dataset.total_timesteps,
        "train_episode_ids": sorted(train_ids),
        "validation_episode_ids": sorted(validation_ids),
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "best_validation_loss": best_validation,
        "training_seconds": training_seconds,
        "offline_metrics": offline,
        "history": history,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "architecture": asdict(architecture),
            "normalization": serializable_stats,
            "metadata": metadata,
        },
        output,
    )
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    summary = {
        "checkpoint": str(output.resolve()),
        "metrics": str(metrics_path.resolve()),
        **metadata,
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/push_demos_50_seed123.npz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/push_act_seed17.pt")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    train(
        dataset_path=args.dataset,
        output=args.output,
        device_name=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
