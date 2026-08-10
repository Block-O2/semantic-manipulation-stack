"""Train local LeRobot ACT on the Bottle demonstrations."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from agent_act_reproduction.data.dataset import (
    BottleACTDataset,
    compute_normalization,
    load_bottle_dataset,
)
from agent_act_reproduction.policies import ACTArchitecture, build_act_policy
from agent_act_reproduction.policies.act import resolve_device


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def validate(policy, loader: DataLoader, device: torch.device, batches: int = 20) -> dict[str, float]:
    # LeRobot 0.4.4 computes ACT's VAE latent only while the module is in
    # training mode, while its public loss wrapper still expects that latent
    # whenever use_vae=True. Keep training mode under no_grad for validation;
    # this is a pinned upstream behavior, not a custom ACT implementation.
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
        if "kld_loss" in metrics:
            kld_losses.append(float(metrics["kld_loss"]))
    return {
        "loss": float(np.mean(losses)),
        "l1_loss": float(np.mean(l1_losses)),
        "kld_loss": float(np.mean(kld_losses)) if kld_losses else 0.0,
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
    task_name: str = "bottle",
    n_action_steps: int = 8,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device_name)
    arrays = load_bottle_dataset(dataset_path)
    episode_count = len(arrays.episode_lengths)
    validation_count = max(1, episode_count // 10)
    train_ids = set(range(episode_count - validation_count))
    validation_ids = set(range(episode_count - validation_count, episode_count))
    architecture = ACTArchitecture(n_action_steps=n_action_steps)
    normalization = compute_normalization(arrays, train_ids)
    train_dataset = BottleACTDataset(
        arrays,
        episode_ids=train_ids,
        chunk_size=architecture.chunk_size,
        normalization=normalization,
    )
    validation_dataset = BottleACTDataset(
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
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    policy = build_act_policy(architecture).to(device)
    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=1e-4)
    iterator = iter(train_loader)
    history: list[dict[str, object]] = []
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    start_time = time.perf_counter()

    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss, train_metrics = policy(_move(batch, device))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == steps:
            validation = validate(policy, validation_loader, device)
            record = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "train_l1_loss": float(train_metrics["l1_loss"]),
                "train_kld_loss": float(train_metrics.get("kld_loss", 0.0)),
                "validation_loss": validation["loss"],
                "validation_l1_loss": validation["l1_loss"],
                "validation_kld_loss": validation["kld_loss"],
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if validation["loss"] < best_validation:
                best_validation = validation["loss"]
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in policy.state_dict().items()
                }

    assert best_state is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    serializable_stats = {
        key: {name: value.tolist() for name, value in stat.items()}
        for key, stat in normalization.items()
    }
    metadata = {
        "task_name": task_name,
        "act_implementation": "Hugging Face LeRobot ACTPolicy",
        "lerobot_version": "0.4.4",
        "torch_version": torch.__version__,
        "device": str(device),
        "dataset_path": str(dataset_path.resolve()),
        "train_episode_ids": sorted(train_ids),
        "validation_episode_ids": sorted(validation_ids),
        "dataset_timesteps": int(arrays.action.shape[0]),
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "best_validation_loss": best_validation,
        "training_seconds": time.perf_counter() - start_time,
        "history": history,
        "feature_names": arrays.feature_names,
    }
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
        "device": str(device),
        "best_validation_loss": best_validation,
        "final_train_loss": history[-1]["train_loss"],
        "training_seconds": metadata["training_seconds"],
        "parameters": sum(parameter.numel() for parameter in policy.parameters()),
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/bottle_demos_50.npz"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/bottle_act.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task-name", choices=("bottle", "tissue", "draw"), default="bottle")
    parser.add_argument("--n-action-steps", type=int, default=8)
    args = parser.parse_args()
    train(
        dataset_path=args.dataset,
        output=args.output,
        device_name=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        task_name=args.task_name,
        n_action_steps=args.n_action_steps,
    )


if __name__ == "__main__":
    main()
