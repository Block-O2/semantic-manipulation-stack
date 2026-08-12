"""Train one state-based Diffusion Policy on the existing Push dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import load_push_dataset, split_episode_indices, validate_push_dataset
from learning.diffusion_push import (
    DiffusionArchitecture,
    PushDiffusionDataset,
    build_diffusion_model,
    build_noise_scheduler,
    compute_diffusion_normalization,
    diffusion_loss,
    sample_action_trajectory,
    unnormalize_tensor,
)
from learning.act_push import resolve_device


def _move(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def validate_loss(
    model,
    scheduler,
    loader: DataLoader,
    device: torch.device,
    *,
    seed: int,
    maximum_batches: int = 24,
) -> float:
    model.eval()
    generator = (
        torch.Generator(device="cpu").manual_seed(seed)
        if device.type == "cpu"
        else None
    )
    losses: list[float] = []
    for index, batch in enumerate(loader):
        if index >= maximum_batches:
            break
        loss = diffusion_loss(
            model,
            scheduler,
            _move(batch, device),
            generator=generator,
        )
        losses.append(float(loss.cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def offline_action_metrics(
    model,
    scheduler,
    loader: DataLoader,
    device: torch.device,
    architecture: DiffusionArchitecture,
    normalization: dict[str, dict[str, np.ndarray]],
    *,
    seed: int,
    maximum_batches: int = 8,
) -> dict[str, object]:
    model.eval()
    generator = (
        torch.Generator(device="cpu").manual_seed(seed)
        if device.type == "cpu"
        else None
    )
    all_errors: list[np.ndarray] = []
    first_errors: list[np.ndarray] = []
    horizon_errors: list[list[np.ndarray]] = [
        [] for _ in range(architecture.prediction_horizon)
    ]
    sample_count = 0
    for index, batch in enumerate(loader):
        if index >= maximum_batches:
            break
        moved = _move(batch, device)
        normalized, _ = sample_action_trajectory(
            model,
            scheduler,
            moved["observation"],
            architecture,
            generator=generator,
        )
        predictions = unnormalize_tensor(normalized, normalization["action"])
        targets = unnormalize_tensor(moved["action"], normalization["action"])
        errors = torch.abs(predictions - targets).cpu().numpy()
        valid = moved["action_valid"].cpu().numpy()
        all_errors.append(errors[valid])
        executable = architecture.observation_horizon - 1
        first_errors.append(errors[:, executable])
        for horizon in range(architecture.prediction_horizon):
            if np.any(valid[:, horizon]):
                horizon_errors[horizon].append(
                    errors[valid[:, horizon], horizon]
                )
        sample_count += errors.shape[0]
    combined = np.concatenate(all_errors)
    first = np.concatenate(first_errors)
    return {
        "sampled_validation_windows": sample_count,
        "sampled_trajectory_action_l1_m": float(np.mean(combined)),
        "first_executable_action_l1_m": float(np.mean(first)),
        "first_executable_action_l1_per_axis_m": [
            float(value) for value in np.mean(first, axis=0)
        ],
        "horizon_l1_m": [
            float(np.mean(np.concatenate(values))) for values in horizon_errors
        ],
    }


@torch.no_grad()
def denoising_sanity_check(
    model,
    scheduler,
    loader: DataLoader,
    device: torch.device,
    architecture: DiffusionArchitecture,
    normalization: dict[str, dict[str, np.ndarray]],
    *,
    seed: int,
) -> list[dict[str, float | int]]:
    batch = next(iter(loader))
    observation = batch["observation"][:1].to(device)
    generator = (
        torch.Generator(device="cpu").manual_seed(seed)
        if device.type == "cpu"
        else None
    )
    _, diagnostic = sample_action_trajectory(
        model,
        scheduler,
        observation,
        architecture,
        generator=generator,
        diagnostic_stats=normalization["action"],
    )
    return diagnostic


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
    if steps <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("training parameters must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device_name)
    dataset = load_push_dataset(dataset_path)
    validation_report = validate_push_dataset(dataset)
    if not validation_report.valid:
        raise ValueError(f"dataset validation failed: {validation_report.errors}")
    train_indices, validation_indices = split_episode_indices(
        len(dataset.episodes), validation_fraction=0.2, seed=seed
    )
    train_ids = set(int(value) for value in train_indices)
    validation_ids = set(int(value) for value in validation_indices)
    architecture = DiffusionArchitecture()
    normalization = compute_diffusion_normalization(dataset, train_ids)
    train_dataset = PushDiffusionDataset(
        dataset,
        episode_ids=train_ids,
        architecture=architecture,
        normalization=normalization,
    )
    validation_dataset = PushDiffusionDataset(
        dataset,
        episode_ids=validation_ids,
        architecture=architecture,
        normalization=normalization,
    )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = build_diffusion_model(architecture).to(device)
    noise_scheduler = build_noise_scheduler(architecture)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=1e-6,
    )
    warmup_steps = min(500, max(1, steps // 10))
    learning_rate_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=steps,
    )
    ema = EMAModel(
        model.parameters(),
        update_after_step=0,
        use_ema_warmup=True,
        inv_gamma=1.0,
        power=0.75,
        min_decay=0.0,
        max_decay=0.9999,
    )
    iterator = iter(train_loader)
    validation_interval = 250
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion_loss(
            model, noise_scheduler, _move(batch, device)
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite diffusion loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        learning_rate_scheduler.step()
        ema.step(model.parameters())
        if step == 1 or step % validation_interval == 0 or step == steps:
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            validation_loss = validate_loss(
                model,
                noise_scheduler,
                validation_loader,
                device,
                seed=seed + 1000,
            )
            record = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "validation_loss": validation_loss,
                "learning_rate": learning_rate_scheduler.get_last_lr()[0],
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            ema.restore(model.parameters())
    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("training produced no checkpoint state")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    offline = offline_action_metrics(
        model,
        noise_scheduler,
        validation_loader,
        device,
        architecture,
        normalization,
        seed=seed + 2000,
    )
    denoising = denoising_sanity_check(
        model,
        noise_scheduler,
        validation_loader,
        device,
        architecture,
        normalization,
        seed=seed + 3000,
    )
    serializable_normalization = {
        key: {name: values.tolist() for name, values in group.items()}
        for key, group in normalization.items()
    }
    metadata = {
        "implementation": "real-stanford Diffusion Policy low-dimensional CNN adaptation",
        "official_source_commit": "5ba07ac6661db573af695b419a7947ecb704690f",
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "device": str(device),
        "dataset_path": str(dataset_path.resolve()),
        "episodes": len(dataset.episodes),
        "timesteps": dataset.total_timesteps,
        "train_episode_ids": sorted(train_ids),
        "validation_episode_ids": sorted(validation_ids),
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "AdamW(betas=(0.95,0.999), weight_decay=1e-6)",
        "lr_scheduler": f"cosine with {warmup_steps} warmup steps",
        "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation_loss": best_validation,
        "training_seconds": training_seconds,
        "offline_metrics": offline,
        "denoising_sanity": denoising,
        "history": history,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "architecture": asdict(architecture),
            "normalization": serializable_normalization,
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
        "--output",
        type=Path,
        default=Path("checkpoints/push_diffusion_seed17.pt"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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
