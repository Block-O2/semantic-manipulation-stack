"""State-based Diffusion Policy components for the semantic Push backend.

The conditional 1D U-Net and inference layout are minimally adapted from
real-stanford/diffusion_policy (MIT, commit 5ba07ac). The adaptation keeps the
official global observation conditioning, epsilon objective, DDPM scheduler,
and receding-horizon action slicing while using this repository's 10D state,
3D absolute Cartesian action, episode split, and padding mask.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Any

from diffusers import DDPMScheduler
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets import PushDataset
from learning.act_push import resolve_device


@dataclass(frozen=True)
class DiffusionArchitecture:
    observation_horizon: int = 2
    prediction_horizon: int = 16
    execution_horizon: int = 8
    observation_dimension: int = 10
    action_dimension: int = 3
    diffusion_step_embed_dim: int = 128
    down_dims: tuple[int, ...] = (64, 128, 256)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"

    def __post_init__(self) -> None:
        if self.observation_horizon <= 0:
            raise ValueError("observation_horizon must be positive")
        if not 1 <= self.execution_horizon <= self.prediction_horizon:
            raise ValueError("execution horizon must fit prediction horizon")
        start = self.observation_horizon - 1
        if start + self.execution_horizon > self.prediction_horizon:
            raise ValueError("executable action slice exceeds prediction horizon")
        if self.prediction_horizon % (2 ** (len(self.down_dims) - 1)) != 0:
            raise ValueError("prediction horizon is incompatible with U-Net downsamples")
        if any(dim <= 0 or dim % self.n_groups for dim in self.down_dims):
            raise ValueError("U-Net channels must be positive multiples of n_groups")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if self.num_train_timesteps <= 0 or self.num_inference_steps <= 0:
            raise ValueError("diffusion timestep counts must be positive")
        if self.num_inference_steps > self.num_train_timesteps:
            raise ValueError("inference steps cannot exceed training timesteps")


def build_noise_scheduler(architecture: DiffusionArchitecture) -> DDPMScheduler:
    """Build the scheduler used by the official low-dimensional formulation."""

    return DDPMScheduler(
        num_train_timesteps=architecture.num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=architecture.beta_schedule,
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type=architecture.prediction_type,
    )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10000) / (half - 1)
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device) * -scale
        )
        embedding = timesteps[:, None] * frequencies[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        n_groups: int,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(n_groups, output_channels),
            nn.Mish(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        condition_dimension: int,
        *,
        kernel_size: int,
        n_groups: int,
        cond_predict_scale: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(
                    input_channels, output_channels, kernel_size, n_groups
                ),
                Conv1dBlock(
                    output_channels, output_channels, kernel_size, n_groups
                ),
            ]
        )
        condition_channels = output_channels * (2 if cond_predict_scale else 1)
        self.condition_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dimension, condition_channels),
        )
        self.cond_predict_scale = cond_predict_scale
        self.output_channels = output_channels
        self.residual = (
            nn.Conv1d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(
        self, values: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        output = self.blocks[0](values)
        encoded = self.condition_encoder(condition).unsqueeze(-1)
        if self.cond_predict_scale:
            encoded = encoded.reshape(
                encoded.shape[0], 2, self.output_channels, 1
            )
            scale = encoded[:, 0]
            bias = encoded[:, 1]
            output = scale * output + bias
        else:
            output = output + encoded
        output = self.blocks[1](output)
        return output + self.residual(values)


class ConditionalUnet1D(nn.Module):
    """Official-style FiLM-conditioned temporal 1D U-Net."""

    def __init__(self, architecture: DiffusionArchitecture) -> None:
        super().__init__()
        dimensions = [architecture.action_dimension, *architecture.down_dims]
        pairs = list(zip(dimensions[:-1], dimensions[1:]))
        condition_dimension = (
            architecture.diffusion_step_embed_dim
            + architecture.observation_horizon
            * architecture.observation_dimension
        )
        embed = architecture.diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(embed),
            nn.Linear(embed, embed * 4),
            nn.Mish(),
            nn.Linear(embed * 4, embed),
        )

        def residual(input_channels: int, output_channels: int):
            return ConditionalResidualBlock1D(
                input_channels,
                output_channels,
                condition_dimension,
                kernel_size=architecture.kernel_size,
                n_groups=architecture.n_groups,
                cond_predict_scale=architecture.cond_predict_scale,
            )

        self.down_modules = nn.ModuleList()
        for index, (input_channels, output_channels) in enumerate(pairs):
            is_last = index == len(pairs) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        residual(input_channels, output_channels),
                        residual(output_channels, output_channels),
                        (
                            nn.Identity()
                            if is_last
                            else nn.Conv1d(
                                output_channels,
                                output_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                            )
                        ),
                    ]
                )
            )
        middle = dimensions[-1]
        self.mid_modules = nn.ModuleList(
            [residual(middle, middle), residual(middle, middle)]
        )
        self.up_modules = nn.ModuleList()
        for input_channels, output_channels in reversed(pairs[1:]):
            self.up_modules.append(
                nn.ModuleList(
                    [
                        residual(output_channels * 2, input_channels),
                        residual(input_channels, input_channels),
                        nn.ConvTranspose1d(
                            input_channels,
                            input_channels,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                    ]
                )
            )
        first = architecture.down_dims[0]
        self.final = nn.Sequential(
            Conv1dBlock(
                first,
                first,
                architecture.kernel_size,
                architecture.n_groups,
            ),
            nn.Conv1d(first, architecture.action_dimension, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        global_condition: torch.Tensor,
    ) -> torch.Tensor:
        values = sample.permute(0, 2, 1)
        if not torch.is_tensor(timestep):
            timesteps = torch.tensor(
                [timestep], dtype=torch.long, device=values.device
            )
        elif timestep.ndim == 0:
            timesteps = timestep[None].to(values.device)
        else:
            timesteps = timestep.to(values.device)
        timesteps = timesteps.expand(values.shape[0])
        condition = torch.cat(
            [self.diffusion_step_encoder(timesteps), global_condition], dim=-1
        )
        skips: list[torch.Tensor] = []
        for first, second, downsample in self.down_modules:
            values = first(values, condition)
            values = second(values, condition)
            skips.append(values)
            values = downsample(values)
        for middle in self.mid_modules:
            values = middle(values, condition)
        for first, second, upsample in self.up_modules:
            values = torch.cat((values, skips.pop()), dim=1)
            values = first(values, condition)
            values = second(values, condition)
            values = upsample(values)
        return self.final(values).permute(0, 2, 1)


def build_diffusion_model(architecture: DiffusionArchitecture) -> ConditionalUnet1D:
    return ConditionalUnet1D(architecture)


def _fit_limit_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32).reshape(-1, values.shape[-1])
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0)
    value_range = maximum - minimum
    constant = value_range < np.float32(1e-4)
    safe_range = value_range.copy()
    safe_range[constant] = np.float32(2.0)
    scale = np.float32(2.0) / safe_range
    offset = np.float32(-1.0) - scale * minimum
    scale[constant] = np.float32(1.0)
    offset[constant] = -mean[constant]
    return {
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": standard_deviation,
        "scale": scale,
        "offset": offset,
    }


def compute_diffusion_normalization(
    dataset: PushDataset, train_episode_ids: set[int]
) -> dict[str, dict[str, np.ndarray]]:
    if not train_episode_ids:
        raise ValueError("normalization requires training episodes")
    observations = np.concatenate(
        [dataset.episodes[index].observations for index in sorted(train_episode_ids)]
    )
    actions = np.concatenate(
        [dataset.episodes[index].actions for index in sorted(train_episode_ids)]
    )
    return {
        "observation": _fit_limit_stats(observations),
        "action": _fit_limit_stats(actions),
    }


def normalize_array(values: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return (
        np.asarray(values, dtype=np.float32) * stats["scale"] + stats["offset"]
    ).astype(np.float32)


def unnormalize_tensor(
    values: torch.Tensor, stats: dict[str, np.ndarray]
) -> torch.Tensor:
    scale = torch.as_tensor(stats["scale"], device=values.device)
    offset = torch.as_tensor(stats["offset"], device=values.device)
    return (values - offset) / scale


class PushDiffusionDataset(Dataset[dict[str, torch.Tensor]]):
    """Episode-bounded observation/action windows with boundary padding masks."""

    def __init__(
        self,
        dataset: PushDataset,
        *,
        episode_ids: set[int],
        architecture: DiffusionArchitecture,
        normalization: dict[str, dict[str, np.ndarray]],
    ) -> None:
        self.dataset = dataset
        self.architecture = architecture
        self.normalization = normalization
        self.indices = [
            (episode_id, timestep)
            for episode_id in sorted(episode_ids)
            for timestep in range(dataset.episodes[episode_id].length)
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_id, latest_timestep = self.indices[index]
        episode = self.dataset.episodes[episode_id]
        start = latest_timestep - (self.architecture.observation_horizon - 1)
        observation_indices = np.clip(
            np.arange(start, start + self.architecture.observation_horizon),
            0,
            episode.length - 1,
        )
        raw_action_indices = np.arange(
            start, start + self.architecture.prediction_horizon
        )
        action_valid = (raw_action_indices >= 0) & (
            raw_action_indices < episode.length
        )
        action_indices = np.clip(raw_action_indices, 0, episode.length - 1)
        observations = normalize_array(
            episode.observations[observation_indices],
            self.normalization["observation"],
        )
        actions = normalize_array(
            episode.actions[action_indices], self.normalization["action"]
        )
        return {
            "observation": torch.from_numpy(observations),
            "action": torch.from_numpy(actions),
            "action_valid": torch.from_numpy(action_valid),
            "episode_id": torch.tensor(episode_id, dtype=torch.int64),
            "latest_timestep": torch.tensor(latest_timestep, dtype=torch.int64),
        }


def diffusion_loss(
    model: ConditionalUnet1D,
    scheduler: DDPMScheduler,
    batch: dict[str, torch.Tensor],
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    actions = batch["action"]
    observations = batch["observation"]
    noise = torch.randn(
        actions.shape,
        dtype=actions.dtype,
        device=actions.device,
        generator=generator,
    )
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (actions.shape[0],),
        device=actions.device,
        generator=generator,
    ).long()
    noisy_actions = scheduler.add_noise(actions, noise, timesteps)
    predicted_noise = model(
        noisy_actions, timesteps, observations.reshape(observations.shape[0], -1)
    )
    squared_error = F.mse_loss(predicted_noise, noise, reduction="none")
    valid = batch["action_valid"].unsqueeze(-1).to(squared_error.dtype)
    denominator = valid.sum() * actions.shape[-1]
    return (squared_error * valid).sum() / denominator.clamp_min(1.0)


@torch.no_grad()
def sample_action_trajectory(
    model: ConditionalUnet1D,
    scheduler: DDPMScheduler,
    normalized_observations: torch.Tensor,
    architecture: DiffusionArchitecture,
    *,
    generator: torch.Generator | None = None,
    diagnostic_stats: dict[str, np.ndarray] | None = None,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    batch_size = normalized_observations.shape[0]
    trajectory = torch.randn(
        (
            batch_size,
            architecture.prediction_horizon,
            architecture.action_dimension,
        ),
        dtype=normalized_observations.dtype,
        device=normalized_observations.device,
        generator=generator,
    )
    scheduler.set_timesteps(
        architecture.num_inference_steps, device=normalized_observations.device
    )
    global_condition = normalized_observations.reshape(batch_size, -1)
    diagnostics: list[dict[str, float | int]] = []
    previous_physical: torch.Tensor | None = None
    for index, timestep in enumerate(scheduler.timesteps):
        predicted_noise = model(trajectory, timestep, global_condition)
        trajectory = scheduler.step(
            predicted_noise, timestep, trajectory, generator=generator
        ).prev_sample
        if diagnostic_stats is not None:
            physical = unnormalize_tensor(trajectory, diagnostic_stats)
            change = (
                0.0
                if previous_physical is None
                else float(torch.mean(torch.abs(physical - previous_physical)).cpu())
            )
            diagnostics.append(
                {
                    "iteration": index + 1,
                    "noise_timestep": int(timestep),
                    "normalized_mean": float(trajectory.mean().cpu()),
                    "normalized_std": float(trajectory.std().cpu()),
                    "action_mean_m": float(physical.mean().cpu()),
                    "action_std_m": float(physical.std().cpu()),
                    "mean_abs_change_from_previous_m": change,
                }
            )
            previous_physical = physical
    return trajectory, diagnostics


class DiffusionMotorPolicy:
    """Checkpoint-backed receding-horizon Diffusion Policy runtime."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint).resolve()
        self.checkpoint_sha256 = hashlib.sha256(
            self.checkpoint_path.read_bytes()
        ).hexdigest()
        self.device = resolve_device(device)
        payload = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        self.architecture = DiffusionArchitecture(**payload["architecture"])
        self.model = build_diffusion_model(self.architecture)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.scheduler = build_noise_scheduler(self.architecture)
        self.normalization = {
            key: {
                name: np.asarray(values, dtype=np.float32)
                for name, values in group.items()
            }
            for key, group in payload["normalization"].items()
        }
        self.metadata: dict[str, Any] = payload["metadata"]
        self.sampling_seed = int(self.metadata.get("seed", 17))
        self._history: deque[np.ndarray] = deque(
            maxlen=self.architecture.observation_horizon
        )
        self.inference_calls = 0
        self.inference_latencies: list[float] = []
        self.last_inference_latency = 0.0
        self.reset()

    @property
    def history_size(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        self._history.clear()
        self.inference_calls = 0
        self.inference_latencies = []
        self.last_inference_latency = 0.0
        self._generator = torch.Generator(device="cpu").manual_seed(
            self.sampling_seed
        )

    def observe(self, observation: np.ndarray) -> None:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (self.architecture.observation_dimension,):
            raise ValueError("Diffusion observation has the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("Diffusion observation must be finite")
        self._history.append(values.copy())

    def _condition(self, observation: np.ndarray) -> np.ndarray:
        current = np.asarray(observation, dtype=np.float32)
        if not self._history or not np.array_equal(self._history[-1], current):
            self.observe(current)
        history = list(self._history)
        while len(history) < self.architecture.observation_horizon:
            history.insert(0, history[0].copy())
        return normalize_array(
            np.stack(history), self.normalization["observation"]
        )

    @torch.no_grad()
    def predict(self, observation: np.ndarray) -> np.ndarray:
        condition = torch.from_numpy(self._condition(observation)).unsqueeze(0)
        condition = condition.to(self.device)
        started = time.perf_counter()
        generator = self._generator if self.device.type == "cpu" else None
        normalized, _ = sample_action_trajectory(
            self.model,
            self.scheduler,
            condition,
            self.architecture,
            generator=generator,
        )
        actions = unnormalize_tensor(normalized, self.normalization["action"])
        start = self.architecture.observation_horizon - 1
        end = start + self.architecture.execution_horizon
        selected = actions[0, start:end].cpu().numpy()
        self.last_inference_latency = time.perf_counter() - started
        self.inference_latencies.append(self.last_inference_latency)
        self.inference_calls += 1
        return selected

    @staticmethod
    def architecture_dict(
        architecture: DiffusionArchitecture,
    ) -> dict[str, object]:
        return asdict(architecture)
