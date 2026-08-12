"""Standard Hugging Face LeRobot ACT integration for state-based Push."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import PushDataset, split_episode_indices
from learning.lerobot_compat import load_lerobot_act_classes


@dataclass(frozen=True)
class ACTArchitecture:
    chunk_size: int = 32
    n_action_steps: int = 8
    dim_model: int = 128
    n_heads: int = 4
    dim_feedforward: int = 512
    n_encoder_layers: int = 2
    n_decoder_layers: int = 1
    use_vae: bool = True
    latent_dim: int = 16
    n_vae_encoder_layers: int = 2
    dropout: float = 0.1
    kl_weight: float = 1.0
    temporal_ensemble_coeff: float | None = None

    def __post_init__(self) -> None:
        if self.chunk_size <= 0 or not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("ACT horizons must satisfy 1 <= n_action_steps <= chunk_size")
        if self.temporal_ensemble_coeff is not None and self.n_action_steps != 1:
            raise ValueError("ACT temporal ensembling requires n_action_steps=1")


class ACTExecutionMode(str, Enum):
    QUEUE = "queue"
    TEMPORAL_ENSEMBLE = "temporal_ensemble"


def build_act_policy(architecture: ACTArchitecture):
    """Build LeRobot 0.4.4 ACTPolicy without Hub or image dependencies."""

    FeatureType, PolicyFeature, ACTConfig, ACTPolicy = load_lerobot_act_classes()
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(10,)),
            # LeRobot 0.4.4 requires an ENV or image token even for state-only
            # ACT.  This empty tensor carries no additional information.
            "observation.environment_state": PolicyFeature(
                type=FeatureType.ENV, shape=(0,)
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
        },
        chunk_size=architecture.chunk_size,
        n_action_steps=architecture.n_action_steps,
        dim_model=architecture.dim_model,
        n_heads=architecture.n_heads,
        dim_feedforward=architecture.dim_feedforward,
        n_encoder_layers=architecture.n_encoder_layers,
        n_decoder_layers=architecture.n_decoder_layers,
        use_vae=architecture.use_vae,
        latent_dim=architecture.latent_dim,
        n_vae_encoder_layers=architecture.n_vae_encoder_layers,
        dropout=architecture.dropout,
        kl_weight=architecture.kl_weight,
        temporal_ensemble_coeff=architecture.temporal_ensemble_coeff,
        pretrained_backbone_weights=None,
    )
    return ACTPolicy(config)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")
    return torch.device(requested)


@dataclass(frozen=True)
class PushACTArrays:
    observations: np.ndarray
    actions: np.ndarray
    episode_ids: np.ndarray
    episode_lengths: np.ndarray


def push_act_arrays(dataset: PushDataset) -> PushACTArrays:
    lengths = np.asarray(
        [episode.length for episode in dataset.episodes], dtype=np.int64
    )
    return PushACTArrays(
        observations=np.concatenate(
            [episode.observations for episode in dataset.episodes]
        ).astype(np.float32),
        actions=np.concatenate(
            [episode.actions for episode in dataset.episodes]
        ).astype(np.float32),
        episode_ids=np.concatenate(
            [
                np.full(episode.length, index, dtype=np.int64)
                for index, episode in enumerate(dataset.episodes)
            ]
        ),
        episode_lengths=lengths,
    )


def compute_act_normalization(
    arrays: PushACTArrays,
    train_episode_ids: set[int],
) -> dict[str, dict[str, np.ndarray]]:
    mask = np.isin(arrays.episode_ids, np.asarray(sorted(train_episode_ids)))
    result: dict[str, dict[str, np.ndarray]] = {}
    for key, values in (
        ("observation.state", arrays.observations[mask]),
        ("action", arrays.actions[mask]),
    ):
        mean = values.mean(axis=0).astype(np.float32)
        std = np.maximum(values.std(axis=0), np.float32(1e-4)).astype(np.float32)
        result[key] = {"mean": mean, "std": std}
    return result


class PushACTDataset(Dataset[dict[str, torch.Tensor]]):
    """Episode-bounded future-action chunks with explicit padding masks."""

    def __init__(
        self,
        arrays: PushACTArrays,
        *,
        episode_ids: set[int],
        chunk_size: int,
        normalization: dict[str, dict[str, np.ndarray]],
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.arrays = arrays
        self.chunk_size = chunk_size
        self.normalization = normalization
        self.indices = np.flatnonzero(np.isin(arrays.episode_ids, sorted(episode_ids)))
        offsets = np.concatenate(([0], np.cumsum(arrays.episode_lengths)))
        self._episode_ends = {
            episode: int(offsets[episode + 1])
            for episode in range(len(arrays.episode_lengths))
        }

    def __len__(self) -> int:
        return int(self.indices.size)

    def _normalize(self, key: str, values: np.ndarray) -> np.ndarray:
        stats = self.normalization[key]
        return (values - stats["mean"]) / stats["std"]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        episode = int(self.arrays.episode_ids[row])
        episode_end = self._episode_ends[episode]
        valid = min(self.chunk_size, episode_end - row)
        action_chunk = np.empty((self.chunk_size, 3), dtype=np.float32)
        action_chunk[:valid] = self.arrays.actions[row : row + valid]
        action_chunk[valid:] = self.arrays.actions[episode_end - 1]
        is_pad = np.ones(self.chunk_size, dtype=np.bool_)
        is_pad[:valid] = False
        return {
            "observation.state": torch.from_numpy(
                self._normalize("observation.state", self.arrays.observations[row])
            ),
            "observation.environment_state": torch.empty(0, dtype=torch.float32),
            "action": torch.from_numpy(self._normalize("action", action_chunk)),
            "action_is_pad": torch.from_numpy(is_pad),
        }


class ACTMotorPolicy:
    """Local LeRobot checkpoint with queue or native ensemble inference."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "auto",
        execution_mode: ACTExecutionMode | str = ACTExecutionMode.QUEUE,
        temporal_ensemble_coeff: float = 0.01,
    ) -> None:
        self.checkpoint_path = Path(checkpoint).resolve()
        self.checkpoint_sha256 = hashlib.sha256(
            self.checkpoint_path.read_bytes()
        ).hexdigest()
        self.device = resolve_device(device)
        payload = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint_architecture = ACTArchitecture(**payload["architecture"])
        self.execution_mode = ACTExecutionMode(execution_mode)
        if self.execution_mode is ACTExecutionMode.TEMPORAL_ENSEMBLE:
            self.architecture = replace(
                checkpoint_architecture,
                n_action_steps=1,
                temporal_ensemble_coeff=temporal_ensemble_coeff,
            )
        else:
            self.architecture = replace(
                checkpoint_architecture,
                temporal_ensemble_coeff=None,
            )
        self.policy = build_act_policy(self.architecture)
        self.policy.load_state_dict(payload["model_state_dict"], strict=True)
        self.policy.to(self.device)
        self.policy.eval()
        self.stats = {
            key: {
                name: np.asarray(values, dtype=np.float32)
                for name, values in stat.items()
            }
            for key, stat in payload["normalization"].items()
        }
        self.metadata: dict[str, Any] = payload["metadata"]
        self.inference_calls = 0
        self._actions_until_inference = 0
        self._action_timestep = 0
        self.current_overlap_count = 1
        self.ensemble_diagnostics: list[dict[str, object]] = []

    def reset(self) -> None:
        self.policy.reset()
        self.inference_calls = 0
        self._actions_until_inference = 0
        self._action_timestep = 0
        self.current_overlap_count = 1
        self.ensemble_diagnostics = []

    def _normalize_observation(self, observation: np.ndarray) -> torch.Tensor:
        stat = self.stats["observation.state"]
        values = (np.asarray(observation, dtype=np.float32) - stat["mean"]) / stat["std"]
        return torch.from_numpy(values).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def select_action(self, observation: np.ndarray) -> np.ndarray:
        if self.execution_mode is ACTExecutionMode.TEMPORAL_ENSEMBLE:
            self.inference_calls += 1
        elif self._actions_until_inference == 0:
            self.inference_calls += 1
            self._actions_until_inference = self.architecture.n_action_steps
        normalized = self.policy.select_action(
            {
                "observation.state": self._normalize_observation(observation),
                "observation.environment_state": torch.empty(
                    (1, 0), dtype=torch.float32, device=self.device
                ),
            }
        )[0].detach().cpu().numpy()
        stat = self.stats["action"]
        action = normalized * stat["std"] + stat["mean"]
        if self.execution_mode is ACTExecutionMode.TEMPORAL_ENSEMBLE:
            count = min(self._action_timestep + 1, self.architecture.chunk_size)
            first_source = self._action_timestep - count + 1
            source_timesteps = list(range(first_source, self._action_timestep + 1))
            native_weights = (
                self.policy.temporal_ensembler.ensemble_weights[:count]
                .detach()
                .cpu()
                .numpy()
            )
            weights = native_weights / native_weights.sum()
            self.current_overlap_count = count
            if self._action_timestep in {0, 1, 2, 4, 8, 16, 31, 32, 64}:
                self.ensemble_diagnostics.append(
                    {
                        "timestep": self._action_timestep,
                        "overlap_count": count,
                        "source_prediction_timesteps": source_timesteps,
                        "normalized_weights_oldest_to_newest": weights.tolist(),
                        "combined_action_xyz": action.tolist(),
                    }
                )
        else:
            self._actions_until_inference -= 1
            self.current_overlap_count = 1
        self._action_timestep += 1
        return action

    @staticmethod
    def architecture_dict(architecture: ACTArchitecture) -> dict[str, object]:
        return asdict(architecture)
