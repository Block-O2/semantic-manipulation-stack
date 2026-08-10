"""Pinned LeRobot ACT construction and inference wrapper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent_act_reproduction.tasks import BottleObservation


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


def build_act_policy(architecture: ACTArchitecture):
    """Construct Hugging Face LeRobot 0.4.4 ACTPolicy without Hub access."""

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(11,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
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
        pretrained_backbone_weights=None,
    )
    return ACTPolicy(config)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")
    return torch.device(requested)


class ACTMotorPolicy:
    """Local checkpoint -> observation -> ACT action chunk -> one Cartesian action."""

    def __init__(self, checkpoint: Path, *, device: str = "auto") -> None:
        self.checkpoint_path = checkpoint.resolve()
        self.device = resolve_device(device)
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.architecture = ACTArchitecture(**payload["architecture"])
        self.policy = build_act_policy(self.architecture)
        self.policy.load_state_dict(payload["model_state_dict"], strict=True)
        self.policy.to(self.device)
        self.policy.eval()
        self.stats: dict[str, dict[str, np.ndarray]] = {
            key: {
                name: np.asarray(values, dtype=np.float32)
                for name, values in stat.items()
            }
            for key, stat in payload["normalization"].items()
        }
        self.metadata: dict[str, Any] = payload["metadata"]

    def reset(self) -> None:
        self.policy.reset()

    def _normalize(self, key: str, values: np.ndarray) -> torch.Tensor:
        stat = self.stats[key]
        normalized = (values.astype(np.float32) - stat["mean"]) / stat["std"]
        return torch.from_numpy(normalized).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def select_action(self, observation: BottleObservation) -> np.ndarray:
        batch = {
            "observation.state": self._normalize("observation.state", observation.robot_state),
            "observation.environment_state": self._normalize(
                "observation.environment_state", observation.environment_state
            ),
        }
        normalized_action = self.policy.select_action(batch)[0].detach().cpu().numpy()
        stat = self.stats["action"]
        return normalized_action * stat["std"] + stat["mean"]

    @staticmethod
    def architecture_dict(architecture: ACTArchitecture) -> dict[str, object]:
        return asdict(architecture)
