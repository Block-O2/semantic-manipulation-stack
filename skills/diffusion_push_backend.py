"""State-based Diffusion Policy behind the trusted Cartesian Push boundary."""

from __future__ import annotations

import numpy as np

from datasets.specs import encode_push_state
from learning.diffusion_push import DiffusionMotorPolicy
from skills.bc_push_backend import BCPushBackend, BCPushConfig
from skills.push_backend import PushExecutionContext, PushRequest


class DiffusionPushBackend(BCPushBackend):
    """Execute receding-horizon diffusion trajectories through BCPushBackend."""

    name = "diffusion"

    def __init__(
        self,
        policy: DiffusionMotorPolicy,
        *,
        config: BCPushConfig = BCPushConfig(),
    ) -> None:
        super().__init__(policy, config=config)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        *,
        device: str = "auto",
        config: BCPushConfig = BCPushConfig(),
    ) -> "DiffusionPushBackend":
        return cls(
            DiffusionMotorPolicy(checkpoint, device=device),
            config=config,
        )

    def _reset_policy(self) -> None:
        self.policy.reset()

    def _predict_actions(
        self,
        observation: np.ndarray,
        rollout_step: int,
    ) -> np.ndarray:
        del rollout_step
        return np.asarray(self.policy.predict(observation), dtype=np.float64)

    def _after_action(
        self,
        request: PushRequest,
        context: PushExecutionContext,
    ) -> None:
        self.policy.observe(
            encode_push_state(
                context.primitives.current_pose,
                context.world,
                request.object_name,
                request.target_name,
            )
        )

    def _action_log_metadata(self) -> dict[str, object]:
        return {
            "diffusion_inference_calls": self.policy.inference_calls,
            "diffusion_inference_latency_s": self.policy.last_inference_latency,
            "diffusion_observation_history_size": self.policy.history_size,
            "diffusion_denoising_steps": (
                self.policy.architecture.num_inference_steps
            ),
        }
