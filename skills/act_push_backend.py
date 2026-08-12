"""LeRobot ACT Push backend behind the existing trusted Cartesian boundary."""

from __future__ import annotations

import numpy as np

from learning.act_push import ACTMotorPolicy
from skills.bc_push_backend import BCPushBackend, BCPushConfig


class ACTPushBackend(BCPushBackend):
    name = "act"

    def __init__(
        self,
        policy: ACTMotorPolicy,
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
    ) -> "ACTPushBackend":
        return cls(ACTMotorPolicy(checkpoint, device=device), config=config)

    def _reset_policy(self) -> None:
        self.policy.reset()

    def _predict_actions(self, observation: np.ndarray, rollout_step: int) -> np.ndarray:
        del rollout_step
        return np.atleast_2d(self.policy.select_action(observation))
