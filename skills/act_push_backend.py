"""LeRobot ACT Push backend behind the existing trusted Cartesian boundary."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from skills.bc_push_backend import BCPushBackend, BCPushConfig

if TYPE_CHECKING:
    from learning.act_push import ACTExecutionMode, ACTMotorPolicy


class ACTPushBackend(BCPushBackend):
    name = "act"

    def __init__(
        self,
        policy: ACTMotorPolicy,
        *,
        config: BCPushConfig = BCPushConfig(),
    ) -> None:
        super().__init__(policy, config=config)
        execution_mode = getattr(policy, "execution_mode", "queue")
        self.name = (
            "act_temporal_ensemble"
            if execution_mode == "temporal_ensemble"
            else "act"
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        *,
        device: str = "auto",
        execution_mode: ACTExecutionMode | str = "queue",
        temporal_ensemble_coeff: float = 0.01,
        config: BCPushConfig = BCPushConfig(),
    ) -> "ACTPushBackend":
        from learning.act_push import ACTMotorPolicy

        return cls(
            ACTMotorPolicy(
                checkpoint,
                device=device,
                execution_mode=execution_mode,
                temporal_ensemble_coeff=temporal_ensemble_coeff,
            ),
            config=config,
        )

    def _reset_policy(self) -> None:
        self.policy.reset()

    def _predict_actions(self, observation: np.ndarray, rollout_step: int) -> np.ndarray:
        del rollout_step
        return np.atleast_2d(self.policy.select_action(observation))

    def _action_log_metadata(self) -> dict[str, object]:
        return {"ensemble_overlap_count": self.policy.current_overlap_count}
