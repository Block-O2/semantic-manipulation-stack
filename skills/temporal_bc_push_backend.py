"""Progress and action-chunk BC Push backends using the trusted BC boundary."""

from __future__ import annotations

import numpy as np

from learning.temporal_push_bc import ChunkBCPolicy, ProgressBCPolicy
from skills.bc_push_backend import BCPushBackend, BCPushConfig


class ProgressBCPushBackend(BCPushBackend):
    name = "progress_bc"

    def __init__(
        self,
        policy: ProgressBCPolicy,
        *,
        config: BCPushConfig = BCPushConfig(),
    ) -> None:
        super().__init__(policy, config=config)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        *,
        config: BCPushConfig = BCPushConfig(),
    ) -> "ProgressBCPushBackend":
        policy, _ = ProgressBCPolicy.load(checkpoint)
        return cls(policy, config=config)

    def _predict_actions(self, observation: np.ndarray, rollout_step: int) -> np.ndarray:
        denominator = max(1, self.config.maximum_control_steps - 1)
        progress = min(1.0, rollout_step / denominator)
        return np.atleast_2d(self.policy.predict(observation, progress))


class ChunkBCPushBackend(BCPushBackend):
    def __init__(
        self,
        policy: ChunkBCPolicy,
        *,
        execution_horizon: int,
        config: BCPushConfig = BCPushConfig(),
    ) -> None:
        if not 1 <= execution_horizon <= policy.prediction_horizon:
            raise ValueError("execution horizon H must satisfy 1 <= H <= K")
        super().__init__(policy, config=config)
        self.execution_horizon = execution_horizon
        self.name = (
            f"chunk_bc_k{policy.prediction_horizon}_h{self.execution_horizon}"
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        *,
        execution_horizon: int,
        config: BCPushConfig = BCPushConfig(),
    ) -> "ChunkBCPushBackend":
        policy, _ = ChunkBCPolicy.load(checkpoint)
        return cls(policy, execution_horizon=execution_horizon, config=config)

    def _predict_actions(self, observation: np.ndarray, rollout_step: int) -> np.ndarray:
        del rollout_step
        return self.policy.predict(observation)[: self.execution_horizon]
