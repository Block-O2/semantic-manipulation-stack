"""Evaluation-only simulator disturbances for closed-loop agent experiments."""

from __future__ import annotations

import numpy as np

from runtime import AgentStepEvent
from sim import SemanticTabletopEnv


class ObjectLostAfterPick:
    """Teleport the cube back to the table once after a successful Pick."""

    def __init__(
        self,
        env: SemanticTabletopEnv,
        *,
        table_position: tuple[float, float, float] = (-0.05, -0.10, 0.825),
    ) -> None:
        self._env = env
        self._position = np.asarray(table_position, dtype=np.float64)
        self.triggered = False

    def __call__(self, event: AgentStepEvent) -> None:
        if self.triggered or event.step.skill != "pick" or not event.task_result.success:
            return
        joint = self._env.red_cube.joints[0]
        qpos = np.concatenate([self._position, np.array([1.0, 0.0, 0.0, 0.0])])
        self._env.sim.data.set_joint_qpos(joint, qpos)
        self._env.sim.data.set_joint_qvel(joint, np.zeros(6, dtype=np.float64))
        self._env.sim.forward()
        self.triggered = True


class TargetUnreachableAfterPick:
    """Move the visual target outside semantic workspace once after Pick."""

    def __init__(
        self,
        env: SemanticTabletopEnv,
        *,
        target_position: tuple[float, float, float] = (0.50, 0.50, 0.803),
    ) -> None:
        self._env = env
        self._position = np.asarray(target_position, dtype=np.float64)
        self.triggered = False

    def __call__(self, event: AgentStepEvent) -> None:
        if self.triggered or event.step.skill != "pick" or not event.task_result.success:
            return
        body_id = self._env._object_body_ids["blue_target"]
        self._env.sim.model.body_pos[body_id] = self._position
        self._env.sim.forward()
        self.triggered = True
