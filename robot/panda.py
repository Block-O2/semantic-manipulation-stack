"""Simulator-independent public abstraction for the Franka Panda."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robot.controller import CartesianController, MotionResult

if TYPE_CHECKING:
    from sim.environment import SemanticTabletopEnv


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Pose:
    """Cartesian pose with position in metres and quaternion in xyzw order."""

    position: FloatArray
    quaternion: FloatArray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64)
        quaternion = np.asarray(self.quaternion, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position must be a finite vector with shape (3,)")
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("quaternion must be a finite vector with shape (4,)")
        norm = np.linalg.norm(quaternion)
        if norm < 1e-8:
            raise ValueError("quaternion norm must be non-zero")
        object.__setattr__(self, "position", position.copy())
        object.__setattr__(self, "quaternion", quaternion / norm)

    @classmethod
    def from_values(cls, position: ArrayLike, quaternion: ArrayLike) -> "Pose":
        return cls(np.asarray(position, dtype=np.float64), np.asarray(quaternion, dtype=np.float64))


class PandaRobot:
    """The robot-facing API used by future primitives and skills."""

    def __init__(self, env: "SemanticTabletopEnv") -> None:
        self._env = env
        self._controller = CartesianController(env)

    @property
    def pose(self) -> Pose:
        return self._env.end_effector_pose()

    def move_end_effector(
        self,
        target: Pose,
        *,
        max_steps: int = 250,
        position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
    ) -> MotionResult:
        return self._controller.move_to_pose(
            target,
            max_steps=max_steps,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )

    def open_gripper(self, *, steps: int = 40) -> None:
        self._controller.set_gripper(open=True, steps=steps)

    def close_gripper(self, *, steps: int = 40) -> None:
        self._controller.set_gripper(open=False, steps=steps)

    def hold(self, steps: int) -> None:
        self._controller.hold(steps)
