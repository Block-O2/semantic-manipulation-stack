"""Trusted adapter from bounded Cartesian commands to robosuite OSC actions."""

from __future__ import annotations

import numpy as np
from robosuite.utils import transform_utils as T

class PandaCartesianAdapter:
    """Execute one absolute XYZ + gripper command per policy timestep.

    This is the only runtime component aware of robosuite's normalized action
    layout. It applies clipping and orientation holding, but contains no task
    waypoints, phases, or scripted recovery.
    """

    WORKSPACE_LOW = np.array((-0.30, -0.30, 0.825), dtype=np.float64)
    WORKSPACE_HIGH = np.array((0.30, 0.32, 1.18), dtype=np.float64)

    def __init__(
        self,
        env,
        *,
        max_position_delta_m: float = 0.05,
        max_rotation_delta_rad: float = 0.5,
    ) -> None:
        self.env = env
        self.max_position_delta_m = float(max_position_delta_m)
        self.max_rotation_delta_rad = float(max_rotation_delta_rad)
        self._fixed_orientation = env.end_effector_quaternion_xyzw()
        low, high = env.action_spec
        if low.shape != (7,) or high.shape != (7,):
            raise ValueError(f"Expected Panda OSC action shape (7,), got {low.shape}")

    @staticmethod
    def _orientation_error(target_xyzw: np.ndarray, current_xyzw: np.ndarray) -> np.ndarray:
        target = target_xyzw / np.linalg.norm(target_xyzw)
        current = current_xyzw / np.linalg.norm(current_xyzw)
        delta = T.quat2mat(target) @ T.quat2mat(current).T
        return np.asarray(T.quat2axisangle(T.mat2quat(delta)), dtype=np.float64)

    def execute(self, command: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (4,) or not np.all(np.isfinite(command)):
            raise ValueError("Cartesian policy action must be a finite vector with shape (4,)")

        desired = np.clip(command[:3], self.WORKSPACE_LOW, self.WORKSPACE_HIGH)
        current = self.env.end_effector_position()
        current_quat = self.env.end_effector_quaternion_xyzw()
        rotation_error = self._orientation_error(self._fixed_orientation, current_quat)

        osc_action = np.zeros(7, dtype=np.float64)
        osc_action[:3] = np.clip(
            (desired - current) / self.max_position_delta_m,
            -1.0,
            1.0,
        )
        osc_action[3:6] = np.clip(
            rotation_error / self.max_rotation_delta_rad,
            -1.0,
            1.0,
        )
        osc_action[6] = float(np.clip(command[3], -1.0, 1.0))
        self.env.step(osc_action)
        self.env.render_if_enabled()
        return np.concatenate((desired, [osc_action[6]]))
