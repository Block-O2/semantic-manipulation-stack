"""Cartesian control wrapper around robosuite's low-level action vector."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from robosuite.utils import transform_utils as T

if TYPE_CHECKING:
    from sim.environment import SemanticTabletopEnv
    from robot.panda import Pose


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MotionResult:
    """Outcome of a bounded Cartesian motion request."""

    reached: bool
    steps: int
    position_error: float
    orientation_error: float


@dataclass(frozen=True)
class CartesianCommandEvent:
    """One absolute Cartesian command at one simulator control timestep."""

    step: int
    current_pose: "Pose"
    target_pose: "Pose"
    gripper_command: float


class CartesianController:
    """Translate absolute end-effector targets into OSC pose delta actions.

    This is intentionally the only project layer that knows the simulator's
    action-vector layout. All callers operate on absolute Cartesian poses.
    """

    def __init__(
        self,
        env: "SemanticTabletopEnv",
        *,
        max_position_delta: float = 0.05,
        max_rotation_delta: float = 0.5,
        position_tolerance: float = 0.008,
        orientation_tolerance: float = 0.08,
    ) -> None:
        self._env = env
        self.max_position_delta = float(max_position_delta)
        self.max_rotation_delta = float(max_rotation_delta)
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        # Start neutral until the robot abstraction receives an explicit
        # open / close request. This prevents an arm-only first motion from
        # unexpectedly sweeping nearby objects with the fingers.
        # robosuite standard: -1 opens, +1 closes.
        self._gripper_command = 0.0
        self._command_step = 0
        self._command_observers: list[Callable[[CartesianCommandEvent], None]] = []

        low, high = self._env.action_spec
        if low.shape != (7,) or high.shape != (7,):
            raise ValueError(
                "Expected Panda OSC_POSE action shape (7,), "
                f"but environment exposes {low.shape}."
            )

    @staticmethod
    def _validate_vector(value: ArrayLike, size: int, name: str) -> FloatArray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite vector with shape ({size},)")
        return result

    @staticmethod
    def _normalize_quaternion(quaternion: ArrayLike) -> FloatArray:
        quat = CartesianController._validate_vector(quaternion, 4, "quaternion")
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            raise ValueError("quaternion norm must be non-zero")
        return quat / norm

    @staticmethod
    def orientation_error(target_xyzw: ArrayLike, current_xyzw: ArrayLike) -> FloatArray:
        """Return world-frame axis-angle error from current to target."""

        target = CartesianController._normalize_quaternion(target_xyzw)
        current = CartesianController._normalize_quaternion(current_xyzw)
        target_matrix = T.quat2mat(target)
        current_matrix = T.quat2mat(current)
        delta_matrix = target_matrix @ current_matrix.T
        return np.asarray(T.quat2axisangle(T.mat2quat(delta_matrix)), dtype=np.float64)

    def _action_for_pose(self, target: "Pose") -> tuple[FloatArray, float, float]:
        current = self._env.end_effector_pose()
        position_error = target.position - current.position
        rotation_error = self.orientation_error(target.quaternion, current.quaternion)

        action = np.zeros(7, dtype=np.float64)
        action[:3] = np.clip(
            position_error / self.max_position_delta,
            -1.0,
            1.0,
        )
        action[3:6] = np.clip(
            rotation_error / self.max_rotation_delta,
            -1.0,
            1.0,
        )
        action[6] = self._gripper_command
        return action, float(np.linalg.norm(position_error)), float(np.linalg.norm(rotation_error))

    def add_command_observer(
        self,
        observer: Callable[[CartesianCommandEvent], None],
    ) -> None:
        if observer not in self._command_observers:
            self._command_observers.append(observer)

    def remove_command_observer(
        self,
        observer: Callable[[CartesianCommandEvent], None],
    ) -> None:
        if observer in self._command_observers:
            self._command_observers.remove(observer)

    def _notify_command(self, current: "Pose", target: "Pose") -> None:
        event = CartesianCommandEvent(
            self._command_step,
            current,
            target,
            self._gripper_command,
        )
        self._command_step += 1
        for observer in tuple(self._command_observers):
            observer(event)

    def command_pose_once(self, target: "Pose") -> MotionResult:
        """Submit one trusted OSC pose command for one control timestep."""

        current = self._env.end_effector_pose()
        action, _, _ = self._action_for_pose(target)
        self._notify_command(current, target)
        self._env.step(action)
        self._env.render_if_enabled()
        updated = self._env.end_effector_pose()
        position_error = float(np.linalg.norm(target.position - updated.position))
        orientation_error = float(
            np.linalg.norm(self.orientation_error(target.quaternion, updated.quaternion))
        )
        return MotionResult(
            position_error <= self.position_tolerance
            and orientation_error <= self.orientation_tolerance,
            1,
            position_error,
            orientation_error,
        )

    def move_to_pose(
        self,
        target: "Pose",
        *,
        max_steps: int = 250,
        position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
    ) -> MotionResult:
        """Run a bounded closed loop toward an absolute Cartesian target."""

        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        position_tolerance = (
            self.position_tolerance
            if position_tolerance is None
            else float(position_tolerance)
        )
        orientation_tolerance = (
            self.orientation_tolerance
            if orientation_tolerance is None
            else float(orientation_tolerance)
        )
        if position_tolerance <= 0.0 or orientation_tolerance <= 0.0:
            raise ValueError("motion tolerances must be positive")

        position_error = float("inf")
        orientation_error = float("inf")
        stable_steps = 0
        for step in range(1, max_steps + 1):
            current = self._env.end_effector_pose()
            action, position_error, orientation_error = self._action_for_pose(target)
            self._notify_command(current, target)
            self._env.step(action)
            self._env.render_if_enabled()

            if (
                position_error <= position_tolerance
                and orientation_error <= orientation_tolerance
            ):
                stable_steps += 1
                if stable_steps >= 5:
                    return MotionResult(True, step, position_error, orientation_error)
            else:
                stable_steps = 0

        return MotionResult(False, max_steps, position_error, orientation_error)

    def set_gripper(self, *, open: bool, steps: int = 40) -> None:
        """Open or close the Panda gripper while holding the current arm pose."""

        if steps <= 0:
            raise ValueError("steps must be positive")
        self._gripper_command = -1.0 if open else 1.0
        for _ in range(steps):
            current = self._env.end_effector_pose()
            action = np.zeros(7, dtype=np.float64)
            action[6] = self._gripper_command
            self._notify_command(current, current)
            self._env.step(action)
            self._env.render_if_enabled()

    def hold(self, steps: int) -> None:
        """Hold the arm and current gripper command for a number of policy steps."""

        if steps < 0:
            raise ValueError("steps must be non-negative")
        for _ in range(steps):
            current = self._env.end_effector_pose()
            action = np.zeros(7, dtype=np.float64)
            action[6] = self._gripper_command
            self._notify_command(current, current)
            self._env.step(action)
            self._env.render_if_enabled()
