"""Cartesian and gripper primitives built exclusively on :class:`PandaRobot`."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from primitives.results import PrimitiveFailure, PrimitiveResult
from primitives.workspace import DEFAULT_WORKSPACE, WorkspaceBounds
from robot import PandaRobot, Pose


FloatArray = NDArray[np.float64]


def _quaternion_angle(first: FloatArray, second: FloatArray) -> float:
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def _slerp(start: FloatArray, target: FloatArray, fraction: float) -> FloatArray:
    """Shortest-path quaternion interpolation in xyzw order."""

    start = start / np.linalg.norm(start)
    target = target / np.linalg.norm(target)
    dot = float(np.dot(start, target))
    if dot < 0.0:
        target = -target
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = start + fraction * (target - start)
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    sin_angle = math.sin(angle)
    return (
        math.sin((1.0 - fraction) * angle) / sin_angle * start
        + math.sin(fraction * angle) / sin_angle * target
    )


class ManipulationPrimitives:
    """Safe primitive API intended for future task-level skills."""

    def __init__(
        self,
        robot: PandaRobot,
        *,
        workspace: WorkspaceBounds = DEFAULT_WORKSPACE,
    ) -> None:
        self._robot = robot
        self.workspace = workspace

    @property
    def current_pose(self) -> Pose:
        """Read the robot pose through the primitive-layer boundary."""

        return self._robot.pose

    @staticmethod
    def linear_waypoints(
        start: Pose,
        target: Pose,
        *,
        max_cartesian_step: float,
        max_orientation_step: float,
    ) -> tuple[Pose, ...]:
        """Create fixed straight-line position waypoints with quaternion SLERP."""

        if max_cartesian_step <= 0.0 or max_orientation_step <= 0.0:
            raise ValueError("trajectory step sizes must be positive")
        distance = float(np.linalg.norm(target.position - start.position))
        angle = _quaternion_angle(start.quaternion, target.quaternion)
        segment_count = max(
            1,
            math.ceil(distance / max_cartesian_step),
            math.ceil(angle / max_orientation_step),
        )
        return tuple(
            Pose(
                start.position + (target.position - start.position) * (index / segment_count),
                _slerp(start.quaternion, target.quaternion, index / segment_count),
            )
            for index in range(1, segment_count + 1)
        )

    @staticmethod
    def _errors(current: Pose, target: Pose) -> tuple[float, float]:
        return (
            float(np.linalg.norm(target.position - current.position)),
            _quaternion_angle(current.quaternion, target.quaternion),
        )

    @staticmethod
    def _failure_for_errors(
        position_error: float,
        orientation_error: float,
        position_tolerance: float,
        orientation_tolerance: float,
    ) -> PrimitiveFailure | None:
        if position_error > position_tolerance:
            return PrimitiveFailure.POSITION_NOT_CONVERGED
        if orientation_error > orientation_tolerance:
            return PrimitiveFailure.ORIENTATION_NOT_CONVERGED
        return None

    def _outside_workspace(self, poses: Iterable[Pose]) -> bool:
        return any(not self.workspace.contains(pose.position) for pose in poses)

    def _execute_waypoints(
        self,
        waypoints: tuple[Pose, ...],
        final_target: Pose,
        *,
        position_tolerance: float,
        orientation_tolerance: float,
        max_steps: int,
    ) -> PrimitiveResult:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if position_tolerance <= 0.0 or orientation_tolerance <= 0.0:
            raise ValueError("motion tolerances must be positive")
        if self._outside_workspace(waypoints):
            return PrimitiveResult(
                False,
                PrimitiveFailure.TARGET_OUTSIDE_WORKSPACE,
                0,
                None,
                None,
            )

        total_steps = 0
        for waypoint in waypoints:
            remaining_steps = max_steps - total_steps
            if remaining_steps <= 0:
                position_error, orientation_error = self._errors(self._robot.pose, final_target)
                return PrimitiveResult(
                    False,
                    PrimitiveFailure.TIMEOUT,
                    total_steps,
                    position_error,
                    orientation_error,
                )

            motion = self._robot.move_end_effector(
                waypoint,
                max_steps=remaining_steps,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
            )
            total_steps += motion.steps
            if not motion.reached:
                position_error, orientation_error = self._errors(self._robot.pose, final_target)
                reason = (
                    PrimitiveFailure.TIMEOUT
                    if total_steps >= max_steps
                    else self._failure_for_errors(
                        motion.position_error,
                        motion.orientation_error,
                        position_tolerance,
                        orientation_tolerance,
                    )
                )
                return PrimitiveResult(
                    False,
                    reason or PrimitiveFailure.TIMEOUT,
                    total_steps,
                    position_error,
                    orientation_error,
                )

        position_error, orientation_error = self._errors(self._robot.pose, final_target)
        reason = self._failure_for_errors(
            position_error,
            orientation_error,
            position_tolerance,
            orientation_tolerance,
        )
        return PrimitiveResult(
            reason is None,
            reason,
            total_steps,
            position_error,
            orientation_error,
        )

    def move_linear(
        self,
        target: Pose,
        *,
        position_tolerance: float = 0.006,
        orientation_tolerance: float = 0.06,
        max_steps: int = 300,
        max_cartesian_step: float = 0.025,
        max_orientation_step: float = 0.15,
    ) -> PrimitiveResult:
        """Follow fixed waypoints on the straight segment to ``target``."""

        start = self._robot.pose
        if not self.workspace.contains(target.position):
            return PrimitiveResult(
                False,
                PrimitiveFailure.TARGET_OUTSIDE_WORKSPACE,
                0,
                None,
                None,
            )
        waypoints = self.linear_waypoints(
            start,
            target,
            max_cartesian_step=max_cartesian_step,
            max_orientation_step=max_orientation_step,
        )
        return self._execute_waypoints(
            waypoints,
            target,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            max_steps=max_steps,
        )

    def move_to_pose(
        self,
        target: Pose,
        *,
        position_tolerance: float = 0.006,
        orientation_tolerance: float = 0.06,
        max_steps: int = 300,
        max_cartesian_step: float = 0.03,
        max_orientation_step: float = 0.20,
    ) -> PrimitiveResult:
        """Move using bounded receding Cartesian targets.

        Unlike :meth:`move_linear`, each next waypoint is recomputed from the
        measured pose, allowing correction after disturbances without claiming
        that the executed path remained on one fixed Cartesian line.
        """

        if not self.workspace.contains(target.position):
            return PrimitiveResult(
                False,
                PrimitiveFailure.TARGET_OUTSIDE_WORKSPACE,
                0,
                None,
                None,
            )
        if max_cartesian_step <= 0.0 or max_orientation_step <= 0.0:
            raise ValueError("trajectory step sizes must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        total_steps = 0
        while total_steps < max_steps:
            current = self._robot.pose
            position_error, orientation_error = self._errors(current, target)
            if (
                position_error <= position_tolerance
                and orientation_error <= orientation_tolerance
            ):
                return PrimitiveResult(True, None, total_steps, position_error, orientation_error)

            position_fraction = min(1.0, max_cartesian_step / max(position_error, 1e-12))
            orientation_fraction = min(
                1.0, max_orientation_step / max(orientation_error, 1e-12)
            )
            waypoint = Pose(
                current.position + (target.position - current.position) * position_fraction,
                _slerp(current.quaternion, target.quaternion, orientation_fraction),
            )
            if not self.workspace.contains(waypoint.position):
                return PrimitiveResult(
                    False,
                    PrimitiveFailure.TARGET_OUTSIDE_WORKSPACE,
                    total_steps,
                    position_error,
                    orientation_error,
                )

            remaining_steps = max_steps - total_steps
            motion = self._robot.move_end_effector(
                waypoint,
                max_steps=remaining_steps,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
            )
            total_steps += motion.steps
            if not motion.reached:
                final_position_error, final_orientation_error = self._errors(
                    self._robot.pose, target
                )
                reason = (
                    PrimitiveFailure.TIMEOUT
                    if total_steps >= max_steps
                    else self._failure_for_errors(
                        motion.position_error,
                        motion.orientation_error,
                        position_tolerance,
                        orientation_tolerance,
                    )
                )
                return PrimitiveResult(
                    False,
                    reason or PrimitiveFailure.TIMEOUT,
                    total_steps,
                    final_position_error,
                    final_orientation_error,
                )

        position_error, orientation_error = self._errors(self._robot.pose, target)
        return PrimitiveResult(
            False,
            PrimitiveFailure.TIMEOUT,
            total_steps,
            position_error,
            orientation_error,
        )

    def open_gripper(self, *, steps: int = 40) -> PrimitiveResult:
        self._robot.open_gripper(steps=steps)
        return PrimitiveResult(True, None, steps, None, None)

    def close_gripper(self, *, steps: int = 50) -> PrimitiveResult:
        self._robot.close_gripper(steps=steps)
        return PrimitiveResult(True, None, steps, None, None)

    def wait(self, *, steps: int) -> PrimitiveResult:
        """Advance bounded control steps while preserving the robot command."""

        if steps < 0:
            raise ValueError("steps must be non-negative")
        self._robot.hold(steps)
        return PrimitiveResult(True, None, steps, None, None)
