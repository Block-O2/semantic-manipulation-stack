from __future__ import annotations

import numpy as np

from primitives import (
    ManipulationPrimitives,
    PrimitiveFailure,
    PrimitiveResult,
    WorkspaceBounds,
)
from robot import MotionResult, Pose


class FakeRobot:
    def __init__(self, pose: Pose, *, fail_motion: bool = False) -> None:
        self._pose = pose
        self.fail_motion = fail_motion
        self.targets: list[Pose] = []
        self.gripper_events: list[tuple[str, int]] = []

    @property
    def pose(self) -> Pose:
        return self._pose

    def move_end_effector(
        self,
        target: Pose,
        *,
        max_steps: int,
        position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
    ) -> MotionResult:
        self.targets.append(target)
        if self.fail_motion:
            return MotionResult(False, max_steps, 1.0, 1.0)
        self._pose = target
        return MotionResult(True, 1, 0.0, 0.0)

    def open_gripper(self, *, steps: int) -> None:
        self.gripper_events.append(("open", steps))

    def close_gripper(self, *, steps: int) -> None:
        self.gripper_events.append(("close", steps))


START = Pose.from_values([0.0, 0.0, 0.9], [0.0, 0.0, 0.0, 1.0])
WORKSPACE = WorkspaceBounds(
    lower=np.array([-0.5, -0.5, 0.8]),
    upper=np.array([0.5, 0.5, 1.2]),
)


def test_move_linear_interpolates_on_straight_line_with_bounded_steps() -> None:
    robot = FakeRobot(START)
    primitives = ManipulationPrimitives(robot, workspace=WORKSPACE)
    target = Pose.from_values([0.12, 0.06, 0.96], [0.0, 0.0, 0.0, 1.0])

    result = primitives.move_linear(target, max_cartesian_step=0.025)

    assert result.success
    assert len(robot.targets) > 1
    points = np.vstack([START.position, *(pose.position for pose in robot.targets)])
    step_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert np.all(step_lengths <= 0.025 + 1e-12)
    direction = target.position - START.position
    relative = points - START.position
    assert np.all(np.linalg.norm(np.cross(relative, direction), axis=1) < 1e-12)
    np.testing.assert_allclose(points[-1], target.position)


def test_linear_waypoints_slerp_orientation_with_bounded_angle() -> None:
    target = Pose.from_values(
        START.position,
        [0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)],
    )

    waypoints = ManipulationPrimitives.linear_waypoints(
        START,
        target,
        max_cartesian_step=0.02,
        max_orientation_step=0.2,
    )

    quaternions = [START.quaternion, *(pose.quaternion for pose in waypoints)]
    angular_steps = [
        2.0 * np.arccos(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
        for first, second in zip(quaternions, quaternions[1:])
    ]
    assert len(waypoints) == 8
    assert max(angular_steps) <= 0.2 + 1e-12
    np.testing.assert_allclose(waypoints[-1].quaternion, target.quaternion)


def test_move_to_pose_uses_multiple_bounded_receding_targets() -> None:
    robot = FakeRobot(START)
    primitives = ManipulationPrimitives(robot, workspace=WORKSPACE)
    target = Pose.from_values([0.10, 0.0, 0.9], [0.0, 0.0, 0.0, 1.0])

    result = primitives.move_to_pose(target, max_cartesian_step=0.03)

    assert result == PrimitiveResult(True, None, 4, 0.0, 0.0)
    points = np.vstack([START.position, *(pose.position for pose in robot.targets)])
    assert np.all(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0.03 + 1e-12)


def test_workspace_rejection_does_not_command_robot() -> None:
    robot = FakeRobot(START)
    primitives = ManipulationPrimitives(robot, workspace=WORKSPACE)
    target = Pose.from_values([0.0, 0.0, 0.7], [0.0, 0.0, 0.0, 1.0])

    result = primitives.move_linear(target)

    assert result == PrimitiveResult(
        False,
        PrimitiveFailure.TARGET_OUTSIDE_WORKSPACE,
        0,
        None,
        None,
    )
    assert robot.targets == []


def test_motion_timeout_is_structured() -> None:
    robot = FakeRobot(START, fail_motion=True)
    primitives = ManipulationPrimitives(robot, workspace=WORKSPACE)
    target = Pose.from_values([0.05, 0.0, 0.9], [0.0, 0.0, 0.0, 1.0])

    result = primitives.move_to_pose(target, max_steps=7)

    assert not result.success
    assert result.reason is PrimitiveFailure.TIMEOUT
    assert result.steps == 7
    assert result.final_position_error == 0.05


def test_gripper_primitives_are_thin_robot_wrappers() -> None:
    robot = FakeRobot(START)
    primitives = ManipulationPrimitives(robot, workspace=WORKSPACE)

    opened = primitives.open_gripper(steps=12)
    closed = primitives.close_gripper(steps=18)

    assert opened.success and closed.success
    assert robot.gripper_events == [("open", 12), ("close", 18)]
