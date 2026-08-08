"""Shared physically valid initial-state sampling for Push experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from playground import WorldController
from primitives import ManipulationPrimitives
from robot import PandaRobot, Pose
from world import WorldModel


EVALUATION_STANDBY_POSITION = np.array([-0.25, -0.25, 1.10])
DEFAULT_PUSH_X_RANGE = (-0.10, 0.08)
DEFAULT_PUSH_Y_RANGE = (-0.15, 0.12)


def prepare_push_scene(
    world: WorldModel,
    robot: PandaRobot,
    primitives: ManipulationPrimitives,
    controller: WorldController,
) -> None:
    """Move the robot away and isolate red_cube before randomized placement."""

    standby = Pose(EVALUATION_STANDBY_POSITION, robot.pose.quaternion)
    if not primitives.move_to_pose(standby).success:
        raise RuntimeError("could not move robot to Push evaluation standby pose")
    if not primitives.open_gripper().success:
        raise RuntimeError("could not open gripper during Push evaluation setup")
    for name in ("green_cube", "blue_cube"):
        if world.exists(name):
            controller.remove_object(name)


def sample_valid_cube_position(
    world: WorldModel,
    primitives: ManipulationPrimitives,
    controller: WorldController,
    rng: np.random.Generator,
    *,
    x_range: tuple[float, float] = DEFAULT_PUSH_X_RANGE,
    y_range: tuple[float, float] = DEFAULT_PUSH_Y_RANGE,
    maximum_attempts: int = 100,
    settle_steps: int = 10,
    maximum_settle_displacement: float = 0.003,
) -> tuple[NDArray[np.float64], int]:
    """Sample broadly, rejecting only configurations unstable before execution."""

    if maximum_attempts <= 0 or settle_steps < 0:
        raise ValueError("sampling limits must be positive")
    for rejection_count in range(maximum_attempts):
        sampled = np.array(
            [rng.uniform(*x_range), rng.uniform(*y_range)],
            dtype=np.float64,
        )
        controller.move_object("red_cube", float(sampled[0]), float(sampled[1]))
        primitives.wait(steps=settle_steps)
        settled = world.pose("red_cube").position
        if (
            np.linalg.norm(settled[:2] - sampled) <= maximum_settle_displacement
            and world.is_on_table("red_cube")
        ):
            return settled.copy(), rejection_count
    raise RuntimeError("could not sample a physically valid cube position")
