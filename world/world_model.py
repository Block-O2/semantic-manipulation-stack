"""Ground-truth world state exposed independently of robosuite observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from primitives import DEFAULT_WORKSPACE
from robot.panda import Pose
from world.state import (
    ObjectState,
    PoseState,
    RobotState,
    SemanticThresholds,
    TargetState,
    WorldState,
)

if TYPE_CHECKING:
    from sim.environment import SemanticTabletopEnv


class WorldModel:
    """Read-only object-pose view for planners and future task primitives."""

    def __init__(
        self,
        env: "SemanticTabletopEnv",
        *,
        thresholds: SemanticThresholds = SemanticThresholds(),
    ) -> None:
        self._env = env
        self.thresholds = thresholds

    @property
    def object_names(self) -> tuple[str, ...]:
        return self._env.object_names

    def pose(self, name: str) -> Pose:
        return self._env.object_pose(name)

    def exists(self, name: str) -> bool:
        return self._env.object_exists(name)

    def snapshot(self) -> dict[str, Pose]:
        return {name: self.pose(name) for name in self.object_names}

    def is_grasped(self, name: str) -> bool:
        """Read the simulator contact-based grasp state without exposing it upstream."""

        return self._env.is_grasping_object(name)

    def linear_velocity(self, name: str) -> NDArray[np.float64]:
        """Return ground-truth world-frame linear velocity in metres/second."""

        return self._env.object_linear_velocity(name)

    def target_half_extents(self, name: str) -> NDArray[np.float64]:
        """Return target-region half extents without exposing simulator objects."""

        return self._env.target_half_extents(name)

    def is_inside_target(
        self,
        object_name: str,
        target_name: str,
        *,
        margin: float = 0.0,
    ) -> bool:
        """Return whether the object's centre is within the target XY bounds."""

        return self._env.is_object_center_inside_target(
            object_name,
            target_name,
            margin=margin,
        )

    def is_target_occupied(
        self,
        target_name: str,
        *,
        exclude_object: str | None = None,
    ) -> bool:
        """Return whether any existing cube other than an optional exclusion is inside."""

        return any(
            name != exclude_object
            and self.exists(name)
            and self.is_inside_target(
                name,
                target_name,
                margin=self.thresholds.inside_margin,
            )
            for name in self._env.manipulable_object_names
        )

    def is_stable(self, name: str, *, maximum_speed: float = 0.03) -> bool:
        """Return whether object translation is below a configured speed."""

        if maximum_speed <= 0.0:
            raise ValueError("maximum_speed must be positive")
        return bool(np.linalg.norm(self.linear_velocity(name)) <= maximum_speed)

    @staticmethod
    def _semantically_reachable(pose: Pose) -> bool:
        """Conservative point-level reachability for task-state validation."""

        position = pose.position
        return bool(
            np.all(position[:2] >= DEFAULT_WORKSPACE.lower[:2])
            and np.all(position[:2] <= DEFAULT_WORKSPACE.upper[:2])
            and position[2] >= DEFAULT_WORKSPACE.lower[2] - 0.01
            and position[2] <= DEFAULT_WORKSPACE.upper[2]
        )

    def semantic_state(self) -> WorldState:
        """Build a serializable snapshot containing only task-relevant facts."""

        object_names = self._env.manipulable_object_names
        target_names = self._env.target_names
        objects: dict[str, ObjectState] = {}
        holding: str | None = None
        for name in object_names:
            exists = self.exists(name)
            if not exists:
                objects[name] = ObjectState(
                    exists=False,
                    pose=None,
                    grasped=False,
                    reachable=False,
                )
                continue
            pose = self.pose(name)
            grasped = self.is_grasped(name)
            if grasped:
                holding = name
            objects[name] = ObjectState(
                exists=exists,
                pose=PoseState.from_pose(pose),
                grasped=grasped,
                reachable=self._semantically_reachable(pose),
            )

        relations: dict[str, bool] = {}
        targets: dict[str, TargetState] = {}
        for target_name in target_names:
            target_pose = self.pose(target_name)
            occupied = False
            for object_name in object_names:
                inside = self.is_inside_target(
                    object_name,
                    target_name,
                    margin=self.thresholds.inside_margin,
                )
                relations[WorldState.relation_key(object_name, target_name)] = inside
                occupied = occupied or inside
            targets[target_name] = TargetState(
                exists=True,
                pose=PoseState.from_pose(target_pose),
                reachable=self._semantically_reachable(target_pose),
                occupied=occupied,
            )

        for first_name in object_names:
            first = objects[first_name]
            for second_name in object_names:
                if first_name == second_name:
                    continue
                second = objects[second_name]
                if not first.exists or not second.exists:
                    left_of = right_of = near = False
                else:
                    assert first.pose is not None and second.pose is not None
                    first_xy = np.asarray(first.pose.position[:2])
                    second_xy = np.asarray(second.pose.position[:2])
                    delta_x = float(first_xy[0] - second_xy[0])
                    left_of = delta_x < -self.thresholds.left_right_tolerance
                    right_of = delta_x > self.thresholds.left_right_tolerance
                    near = bool(
                        np.linalg.norm(first_xy - second_xy)
                        <= self.thresholds.near_distance
                    )
                relations[WorldState.left_of_key(first_name, second_name)] = left_of
                relations[WorldState.right_of_key(first_name, second_name)] = right_of
                relations[WorldState.near_key(first_name, second_name)] = near

        return WorldState(
            robot=RobotState(holding=holding),
            objects=objects,
            targets=targets,
            relations=relations,
        )
