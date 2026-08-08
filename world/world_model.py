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
    PushRegionState,
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

    @property
    def push_region_names(self) -> tuple[str, ...]:
        return self._env.scene_config.push_region_names

    def pose(self, name: str) -> Pose:
        return self._env.object_pose(name)

    def exists(self, name: str) -> bool:
        return self._env.object_exists(name)

    def is_reachable(self, name: str) -> bool:
        return self.exists(name) and self._semantically_reachable(self.pose(name))

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

    def push_region_bounds(self, name: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        try:
            lower, upper = self._env.scene_config.push_region_bounds[name]
        except KeyError as exc:
            raise KeyError(f"Unknown push region {name!r}") from exc
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    def push_region_center(self, name: str) -> NDArray[np.float64]:
        try:
            return self._env.scene_config.push_region_center(name)
        except KeyError as exc:
            raise KeyError(f"Unknown push region {name!r}") from exc

    def is_inside_push_region(self, object_name: str, region_name: str) -> bool:
        if not self.exists(object_name):
            return False
        lower, upper = self.push_region_bounds(region_name)
        xy = self.pose(object_name).position[:2]
        return bool(np.all(xy >= lower) and np.all(xy <= upper))

    def is_on_table(self, object_name: str, *, height_tolerance: float = 0.04) -> bool:
        if height_tolerance <= 0.0:
            raise ValueError("height_tolerance must be positive")
        if not self.exists(object_name):
            return False
        pose = self.pose(object_name)
        expected_z = (
            self._env.scene_config.table_top_z
            + self._env.scene_config.cube_half_size
        )
        half_x = 0.5 * self._env.scene_config.table_size[0]
        half_y = 0.5 * self._env.scene_config.table_size[1]
        return bool(
            abs(float(pose.position[2]) - expected_z) <= height_tolerance
            and abs(float(pose.position[0])) <= half_x
            and abs(float(pose.position[1])) <= half_y
        )

    def is_push_path_safe(
        self,
        start_position: NDArray[np.float64],
        end_position: NDArray[np.float64],
        *,
        margin: float | None = None,
    ) -> bool:
        clearance = (
            self._env.scene_config.cube_half_size if margin is None else margin
        )
        if clearance < 0.0:
            raise ValueError("push-path margin must be non-negative")
        half_x = 0.5 * self._env.scene_config.table_size[0] - clearance
        half_y = 0.5 * self._env.scene_config.table_size[1] - clearance
        for position in (start_position, end_position):
            if (
                abs(float(position[0])) > half_x
                or abs(float(position[1])) > half_y
            ):
                return False
        return True

    def holding_object(self) -> str | None:
        for name in self._env.manipulable_object_names:
            if self.is_grasped(name):
                return name
        return None

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
            occupied_by: list[str] = []
            for object_name in object_names:
                inside = self.is_inside_target(
                    object_name,
                    target_name,
                    margin=self.thresholds.inside_margin,
                )
                relations[WorldState.relation_key(object_name, target_name)] = inside
                if inside:
                    occupied_by.append(object_name)
            targets[target_name] = TargetState(
                exists=True,
                pose=PoseState.from_pose(target_pose),
                reachable=self._semantically_reachable(target_pose),
                occupied=bool(occupied_by),
                occupied_by=tuple(sorted(occupied_by)),
                role=(
                    "temporary"
                    if target_name in self._env.scene_config.temporary_target_names
                    else "destination"
                ),
            )

        push_regions: dict[str, PushRegionState] = {}
        for region_name in self.push_region_names:
            lower, upper = self.push_region_bounds(region_name)
            center = self.push_region_center(region_name)
            push_regions[region_name] = PushRegionState(
                exists=True,
                reachable=self._semantically_reachable(
                    Pose(center, np.array([0.0, 0.0, 0.0, 1.0]))
                ),
                center=tuple(float(value) for value in center),
                lower_xy=tuple(float(value) for value in lower),
                upper_xy=tuple(float(value) for value in upper),
            )
            for object_name in object_names:
                relations[
                    WorldState.push_region_key(object_name, region_name)
                ] = self.is_inside_push_region(object_name, region_name)

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
            push_regions=push_regions,
        )
