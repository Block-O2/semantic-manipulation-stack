"""Dedicated external controller for deterministic simulated-world changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from robosuite.utils.transform_utils import convert_quat

from sim import SemanticTabletopEnv
from world import WorldModel


@dataclass(frozen=True)
class WorldUpdate:
    operation: str
    entity: str
    position: tuple[float, float, float] | None


class WorldController:
    """Modify MuJoCo world state outside all production execution layers."""

    def __init__(self, env: SemanticTabletopEnv, world: WorldModel) -> None:
        self._env = env
        self._world = world
        self._restore_poses = {
            name: world.pose(name) for name in env.manipulable_object_names
        }

    def _require_object(self, name: str) -> None:
        if name not in self._env.manipulable_object_names:
            raise ValueError(f"Unknown movable object {name!r}")

    def _require_target(self, name: str) -> None:
        if name not in self._env.target_names:
            raise ValueError(f"Unknown target {name!r}")

    def _set_object_pose(
        self,
        name: str,
        position: np.ndarray,
        quaternion_xyzw: np.ndarray,
    ) -> None:
        cube = self._env._cube_objects[name]
        quaternion_wxyz = np.asarray(convert_quat(quaternion_xyzw, to="wxyz"))
        self._env.sim.data.set_joint_qpos(
            cube.joints[0],
            np.concatenate([position, quaternion_wxyz]),
        )
        self._env.sim.data.set_joint_qvel(cube.joints[0], np.zeros(6))
        self._env.sim.forward()

    def move_object(
        self,
        name: str,
        x: float,
        y: float,
        *,
        z: float | None = None,
    ) -> WorldUpdate:
        self._require_object(name)
        if not self._env.object_exists(name):
            raise ValueError(f"Object {name!r} is removed; restore it before moving")
        pose = self._world.pose(name)
        position = np.array(
            [
                float(x),
                float(y),
                self._env.scene_config.table_top_z
                + self._env.scene_config.cube_half_size
                if z is None
                else float(z),
            ],
            dtype=np.float64,
        )
        self._set_object_pose(name, position, pose.quaternion)
        return WorldUpdate("move_object", name, tuple(float(v) for v in position))

    def move_target(self, name: str, x: float, y: float) -> WorldUpdate:
        self._require_target(name)
        body_id = self._env._object_body_ids[name]
        position = np.array(
            [float(x), float(y), self._env.scene_config.target_position(name)[2]],
            dtype=np.float64,
        )
        self._env.sim.model.body_pos[body_id] = position
        self._env.sim.forward()
        return WorldUpdate("move_target", name, tuple(float(v) for v in position))

    def drop_held_object(
        self,
        *,
        x: float | None = None,
        y: float | None = None,
    ) -> WorldUpdate:
        holding = self._world.semantic_state().robot.holding
        if holding is None:
            raise ValueError("Robot is not holding an object")
        pose = self._world.pose(holding)
        return self.move_object(
            holding,
            float(pose.position[0] if x is None else x),
            float(pose.position[1] if y is None else y),
        )

    def place_object_in_target(self, object_name: str, target_name: str) -> WorldUpdate:
        self._require_object(object_name)
        self._require_target(target_name)
        target = self._world.pose(target_name)
        update = self.move_object(
            object_name,
            float(target.position[0]),
            float(target.position[1]),
        )
        return WorldUpdate(
            "place_object_in_target",
            f"{object_name}:{target_name}",
            update.position,
        )

    def remove_object(self, name: str) -> WorldUpdate:
        self._require_object(name)
        if not self._env.object_exists(name):
            raise ValueError(f"Object {name!r} is already removed")
        self._restore_poses[name] = self._world.pose(name)
        pose = self._restore_poses[name]
        self._env._active_objects.remove(name)
        self._set_object_pose(
            name,
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
            pose.quaternion,
        )
        return WorldUpdate("remove_object", name, None)

    def restore_object(self, name: str) -> WorldUpdate:
        self._require_object(name)
        if self._env.object_exists(name):
            raise ValueError(f"Object {name!r} already exists")
        pose = self._restore_poses[name]
        position = pose.position.copy()
        position[2] = (
            self._env.scene_config.table_top_z + self._env.scene_config.cube_half_size
        )
        self._env._active_objects.add(name)
        self._set_object_pose(name, position, pose.quaternion)
        return WorldUpdate("restore_object", name, tuple(float(v) for v in position))
