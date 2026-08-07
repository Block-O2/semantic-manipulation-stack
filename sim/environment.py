"""robosuite environment for the dynamic semantic tabletop playground."""

from __future__ import annotations

from typing import Any

import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.transform_utils import convert_quat

from robot.panda import Pose
from sim.scene import DEFAULT_SCENE, SceneConfig


class SemanticTabletopEnv(ManipulationEnv):
    """Panda tabletop environment with three cubes and three target regions."""

    def __init__(
        self,
        *,
        scene: SceneConfig = DEFAULT_SCENE,
        controller_configs: dict[str, Any] | None = None,
        has_renderer: bool = False,
        randomize_cube: bool = False,
        control_freq: int = 20,
        seed: int | None = None,
    ) -> None:
        self.scene_config = scene
        self.table_full_size = scene.table_size
        self.table_friction = scene.table_friction
        self.table_offset = np.asarray(scene.table_offset, dtype=np.float64)
        self.randomize_cube = randomize_cube
        self._render_enabled = has_renderer
        self._object_body_ids: dict[str, int] = {}
        self._cube_objects: dict[str, BoxObject] = {}
        self._target_objects: dict[str, BoxObject] = {}
        self._active_objects = set(scene.cube_names)

        if controller_configs is None:
            basic_config = load_composite_controller_config(controller="BASIC")
            controller_configs = {
                "type": basic_config["type"],
                "body_parts": {"right": basic_config["body_parts"]["right"]},
            }

        super().__init__(
            robots="Panda",
            env_configuration="default",
            controller_configs=controller_configs,
            gripper_types="default",
            base_types="default",
            initialization_noise=None,
            use_camera_obs=False,
            has_renderer=has_renderer,
            has_offscreen_renderer=False,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            control_freq=control_freq,
            horizon=100_000,
            ignore_done=True,
            hard_reset=False,
            renderer="mjviewer",
            seed=seed,
        )

    @property
    def object_names(self) -> tuple[str, ...]:
        return self.manipulable_object_names + self.target_names

    @property
    def manipulable_object_names(self) -> tuple[str, ...]:
        return self.scene_config.cube_names

    @property
    def target_names(self) -> tuple[str, ...]:
        return self.scene_config.target_names

    def reward(self, action: np.ndarray | None = None) -> float:
        return 0.0

    def _load_model(self) -> None:
        super()._load_model()
        base_position = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(base_position)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        half = self.scene_config.cube_half_size
        self._cube_objects = {}
        for name in self.scene_config.cube_names:
            cube = BoxObject(
                name=name,
                size_min=[half, half, half],
                size_max=[half, half, half],
                rgba=self.scene_config.cube_rgba[name],
                friction=self.scene_config.object_friction,
                rng=self.rng,
            )
            self._cube_objects[name] = cube
            setattr(self, name, cube)

        self._target_objects = {}
        for name in self.scene_config.target_names:
            target = BoxObject(
                name=name,
                size_min=self.scene_config.target_half_size,
                size_max=self.scene_config.target_half_size,
                rgba=self.scene_config.target_rgba[name],
                joints=None,
                obj_type="visual",
                duplicate_collision_geoms=False,
                rng=self.rng,
            )
            target.get_obj().set(
                "pos", array_to_string(self.scene_config.target_position(name))
            )
            self._target_objects[name] = target
            setattr(self, name, target)

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[*self._cube_objects.values(), *self._target_objects.values()],
        )

    def _setup_references(self) -> None:
        super()._setup_references()
        all_objects = {**self._cube_objects, **self._target_objects}
        self._object_body_ids = {
            name: self.sim.model.body_name2id(obj.root_body)
            for name, obj in all_objects.items()
        }
        eef_ids = self.robots[0].eef_site_id
        if isinstance(eef_ids, dict):
            self._eef_site_id = eef_ids.get("right", next(iter(eef_ids.values())))
        else:
            self._eef_site_id = int(eef_ids)

    def _random_cube_positions(self) -> dict[str, np.ndarray]:
        x_range, y_range = self.scene_config.placement_xy_range
        positions: dict[str, np.ndarray] = {}
        for name in self.manipulable_object_names:
            for _ in range(100):
                candidate = np.array(
                    [
                        self.rng.uniform(*x_range),
                        self.rng.uniform(*y_range),
                        self.scene_config.table_top_z + self.scene_config.cube_half_size,
                    ],
                    dtype=np.float64,
                )
                if all(
                    np.linalg.norm(candidate[:2] - other[:2])
                    >= self.scene_config.minimum_random_cube_separation
                    for other in positions.values()
                ):
                    positions[name] = candidate
                    break
            else:
                raise RuntimeError("could not sample non-overlapping cube positions")
        return positions

    def _reset_internal(self) -> None:
        super()._reset_internal()
        self._active_objects = set(self.manipulable_object_names)
        positions = (
            self._random_cube_positions()
            if self.randomize_cube
            else {
                name: self.scene_config.cube_position(name)
                for name in self.manipulable_object_names
            }
        )
        for name, cube in self._cube_objects.items():
            qpos = np.concatenate(
                [positions[name], np.array([1.0, 0.0, 0.0, 0.0])]
            )
            self.sim.data.set_joint_qpos(cube.joints[0], qpos)
            self.sim.data.set_joint_qvel(cube.joints[0], np.zeros(6))
        for name in self.target_names:
            self.sim.model.body_pos[self._object_body_ids[name]] = (
                self.scene_config.target_position(name)
            )
        self.sim.forward()

    def object_exists(self, name: str) -> bool:
        if name in self._cube_objects:
            return name in self._active_objects
        if name in self._target_objects:
            return True
        raise KeyError(f"Unknown entity {name!r}")

    def end_effector_pose(self) -> Pose:
        position = np.array(self.sim.data.site_xpos[self._eef_site_id], dtype=np.float64)
        rotation = np.array(
            self.sim.data.site_xmat[self._eef_site_id], dtype=np.float64
        ).reshape(3, 3)
        from robosuite.utils.transform_utils import mat2quat

        return Pose(position=position, quaternion=np.asarray(mat2quat(rotation)))

    def object_pose(self, name: str) -> Pose:
        try:
            body_id = self._object_body_ids[name]
        except KeyError as exc:
            available = ", ".join(self.object_names)
            raise KeyError(f"Unknown entity {name!r}; available: {available}") from exc
        position = np.array(self.sim.data.body_xpos[body_id], dtype=np.float64)
        quaternion_wxyz = np.array(self.sim.data.body_xquat[body_id], dtype=np.float64)
        quaternion_xyzw = np.asarray(convert_quat(quaternion_wxyz, to="xyzw"))
        return Pose(position=position, quaternion=quaternion_xyzw)

    def object_linear_velocity(self, name: str) -> np.ndarray:
        if name not in self._cube_objects:
            raise ValueError(f"Entity {name!r} is not a movable object")
        body_name = self.sim.model.body_id2name(self._object_body_ids[name])
        return np.asarray(
            self.sim.data.get_body_xvelp(body_name), dtype=np.float64
        ).copy()

    def target_half_extents(self, name: str) -> np.ndarray:
        if name not in self._target_objects:
            if name not in self.object_names:
                raise KeyError(f"Unknown entity {name!r}")
            raise ValueError(f"Entity {name!r} is not a target region")
        return np.asarray(self.scene_config.target_half_size, dtype=np.float64).copy()

    def is_object_center_inside_target(
        self,
        object_name: str,
        target_name: str,
        *,
        margin: float = 0.0,
    ) -> bool:
        if margin < 0.0:
            raise ValueError("margin must be non-negative")
        if not self.object_exists(object_name):
            return False
        object_position = self.object_pose(object_name).position
        target_position = self.object_pose(target_name).position
        usable_xy = self.target_half_extents(target_name)[:2] - margin
        return bool(
            np.all(usable_xy >= 0.0)
            and np.all(np.abs(object_position[:2] - target_position[:2]) <= usable_xy)
        )

    def is_grasping_object(self, name: str) -> bool:
        if name not in self._cube_objects:
            if name not in self.object_names:
                raise KeyError(f"Unknown entity {name!r}")
            return False
        if not self.object_exists(name):
            return False
        return bool(
            self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self._cube_objects[name],
            )
        )

    def render_if_enabled(self) -> None:
        if self._render_enabled:
            self.render()


def make_environment(
    *,
    scene: SceneConfig = DEFAULT_SCENE,
    render: bool = False,
    randomize_cube: bool = False,
    seed: int | None = None,
) -> SemanticTabletopEnv:
    return SemanticTabletopEnv(
        scene=scene,
        has_renderer=render,
        randomize_cube=randomize_cube,
        seed=seed,
    )
