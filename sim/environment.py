"""robosuite environment for the semantic manipulation tabletop scene."""

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
    """Minimal Panda/table/cube/target environment.

    The environment owns MuJoCo-specific state access. Higher layers should use
    :class:`PandaRobot` and :class:`WorldModel`, rather than this class directly.
    """

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

        if controller_configs is None:
            basic_config = load_composite_controller_config(controller="BASIC")
            # The generic BASIC file also contains left arm, mobile-base,
            # torso, head, and leg entries. Panda only owns the right arm.
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
        return ("red_cube", "blue_target")

    @property
    def manipulable_object_names(self) -> tuple[str, ...]:
        return ("red_cube",)

    @property
    def target_names(self) -> tuple[str, ...]:
        return ("blue_target",)

    def reward(self, action: np.ndarray | None = None) -> float:
        """V0 is a control foundation, so it intentionally has no task reward."""

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
        self.red_cube = BoxObject(
            name="red_cube",
            size_min=[half, half, half],
            size_max=[half, half, half],
            rgba=self.scene_config.red_rgba,
            friction=self.scene_config.object_friction,
            rng=self.rng,
        )

        self.blue_target = BoxObject(
            name="blue_target",
            size_min=self.scene_config.target_half_size,
            size_max=self.scene_config.target_half_size,
            rgba=self.scene_config.blue_rgba,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=False,
            rng=self.rng,
        )
        self.blue_target.get_obj().set(
            "pos", array_to_string(self.scene_config.target_position)
        )

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.red_cube, self.blue_target],
        )

    def _setup_references(self) -> None:
        super()._setup_references()
        self._object_body_ids = {
            "red_cube": self.sim.model.body_name2id(self.red_cube.root_body),
            "blue_target": self.sim.model.body_name2id(self.blue_target.root_body),
        }

        eef_ids = self.robots[0].eef_site_id
        if isinstance(eef_ids, dict):
            self._eef_site_id = eef_ids.get("right", next(iter(eef_ids.values())))
        else:
            self._eef_site_id = int(eef_ids)

    def _reset_internal(self) -> None:
        super()._reset_internal()
        cube_position = self.scene_config.cube_position.copy()
        if self.randomize_cube:
            x_range, y_range = self.scene_config.placement_xy_range
            cube_position[0] = self.rng.uniform(*x_range)
            cube_position[1] = self.rng.uniform(*y_range)

        # Free-joint quaternions use MuJoCo's wxyz convention.
        cube_qpos = np.concatenate([cube_position, np.array([1.0, 0.0, 0.0, 0.0])])
        self.sim.data.set_joint_qpos(self.red_cube.joints[0], cube_qpos)
        self.sim.forward()

    def end_effector_pose(self) -> Pose:
        position = np.array(self.sim.data.site_xpos[self._eef_site_id], dtype=np.float64)
        rotation = np.array(self.sim.data.site_xmat[self._eef_site_id], dtype=np.float64).reshape(3, 3)
        from robosuite.utils.transform_utils import mat2quat

        return Pose(position=position, quaternion=np.asarray(mat2quat(rotation)))

    def object_pose(self, name: str) -> Pose:
        try:
            body_id = self._object_body_ids[name]
        except KeyError as exc:
            available = ", ".join(self.object_names)
            raise KeyError(f"Unknown object {name!r}; available objects: {available}") from exc

        position = np.array(self.sim.data.body_xpos[body_id], dtype=np.float64)
        quaternion_wxyz = np.array(self.sim.data.body_xquat[body_id], dtype=np.float64)
        quaternion_xyzw = np.asarray(convert_quat(quaternion_wxyz, to="xyzw"))
        return Pose(position=position, quaternion=quaternion_xyzw)

    def object_linear_velocity(self, name: str) -> np.ndarray:
        """Return one object's world-frame linear velocity in metres/second."""

        try:
            body_id = self._object_body_ids[name]
        except KeyError as exc:
            available = ", ".join(self.object_names)
            raise KeyError(f"Unknown object {name!r}; available objects: {available}") from exc
        body_name = self.sim.model.body_id2name(body_id)
        return np.asarray(self.sim.data.get_body_xvelp(body_name), dtype=np.float64).copy()

    def target_half_extents(self, name: str) -> np.ndarray:
        """Return the configured half extents for a supported target region."""

        if name != "blue_target":
            if name not in self.object_names:
                available = ", ".join(self.object_names)
                raise KeyError(f"Unknown object {name!r}; available objects: {available}")
            raise ValueError(f"Object {name!r} is not a target region")
        return np.asarray(self.scene_config.target_half_size, dtype=np.float64).copy()

    def is_object_center_inside_target(
        self,
        object_name: str,
        target_name: str,
        *,
        margin: float = 0.0,
    ) -> bool:
        """Test target membership from ground-truth object-center XY state."""

        if margin < 0.0:
            raise ValueError("margin must be non-negative")
        object_position = self.object_pose(object_name).position
        target_position = self.object_pose(target_name).position
        half_extents = self.target_half_extents(target_name)
        usable_xy = half_extents[:2] - margin
        if np.any(usable_xy < 0.0):
            return False
        return bool(np.all(np.abs(object_position[:2] - target_position[:2]) <= usable_xy))

    def is_grasping_object(self, name: str) -> bool:
        """Return robosuite's two-finger contact grasp predicate."""

        if name != "red_cube":
            if name not in self.object_names:
                available = ", ".join(self.object_names)
                raise KeyError(f"Unknown object {name!r}; available objects: {available}")
            return False
        return bool(
            self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self.red_cube,
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
    """Create the supported V0 environment with a Panda OSC pose controller."""

    return SemanticTabletopEnv(
        scene=scene,
        has_renderer=render,
        randomize_cube=randomize_cube,
        seed=seed,
    )
