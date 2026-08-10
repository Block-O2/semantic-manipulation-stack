"""Bottle-only Panda scene built directly on robosuite and MuJoCo."""

from __future__ import annotations

from typing import Any

import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string

from agent_act_reproduction.config import BottleSceneConfig, DEFAULT_SCENE


class BottleEnv(ManipulationEnv):
    """Fixed-workspace Panda, bottle, and raised shelf scene.

    The environment exposes only physical stepping and state reads. It contains
    no task trajectory and no learned-policy fallback.
    """

    def __init__(
        self,
        *,
        scene: BottleSceneConfig = DEFAULT_SCENE,
        has_renderer: bool = False,
        seed: int = 0,
        controller_configs: dict[str, Any] | None = None,
    ) -> None:
        self.scene = scene
        self.table_full_size = scene.table_size
        self.table_friction = (1.0, 0.005, 0.0001)
        self.table_offset = np.asarray(scene.table_offset, dtype=np.float64)
        self._render_enabled = has_renderer
        self._episode_seed = int(seed)
        self._perturbation_m = scene.random_xy_m
        self._yaw_perturbation_rad = scene.random_yaw_rad
        self._step_count = 0

        if controller_configs is None:
            basic = load_composite_controller_config(controller="BASIC")
            controller_configs = {
                "type": basic["type"],
                "body_parts": {"right": basic["body_parts"]["right"]},
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
            control_freq=scene.control_freq_hz,
            horizon=100_000,
            ignore_done=True,
            hard_reset=False,
            renderer="mjviewer",
            seed=seed,
        )

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

        self.bottle = CylinderObject(
            name="bottle",
            size=(self.scene.bottle_radius, self.scene.bottle_half_height),
            rgba=(0.10, 0.55, 0.95, 1.0),
            density=350.0,
            friction=(1.2, 0.01, 0.001),
            rng=self.rng,
        )
        shelf_rgba = (0.55, 0.34, 0.16, 1.0)
        self.shelf_deck = BoxObject(
            name="shelf_deck",
            size_min=self.scene.shelf_deck_half_size,
            size_max=self.scene.shelf_deck_half_size,
            rgba=shelf_rgba,
            joints=None,
            friction=(1.0, 0.005, 0.0001),
            rng=self.rng,
        )
        self.shelf_left_post = BoxObject(
            name="shelf_left_post",
            size_min=self.scene.shelf_post_half_size,
            size_max=self.scene.shelf_post_half_size,
            rgba=shelf_rgba,
            joints=None,
            rng=self.rng,
        )
        self.shelf_right_post = BoxObject(
            name="shelf_right_post",
            size_min=self.scene.shelf_post_half_size,
            size_max=self.scene.shelf_post_half_size,
            rgba=shelf_rgba,
            joints=None,
            rng=self.rng,
        )
        target_half_size = (0.045, 0.045, 0.002)
        self.shelf_target_marker = BoxObject(
            name="shelf_target_marker",
            size_min=target_half_size,
            size_max=target_half_size,
            rgba=(0.15, 0.9, 0.2, 0.35),
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=False,
            rng=self.rng,
        )

        sx, sy = self.scene.shelf_xy
        self.shelf_deck.get_obj().set(
            "pos", array_to_string((sx, sy, self.scene.shelf_deck_center_z))
        )
        post_z = self.scene.table_top_z + self.scene.shelf_post_half_size[2]
        post_x_offset = self.scene.shelf_deck_half_size[0] - self.scene.shelf_post_half_size[0]
        self.shelf_left_post.get_obj().set(
            "pos", array_to_string((sx - post_x_offset, sy, post_z))
        )
        self.shelf_right_post.get_obj().set(
            "pos", array_to_string((sx + post_x_offset, sy, post_z))
        )
        self.shelf_target_marker.get_obj().set(
            "pos", array_to_string((sx, sy, self.scene.shelf_top_z + target_half_size[2]))
        )

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[
                self.bottle,
                self.shelf_deck,
                self.shelf_left_post,
                self.shelf_right_post,
                self.shelf_target_marker,
            ],
        )

    def _setup_references(self) -> None:
        super()._setup_references()
        self._bottle_body_id = self.sim.model.body_name2id(self.bottle.root_body)
        eef_ids = self.robots[0].eef_site_id
        self._eef_site_id = (
            eef_ids.get("right", next(iter(eef_ids.values())))
            if isinstance(eef_ids, dict)
            else int(eef_ids)
        )

    def configure_episode(
        self,
        *,
        seed: int,
        perturbation_m: float,
        yaw_perturbation_rad: float | None = None,
    ) -> None:
        if perturbation_m < 0.0:
            raise ValueError("perturbation_m must be non-negative")
        self._episode_seed = int(seed)
        self._perturbation_m = float(perturbation_m)
        self._yaw_perturbation_rad = (
            self.scene.random_yaw_rad
            if yaw_perturbation_rad is None
            else float(yaw_perturbation_rad)
        )

    def _reset_internal(self) -> None:
        super()._reset_internal()
        rng = np.random.default_rng(self._episode_seed)
        dx, dy = rng.uniform(-self._perturbation_m, self._perturbation_m, size=2)
        yaw = rng.uniform(-self._yaw_perturbation_rad, self._yaw_perturbation_rad)
        qpos = np.array(
            [
                self.scene.bottle_xy[0] + dx,
                self.scene.bottle_xy[1] + dy,
                self.scene.bottle_initial_z,
                np.cos(0.5 * yaw),
                0.0,
                0.0,
                np.sin(0.5 * yaw),
            ],
            dtype=np.float64,
        )
        self.sim.data.set_joint_qpos(self.bottle.joints[0], qpos)
        self.sim.data.set_joint_qvel(self.bottle.joints[0], np.zeros(6))
        self._step_count = 0
        self.sim.forward()

    def step(self, action: np.ndarray):
        result = super().step(action)
        self._step_count += 1
        return result

    @property
    def step_count(self) -> int:
        return self._step_count

    def end_effector_position(self) -> np.ndarray:
        return np.asarray(self.sim.data.site_xpos[self._eef_site_id], dtype=np.float64).copy()

    def end_effector_quaternion_xyzw(self) -> np.ndarray:
        from robosuite.utils.transform_utils import mat2quat

        rotation = np.asarray(self.sim.data.site_xmat[self._eef_site_id], dtype=np.float64).reshape(3, 3)
        return np.asarray(mat2quat(rotation), dtype=np.float64)

    def bottle_position(self) -> np.ndarray:
        return np.asarray(self.sim.data.body_xpos[self._bottle_body_id], dtype=np.float64).copy()

    def bottle_quaternion_wxyz(self) -> np.ndarray:
        return np.asarray(self.sim.data.body_xquat[self._bottle_body_id], dtype=np.float64).copy()

    def bottle_linear_velocity(self) -> np.ndarray:
        body_name = self.sim.model.body_id2name(self._bottle_body_id)
        return np.asarray(self.sim.data.get_body_xvelp(body_name), dtype=np.float64).copy()

    def gripper_width(self) -> float:
        qpos = np.asarray(self.robots[0].get_gripper_joint_positions("right"), dtype=np.float64)
        return float(np.sum(np.abs(qpos)))

    def is_grasping_bottle(self) -> bool:
        return bool(self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.bottle))

    def shelf_target_position(self) -> np.ndarray:
        return np.asarray(self.scene.shelf_target, dtype=np.float64)

    def render_if_enabled(self) -> None:
        if self._render_enabled:
            self.render()


def make_bottle_env(*, render: bool = False, seed: int = 0) -> BottleEnv:
    return BottleEnv(has_renderer=render, seed=seed)
