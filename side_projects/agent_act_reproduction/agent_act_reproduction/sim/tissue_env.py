"""Rigid-sheet tissue extraction scene for Panda / robosuite."""

from __future__ import annotations

from typing import Any

import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string

from agent_act_reproduction.config import DEFAULT_TISSUE_SCENE, TissueSceneConfig


class TissueEnv(ManipulationEnv):
    """A thin rigid sheet protrudes from a fixed low box.

    The simplification is deliberate: no deformable cloth model is claimed.
    """

    def __init__(
        self,
        *,
        scene: TissueSceneConfig = DEFAULT_TISSUE_SCENE,
        has_renderer: bool = False,
        has_offscreen_renderer: bool = False,
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
            has_offscreen_renderer=has_offscreen_renderer,
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

        self.tissue = BoxObject(
            name="tissue",
            size_min=self.scene.tissue_half_size,
            size_max=self.scene.tissue_half_size,
            rgba=(0.96, 0.97, 1.0, 1.0),
            density=120.0,
            friction=(1.5, 0.01, 0.002),
            rng=self.rng,
        )
        box_color = (0.16, 0.55, 0.70, 1.0)
        box_specs = (
            ("box_base", (0.065, 0.095, 0.016), (0.0, 0.0, 0.0)),
            ("slot_back", (0.052, 0.003, 0.020), (0.0, 0.052, 0.0)),
            ("slot_front", (0.052, 0.003, 0.020), (0.0, 0.068, 0.0)),
            ("box_back", (0.065, 0.012, 0.026), (0.0, -0.050, 0.0)),
            ("box_left", (0.012, 0.050, 0.026), (-0.053, 0.0, 0.0)),
            ("box_right", (0.012, 0.050, 0.026), (0.053, 0.0, 0.0)),
        )
        self.box_parts = []
        bx, by = self.scene.box_xy
        for name, half_size, offset in box_specs:
            part = BoxObject(
                name=name,
                size_min=half_size,
                size_max=half_size,
                rgba=box_color,
                joints=None,
                obj_type="all" if name.startswith("slot_") else "visual",
                duplicate_collision_geoms=name.startswith("slot_"),
                rng=self.rng,
            )
            part.get_obj().set(
                "pos",
                array_to_string(
                    (bx + offset[0], by + offset[1], self.scene.table_top_z + half_size[2])
                ),
            )
            self.box_parts.append(part)
        self.pull_marker = BoxObject(
            name="pull_target",
            size_min=(0.055, 0.035, 0.0015),
            size_max=(0.055, 0.035, 0.0015),
            rgba=(0.20, 0.85, 0.38, 0.28),
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=False,
            rng=self.rng,
        )
        self.pull_marker.get_obj().set(
            "pos",
            array_to_string(
                (
                    self.scene.pull_target[0],
                    self.scene.pull_target[1],
                    self.scene.table_top_z + 0.002,
                )
            ),
        )
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.tissue, *self.box_parts, self.pull_marker],
        )

    def _setup_references(self) -> None:
        super()._setup_references()
        self._tissue_body_id = self.sim.model.body_name2id(self.tissue.root_body)
        eef_ids = self.robots[0].eef_site_id
        self._eef_site_id = (
            eef_ids.get("right", next(iter(eef_ids.values())))
            if isinstance(eef_ids, dict)
            else int(eef_ids)
        )

    def configure_episode(self, *, seed: int, perturbation_m: float) -> None:
        if perturbation_m < 0.0:
            raise ValueError("perturbation_m must be non-negative")
        self._episode_seed = int(seed)
        self._perturbation_m = float(perturbation_m)

    def _reset_internal(self) -> None:
        super()._reset_internal()
        rng = np.random.default_rng(self._episode_seed)
        dx, dy = rng.uniform(-self._perturbation_m, self._perturbation_m, size=2)
        yaw = rng.uniform(-self.scene.random_yaw_rad, self.scene.random_yaw_rad)
        qpos = np.array(
            [
                self.scene.tissue_xy[0] + dx,
                self.scene.tissue_xy[1] + dy,
                self.scene.tissue_center_z,
                np.cos(0.5 * yaw),
                0.0,
                0.0,
                np.sin(0.5 * yaw),
            ],
            dtype=np.float64,
        )
        self.sim.data.set_joint_qpos(self.tissue.joints[0], qpos)
        self.sim.data.set_joint_qvel(self.tissue.joints[0], np.zeros(6))
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

        rotation = np.asarray(self.sim.data.site_xmat[self._eef_site_id]).reshape(3, 3)
        return np.asarray(mat2quat(rotation), dtype=np.float64)

    def tissue_position(self) -> np.ndarray:
        return np.asarray(self.sim.data.body_xpos[self._tissue_body_id], dtype=np.float64).copy()

    def tissue_quaternion_wxyz(self) -> np.ndarray:
        return np.asarray(self.sim.data.body_xquat[self._tissue_body_id], dtype=np.float64).copy()

    def tissue_linear_velocity(self) -> np.ndarray:
        name = self.sim.model.body_id2name(self._tissue_body_id)
        return np.asarray(self.sim.data.get_body_xvelp(name), dtype=np.float64).copy()

    def gripper_width(self) -> float:
        qpos = np.asarray(self.robots[0].get_gripper_joint_positions("right"))
        return float(np.sum(np.abs(qpos)))

    def is_grasping_tissue(self) -> bool:
        return bool(self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.tissue))

    def pull_target_position(self) -> np.ndarray:
        return np.asarray(self.scene.pull_target, dtype=np.float64)

    def render_if_enabled(self) -> None:
        if self._render_enabled:
            self.render()


def make_tissue_env(
    *, render: bool = False, offscreen: bool = False, seed: int = 0
) -> TissueEnv:
    return TissueEnv(
        has_renderer=render,
        has_offscreen_renderer=offscreen,
        seed=seed,
    )
