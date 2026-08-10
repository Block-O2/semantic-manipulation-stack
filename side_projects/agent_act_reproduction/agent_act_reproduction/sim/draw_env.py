"""Pen, paper and contact-derived stroke scene for Panda / robosuite."""

from __future__ import annotations

from typing import Any

import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, new_geom

from agent_act_reproduction.config import DEFAULT_DRAW_SCENE, DrawSceneConfig


class DrawEnv(ManipulationEnv):
    """A physical pen whose tip contacts are the sole source of the dark line."""

    def __init__(
        self,
        *,
        scene: DrawSceneConfig = DEFAULT_DRAW_SCENE,
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
        self._contact_points: list[np.ndarray] = []
        self._ever_grasped = False
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
        self.pen = BoxObject(
            name="pen",
            size_min=(0.026, 0.020, 0.026),
            size_max=(0.026, 0.020, 0.026),
            rgba=(0.20, 0.30, 0.45, 1.0),
            density=420.0,
            friction=(5.0, 0.02, 0.002),
            rng=self.rng,
        )
        self.pen.get_obj().append(
            new_geom(
                name="pen_shaft_geom",
                type="cylinder",
                size=(0.008, 0.037),
                pos=(0.0, 0.0, -0.063),
                rgba=(0.12, 0.18, 0.26, 1.0),
                group=1,
                contype="0",
                conaffinity="0",
                mass="0",
            )
        )
        self.pen.get_obj().append(
            new_geom(
                name="pen_tip_geom",
                type="sphere",
                size=(0.005,),
                pos=(0.0, 0.0, -0.100),
                rgba=(0.02, 0.02, 0.025, 1.0),
                group=0,
                contype="1",
                conaffinity="1",
                density="420",
                friction="0.05 0.001 0.0001",
            )
        )
        self.paper = BoxObject(
            name="paper",
            size_min=self.scene.paper_half_size,
            size_max=self.scene.paper_half_size,
            rgba=(0.97, 0.95, 0.88, 1.0),
            joints=None,
            friction=(1.0, 0.005, 0.0001),
            rng=self.rng,
        )
        self.paper.get_obj().set(
            "pos",
            array_to_string(
                (
                    self.scene.paper_xy[0],
                    self.scene.paper_xy[1],
                    self.scene.table_top_z + self.scene.paper_half_size[2],
                )
            ),
        )
        self.markers = []
        for index in range(self.scene.marker_count):
            marker = BoxObject(
                name=f"ink_{index:02d}",
                size_min=(0.0034, 0.0034, 0.0006),
                size_max=(0.0034, 0.0034, 0.0006),
                rgba=(0.025, 0.025, 0.03, 1.0),
                joints=None,
                obj_type="visual",
                duplicate_collision_geoms=False,
                rng=self.rng,
            )
            marker.get_obj().set("pos", array_to_string((0.0, 0.0, 0.20)))
            self.markers.append(marker)
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.pen, self.paper, *self.markers],
        )

    def _setup_references(self) -> None:
        super()._setup_references()
        self._pen_body_id = self.sim.model.body_name2id(self.pen.root_body)
        self._marker_body_ids = [
            self.sim.model.body_name2id(marker.root_body) for marker in self.markers
        ]
        geom_names = [self.sim.model.geom_id2name(i) for i in range(self.sim.model.ngeom)]
        self._tip_geom_id = next(
            i for i, name in enumerate(geom_names) if name and name.endswith("pen_tip_geom")
        )
        self._paper_geom_ids = {
            self.sim.model.geom_name2id(name) for name in self.paper.contact_geoms
        }
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
        qpos = np.array(
            [
                self.scene.pen_xy[0] + dx,
                self.scene.pen_xy[1] + dy,
                self.scene.pen_initial_z,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        self.sim.data.set_joint_qpos(self.pen.joints[0], qpos)
        self.sim.data.set_joint_qvel(self.pen.joints[0], np.zeros(6))
        self._step_count = 0
        self._contact_points.clear()
        self._ever_grasped = False
        for body_id in self._marker_body_ids:
            self.sim.model.body_pos[body_id] = (0.0, 0.0, 0.20)
        self.sim.forward()

    def _record_tip_contacts(self) -> None:
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if self._tip_geom_id not in pair or not (pair & self._paper_geom_ids):
                continue
            point = np.asarray(contact.pos, dtype=np.float64).copy()
            px, py = self.scene.paper_xy
            hx, hy, _ = self.scene.paper_half_size
            if not (px - hx <= point[0] <= px + hx and py - hy <= point[1] <= py + hy):
                continue
            if self._contact_points and np.linalg.norm(point[:2] - self._contact_points[-1][:2]) < 0.004:
                continue
            if len(self._contact_points) >= len(self._marker_body_ids):
                continue
            self._contact_points.append(point)
            body_id = self._marker_body_ids[len(self._contact_points) - 1]
            self.sim.model.body_pos[body_id] = (
                float(point[0]),
                float(point[1]),
                self.scene.paper_top_z + 0.0008,
            )
            self.sim.forward()

    def step(self, action: np.ndarray):
        result = super().step(action)
        self._step_count += 1
        self._ever_grasped = self._ever_grasped or self.is_grasping_pen()
        self._record_tip_contacts()
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

    def pen_position(self) -> np.ndarray:
        return np.asarray(self.sim.data.body_xpos[self._pen_body_id], dtype=np.float64).copy()

    def pen_quaternion_wxyz(self) -> np.ndarray:
        return np.asarray(self.sim.data.body_xquat[self._pen_body_id], dtype=np.float64).copy()

    def pen_tip_position(self) -> np.ndarray:
        return np.asarray(self.sim.data.geom_xpos[self._tip_geom_id], dtype=np.float64).copy()

    def gripper_width(self) -> float:
        qpos = np.asarray(self.robots[0].get_gripper_joint_positions("right"))
        return float(np.sum(np.abs(qpos)))

    def is_grasping_pen(self) -> bool:
        return bool(self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.pen))

    def ever_grasped_pen(self) -> bool:
        return self._ever_grasped

    def contact_trajectory(self) -> np.ndarray:
        if not self._contact_points:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(self._contact_points, dtype=np.float64)

    def stroke_progress(self) -> float:
        trajectory = self.contact_trajectory()
        if len(trajectory) < 2:
            return 0.0
        return min(1.0, float(np.ptp(trajectory[:, 0])) / 0.16)

    def line_end_eef_target(self) -> np.ndarray:
        return np.asarray((*self.scene.line_end_xy, 0.915), dtype=np.float64)

    def render_if_enabled(self) -> None:
        if self._render_enabled:
            self.render()


def make_draw_env(
    *, render: bool = False, offscreen: bool = False, seed: int = 0
) -> DrawEnv:
    return DrawEnv(
        has_renderer=render,
        has_offscreen_renderer=offscreen,
        seed=seed,
    )
