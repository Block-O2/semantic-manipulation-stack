"""Observation and physical success definition for rigid Tissue."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent_act_reproduction.sim.tissue_env import TissueEnv


@dataclass(frozen=True)
class TissueObservation:
    robot_state: np.ndarray
    environment_state: np.ndarray

    @classmethod
    def read(cls, env: TissueEnv) -> "TissueObservation":
        robot_state = np.concatenate(
            (env.end_effector_position(), np.array([env.gripper_width()]))
        )
        environment_state = np.concatenate(
            (
                env.tissue_position(),
                env.tissue_linear_velocity(),
                env.pull_target_position(),
                np.array(
                    [
                        float(env.is_grasping_tissue()),
                        min(1.0, env.step_count / env.scene.max_episode_steps),
                    ]
                ),
            )
        )
        return cls(robot_state=robot_state, environment_state=environment_state)


@dataclass(frozen=True)
class TissueSuccess:
    success: bool
    extracted: bool
    forward_distance_m: float
    lateral_error_m: float
    height_ok: bool
    stable: bool
    speed_mps: float


class TissueSuccessChecker:
    def check(self, env: TissueEnv) -> TissueSuccess:
        position = env.tissue_position()
        speed = float(np.linalg.norm(env.tissue_linear_velocity()))
        forward = float(position[1] - env.scene.tissue_xy[1])
        lateral = abs(float(position[0] - env.scene.pull_target[0]))
        extracted = forward >= 0.16
        height_ok = position[2] >= env.scene.table_top_z + 0.02
        stable = speed <= 0.08
        return TissueSuccess(
            success=extracted and lateral <= 0.06 and height_ok and stable,
            extracted=extracted,
            forward_distance_m=forward,
            lateral_error_m=lateral,
            height_ok=bool(height_ok),
            stable=stable,
            speed_mps=speed,
        )
