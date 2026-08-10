"""State observation and physical success definition for Bottle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent_act_reproduction.sim import BottleEnv


@dataclass(frozen=True)
class BottleObservation:
    robot_state: np.ndarray
    environment_state: np.ndarray

    @classmethod
    def read(cls, env: BottleEnv) -> "BottleObservation":
        robot_state = np.concatenate(
            (env.end_effector_position(), np.array([env.gripper_width()], dtype=np.float64))
        )
        environment_state = np.concatenate(
            (
                env.bottle_position(),
                env.bottle_linear_velocity(),
                env.shelf_target_position(),
                np.array(
                    [
                        float(env.is_grasping_bottle()),
                        min(1.0, env.step_count / env.scene.max_episode_steps),
                    ],
                    dtype=np.float64,
                ),
            )
        )
        return cls(robot_state=robot_state, environment_state=environment_state)


@dataclass(frozen=True)
class BottleSuccess:
    success: bool
    inside_shelf_xy: bool
    released: bool
    height_ok: bool
    stable: bool
    position_error_m: tuple[float, float, float]
    speed_mps: float


class BottleSuccessChecker:
    def __init__(
        self,
        *,
        xy_tolerance_m: float = 0.045,
        z_tolerance_m: float = 0.025,
        maximum_speed_mps: float = 0.04,
    ) -> None:
        self.xy_tolerance_m = float(xy_tolerance_m)
        self.z_tolerance_m = float(z_tolerance_m)
        self.maximum_speed_mps = float(maximum_speed_mps)

    def check(self, env: BottleEnv) -> BottleSuccess:
        error = env.bottle_position() - env.shelf_target_position()
        speed = float(np.linalg.norm(env.bottle_linear_velocity()))
        inside_xy = bool(np.all(np.abs(error[:2]) <= self.xy_tolerance_m))
        released = not env.is_grasping_bottle()
        height_ok = abs(float(error[2])) <= self.z_tolerance_m
        stable = speed <= self.maximum_speed_mps
        return BottleSuccess(
            success=inside_xy and released and height_ok and stable,
            inside_shelf_xy=inside_xy,
            released=released,
            height_ok=height_ok,
            stable=stable,
            position_error_m=tuple(float(value) for value in error),
            speed_mps=speed,
        )
