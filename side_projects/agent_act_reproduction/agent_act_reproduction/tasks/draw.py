"""Observation and contact-trajectory success definition for Draw."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent_act_reproduction.sim.draw_env import DrawEnv


@dataclass(frozen=True)
class DrawObservation:
    robot_state: np.ndarray
    environment_state: np.ndarray

    @classmethod
    def read(cls, env: DrawEnv) -> "DrawObservation":
        robot_state = np.concatenate(
            (env.end_effector_position(), np.array([env.gripper_width()]))
        )
        environment_state = np.concatenate(
            (
                env.pen_position(),
                env.pen_tip_position(),
                env.line_end_eef_target(),
                np.array(
                    [
                        float(env.is_grasping_pen()),
                        env.stroke_progress(),
                    ]
                ),
            )
        )
        return cls(robot_state=robot_state, environment_state=environment_state)


@dataclass(frozen=True)
class DrawSuccess:
    success: bool
    pen_was_grasped: bool
    contacted_paper: bool
    contact_points: int
    stroke_length_m: float
    lateral_span_m: float
    maximum_y_deviation_m: float
    lifted_after_stroke: bool


class DrawSuccessChecker:
    def check(self, env: DrawEnv) -> DrawSuccess:
        trajectory = env.contact_trajectory()
        contacted = len(trajectory) >= 2
        if contacted:
            x_span = float(trajectory[:, 0].max() - trajectory[:, 0].min())
            y_span = float(trajectory[:, 1].max() - trajectory[:, 1].min())
            y_deviation = float(np.max(np.abs(trajectory[:, 1] - env.scene.line_start_xy[1])))
        else:
            x_span = y_span = y_deviation = 0.0
        lifted = contacted and env.pen_tip_position()[2] >= env.scene.paper_top_z + 0.025
        grasped = env.ever_grasped_pen()
        success = grasped and contacted and x_span >= 0.11 and y_deviation <= 0.03 and lifted
        return DrawSuccess(
            success=bool(success),
            pen_was_grasped=grasped,
            contacted_paper=contacted,
            contact_points=len(trajectory),
            stroke_length_m=x_span,
            lateral_span_m=y_span,
            maximum_y_deviation_m=y_deviation,
            lifted_after_stroke=bool(lifted),
        )
