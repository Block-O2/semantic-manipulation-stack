"""Deterministic pen-pick and horizontal-stroke expert for demos only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from agent_act_reproduction.robot import PandaCartesianAdapter
from agent_act_reproduction.sim.draw_env import DrawEnv
from agent_act_reproduction.tasks.draw import DrawObservation, DrawSuccessChecker


class DrawPhase(str, Enum):
    OPEN = "OPEN"
    PREGRASP = "PREGRASP"
    DESCEND = "DESCEND"
    CLOSE = "CLOSE"
    LIFT = "LIFT"
    ABOVE_START = "ABOVE_START"
    CONTACT = "CONTACT"
    STROKE = "STROKE"
    LIFT_OFF = "LIFT_OFF"
    HOLD = "HOLD"
    DONE = "DONE"


@dataclass
class DrawExpertEpisode:
    seed: int
    initial_pen_pose: list[float]
    observations_state: np.ndarray
    observations_environment_state: np.ndarray
    actions: np.ndarray
    phases: list[str]
    episode_length: int
    success: bool
    final_pen_pose: list[float]
    success_details: dict[str, object]


class DrawExpert:
    def __init__(self, env: DrawEnv, adapter: PandaCartesianAdapter) -> None:
        self.env = env
        self.adapter = adapter
        self.checker = DrawSuccessChecker()

    @staticmethod
    def _near(a: np.ndarray, b: np.ndarray, tolerance: float = 0.012) -> bool:
        return float(np.linalg.norm(a - b)) <= tolerance

    def run(self, *, seed: int, perturbation_m: float = 0.006) -> DrawExpertEpisode:
        self.env.configure_episode(seed=seed, perturbation_m=perturbation_m)
        self.env.reset()
        initial_position = self.env.pen_position().copy()
        initial_pose = np.concatenate((initial_position, self.env.pen_quaternion_wxyz()))
        phase = DrawPhase.OPEN
        phase_steps = 0
        state_rows: list[np.ndarray] = []
        env_rows: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        phases: list[str] = []
        start = np.asarray((*self.env.scene.line_start_xy, 0.915), dtype=np.float64)
        end = self.env.line_end_eef_target()
        for _ in range(self.env.scene.max_episode_steps):
            eef = self.env.end_effector_position()
            if phase is DrawPhase.OPEN:
                target, grip = eef.copy(), -1.0
                if phase_steps >= 20:
                    phase, phase_steps = DrawPhase.PREGRASP, 0
            elif phase is DrawPhase.PREGRASP:
                target, grip = initial_position + np.array((0.0, 0.0, 0.15)), -1.0
                if self._near(eef, target) and phase_steps >= 5:
                    phase, phase_steps = DrawPhase.DESCEND, 0
            elif phase is DrawPhase.DESCEND:
                target, grip = initial_position + np.array((0.0, 0.0, 0.020)), -1.0
                if self._near(eef, target) and phase_steps >= 5:
                    phase, phase_steps = DrawPhase.CLOSE, 0
            elif phase is DrawPhase.CLOSE:
                target, grip = eef.copy(), 1.0
                if phase_steps >= 42:
                    phase, phase_steps = DrawPhase.LIFT, 0
            elif phase is DrawPhase.LIFT:
                target, grip = np.array((initial_position[0], initial_position[1], 1.055)), 1.0
                if self._near(eef, target) and phase_steps >= 5:
                    phase, phase_steps = DrawPhase.ABOVE_START, 0
            elif phase is DrawPhase.ABOVE_START:
                target, grip = np.array((start[0], start[1], 1.02)), 1.0
                if self._near(eef, target) and phase_steps >= 5:
                    phase, phase_steps = DrawPhase.CONTACT, 0
            elif phase is DrawPhase.CONTACT:
                target, grip = start, 1.0
                if len(self.env.contact_trajectory()) >= 1 and phase_steps >= 5:
                    phase, phase_steps = DrawPhase.STROKE, 0
                elif phase_steps >= 45:
                    phase, phase_steps = DrawPhase.STROKE, 0
            elif phase is DrawPhase.STROKE:
                target, grip = end, 1.0
                if (
                    self.env.stroke_progress() >= 0.95 and phase_steps >= 5
                ) or phase_steps >= 60:
                    phase, phase_steps = DrawPhase.LIFT_OFF, 0
            elif phase is DrawPhase.LIFT_OFF:
                target, grip = np.array((eef[0], eef[1], 1.10)), 1.0
                if eef[2] >= 1.07 or phase_steps >= 60:
                    phase, phase_steps = DrawPhase.HOLD, 0
            elif phase is DrawPhase.HOLD:
                target, grip = eef.copy(), 1.0
                if phase_steps >= 22:
                    phase = DrawPhase.DONE
            else:
                break
            observation = DrawObservation.read(self.env)
            command = np.array((*target, grip), dtype=np.float64)
            state_rows.append(observation.robot_state)
            env_rows.append(observation.environment_state)
            actions.append(command)
            phases.append(phase.value)
            self.adapter.execute(command)
            phase_steps += 1
        details = self.checker.check(self.env)
        final_pose = np.concatenate((self.env.pen_position(), self.env.pen_quaternion_wxyz()))
        return DrawExpertEpisode(
            seed=seed,
            initial_pen_pose=initial_pose.tolist(),
            observations_state=np.asarray(state_rows, dtype=np.float32),
            observations_environment_state=np.asarray(env_rows, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            phases=phases,
            episode_length=len(actions),
            success=details.success,
            final_pen_pose=final_pose.tolist(),
            success_details=asdict(details),
        )
