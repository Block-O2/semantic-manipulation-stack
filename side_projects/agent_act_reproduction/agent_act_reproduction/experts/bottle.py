"""Deterministic Bottle FSM used only to generate demonstrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from agent_act_reproduction.robot import PandaCartesianAdapter
from agent_act_reproduction.sim import BottleEnv
from agent_act_reproduction.tasks import BottleObservation, BottleSuccessChecker


class BottlePhase(str, Enum):
    OPEN_GRIPPER = "OPEN_GRIPPER"
    MOVE_PREGRASP = "MOVE_PREGRASP"
    DESCEND = "DESCEND"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    LIFT = "LIFT"
    MOVE_ABOVE_SHELF = "MOVE_ABOVE_SHELF"
    PLACE_DESCEND = "PLACE_DESCEND"
    OPEN_RELEASE = "OPEN_RELEASE"
    RETREAT = "RETREAT"
    SETTLE = "SETTLE"
    DONE = "DONE"


@dataclass
class ExpertEpisode:
    seed: int
    initial_bottle_pose: list[float]
    observations_state: np.ndarray
    observations_environment_state: np.ndarray
    actions: np.ndarray
    phases: list[str]
    episode_length: int
    success: bool
    final_bottle_pose: list[float]
    success_details: dict[str, object]


class BottleExpert:
    """Closed-loop waypoint FSM; never imported by policy runtime."""

    def __init__(self, env: BottleEnv, adapter: PandaCartesianAdapter) -> None:
        self.env = env
        self.adapter = adapter
        self.checker = BottleSuccessChecker()

    @staticmethod
    def _near(current: np.ndarray, target: np.ndarray, tolerance: float = 0.012) -> bool:
        return float(np.linalg.norm(current - target)) <= tolerance

    def run(self, *, seed: int, perturbation_m: float = 0.01) -> ExpertEpisode:
        self.env.configure_episode(seed=seed, perturbation_m=perturbation_m)
        self.env.reset()
        initial_pose = np.concatenate(
            (self.env.bottle_position(), self.env.bottle_quaternion_wxyz())
        )
        initial_bottle_position = self.env.bottle_position().copy()
        phase = BottlePhase.OPEN_GRIPPER
        phase_steps = 0
        state_rows: list[np.ndarray] = []
        env_rows: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        phases: list[str] = []

        for _ in range(self.env.scene.max_episode_steps):
            eef = self.env.end_effector_position()
            bottle = self.env.bottle_position()
            shelf_target = self.env.shelf_target_position()

            if phase is BottlePhase.OPEN_GRIPPER:
                target = eef.copy()
                gripper = -1.0
                if phase_steps >= 24:
                    phase, phase_steps = BottlePhase.MOVE_PREGRASP, 0
            elif phase is BottlePhase.MOVE_PREGRASP:
                target = initial_bottle_position + np.array((0.0, 0.0, 0.145))
                gripper = -1.0
                if self._near(eef, target) and phase_steps >= 5:
                    phase, phase_steps = BottlePhase.DESCEND, 0
            elif phase is BottlePhase.DESCEND:
                # Hold the sampled XY rather than chasing a contacted object.
                # The Panda grip site sits above the finger contact centre, so
                # a 2 cm offset centers the fingertips on this tall cylinder.
                target = initial_bottle_position + np.array((0.0, 0.0, 0.020))
                gripper = -1.0
                if self._near(eef, target, 0.012) and phase_steps >= 5:
                    phase, phase_steps = BottlePhase.CLOSE_GRIPPER, 0
            elif phase is BottlePhase.CLOSE_GRIPPER:
                target = eef.copy()
                gripper = 1.0
                if phase_steps >= 42:
                    phase, phase_steps = BottlePhase.LIFT, 0
            elif phase is BottlePhase.LIFT:
                target = np.array((bottle[0], bottle[1], 1.075))
                gripper = 1.0
                if self._near(eef, target) and phase_steps >= 5:
                    phase, phase_steps = BottlePhase.MOVE_ABOVE_SHELF, 0
            elif phase is BottlePhase.MOVE_ABOVE_SHELF:
                target = shelf_target + np.array((0.0, 0.0, 0.12))
                gripper = 1.0
                if self._near(eef, target) and phase_steps >= 5:
                    phase, phase_steps = BottlePhase.PLACE_DESCEND, 0
            elif phase is BottlePhase.PLACE_DESCEND:
                # Preserve the measured grip-site-to-bottle-centre offset so
                # the bottle base, rather than the fingers, meets the deck.
                target = shelf_target + np.array((0.0, 0.0, 0.020))
                gripper = 1.0
                if self._near(eef, target, 0.012) and phase_steps >= 5:
                    phase, phase_steps = BottlePhase.OPEN_RELEASE, 0
            elif phase is BottlePhase.OPEN_RELEASE:
                target = eef.copy()
                gripper = -1.0
                if phase_steps >= 38:
                    phase, phase_steps = BottlePhase.RETREAT, 0
            elif phase is BottlePhase.RETREAT:
                target = shelf_target + np.array((0.0, 0.0, 0.15))
                gripper = -1.0
                if self._near(eef, target) and phase_steps >= 5:
                    phase, phase_steps = BottlePhase.SETTLE, 0
            elif phase is BottlePhase.SETTLE:
                target = eef.copy()
                gripper = -1.0
                if phase_steps >= 28:
                    phase = BottlePhase.DONE
            else:
                break

            observation = BottleObservation.read(self.env)
            command = np.array((*target, gripper), dtype=np.float64)
            state_rows.append(observation.robot_state)
            env_rows.append(observation.environment_state)
            actions.append(command)
            phases.append(phase.value)
            self.adapter.execute(command)
            phase_steps += 1

        details = self.checker.check(self.env)
        final_pose = np.concatenate(
            (self.env.bottle_position(), self.env.bottle_quaternion_wxyz())
        )
        return ExpertEpisode(
            seed=seed,
            initial_bottle_pose=initial_pose.tolist(),
            observations_state=np.asarray(state_rows, dtype=np.float32),
            observations_environment_state=np.asarray(env_rows, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            phases=phases,
            episode_length=len(actions),
            success=details.success,
            final_bottle_pose=final_pose.tolist(),
            success_details=asdict(details),
        )
