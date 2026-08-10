"""Deterministic Tissue FSM used only for demonstrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from agent_act_reproduction.robot import PandaCartesianAdapter
from agent_act_reproduction.sim.tissue_env import TissueEnv
from agent_act_reproduction.tasks.tissue import TissueObservation, TissueSuccessChecker


class TissuePhase(str, Enum):
    OPEN = "OPEN"
    PREGRASP = "PREGRASP"
    DESCEND = "DESCEND"
    CLOSE = "CLOSE"
    PULL = "PULL"
    HOLD = "HOLD"
    DONE = "DONE"


@dataclass
class TissueExpertEpisode:
    seed: int
    initial_tissue_pose: list[float]
    observations_state: np.ndarray
    observations_environment_state: np.ndarray
    actions: np.ndarray
    phases: list[str]
    episode_length: int
    success: bool
    final_tissue_pose: list[float]
    success_details: dict[str, object]


class TissueExpert:
    """Closed-loop approach, close and pull controller; collection only."""

    def __init__(self, env: TissueEnv, adapter: PandaCartesianAdapter) -> None:
        self.env = env
        self.adapter = adapter
        self.checker = TissueSuccessChecker()

    @staticmethod
    def _near(a: np.ndarray, b: np.ndarray, tolerance: float = 0.012) -> bool:
        return float(np.linalg.norm(a - b)) <= tolerance

    def run(self, *, seed: int, perturbation_m: float = 0.006) -> TissueExpertEpisode:
        self.env.configure_episode(seed=seed, perturbation_m=perturbation_m)
        self.env.reset()
        initial_position = self.env.tissue_position().copy()
        initial_pose = np.concatenate((initial_position, self.env.tissue_quaternion_wxyz()))
        phase = TissuePhase.OPEN
        phase_steps = 0
        state_rows: list[np.ndarray] = []
        env_rows: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        phases: list[str] = []
        for _ in range(self.env.scene.max_episode_steps):
            eef = self.env.end_effector_position()
            if phase is TissuePhase.OPEN:
                target, grip = eef.copy(), -1.0
                if phase_steps >= 20:
                    phase, phase_steps = TissuePhase.PREGRASP, 0
            elif phase is TissuePhase.PREGRASP:
                target, grip = initial_position + np.array((0.0, 0.0, 0.11)), -1.0
                if self._near(eef, target) and phase_steps >= 5:
                    phase, phase_steps = TissuePhase.DESCEND, 0
            elif phase is TissuePhase.DESCEND:
                target, grip = initial_position + np.array((0.0, 0.0, 0.020)), -1.0
                if self._near(eef, target) and phase_steps >= 5:
                    phase, phase_steps = TissuePhase.CLOSE, 0
            elif phase is TissuePhase.CLOSE:
                target, grip = eef.copy(), 1.0
                if phase_steps >= 38:
                    phase, phase_steps = TissuePhase.PULL, 0
            elif phase is TissuePhase.PULL:
                target, grip = self.env.pull_target_position(), 1.0
                if self._near(eef, target, 0.014) and phase_steps >= 5:
                    phase, phase_steps = TissuePhase.HOLD, 0
            elif phase is TissuePhase.HOLD:
                target, grip = eef.copy(), 1.0
                if phase_steps >= 24:
                    phase = TissuePhase.DONE
            else:
                break
            observation = TissueObservation.read(self.env)
            command = np.array((*target, grip), dtype=np.float64)
            state_rows.append(observation.robot_state)
            env_rows.append(observation.environment_state)
            actions.append(command)
            phases.append(phase.value)
            self.adapter.execute(command)
            phase_steps += 1
        details = self.checker.check(self.env)
        final_pose = np.concatenate((self.env.tissue_position(), self.env.tissue_quaternion_wxyz()))
        return TissueExpertEpisode(
            seed=seed,
            initial_tissue_pose=initial_pose.tolist(),
            observations_state=np.asarray(state_rows, dtype=np.float32),
            observations_environment_state=np.asarray(env_rows, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            phases=phases,
            episode_length=len(actions),
            success=details.success,
            final_tissue_pose=final_pose.tolist(),
            success_details=asdict(details),
        )
