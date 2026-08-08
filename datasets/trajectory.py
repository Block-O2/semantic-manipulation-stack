"""Control-timestep Push trajectory recorder and episode representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from datasets.specs import encode_push_action, encode_push_observation
from robot import CartesianCommandEvent, PandaRobot
from world import WorldModel


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PushEpisode:
    observations: FloatArray
    actions: FloatArray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        observations = np.asarray(self.observations, dtype=np.float64)
        actions = np.asarray(self.actions, dtype=np.float64)
        if observations.ndim != 2 or observations.shape[1] != 10:
            raise ValueError("episode observations must have shape (T, 10)")
        if actions.ndim != 2 or actions.shape[1] != 3:
            raise ValueError("episode actions must have shape (T, 3)")
        if observations.shape[0] != actions.shape[0] or observations.shape[0] == 0:
            raise ValueError("episode must contain aligned non-empty observations/actions")
        object.__setattr__(self, "observations", observations.copy())
        object.__setattr__(self, "actions", actions.copy())
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def length(self) -> int:
        return self.observations.shape[0]


class PushTrajectoryRecorder:
    """Observe the trusted Cartesian command path without changing execution."""

    def __init__(
        self,
        world: WorldModel,
        object_name: str,
        target_name: str,
    ) -> None:
        self.world = world
        self.object_name = object_name
        self.target_name = target_name
        self._observations: list[FloatArray] = []
        self._actions: list[FloatArray] = []
        self._robot: PandaRobot | None = None

    def _record(self, event: CartesianCommandEvent) -> None:
        self._observations.append(
            encode_push_observation(
                event,
                self.world,
                self.object_name,
                self.target_name,
            )
        )
        self._actions.append(encode_push_action(event))

    def attach(self, robot: PandaRobot) -> None:
        if self._robot is not None:
            raise RuntimeError("recorder is already attached")
        self._robot = robot
        robot.add_command_observer(self._record)

    def detach(self) -> None:
        if self._robot is not None:
            self._robot.remove_command_observer(self._record)
            self._robot = None

    @property
    def length(self) -> int:
        return len(self._actions)

    def episode(self, metadata: dict[str, Any]) -> PushEpisode:
        if self._robot is not None:
            raise RuntimeError("detach recorder before finalizing an episode")
        return PushEpisode(
            np.asarray(self._observations, dtype=np.float64),
            np.asarray(self._actions, dtype=np.float64),
            metadata,
        )
