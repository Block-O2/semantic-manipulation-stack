"""Evaluation-only simulator disturbances for closed-loop agent experiments."""

from __future__ import annotations

from playground import WorldController
from runtime import AgentStepEvent
from sim import SemanticTabletopEnv
from world import WorldModel


class ObjectLostAfterPick:
    """Teleport the cube back to the table once after a successful Pick."""

    def __init__(
        self,
        env: SemanticTabletopEnv,
        *,
        table_position: tuple[float, float, float] = (-0.05, -0.10, 0.825),
    ) -> None:
        self._controller = WorldController(env, WorldModel(env))
        self._position = table_position
        self.triggered = False

    def __call__(self, event: AgentStepEvent) -> None:
        if self.triggered or event.step.skill != "pick" or not event.task_result.success:
            return
        self._controller.move_object(
            "red_cube", self._position[0], self._position[1], z=self._position[2]
        )
        self.triggered = True


class TargetUnreachableAfterPick:
    """Move the visual target outside semantic workspace once after Pick."""

    def __init__(
        self,
        env: SemanticTabletopEnv,
        *,
        target_position: tuple[float, float, float] = (0.50, 0.50, 0.803),
    ) -> None:
        self._controller = WorldController(env, WorldModel(env))
        self._position = target_position
        self.triggered = False

    def __call__(self, event: AgentStepEvent) -> None:
        if self.triggered or event.step.skill != "pick" or not event.task_result.success:
            return
        self._controller.move_target(
            "blue_target", self._position[0], self._position[1]
        )
        self.triggered = True
