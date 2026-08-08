"""Validated PlanStep to semantic Skill construction boundary."""

from __future__ import annotations

from typing import Protocol

from planner import PlanStep
from primitives import ManipulationPrimitives
from skills import ClassicalPushBackend, PickSkill, PlaceSkill, PushBackend, PushSkill, Skill
from world import WorldModel


class SkillFactory(Protocol):
    def create(self, step: PlanStep) -> Skill:
        """Create one semantic Skill from an already-validated plan step."""


class SemanticSkillFactory:
    """The only runtime adapter mapping registry names to production skills."""

    def __init__(
        self,
        world: WorldModel,
        primitives: ManipulationPrimitives,
        *,
        push_backend: PushBackend | None = None,
    ) -> None:
        self._world = world
        self._primitives = primitives
        self._push_backend = push_backend or ClassicalPushBackend()

    def create(self, step: PlanStep) -> Skill:
        if step.skill == "pick":
            return PickSkill(step.args["object"], self._world, self._primitives)
        if step.skill == "place":
            return PlaceSkill(
                step.args["object"],
                step.args["target"],
                self._world,
                self._primitives,
            )
        if step.skill == "push":
            return PushSkill(
                step.args["object"],
                step.args["target"],
                self._world,
                self._primitives,
                backend=self._push_backend,
            )
        raise ValueError(f"validated skill has no implementation: {step.skill}")
