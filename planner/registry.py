"""Machine-readable semantic skill contracts for the current skill library."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from planner.schema import PlanStep
from world import WorldState


class ArgumentKind(str, Enum):
    OBJECT = "object"
    TARGET = "target"


@dataclass(frozen=True)
class SkillArgument:
    name: str
    kind: ArgumentKind
    description: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "description": self.description,
        }


def _render(value: Any, args: Mapping[str, str]) -> Any:
    if not isinstance(value, str):
        return value
    rendered = value
    for name, argument in args.items():
        rendered = rendered.replace("{" + name + "}", argument)
    return rendered


@dataclass(frozen=True)
class ConditionTemplate:
    path: str
    expected: Any
    failure_reason: str

    def render(self, args: Mapping[str, str]) -> tuple[str, Any]:
        return _render(self.path, args), _render(self.expected, args)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "expected": self.expected,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class EffectTemplate:
    path: str
    expected: Any
    mismatch_code: str

    def render(self, args: Mapping[str, str]) -> tuple[str, Any]:
        return _render(self.path, args), _render(self.expected, args)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "expected": self.expected,
            "mismatch_code": self.mismatch_code,
        }


@dataclass(frozen=True)
class SkillSpec:
    name: str
    arguments: tuple[SkillArgument, ...]
    preconditions: tuple[ConditionTemplate, ...]
    expected_effects: tuple[EffectTemplate, ...]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": [argument.to_dict() for argument in self.arguments],
            "preconditions": [item.to_dict() for item in self.preconditions],
            "expected_effects": [item.to_dict() for item in self.expected_effects],
            "description": self.description,
        }


@dataclass(frozen=True)
class PreconditionFailure:
    path: str
    expected: Any
    observed: Any
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "expected": self.expected,
            "observed": self.observed,
            "reason": self.reason,
        }


class SkillRegistry:
    def __init__(self, specs: tuple[SkillSpec, ...]) -> None:
        by_name = {spec.name: spec for spec in specs}
        if len(by_name) != len(specs):
            raise ValueError("skill names must be unique")
        self._specs = by_name

    @classmethod
    def standard(cls) -> "SkillRegistry":
        pick = SkillSpec(
            name="pick",
            arguments=(SkillArgument("object", ArgumentKind.OBJECT, "Object to grasp"),),
            preconditions=(
                ConditionTemplate("objects.{object}.exists", True, "OBJECT_NOT_FOUND"),
                ConditionTemplate("objects.{object}.reachable", True, "OBJECT_UNREACHABLE"),
                ConditionTemplate("robot.holding", None, "GRIPPER_NOT_EMPTY"),
            ),
            expected_effects=(
                EffectTemplate("robot.holding", "{object}", "OBJECT_LOST"),
                EffectTemplate(
                    "objects.{object}.grasped",
                    True,
                    "GRASP_EFFECT_NOT_ACHIEVED",
                ),
            ),
            description="Grasp one reachable known object with an empty gripper.",
        )
        place = SkillSpec(
            name="place",
            arguments=(
                SkillArgument("object", ArgumentKind.OBJECT, "Held object to release"),
                SkillArgument("target", ArgumentKind.TARGET, "Destination target region"),
            ),
            preconditions=(
                ConditionTemplate("objects.{object}.exists", True, "OBJECT_NOT_FOUND"),
                ConditionTemplate("robot.holding", "{object}", "OBJECT_NOT_GRASPED"),
                ConditionTemplate("targets.{target}.exists", True, "TARGET_NOT_FOUND"),
                ConditionTemplate("targets.{target}.reachable", True, "TARGET_UNREACHABLE"),
                ConditionTemplate("targets.{target}.occupied", False, "TARGET_OCCUPIED"),
            ),
            expected_effects=(
                EffectTemplate("robot.holding", None, "RELEASE_EFFECT_NOT_ACHIEVED"),
                EffectTemplate(
                    "objects.{object}.grasped",
                    False,
                    "RELEASE_EFFECT_NOT_ACHIEVED",
                ),
                EffectTemplate(
                    "relations.{object}_inside_{target}",
                    True,
                    "PLACE_EFFECT_NOT_ACHIEVED",
                ),
                EffectTemplate("targets.{target}.occupied", True, "TARGET_NOT_OCCUPIED"),
            ),
            description="Release one held object at the centre of a reachable target.",
        )
        return cls((pick, place))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def get(self, name: str) -> SkillSpec | None:
        return self._specs.get(name)

    def require(self, name: str) -> SkillSpec:
        spec = self.get(name)
        if spec is None:
            raise KeyError(name)
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {name: self._specs[name].to_dict() for name in self.names}

    def check_preconditions(
        self,
        step: PlanStep,
        state: WorldState | Mapping[str, Any],
    ) -> tuple[PreconditionFailure, ...]:
        spec = self.require(step.skill)
        values = state.flattened() if isinstance(state, WorldState) else state
        failures: list[PreconditionFailure] = []
        for condition in spec.preconditions:
            path, expected = condition.render(step.args)
            observed = values.get(path, "<missing>")
            if observed != expected:
                failures.append(
                    PreconditionFailure(path, expected, observed, condition.failure_reason)
                )
        return tuple(failures)
