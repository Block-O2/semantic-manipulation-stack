"""Symbolic expected effects and explicit post-skill semantic residuals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from planner.registry import SkillRegistry
from planner.schema import PlanStep
from world import WorldState


@dataclass(frozen=True)
class ExpectedEffect:
    path: str
    expected: Any
    mismatch_code: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "expected": self.expected,
            "mismatch_code": self.mismatch_code,
        }


@dataclass(frozen=True)
class ExpectedWorldEffects:
    skill: str
    args: dict[str, str]
    effects: tuple[ExpectedEffect, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "args": dict(sorted(self.args.items())),
            "effects": [effect.to_dict() for effect in self.effects],
        }


class ResidualSeverity(str, Enum):
    NONE = "none"
    CRITICAL = "critical"


@dataclass(frozen=True)
class EffectMismatch:
    path: str
    expected: Any
    observed: Any
    code: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "expected": self.expected,
            "observed": self.observed,
            "code": self.code,
        }


@dataclass(frozen=True)
class SemanticResidual:
    consistent: bool
    mismatches: tuple[EffectMismatch, ...]
    severity: ResidualSeverity

    def to_dict(self) -> dict[str, Any]:
        return {
            "consistent": self.consistent,
            "mismatches": [item.to_dict() for item in self.mismatches],
            "severity": self.severity.value,
        }


def derive_expected_effects(
    registry: SkillRegistry,
    step: PlanStep,
) -> ExpectedWorldEffects:
    spec = registry.require(step.skill)
    effects = tuple(
        ExpectedEffect(*template.render(step.args), template.mismatch_code)
        for template in spec.expected_effects
    )
    return ExpectedWorldEffects(step.skill, dict(step.args), effects)


def compare_expected_effects(
    expected: ExpectedWorldEffects,
    observed: WorldState,
) -> SemanticResidual:
    mismatches: list[EffectMismatch] = []
    for effect in expected.effects:
        try:
            actual = observed.value(effect.path)
        except KeyError:
            actual = "<missing>"
        if actual != effect.expected:
            mismatches.append(
                EffectMismatch(
                    path=effect.path,
                    expected=effect.expected,
                    observed=actual,
                    code=effect.mismatch_code,
                )
            )
    return SemanticResidual(
        consistent=not mismatches,
        mismatches=tuple(mismatches),
        severity=(ResidualSeverity.NONE if not mismatches else ResidualSeverity.CRITICAL),
    )
