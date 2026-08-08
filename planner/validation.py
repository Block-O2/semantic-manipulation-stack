"""Strict deterministic validation for all planner-produced plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from planner.composition import SymbolicManipulationState
from planner.effects import derive_expected_effects
from planner.registry import ArgumentKind, SkillRegistry
from planner.schema import Plan, PlannerResult, PlanStatus
from world import WorldState


@dataclass(frozen=True)
class PlanValidationResult:
    valid: bool
    plan: Plan | None
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "plan": self.plan.to_dict() if self.plan else None,
            "errors": list(self.errors),
        }


class PlanValidator:
    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def validate(self, result: PlannerResult, world_state: WorldState) -> PlanValidationResult:
        if result.error is not None:
            return PlanValidationResult(False, None, (f"PLANNER_OUTPUT_ERROR: {result.error}",))

        plan = result.plan
        if plan is None:
            if not isinstance(result.raw_response, Mapping):
                return PlanValidationResult(
                    False,
                    None,
                    ("PLAN_SCHEMA_ERROR: planner response must be a JSON object",),
                )
            try:
                plan = Plan.from_dict(result.raw_response)
            except ValueError as exc:
                return PlanValidationResult(False, None, (f"PLAN_SCHEMA_ERROR: {exc}",))

        if plan.status is PlanStatus.CANNOT_PLAN:
            return PlanValidationResult(True, plan, ())

        projection = SymbolicManipulationState.from_world_state(world_state)
        errors: list[str] = []
        for index, step in enumerate(plan.steps):
            spec = self.registry.get(step.skill)
            if spec is None:
                errors.append(f"STEP_{index}_UNKNOWN_SKILL: {step.skill}")
                continue

            expected_args = {argument.name for argument in spec.arguments}
            actual_args = set(step.args)
            missing = expected_args - actual_args
            extra = actual_args - expected_args
            if missing:
                errors.append(f"STEP_{index}_MISSING_ARGS: {sorted(missing)}")
            if extra:
                errors.append(f"STEP_{index}_UNKNOWN_ARGS: {sorted(extra)}")
            if missing or extra:
                continue

            entity_error = False
            for argument in spec.arguments:
                value = step.args[argument.name]
                if argument.kind is ArgumentKind.OBJECT and value not in world_state.objects:
                    errors.append(f"STEP_{index}_UNKNOWN_OBJECT: {value}")
                    entity_error = True
                if argument.kind is ArgumentKind.TARGET and value not in world_state.targets:
                    errors.append(f"STEP_{index}_UNKNOWN_TARGET: {value}")
                    entity_error = True
                if (
                    argument.kind is ArgumentKind.PUSH_REGION
                    and value not in world_state.push_regions
                ):
                    errors.append(f"STEP_{index}_UNKNOWN_PUSH_REGION: {value}")
                    entity_error = True
            if entity_error:
                continue

            failures = self.registry.check_preconditions(step, projection.flattened())
            for failure in failures:
                errors.append(
                    f"STEP_{index}_PRECONDITION_{failure.reason}: "
                    f"{failure.path} expected {failure.expected!r}, "
                    f"observed {failure.observed!r}"
                )
            if failures:
                continue

            expected = derive_expected_effects(self.registry, step)
            try:
                projection = projection.apply(step)
            except ValueError as exc:
                errors.append(f"STEP_{index}_NO_SYMBOLIC_TRANSITION: {exc}")
                continue
            projected_values = projection.flattened()
            for effect in expected.effects:
                if projected_values.get(effect.path, "<missing>") != effect.expected:
                    errors.append(
                        f"STEP_{index}_INVALID_SYMBOLIC_EFFECT: {effect.path}"
                    )

        return PlanValidationResult(not errors, plan, tuple(errors))
