"""Provider-independent goal, plan, and planner-response schemas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class GoalRelation(str, Enum):
    INSIDE = "inside"


@dataclass(frozen=True)
class Goal:
    original_text: str
    relation: GoalRelation
    object_name: str
    target_name: str

    @classmethod
    def put_inside(cls, text: str, object_name: str, target_name: str) -> "Goal":
        if not text.strip() or not object_name or not target_name:
            raise ValueError("goal text, object name, and target name are required")
        return cls(text, GoalRelation.INSIDE, object_name, target_name)

    def to_dict(self) -> dict[str, str]:
        return {
            "original_text": self.original_text,
            "relation": self.relation.value,
            "object": self.object_name,
            "target": self.target_name,
        }


class PlanStatus(str, Enum):
    READY = "ready"
    CANNOT_PLAN = "cannot_plan"


@dataclass(frozen=True)
class PlanStep:
    skill: str
    args: dict[str, str]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanStep":
        if set(payload) != {"skill", "args"}:
            raise ValueError("plan step must contain exactly 'skill' and 'args'")
        skill = payload["skill"]
        args = payload["args"]
        if not isinstance(skill, str) or not skill:
            raise ValueError("plan step skill must be a non-empty string")
        if not isinstance(args, Mapping):
            raise ValueError("plan step args must be an object")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in args.items()
        ):
            raise ValueError("plan step args must map strings to strings")
        return cls(skill=skill, args=dict(args))

    def to_dict(self) -> dict[str, Any]:
        return {"skill": self.skill, "args": dict(sorted(self.args.items()))}


@dataclass(frozen=True)
class Plan:
    status: PlanStatus
    steps: tuple[PlanStep, ...]
    reason: str | None = None

    @classmethod
    def ready(cls, steps: list[PlanStep] | tuple[PlanStep, ...]) -> "Plan":
        return cls(PlanStatus.READY, tuple(steps), None)

    @classmethod
    def cannot_plan(cls, reason: str) -> "Plan":
        if not reason.strip():
            raise ValueError("CANNOT_PLAN requires a semantic reason")
        return cls(PlanStatus.CANNOT_PLAN, (), reason)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Plan":
        allowed = {"status", "steps", "reason"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown plan fields: {sorted(unknown)}")
        status_raw = payload.get("status", PlanStatus.READY.value)
        try:
            status = PlanStatus(status_raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid plan status: {status_raw!r}") from exc

        if status is PlanStatus.CANNOT_PLAN:
            if set(payload) - {"status", "reason"}:
                raise ValueError("CANNOT_PLAN must not contain steps")
            reason = payload.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("CANNOT_PLAN requires a semantic reason")
            return cls.cannot_plan(reason)

        if "reason" in payload:
            raise ValueError("ready plans must not contain a reason")
        steps_raw = payload.get("steps")
        if not isinstance(steps_raw, list):
            raise ValueError("ready plan requires a steps array")
        steps = []
        for raw_step in steps_raw:
            if not isinstance(raw_step, Mapping):
                raise ValueError("every plan step must be an object")
            steps.append(PlanStep.from_dict(raw_step))
        return cls.ready(steps)

    def to_dict(self) -> dict[str, Any]:
        if self.status is PlanStatus.CANNOT_PLAN:
            return {"status": self.status.value, "reason": self.reason}
        return {
            "status": self.status.value,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class PlannerResult:
    """Raw planner output plus optional already-parsed deterministic plan."""

    raw_response: Any
    plan: Plan | None = None
    error: str | None = None

    @classmethod
    def from_plan(cls, plan: Plan) -> "PlannerResult":
        return cls(raw_response=plan.to_dict(), plan=plan)


@dataclass(frozen=True)
class PlannerFailureContext:
    event: str
    completed_steps: int
    failed_step: dict[str, Any] | None = None
    skill_result: dict[str, Any] | None = None
    semantic_residual: dict[str, Any] | None = None
    validation_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "completed_steps": self.completed_steps,
            "failed_step": self.failed_step,
            "skill_result": self.skill_result,
            "semantic_residual": self.semantic_residual,
            "validation_errors": list(self.validation_errors),
        }
