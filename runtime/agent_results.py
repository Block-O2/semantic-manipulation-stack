"""Structured closed-loop agent outcomes preserving lower result layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from planner import ExpectedWorldEffects, Goal, Plan, PlanStep, SemanticResidual
from runtime.results import TaskResult
from world import WorldState


class AgentFailure(str, Enum):
    CANNOT_PLAN = "CANNOT_PLAN"
    CAPABILITY_GAP = "CAPABILITY_GAP"
    AGENT_REPLAN_EXHAUSTED = "AGENT_REPLAN_EXHAUSTED"


def _skill_result_dict(result: Any) -> dict[str, Any]:
    return {
        "success": result.success,
        "skill": result.skill,
        "object_name": result.object_name,
        "phase": result.phase.value,
        "reason": result.reason.value if result.reason else None,
        "attempts": result.attempts,
        "trace": list(result.trace),
    }


def task_result_dict(result: TaskResult) -> dict[str, Any]:
    return {
        "success": result.success,
        "completed_steps": result.completed_steps,
        "failed_step": result.failed_step,
        "failed_skill": result.failed_skill,
        "reason": result.reason.value if result.reason else None,
        "skill_results": [_skill_result_dict(item) for item in result.skill_results],
        "trace": list(result.trace),
    }


@dataclass(frozen=True)
class ExecutedAgentStep:
    sequence: int
    plan_version: int
    plan_step_index: int
    step: PlanStep
    task_result: TaskResult
    expected_effects: ExpectedWorldEffects
    residual: SemanticResidual
    world_before: WorldState
    world_after: WorldState

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "plan_version": self.plan_version,
            "plan_step_index": self.plan_step_index,
            "step": self.step.to_dict(),
            "task_result": task_result_dict(self.task_result),
            "expected_effects": self.expected_effects.to_dict(),
            "residual": self.residual.to_dict(),
            "world_before": self.world_before.to_dict(),
            "world_after": self.world_after.to_dict(),
        }


@dataclass(frozen=True)
class AgentResult:
    success: bool
    goal: Goal
    planner_calls: int
    replans: int
    executed_steps: tuple[ExecutedAgentStep, ...]
    final_world_state: WorldState
    failure_reason: AgentFailure | None
    failure_detail: str | None
    plan_history: tuple[Plan, ...]
    trace: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "goal": self.goal.to_dict(),
            "planner_calls": self.planner_calls,
            "replans": self.replans,
            "executed_steps": [step.to_dict() for step in self.executed_steps],
            "final_world_state": self.final_world_state.to_dict(),
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "failure_detail": self.failure_detail,
            "plan_history": [plan.to_dict() for plan in self.plan_history],
            "trace": list(self.trace),
        }


@dataclass(frozen=True)
class AgentBoundaryEvent:
    sequence: int
    plan_version: int
    plan_step_index: int
    step: PlanStep
    world_state: WorldState


@dataclass(frozen=True)
class AgentStepEvent:
    sequence: int
    plan_version: int
    plan_step_index: int
    step: PlanStep
    task_result: TaskResult
