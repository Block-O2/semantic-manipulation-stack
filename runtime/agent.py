"""Event-triggered closed-loop semantic agent runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from planner import (
    Goal,
    Plan,
    PlannerBackend,
    PlannerFailureContext,
    PlanStatus,
    PlanValidator,
    SemanticResidual,
    SkillRegistry,
    compare_expected_effects,
    derive_expected_effects,
)
from runtime.agent_results import (
    AgentFailure,
    AgentResult,
    AgentStepEvent,
    ExecutedAgentStep,
    task_result_dict,
)
from runtime.executor import SkillExecutor
from runtime.skill_factory import SkillFactory
from world import WorldState


WorldObserver = Callable[[], WorldState]
StepHook = Callable[[AgentStepEvent], None]


def goal_is_satisfied(goal: Goal, state: WorldState) -> bool:
    relation = WorldState.relation_key(goal.object_name, goal.target_name)
    return state.relations.get(relation, False)


class AgentRuntime:
    """Plan, validate, execute semantic skills, observe, compare, and replan."""

    def __init__(
        self,
        *,
        planner: PlannerBackend,
        registry: SkillRegistry,
        validator: PlanValidator,
        observer: WorldObserver,
        skill_factory: SkillFactory,
        executor: SkillExecutor | None = None,
        max_replans: int = 2,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        if max_replans < 0:
            raise ValueError("max_replans must be non-negative")
        self._planner = planner
        self._registry = registry
        self._validator = validator
        self._observer = observer
        self._skill_factory = skill_factory
        self._executor = executor or SkillExecutor()
        self.max_replans = max_replans
        self._logger = logger

    def _emit(self, trace: list[str], message: str) -> None:
        trace.append(message)
        if self._logger is not None:
            self._logger(message)

    @staticmethod
    def _result(
        *,
        success: bool,
        goal: Goal,
        planner_calls: int,
        replans: int,
        executed: list[ExecutedAgentStep],
        state: WorldState,
        failure_reason: AgentFailure | None,
        failure_detail: str | None,
        plans: list[Plan],
        trace: list[str],
    ) -> AgentResult:
        return AgentResult(
            success=success,
            goal=goal,
            planner_calls=planner_calls,
            replans=replans,
            executed_steps=tuple(executed),
            final_world_state=state,
            failure_reason=failure_reason,
            failure_detail=failure_detail,
            plan_history=tuple(plans),
            trace=tuple(trace),
        )

    def run(self, goal: Goal, *, after_step: StepHook | None = None) -> AgentResult:
        trace: list[str] = []
        executed: list[ExecutedAgentStep] = []
        plans: list[Plan] = []
        planner_calls = 0
        replans = 0
        previous_plan: Plan | None = None
        failure_context: PlannerFailureContext | None = None
        state = self._observer()
        self._emit(trace, f"GOAL: {goal.original_text}")

        if goal_is_satisfied(goal, state):
            self._emit(trace, "GOAL CHECK: SATISFIED")
            self._emit(trace, "AGENT SUCCESS")
            return self._result(
                success=True,
                goal=goal,
                planner_calls=0,
                replans=0,
                executed=executed,
                state=state,
                failure_reason=None,
                failure_detail=None,
                plans=plans,
                trace=trace,
            )

        while True:
            planner_calls += 1
            planner_result = self._planner.plan(
                goal,
                state,
                self._registry,
                previous_plan=previous_plan,
                failure_context=failure_context,
            )
            validation = self._validator.validate(planner_result, state)
            candidate_plan = validation.plan

            if not validation.valid:
                self._emit(trace, f"PLAN v{planner_calls}: INVALID")
                for error in validation.errors:
                    self._emit(trace, f"VALIDATION ERROR: {error}")
                failure_context = PlannerFailureContext(
                    event="PLAN_VALIDATION_FAILED",
                    completed_steps=len(executed),
                    validation_errors=validation.errors,
                )
                if replans >= self.max_replans:
                    self._emit(trace, "AGENT REPLAN EXHAUSTED")
                    return self._result(
                        success=False,
                        goal=goal,
                        planner_calls=planner_calls,
                        replans=replans,
                        executed=executed,
                        state=state,
                        failure_reason=AgentFailure.AGENT_REPLAN_EXHAUSTED,
                        failure_detail="; ".join(validation.errors),
                        plans=plans,
                        trace=trace,
                    )
                if candidate_plan is not None:
                    plans.append(candidate_plan)
                    previous_plan = candidate_plan
                replans += 1
                self._emit(trace, "REPLAN REQUIRED: PLAN_VALIDATION_FAILED")
                continue

            assert candidate_plan is not None
            plans.append(candidate_plan)
            self._emit(trace, f"PLAN v{len(plans)}")
            for index, step in enumerate(candidate_plan.steps):
                self._emit(trace, f"{index + 1}. {step.skill}({step.args})")

            if candidate_plan.status is PlanStatus.CANNOT_PLAN:
                self._emit(trace, f"CANNOT_PLAN: {candidate_plan.reason}")
                return self._result(
                    success=False,
                    goal=goal,
                    planner_calls=planner_calls,
                    replans=replans,
                    executed=executed,
                    state=state,
                    failure_reason=AgentFailure.CANNOT_PLAN,
                    failure_detail=candidate_plan.reason,
                    plans=plans,
                    trace=trace,
                )

            event_context: PlannerFailureContext | None = None
            for plan_step_index, step in enumerate(candidate_plan.steps):
                state = self._observer()
                if goal_is_satisfied(goal, state):
                    self._emit(trace, "GOAL CHECK: SATISFIED")
                    self._emit(trace, "AGENT SUCCESS")
                    return self._result(
                        success=True,
                        goal=goal,
                        planner_calls=planner_calls,
                        replans=replans,
                        executed=executed,
                        state=state,
                        failure_reason=None,
                        failure_detail=None,
                        plans=plans,
                        trace=trace,
                    )

                precondition_failures = self._registry.check_preconditions(step, state)
                if precondition_failures:
                    reasons = tuple(item.reason for item in precondition_failures)
                    self._emit(
                        trace,
                        "REPLAN REQUIRED: NEXT_STEP_PRECONDITION_FAILED " + ",".join(reasons),
                    )
                    event_context = PlannerFailureContext(
                        event="NEXT_STEP_PRECONDITION_FAILED",
                        completed_steps=len(executed),
                        failed_step=step.to_dict(),
                        validation_errors=reasons,
                    )
                    break

                expected = derive_expected_effects(self._registry, step)
                self._emit(trace, f"EXECUTE {len(executed) + 1}: {step.skill}({step.args})")
                self._emit(trace, f"EXPECTED EFFECT: {expected.to_dict()}")
                task_result = self._executor.execute([self._skill_factory.create(step)])

                event = AgentStepEvent(
                    sequence=len(executed),
                    plan_version=len(plans),
                    plan_step_index=plan_step_index,
                    step=step,
                    task_result=task_result,
                )
                if after_step is not None:
                    after_step(event)

                observed = self._observer()
                residual = compare_expected_effects(expected, observed)
                executed.append(
                    ExecutedAgentStep(
                        sequence=event.sequence,
                        plan_version=event.plan_version,
                        plan_step_index=plan_step_index,
                        step=step,
                        task_result=task_result,
                        expected_effects=expected,
                        residual=residual,
                        world_before=state,
                        world_after=observed,
                    )
                )
                self._emit(
                    trace,
                    "SEMANTIC RESIDUAL: "
                    + ("CONSISTENT" if residual.consistent else str(residual.to_dict())),
                )

                if not task_result.success:
                    event_context = PlannerFailureContext(
                        event="SKILL_FAILURE",
                        completed_steps=len(executed),
                        failed_step=step.to_dict(),
                        skill_result=task_result_dict(task_result),
                        semantic_residual=residual.to_dict(),
                    )
                    self._emit(trace, "REPLAN REQUIRED: SKILL_FAILURE")
                    state = observed
                    break
                if not residual.consistent:
                    event_context = PlannerFailureContext(
                        event="SEMANTIC_RESIDUAL",
                        completed_steps=len(executed),
                        failed_step=step.to_dict(),
                        skill_result=task_result_dict(task_result),
                        semantic_residual=residual.to_dict(),
                    )
                    self._emit(trace, "REPLAN REQUIRED: SEMANTIC_RESIDUAL")
                    state = observed
                    break
                if goal_is_satisfied(goal, observed):
                    self._emit(trace, "GOAL CHECK: SATISFIED")
                    self._emit(trace, "AGENT SUCCESS")
                    return self._result(
                        success=True,
                        goal=goal,
                        planner_calls=planner_calls,
                        replans=replans,
                        executed=executed,
                        state=observed,
                        failure_reason=None,
                        failure_detail=None,
                        plans=plans,
                        trace=trace,
                    )
                state = observed

            if event_context is None:
                state = self._observer()
                if goal_is_satisfied(goal, state):
                    self._emit(trace, "GOAL CHECK: SATISFIED")
                    self._emit(trace, "AGENT SUCCESS")
                    return self._result(
                        success=True,
                        goal=goal,
                        planner_calls=planner_calls,
                        replans=replans,
                        executed=executed,
                        state=state,
                        failure_reason=None,
                        failure_detail=None,
                        plans=plans,
                        trace=trace,
                    )
                event_context = PlannerFailureContext(
                    event="PLAN_EXHAUSTED_WITHOUT_GOAL",
                    completed_steps=len(executed),
                )
                self._emit(trace, "REPLAN REQUIRED: PLAN_EXHAUSTED_WITHOUT_GOAL")

            if replans >= self.max_replans:
                self._emit(trace, "AGENT REPLAN EXHAUSTED")
                return self._result(
                    success=False,
                    goal=goal,
                    planner_calls=planner_calls,
                    replans=replans,
                    executed=executed,
                    state=state,
                    failure_reason=AgentFailure.AGENT_REPLAN_EXHAUSTED,
                    failure_detail=event_context.event,
                    plans=plans,
                    trace=trace,
                )
            previous_plan = candidate_plan
            failure_context = event_context
            replans += 1
