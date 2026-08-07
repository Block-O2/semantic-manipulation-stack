"""Run controlled object-loss and unreachable-target AgentRuntime scenarios."""

from __future__ import annotations

import json
from typing import Any

from evaluation.disturbances import ObjectLostAfterPick, TargetUnreachableAfterPick
from planner import Goal, PlanValidator, RuleBasedPlanner, SkillRegistry
from primitives import ManipulationPrimitives
from robot import PandaRobot
from runtime import AgentRuntime, SemanticSkillFactory
from sim import make_environment
from world import WorldModel


GOAL = Goal.put_inside(
    "Put the red cube inside the blue target.",
    "red_cube",
    "blue_target",
)


def _run_scenario(kind: str) -> dict[str, Any]:
    env = make_environment(render=False, randomize_cube=False, seed=61)
    try:
        env.reset()
        world = WorldModel(env)
        primitives = ManipulationPrimitives(PandaRobot(env))
        registry = SkillRegistry.standard()
        disturbance = (
            ObjectLostAfterPick(env)
            if kind == "object_lost"
            else TargetUnreachableAfterPick(env)
        )
        result = AgentRuntime(
            planner=RuleBasedPlanner(),
            registry=registry,
            validator=PlanValidator(registry),
            observer=world.semantic_state,
            skill_factory=SemanticSkillFactory(world, primitives),
            max_replans=2,
        ).run(GOAL, after_step=disturbance)
        return {
            "scenario": kind,
            "disturbance_triggered": disturbance.triggered,
            "success": result.success,
            "planner_calls": result.planner_calls,
            "replans": result.replans,
            "executed_steps": len(result.executed_steps),
            "failure_reason": (
                result.failure_reason.value if result.failure_reason else None
            ),
            "failure_detail": result.failure_detail,
            "goal_satisfied": result.final_world_state.relations[
                "red_cube_inside_blue_target"
            ],
            "plan_history": [plan.to_dict() for plan in result.plan_history],
            "residuals": [step.residual.to_dict() for step in result.executed_steps],
            "trace": list(result.trace),
        }
    finally:
        env.close()


def main() -> None:
    object_lost = _run_scenario("object_lost")
    target_unreachable = _run_scenario("target_unreachable")
    print(json.dumps(object_lost, indent=2, sort_keys=True))
    print(json.dumps(target_unreachable, indent=2, sort_keys=True))
    if not (
        object_lost["success"]
        and object_lost["replans"] == 1
        and target_unreachable["failure_reason"] == "CANNOT_PLAN"
        and target_unreachable["replans"] == 1
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
