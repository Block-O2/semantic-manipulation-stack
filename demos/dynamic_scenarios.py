"""Run scripted Dynamic World Playground scenarios."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from planner import Goal, PlanValidator, RuleBasedPlanner, SkillRegistry
from playground import WorldController
from primitives import ManipulationPrimitives
from robot import PandaRobot
from runtime import AgentBoundaryEvent, AgentResult, AgentRuntime, SemanticSkillFactory
from sim import make_environment
from world import WorldModel


INSIDE_GOAL = Goal.put_inside(
    "Put the red cube inside the blue target.",
    "red_cube",
    "blue_target",
)


def _result_record(name: str, result: AgentResult) -> dict[str, Any]:
    return {
        "scenario": name,
        "success": result.success,
        "planner_calls": result.planner_calls,
        "replans": result.replans,
        "executed_steps": len(result.executed_steps),
        "failure_reason": (
            result.failure_reason.value if result.failure_reason else None
        ),
        "failure_detail": result.failure_detail,
        "missing_capabilities": (
            list(result.plan_history[-1].missing_capabilities)
            if result.plan_history
            else []
        ),
        "goal_satisfied": result.final_world_state.relations.get(
            "red_cube_inside_blue_target", False
        ),
        "final_occupancy": {
            target: list(state.occupied_by)
            for target, state in result.final_world_state.targets.items()
        },
        "trace": list(result.trace),
    }


def run_scenario(name: str) -> dict[str, Any]:
    env = make_environment(render=False, randomize_cube=False, seed=73)
    try:
        env.reset()
        robot = PandaRobot(env)
        world = WorldModel(env)
        controller = WorldController(env, world)
        registry = SkillRegistry.standard()
        triggered = False

        if name == "occupy":
            controller.place_object_in_target("green_cube", "blue_target")

        def change_world(event: AgentBoundaryEvent) -> None:
            nonlocal triggered
            if triggered or event.step.skill != "place":
                return
            if name == "drop":
                controller.drop_held_object()
            elif name == "move_reachable":
                controller.move_target("blue_target", 0.20, 0.10)
            elif name == "move_unreachable":
                controller.move_target("blue_target", 0.50, 0.50)
            elif name == "occupy_after_pick":
                controller.place_object_in_target("green_cube", "blue_target")
            else:
                return
            triggered = True

        goal = (
            Goal.open_drawer("Open the drawer.")
            if name == "capability_gap"
            else INSIDE_GOAL
        )
        result = AgentRuntime(
            planner=RuleBasedPlanner(),
            registry=registry,
            validator=PlanValidator(registry),
            observer=world.semantic_state,
            skill_factory=SemanticSkillFactory(
                world,
                ManipulationPrimitives(robot),
            ),
        ).run(goal, before_step=change_world)
        record = _result_record(name, result)
        record["disturbance_triggered"] = triggered
        return record
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = (
        "nominal",
        "drop",
        "move_reachable",
        "move_unreachable",
        "occupy",
        "occupy_after_pick",
        "capability_gap",
    )
    parser.add_argument("--scenario", choices=(*choices, "all"), default="all")
    args = parser.parse_args()
    selected = choices if args.scenario == "all" else (args.scenario,)
    records = [run_scenario(name) for name in selected]
    for record in records:
        print(json.dumps(record, indent=2, sort_keys=True))

    expectations: dict[str, Callable[[dict[str, Any]], bool]] = {
        "nominal": lambda item: bool(item["success"]),
        "drop": lambda item: bool(item["success"] and item["replans"] == 1),
        "move_reachable": lambda item: bool(
            item["success"] and item["replans"] == 0
        ),
        "move_unreachable": lambda item: bool(
            not item["success"] and item["failure_reason"] == "CANNOT_PLAN"
        ),
        "occupy": lambda item: bool(
            item["success"]
            and item["planner_calls"] == 1
            and item["replans"] == 0
            and item["executed_steps"] == 4
            and item["final_occupancy"]["blue_target"] == ["red_cube"]
            and item["final_occupancy"]["temporary_area"] == ["green_cube"]
        ),
        "occupy_after_pick": lambda item: bool(
            item["success"]
            and item["planner_calls"] == 2
            and item["replans"] == 1
            and item["executed_steps"] == 6
        ),
        "capability_gap": lambda item: bool(
            not item["success"]
            and item["failure_reason"] == "CAPABILITY_GAP"
            and item["missing_capabilities"] == ["open_drawer"]
        ),
    }
    if not all(expectations[item["scenario"]](item) for item in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
