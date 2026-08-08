from __future__ import annotations

import numpy as np

from planner import Goal, PlanValidator, RuleBasedPlanner, SkillRegistry
from primitives import ManipulationPrimitives
from robot import PandaRobot
from runtime import AgentRuntime, SemanticSkillFactory
from sim import make_environment
from world import WorldModel


def test_full_agent_executes_physical_push_to_right_side() -> None:
    env = make_environment(render=False, seed=109)
    try:
        env.reset()
        robot = PandaRobot(env)
        world = WorldModel(env)
        registry = SkillRegistry.standard()
        initial = world.pose("red_cube").position.copy()

        result = AgentRuntime(
            planner=RuleBasedPlanner(),
            registry=registry,
            validator=PlanValidator(registry),
            observer=world.semantic_state,
            skill_factory=SemanticSkillFactory(
                world,
                ManipulationPrimitives(robot),
            ),
        ).run(
            Goal.push_to_region(
                "Push the red cube toward the right side of the table.",
                "red_cube",
                "right_side",
            )
        )

        final = world.pose("red_cube").position
        assert result.success
        assert result.planner_calls == 1
        assert result.replans == 0
        assert len(result.executed_steps) == 1
        assert result.executed_steps[0].step.skill == "push"
        assert result.executed_steps[0].residual.consistent
        assert np.linalg.norm(final[:2] - initial[:2]) >= 0.06
        assert result.final_world_state.relations["red_cube_inside_right_side"]
    finally:
        env.close()
