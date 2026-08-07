from __future__ import annotations

from planner import Goal, PlanValidator, RuleBasedPlanner, SkillRegistry
from playground import WorldController
from primitives import ManipulationPrimitives
from robot import PandaRobot
from runtime import (
    AgentBoundaryEvent,
    AgentRuntime,
    SemanticSkillFactory,
)
from sim import make_environment
from skills import PickSkill
from world import WorldModel, WorldState


def test_geometric_relations_and_temporary_area() -> None:
    env = make_environment(render=False, seed=79)
    try:
        env.reset()
        world = WorldModel(env)
        state = world.semantic_state()

        assert state.relations[WorldState.left_of_key("green_cube", "red_cube")]
        assert state.relations[WorldState.right_of_key("red_cube", "green_cube")]
        assert state.relations[WorldState.near_key("red_cube", "blue_cube")]
        assert state.relations[WorldState.near_key("blue_cube", "red_cube")]
        assert not state.relations[WorldState.near_key("green_cube", "blue_cube")]
        assert "temporary_area" in state.targets
        assert not state.targets["temporary_area"].occupied
    finally:
        env.close()


def test_occupancy_move_remove_and_restore_are_observed() -> None:
    env = make_environment(render=False, seed=83)
    try:
        env.reset()
        world = WorldModel(env)
        controller = WorldController(env, world)

        controller.place_object_in_target("green_cube", "temporary_area")
        occupied = world.semantic_state()
        assert occupied.targets["temporary_area"].occupied
        assert occupied.targets["temporary_area"].occupied_by == ("green_cube",)
        assert occupied.relations[
            WorldState.relation_key("green_cube", "temporary_area")
        ]

        controller.remove_object("red_cube")
        removed = world.semantic_state()
        assert not removed.objects["red_cube"].exists
        assert removed.objects["red_cube"].pose is None

        controller.restore_object("red_cube")
        restored = world.semantic_state()
        assert restored.objects["red_cube"].exists
        assert restored.objects["red_cube"].reachable
    finally:
        env.close()


def test_target_motion_distinguishes_reachable_and_unreachable() -> None:
    env = make_environment(render=False, seed=89)
    try:
        env.reset()
        world = WorldModel(env)
        controller = WorldController(env, world)

        controller.move_target("blue_target", 0.20, 0.10)
        assert world.semantic_state().targets["blue_target"].reachable

        controller.move_target("blue_target", 0.50, 0.50)
        assert not world.semantic_state().targets["blue_target"].reachable
    finally:
        env.close()


def test_external_drop_changes_holding_without_agent_simulator_access() -> None:
    env = make_environment(render=False, seed=97)
    try:
        env.reset()
        robot = PandaRobot(env)
        world = WorldModel(env)
        primitives = ManipulationPrimitives(robot)
        result = PickSkill("red_cube", world, primitives).execute()
        assert result.success
        assert world.semantic_state().robot.holding == "red_cube"

        WorldController(env, world).drop_held_object(x=-0.05, y=-0.10)

        state = world.semantic_state()
        assert state.robot.holding is None
        assert state.objects["red_cube"].exists
        assert state.objects["red_cube"].reachable
    finally:
        env.close()


def test_initial_occupied_target_is_solved_by_skill_composition() -> None:
    env = make_environment(render=False, seed=101)
    try:
        env.reset()
        robot = PandaRobot(env)
        world = WorldModel(env)
        WorldController(env, world).place_object_in_target(
            "green_cube", "blue_target"
        )
        registry = SkillRegistry.standard()
        result = AgentRuntime(
            planner=RuleBasedPlanner(),
            registry=registry,
            validator=PlanValidator(registry),
            observer=world.semantic_state,
            skill_factory=SemanticSkillFactory(
                world, ManipulationPrimitives(robot)
            ),
        ).run(
            Goal.put_inside(
                "Put the red cube inside the blue target.",
                "red_cube",
                "blue_target",
            )
        )

        assert result.success
        assert result.planner_calls == 1
        assert result.replans == 0
        assert [entry.step.to_dict() for entry in result.executed_steps] == [
            {"skill": "pick", "args": {"object": "green_cube"}},
            {
                "skill": "place",
                "args": {
                    "object": "green_cube",
                    "target": "temporary_area",
                },
            },
            {"skill": "pick", "args": {"object": "red_cube"}},
            {
                "skill": "place",
                "args": {"object": "red_cube", "target": "blue_target"},
            },
        ]
        assert result.final_world_state.targets["blue_target"].occupied_by == (
            "red_cube",
        )
        assert result.final_world_state.targets["temporary_area"].occupied_by == (
            "green_cube",
        )
    finally:
        env.close()


def test_dynamic_occupancy_after_pick_replans_to_valid_composition() -> None:
    env = make_environment(render=False, seed=107)
    try:
        env.reset()
        robot = PandaRobot(env)
        world = WorldModel(env)
        controller = WorldController(env, world)
        registry = SkillRegistry.standard()
        changed = False

        def occupy_before_place(event: AgentBoundaryEvent) -> None:
            nonlocal changed
            if not changed and event.step.skill == "place":
                controller.place_object_in_target("green_cube", "blue_target")
                changed = True

        result = AgentRuntime(
            planner=RuleBasedPlanner(),
            registry=registry,
            validator=PlanValidator(registry),
            observer=world.semantic_state,
            skill_factory=SemanticSkillFactory(
                world, ManipulationPrimitives(robot)
            ),
        ).run(
            Goal.put_inside(
                "Put the red cube inside the blue target.",
                "red_cube",
                "blue_target",
            ),
            before_step=occupy_before_place,
        )

        assert changed
        assert result.success
        assert result.planner_calls == 2
        assert result.replans == 1
        assert any("TARGET_OCCUPIED" in line for line in result.trace)
        assert [step.to_dict() for step in result.plan_history[1].steps] == [
            {
                "skill": "place",
                "args": {"object": "red_cube", "target": "temporary_area"},
            },
            {"skill": "pick", "args": {"object": "green_cube"}},
            {
                "skill": "place",
                "args": {"object": "green_cube", "target": "red_target"},
            },
            {"skill": "pick", "args": {"object": "red_cube"}},
            {
                "skill": "place",
                "args": {"object": "red_cube", "target": "blue_target"},
            },
        ]
        assert [step.step.skill for step in result.executed_steps] == [
            "pick",
            "place",
            "pick",
            "place",
            "pick",
            "place",
        ]
        assert result.final_world_state.targets["blue_target"].occupied_by == (
            "red_cube",
        )
        assert result.final_world_state.targets["red_target"].occupied_by == (
            "green_cube",
        )
    finally:
        env.close()


def test_temporary_area_accepts_normal_pick_and_place() -> None:
    env = make_environment(render=False, seed=103)
    try:
        env.reset()
        robot = PandaRobot(env)
        world = WorldModel(env)
        registry = SkillRegistry.standard()
        result = AgentRuntime(
            planner=RuleBasedPlanner(),
            registry=registry,
            validator=PlanValidator(registry),
            observer=world.semantic_state,
            skill_factory=SemanticSkillFactory(
                world, ManipulationPrimitives(robot)
            ),
        ).run(
            Goal.put_inside(
                "Put the green cube inside the temporary area.",
                "green_cube",
                "temporary_area",
            )
        )

        assert result.success
        assert result.planner_calls == 1
        assert result.replans == 0
        assert result.final_world_state.relations[
            WorldState.relation_key("green_cube", "temporary_area")
        ]
    finally:
        env.close()
