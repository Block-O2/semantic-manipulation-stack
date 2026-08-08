from __future__ import annotations

import inspect

from planner import (
    Goal,
    GoalRelation,
    Plan,
    PlannerResult,
    PlanStep,
    PlanValidator,
    RuleBasedPlanner,
    SkillRegistry,
    derive_expected_effects,
)
from runtime import AgentRuntime, SkillExecutor
from world import (
    ObjectState,
    PoseState,
    PushRegionState,
    RobotState,
    WorldState,
)


def push_world(*, holding: str | None = None) -> WorldState:
    pose = PoseState((0.05, -0.10, 0.825), (0.0, 0.0, 0.0, 1.0))
    return WorldState(
        robot=RobotState(holding),
        objects={
            "test_object": ObjectState(
                True,
                pose,
                holding == "test_object",
                True,
            )
        },
        targets={},
        relations={"test_object_inside_right_side": False},
        push_regions={
            "right_side": PushRegionState(
                True,
                True,
                (0.24, 0.0, 0.825),
                (0.18, -0.25),
                (0.30, 0.25),
            )
        },
    )


def test_push_goal_produces_valid_registered_plan() -> None:
    state = push_world()
    registry = SkillRegistry.standard()
    goal = Goal.push_to_region(
        "Push the object toward the right side.",
        "test_object",
        "right_side",
    )

    result = RuleBasedPlanner().plan(goal, state, registry)

    assert goal.relation is GoalRelation.PUSH_TO_REGION
    assert result.plan is not None
    assert result.plan.missing_capabilities == ()
    assert [step.to_dict() for step in result.plan.steps] == [
        {
            "skill": "push",
            "args": {"object": "test_object", "target": "right_side"},
        }
    ]
    assert PlanValidator(registry).validate(result, state).valid


def test_push_to_edge_shorthand_targets_supported_right_side() -> None:
    goal = Goal.push_to_edge("Push the object to the edge.", "test_object")

    result = RuleBasedPlanner().plan(
        goal,
        push_world(),
        SkillRegistry.standard(),
    )

    assert goal.relation is GoalRelation.PUSH_TO_EDGE
    assert goal.target_name == "right_side"
    assert result.plan is not None
    assert result.plan.steps == (
        PlanStep(
            "push",
            {"object": "test_object", "target": "right_side"},
        ),
    )


def test_push_validation_rejects_invalid_region_and_nonempty_gripper() -> None:
    registry = SkillRegistry.standard()
    invalid_target = PlannerResult.from_plan(
        Plan.ready(
            [
                PlanStep(
                    "push",
                    {"object": "test_object", "target": "not_a_region"},
                )
            ]
        )
    )
    holding_plan = PlannerResult.from_plan(
        Plan.ready(
            [
                PlanStep(
                    "push",
                    {"object": "test_object", "target": "right_side"},
                )
            ]
        )
    )

    invalid = PlanValidator(registry).validate(invalid_target, push_world())
    holding = PlanValidator(registry).validate(
        holding_plan,
        push_world(holding="test_object"),
    )

    assert not invalid.valid
    assert any("UNKNOWN_PUSH_REGION" in error for error in invalid.errors)
    assert not holding.valid
    assert any("GRIPPER_NOT_EMPTY" in error for error in holding.errors)


def test_push_expected_effect_is_semantic_region_membership() -> None:
    expected = derive_expected_effects(
        SkillRegistry.standard(),
        PlanStep(
            "push",
            {"object": "test_object", "target": "right_side"},
        ),
    )

    assert len(expected.effects) == 1
    assert expected.effects[0].path == "relations.test_object_inside_right_side"
    assert expected.effects[0].expected is True
    assert expected.effects[0].mismatch_code == "PUSH_TARGET_NOT_REACHED"


def test_genuinely_unsupported_open_drawer_remains_capability_gap() -> None:
    result = RuleBasedPlanner().plan(
        Goal.open_drawer("Open the drawer."),
        push_world(),
        SkillRegistry.standard(),
    )

    assert result.plan is not None
    assert result.plan.reason == "CAPABILITY_GAP"
    assert result.plan.missing_capabilities == ("open_drawer",)


def test_runtime_and_executor_contain_no_push_execution_policy() -> None:
    runtime_source = inspect.getsource(AgentRuntime)
    executor_source = inspect.getsource(SkillExecutor)

    assert "PushSkill" not in runtime_source
    assert "PUSH_LINEAR" not in runtime_source
    assert "push" not in executor_source.lower()
