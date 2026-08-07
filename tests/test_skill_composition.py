from __future__ import annotations

import inspect
from dataclasses import replace

from planner import (
    Goal,
    OpenAICompatiblePlanner,
    PlanValidator,
    RuleBasedPlanner,
    SkillRegistry,
)
from runtime import AgentRuntime
from world import ObjectState, PoseState, RobotState, TargetState, WorldState


def generic_occupied_world() -> WorldState:
    pose = PoseState((0.0, 0.0, 0.825), (0.0, 0.0, 0.0, 1.0))
    return WorldState(
        robot=RobotState(None),
        objects={
            name: ObjectState(True, pose, False, True)
            for name in ("payload", "blocker", "spare")
        },
        targets={
            "destination": TargetState(
                True,
                pose,
                True,
                True,
                ("blocker",),
            ),
            "staging": TargetState(
                True,
                pose,
                True,
                False,
                (),
                "temporary",
            ),
            "other_target": TargetState(True, pose, True, False),
        },
        relations={
            "payload_inside_destination": False,
            "blocker_inside_destination": True,
            "spare_inside_destination": False,
            "payload_inside_staging": False,
            "blocker_inside_staging": False,
            "spare_inside_staging": False,
            "payload_inside_other_target": False,
            "blocker_inside_other_target": False,
            "spare_inside_other_target": False,
        },
    )


def test_occupied_by_is_serialized_and_flattened() -> None:
    state = generic_occupied_world()

    assert state.to_dict()["targets"]["destination"]["occupied_by"] == [
        "blocker"
    ]
    assert state.flattened()["targets.destination.occupied_by"] == ("blocker",)


def test_generic_occupied_target_composes_registered_pick_and_place() -> None:
    state = generic_occupied_world()
    registry = SkillRegistry.standard()
    goal = Goal.put_inside("Move payload to destination.", "payload", "destination")

    result = RuleBasedPlanner().plan(goal, state, registry)

    assert result.plan is not None
    assert result.plan.missing_capabilities == ()
    assert [step.to_dict() for step in result.plan.steps] == [
        {"skill": "pick", "args": {"object": "blocker"}},
        {
            "skill": "place",
            "args": {"object": "blocker", "target": "staging"},
        },
        {"skill": "pick", "args": {"object": "payload"}},
        {
            "skill": "place",
            "args": {"object": "payload", "target": "destination"},
        },
    ]
    validation = PlanValidator(registry).validate(result, state)
    assert validation.valid
    assert validation.errors == ()


def test_llm_prompt_exposes_occupancy_and_skill_composition_example() -> None:
    state = generic_occupied_world()
    payload = OpenAICompatiblePlanner._prompt_payload(
        Goal.put_inside("Move payload to destination.", "payload", "destination"),
        state,
        SkillRegistry.standard(),
        None,
        None,
    )

    assert payload["current_world_state"]["targets"]["destination"][
        "occupied_by"
    ] == ["blocker"]
    assert len(payload["required_output"]["ready"]["steps"]) == 4


def test_known_blocker_without_free_target_is_clean_state_limit() -> None:
    state = generic_occupied_world()
    state = replace(
        state,
        targets={"destination": state.targets["destination"]},
    )

    result = RuleBasedPlanner().plan(
        Goal.put_inside("Move payload to destination.", "payload", "destination"),
        state,
        SkillRegistry.standard(),
    )

    assert result.plan is not None
    assert result.plan.reason != "CAPABILITY_GAP"
    assert result.plan.missing_capabilities == ()
    assert "No legal Pick / Place composition" in (result.plan.reason or "")


def test_agent_runtime_contains_no_occupancy_recovery_policy() -> None:
    source = inspect.getsource(AgentRuntime)

    assert "TARGET_OCCUPIED" not in source
    assert "occupied_by" not in source
    assert "temporary_area" not in source
