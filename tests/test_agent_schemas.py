from __future__ import annotations

import json

import pytest

from planner import (
    Goal,
    Plan,
    PlannerResult,
    PlanStep,
    PlanValidator,
    SkillRegistry,
    compare_expected_effects,
    derive_expected_effects,
)
from world import ObjectState, PoseState, RobotState, TargetState, WorldState


def world_state(*, holding: str | None = None, inside: bool = False) -> WorldState:
    pose = PoseState((0.0, 0.0, 0.825), (0.0, 0.0, 0.0, 1.0))
    return WorldState(
        robot=RobotState(holding),
        objects={
            "red_cube": ObjectState(True, pose, holding == "red_cube", True),
        },
        targets={
            "blue_target": TargetState(True, pose, True, inside),
        },
        relations={"red_cube_inside_blue_target": inside},
    )


def test_world_state_is_json_serializable() -> None:
    state = world_state()

    encoded = json.dumps(state.to_dict(), sort_keys=True)

    assert '"holding": null' in encoded
    assert state.value("objects.red_cube.reachable") is True
    assert state.value("relations.red_cube_inside_blue_target") is False


def test_skill_registry_contains_machine_readable_pick_and_place_contracts() -> None:
    registry = SkillRegistry.standard()

    payload = registry.to_dict()

    assert registry.names == ("pick", "place")
    assert payload["pick"]["arguments"][0]["kind"] == "object"
    assert payload["pick"]["expected_effects"][0]["path"] == "robot.holding"
    assert any(
        effect["mismatch_code"] == "PLACE_EFFECT_NOT_ACHIEVED"
        for effect in payload["place"]["expected_effects"]
    )


def test_plan_parsing_is_strict_and_supports_cannot_plan() -> None:
    plan = Plan.from_dict(
        {
            "steps": [
                {"skill": "pick", "args": {"object": "red_cube"}},
                {
                    "skill": "place",
                    "args": {"object": "red_cube", "target": "blue_target"},
                },
            ]
        }
    )
    cannot = Plan.from_dict(
        {"status": "cannot_plan", "reason": "Target is unreachable."}
    )

    assert len(plan.steps) == 2
    assert cannot.reason == "Target is unreachable."
    with pytest.raises(ValueError, match="unknown plan fields"):
        Plan.from_dict({"steps": [], "python": "move_robot()"})


def test_capability_gap_round_trips_without_inventing_a_skill() -> None:
    plan = Plan.capability_gap(["push"], "Push is not registered.")

    parsed = Plan.from_dict(plan.to_dict())

    assert parsed.reason == "CAPABILITY_GAP"
    assert parsed.missing_capabilities == ("push",)
    assert parsed.steps == ()


def test_validator_accepts_projected_pick_place_preconditions() -> None:
    registry = SkillRegistry.standard()
    validator = PlanValidator(registry)
    plan = Plan.ready(
        [
            PlanStep("pick", {"object": "red_cube"}),
            PlanStep(
                "place", {"object": "red_cube", "target": "blue_target"}
            ),
        ]
    )

    result = validator.validate(PlannerResult.from_plan(plan), world_state())

    assert result.valid
    assert result.errors == ()


@pytest.mark.parametrize(
    "payload, expected_error",
    [
        (
            {"steps": [{"skill": "move_joints", "args": {"joints": "all"}}]},
            "UNKNOWN_SKILL",
        ),
        (
            {
                "steps": [
                    {
                        "skill": "pick",
                        "args": {"object": "red_cube", "controller": "OSC_POSE"},
                    }
                ]
            },
            "UNKNOWN_ARGS",
        ),
        (
            {"steps": [{"skill": "pick", "args": {"object": "green_cube"}}]},
            "UNKNOWN_OBJECT",
        ),
    ],
)
def test_validator_rejects_hallucinated_or_unsafe_plans(
    payload: dict[str, object], expected_error: str
) -> None:
    result = PlanValidator(SkillRegistry.standard()).validate(
        PlannerResult(raw_response=payload),
        world_state(),
    )

    assert not result.valid
    assert any(expected_error in error for error in result.errors)


def test_expected_effects_produce_explicit_semantic_residual() -> None:
    registry = SkillRegistry.standard()
    step = PlanStep("pick", {"object": "red_cube"})
    expected = derive_expected_effects(registry, step)

    residual = compare_expected_effects(expected, world_state())

    assert not residual.consistent
    assert {item.code for item in residual.mismatches} == {
        "OBJECT_LOST",
        "GRASP_EFFECT_NOT_ACHIEVED",
    }
