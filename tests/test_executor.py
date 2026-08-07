from __future__ import annotations

from skills import Skill, SkillFailure, SkillPhase, SkillResult
from runtime import SkillExecutor


def _result(*, skill: str, success: bool, reason: SkillFailure | None = None) -> SkillResult:
    return SkillResult(
        success=success,
        skill=skill,
        object_name="red_cube",
        phase=SkillPhase.SUCCESS if success else SkillPhase.CHECK_PRECONDITIONS,
        reason=reason,
        attempts=1,
        primitive_result=None,
        trace=(),
    )


class FakeSkill(Skill):
    def __init__(self, result: SkillResult) -> None:
        self.result = result
        self.executions = 0

    def check_preconditions(self) -> SkillResult | None:
        return None

    def execute(self) -> SkillResult:
        self.executions += 1
        return self.result


def test_pick_failure_stops_task_before_place() -> None:
    pick = FakeSkill(
        _result(
            skill="PickSkill",
            success=False,
            reason=SkillFailure.NO_CONTACT_GRASP,
        )
    )
    place = FakeSkill(_result(skill="PlaceSkill", success=True))

    result = SkillExecutor().execute([pick, place])

    assert not result.success
    assert result.completed_steps == 0
    assert result.failed_step == 0
    assert result.failed_skill == "PickSkill"
    assert result.reason is SkillFailure.NO_CONTACT_GRASP
    assert pick.executions == 1
    assert place.executions == 0
    assert len(result.skill_results) == 1


def test_successful_pick_then_place_completes_task() -> None:
    pick = FakeSkill(_result(skill="PickSkill", success=True))
    place = FakeSkill(_result(skill="PlaceSkill", success=True))

    result = SkillExecutor().execute([pick, place])

    assert result.success
    assert result.completed_steps == 2
    assert result.failed_step is None
    assert result.failed_skill is None
    assert result.reason is None
    assert pick.executions == 1
    assert place.executions == 1
    assert len(result.skill_results) == 2


def test_place_failure_is_propagated_as_task_failure() -> None:
    pick = FakeSkill(_result(skill="PickSkill", success=True))
    place = FakeSkill(
        _result(
            skill="PlaceSkill",
            success=False,
            reason=SkillFailure.OBJECT_OUTSIDE_TARGET,
        )
    )

    result = SkillExecutor().execute([pick, place])

    assert not result.success
    assert result.completed_steps == 1
    assert result.failed_step == 1
    assert result.failed_skill == "PlaceSkill"
    assert result.reason is SkillFailure.OBJECT_OUTSIDE_TARGET
    assert len(result.skill_results) == 2
    assert place.executions == 1
