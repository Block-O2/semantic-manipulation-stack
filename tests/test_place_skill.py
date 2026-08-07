from __future__ import annotations

import numpy as np

from primitives import PrimitiveResult, WorkspaceBounds
from robot import Pose
from skills import PlaceSkill, SkillFailure, SkillPhase


SUCCESS = PrimitiveResult(True, None, 1, 0.0, 0.0)


class FakeWorld:
    def __init__(
        self,
        *,
        object_names: tuple[str, ...] = ("red_cube", "blue_target"),
        grasp_states: list[bool] | None = None,
        inside_target: bool = True,
        stable: bool = True,
    ) -> None:
        self.object_names = object_names
        self._poses = {
            "red_cube": Pose.from_values([0.0, -0.05, 0.98], [0.0, 0.0, 0.0, 1.0]),
            "blue_target": Pose.from_values([0.12, 0.14, 0.803], [0.0, 0.0, 0.0, 1.0]),
        }
        self.grasp_states = list(grasp_states or [True])
        self.inside_target = inside_target
        self.stable = stable

    def pose(self, name: str) -> Pose:
        if name not in self.object_names:
            raise KeyError(name)
        return self._poses[name]

    def is_grasped(self, name: str) -> bool:
        if name not in self.object_names:
            raise KeyError(name)
        if len(self.grasp_states) > 1:
            return self.grasp_states.pop(0)
        return self.grasp_states[0]

    def target_half_extents(self, name: str) -> np.ndarray:
        if name != "blue_target" or name not in self.object_names:
            raise KeyError(name)
        return np.array([0.09, 0.09, 0.003])

    def is_inside_target(self, object_name: str, target_name: str, *, margin: float) -> bool:
        return self.inside_target

    def is_stable(self, name: str, *, maximum_speed: float) -> bool:
        return self.stable


class FakePrimitives:
    def __init__(self, *, upper_z: float = 1.25) -> None:
        self.workspace = WorkspaceBounds(
            lower=np.array([-0.35, -0.35, 0.805]),
            upper=np.array([0.35, 0.35, upper_z]),
        )
        self._pose = Pose.from_values(
            [0.0, -0.05, 0.98],
            [0.7055, 0.7058, 0.0449, 0.0450],
        )
        self.calls: list[str] = []

    @property
    def current_pose(self) -> Pose:
        return self._pose

    def move_to_pose(self, target: Pose) -> PrimitiveResult:
        self.calls.append("move_to_pose")
        self._pose = target
        return SUCCESS

    def move_linear(self, target: Pose, *, max_cartesian_step: float) -> PrimitiveResult:
        self.calls.append("move_linear")
        self._pose = target
        return SUCCESS

    def open_gripper(self, *, steps: int) -> PrimitiveResult:
        self.calls.append("open_gripper")
        return SUCCESS

    def wait(self, *, steps: int) -> PrimitiveResult:
        self.calls.append("wait")
        return SUCCESS


def test_object_lost_before_place_fails_without_robot_motion() -> None:
    world = FakeWorld(grasp_states=[False])
    primitives = FakePrimitives()

    result = PlaceSkill("red_cube", "blue_target", world, primitives).execute()

    assert not result.success
    assert result.phase is SkillPhase.CHECK_PRECONDITIONS
    assert result.reason is SkillFailure.OBJECT_NOT_GRASPED
    assert result.attempts == 0
    assert primitives.calls == []


def test_place_succeeds_with_release_target_membership_and_stability() -> None:
    world = FakeWorld(grasp_states=[True, False, False])
    primitives = FakePrimitives()

    result = PlaceSkill("red_cube", "blue_target", world, primitives).execute()

    assert result.success
    assert result.phase is SkillPhase.SUCCESS
    assert result.attempts == 1
    assert result.reason is None
    assert primitives.calls == [
        "move_to_pose",
        "move_linear",
        "open_gripper",
        "wait",
        "move_linear",
    ]
    assert result.trace[-1] == SkillPhase.SUCCESS.value


def test_unreachable_place_candidate_fails_before_motion() -> None:
    world = FakeWorld(grasp_states=[True])
    primitives = FakePrimitives(upper_z=0.90)

    result = PlaceSkill("red_cube", "blue_target", world, primitives).execute()

    assert not result.success
    assert result.phase is SkillPhase.GENERATE_PLACE_POSE
    assert result.reason is SkillFailure.TARGET_UNREACHABLE
    assert primitives.calls == []


def test_object_outside_target_is_a_structural_place_failure() -> None:
    world = FakeWorld(grasp_states=[True, False, False], inside_target=False)
    primitives = FakePrimitives()

    result = PlaceSkill("red_cube", "blue_target", world, primitives).execute()

    assert not result.success
    assert result.phase is SkillPhase.VERIFY_SUCCESS
    assert result.reason is SkillFailure.OBJECT_OUTSIDE_TARGET
    assert SkillPhase.RECOVERY.value not in result.trace


def test_still_grasped_release_uses_one_local_retry() -> None:
    world = FakeWorld(grasp_states=[True, True, True, False])
    primitives = FakePrimitives()

    result = PlaceSkill("red_cube", "blue_target", world, primitives).execute()

    assert result.success
    assert result.attempts == 2
    assert "ATTEMPT_FAILED: OBJECT_STILL_GRASPED" in result.trace
    assert SkillPhase.RECOVERY.value in result.trace
    assert SkillPhase.RECOVERY_RETREAT_SLIGHTLY.value in result.trace
    assert SkillPhase.RECOVERY_REFRESH_WORLD_STATE.value in result.trace
    assert SkillPhase.RECOVERY_RECOMPUTE_PLACE.value in result.trace
    assert primitives.calls.count("open_gripper") == 2
