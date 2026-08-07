from __future__ import annotations

import numpy as np

from primitives import PrimitiveResult, WorkspaceBounds
from robot import Pose
from skills import PickSkill, SkillFailure, SkillPhase


SUCCESS = PrimitiveResult(True, None, 1, 0.0, 0.0)


class FakeWorld:
    def __init__(
        self,
        *,
        object_names: tuple[str, ...] = ("red_cube",),
        grasp_states: list[bool] | None = None,
    ) -> None:
        self.object_names = object_names
        self.object_pose = Pose.from_values(
            [0.0, -0.05, 0.825],
            [0.0, 0.0, 0.0, 1.0],
        )
        self.grasp_states = list(grasp_states or [False])

    def pose(self, name: str) -> Pose:
        if name not in self.object_names:
            raise KeyError(name)
        return self.object_pose

    def is_grasped(self, name: str) -> bool:
        if name not in self.object_names:
            raise KeyError(name)
        if len(self.grasp_states) > 1:
            return self.grasp_states.pop(0)
        return self.grasp_states[0]

    def lift_object(self, height: float) -> None:
        self.object_pose = Pose(
            self.object_pose.position + np.array([0.0, 0.0, height]),
            self.object_pose.quaternion,
        )


class FakePrimitives:
    def __init__(self, world: FakeWorld, *, upper_z: float = 1.25) -> None:
        self.world = world
        self.workspace = WorkspaceBounds(
            lower=np.array([-0.35, -0.35, 0.805]),
            upper=np.array([0.35, 0.35, upper_z]),
        )
        self._pose = Pose.from_values(
            [-0.10, 0.0, 1.01],
            [0.7055, 0.7058, 0.0449, 0.0450],
        )
        self.calls: list[str] = []

    @property
    def current_pose(self) -> Pose:
        return self._pose

    def open_gripper(self, *, steps: int) -> PrimitiveResult:
        self.calls.append("open_gripper")
        return SUCCESS

    def close_gripper(self, *, steps: int) -> PrimitiveResult:
        self.calls.append("close_gripper")
        return SUCCESS

    def move_to_pose(self, target: Pose) -> PrimitiveResult:
        self.calls.append("move_to_pose")
        self._pose = target
        return SUCCESS

    def move_linear(self, target: Pose, *, max_cartesian_step: float) -> PrimitiveResult:
        self.calls.append("move_linear")
        previous_z = self._pose.position[2]
        self._pose = target
        if target.position[2] > previous_z + 0.05:
            self.world.lift_object(target.position[2] - previous_z)
        return SUCCESS


def test_missing_object_fails_without_recovery() -> None:
    world = FakeWorld(object_names=())
    primitives = FakePrimitives(world)

    result = PickSkill("nonexistent_object", world, primitives).execute()

    assert not result.success
    assert result.phase is SkillPhase.CHECK_PRECONDITIONS
    assert result.reason is SkillFailure.OBJECT_NOT_FOUND
    assert result.attempts == 0
    assert primitives.calls == []
    assert SkillPhase.RECOVERY.value not in result.trace


def test_unreachable_grasp_fails_once_without_recovery() -> None:
    world = FakeWorld(grasp_states=[False])
    primitives = FakePrimitives(world, upper_z=0.90)

    result = PickSkill("red_cube", world, primitives).execute()

    assert not result.success
    assert result.phase is SkillPhase.GENERATE_GRASP
    assert result.reason is SkillFailure.TARGET_UNREACHABLE
    assert result.attempts == 1
    assert primitives.calls == []
    assert SkillPhase.RECOVERY.value not in result.trace


def test_no_contact_grasp_enters_recovery_and_succeeds_on_retry() -> None:
    world = FakeWorld(grasp_states=[False, False, True, True])
    primitives = FakePrimitives(world)

    result = PickSkill("red_cube", world, primitives).execute()

    assert result.success
    assert result.phase is SkillPhase.SUCCESS
    assert result.reason is None
    assert result.attempts == 2
    assert "ATTEMPT_FAILED: NO_CONTACT_GRASP" in result.trace
    assert SkillPhase.RECOVERY.value in result.trace
    assert SkillPhase.RECOVERY_OPEN_GRIPPER.value in result.trace
    assert SkillPhase.RECOVERY_RETREAT.value in result.trace
    assert SkillPhase.RECOVERY_REFRESH_OBJECT_STATE.value in result.trace
    assert SkillPhase.RECOVERY_RECOMPUTE_GRASP.value in result.trace
    assert primitives.calls.count("open_gripper") == 3
