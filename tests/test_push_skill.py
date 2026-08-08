from __future__ import annotations

import numpy as np

from primitives import PrimitiveResult, WorkspaceBounds
from robot import Pose
from skills import (
    AxisAlignedCubePushPoseGenerator,
    PushSkill,
    SkillFailure,
    SkillPhase,
    evaluate_push_outcome,
)


SUCCESS = PrimitiveResult(True, None, 1, 0.0, 0.0)


class FakeWorld:
    def __init__(
        self,
        *,
        push_displacements: list[float] | None = None,
        holding: str | None = None,
        path_safe: bool = True,
    ) -> None:
        self.object_names = ("red_cube",)
        self.push_region_names = ("right_side",)
        self.object_pose = Pose.from_values(
            [0.0, -0.05, 0.825],
            [0.0, 0.0, 0.0, 1.0],
        )
        self.push_displacements = list(push_displacements or [0.21])
        self.holding = holding
        self.path_safe = path_safe

    def exists(self, name: str) -> bool:
        return name in self.object_names

    def is_reachable(self, name: str) -> bool:
        return self.exists(name)

    def pose(self, name: str) -> Pose:
        if not self.exists(name):
            raise KeyError(name)
        return self.object_pose

    def holding_object(self) -> str | None:
        return self.holding

    def push_region_bounds(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name == "right_side":
            return np.array([0.18, -0.25]), np.array([0.30, 0.25])
        raise KeyError(name)

    def is_push_path_safe(self, start, end, *, margin=None) -> bool:
        return self.path_safe

    def is_inside_push_region(self, object_name: str, region_name: str) -> bool:
        lower, upper = self.push_region_bounds(region_name)
        xy = self.object_pose.position[:2]
        return bool(np.all(xy >= lower) and np.all(xy <= upper))

    def is_on_table(self, object_name: str, *, height_tolerance: float) -> bool:
        return abs(float(self.object_pose.position[2]) - 0.825) <= height_tolerance

    def apply_next_push(self) -> None:
        displacement = self.push_displacements.pop(0)
        self.object_pose = Pose(
            self.object_pose.position + np.array([displacement, 0.0, 0.0]),
            self.object_pose.quaternion,
        )


class FakePrimitives:
    def __init__(self, world: FakeWorld, *, upper_x: float = 0.35) -> None:
        self.world = world
        self.workspace = WorkspaceBounds(
            lower=np.array([-0.35, -0.35, 0.805]),
            upper=np.array([upper_x, 0.35, 1.25]),
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

    def move_to_pose(self, target: Pose) -> PrimitiveResult:
        self.calls.append("move_to_pose")
        self._pose = target
        return SUCCESS

    def move_linear(
        self,
        target: Pose,
        *,
        max_cartesian_step: float,
        max_steps: int = 300,
    ) -> PrimitiveResult:
        self.calls.append("move_linear")
        horizontal_distance = float(
            np.linalg.norm(target.position[:2] - self._pose.position[:2])
        )
        self._pose = target
        if horizontal_distance > 0.10:
            self.world.apply_next_push()
        return SUCCESS

    def wait(self, *, steps: int) -> PrimitiveResult:
        self.calls.append("wait")
        return SUCCESS


def test_push_geometry_approaches_opposite_requested_direction() -> None:
    generator = AxisAlignedCubePushPoseGenerator()
    object_pose = Pose.from_values([0.0, -0.05, 0.825], [0.0, 0.0, 0.0, 1.0])
    orientation = np.array([0.7, 0.7, 0.0, 0.0])

    right = generator.generate(
        "red_cube",
        "right_side",
        object_pose,
        np.array([0.18, -0.25]),
        np.array([0.30, 0.25]),
        orientation,
    )
    assert right is not None
    assert right.direction[0] > 0.99
    assert right.contact_pose.position[0] < object_pose.position[0]
    assert right.end_pose.position[0] > right.contact_pose.position[0]


def test_invalid_target_and_nonempty_gripper_fail_before_motion() -> None:
    invalid_world = FakeWorld()
    invalid_primitives = FakePrimitives(invalid_world)
    invalid = PushSkill(
        "red_cube", "not_a_region", invalid_world, invalid_primitives
    ).execute()

    held_world = FakeWorld(holding="green_cube")
    held_primitives = FakePrimitives(held_world)
    held = PushSkill(
        "red_cube", "right_side", held_world, held_primitives
    ).execute()

    assert invalid.reason is SkillFailure.INVALID_PUSH_TARGET
    assert invalid.attempts == 0
    assert invalid_primitives.calls == []
    assert held.reason is SkillFailure.GRIPPER_NOT_EMPTY
    assert held.attempts == 0
    assert held_primitives.calls == []


def test_workspace_safety_rejects_push_before_motion() -> None:
    world = FakeWorld()
    primitives = FakePrimitives(world, upper_x=0.10)

    result = PushSkill("red_cube", "right_side", world, primitives).execute()

    assert not result.success
    assert result.phase is SkillPhase.GENERATE_PUSH
    assert result.reason is SkillFailure.PUSH_PATH_OUTSIDE_WORKSPACE
    assert primitives.calls == []


def test_push_outcome_requires_displacement_target_and_table() -> None:
    start = np.array([0.0, 0.0, 0.825])

    assert evaluate_push_outcome(
        start,
        np.array([0.02, 0.0, 0.825]),
        minimum_displacement=0.06,
        target_reached=True,
        object_on_table=True,
    ) is SkillFailure.INSUFFICIENT_DISPLACEMENT
    assert evaluate_push_outcome(
        start,
        np.array([0.10, 0.0, 0.825]),
        minimum_displacement=0.06,
        target_reached=False,
        object_on_table=True,
    ) is SkillFailure.PUSH_TARGET_NOT_REACHED
    assert evaluate_push_outcome(
        start,
        np.array([0.20, 0.0, 0.60]),
        minimum_displacement=0.06,
        target_reached=True,
        object_on_table=False,
    ) is SkillFailure.OBJECT_LEFT_WORKSPACE


def test_insufficient_displacement_uses_one_local_retry_then_succeeds() -> None:
    world = FakeWorld(push_displacements=[0.0, 0.21])
    primitives = FakePrimitives(world)

    result = PushSkill("red_cube", "right_side", world, primitives).execute()

    assert result.success
    assert result.attempts == 2
    assert "ATTEMPT_FAILED: INSUFFICIENT_DISPLACEMENT" in result.trace
    assert SkillPhase.RECOVERY.value in result.trace
    assert SkillPhase.RECOVERY_REFRESH_PUSH_STATE.value in result.trace
    assert SkillPhase.RECOVERY_RECOMPUTE_PUSH.value in result.trace


def test_failed_push_returns_structured_semantic_reason() -> None:
    world = FakeWorld(push_displacements=[0.0, 0.0])
    primitives = FakePrimitives(world)

    result = PushSkill("red_cube", "right_side", world, primitives).execute()

    assert not result.success
    assert result.reason is SkillFailure.INSUFFICIENT_DISPLACEMENT
    assert result.phase is SkillPhase.VERIFY_SUCCESS
    assert result.attempts == 2
