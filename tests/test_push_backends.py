from __future__ import annotations

import inspect

import numpy as np

from learning import NormalizationStats, OneStepBCPolicy
from primitives import PrimitiveResult, WorkspaceBounds
from robot import Pose
from runtime import AgentRuntime, SemanticSkillFactory, create_push_backend
from skills import (
    ACTPushBackend,
    BCPushBackend,
    ChunkBCPushBackend,
    ClassicalPushBackend,
    PushBackend,
    PushExecutionContext,
    PushRequest,
    SkillFailure,
)


class FakeWorld:
    object_names = ("red_cube",)
    push_region_names = ("right_side",)

    def __init__(self) -> None:
        self.cube = Pose.from_values([0.0, -0.1, 0.825], [0.0, 0.0, 0.0, 1.0])

    def pose(self, name: str) -> Pose:
        return self.cube

    def push_region_bounds(self, name: str):
        return np.array([0.18, -0.25]), np.array([0.34, 0.25])

    def is_inside_push_region(self, object_name: str, target_name: str) -> bool:
        return False

    def is_on_table(self, object_name: str) -> bool:
        return True


class FakePrimitives:
    def __init__(self) -> None:
        self._pose = Pose.from_values([-0.25, -0.25, 1.1], [0.0, 0.0, 0.0, 1.0])
        self.workspace = WorkspaceBounds(
            np.array([-0.35, -0.35, 0.805]),
            np.array([0.35, 0.35, 1.25]),
        )
        self.commands: list[Pose] = []

    @property
    def current_pose(self) -> Pose:
        return self._pose

    def open_gripper(self, *, steps: int) -> PrimitiveResult:
        return PrimitiveResult(True, None, steps, None, None)

    def command_cartesian_once(self, target: Pose, *, max_cartesian_step: float):
        self.commands.append(target)
        self._pose = target
        return PrimitiveResult(True, None, 1, 0.0, 0.0)


class ConstantPolicy:
    def __init__(self, action) -> None:
        self.action = np.asarray(action, dtype=np.float64)

    def predict(self, observation):
        return self.action.copy()


def test_push_backend_protocol_and_factory_selection(tmp_path) -> None:
    classical = create_push_backend("classical")
    assert isinstance(classical, PushBackend)
    assert isinstance(classical, ClassicalPushBackend)

    normalization = NormalizationStats(
        np.zeros(10), np.ones(10), np.zeros(3), np.ones(3)
    )
    policy = OneStepBCPolicy(normalization=normalization)
    checkpoint = tmp_path / "policy.npz"
    policy.save(checkpoint, metadata={"test": True})
    assert isinstance(create_push_backend("bc", checkpoint=str(checkpoint)), BCPushBackend)
    with np.testing.assert_raises(ValueError):
        create_push_backend(
            "bc", checkpoint=str(checkpoint), execution_horizon=2
        )


def test_semantic_skill_factory_defaults_to_classical_backend() -> None:
    factory = SemanticSkillFactory(FakeWorld(), FakePrimitives())
    skill = factory.create(
        type("Step", (), {"skill": "push", "args": {
            "object": "red_cube", "target": "right_side"
        }})()
    )

    assert skill.backend_name == "classical"

    bc_factory = SemanticSkillFactory(
        FakeWorld(),
        FakePrimitives(),
        push_backend=BCPushBackend(ConstantPolicy([-0.25, -0.25, 1.1])),
    )
    bc_skill = bc_factory.create(
        type("Step", (), {"skill": "push", "args": {
            "object": "red_cube", "target": "right_side"
        }})()
    )
    assert bc_skill.backend_name == "bc"


def test_bc_backend_rejects_nonfinite_and_extreme_actions() -> None:
    context = PushExecutionContext(FakeWorld(), FakePrimitives())
    request = PushRequest("red_cube", "right_side")

    nonfinite = BCPushBackend(ConstantPolicy([np.nan, 0.0, 1.0])).execute(
        request, context
    )
    unsafe = BCPushBackend(ConstantPolicy([0.35, 0.35, 1.2])).execute(
        request, PushExecutionContext(FakeWorld(), FakePrimitives())
    )

    assert nonfinite.reason is SkillFailure.POLICY_ACTION_NONFINITE
    assert unsafe.reason is SkillFailure.POLICY_ACTION_UNSAFE


def test_agent_runtime_remains_push_backend_independent() -> None:
    source = inspect.getsource(AgentRuntime)
    assert "ClassicalPushBackend" not in source
    assert "BCPushBackend" not in source
    assert "ACTPushBackend" not in source
    assert "temporal_ensemble" not in source
    assert "checkpoint" not in source


def test_chunk_backend_reuses_unsafe_action_rejection() -> None:
    class UnsafeChunkPolicy:
        prediction_horizon = 2

        def predict(self, observation):
            return np.array([[0.35, 0.35, 1.2], [0.35, 0.35, 1.2]])

    backend = ChunkBCPushBackend(UnsafeChunkPolicy(), execution_horizon=2)
    result = backend.execute(
        PushRequest("red_cube", "right_side"),
        PushExecutionContext(FakeWorld(), FakePrimitives()),
    )
    assert result.reason is SkillFailure.POLICY_ACTION_UNSAFE


def test_act_backend_reuses_unsafe_action_rejection_and_resets_queue() -> None:
    class UnsafeACTPolicy:
        def __init__(self) -> None:
            self.resets = 0

        def reset(self) -> None:
            self.resets += 1

        def select_action(self, observation):
            return np.array([0.35, 0.35, 1.2])

    policy = UnsafeACTPolicy()
    backend = ACTPushBackend(policy)
    result = backend.execute(
        PushRequest("red_cube", "right_side"),
        PushExecutionContext(FakeWorld(), FakePrimitives()),
    )
    assert result.reason is SkillFailure.POLICY_ACTION_UNSAFE
    assert policy.resets == 1
