from __future__ import annotations

from dataclasses import asdict
import inspect

import numpy as np
import torch

from datasets import PushDataset, PushEpisode
from learning.diffusion_push import (
    DiffusionArchitecture,
    DiffusionMotorPolicy,
    PushDiffusionDataset,
    build_diffusion_model,
    build_noise_scheduler,
    compute_diffusion_normalization,
    normalize_array,
)
from primitives import PrimitiveResult, WorkspaceBounds
from robot import Pose
from skills import (
    BCPushConfig,
    DiffusionPushBackend,
    PushExecutionContext,
    PushRequest,
    SkillFailure,
)


def _architecture() -> DiffusionArchitecture:
    return DiffusionArchitecture(
        prediction_horizon=4,
        execution_horizon=2,
        diffusion_step_embed_dim=8,
        down_dims=(8, 16),
        n_groups=4,
        num_train_timesteps=2,
        num_inference_steps=2,
    )


def _dataset() -> PushDataset:
    episodes = []
    for episode_id, length in enumerate((3, 4)):
        observations = np.arange(length * 10, dtype=np.float64).reshape(length, 10)
        observations += episode_id * 1000
        actions = np.arange(length * 3, dtype=np.float64).reshape(length, 3)
        actions += episode_id * 100
        episodes.append(PushEpisode(observations, actions, {"episode": episode_id}))
    return PushDataset(tuple(episodes))


def _checkpoint(path) -> DiffusionArchitecture:
    architecture = _architecture()
    model = build_diffusion_model(architecture)
    normalization = {
        "observation": {
            "min": np.zeros(10), "max": np.ones(10),
            "mean": np.zeros(10), "std": np.ones(10),
            "scale": np.ones(10), "offset": np.zeros(10),
        },
        "action": {
            "min": np.zeros(3), "max": np.ones(3),
            "mean": np.zeros(3), "std": np.ones(3),
            "scale": np.ones(3), "offset": np.zeros(3),
        },
    }
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": asdict(architecture),
        "normalization": {
            key: {name: values.tolist() for name, values in group.items()}
            for key, group in normalization.items()
        },
        "metadata": {"seed": 17},
    }, path)
    return architecture


def test_diffusion_config_model_and_scheduler_construction() -> None:
    architecture = DiffusionArchitecture()
    scheduler = build_noise_scheduler(architecture)
    model = build_diffusion_model(architecture)
    output = model(
        torch.randn(2, 16, 3),
        torch.tensor([0, 99]),
        torch.randn(2, 20),
    )

    assert (architecture.observation_horizon, architecture.prediction_horizon) == (2, 16)
    assert architecture.execution_horizon == 8
    assert scheduler.config.beta_schedule == "squaredcos_cap_v2"
    assert scheduler.config.prediction_type == "epsilon"
    assert output.shape == (2, 16, 3)


def test_temporal_windows_pad_without_crossing_episode_boundaries() -> None:
    dataset = _dataset()
    architecture = _architecture()
    normalization = compute_diffusion_normalization(dataset, {0})
    windows = PushDiffusionDataset(
        dataset,
        episode_ids={0, 1},
        architecture=architecture,
        normalization=normalization,
    )

    first = windows[0]
    np.testing.assert_allclose(first["observation"][0], first["observation"][1])
    assert first["action_valid"].tolist() == [False, True, True, True]
    last_episode_zero = windows[2]
    assert last_episode_zero["action_valid"].tolist() == [True, True, False, False]
    first_episode_one = windows[3]
    assert int(first_episode_one["episode_id"]) == 1
    assert first_episode_one["action_valid"].tolist() == [False, True, True, True]
    expected = normalize_array(dataset.episodes[1].actions[0], normalization["action"])
    np.testing.assert_allclose(first_episode_one["action"][0], expected)


def test_normalization_uses_training_episodes_only() -> None:
    dataset = _dataset()
    stats = compute_diffusion_normalization(dataset, {0})

    np.testing.assert_allclose(stats["observation"]["min"], dataset.episodes[0].observations.min(axis=0))
    np.testing.assert_allclose(stats["observation"]["max"], dataset.episodes[0].observations.max(axis=0))
    assert float(stats["observation"]["max"].max()) < 1000.0


def test_checkpoint_load_action_shape_history_and_reset(tmp_path) -> None:
    checkpoint = tmp_path / "diffusion.pt"
    architecture = _checkpoint(checkpoint)
    policy = DiffusionMotorPolicy(checkpoint, device="cpu")
    observation = np.linspace(0.0, 0.9, 10, dtype=np.float32)

    action = policy.predict(observation)
    assert action.shape == (architecture.execution_horizon, 3)
    assert np.all(np.isfinite(action))
    assert policy.history_size == 1
    policy.observe(observation + 1.0)
    assert policy.history_size == architecture.observation_horizon
    policy.reset()
    assert policy.history_size == 0
    assert policy.inference_calls == 0


class _World:
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


class _Primitives:
    def __init__(self) -> None:
        self._pose = Pose.from_values([-0.25, -0.25, 1.1], [0.0, 0.0, 0.0, 1.0])
        self.workspace = WorkspaceBounds(
            np.array([-0.35, -0.35, 0.805]), np.array([0.35, 0.35, 1.25])
        )

    @property
    def current_pose(self) -> Pose:
        return self._pose

    def open_gripper(self, *, steps: int) -> PrimitiveResult:
        return PrimitiveResult(True, None, steps, None, None)

    def command_cartesian_once(self, target: Pose, *, max_cartesian_step: float):
        self._pose = target
        return PrimitiveResult(True, None, 1, 0.0, 0.0)


def test_backend_interface_history_updates_and_safety_boundary() -> None:
    class UnsafePolicy:
        architecture = _architecture()
        inference_calls = 0
        inference_latencies: list[float] = []
        last_inference_latency = 0.0

        def __init__(self) -> None:
            self.observations: list[np.ndarray] = []
            self.resets = 0

        @property
        def history_size(self) -> int:
            return len(self.observations)

        def reset(self) -> None:
            self.resets += 1
            self.observations.clear()

        def observe(self, observation: np.ndarray) -> None:
            self.observations.append(observation.copy())

        def predict(self, observation: np.ndarray) -> np.ndarray:
            return np.tile(np.array([0.35, 0.35, 1.2]), (2, 1))

    policy = UnsafePolicy()
    backend = DiffusionPushBackend(policy)
    result = backend.execute(
        PushRequest("red_cube", "right_side"),
        PushExecutionContext(_World(), _Primitives()),
    )

    assert result.reason is SkillFailure.POLICY_ACTION_UNSAFE
    assert policy.resets == 1
    assert policy.observations == []
    source = inspect.getsource(DiffusionPushBackend)
    assert "env.step" not in source
    assert "expert" not in source.lower()
    assert "ClassicalPushBackend" not in source
    assert "fallback" not in source.lower()


def test_backend_records_realized_observations_between_action_chunks() -> None:
    class ShortPolicy:
        architecture = _architecture()
        inference_calls = 1
        inference_latencies = [0.01]
        last_inference_latency = 0.01

        def __init__(self) -> None:
            self.observations: list[np.ndarray] = []

        @property
        def history_size(self) -> int:
            return len(self.observations)

        def reset(self) -> None:
            self.observations.clear()

        def observe(self, observation: np.ndarray) -> None:
            self.observations.append(observation.copy())

        def predict(self, observation: np.ndarray) -> np.ndarray:
            current = observation[:3]
            return np.stack((current + np.array([0.001, 0.0, 0.0]),) * 2)

    policy = ShortPolicy()
    backend = DiffusionPushBackend(
        policy,
        config=BCPushConfig(maximum_control_steps=1),
    )
    result = backend.execute(
        PushRequest("red_cube", "right_side"),
        PushExecutionContext(_World(), _Primitives()),
    )

    assert result.reason is SkillFailure.POLICY_TIMEOUT
    assert len(policy.observations) == 1
    assert policy.observations[0].shape == (10,)
