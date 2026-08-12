from __future__ import annotations

from dataclasses import asdict
import inspect

import numpy as np
import torch

from datasets import PushDataset, PushEpisode, split_episode_indices
from learning.act_push import (
    ACTArchitecture,
    ACTMotorPolicy,
    PushACTDataset,
    build_act_policy,
    compute_act_normalization,
    push_act_arrays,
)
from skills.act_push_backend import ACTPushBackend


def _dataset() -> PushDataset:
    episodes = []
    for episode_id in range(5):
        observations = np.zeros((4, 10), dtype=np.float64)
        observations[:, 0] = episode_id + np.arange(4) / 10
        actions = observations[:, :3] + 0.01
        episodes.append(PushEpisode(observations, actions, {"id": episode_id}))
    return PushDataset(tuple(episodes))


def test_act_config_is_standard_state_to_action_contract() -> None:
    policy = build_act_policy(ACTArchitecture())
    assert policy.config.input_features["observation.state"].shape == (10,)
    assert policy.config.input_features["observation.environment_state"].shape == (0,)
    assert policy.config.output_features["action"].shape == (3,)
    assert policy.config.chunk_size == 32
    assert policy.config.n_action_steps == 8


def test_act_dataset_stays_inside_episode_and_masks_padding() -> None:
    arrays = push_act_arrays(_dataset())
    normalization = compute_act_normalization(arrays, {0, 1, 2, 3})
    dataset = PushACTDataset(
        arrays, episode_ids={0}, chunk_size=3, normalization=normalization
    )
    sample = dataset[3]
    assert sample["action"].shape == (3, 3)
    np.testing.assert_array_equal(sample["action_is_pad"], [False, True, True])
    expected = (
        arrays.actions[3] - normalization["action"]["mean"]
    ) / normalization["action"]["std"]
    np.testing.assert_allclose(sample["action"].numpy(), np.tile(expected, (3, 1)))


def test_act_split_and_normalization_are_episode_level_and_train_only() -> None:
    dataset = _dataset()
    arrays = push_act_arrays(dataset)
    train, validation = split_episode_indices(5, validation_fraction=0.2, seed=17)
    assert set(train).isdisjoint(validation)
    stats = compute_act_normalization(arrays, set(int(value) for value in train))
    expected = np.concatenate([
        dataset.episodes[int(index)].observations for index in train
    ]).mean(axis=0)
    np.testing.assert_allclose(stats["observation.state"]["mean"], expected)


def test_act_checkpoint_round_trip_and_action_dimension(tmp_path) -> None:
    architecture = ACTArchitecture(
        chunk_size=4, n_action_steps=2, dim_model=32, n_heads=4,
        dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1,
    )
    policy = build_act_policy(architecture)
    stats = {
        "observation.state": {"mean": [0.0] * 10, "std": [1.0] * 10},
        "action": {"mean": [0.0] * 3, "std": [1.0] * 3},
    }
    checkpoint = tmp_path / "act.pt"
    torch.save({
        "model_state_dict": policy.state_dict(),
        "architecture": asdict(architecture),
        "normalization": stats,
        "metadata": {"test": True},
    }, checkpoint)
    loaded = ACTMotorPolicy(checkpoint, device="cpu")
    loaded.reset()
    assert loaded.select_action(np.zeros(10)).shape == (3,)
    assert loaded.metadata["test"] is True


def test_act_backend_has_no_expert_or_raw_environment_path() -> None:
    source = inspect.getsource(ACTPushBackend) + inspect.getsource(ACTMotorPolicy)
    assert "expert" not in source.lower()
    assert "ClassicalPushBackend" not in source
    assert "env.step" not in source
