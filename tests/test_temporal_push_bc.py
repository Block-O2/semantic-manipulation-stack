from __future__ import annotations

import inspect

import numpy as np
import pytest

from analysis.push_dataset_diagnostics import (
    action_motion_magnitudes,
    motion_mask,
    transition_mask,
)
from datasets import PushDataset, PushEpisode
from learning.temporal_push_bc import (
    ChunkBCPolicy,
    ProgressBCPolicy,
    episode_progress,
    masked_mse,
    train_chunk_bc,
    train_progress_bc,
)
from skills import ChunkBCPushBackend


def temporal_dataset() -> PushDataset:
    episodes = []
    for index in range(6):
        observations = np.zeros((8, 10), dtype=np.float64)
        observations[:, 0] = np.linspace(0.0, 0.07, 8) + index * 0.01
        observations[:, 2] = 1.0
        observations[:, 3] = index * 0.01
        observations[:, 5] = 0.825
        observations[:, 6:] = [0.18, -0.25, 0.34, 0.25]
        actions = observations[:, :3].copy()
        actions[:, 0] += np.linspace(0.0, 0.035, 8)
        episodes.append(PushEpisode(observations, actions, {"episode_id": str(index)}))
    return PushDataset(tuple(episodes))


def test_motion_classification_and_transition_window() -> None:
    observations = np.zeros((8, 10))
    actions = np.zeros((8, 3))
    actions[3:] = [0.01, 0.0, 0.0]
    episode = PushEpisode(observations, actions, {})

    np.testing.assert_allclose(action_motion_magnitudes(episode), [0, 0, 0, .01, 0, 0, 0, 0])
    np.testing.assert_array_equal(motion_mask(episode), [0, 0, 0, 1, 0, 0, 0, 0])
    marked = transition_mask(episode, radius_steps=1, motion_gap_steps=0)
    np.testing.assert_array_equal(marked, [0, 0, 1, 1, 1, 1, 0, 0])


def test_progress_range_and_episode_reset() -> None:
    np.testing.assert_allclose(episode_progress(3), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(episode_progress(2), [0.0, 1.0])
    np.testing.assert_allclose(episode_progress(1), [0.0])


def test_masked_loss_ignores_padding() -> None:
    prediction = np.array([[[1.0], [100.0]]])
    target = np.zeros_like(prediction)
    assert masked_mse(prediction, target, np.array([[True, False]])) == 1.0


def test_progress_and_chunk_training_split_dimensions_and_round_trip(tmp_path) -> None:
    dataset = temporal_dataset()
    progress, progress_result = train_progress_bc(
        dataset, epochs=2, batch_size=16, hidden_dimension=8, seed=4
    )
    chunk, chunk_result = train_chunk_bc(
        dataset,
        prediction_horizon=4,
        epochs=2,
        batch_size=16,
        hidden_dimension=8,
        seed=4,
    )
    assert set(progress_result.train_episode_indices).isdisjoint(
        progress_result.validation_episode_indices
    )
    assert progress.predict(np.zeros(10), 0.0).shape == (3,)
    assert chunk.predict(np.zeros(10)).shape == (4, 3)
    checkpoint = tmp_path / "chunk.npz"
    chunk.save(checkpoint, metadata={"test": True})
    loaded, metadata = ChunkBCPolicy.load(checkpoint)
    np.testing.assert_allclose(loaded.predict(np.zeros(10)), chunk.predict(np.zeros(10)))
    assert metadata["test"] is True
    assert chunk_result.metrics["first_action_l1"] >= 0.0


def test_normalization_uses_training_episodes_only() -> None:
    dataset = temporal_dataset()
    policy, result = train_progress_bc(
        dataset, epochs=1, batch_size=32, hidden_dimension=8, seed=7
    )
    expected = np.concatenate([
        np.column_stack((dataset.episodes[i].observations, episode_progress(8)))
        for i in result.train_episode_indices
    ]).mean(axis=0)
    np.testing.assert_allclose(policy.model.normalization.observation_mean, expected)


def test_execution_horizon_validation_and_no_raw_env_interface() -> None:
    chunk, _ = train_chunk_bc(
        temporal_dataset(), prediction_horizon=3, epochs=1, hidden_dimension=8
    )
    with pytest.raises(ValueError):
        ChunkBCPushBackend(chunk, execution_horizon=0)
    with pytest.raises(ValueError):
        ChunkBCPushBackend(chunk, execution_horizon=4)
    source = inspect.getsource(ChunkBCPushBackend)
    assert "env.step" not in source
    assert "command_cartesian_once" not in source


def test_progress_checkpoint_type_is_enforced(tmp_path) -> None:
    progress, _ = train_progress_bc(
        temporal_dataset(), epochs=1, hidden_dimension=8
    )
    checkpoint = tmp_path / "progress.npz"
    progress.save(checkpoint, metadata={})
    loaded, _ = ProgressBCPolicy.load(checkpoint)
    assert loaded.predict(np.zeros(10), 1.0).shape == (3,)
