from __future__ import annotations

import numpy as np

from datasets import PushDataset, PushEpisode
from learning import OneStepBCPolicy, train_one_step_bc


def synthetic_dataset() -> PushDataset:
    rng = np.random.default_rng(3)
    episodes = []
    for index in range(6):
        observations = rng.normal(size=(12, 10))
        actions = observations[:, :3] * 0.02 + np.array([0.0, 0.0, 0.9])
        episodes.append(
            PushEpisode(
                observations,
                actions,
                {
                    "episode_id": str(index),
                    "success": True,
                    "target_reached": True,
                    "total_displacement": 0.2,
                },
            )
        )
    return PushDataset(tuple(episodes))


def test_bc_model_dimensions_training_and_checkpoint_round_trip(tmp_path) -> None:
    policy, result = train_one_step_bc(
        synthetic_dataset(),
        epochs=12,
        batch_size=16,
        hidden_dimension=16,
        seed=11,
    )
    observation = np.zeros(10)
    prediction = policy.predict(observation)
    checkpoint = tmp_path / "bc.npz"
    policy.save(checkpoint, metadata={"epochs": 12})
    loaded, metadata = OneStepBCPolicy.load(checkpoint)

    assert prediction.shape == (3,)
    assert result.validation_mse >= 0.0
    assert len(result.train_episode_indices) == 5
    assert len(result.validation_episode_indices) == 1
    np.testing.assert_allclose(loaded.predict(observation), prediction)
    assert metadata["epochs"] == 12
