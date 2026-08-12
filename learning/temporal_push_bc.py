"""Progress-conditioned and deterministic action-chunk Push BC baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.push_dataset_diagnostics import motion_mask, transition_mask
from datasets import PushDataset, get_action_chunk, split_episode_indices
from learning.push_bc import NormalizationStats, OneStepBCPolicy


@dataclass(frozen=True)
class TemporalTrainingResult:
    train_episode_indices: tuple[int, ...]
    validation_episode_indices: tuple[int, ...]
    loss_history: tuple[float, ...]
    metrics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "train_episode_indices": list(self.train_episode_indices),
            "validation_episode_indices": list(self.validation_episode_indices),
            "loss_history": list(self.loss_history),
            "metrics": self.metrics,
        }


def episode_progress(length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("episode length must be positive")
    if length == 1:
        return np.zeros(1, dtype=np.float64)
    return np.arange(length, dtype=np.float64) / float(length - 1)


def masked_mse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    expanded = np.broadcast_to(mask[..., None], prediction.shape)
    if not np.any(expanded):
        raise ValueError("masked loss requires at least one valid element")
    return float(np.mean((prediction[expanded] - target[expanded]) ** 2))


class ProgressBCPolicy:
    """11D [state, normalized episode progress] to one Cartesian action."""

    def __init__(self, model: OneStepBCPolicy) -> None:
        if model.input_dimension != 11 or model.output_dimension != 3:
            raise ValueError("progress policy model must have dimensions 11 -> 3")
        self.model = model

    def predict(self, observation: np.ndarray, progress: float) -> np.ndarray:
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be in [0, 1]")
        augmented = np.concatenate((np.asarray(observation, dtype=np.float64), [progress]))
        return self.model.predict(augmented)

    def save(self, path: str | Path, *, metadata: dict[str, object]) -> None:
        self.model.save(path, metadata={"policy_type": "progress_bc", **metadata})

    @classmethod
    def load(cls, path: str | Path) -> tuple["ProgressBCPolicy", dict[str, object]]:
        model, metadata = OneStepBCPolicy.load(path)
        if metadata.get("policy_type") != "progress_bc":
            raise ValueError("checkpoint is not a progress BC policy")
        return cls(model), metadata


class ChunkBCPolicy:
    """10D state to K absolute Cartesian targets without temporal ensembling."""

    def __init__(self, model: OneStepBCPolicy, prediction_horizon: int) -> None:
        if prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")
        if model.input_dimension != 10 or model.output_dimension != 3 * prediction_horizon:
            raise ValueError("chunk model dimensions do not match prediction horizon")
        self.model = model
        self.prediction_horizon = prediction_horizon

    def predict(self, observation: np.ndarray) -> np.ndarray:
        return self.model.predict(observation).reshape(self.prediction_horizon, 3)

    def save(self, path: str | Path, *, metadata: dict[str, object]) -> None:
        self.model.save(
            path,
            metadata={
                "policy_type": "chunk_bc",
                "prediction_horizon": self.prediction_horizon,
                **metadata,
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple["ChunkBCPolicy", dict[str, object]]:
        model, metadata = OneStepBCPolicy.load(path)
        if metadata.get("policy_type") != "chunk_bc":
            raise ValueError("checkpoint is not a chunk BC policy")
        return cls(model, int(metadata["prediction_horizon"])), metadata


def _adam_train(
    policy: OneStepBCPolicy,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[float, ...]:
    first = {name: np.zeros_like(value) for name, value in policy.params.items()}
    second = {name: np.zeros_like(value) for name, value in policy.params.items()}
    rng = np.random.default_rng(seed)
    history: list[float] = []
    update = 0
    for _ in range(epochs):
        order = rng.permutation(len(train_x))
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            x = train_x[batch]
            y = train_y[batch]
            mask = train_mask[batch]
            prediction, cache = policy._forward_normalized(x)
            difference = (prediction - y) * mask
            valid = float(mask.sum())
            losses.append(float(np.sum(difference**2) / valid))
            gradient = 2.0 * difference / valid
            x_cache, z1, h1, z2, h2 = cache
            gradients: dict[str, np.ndarray] = {}
            gradients["w3"] = h2.T @ gradient
            gradients["b3"] = gradient.sum(axis=0)
            dh2 = gradient @ policy.params["w3"].T
            dz2 = dh2 * (z2 > 0.0)
            gradients["w2"] = h1.T @ dz2
            gradients["b2"] = dz2.sum(axis=0)
            dh1 = dz2 @ policy.params["w2"].T
            dz1 = dh1 * (z1 > 0.0)
            gradients["w1"] = x_cache.T @ dz1
            gradients["b1"] = dz1.sum(axis=0)
            update += 1
            for name, parameter in policy.params.items():
                first[name] = 0.9 * first[name] + 0.1 * gradients[name]
                second[name] = 0.999 * second[name] + 0.001 * gradients[name] ** 2
                corrected_first = first[name] / (1.0 - 0.9**update)
                corrected_second = second[name] / (1.0 - 0.999**update)
                parameter -= learning_rate * corrected_first / (
                    np.sqrt(corrected_second) + 1e-8
                )
        history.append(float(np.mean(losses)))
    return tuple(history)


def _stratified_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    episodes,
) -> dict[str, object]:
    moving = np.concatenate([motion_mask(ep) for ep in episodes])
    transitions = np.concatenate([transition_mask(ep) for ep in episodes])
    errors = predictions - targets

    def metrics(selected: np.ndarray) -> dict[str, object]:
        chosen = errors[selected]
        return {
            "samples": int(len(chosen)),
            "mse": float(np.mean(chosen**2)),
            "l1": float(np.mean(np.abs(chosen))),
            "l1_per_axis": [float(x) for x in np.mean(np.abs(chosen), axis=0)],
        }

    return {
        "overall": metrics(np.ones(len(errors), dtype=np.bool_)),
        "hold_only": metrics(~moving),
        "moving_only": metrics(moving),
        "transition_only": metrics(transitions),
    }


def train_progress_bc(
    dataset: PushDataset,
    *,
    epochs: int = 60,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_dimension: int = 64,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> tuple[ProgressBCPolicy, TemporalTrainingResult]:
    train_indices, validation_indices = split_episode_indices(
        len(dataset.episodes), validation_fraction=validation_fraction, seed=seed
    )

    def arrays(indices):
        episodes = [dataset.episodes[int(i)] for i in indices]
        observations = np.concatenate([
            np.column_stack((ep.observations, episode_progress(ep.length))) for ep in episodes
        ])
        actions = np.concatenate([ep.actions for ep in episodes])
        return episodes, observations, actions

    _, train_observations, train_actions = arrays(train_indices)
    validation_episodes, validation_observations, validation_actions = arrays(validation_indices)
    normalization = NormalizationStats.fit(train_observations, train_actions)
    train_x = (train_observations - normalization.observation_mean) / normalization.observation_std
    train_y = (train_actions - normalization.action_mean) / normalization.action_std
    model = OneStepBCPolicy(
        input_dimension=11,
        hidden_dimension=hidden_dimension,
        output_dimension=3,
        seed=seed,
        normalization=normalization,
    )
    history = _adam_train(
        model, train_x, train_y, np.ones_like(train_y),
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, seed=seed,
    )
    predictions = model.predict(validation_observations)
    metrics = _stratified_metrics(predictions, validation_actions, validation_episodes)
    return ProgressBCPolicy(model), TemporalTrainingResult(
        tuple(int(i) for i in train_indices),
        tuple(int(i) for i in validation_indices),
        history,
        metrics,
    )


def _chunk_arrays(dataset: PushDataset, indices, horizon: int):
    observations = []
    targets = []
    masks = []
    episodes = [dataset.episodes[int(i)] for i in indices]
    for episode in episodes:
        for timestep in range(episode.length):
            chunk, mask = get_action_chunk(episode, timestep, horizon)
            observations.append(episode.observations[timestep])
            targets.append(chunk)
            masks.append(mask)
    return episodes, np.asarray(observations), np.asarray(targets), np.asarray(masks)


def train_chunk_bc(
    dataset: PushDataset,
    *,
    prediction_horizon: int,
    epochs: int = 60,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_dimension: int = 128,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> tuple[ChunkBCPolicy, TemporalTrainingResult]:
    if prediction_horizon <= 0:
        raise ValueError("prediction_horizon must be positive")
    train_indices, validation_indices = split_episode_indices(
        len(dataset.episodes), validation_fraction=validation_fraction, seed=seed
    )
    _, train_observations, train_targets, train_masks = _chunk_arrays(
        dataset, train_indices, prediction_horizon
    )
    _, validation_observations, validation_targets, validation_masks = _chunk_arrays(
        dataset, validation_indices, prediction_horizon
    )
    observation_mean = train_observations.mean(axis=0)
    observation_std = NormalizationStats._safe_std(train_observations)
    action_mean = np.zeros((prediction_horizon, 3), dtype=np.float64)
    action_std = np.ones((prediction_horizon, 3), dtype=np.float64)
    for horizon in range(prediction_horizon):
        valid = train_targets[train_masks[:, horizon], horizon]
        action_mean[horizon] = valid.mean(axis=0)
        action_std[horizon] = NormalizationStats._safe_std(valid)
    normalization = NormalizationStats(
        observation_mean, observation_std, action_mean.reshape(-1), action_std.reshape(-1)
    )
    train_x = (train_observations - observation_mean) / observation_std
    train_y = ((train_targets - action_mean) / action_std).reshape(len(train_targets), -1)
    train_mask = np.repeat(train_masks[:, :, None], 3, axis=2).reshape(len(train_masks), -1)
    model = OneStepBCPolicy(
        input_dimension=10,
        hidden_dimension=hidden_dimension,
        output_dimension=3 * prediction_horizon,
        seed=seed,
        normalization=normalization,
    )
    history = _adam_train(
        model, train_x, train_y, train_mask,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, seed=seed,
    )
    predictions = model.predict(validation_observations).reshape(-1, prediction_horizon, 3)
    valid_values = np.broadcast_to(validation_masks[:, :, None], predictions.shape)
    absolute_errors = np.abs(predictions - validation_targets)
    squared_errors = (predictions - validation_targets) ** 2
    horizon_l1 = []
    for horizon in range(prediction_horizon):
        valid = validation_masks[:, horizon]
        horizon_l1.append(float(np.mean(absolute_errors[valid, horizon])))
    metrics = {
        "masked_chunk_mse": float(np.mean(squared_errors[valid_values])),
        "masked_chunk_l1": float(np.mean(absolute_errors[valid_values])),
        "first_action_l1": horizon_l1[0],
        "horizon_l1": horizon_l1,
        "validation_valid_actions": int(validation_masks.sum()),
    }
    return ChunkBCPolicy(model, prediction_horizon), TemporalTrainingResult(
        tuple(int(i) for i in train_indices),
        tuple(int(i) for i in validation_indices),
        history,
        metrics,
    )
