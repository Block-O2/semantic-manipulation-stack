"""Dependency-free NumPy MLP for a one-step state-to-Cartesian BC baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from datasets import PushDataset, split_episode_indices


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NormalizationStats:
    observation_mean: FloatArray
    observation_std: FloatArray
    action_mean: FloatArray
    action_std: FloatArray

    @staticmethod
    def _safe_std(values: FloatArray) -> FloatArray:
        std = values.std(axis=0)
        return np.where(std < 1e-6, 1.0, std)

    @classmethod
    def fit(cls, observations: FloatArray, actions: FloatArray) -> "NormalizationStats":
        return cls(
            observations.mean(axis=0),
            cls._safe_std(observations),
            actions.mean(axis=0),
            cls._safe_std(actions),
        )


@dataclass(frozen=True)
class BCTrainingResult:
    train_episode_indices: tuple[int, ...]
    validation_episode_indices: tuple[int, ...]
    loss_history: tuple[float, ...]
    validation_mse: float
    validation_l1: float
    validation_mse_per_dimension: tuple[float, ...]
    validation_l1_per_dimension: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "train_episode_indices": list(self.train_episode_indices),
            "validation_episode_indices": list(self.validation_episode_indices),
            "loss_history": list(self.loss_history),
            "validation_mse": self.validation_mse,
            "validation_l1": self.validation_l1,
            "validation_mse_per_dimension": list(
                self.validation_mse_per_dimension
            ),
            "validation_l1_per_dimension": list(self.validation_l1_per_dimension),
        }


class OneStepBCPolicy:
    """Two-hidden-layer ReLU MLP with explicit normalization metadata."""

    def __init__(
        self,
        *,
        input_dimension: int = 10,
        hidden_dimension: int = 64,
        output_dimension: int = 3,
        seed: int = 0,
        normalization: NormalizationStats | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.params: dict[str, FloatArray] = {
            "w1": rng.normal(
                0.0,
                np.sqrt(2.0 / input_dimension),
                (input_dimension, hidden_dimension),
            ),
            "b1": np.zeros(hidden_dimension),
            "w2": rng.normal(
                0.0,
                np.sqrt(2.0 / hidden_dimension),
                (hidden_dimension, hidden_dimension),
            ),
            "b2": np.zeros(hidden_dimension),
            "w3": rng.normal(
                0.0,
                np.sqrt(2.0 / hidden_dimension),
                (hidden_dimension, output_dimension),
            ),
            "b3": np.zeros(output_dimension),
        }
        self.normalization = normalization

    def _forward_normalized(
        self,
        normalized_observations: FloatArray,
    ) -> tuple[FloatArray, tuple[FloatArray, ...]]:
        z1 = normalized_observations @ self.params["w1"] + self.params["b1"]
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.params["w2"] + self.params["b2"]
        h2 = np.maximum(z2, 0.0)
        output = h2 @ self.params["w3"] + self.params["b3"]
        return output, (normalized_observations, z1, h1, z2, h2)

    def predict(self, observations: FloatArray) -> FloatArray:
        if self.normalization is None:
            raise RuntimeError("policy normalization is not configured")
        values = np.asarray(observations, dtype=np.float64)
        single = values.ndim == 1
        values = np.atleast_2d(values)
        if values.shape[1] != self.input_dimension:
            raise ValueError("observation dimension does not match policy")
        normalized = (
            values - self.normalization.observation_mean
        ) / self.normalization.observation_std
        output, _ = self._forward_normalized(normalized)
        actions = (
            output * self.normalization.action_std
            + self.normalization.action_mean
        )
        return actions[0] if single else actions

    def save(self, path: str | Path, *, metadata: dict[str, object]) -> None:
        if self.normalization is None:
            raise RuntimeError("cannot save a policy without normalization")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            **self.params,
            observation_mean=self.normalization.observation_mean,
            observation_std=self.normalization.observation_std,
            action_mean=self.normalization.action_mean,
            action_std=self.normalization.action_std,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple["OneStepBCPolicy", dict[str, object]]:
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            policy = cls(
                input_dimension=int(payload["w1"].shape[0]),
                hidden_dimension=int(payload["w1"].shape[1]),
                output_dimension=int(payload["w3"].shape[1]),
                normalization=NormalizationStats(
                    np.asarray(payload["observation_mean"], dtype=np.float64),
                    np.asarray(payload["observation_std"], dtype=np.float64),
                    np.asarray(payload["action_mean"], dtype=np.float64),
                    np.asarray(payload["action_std"], dtype=np.float64),
                ),
            )
            policy.params = {
                name: np.asarray(payload[name], dtype=np.float64)
                for name in ("w1", "b1", "w2", "b2", "w3", "b3")
            }
        return policy, metadata


def _episode_arrays(
    dataset: PushDataset,
    indices: NDArray[np.int64],
) -> tuple[FloatArray, FloatArray]:
    observations = np.concatenate(
        [dataset.episodes[int(index)].observations for index in indices]
    )
    actions = np.concatenate(
        [dataset.episodes[int(index)].actions for index in indices]
    )
    return observations, actions


def train_one_step_bc(
    dataset: PushDataset,
    *,
    epochs: int = 60,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_dimension: int = 64,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[OneStepBCPolicy, BCTrainingResult]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("training hyperparameters must be positive")
    train_indices, validation_indices = split_episode_indices(
        len(dataset.episodes),
        validation_fraction=validation_fraction,
        seed=seed,
    )
    train_observations, train_actions = _episode_arrays(dataset, train_indices)
    validation_observations, validation_actions = _episode_arrays(
        dataset,
        validation_indices,
    )
    normalization = NormalizationStats.fit(train_observations, train_actions)
    train_x = (
        train_observations - normalization.observation_mean
    ) / normalization.observation_std
    train_y = (train_actions - normalization.action_mean) / normalization.action_std
    policy = OneStepBCPolicy(
        hidden_dimension=hidden_dimension,
        seed=seed,
        normalization=normalization,
    )

    first_moment = {name: np.zeros_like(value) for name, value in policy.params.items()}
    second_moment = {name: np.zeros_like(value) for name, value in policy.params.items()}
    rng = np.random.default_rng(seed)
    history: list[float] = []
    update = 0
    for _ in range(epochs):
        order = rng.permutation(train_x.shape[0])
        epoch_losses: list[float] = []
        for start in range(0, order.size, batch_size):
            batch = order[start : start + batch_size]
            x = train_x[batch]
            y = train_y[batch]
            prediction, cache = policy._forward_normalized(x)
            difference = prediction - y
            epoch_losses.append(float(np.mean(difference**2)))
            gradient = 2.0 * difference / difference.size
            x_cache, z1, h1, z2, h2 = cache
            gradients: dict[str, FloatArray] = {}
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
                first_moment[name] = 0.9 * first_moment[name] + 0.1 * gradients[name]
                second_moment[name] = (
                    0.999 * second_moment[name] + 0.001 * gradients[name] ** 2
                )
                corrected_first = first_moment[name] / (1.0 - 0.9**update)
                corrected_second = second_moment[name] / (1.0 - 0.999**update)
                parameter -= learning_rate * corrected_first / (
                    np.sqrt(corrected_second) + 1e-8
                )
        history.append(float(np.mean(epoch_losses)))

    predicted = policy.predict(validation_observations)
    errors = predicted - validation_actions
    mse_per_dimension = np.mean(errors**2, axis=0)
    l1_per_dimension = np.mean(np.abs(errors), axis=0)
    result = BCTrainingResult(
        tuple(int(value) for value in train_indices),
        tuple(int(value) for value in validation_indices),
        tuple(history),
        float(np.mean(errors**2)),
        float(np.mean(np.abs(errors))),
        tuple(float(value) for value in mse_per_dimension),
        tuple(float(value) for value in l1_per_dimension),
    )
    return policy, result
