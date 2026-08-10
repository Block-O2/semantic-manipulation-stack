"""Local, episode-aware action-chunk dataset for LeRobot ACT."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BottleDatasetArrays:
    observation_state: np.ndarray
    observation_environment_state: np.ndarray
    action: np.ndarray
    episode_id: np.ndarray
    episode_lengths: np.ndarray
    metadata: list[dict[str, object]]
    feature_names: dict[str, tuple[str, ...]]


def load_bottle_dataset(path: Path) -> BottleDatasetArrays:
    with np.load(path, allow_pickle=False) as archive:
        return BottleDatasetArrays(
            observation_state=np.asarray(archive["observation_state"], dtype=np.float32),
            observation_environment_state=np.asarray(
                archive["observation_environment_state"], dtype=np.float32
            ),
            action=np.asarray(archive["action"], dtype=np.float32),
            episode_id=np.asarray(archive["episode_id"], dtype=np.int32),
            episode_lengths=np.asarray(archive["episode_lengths"], dtype=np.int32),
            metadata=json.loads(str(archive["metadata_json"])),
            feature_names={
                "observation.state": tuple(str(x) for x in archive["robot_state_features"]),
                "observation.environment_state": tuple(
                    str(x) for x in archive["environment_state_features"]
                ),
                "action": tuple(str(x) for x in archive["action_features"]),
            },
        )


def compute_normalization(arrays: BottleDatasetArrays, train_episode_ids: set[int]) -> dict[str, dict[str, np.ndarray]]:
    mask = np.isin(arrays.episode_id, np.asarray(sorted(train_episode_ids), dtype=np.int32))
    result: dict[str, dict[str, np.ndarray]] = {}
    for key, values in (
        ("observation.state", arrays.observation_state[mask]),
        ("observation.environment_state", arrays.observation_environment_state[mask]),
        ("action", arrays.action[mask]),
    ):
        mean = values.mean(axis=0).astype(np.float32)
        std = values.std(axis=0).astype(np.float32)
        std = np.maximum(std, np.float32(1e-4))
        result[key] = {"mean": mean, "std": std}
    return result


class BottleACTDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        arrays: BottleDatasetArrays,
        *,
        episode_ids: set[int],
        chunk_size: int,
        normalization: dict[str, dict[str, np.ndarray]],
    ) -> None:
        self.arrays = arrays
        self.chunk_size = int(chunk_size)
        self.normalization = normalization
        self.indices = np.flatnonzero(
            np.isin(arrays.episode_id, np.asarray(sorted(episode_ids), dtype=np.int32))
        )
        self._episode_ends: dict[int, int] = {}
        cursor = 0
        for episode, length in enumerate(arrays.episode_lengths.tolist()):
            cursor += int(length)
            self._episode_ends[episode] = cursor

    def __len__(self) -> int:
        return int(self.indices.size)

    def _normalize(self, key: str, values: np.ndarray) -> np.ndarray:
        stats = self.normalization[key]
        return (values - stats["mean"]) / stats["std"]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        episode = int(self.arrays.episode_id[row])
        episode_end = self._episode_ends[episode]
        valid = min(self.chunk_size, episode_end - row)
        action_chunk = np.zeros((self.chunk_size, self.arrays.action.shape[1]), dtype=np.float32)
        action_chunk[:valid] = self.arrays.action[row : row + valid]
        if valid < self.chunk_size:
            action_chunk[valid:] = self.arrays.action[episode_end - 1]
        is_pad = np.ones(self.chunk_size, dtype=bool)
        is_pad[:valid] = False
        return {
            "observation.state": torch.from_numpy(
                self._normalize("observation.state", self.arrays.observation_state[row])
            ),
            "observation.environment_state": torch.from_numpy(
                self._normalize(
                    "observation.environment_state",
                    self.arrays.observation_environment_state[row],
                )
            ),
            "action": torch.from_numpy(self._normalize("action", action_chunk)),
            "action_is_pad": torch.from_numpy(is_pad),
        }
