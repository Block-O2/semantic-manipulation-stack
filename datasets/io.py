"""Deterministic, pickle-free NPZ storage for variable-length Push episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datasets.specs import PUSH_ACTION_SPEC, PUSH_OBSERVATION_SPEC
from datasets.trajectory import PushEpisode


DATASET_VERSION = 1


@dataclass(frozen=True)
class PushDataset:
    episodes: tuple[PushEpisode, ...]

    def __post_init__(self) -> None:
        if not self.episodes:
            raise ValueError("dataset must contain at least one episode")

    @property
    def total_timesteps(self) -> int:
        return sum(episode.length for episode in self.episodes)


def save_push_dataset(path: str | Path, dataset: PushDataset) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    observations = np.concatenate(
        [episode.observations for episode in dataset.episodes],
        axis=0,
    )
    actions = np.concatenate([episode.actions for episode in dataset.episodes], axis=0)
    lengths = np.asarray([episode.length for episode in dataset.episodes], dtype=np.int64)
    offsets = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(lengths)])
    metadata_json = json.dumps(
        [episode.metadata for episode in dataset.episodes],
        sort_keys=True,
        separators=(",", ":"),
    )
    schema_json = json.dumps(
        {
            "version": DATASET_VERSION,
            "observation": PUSH_OBSERVATION_SPEC.to_dict(),
            "action": PUSH_ACTION_SPEC.to_dict(),
            "episode_storage": "concatenated arrays with exclusive offsets",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    np.savez_compressed(
        destination,
        observations=observations,
        actions=actions,
        episode_offsets=offsets,
        metadata_json=np.asarray(metadata_json),
        schema_json=np.asarray(schema_json),
    )


def load_push_dataset(path: str | Path) -> PushDataset:
    with np.load(Path(path), allow_pickle=False) as payload:
        observations = np.asarray(payload["observations"], dtype=np.float64)
        actions = np.asarray(payload["actions"], dtype=np.float64)
        offsets = np.asarray(payload["episode_offsets"], dtype=np.int64)
        metadata = json.loads(str(payload["metadata_json"].item()))
        schema = json.loads(str(payload["schema_json"].item()))
    if schema.get("version") != DATASET_VERSION:
        raise ValueError(f"unsupported Push dataset version {schema.get('version')!r}")
    if offsets.ndim != 1 or offsets.size < 2 or offsets[0] != 0:
        raise ValueError("invalid episode offsets")
    if offsets[-1] != observations.shape[0] or actions.shape[0] != observations.shape[0]:
        raise ValueError("dataset arrays and offsets are inconsistent")
    if len(metadata) != offsets.size - 1:
        raise ValueError("metadata count does not match episode count")
    episodes = tuple(
        PushEpisode(
            observations[offsets[index] : offsets[index + 1]],
            actions[offsets[index] : offsets[index + 1]],
            metadata[index],
        )
        for index in range(offsets.size - 1)
    )
    return PushDataset(episodes)
