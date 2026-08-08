"""Episode-level splitting and future action-chunk extraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from datasets.trajectory import PushEpisode


def get_action_chunk(
    episode: PushEpisode,
    start_t: int,
    chunk_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return K actions with last-action padding and a valid-timestep mask."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if start_t < 0 or start_t >= episode.length:
        raise IndexError("start_t is outside the episode")
    available = min(chunk_size, episode.length - start_t)
    chunk = np.empty((chunk_size, episode.actions.shape[1]), dtype=np.float64)
    chunk[:available] = episode.actions[start_t : start_t + available]
    if available < chunk_size:
        chunk[available:] = episode.actions[-1]
    mask = np.zeros(chunk_size, dtype=np.bool_)
    mask[:available] = True
    return chunk, mask


def split_episode_indices(
    episode_count: int,
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Deterministically split whole episodes, never individual timesteps."""

    if episode_count < 2:
        raise ValueError("at least two episodes are required")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    order = np.random.default_rng(seed).permutation(episode_count)
    validation_count = max(1, round(episode_count * validation_fraction))
    validation = np.sort(order[:validation_count]).astype(np.int64)
    train = np.sort(order[validation_count:]).astype(np.int64)
    if train.size == 0:
        raise ValueError("validation split leaves no training episodes")
    return train, validation
