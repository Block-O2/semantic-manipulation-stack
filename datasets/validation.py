"""Explicit semantic and numerical validation for Push demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datasets.io import PushDataset
from datasets.specs import PUSH_ACTION_SPEC, PUSH_OBSERVATION_SPEC
from primitives import DEFAULT_WORKSPACE


@dataclass(frozen=True)
class DatasetValidationReport:
    valid: bool
    errors: tuple[str, ...]
    statistics: dict[str, object]


def validate_push_dataset(dataset: PushDataset) -> DatasetValidationReport:
    errors: list[str] = []
    lengths = np.asarray([episode.length for episode in dataset.episodes])
    observations = np.concatenate([item.observations for item in dataset.episodes])
    actions = np.concatenate([item.actions for item in dataset.episodes])
    if observations.shape[1] != PUSH_OBSERVATION_SPEC.dimension:
        errors.append("OBSERVATION_DIMENSION_MISMATCH")
    if actions.shape[1] != PUSH_ACTION_SPEC.dimension:
        errors.append("ACTION_DIMENSION_MISMATCH")
    if not np.all(np.isfinite(observations)):
        errors.append("NONFINITE_OBSERVATION")
    if not np.all(np.isfinite(actions)):
        errors.append("NONFINITE_ACTION")
    if np.any(lengths <= 0):
        errors.append("EMPTY_EPISODE")

    for index, episode in enumerate(dataset.episodes):
        metadata = episode.metadata
        if not metadata.get("success", False):
            errors.append(f"EPISODE_{index}_NOT_SUCCESSFUL")
        if not metadata.get("target_reached", False):
            errors.append(f"EPISODE_{index}_TARGET_NOT_REACHED")
        if float(metadata.get("total_displacement", 0.0)) < 0.06:
            errors.append(f"EPISODE_{index}_INSUFFICIENT_DISPLACEMENT")
        if int(metadata.get("episode_length", episode.length)) != episode.length:
            errors.append(f"EPISODE_{index}_LENGTH_METADATA_MISMATCH")

    lower = DEFAULT_WORKSPACE.lower
    upper = DEFAULT_WORKSPACE.upper
    if np.any(actions < lower - 1e-9) or np.any(actions > upper + 1e-9):
        errors.append("ACTION_OUTSIDE_WORKSPACE")
    if np.any(observations[:, :3] < lower - 0.01) or np.any(
        observations[:, :3] > upper + 0.01
    ):
        errors.append("EE_OBSERVATION_OUTSIDE_EXPECTED_WORKSPACE")
    cube = observations[:, 3:6]
    if (
        np.any(np.abs(cube[:, :2]) > 0.4)
        or np.any(cube[:, 2] < 0.75)
        or np.any(cube[:, 2] > 1.25)
    ):
        errors.append("CUBE_OBSERVATION_OUTSIDE_EXPECTED_TABLETOP")
    if np.any(observations[:, 6:8] >= observations[:, 8:10]):
        errors.append("INVALID_TARGET_REGION_OBSERVATION")
    statistics: dict[str, object] = {
        "episodes": len(dataset.episodes),
        "total_timesteps": int(lengths.sum()),
        "episode_length": {
            "min": int(lengths.min()),
            "mean": float(lengths.mean()),
            "max": int(lengths.max()),
        },
        "observation_dimension": observations.shape[1],
        "action_dimension": actions.shape[1],
        "observation_min": observations.min(axis=0).tolist(),
        "observation_max": observations.max(axis=0).tolist(),
        "observation_mean": observations.mean(axis=0).tolist(),
        "observation_std": observations.std(axis=0).tolist(),
        "action_min": actions.min(axis=0).tolist(),
        "action_max": actions.max(axis=0).tolist(),
        "action_mean": actions.mean(axis=0).tolist(),
        "action_std": actions.std(axis=0).tolist(),
    }
    return DatasetValidationReport(not errors, tuple(errors), statistics)
