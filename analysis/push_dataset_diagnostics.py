"""Diagnose hold dominance, transitions, and state aliasing in Push demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from datasets import PushDataset, PushEpisode, load_push_dataset
from learning import OneStepBCPolicy


DEFAULT_MOTION_EPSILON_M = 0.003
DEFAULT_TRANSITION_RADIUS_STEPS = 5
DEFAULT_MOTION_GAP_STEPS = 12


def action_motion_magnitudes(episode: PushEpisode) -> np.ndarray:
    """Return ||a_t-a_(t-1)||, defining the first sample as zero."""

    magnitudes = np.zeros(episode.length, dtype=np.float64)
    if episode.length > 1:
        magnitudes[1:] = np.linalg.norm(np.diff(episode.actions, axis=0), axis=1)
    return magnitudes


def motion_mask(
    episode: PushEpisode,
    *,
    epsilon_m: float = DEFAULT_MOTION_EPSILON_M,
) -> np.ndarray:
    if epsilon_m < 0.0:
        raise ValueError("epsilon_m must be non-negative")
    return action_motion_magnitudes(episode) > epsilon_m


def transition_mask(
    episode: PushEpisode,
    *,
    epsilon_m: float = DEFAULT_MOTION_EPSILON_M,
    radius_steps: int = DEFAULT_TRANSITION_RADIUS_STEPS,
    motion_gap_steps: int = DEFAULT_MOTION_GAP_STEPS,
    direction_change_degrees: float = 60.0,
) -> np.ndarray:
    """Mark windows around debounced motion-state or direction changes."""

    if radius_steps < 0 or motion_gap_steps < 0:
        raise ValueError("transition window parameters must be non-negative")
    raw_moving = motion_mask(episode, epsilon_m=epsilon_m)
    moving = raw_moving.copy()
    active = np.flatnonzero(raw_moving)
    for previous, current in zip(active[:-1], active[1:]):
        if current - previous - 1 <= motion_gap_steps:
            moving[previous : current + 1] = True
    vectors = np.zeros_like(episode.actions)
    vectors[1:] = np.diff(episode.actions, axis=0)
    boundaries = set(int(i) for i in np.flatnonzero(moving[1:] != moving[:-1]) + 1)
    cosine_limit = float(np.cos(np.deg2rad(direction_change_degrees)))
    for previous_index, index in zip(active[:-1], active[1:]):
        previous = vectors[previous_index]
        current = vectors[index]
        cosine = float(previous @ current / (np.linalg.norm(previous) * np.linalg.norm(current)))
        if cosine <= cosine_limit:
            boundaries.add(index)
    mask = np.zeros(episode.length, dtype=np.bool_)
    for index in boundaries:
        lower = max(0, index - radius_steps)
        upper = min(episode.length, index + radius_steps + 1)
        mask[lower:upper] = True
    return mask


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _error_metrics(errors: np.ndarray) -> dict[str, object]:
    if errors.size == 0:
        return {"samples": 0, "mse": None, "l1": None, "l1_per_axis": None}
    return {
        "samples": int(errors.shape[0]),
        "mse": float(np.mean(errors**2)),
        "l1": float(np.mean(np.abs(errors))),
        "l1_per_axis": [float(x) for x in np.mean(np.abs(errors), axis=0)],
    }


def _flatten(dataset: PushDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observations = np.concatenate([episode.observations for episode in dataset.episodes])
    actions = np.concatenate([episode.actions for episode in dataset.episodes])
    episode_ids = np.concatenate([
        np.full(episode.length, index, dtype=np.int64)
        for index, episode in enumerate(dataset.episodes)
    ])
    timesteps = np.concatenate([
        np.arange(episode.length, dtype=np.int64) for episode in dataset.episodes
    ])
    return observations, actions, episode_ids, timesteps


def _aliasing_examples(
    dataset: PushDataset,
    moving: np.ndarray,
    *,
    maximum_examples: int = 12,
) -> dict[str, object]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "state-aliasing analysis requires the optional 'analysis' dependencies"
        ) from exc
    observations, actions, episode_ids, timesteps = _flatten(dataset)
    std = observations.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (observations - observations.mean(axis=0)) / std
    tree = cKDTree(normalized)
    distances, neighbors = tree.query(normalized, k=24)
    candidates: list[dict[str, object]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for source in range(len(observations)):
        for distance, target in zip(distances[source, 1:], neighbors[source, 1:]):
            target = int(target)
            if bool(moving[source]) == bool(moving[target]):
                continue
            pair = (min(source, target), max(source, target))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            action_distance = float(np.linalg.norm(actions[source] - actions[target]))
            candidates.append({
                "normalized_observation_distance": float(distance),
                "action_distance_m": action_distance,
                "same_episode": bool(episode_ids[source] == episode_ids[target]),
                "first": {
                    "episode": int(episode_ids[source]),
                    "timestep": int(timesteps[source]),
                    "motion": "moving" if moving[source] else "hold",
                },
                "second": {
                    "episode": int(episode_ids[target]),
                    "timestep": int(timesteps[target]),
                    "motion": "moving" if moving[target] else "hold",
                },
            })
            break
    candidates.sort(
        key=lambda item: (
            item["normalized_observation_distance"],
            -item["action_distance_m"],
        )
    )
    close = [item for item in candidates if item["normalized_observation_distance"] <= 0.05]
    materially_different = [item for item in close if item["action_distance_m"] >= 0.01]
    same_episode = [item for item in materially_different if item["same_episode"]]
    cross_episode = [item for item in materially_different if not item["same_episode"]]
    return {
        "normalization": "per-feature dataset mean/std; Euclidean distance",
        "close_pair_threshold": 0.05,
        "material_action_difference_m": 0.01,
        "hold_move_pairs_within_threshold": len(close),
        "materially_ambiguous_pairs": len(materially_different),
        "same_episode_material_pairs": len(same_episode),
        "cross_episode_material_pairs": len(cross_episode),
        "same_episode_examples": same_episode[: maximum_examples // 2],
        "cross_episode_examples": cross_episode[: maximum_examples // 2],
        "examples": (same_episode[: maximum_examples // 2] + cross_episode[: maximum_examples // 2])
        or (materially_different or close or candidates)[:maximum_examples],
    }


def diagnose(
    dataset: PushDataset,
    *,
    epsilon_m: float = DEFAULT_MOTION_EPSILON_M,
    transition_radius_steps: int = DEFAULT_TRANSITION_RADIUS_STEPS,
    checkpoint: str | Path | None = None,
) -> dict[str, object]:
    magnitudes_by_episode = [action_motion_magnitudes(ep) for ep in dataset.episodes]
    moving_by_episode = [motion_mask(ep, epsilon_m=epsilon_m) for ep in dataset.episodes]
    transition_by_episode = [
        transition_mask(ep, epsilon_m=epsilon_m, radius_steps=transition_radius_steps)
        for ep in dataset.episodes
    ]
    magnitudes = np.concatenate(magnitudes_by_episode)
    moving = np.concatenate(moving_by_episode)
    transitions = np.concatenate(transition_by_episode)
    hold_ratios = np.asarray([1.0 - mask.mean() for mask in moving_by_episode])
    report: dict[str, object] = {
        "dataset": {
            "episodes": len(dataset.episodes),
            "timesteps": dataset.total_timesteps,
        },
        "motion_definition": {
            "delta": "||a_t - a_(t-1)||; t=0 is defined as 0",
            "motion_epsilon_m": epsilon_m,
            "threshold_reason": "fixed in the empirical gap between sub-millimetre hold jitter and roughly 7 mm moving commands",
        },
        "action_delta_m": _distribution(magnitudes),
        "motion_counts": {
            "hold_timesteps": int((~moving).sum()),
            "moving_timesteps": int(moving.sum()),
            "hold_ratio": float((~moving).mean()),
            "moving_ratio": float(moving.mean()),
            "episode_hold_ratio": {
                "min": float(hold_ratios.min()),
                "mean": float(hold_ratios.mean()),
                "max": float(hold_ratios.max()),
            },
        },
        "transition_definition": {
            "radius_steps": transition_radius_steps,
            "seconds_each_side_at_20hz": transition_radius_steps / 20.0,
            "events": "STATIC/MOVING boundary or >=60 degree direction change while moving",
            "debounce": "raw moving pulses separated by <=12 steps are one moving segment",
            "timesteps": int(transitions.sum()),
            "ratio": float(transitions.mean()),
        },
        "state_aliasing": _aliasing_examples(dataset, moving),
    }
    if checkpoint is not None:
        policy, metadata = OneStepBCPolicy.load(checkpoint)
        validation_indices = metadata["training_result"]["validation_episode_indices"]
        validation_episodes = [dataset.episodes[int(i)] for i in validation_indices]
        predictions = np.concatenate([policy.predict(ep.observations) for ep in validation_episodes])
        targets = np.concatenate([ep.actions for ep in validation_episodes])
        validation_moving = np.concatenate([moving_by_episode[int(i)] for i in validation_indices])
        validation_transitions = np.concatenate([
            transition_by_episode[int(i)] for i in validation_indices
        ])
        errors = predictions - targets
        report["one_step_validation"] = {
            "episode_indices": validation_indices,
            "overall": _error_metrics(errors),
            "hold_only": _error_metrics(errors[~validation_moving]),
            "moving_only": _error_metrics(errors[validation_moving]),
            "transition_only": _error_metrics(errors[validation_transitions]),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--motion-epsilon-m", type=float, default=DEFAULT_MOTION_EPSILON_M)
    parser.add_argument("--transition-radius-steps", type=int, default=DEFAULT_TRANSITION_RADIUS_STEPS)
    args = parser.parse_args()
    report = diagnose(
        load_push_dataset(args.dataset),
        epsilon_m=args.motion_epsilon_m,
        transition_radius_steps=args.transition_radius_steps,
        checkpoint=args.checkpoint,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
