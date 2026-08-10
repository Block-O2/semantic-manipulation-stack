"""Collect successful BottleExpert demonstrations into a local NPZ dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agent_act_reproduction.config import (
    ACTION_FEATURES,
    ENV_STATE_FEATURES,
    ROBOT_STATE_FEATURES,
)
from agent_act_reproduction.experts import BottleExpert
from agent_act_reproduction.robot import PandaCartesianAdapter
from agent_act_reproduction.sim import make_bottle_env


def collect(
    *,
    output: Path,
    episodes: int,
    seed: int,
    perturbation_m: float,
) -> dict[str, object]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = make_bottle_env(render=False, seed=seed)
    state_rows: list[np.ndarray] = []
    environment_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    try:
        expert = BottleExpert(env, PandaCartesianAdapter(env))
        attempt = 0
        while len(metadata) < episodes:
            episode_seed = seed + attempt
            attempt += 1
            episode = expert.run(seed=episode_seed, perturbation_m=perturbation_m)
            if not episode.success:
                print(json.dumps({"seed": episode_seed, "accepted": False}))
                continue
            episode_index = len(metadata)
            state_rows.append(episode.observations_state)
            environment_rows.append(episode.observations_environment_state)
            action_rows.append(episode.actions)
            episode_ids.append(np.full(episode.episode_length, episode_index, dtype=np.int32))
            metadata.append(
                {
                    "episode_index": episode_index,
                    "seed": episode.seed,
                    "initial_bottle_pose": episode.initial_bottle_pose,
                    "episode_length": episode.episode_length,
                    "success": episode.success,
                    "final_bottle_pose": episode.final_bottle_pose,
                }
            )
            print(
                json.dumps(
                    {
                        "accepted_episode": episode_index + 1,
                        "seed": episode.seed,
                        "episode_length": episode.episode_length,
                    },
                    sort_keys=True,
                )
            )
    finally:
        env.close()

    state = np.concatenate(state_rows, axis=0)
    environment_state = np.concatenate(environment_rows, axis=0)
    actions = np.concatenate(action_rows, axis=0)
    episode_id = np.concatenate(episode_ids, axis=0)
    lengths = np.asarray([item["episode_length"] for item in metadata], dtype=np.int32)
    np.savez_compressed(
        output,
        observation_state=state,
        observation_environment_state=environment_state,
        action=actions,
        episode_id=episode_id,
        episode_lengths=lengths,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        robot_state_features=np.asarray(ROBOT_STATE_FEATURES),
        environment_state_features=np.asarray(ENV_STATE_FEATURES),
        action_features=np.asarray(ACTION_FEATURES),
    )
    summary = {
        "path": str(output.resolve()),
        "episodes": episodes,
        "successful_episodes": len(metadata),
        "attempts": attempt,
        "timesteps": int(state.shape[0]),
        "minimum_episode_length": int(lengths.min()),
        "maximum_episode_length": int(lengths.max()),
        "mean_episode_length": float(lengths.mean()),
        "state_dimension": int(state.shape[1]),
        "environment_state_dimension": int(environment_state.shape[1]),
        "action_dimension": int(actions.shape[1]),
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/bottle_demos_50.npz"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--perturbation-m", type=float, default=0.01)
    args = parser.parse_args()
    collect(
        output=args.output,
        episodes=args.episodes,
        seed=args.seed,
        perturbation_m=args.perturbation_m,
    )


if __name__ == "__main__":
    main()
