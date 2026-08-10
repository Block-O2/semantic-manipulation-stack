"""Collect successful expert demonstrations for Tissue or Draw."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agent_act_reproduction.config import ACTION_FEATURES, ROBOT_STATE_FEATURES
from agent_act_reproduction.experts import DrawExpert, TissueExpert
from agent_act_reproduction.robot import PandaCartesianAdapter
from agent_act_reproduction.sim import make_draw_env, make_tissue_env


FEATURES = {
    "tissue": (
        "tissue_position_x_m", "tissue_position_y_m", "tissue_position_z_m",
        "tissue_velocity_x_mps", "tissue_velocity_y_mps", "tissue_velocity_z_mps",
        "pull_target_x_m", "pull_target_y_m", "pull_target_z_m",
        "tissue_grasped_bool", "episode_progress_fraction",
    ),
    "draw": (
        "pen_position_x_m", "pen_position_y_m", "pen_position_z_m",
        "pen_tip_x_m", "pen_tip_y_m", "pen_tip_z_m",
        "line_end_eef_x_m", "line_end_eef_y_m", "line_end_eef_z_m",
        "pen_grasped_bool", "stroke_progress_fraction",
    ),
}


def collect(*, task: str, output: Path, episodes: int, seed: int, perturbation_m: float) -> dict[str, object]:
    if task not in FEATURES:
        raise ValueError(f"Unsupported task: {task}")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = (make_tissue_env if task == "tissue" else make_draw_env)(seed=seed)
    expert_type = TissueExpert if task == "tissue" else DrawExpert
    state_rows: list[np.ndarray] = []
    environment_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    attempt = 0
    try:
        expert = expert_type(env, PandaCartesianAdapter(env))
        while len(metadata) < episodes:
            episode_seed = seed + attempt
            attempt += 1
            episode = expert.run(seed=episode_seed, perturbation_m=perturbation_m)
            if not episode.success:
                print(json.dumps({"task": task, "seed": episode_seed, "accepted": False}))
                continue
            episode_index = len(metadata)
            state_rows.append(episode.observations_state)
            environment_rows.append(episode.observations_environment_state)
            action_rows.append(episode.actions)
            episode_ids.append(np.full(episode.episode_length, episode_index, dtype=np.int32))
            metadata.append({
                "episode_index": episode_index,
                "task": task,
                "seed": episode.seed,
                "episode_length": episode.episode_length,
                "success": True,
                "success_details": episode.success_details,
            })
            print(json.dumps({
                "task": task,
                "accepted_episode": episode_index + 1,
                "seed": episode.seed,
                "episode_length": episode.episode_length,
            }, sort_keys=True))
    finally:
        env.close()
    state = np.concatenate(state_rows)
    environment_state = np.concatenate(environment_rows)
    actions = np.concatenate(action_rows)
    episode_id = np.concatenate(episode_ids)
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
        environment_state_features=np.asarray(FEATURES[task]),
        action_features=np.asarray(ACTION_FEATURES),
    )
    summary = {
        "task": task, "path": str(output.resolve()), "episodes": episodes,
        "successful_episodes": len(metadata), "attempts": attempt,
        "timesteps": int(len(actions)), "minimum_episode_length": int(lengths.min()),
        "maximum_episode_length": int(lengths.max()), "mean_episode_length": float(lengths.mean()),
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("tissue", "draw"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--perturbation-m", type=float, default=0.006)
    args = parser.parse_args()
    output = args.output or Path(f"data/{args.task}_demos_50.npz")
    collect(task=args.task, output=output, episodes=args.episodes, seed=args.seed, perturbation_m=args.perturbation_m)


if __name__ == "__main__":
    main()
