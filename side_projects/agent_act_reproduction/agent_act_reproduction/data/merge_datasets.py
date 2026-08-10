"""Merge episode-complete NPZ shards produced by collect_tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def merge(inputs: list[Path], output: Path) -> None:
    archives = [np.load(path, allow_pickle=False) for path in inputs]
    try:
        states = []
        environments = []
        actions = []
        episode_ids = []
        lengths = []
        metadata = []
        episode_offset = 0
        for archive in archives:
            states.append(np.asarray(archive["observation_state"]))
            environments.append(np.asarray(archive["observation_environment_state"]))
            actions.append(np.asarray(archive["action"]))
            episode_ids.append(np.asarray(archive["episode_id"]) + episode_offset)
            shard_lengths = np.asarray(archive["episode_lengths"])
            lengths.append(shard_lengths)
            for item in json.loads(str(archive["metadata_json"])):
                copied = dict(item)
                copied["episode_index"] = len(metadata)
                metadata.append(copied)
            episode_offset += len(shard_lengths)
        output.parent.mkdir(parents=True, exist_ok=True)
        first = archives[0]
        np.savez_compressed(
            output,
            observation_state=np.concatenate(states),
            observation_environment_state=np.concatenate(environments),
            action=np.concatenate(actions),
            episode_id=np.concatenate(episode_ids),
            episode_lengths=np.concatenate(lengths),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            robot_state_features=first["robot_state_features"],
            environment_state_features=first["environment_state_features"],
            action_features=first["action_features"],
        )
    finally:
        for archive in archives:
            archive.close()
    print(json.dumps({"output": str(output.resolve()), "episodes": len(metadata), "timesteps": sum(int(array.shape[0]) for array in actions)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    merge(args.inputs, args.output)


if __name__ == "__main__":
    main()
