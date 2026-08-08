from __future__ import annotations

import numpy as np

from datasets import (
    PUSH_ACTION_SPEC,
    PUSH_OBSERVATION_SPEC,
    PushDataset,
    PushEpisode,
    PushTrajectoryRecorder,
    get_action_chunk,
    load_push_dataset,
    save_push_dataset,
    split_episode_indices,
    validate_push_dataset,
)
from robot import CartesianCommandEvent, Pose


def make_episode(identifier: str, length: int = 4) -> PushEpisode:
    observations = np.zeros((length, 10), dtype=np.float64)
    observations[:, 2] = 1.0
    observations[:, 5] = 0.825
    observations[:, 6:] = np.array([0.18, -0.25, 0.34, 0.25])
    actions = np.tile(np.array([0.0, 0.0, 1.0]), (length, 1))
    return PushEpisode(
        observations,
        actions,
        {
            "episode_id": identifier,
            "success": True,
            "target_reached": True,
            "total_displacement": 0.2,
        },
    )


class FakeWorld:
    def pose(self, name: str) -> Pose:
        return Pose.from_values([0.0, -0.1, 0.825], [0.0, 0.0, 0.0, 1.0])

    def push_region_bounds(self, name: str):
        return np.array([0.18, -0.25]), np.array([0.34, 0.25])


class FakeRobot:
    def __init__(self) -> None:
        self.observer = None

    def add_command_observer(self, observer) -> None:
        self.observer = observer

    def remove_command_observer(self, observer) -> None:
        assert observer == self.observer
        self.observer = None


def test_feature_contract_and_control_timestep_recording() -> None:
    assert PUSH_OBSERVATION_SPEC.dimension == 10
    assert PUSH_ACTION_SPEC.dimension == 3
    assert PUSH_OBSERVATION_SPEC.names[:3] == ("ee_x", "ee_y", "ee_z")
    assert PUSH_ACTION_SPEC.names == (
        "desired_ee_x",
        "desired_ee_y",
        "desired_ee_z",
    )

    robot = FakeRobot()
    recorder = PushTrajectoryRecorder(FakeWorld(), "red_cube", "right_side")
    recorder.attach(robot)
    current = Pose.from_values([-0.2, -0.2, 1.0], [0.0, 0.0, 0.0, 1.0])
    target = Pose.from_values([-0.18, -0.2, 1.0], [0.0, 0.0, 0.0, 1.0])
    robot.observer(CartesianCommandEvent(0, current, target, -1.0))
    recorder.detach()
    episode = recorder.episode(
        {"success": True, "target_reached": True, "total_displacement": 0.2}
    )

    assert episode.length == 1
    np.testing.assert_allclose(episode.observations[0, :3], current.position)
    np.testing.assert_allclose(episode.actions[0], target.position)


def test_npz_round_trip_and_validation(tmp_path) -> None:
    dataset = PushDataset((make_episode("a", 3), make_episode("b", 5)))
    path = tmp_path / "dataset.npz"

    save_push_dataset(path, dataset)
    loaded = load_push_dataset(path)
    report = validate_push_dataset(loaded)

    assert [episode.length for episode in loaded.episodes] == [3, 5]
    assert loaded.episodes[1].metadata["episode_id"] == "b"
    assert report.valid
    assert report.statistics["total_timesteps"] == 8


def test_dataset_validation_rejects_nonfinite_and_unsuccessful_episode() -> None:
    episode = make_episode("invalid", 2)
    episode.observations[0, 0] = np.nan
    episode.actions[0, 0] = 2.0
    episode.metadata["success"] = False
    episode.metadata["target_reached"] = False
    episode.metadata["total_displacement"] = 0.0

    report = validate_push_dataset(PushDataset((episode,)))

    assert not report.valid
    assert "NONFINITE_OBSERVATION" in report.errors
    assert "ACTION_OUTSIDE_WORKSPACE" in report.errors
    assert "EPISODE_0_NOT_SUCCESSFUL" in report.errors
    assert "EPISODE_0_TARGET_NOT_REACHED" in report.errors


def test_action_chunk_uses_last_action_padding_and_mask() -> None:
    episode = make_episode("chunk", 3)
    episode.actions[:] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    chunk, mask = get_action_chunk(episode, 1, 4)

    np.testing.assert_allclose(
        chunk,
        np.array([[4, 5, 6], [7, 8, 9], [7, 8, 9], [7, 8, 9]]),
    )
    np.testing.assert_array_equal(mask, [True, True, False, False])


def test_train_validation_split_is_by_episode_and_deterministic() -> None:
    first = split_episode_indices(10, validation_fraction=0.2, seed=9)
    second = split_episode_indices(10, validation_fraction=0.2, seed=9)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert set(first[0]).isdisjoint(first[1])
    assert len(first[0]) == 8
    assert len(first[1]) == 2
