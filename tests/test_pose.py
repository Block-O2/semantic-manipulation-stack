import numpy as np
import pytest

from robot.panda import Pose


def test_pose_normalizes_quaternion_and_copies_inputs() -> None:
    position = np.array([1.0, 2.0, 3.0])
    quaternion = np.array([0.0, 0.0, 0.0, 2.0])

    pose = Pose(position, quaternion)
    position[0] = 99.0

    np.testing.assert_allclose(pose.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(pose.quaternion, [0.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    ("position", "quaternion"),
    [
        ([0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_pose_rejects_invalid_values(position, quaternion) -> None:
    with pytest.raises(ValueError):
        Pose.from_values(position, quaternion)
