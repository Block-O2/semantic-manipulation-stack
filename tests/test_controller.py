import numpy as np

from robot.controller import CartesianController
from robosuite.utils import transform_utils as T


def test_orientation_error_is_zero_for_same_rotation() -> None:
    quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        CartesianController.orientation_error(quaternion, quaternion),
        np.zeros(3),
        atol=1e-10,
    )


def test_orientation_error_uses_world_axis() -> None:
    target = T.axisangle2quat(np.array([0.0, 0.0, np.pi / 4.0]))
    error = CartesianController.orientation_error(target, [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(error, [0.0, 0.0, np.pi / 4.0], atol=1e-7)
