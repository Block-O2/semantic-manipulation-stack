"""Unambiguous feature contracts for state-based Push imitation learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from robot import CartesianCommandEvent, Pose
from world import WorldModel


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PushFeatureSpec:
    names: tuple[str, ...]
    units: tuple[str, ...]
    convention: str

    def __post_init__(self) -> None:
        if not self.names or len(self.names) != len(self.units):
            raise ValueError("feature names and units must have equal non-zero length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("feature names must be unique")

    @property
    def dimension(self) -> int:
        return len(self.names)

    def to_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "units": list(self.units),
            "dimension": self.dimension,
            "convention": self.convention,
        }


PUSH_OBSERVATION_SPEC = PushFeatureSpec(
    names=(
        "ee_x",
        "ee_y",
        "ee_z",
        "cube_x",
        "cube_y",
        "cube_z",
        "target_lower_x",
        "target_lower_y",
        "target_upper_x",
        "target_upper_y",
    ),
    units=("m",) * 10,
    convention=(
        "world-frame absolute Cartesian state; no normalization is stored in "
        "the raw dataset"
    ),
)

PUSH_ACTION_SPEC = PushFeatureSpec(
    names=("desired_ee_x", "desired_ee_y", "desired_ee_z"),
    units=("m", "m", "m"),
    convention=(
        "world-frame absolute desired end-effector xyz for the current 20 Hz "
        "control timestep; orientation is held by the deterministic wrapper"
    ),
)


def encode_push_observation(
    event: CartesianCommandEvent,
    world: WorldModel,
    object_name: str,
    target_name: str,
) -> FloatArray:
    return encode_push_state(
        event.current_pose,
        world,
        object_name,
        target_name,
    )


def encode_push_state(
    end_effector_pose: Pose,
    world: WorldModel,
    object_name: str,
    target_name: str,
) -> FloatArray:
    cube = world.pose(object_name).position
    lower, upper = world.push_region_bounds(target_name)
    observation = np.concatenate(
        [end_effector_pose.position, cube, lower, upper]
    ).astype(np.float64)
    if observation.shape != (PUSH_OBSERVATION_SPEC.dimension,):
        raise RuntimeError("push observation encoder produced an invalid shape")
    return observation


def encode_push_action(event: CartesianCommandEvent) -> FloatArray:
    action = np.asarray(event.target_pose.position, dtype=np.float64).copy()
    if action.shape != (PUSH_ACTION_SPEC.dimension,):
        raise RuntimeError("push action encoder produced an invalid shape")
    return action
