"""Deterministic placement geometry for the current cube and target scene."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from robot import Pose


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PlaceGeometry:
    """Top-down geometry relative to the target-region body origin."""

    preplace_height: float = 0.14
    place_z_offset: float = 0.027
    retreat_height: float = 0.14

    def __post_init__(self) -> None:
        if (
            self.preplace_height <= 0.0
            or self.place_z_offset <= 0.0
            or self.retreat_height <= 0.0
        ):
            raise ValueError("place geometry offsets must be positive")


@dataclass(frozen=True)
class PlaceCandidate:
    object_pose: Pose
    target_pose: Pose
    preplace_pose: Pose
    place_pose: Pose
    retreat_pose: Pose

    @property
    def poses(self) -> tuple[Pose, Pose, Pose]:
        return (self.preplace_pose, self.place_pose, self.retreat_pose)


class PlacePoseGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        object_name: str,
        target_name: str,
        object_pose: Pose,
        target_pose: Pose,
        gripper_orientation: FloatArray,
    ) -> PlaceCandidate | None:
        """Return one candidate or ``None`` for unsupported semantics."""


class TopDownCubePlacePoseGenerator(PlacePoseGenerator):
    """One fixed-orientation top-down candidate at the target centre."""

    def __init__(
        self,
        *,
        geometry: PlaceGeometry = PlaceGeometry(),
        supported_objects: frozenset[str] = frozenset({"red_cube"}),
        supported_targets: frozenset[str] = frozenset({"blue_target"}),
    ) -> None:
        self.geometry = geometry
        self.supported_objects = supported_objects
        self.supported_targets = supported_targets

    def generate(
        self,
        object_name: str,
        target_name: str,
        object_pose: Pose,
        target_pose: Pose,
        gripper_orientation: FloatArray,
    ) -> PlaceCandidate | None:
        if object_name not in self.supported_objects or target_name not in self.supported_targets:
            return None
        orientation = np.asarray(gripper_orientation, dtype=np.float64)
        place_pose = Pose(
            target_pose.position + np.array([0.0, 0.0, self.geometry.place_z_offset]),
            orientation,
        )
        return PlaceCandidate(
            object_pose=object_pose,
            target_pose=target_pose,
            preplace_pose=Pose(
                place_pose.position
                + np.array([0.0, 0.0, self.geometry.preplace_height]),
                orientation,
            ),
            place_pose=place_pose,
            retreat_pose=Pose(
                place_pose.position + np.array([0.0, 0.0, self.geometry.retreat_height]),
                orientation,
            ),
        )
