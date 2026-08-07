"""Deterministic grasp geometry for the currently supported cube scene."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from robot import Pose


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GraspGeometry:
    pregrasp_height: float = 0.14
    grasp_z_offset: float = 0.005
    lift_height: float = 0.15

    def __post_init__(self) -> None:
        if self.pregrasp_height <= 0.0 or self.lift_height <= 0.0:
            raise ValueError("pregrasp and lift heights must be positive")


@dataclass(frozen=True)
class GraspCandidate:
    object_pose: Pose
    pregrasp_pose: Pose
    grasp_pose: Pose
    lift_pose: Pose

    @property
    def poses(self) -> tuple[Pose, Pose, Pose]:
        return (self.pregrasp_pose, self.grasp_pose, self.lift_pose)


class GraspGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        object_name: str,
        object_pose: Pose,
        gripper_orientation: FloatArray,
    ) -> GraspCandidate | None:
        """Return one candidate, or ``None`` when the object is unsupported."""


class TopDownCubeGraspGenerator(GraspGenerator):
    """One deterministic top-down candidate for the known red cube."""

    def __init__(
        self,
        *,
        geometry: GraspGeometry = GraspGeometry(),
        supported_objects: frozenset[str] = frozenset({"red_cube"}),
    ) -> None:
        self.geometry = geometry
        self.supported_objects = supported_objects

    def generate(
        self,
        object_name: str,
        object_pose: Pose,
        gripper_orientation: FloatArray,
    ) -> GraspCandidate | None:
        if object_name not in self.supported_objects:
            return None
        orientation = np.asarray(gripper_orientation, dtype=np.float64)
        grasp_pose = Pose(
            object_pose.position + np.array([0.0, 0.0, self.geometry.grasp_z_offset]),
            orientation,
        )
        return GraspCandidate(
            object_pose=object_pose,
            pregrasp_pose=Pose(
                object_pose.position
                + np.array([0.0, 0.0, self.geometry.pregrasp_height]),
                orientation,
            ),
            grasp_pose=grasp_pose,
            lift_pose=Pose(
                grasp_pose.position + np.array([0.0, 0.0, self.geometry.lift_height]),
                orientation,
            ),
        )
