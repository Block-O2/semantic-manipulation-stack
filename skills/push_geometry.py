"""Deterministic Cartesian geometry for straight tabletop pushes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from robot import Pose


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PushGeometry:
    """Offsets in metres for a fixed-orientation, horizontal cube push."""

    prepush_height: float = 0.12
    precontact_offset: float = 0.065
    contact_offset: float = 0.020
    # Keep the open finger pad near the cube centre of mass. Higher contacts
    # produced large yaw and lateral ejection for otherwise valid starts.
    contact_height_offset: float = 0.0
    lateral_contact_offset: float = 0.043
    target_inset: float = 0.020
    retreat_height: float = 0.12
    minimum_planned_distance: float = 0.04

    def __post_init__(self) -> None:
        values = (
            self.prepush_height,
            self.precontact_offset,
            self.contact_offset,
            self.retreat_height,
            self.minimum_planned_distance,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("push geometry distances must be positive")
        if (
            self.contact_height_offset < 0.0
            or self.lateral_contact_offset < 0.0
            or self.target_inset < 0.0
        ):
            raise ValueError("push contact and target offsets must be non-negative")
        if self.precontact_offset <= self.contact_offset:
            raise ValueError("precontact_offset must exceed contact_offset")


@dataclass(frozen=True)
class PushCandidate:
    object_pose: Pose
    target_object_position: FloatArray
    direction: FloatArray
    prepush_pose: Pose
    push_start_pose: Pose
    contact_pose: Pose
    end_pose: Pose
    retreat_pose: Pose

    @property
    def poses(self) -> tuple[Pose, ...]:
        return (
            self.prepush_pose,
            self.push_start_pose,
            self.contact_pose,
            self.end_pose,
            self.retreat_pose,
        )


class PushPoseGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        object_name: str,
        target_name: str,
        object_pose: Pose,
        region_lower_xy: FloatArray,
        region_upper_xy: FloatArray,
        gripper_orientation: FloatArray,
    ) -> PushCandidate | None:
        """Return a straight push candidate or ``None`` for invalid geometry."""


class AxisAlignedCubePushPoseGenerator(PushPoseGenerator):
    """Push a cube toward the nearest inset point of an axis-aligned region."""

    def __init__(
        self,
        *,
        geometry: PushGeometry = PushGeometry(),
        supported_objects: frozenset[str] = frozenset(
            {"red_cube", "green_cube", "blue_cube"}
        ),
        supported_targets: frozenset[str] = frozenset(
            {"right_side"}
        ),
    ) -> None:
        self.geometry = geometry
        self.supported_objects = supported_objects
        self.supported_targets = supported_targets

    def generate(
        self,
        object_name: str,
        target_name: str,
        object_pose: Pose,
        region_lower_xy: FloatArray,
        region_upper_xy: FloatArray,
        gripper_orientation: FloatArray,
    ) -> PushCandidate | None:
        if object_name not in self.supported_objects:
            return None
        if target_name not in self.supported_targets:
            return None

        lower = np.asarray(region_lower_xy, dtype=np.float64)
        upper = np.asarray(region_upper_xy, dtype=np.float64)
        if lower.shape != (2,) or upper.shape != (2,) or np.any(lower >= upper):
            return None

        inset_lower = lower + self.geometry.target_inset
        inset_upper = upper - self.geometry.target_inset
        if np.any(inset_lower > inset_upper):
            return None
        start_xy = np.asarray(object_pose.position[:2], dtype=np.float64)
        target_xy = np.clip(start_xy, inset_lower, inset_upper)
        delta = target_xy - start_xy
        distance = float(np.linalg.norm(delta))
        if distance < self.geometry.minimum_planned_distance:
            return None
        direction_xy = delta / distance
        lateral_xy = np.array([-direction_xy[1], direction_xy[0]])
        direction = np.array([direction_xy[0], direction_xy[1], 0.0])
        contact_z = float(
            object_pose.position[2] + self.geometry.contact_height_offset
        )
        orientation = np.asarray(gripper_orientation, dtype=np.float64)

        def pose_at(xy: FloatArray, z: float) -> Pose:
            return Pose(np.array([xy[0], xy[1], z]), orientation)

        lateral_offset = lateral_xy * self.geometry.lateral_contact_offset
        push_start_xy = (
            start_xy
            - direction_xy * self.geometry.precontact_offset
            + lateral_offset
        )
        contact_xy = (
            start_xy - direction_xy * self.geometry.contact_offset + lateral_offset
        )
        end_xy = (
            target_xy - direction_xy * self.geometry.contact_offset + lateral_offset
        )
        push_start_pose = pose_at(push_start_xy, contact_z)
        end_pose = pose_at(end_xy, contact_z)
        return PushCandidate(
            object_pose=object_pose,
            target_object_position=np.array(
                [target_xy[0], target_xy[1], object_pose.position[2]],
                dtype=np.float64,
            ),
            direction=direction,
            prepush_pose=pose_at(
                push_start_xy,
                contact_z + self.geometry.prepush_height,
            ),
            push_start_pose=push_start_pose,
            contact_pose=pose_at(contact_xy, contact_z),
            end_pose=end_pose,
            retreat_pose=pose_at(
                end_xy,
                contact_z + self.geometry.retreat_height,
            ),
        )
