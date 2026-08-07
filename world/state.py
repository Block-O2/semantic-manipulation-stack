"""Serializable task-level world state with no simulator implementation details."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robot import Pose


@dataclass(frozen=True)
class SemanticThresholds:
    """Simple geometric thresholds used for deterministic semantic relations."""

    near_distance: float = 0.12
    left_right_tolerance: float = 0.02
    inside_margin: float = 0.0

    def __post_init__(self) -> None:
        if self.near_distance <= 0.0:
            raise ValueError("near_distance must be positive")
        if self.left_right_tolerance < 0.0 or self.inside_margin < 0.0:
            raise ValueError("relation tolerances must be non-negative")


@dataclass(frozen=True)
class PoseState:
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]

    @classmethod
    def from_pose(cls, pose: Pose) -> "PoseState":
        return cls(
            tuple(float(value) for value in pose.position),
            tuple(float(value) for value in pose.quaternion),
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "position": list(self.position),
            "quaternion": list(self.quaternion),
        }


@dataclass(frozen=True)
class RobotState:
    holding: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {"holding": self.holding}


@dataclass(frozen=True)
class ObjectState:
    exists: bool
    pose: PoseState | None
    grasped: bool
    reachable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "pose": self.pose.to_dict() if self.pose else None,
            "grasped": self.grasped,
            "reachable": self.reachable,
        }


@dataclass(frozen=True)
class TargetState:
    exists: bool
    pose: PoseState | None
    reachable: bool
    occupied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "pose": self.pose.to_dict() if self.pose else None,
            "reachable": self.reachable,
            "occupied": self.occupied,
        }


@dataclass(frozen=True)
class WorldState:
    """Semantic snapshot shared by deterministic code and planner prompts."""

    robot: RobotState
    objects: dict[str, ObjectState]
    targets: dict[str, TargetState]
    relations: dict[str, bool]

    @staticmethod
    def relation_key(object_name: str, target_name: str) -> str:
        return f"{object_name}_inside_{target_name}"

    @staticmethod
    def left_of_key(first: str, second: str) -> str:
        return f"{first}_left_of_{second}"

    @staticmethod
    def right_of_key(first: str, second: str) -> str:
        return f"{first}_right_of_{second}"

    @staticmethod
    def near_key(first: str, second: str) -> str:
        return f"{first}_near_{second}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot": self.robot.to_dict(),
            "objects": {
                name: state.to_dict() for name, state in sorted(self.objects.items())
            },
            "targets": {
                name: state.to_dict() for name, state in sorted(self.targets.items())
            },
            "relations": dict(sorted(self.relations.items())),
        }

    def value(self, path: str) -> Any:
        """Resolve one whitelisted semantic path used by skill contracts."""

        parts = path.split(".")
        if parts == ["robot", "holding"]:
            return self.robot.holding
        if len(parts) == 3 and parts[0] == "objects":
            state = self.objects.get(parts[1])
            if state is None or parts[2] not in {"exists", "grasped", "reachable"}:
                raise KeyError(path)
            return getattr(state, parts[2])
        if len(parts) == 3 and parts[0] == "targets":
            state = self.targets.get(parts[1])
            if state is None or parts[2] not in {"exists", "reachable", "occupied"}:
                raise KeyError(path)
            return getattr(state, parts[2])
        if len(parts) == 2 and parts[0] == "relations" and parts[1] in self.relations:
            return self.relations[parts[1]]
        raise KeyError(path)

    def flattened(self) -> dict[str, Any]:
        values: dict[str, Any] = {"robot.holding": self.robot.holding}
        for name, state in self.objects.items():
            values[f"objects.{name}.exists"] = state.exists
            values[f"objects.{name}.grasped"] = state.grasped
            values[f"objects.{name}.reachable"] = state.reachable
        for name, state in self.targets.items():
            values[f"targets.{name}.exists"] = state.exists
            values[f"targets.{name}.reachable"] = state.reachable
            values[f"targets.{name}.occupied"] = state.occupied
        for name, value in self.relations.items():
            values[f"relations.{name}"] = value
        return values
