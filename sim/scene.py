"""Declarative configuration for the semantic tabletop playground."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SceneConfig:
    """Geometry and deterministic placements in MuJoCo world coordinates."""

    table_size: tuple[float, float, float] = (0.8, 0.8, 0.05)
    table_offset: tuple[float, float, float] = (0.0, 0.0, 0.8)
    cube_half_size: float = 0.025
    red_cube_xy: tuple[float, float] = (0.05, -0.10)
    green_cube_xy: tuple[float, float] = (-0.12, -0.02)
    blue_cube_xy: tuple[float, float] = (0.15, -0.12)
    blue_target_xy: tuple[float, float] = (0.12, 0.14)
    red_target_xy: tuple[float, float] = (-0.14, 0.16)
    temporary_area_xy: tuple[float, float] = (0.0, 0.28)
    target_half_size: tuple[float, float, float] = (0.075, 0.075, 0.003)
    placement_xy_range: tuple[tuple[float, float], tuple[float, float]] = (
        (-0.18, 0.18),
        (-0.18, 0.04),
    )
    minimum_random_cube_separation: float = 0.075
    red_rgba: tuple[float, float, float, float] = (0.9, 0.05, 0.05, 1.0)
    green_rgba: tuple[float, float, float, float] = (0.05, 0.75, 0.15, 1.0)
    blue_cube_rgba: tuple[float, float, float, float] = (0.05, 0.25, 0.9, 1.0)
    blue_target_rgba: tuple[float, float, float, float] = (0.05, 0.25, 0.9, 0.55)
    red_target_rgba: tuple[float, float, float, float] = (0.9, 0.05, 0.05, 0.55)
    temporary_area_rgba: tuple[float, float, float, float] = (0.85, 0.7, 0.1, 0.45)
    table_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    object_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)

    @property
    def table_top_z(self) -> float:
        return self.table_offset[2]

    @property
    def cube_names(self) -> tuple[str, ...]:
        return ("red_cube", "green_cube", "blue_cube")

    @property
    def target_names(self) -> tuple[str, ...]:
        return ("red_target", "blue_target", "temporary_area")

    @property
    def temporary_target_names(self) -> tuple[str, ...]:
        """Targets preferred for reversible staging, but usable by normal Place."""

        return ("temporary_area",)

    @property
    def push_region_bounds(
        self,
    ) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
        """Axis-aligned semantic regions expressed as lower / upper XY bounds."""

        return {
            # A named semantic strip, inset from the physical table edge.
            "right_side": ((0.18, -0.25), (0.34, 0.25)),
        }

    @property
    def push_region_names(self) -> tuple[str, ...]:
        return tuple(self.push_region_bounds)

    def push_region_center(self, name: str) -> FloatArray:
        lower, upper = self.push_region_bounds[name]
        return np.array(
            [
                0.5 * (lower[0] + upper[0]),
                0.5 * (lower[1] + upper[1]),
                self.table_top_z + self.cube_half_size,
            ],
            dtype=np.float64,
        )

    @property
    def cube_xy_positions(self) -> dict[str, tuple[float, float]]:
        return {
            "red_cube": self.red_cube_xy,
            "green_cube": self.green_cube_xy,
            "blue_cube": self.blue_cube_xy,
        }

    @property
    def target_xy_positions(self) -> dict[str, tuple[float, float]]:
        return {
            "red_target": self.red_target_xy,
            "blue_target": self.blue_target_xy,
            "temporary_area": self.temporary_area_xy,
        }

    @property
    def cube_rgba(self) -> dict[str, tuple[float, float, float, float]]:
        return {
            "red_cube": self.red_rgba,
            "green_cube": self.green_rgba,
            "blue_cube": self.blue_cube_rgba,
        }

    @property
    def target_rgba(self) -> dict[str, tuple[float, float, float, float]]:
        return {
            "red_target": self.red_target_rgba,
            "blue_target": self.blue_target_rgba,
            "temporary_area": self.temporary_area_rgba,
        }

    def cube_position(self, name: str) -> FloatArray:
        xy = self.cube_xy_positions[name]
        return np.array(
            [xy[0], xy[1], self.table_top_z + self.cube_half_size],
            dtype=np.float64,
        )

    def target_position(self, name: str) -> FloatArray:
        xy = self.target_xy_positions[name]
        return np.array(
            [xy[0], xy[1], self.table_top_z + self.target_half_size[2]],
            dtype=np.float64,
        )


DEFAULT_SCENE = SceneConfig()
