"""Declarative configuration for the first tabletop scene."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SceneConfig:
    """Geometry and placement parameters expressed in MuJoCo world coordinates."""

    table_size: tuple[float, float, float] = (0.8, 0.8, 0.05)
    table_offset: tuple[float, float, float] = (0.0, 0.0, 0.8)
    cube_half_size: float = 0.025
    cube_xy: tuple[float, float] = (0.05, -0.10)
    target_xy: tuple[float, float] = (0.12, 0.14)
    target_half_size: tuple[float, float, float] = (0.09, 0.09, 0.003)
    placement_xy_range: tuple[tuple[float, float], tuple[float, float]] = (
        (-0.12, 0.12),
        (-0.18, 0.02),
    )
    red_rgba: tuple[float, float, float, float] = (0.9, 0.05, 0.05, 1.0)
    blue_rgba: tuple[float, float, float, float] = (0.05, 0.25, 0.9, 0.7)
    table_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    object_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)

    @property
    def table_top_z(self) -> float:
        # TableArena defines table_offset at the top surface, not at the
        # geometric centre of the tabletop slab.
        return self.table_offset[2]

    @property
    def cube_position(self) -> FloatArray:
        return np.array(
            [self.cube_xy[0], self.cube_xy[1], self.table_top_z + self.cube_half_size],
            dtype=np.float64,
        )

    @property
    def target_position(self) -> FloatArray:
        return np.array(
            [
                self.target_xy[0],
                self.target_xy[1],
                self.table_top_z + self.target_half_size[2],
            ],
            dtype=np.float64,
        )


DEFAULT_SCENE = SceneConfig()
