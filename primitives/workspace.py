"""Explicit Cartesian workspace safety limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WorkspaceBounds:
    """Axis-aligned end-effector bounds in world coordinates, in metres."""

    lower: FloatArray
    upper: FloatArray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("workspace bounds must each have shape (3,)")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("workspace bounds must be finite")
        if np.any(lower >= upper):
            raise ValueError("every lower workspace bound must be below its upper bound")
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())

    def contains(self, position: ArrayLike) -> bool:
        point = np.asarray(position, dtype=np.float64)
        return bool(
            point.shape == (3,)
            and np.all(np.isfinite(point))
            and np.all(point >= self.lower)
            and np.all(point <= self.upper)
        )


DEFAULT_WORKSPACE = WorkspaceBounds(
    lower=np.array([-0.35, -0.35, 0.805]),
    upper=np.array([0.35, 0.35, 1.25]),
)
