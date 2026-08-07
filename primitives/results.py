"""Structured outcomes shared by manipulation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PrimitiveFailure(str, Enum):
    TARGET_OUTSIDE_WORKSPACE = "TARGET_OUTSIDE_WORKSPACE"
    TIMEOUT = "TIMEOUT"
    POSITION_NOT_CONVERGED = "POSITION_NOT_CONVERGED"
    ORIENTATION_NOT_CONVERGED = "ORIENTATION_NOT_CONVERGED"


@dataclass(frozen=True)
class PrimitiveResult:
    success: bool
    reason: PrimitiveFailure | None
    steps: int
    final_position_error: float | None
    final_orientation_error: float | None
