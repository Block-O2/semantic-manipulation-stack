"""Physical backend contract for the semantic Push capability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from primitives import ManipulationPrimitives, PrimitiveResult
from skills.base import SkillFailure, SkillPhase
from world import WorldModel


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PushRequest:
    """Semantic identifiers required by any physical Push implementation."""

    object_name: str
    target_name: str


@dataclass(frozen=True)
class PushExecutionContext:
    """Trusted robot and read-only world abstractions exposed to a backend."""

    world: WorldModel
    primitives: ManipulationPrimitives


@dataclass(frozen=True)
class PushBackendResult:
    """Outcome of one physical Push attempt, before semantic verification."""

    success: bool
    phase: SkillPhase
    reason: SkillFailure | None
    primitive_result: PrimitiveResult | None
    initial_object_position: FloatArray | None
    control_steps: int
    trace: tuple[str, ...] = ()
    recovery_token: object | None = None

    def __post_init__(self) -> None:
        if self.success != (self.reason is None):
            raise ValueError("successful backend results must not contain a reason")
        if self.control_steps < 0:
            raise ValueError("control_steps must be non-negative")
        if self.initial_object_position is not None:
            position = np.asarray(self.initial_object_position, dtype=np.float64)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError("initial_object_position must be a finite xyz vector")
            object.__setattr__(self, "initial_object_position", position.copy())


@dataclass(frozen=True)
class PushRecoveryResult:
    """Outcome of backend-specific physical preparation for another attempt."""

    success: bool
    phase: SkillPhase
    reason: SkillFailure | None
    primitive_result: PrimitiveResult | None
    control_steps: int
    trace: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.success != (self.reason is None):
            raise ValueError("successful recovery results must not contain a reason")
        if self.control_steps < 0:
            raise ValueError("control_steps must be non-negative")


@runtime_checkable
class PushBackend(Protocol):
    """Specific physical realization of one semantic Push attempt."""

    name: str

    def execute(
        self,
        request: PushRequest,
        context: PushExecutionContext,
    ) -> PushBackendResult:
        """Execute one bounded attempt without deciding semantic success."""

    def recover(
        self,
        request: PushRequest,
        context: PushExecutionContext,
        previous: PushBackendResult,
    ) -> PushRecoveryResult:
        """Prepare for one retry after a recoverable semantic or physical failure."""
