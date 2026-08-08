"""Semantic PushSkill orchestration independent of physical backend choice."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from primitives import ManipulationPrimitives, PrimitiveResult
from skills.base import Skill, SkillFailure, SkillPhase, SkillResult
from skills.classical_push_backend import ClassicalPushBackend, ClassicalPushConfig
from skills.push_backend import (
    PushBackend,
    PushBackendResult,
    PushExecutionContext,
    PushRequest,
)
from skills.push_geometry import PushPoseGenerator
from world import WorldModel


@dataclass(frozen=True)
class PushConfig:
    max_attempts: int = 2
    opening_steps: int = 35
    settle_steps: int = 20
    minimum_displacement: float = 0.06
    contact_proximity: float = 0.07
    table_height_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.opening_steps <= 0 or self.settle_steps < 0:
            raise ValueError("push gripper steps must be positive and settle non-negative")
        if (
            self.minimum_displacement <= 0.0
            or self.contact_proximity <= 0.0
            or self.table_height_tolerance <= 0.0
        ):
            raise ValueError("push verification thresholds must be positive")


@dataclass(frozen=True)
class _FailureEvent:
    phase: SkillPhase
    reason: SkillFailure
    primitive_result: PrimitiveResult | None


def evaluate_push_outcome(
    initial_position: np.ndarray,
    final_position: np.ndarray,
    *,
    minimum_displacement: float,
    target_reached: bool,
    object_on_table: bool,
) -> SkillFailure | None:
    """Evaluate semantic push success independently of backend execution."""

    if not object_on_table:
        return SkillFailure.OBJECT_LEFT_WORKSPACE
    displacement = float(
        np.linalg.norm(
            np.asarray(final_position, dtype=np.float64)[:2]
            - np.asarray(initial_position, dtype=np.float64)[:2]
        )
    )
    if displacement < minimum_displacement:
        return SkillFailure.INSUFFICIENT_DISPLACEMENT
    if not target_reached:
        return SkillFailure.PUSH_TARGET_NOT_REACHED
    return None


class PushSkill(Skill):
    """Own Push semantics, retry policy, and backend-independent results."""

    RECOVERABLE_FAILURES = frozenset(
        {
            SkillFailure.NO_PUSH_CONTACT,
            SkillFailure.INSUFFICIENT_DISPLACEMENT,
            SkillFailure.PUSH_TARGET_NOT_REACHED,
        }
    )

    def __init__(
        self,
        object_name: str,
        target_name: str,
        world: WorldModel,
        primitives: ManipulationPrimitives,
        *,
        backend: PushBackend | None = None,
        push_generator: PushPoseGenerator | None = None,
        config: PushConfig = PushConfig(),
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.object_name = object_name
        self.target_name = target_name
        self._world = world
        self._primitives = primitives
        self.config = config
        self._backend = backend or ClassicalPushBackend(
            push_generator=push_generator,
            config=ClassicalPushConfig(
                opening_steps=config.opening_steps,
                settle_steps=config.settle_steps,
                contact_proximity=config.contact_proximity,
            ),
        )
        self._logger = logger
        self._trace: list[str] = []
        self.current_phase = SkillPhase.IDLE

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def _emit(self, message: str) -> None:
        self._trace.append(message)
        if self._logger is not None:
            self._logger(message)

    def _transition(self, phase: SkillPhase) -> None:
        self.current_phase = phase
        self._emit(phase.value)

    def _append_backend_trace(self, result: PushBackendResult) -> None:
        self._trace.extend(result.trace)
        self.current_phase = result.phase

    def _failure_result(self, failure: _FailureEvent, *, attempts: int) -> SkillResult:
        self.current_phase = failure.phase
        self._emit(f"FAILED: {failure.reason.value}")
        return SkillResult(
            success=False,
            skill=type(self).__name__,
            object_name=self.object_name,
            phase=failure.phase,
            reason=failure.reason,
            attempts=attempts,
            primitive_result=failure.primitive_result,
            trace=tuple(self._trace),
        )

    def _success_result(
        self,
        *,
        attempts: int,
        primitive_result: PrimitiveResult | None,
    ) -> SkillResult:
        self._transition(SkillPhase.SUCCESS)
        return SkillResult(
            success=True,
            skill=type(self).__name__,
            object_name=self.object_name,
            phase=SkillPhase.SUCCESS,
            reason=None,
            attempts=attempts,
            primitive_result=primitive_result,
            trace=tuple(self._trace),
        )

    def check_preconditions(self) -> SkillResult | None:
        self._transition(SkillPhase.CHECK_PRECONDITIONS)
        if self.object_name not in self._world.object_names or (
            hasattr(self._world, "exists") and not self._world.exists(self.object_name)
        ):
            return self._failure_result(
                _FailureEvent(
                    SkillPhase.CHECK_PRECONDITIONS,
                    SkillFailure.OBJECT_NOT_FOUND,
                    None,
                ),
                attempts=0,
            )
        if self.target_name not in self._world.push_region_names:
            return self._failure_result(
                _FailureEvent(
                    SkillPhase.CHECK_PRECONDITIONS,
                    SkillFailure.INVALID_PUSH_TARGET,
                    None,
                ),
                attempts=0,
            )
        if self._world.holding_object() is not None:
            return self._failure_result(
                _FailureEvent(
                    SkillPhase.CHECK_PRECONDITIONS,
                    SkillFailure.GRIPPER_NOT_EMPTY,
                    None,
                ),
                attempts=0,
            )
        try:
            if not self._world.is_reachable(self.object_name):
                reason = SkillFailure.OBJECT_UNREACHABLE
            else:
                self._world.pose(self.object_name)
                self._world.push_region_bounds(self.target_name)
                reason = None
        except (KeyError, ValueError):
            reason = SkillFailure.OBJECT_POSE_UNAVAILABLE
        if reason is not None:
            return self._failure_result(
                _FailureEvent(SkillPhase.CHECK_PRECONDITIONS, reason, None),
                attempts=0,
            )
        return None

    def _semantic_failure(
        self,
        backend_result: PushBackendResult,
    ) -> SkillFailure | None:
        initial = backend_result.initial_object_position
        if initial is None:
            return SkillFailure.OBJECT_POSE_UNAVAILABLE
        self._transition(SkillPhase.VERIFY_SUCCESS)
        try:
            final_position = self._world.pose(self.object_name).position
            return evaluate_push_outcome(
                initial,
                final_position,
                minimum_displacement=self.config.minimum_displacement,
                target_reached=self._world.is_inside_push_region(
                    self.object_name,
                    self.target_name,
                ),
                object_on_table=self._world.is_on_table(
                    self.object_name,
                    height_tolerance=self.config.table_height_tolerance,
                ),
            )
        except (KeyError, ValueError):
            return SkillFailure.OBJECT_POSE_UNAVAILABLE

    def execute(self) -> SkillResult:
        self._trace = []
        self.current_phase = SkillPhase.IDLE
        self._emit(
            f"PushSkill({self.object_name}, {self.target_name}, "
            f"backend={self._backend.name})"
        )

        precondition_failure = self.check_preconditions()
        if precondition_failure is not None:
            return precondition_failure

        request = PushRequest(self.object_name, self.target_name)
        context = PushExecutionContext(self._world, self._primitives)
        for attempt in range(1, self.config.max_attempts + 1):
            self._emit(f"[attempt {attempt}]")
            backend_result = self._backend.execute(request, context)
            self._append_backend_trace(backend_result)
            if backend_result.success:
                reason = self._semantic_failure(backend_result)
                phase = SkillPhase.VERIFY_SUCCESS
            else:
                assert backend_result.reason is not None
                reason = backend_result.reason
                phase = backend_result.phase

            if reason is None:
                return self._success_result(
                    attempts=attempt,
                    primitive_result=backend_result.primitive_result,
                )

            self._emit(f"ATTEMPT_FAILED: {reason.value}")
            if reason not in self.RECOVERABLE_FAILURES or attempt >= self.config.max_attempts:
                return self._failure_result(
                    _FailureEvent(phase, reason, backend_result.primitive_result),
                    attempts=attempt,
                )

            recovery = self._backend.recover(request, context, backend_result)
            self._trace.extend(recovery.trace)
            self.current_phase = recovery.phase
            if not recovery.success:
                return self._failure_result(
                    _FailureEvent(
                        recovery.phase,
                        recovery.reason or SkillFailure.RECOVERY_FAILED,
                        recovery.primitive_result,
                    ),
                    attempts=attempt,
                )

        raise RuntimeError("PushSkill attempt loop ended unexpectedly")
