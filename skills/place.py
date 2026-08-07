"""Semantic PlaceSkill with bounded, skill-local release recovery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from primitives import ManipulationPrimitives, PrimitiveFailure, PrimitiveResult
from robot import Pose
from skills.base import Skill, SkillFailure, SkillPhase, SkillResult
from skills.place_geometry import (
    PlaceCandidate,
    PlacePoseGenerator,
    TopDownCubePlacePoseGenerator,
)
from world import WorldModel


@dataclass(frozen=True)
class PlaceConfig:
    max_attempts: int = 2
    opening_steps: int = 40
    settle_steps: int = 30
    target_margin: float = 0.0
    maximum_stable_speed: float = 0.03
    recovery_retreat_height: float = 0.03

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.opening_steps <= 0 or self.settle_steps < 0:
            raise ValueError("gripper steps must be positive and settle steps non-negative")
        if self.target_margin < 0.0:
            raise ValueError("target_margin must be non-negative")
        if self.maximum_stable_speed <= 0.0 or self.recovery_retreat_height <= 0.0:
            raise ValueError("stability and recovery thresholds must be positive")


@dataclass(frozen=True)
class _FailureEvent:
    phase: SkillPhase
    reason: SkillFailure
    primitive_result: PrimitiveResult | None


@dataclass(frozen=True)
class _AttemptOutcome:
    failure: _FailureEvent | None
    primitive_result: PrimitiveResult | None


@dataclass(frozen=True)
class _RecoveryOutcome:
    candidate: PlaceCandidate | None
    failure: _FailureEvent | None


class PlaceSkill(Skill):
    """Place one currently grasped object into one known target region."""

    RECOVERABLE_FAILURES = frozenset(
        {
            SkillFailure.MOTION_TIMEOUT,
            SkillFailure.POSITION_NOT_CONVERGED,
            SkillFailure.ORIENTATION_NOT_CONVERGED,
            SkillFailure.OBJECT_STILL_GRASPED,
        }
    )

    def __init__(
        self,
        object_name: str,
        target_name: str,
        world: WorldModel,
        primitives: ManipulationPrimitives,
        *,
        place_generator: PlacePoseGenerator | None = None,
        config: PlaceConfig = PlaceConfig(),
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.object_name = object_name
        self.target_name = target_name
        self._world = world
        self._primitives = primitives
        self._place_generator = place_generator or TopDownCubePlacePoseGenerator()
        self.config = config
        self._logger = logger
        self._trace: list[str] = []
        self.current_phase = SkillPhase.IDLE

    def _emit(self, message: str) -> None:
        self._trace.append(message)
        if self._logger is not None:
            self._logger(message)

    def _transition(self, phase: SkillPhase) -> None:
        self.current_phase = phase
        self._emit(phase.value)

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
        if self.target_name not in self._world.object_names:
            return self._failure_result(
                _FailureEvent(
                    SkillPhase.CHECK_PRECONDITIONS,
                    SkillFailure.TARGET_NOT_FOUND,
                    None,
                ),
                attempts=0,
            )
        try:
            if not self._world.is_grasped(self.object_name):
                return self._failure_result(
                    _FailureEvent(
                        SkillPhase.CHECK_PRECONDITIONS,
                        SkillFailure.OBJECT_NOT_GRASPED,
                        None,
                    ),
                    attempts=0,
                )
            self._world.pose(self.object_name)
        except (KeyError, ValueError):
            return self._failure_result(
                _FailureEvent(
                    SkillPhase.CHECK_PRECONDITIONS,
                    SkillFailure.OBJECT_POSE_UNAVAILABLE,
                    None,
                ),
                attempts=0,
            )
        try:
            self._world.pose(self.target_name)
            self._world.target_half_extents(self.target_name)
        except (KeyError, ValueError):
            return self._failure_result(
                _FailureEvent(
                    SkillPhase.CHECK_PRECONDITIONS,
                    SkillFailure.TARGET_POSE_UNAVAILABLE,
                    None,
                ),
                attempts=0,
            )
        if hasattr(self._world, "is_target_occupied") and self._world.is_target_occupied(
            self.target_name,
            exclude_object=self.object_name,
        ):
            return self._failure_result(
                _FailureEvent(
                    SkillPhase.CHECK_PRECONDITIONS,
                    SkillFailure.TARGET_OCCUPIED,
                    None,
                ),
                attempts=0,
            )
        return None

    @staticmethod
    def _map_primitive_failure(result: PrimitiveResult) -> SkillFailure:
        mapping = {
            PrimitiveFailure.TARGET_OUTSIDE_WORKSPACE: SkillFailure.TARGET_OUTSIDE_WORKSPACE,
            PrimitiveFailure.TIMEOUT: SkillFailure.MOTION_TIMEOUT,
            PrimitiveFailure.POSITION_NOT_CONVERGED: SkillFailure.POSITION_NOT_CONVERGED,
            PrimitiveFailure.ORIENTATION_NOT_CONVERGED: SkillFailure.ORIENTATION_NOT_CONVERGED,
        }
        return mapping.get(result.reason, SkillFailure.MOTION_TIMEOUT)

    def _primitive_failure(
        self,
        phase: SkillPhase,
        result: PrimitiveResult,
    ) -> _FailureEvent | None:
        if result.success:
            return None
        return _FailureEvent(phase, self._map_primitive_failure(result), result)

    def _generate_candidate(self) -> PlaceCandidate | None:
        try:
            object_pose = self._world.pose(self.object_name)
            target_pose = self._world.pose(self.target_name)
        except (KeyError, ValueError):
            return None
        return self._place_generator.generate(
            self.object_name,
            self.target_name,
            object_pose,
            target_pose,
            self._primitives.current_pose.quaternion,
        )

    def _candidate_failure(self, candidate: PlaceCandidate | None) -> SkillFailure | None:
        if candidate is None:
            return SkillFailure.NO_VALID_PLACE
        if any(not self._primitives.workspace.contains(pose.position) for pose in candidate.poses):
            return SkillFailure.TARGET_UNREACHABLE
        return None

    def _execute_attempt(self, candidate: PlaceCandidate) -> _AttemptOutcome:
        self._transition(SkillPhase.MOVE_ABOVE_TARGET)
        result = self._primitives.move_to_pose(candidate.preplace_pose)
        failure = self._primitive_failure(SkillPhase.MOVE_ABOVE_TARGET, result)
        if failure:
            return _AttemptOutcome(failure, result)

        self._transition(SkillPhase.DESCEND)
        result = self._primitives.move_linear(
            candidate.place_pose,
            max_cartesian_step=0.015,
        )
        failure = self._primitive_failure(SkillPhase.DESCEND, result)
        if failure:
            return _AttemptOutcome(failure, result)

        self._transition(SkillPhase.OPEN_GRIPPER)
        result = self._primitives.open_gripper(steps=self.config.opening_steps)
        failure = self._primitive_failure(SkillPhase.OPEN_GRIPPER, result)
        if failure:
            return _AttemptOutcome(failure, result)

        self._transition(SkillPhase.VERIFY_RELEASE)
        result = self._primitives.wait(steps=self.config.settle_steps)
        failure = self._primitive_failure(SkillPhase.VERIFY_RELEASE, result)
        if failure:
            return _AttemptOutcome(failure, result)
        if self._world.is_grasped(self.object_name):
            return _AttemptOutcome(
                _FailureEvent(
                    SkillPhase.VERIFY_RELEASE,
                    SkillFailure.OBJECT_STILL_GRASPED,
                    result,
                ),
                result,
            )

        self._transition(SkillPhase.RETREAT)
        result = self._primitives.move_linear(
            candidate.retreat_pose,
            max_cartesian_step=0.015,
        )
        failure = self._primitive_failure(SkillPhase.RETREAT, result)
        if failure:
            return _AttemptOutcome(failure, result)

        self._transition(SkillPhase.VERIFY_SUCCESS)
        try:
            if self._world.is_grasped(self.object_name):
                reason = SkillFailure.OBJECT_STILL_GRASPED
            elif not self._world.is_inside_target(
                self.object_name,
                self.target_name,
                margin=self.config.target_margin,
            ):
                reason = SkillFailure.OBJECT_OUTSIDE_TARGET
            elif not self._world.is_stable(
                self.object_name,
                maximum_speed=self.config.maximum_stable_speed,
            ):
                reason = SkillFailure.OBJECT_UNSTABLE
            else:
                reason = None
        except (KeyError, ValueError):
            reason = SkillFailure.OBJECT_POSE_UNAVAILABLE
        if reason is not None:
            return _AttemptOutcome(
                _FailureEvent(SkillPhase.VERIFY_SUCCESS, reason, result),
                result,
            )
        return _AttemptOutcome(None, result)

    def _recover(self) -> _RecoveryOutcome:
        self._transition(SkillPhase.RECOVERY)

        self._transition(SkillPhase.RECOVERY_RETREAT_SLIGHTLY)
        current = self._primitives.current_pose
        retreat = Pose(
            current.position + np.array([0.0, 0.0, self.config.recovery_retreat_height]),
            current.quaternion,
        )
        if not self._primitives.workspace.contains(retreat.position):
            return _RecoveryOutcome(
                None,
                _FailureEvent(
                    SkillPhase.RECOVERY_RETREAT_SLIGHTLY,
                    SkillFailure.RECOVERY_FAILED,
                    None,
                ),
            )
        result = self._primitives.move_linear(retreat, max_cartesian_step=0.015)
        if not result.success:
            return _RecoveryOutcome(
                None,
                _FailureEvent(
                    SkillPhase.RECOVERY_RETREAT_SLIGHTLY,
                    SkillFailure.RECOVERY_FAILED,
                    result,
                ),
            )

        self._transition(SkillPhase.RECOVERY_REFRESH_WORLD_STATE)
        if self.object_name not in self._world.object_names:
            return _RecoveryOutcome(
                None,
                _FailureEvent(
                    SkillPhase.RECOVERY_REFRESH_WORLD_STATE,
                    SkillFailure.OBJECT_NOT_FOUND,
                    None,
                ),
            )
        if self.target_name not in self._world.object_names:
            return _RecoveryOutcome(
                None,
                _FailureEvent(
                    SkillPhase.RECOVERY_REFRESH_WORLD_STATE,
                    SkillFailure.TARGET_NOT_FOUND,
                    None,
                ),
            )
        try:
            if not self._world.is_grasped(self.object_name):
                return _RecoveryOutcome(
                    None,
                    _FailureEvent(
                        SkillPhase.RECOVERY_REFRESH_WORLD_STATE,
                        SkillFailure.OBJECT_NOT_GRASPED,
                        None,
                    ),
                )
            self._world.pose(self.object_name)
            self._world.pose(self.target_name)
        except (KeyError, ValueError):
            return _RecoveryOutcome(
                None,
                _FailureEvent(
                    SkillPhase.RECOVERY_REFRESH_WORLD_STATE,
                    SkillFailure.OBJECT_POSE_UNAVAILABLE,
                    None,
                ),
            )

        self._transition(SkillPhase.RECOVERY_RECOMPUTE_PLACE)
        candidate = self._generate_candidate()
        reason = self._candidate_failure(candidate)
        if reason is not None:
            return _RecoveryOutcome(
                None,
                _FailureEvent(SkillPhase.RECOVERY_RECOMPUTE_PLACE, reason, None),
            )
        return _RecoveryOutcome(candidate, None)

    def execute(self) -> SkillResult:
        self._trace = []
        self.current_phase = SkillPhase.IDLE
        self._emit(f"PlaceSkill({self.object_name}, {self.target_name})")

        precondition_failure = self.check_preconditions()
        if precondition_failure is not None:
            return precondition_failure

        candidate: PlaceCandidate | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            self._emit(f"[attempt {attempt}]")
            self._transition(SkillPhase.GENERATE_PLACE_POSE)
            if candidate is None:
                candidate = self._generate_candidate()
            candidate_reason = self._candidate_failure(candidate)
            if candidate_reason is not None:
                return self._failure_result(
                    _FailureEvent(
                        SkillPhase.GENERATE_PLACE_POSE,
                        candidate_reason,
                        None,
                    ),
                    attempts=attempt,
                )
            assert candidate is not None

            outcome = self._execute_attempt(candidate)
            if outcome.failure is None:
                return self._success_result(
                    attempts=attempt,
                    primitive_result=outcome.primitive_result,
                )

            self._emit(f"ATTEMPT_FAILED: {outcome.failure.reason.value}")
            if (
                outcome.failure.reason not in self.RECOVERABLE_FAILURES
                or attempt >= self.config.max_attempts
            ):
                return self._failure_result(outcome.failure, attempts=attempt)

            recovery = self._recover()
            if recovery.failure is not None:
                return self._failure_result(recovery.failure, attempts=attempt)
            candidate = recovery.candidate

        raise RuntimeError("PlaceSkill attempt loop ended unexpectedly")
