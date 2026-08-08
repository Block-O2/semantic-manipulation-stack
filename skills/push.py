"""Classical Cartesian PushSkill with bounded local contact recovery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from primitives import ManipulationPrimitives, PrimitiveFailure, PrimitiveResult
from robot import Pose
from skills.base import Skill, SkillFailure, SkillPhase, SkillResult
from skills.push_geometry import (
    AxisAlignedCubePushPoseGenerator,
    PushCandidate,
    PushPoseGenerator,
)
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


@dataclass(frozen=True)
class _AttemptOutcome:
    failure: _FailureEvent | None
    primitive_result: PrimitiveResult | None


@dataclass(frozen=True)
class _RecoveryOutcome:
    candidate: PushCandidate | None
    failure: _FailureEvent | None


def evaluate_push_outcome(
    initial_position: np.ndarray,
    final_position: np.ndarray,
    *,
    minimum_displacement: float,
    target_reached: bool,
    object_on_table: bool,
) -> SkillFailure | None:
    """Evaluate physical push success independently of robot trajectory success."""

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
    """Push one cube into a named tabletop region using existing primitives."""

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
        push_generator: PushPoseGenerator | None = None,
        config: PushConfig = PushConfig(),
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.object_name = object_name
        self.target_name = target_name
        self._world = world
        self._primitives = primitives
        self._push_generator = push_generator or AxisAlignedCubePushPoseGenerator()
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

    def _emit_object_position(self, label: str) -> None:
        try:
            position = self._world.pose(self.object_name).position.tolist()
        except (KeyError, ValueError):
            self._emit(f"OBJECT_POSITION[{label}]: unavailable")
            return
        self._emit(f"OBJECT_POSITION[{label}]: {position}")

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

    @staticmethod
    def _map_primitive_failure(result: PrimitiveResult) -> SkillFailure:
        mapping = {
            PrimitiveFailure.TARGET_OUTSIDE_WORKSPACE: (
                SkillFailure.PUSH_PATH_OUTSIDE_WORKSPACE
            ),
            PrimitiveFailure.TIMEOUT: SkillFailure.MOTION_TIMEOUT,
            PrimitiveFailure.POSITION_NOT_CONVERGED: (
                SkillFailure.POSITION_NOT_CONVERGED
            ),
            PrimitiveFailure.ORIENTATION_NOT_CONVERGED: (
                SkillFailure.ORIENTATION_NOT_CONVERGED
            ),
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

    def _generate_candidate(self) -> PushCandidate | None:
        try:
            object_pose = self._world.pose(self.object_name)
            lower, upper = self._world.push_region_bounds(self.target_name)
        except (KeyError, ValueError):
            return None
        return self._push_generator.generate(
            self.object_name,
            self.target_name,
            object_pose,
            lower,
            upper,
            self._primitives.current_pose.quaternion,
        )

    def _candidate_failure(self, candidate: PushCandidate | None) -> SkillFailure | None:
        if candidate is None:
            return SkillFailure.INVALID_PUSH_TARGET
        if any(
            not self._primitives.workspace.contains(pose.position)
            for pose in candidate.poses
        ):
            return SkillFailure.PUSH_PATH_OUTSIDE_WORKSPACE
        if not self._world.is_push_path_safe(
            candidate.object_pose.position,
            candidate.target_object_position,
        ):
            return SkillFailure.PUSH_PATH_OUTSIDE_WORKSPACE
        return None

    def _execute_attempt(self, candidate: PushCandidate) -> _AttemptOutcome:
        initial_position = candidate.object_pose.position.copy()

        self._transition(SkillPhase.MOVE_TO_SAFE_HEIGHT)
        current = self._primitives.current_pose
        safe_height_pose = Pose(
            np.array(
                [
                    current.position[0],
                    current.position[1],
                    max(current.position[2], candidate.prepush_pose.position[2]),
                ],
                dtype=np.float64,
            ),
            current.quaternion,
        )
        result = self._primitives.move_linear(
            safe_height_pose,
            max_cartesian_step=0.015,
        )
        failure = self._primitive_failure(SkillPhase.MOVE_TO_SAFE_HEIGHT, result)
        if failure:
            return _AttemptOutcome(failure, result)
        self._emit_object_position("safe_height")

        self._transition(SkillPhase.OPEN_GRIPPER)
        result = self._primitives.open_gripper(steps=self.config.opening_steps)
        failure = self._primitive_failure(SkillPhase.OPEN_GRIPPER, result)
        if failure:
            return _AttemptOutcome(failure, result)
        self._emit_object_position("open")

        self._transition(SkillPhase.MOVE_ABOVE_PREPUSH)
        result = self._primitives.move_to_pose(candidate.prepush_pose)
        failure = self._primitive_failure(SkillPhase.MOVE_ABOVE_PREPUSH, result)
        if failure:
            return _AttemptOutcome(failure, result)
        self._emit_object_position("above_prepush")

        self._transition(SkillPhase.DESCEND_TO_PUSH_HEIGHT)
        result = self._primitives.move_linear(
            candidate.push_start_pose,
            max_cartesian_step=0.012,
        )
        failure = self._primitive_failure(SkillPhase.DESCEND_TO_PUSH_HEIGHT, result)
        if failure:
            return _AttemptOutcome(failure, result)
        self._emit_object_position("push_height")

        self._transition(SkillPhase.MOVE_TO_CONTACT)
        result = self._primitives.move_linear(
            candidate.contact_pose,
            max_cartesian_step=0.008,
        )
        failure = self._primitive_failure(SkillPhase.MOVE_TO_CONTACT, result)
        if failure:
            return _AttemptOutcome(failure, result)
        self._emit_object_position("contact")

        self._transition(SkillPhase.VERIFY_CONTACT)
        try:
            object_position = self._world.pose(self.object_name).position
        except (KeyError, ValueError):
            return _AttemptOutcome(
                _FailureEvent(
                    SkillPhase.VERIFY_CONTACT,
                    SkillFailure.OBJECT_POSE_UNAVAILABLE,
                    result,
                ),
                result,
            )
        proximity = float(
            np.linalg.norm(self._primitives.current_pose.position - object_position)
        )
        if proximity > self.config.contact_proximity:
            return _AttemptOutcome(
                _FailureEvent(
                    SkillPhase.VERIFY_CONTACT,
                    SkillFailure.NO_PUSH_CONTACT,
                    result,
                ),
                result,
            )

        self._transition(SkillPhase.PUSH_LINEAR)
        result = self._primitives.move_linear(
            candidate.end_pose,
            max_cartesian_step=0.008,
            max_steps=500,
        )
        failure = self._primitive_failure(SkillPhase.PUSH_LINEAR, result)
        if failure:
            return _AttemptOutcome(failure, result)

        result = self._primitives.wait(steps=self.config.settle_steps)
        failure = self._primitive_failure(SkillPhase.PUSH_LINEAR, result)
        if failure:
            return _AttemptOutcome(failure, result)

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
            final_position = self._world.pose(self.object_name).position
            reason = evaluate_push_outcome(
                initial_position,
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
            reason = SkillFailure.OBJECT_POSE_UNAVAILABLE
        if reason is not None:
            return _AttemptOutcome(
                _FailureEvent(SkillPhase.VERIFY_SUCCESS, reason, result),
                result,
            )
        return _AttemptOutcome(None, result)

    def _recover(self, candidate: PushCandidate) -> _RecoveryOutcome:
        self._transition(SkillPhase.RECOVERY)
        self._transition(SkillPhase.RECOVERY_RETREAT)
        result = self._primitives.move_to_pose(candidate.retreat_pose)
        if not result.success:
            return _RecoveryOutcome(
                None,
                _FailureEvent(
                    SkillPhase.RECOVERY_RETREAT,
                    SkillFailure.RECOVERY_FAILED,
                    result,
                ),
            )

        self._transition(SkillPhase.RECOVERY_REFRESH_PUSH_STATE)
        try:
            self._world.pose(self.object_name)
        except (KeyError, ValueError):
            return _RecoveryOutcome(
                None,
                _FailureEvent(
                    SkillPhase.RECOVERY_REFRESH_PUSH_STATE,
                    SkillFailure.OBJECT_POSE_UNAVAILABLE,
                    None,
                ),
            )

        self._transition(SkillPhase.RECOVERY_RECOMPUTE_PUSH)
        refreshed = self._generate_candidate()
        reason = self._candidate_failure(refreshed)
        if reason is not None:
            return _RecoveryOutcome(
                None,
                _FailureEvent(SkillPhase.RECOVERY_RECOMPUTE_PUSH, reason, None),
            )
        return _RecoveryOutcome(refreshed, None)

    def execute(self) -> SkillResult:
        self._trace = []
        self.current_phase = SkillPhase.IDLE
        self._emit(f"PushSkill({self.object_name}, {self.target_name})")

        precondition_failure = self.check_preconditions()
        if precondition_failure is not None:
            return precondition_failure

        candidate: PushCandidate | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            self._emit(f"[attempt {attempt}]")
            self._transition(SkillPhase.GENERATE_PUSH)
            if candidate is None:
                candidate = self._generate_candidate()
            candidate_reason = self._candidate_failure(candidate)
            if candidate_reason is not None:
                return self._failure_result(
                    _FailureEvent(
                        SkillPhase.GENERATE_PUSH,
                        candidate_reason,
                        None,
                    ),
                    attempts=attempt,
                )
            assert candidate is not None
            self._emit(
                "PUSH_GEOMETRY: "
                f"object={candidate.object_pose.position.tolist()} "
                f"target={candidate.target_object_position.tolist()} "
                f"direction={candidate.direction.tolist()}"
            )

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

            recovery = self._recover(candidate)
            if recovery.failure is not None:
                return self._failure_result(recovery.failure, attempts=attempt)
            candidate = recovery.candidate

        raise RuntimeError("PushSkill attempt loop ended unexpectedly")
