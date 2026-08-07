"""Semantic PickSkill with bounded, skill-local manipulation recovery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from primitives import ManipulationPrimitives, PrimitiveFailure, PrimitiveResult
from skills.base import Skill, SkillFailure, SkillPhase, SkillResult
from skills.grasp import GraspCandidate, GraspGenerator, TopDownCubeGraspGenerator
from world import WorldModel


@dataclass(frozen=True)
class PickConfig:
    max_attempts: int = 2
    opening_steps: int = 40
    closing_steps: int = 50
    minimum_lift: float = 0.08
    maximum_gripper_distance: float = 0.12

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.opening_steps <= 0 or self.closing_steps <= 0:
            raise ValueError("gripper step counts must be positive")
        if self.minimum_lift <= 0.0 or self.maximum_gripper_distance <= 0.0:
            raise ValueError("pick verification thresholds must be positive")


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
    candidate: GraspCandidate | None
    failure: _FailureEvent | None


class PickSkill(Skill):
    """Pick one known object using only the manipulation primitive layer."""

    RECOVERABLE_FAILURES = frozenset(
        {
            SkillFailure.MOTION_TIMEOUT,
            SkillFailure.POSITION_NOT_CONVERGED,
            SkillFailure.ORIENTATION_NOT_CONVERGED,
            SkillFailure.NO_CONTACT_GRASP,
            SkillFailure.GRASP_LOST,
            SkillFailure.CUBE_NOT_LIFTED,
            SkillFailure.CUBE_NOT_WITH_GRIPPER,
        }
    )

    def __init__(
        self,
        object_name: str,
        world: WorldModel,
        primitives: ManipulationPrimitives,
        *,
        grasp_generator: GraspGenerator | None = None,
        config: PickConfig = PickConfig(),
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.object_name = object_name
        self._world = world
        self._primitives = primitives
        self._grasp_generator = grasp_generator or TopDownCubeGraspGenerator()
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

    def _failure_result(
        self,
        failure: _FailureEvent,
        *,
        attempts: int,
    ) -> SkillResult:
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
        if self.object_name not in self._world.object_names:
            return self._failure_result(
                _FailureEvent(
                    SkillPhase.CHECK_PRECONDITIONS,
                    SkillFailure.OBJECT_NOT_FOUND,
                    None,
                ),
                attempts=0,
            )
        try:
            if self._world.is_grasped(self.object_name):
                return self._failure_result(
                    _FailureEvent(
                        SkillPhase.CHECK_PRECONDITIONS,
                        SkillFailure.OBJECT_ALREADY_GRASPED,
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

    def _candidate_failure(self, candidate: GraspCandidate | None) -> SkillFailure | None:
        if candidate is None:
            return SkillFailure.NO_VALID_GRASP
        if any(not self._primitives.workspace.contains(pose.position) for pose in candidate.poses):
            return SkillFailure.TARGET_UNREACHABLE
        return None

    def _generate_candidate(self) -> GraspCandidate | None:
        try:
            object_pose = self._world.pose(self.object_name)
        except (KeyError, ValueError):
            return None
        return self._grasp_generator.generate(
            self.object_name,
            object_pose,
            self._primitives.current_pose.quaternion,
        )

    def _primitive_failure(
        self,
        phase: SkillPhase,
        result: PrimitiveResult,
    ) -> _FailureEvent | None:
        if result.success:
            return None
        return _FailureEvent(phase, self._map_primitive_failure(result), result)

    def _execute_attempt(self, candidate: GraspCandidate) -> _AttemptOutcome:
        self._transition(SkillPhase.OPEN_GRIPPER)
        result = self._primitives.open_gripper(steps=self.config.opening_steps)
        failure = self._primitive_failure(SkillPhase.OPEN_GRIPPER, result)
        if failure:
            return _AttemptOutcome(failure, result)

        self._transition(SkillPhase.MOVE_TO_PREGRASP)
        result = self._primitives.move_to_pose(candidate.pregrasp_pose)
        failure = self._primitive_failure(SkillPhase.MOVE_TO_PREGRASP, result)
        if failure:
            return _AttemptOutcome(failure, result)

        self._transition(SkillPhase.APPROACH)
        result = self._primitives.move_linear(
            candidate.grasp_pose,
            max_cartesian_step=0.015,
        )
        failure = self._primitive_failure(SkillPhase.APPROACH, result)
        if failure:
            return _AttemptOutcome(failure, result)

        self._transition(SkillPhase.CLOSE_GRIPPER)
        result = self._primitives.close_gripper(steps=self.config.closing_steps)
        failure = self._primitive_failure(SkillPhase.CLOSE_GRIPPER, result)
        if failure:
            return _AttemptOutcome(failure, result)

        self._transition(SkillPhase.VERIFY_GRASP)
        if not self._world.is_grasped(self.object_name):
            return _AttemptOutcome(
                _FailureEvent(
                    SkillPhase.VERIFY_GRASP,
                    SkillFailure.NO_CONTACT_GRASP,
                    result,
                ),
                result,
            )

        self._transition(SkillPhase.LIFT)
        result = self._primitives.move_linear(
            candidate.lift_pose,
            max_cartesian_step=0.015,
        )
        failure = self._primitive_failure(SkillPhase.LIFT, result)
        if failure:
            return _AttemptOutcome(failure, result)

        self._transition(SkillPhase.VERIFY_SUCCESS)
        try:
            final_object_pose = self._world.pose(self.object_name)
        except (KeyError, ValueError):
            return _AttemptOutcome(
                _FailureEvent(
                    SkillPhase.VERIFY_SUCCESS,
                    SkillFailure.OBJECT_POSE_UNAVAILABLE,
                    result,
                ),
                result,
            )
        if not self._world.is_grasped(self.object_name):
            return _AttemptOutcome(
                _FailureEvent(
                    SkillPhase.VERIFY_SUCCESS,
                    SkillFailure.GRASP_LOST,
                    result,
                ),
                result,
            )
        cube_lift = float(final_object_pose.position[2] - candidate.object_pose.position[2])
        if cube_lift < self.config.minimum_lift:
            return _AttemptOutcome(
                _FailureEvent(
                    SkillPhase.VERIFY_SUCCESS,
                    SkillFailure.CUBE_NOT_LIFTED,
                    result,
                ),
                result,
            )
        gripper_distance = float(
            np.linalg.norm(self._primitives.current_pose.position - final_object_pose.position)
        )
        if gripper_distance > self.config.maximum_gripper_distance:
            return _AttemptOutcome(
                _FailureEvent(
                    SkillPhase.VERIFY_SUCCESS,
                    SkillFailure.CUBE_NOT_WITH_GRIPPER,
                    result,
                ),
                result,
            )
        return _AttemptOutcome(None, result)

    def _recover(self, candidate: GraspCandidate) -> _RecoveryOutcome:
        self._transition(SkillPhase.RECOVERY)

        self._transition(SkillPhase.RECOVERY_OPEN_GRIPPER)
        result = self._primitives.open_gripper(steps=self.config.opening_steps)
        if not result.success:
            return _RecoveryOutcome(
                None,
                _FailureEvent(
                    SkillPhase.RECOVERY_OPEN_GRIPPER,
                    SkillFailure.RECOVERY_FAILED,
                    result,
                ),
            )

        self._transition(SkillPhase.RECOVERY_RETREAT)
        result = self._primitives.move_to_pose(candidate.pregrasp_pose)
        if not result.success:
            return _RecoveryOutcome(
                None,
                _FailureEvent(
                    SkillPhase.RECOVERY_RETREAT,
                    SkillFailure.RECOVERY_FAILED,
                    result,
                ),
            )

        self._transition(SkillPhase.RECOVERY_REFRESH_OBJECT_STATE)
        if self.object_name not in self._world.object_names:
            return _RecoveryOutcome(
                None,
                _FailureEvent(
                    SkillPhase.RECOVERY_REFRESH_OBJECT_STATE,
                    SkillFailure.OBJECT_NOT_FOUND,
                    None,
                ),
            )
        try:
            self._world.pose(self.object_name)
        except (KeyError, ValueError):
            return _RecoveryOutcome(
                None,
                _FailureEvent(
                    SkillPhase.RECOVERY_REFRESH_OBJECT_STATE,
                    SkillFailure.OBJECT_POSE_UNAVAILABLE,
                    None,
                ),
            )

        self._transition(SkillPhase.RECOVERY_RECOMPUTE_GRASP)
        refreshed_candidate = self._generate_candidate()
        reason = self._candidate_failure(refreshed_candidate)
        if reason is not None:
            return _RecoveryOutcome(
                None,
                _FailureEvent(SkillPhase.RECOVERY_RECOMPUTE_GRASP, reason, None),
            )
        return _RecoveryOutcome(refreshed_candidate, None)

    def execute(self) -> SkillResult:
        self._trace = []
        self.current_phase = SkillPhase.IDLE
        self._emit(f"PickSkill({self.object_name})")

        precondition_failure = self.check_preconditions()
        if precondition_failure is not None:
            return precondition_failure

        candidate: GraspCandidate | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            self._emit(f"[attempt {attempt}]")
            self._transition(SkillPhase.GENERATE_GRASP)
            if candidate is None:
                candidate = self._generate_candidate()
            candidate_reason = self._candidate_failure(candidate)
            if candidate_reason is not None:
                return self._failure_result(
                    _FailureEvent(SkillPhase.GENERATE_GRASP, candidate_reason, None),
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

            recovery = self._recover(candidate)
            if recovery.failure is not None:
                return self._failure_result(recovery.failure, attempts=attempt)
            candidate = recovery.candidate

        raise RuntimeError("PickSkill attempt loop ended unexpectedly")
