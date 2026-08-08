"""Classical Cartesian implementation of the physical Push backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from primitives import PrimitiveFailure, PrimitiveResult
from robot import Pose
from skills.base import SkillFailure, SkillPhase
from skills.push_backend import (
    PushBackendResult,
    PushExecutionContext,
    PushRecoveryResult,
    PushRequest,
)
from skills.push_geometry import (
    AxisAlignedCubePushPoseGenerator,
    PushCandidate,
    PushPoseGenerator,
)


@dataclass(frozen=True)
class ClassicalPushConfig:
    opening_steps: int = 35
    settle_steps: int = 20
    contact_proximity: float = 0.07

    def __post_init__(self) -> None:
        if self.opening_steps <= 0 or self.settle_steps < 0:
            raise ValueError("push gripper steps must be positive and settle non-negative")
        if self.contact_proximity <= 0.0:
            raise ValueError("contact_proximity must be positive")


def map_primitive_failure(result: PrimitiveResult) -> SkillFailure:
    mapping = {
        PrimitiveFailure.TARGET_OUTSIDE_WORKSPACE: (
            SkillFailure.PUSH_PATH_OUTSIDE_WORKSPACE
        ),
        PrimitiveFailure.TIMEOUT: SkillFailure.MOTION_TIMEOUT,
        PrimitiveFailure.POSITION_NOT_CONVERGED: SkillFailure.POSITION_NOT_CONVERGED,
        PrimitiveFailure.ORIENTATION_NOT_CONVERGED: (
            SkillFailure.ORIENTATION_NOT_CONVERGED
        ),
    }
    return mapping.get(result.reason, SkillFailure.MOTION_TIMEOUT)


class ClassicalPushBackend:
    """Generate fixed geometry and execute one open-gripper Cartesian push."""

    name = "classical"

    def __init__(
        self,
        *,
        push_generator: PushPoseGenerator | None = None,
        config: ClassicalPushConfig = ClassicalPushConfig(),
    ) -> None:
        self.push_generator = push_generator or AxisAlignedCubePushPoseGenerator()
        self.config = config

    @staticmethod
    def _failure(
        phase: SkillPhase,
        reason: SkillFailure,
        primitive_result: PrimitiveResult | None,
        initial_position: np.ndarray | None,
        control_steps: int,
        trace: list[str],
        candidate: PushCandidate | None,
    ) -> PushBackendResult:
        return PushBackendResult(
            False,
            phase,
            reason,
            primitive_result,
            initial_position,
            control_steps,
            tuple(trace),
            candidate,
        )

    @staticmethod
    def _primitive_failure(
        phase: SkillPhase,
        result: PrimitiveResult,
        initial_position: np.ndarray,
        control_steps: int,
        trace: list[str],
        candidate: PushCandidate,
    ) -> PushBackendResult | None:
        if result.success:
            return None
        return ClassicalPushBackend._failure(
            phase,
            map_primitive_failure(result),
            result,
            initial_position,
            control_steps,
            trace,
            candidate,
        )

    @staticmethod
    def _object_trace(
        context: PushExecutionContext,
        request: PushRequest,
        label: str,
    ) -> str:
        try:
            position = context.world.pose(request.object_name).position.tolist()
        except (KeyError, ValueError):
            return f"OBJECT_POSITION[{label}]: unavailable"
        return f"OBJECT_POSITION[{label}]: {position}"

    def _generate_candidate(
        self,
        request: PushRequest,
        context: PushExecutionContext,
    ) -> PushCandidate | None:
        try:
            object_pose = context.world.pose(request.object_name)
            lower, upper = context.world.push_region_bounds(request.target_name)
        except (KeyError, ValueError):
            return None
        return self.push_generator.generate(
            request.object_name,
            request.target_name,
            object_pose,
            lower,
            upper,
            context.primitives.current_pose.quaternion,
        )

    @staticmethod
    def _candidate_failure(
        candidate: PushCandidate | None,
        context: PushExecutionContext,
    ) -> SkillFailure | None:
        if candidate is None:
            return SkillFailure.INVALID_PUSH_TARGET
        if any(
            not context.primitives.workspace.contains(pose.position)
            for pose in candidate.poses
        ):
            return SkillFailure.PUSH_PATH_OUTSIDE_WORKSPACE
        if not context.world.is_push_path_safe(
            candidate.object_pose.position,
            candidate.target_object_position,
        ):
            return SkillFailure.PUSH_PATH_OUTSIDE_WORKSPACE
        return None

    def execute(
        self,
        request: PushRequest,
        context: PushExecutionContext,
    ) -> PushBackendResult:
        trace = [SkillPhase.GENERATE_PUSH.value]
        control_steps = 0
        candidate = self._generate_candidate(request, context)
        reason = self._candidate_failure(candidate, context)
        if reason is not None:
            return self._failure(
                SkillPhase.GENERATE_PUSH,
                reason,
                None,
                None,
                control_steps,
                trace,
                candidate,
            )
        assert candidate is not None
        initial_position = candidate.object_pose.position.copy()
        trace.append(
            "PUSH_GEOMETRY: "
            f"object={initial_position.tolist()} "
            f"target={candidate.target_object_position.tolist()} "
            f"direction={candidate.direction.tolist()}"
        )

        phase = SkillPhase.MOVE_TO_SAFE_HEIGHT
        trace.append(phase.value)
        current = context.primitives.current_pose
        safe_height_pose = Pose(
            np.array(
                [
                    current.position[0],
                    current.position[1],
                    max(current.position[2], candidate.prepush_pose.position[2]),
                ]
            ),
            current.quaternion,
        )
        result = context.primitives.move_linear(
            safe_height_pose,
            max_cartesian_step=0.015,
        )
        control_steps += result.steps
        failure = self._primitive_failure(
            phase, result, initial_position, control_steps, trace, candidate
        )
        if failure:
            return failure
        trace.append(self._object_trace(context, request, "safe_height"))

        phase = SkillPhase.OPEN_GRIPPER
        trace.append(phase.value)
        result = context.primitives.open_gripper(steps=self.config.opening_steps)
        control_steps += result.steps
        failure = self._primitive_failure(
            phase, result, initial_position, control_steps, trace, candidate
        )
        if failure:
            return failure
        trace.append(self._object_trace(context, request, "open"))

        phase = SkillPhase.MOVE_ABOVE_PREPUSH
        trace.append(phase.value)
        result = context.primitives.move_to_pose(candidate.prepush_pose)
        control_steps += result.steps
        failure = self._primitive_failure(
            phase, result, initial_position, control_steps, trace, candidate
        )
        if failure:
            return failure
        trace.append(self._object_trace(context, request, "above_prepush"))

        phase = SkillPhase.DESCEND_TO_PUSH_HEIGHT
        trace.append(phase.value)
        result = context.primitives.move_linear(
            candidate.push_start_pose,
            max_cartesian_step=0.012,
        )
        control_steps += result.steps
        failure = self._primitive_failure(
            phase, result, initial_position, control_steps, trace, candidate
        )
        if failure:
            return failure
        trace.append(self._object_trace(context, request, "push_height"))

        phase = SkillPhase.MOVE_TO_CONTACT
        trace.append(phase.value)
        result = context.primitives.move_linear(
            candidate.contact_pose,
            max_cartesian_step=0.008,
        )
        control_steps += result.steps
        failure = self._primitive_failure(
            phase, result, initial_position, control_steps, trace, candidate
        )
        if failure:
            return failure
        trace.append(self._object_trace(context, request, "contact"))

        phase = SkillPhase.VERIFY_CONTACT
        trace.append(phase.value)
        try:
            object_position = context.world.pose(request.object_name).position
        except (KeyError, ValueError):
            return self._failure(
                phase,
                SkillFailure.OBJECT_POSE_UNAVAILABLE,
                result,
                initial_position,
                control_steps,
                trace,
                candidate,
            )
        proximity = float(
            np.linalg.norm(context.primitives.current_pose.position - object_position)
        )
        if proximity > self.config.contact_proximity:
            return self._failure(
                phase,
                SkillFailure.NO_PUSH_CONTACT,
                result,
                initial_position,
                control_steps,
                trace,
                candidate,
            )

        phase = SkillPhase.PUSH_LINEAR
        trace.append(phase.value)
        result = context.primitives.move_linear(
            candidate.end_pose,
            max_cartesian_step=0.008,
            max_steps=500,
        )
        control_steps += result.steps
        failure = self._primitive_failure(
            phase, result, initial_position, control_steps, trace, candidate
        )
        if failure:
            return failure
        result = context.primitives.wait(steps=self.config.settle_steps)
        control_steps += result.steps
        failure = self._primitive_failure(
            phase, result, initial_position, control_steps, trace, candidate
        )
        if failure:
            return failure

        phase = SkillPhase.RETREAT
        trace.append(phase.value)
        result = context.primitives.move_linear(
            candidate.retreat_pose,
            max_cartesian_step=0.015,
        )
        control_steps += result.steps
        failure = self._primitive_failure(
            phase, result, initial_position, control_steps, trace, candidate
        )
        if failure:
            return failure

        return PushBackendResult(
            True,
            SkillPhase.RETREAT,
            None,
            result,
            initial_position,
            control_steps,
            tuple(trace),
            candidate,
        )

    def recover(
        self,
        request: PushRequest,
        context: PushExecutionContext,
        previous: PushBackendResult,
    ) -> PushRecoveryResult:
        trace = [
            SkillPhase.RECOVERY.value,
            SkillPhase.RECOVERY_RETREAT.value,
        ]
        candidate = previous.recovery_token
        if not isinstance(candidate, PushCandidate):
            return PushRecoveryResult(
                False,
                SkillPhase.RECOVERY_RETREAT,
                SkillFailure.RECOVERY_FAILED,
                None,
                0,
                tuple(trace),
            )
        result = context.primitives.move_to_pose(candidate.retreat_pose)
        if not result.success:
            return PushRecoveryResult(
                False,
                SkillPhase.RECOVERY_RETREAT,
                SkillFailure.RECOVERY_FAILED,
                result,
                result.steps,
                tuple(trace),
            )
        trace.append(SkillPhase.RECOVERY_REFRESH_PUSH_STATE.value)
        try:
            context.world.pose(request.object_name)
        except (KeyError, ValueError):
            return PushRecoveryResult(
                False,
                SkillPhase.RECOVERY_REFRESH_PUSH_STATE,
                SkillFailure.OBJECT_POSE_UNAVAILABLE,
                result,
                result.steps,
                tuple(trace),
            )
        trace.append(SkillPhase.RECOVERY_RECOMPUTE_PUSH.value)
        refreshed = self._generate_candidate(request, context)
        reason = self._candidate_failure(refreshed, context)
        if reason is not None:
            return PushRecoveryResult(
                False,
                SkillPhase.RECOVERY_RECOMPUTE_PUSH,
                reason,
                result,
                result.steps,
                tuple(trace),
            )
        return PushRecoveryResult(
            True,
            SkillPhase.RECOVERY_RECOMPUTE_PUSH,
            None,
            result,
            result.steps,
            tuple(trace),
        )
