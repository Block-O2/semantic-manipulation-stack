"""Closed-loop one-step behavior-cloning Push backend baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datasets.specs import encode_push_state
from learning import OneStepBCPolicy
from robot import Pose
from skills.base import SkillFailure, SkillPhase
from skills.push_backend import (
    PushBackendResult,
    PushExecutionContext,
    PushRecoveryResult,
    PushRequest,
)


@dataclass(frozen=True)
class BCPushConfig:
    maximum_control_steps: int = 650
    maximum_cartesian_step: float = 0.04
    hard_rejection_step: float = 0.12
    opening_steps: int = 35
    minimum_displacement: float = 0.06

    def __post_init__(self) -> None:
        if self.maximum_control_steps <= 0 or self.opening_steps <= 0:
            raise ValueError("BC execution step limits must be positive")
        if not 0.0 < self.maximum_cartesian_step <= self.hard_rejection_step:
            raise ValueError("BC Cartesian safety limits are inconsistent")
        if self.minimum_displacement <= 0.0:
            raise ValueError("minimum_displacement must be positive")


class BCPushBackend:
    """Predict one Cartesian xyz command, execute one step, and re-observe."""

    name = "bc"

    def __init__(
        self,
        policy: OneStepBCPolicy,
        *,
        config: BCPushConfig = BCPushConfig(),
    ) -> None:
        self.policy = policy
        self.config = config

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        *,
        config: BCPushConfig = BCPushConfig(),
    ) -> "BCPushBackend":
        policy, _ = OneStepBCPolicy.load(checkpoint)
        return cls(policy, config=config)

    @staticmethod
    def _failure(
        reason: SkillFailure,
        initial: np.ndarray | None,
        steps: int,
        trace: list[str],
        primitive_result=None,
    ) -> PushBackendResult:
        return PushBackendResult(
            False,
            SkillPhase.PUSH_LINEAR,
            reason,
            primitive_result,
            initial,
            steps,
            tuple(trace),
        )

    def execute(
        self,
        request: PushRequest,
        context: PushExecutionContext,
    ) -> PushBackendResult:
        trace = ["BC_POLICY_ROLLOUT", SkillPhase.OPEN_GRIPPER.value]
        try:
            initial = context.world.pose(request.object_name).position.copy()
        except (KeyError, ValueError):
            return self._failure(
                SkillFailure.OBJECT_POSE_UNAVAILABLE,
                None,
                0,
                trace,
            )
        result = context.primitives.open_gripper(steps=self.config.opening_steps)
        control_steps = result.steps
        if not result.success:
            return self._failure(
                SkillFailure.MOTION_TIMEOUT,
                initial,
                control_steps,
                trace,
                result,
            )
        orientation = context.primitives.current_pose.quaternion
        trace.append(SkillPhase.PUSH_LINEAR.value)
        clipped_predictions = 0
        for _ in range(self.config.maximum_control_steps):
            observation = encode_push_state(
                context.primitives.current_pose,
                context.world,
                request.object_name,
                request.target_name,
            )
            predicted = np.asarray(self.policy.predict(observation), dtype=np.float64)
            if predicted.shape != (3,) or not np.all(np.isfinite(predicted)):
                return self._failure(
                    SkillFailure.POLICY_ACTION_NONFINITE,
                    initial,
                    control_steps,
                    trace,
                )
            current = context.primitives.current_pose.position
            delta = predicted - current
            distance = float(np.linalg.norm(delta))
            if distance > self.config.hard_rejection_step:
                return self._failure(
                    SkillFailure.POLICY_ACTION_UNSAFE,
                    initial,
                    control_steps,
                    trace,
                )
            if distance > self.config.maximum_cartesian_step:
                predicted = current + delta * (
                    self.config.maximum_cartesian_step / distance
                )
                clipped_predictions += 1
            if not context.primitives.workspace.contains(predicted):
                return self._failure(
                    SkillFailure.POLICY_ACTION_UNSAFE,
                    initial,
                    control_steps,
                    trace,
                )
            result = context.primitives.command_cartesian_once(
                Pose(predicted, orientation),
                max_cartesian_step=self.config.maximum_cartesian_step + 1e-9,
            )
            control_steps += result.steps
            if not result.success:
                return self._failure(
                    SkillFailure.POLICY_ACTION_UNSAFE,
                    initial,
                    control_steps,
                    trace,
                    result,
                )
            final = context.world.pose(request.object_name).position
            displacement = float(np.linalg.norm(final[:2] - initial[:2]))
            if (
                displacement >= self.config.minimum_displacement
                and context.world.is_inside_push_region(
                    request.object_name,
                    request.target_name,
                )
                and context.world.is_on_table(request.object_name)
            ):
                trace.append(f"BC_CLIPPED_PREDICTIONS: {clipped_predictions}")
                return PushBackendResult(
                    True,
                    SkillPhase.PUSH_LINEAR,
                    None,
                    result,
                    initial,
                    control_steps,
                    tuple(trace),
                )
        trace.append(f"BC_CLIPPED_PREDICTIONS: {clipped_predictions}")
        return self._failure(
            SkillFailure.POLICY_TIMEOUT,
            initial,
            control_steps,
            trace,
            result,
        )

    def recover(
        self,
        request: PushRequest,
        context: PushExecutionContext,
        previous: PushBackendResult,
    ) -> PushRecoveryResult:
        del request, previous
        current = context.primitives.current_pose
        retreat_position = current.position.copy()
        retreat_position[2] = min(
            context.primitives.workspace.upper[2],
            retreat_position[2] + 0.12,
        )
        result = context.primitives.move_linear(
            Pose(retreat_position, current.quaternion),
            max_cartesian_step=0.015,
        )
        return PushRecoveryResult(
            result.success,
            SkillPhase.RECOVERY_RETREAT,
            None if result.success else SkillFailure.RECOVERY_FAILED,
            result,
            result.steps,
            (SkillPhase.RECOVERY.value, SkillPhase.RECOVERY_RETREAT.value),
        )
