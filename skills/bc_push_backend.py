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
        self.last_rollout_log: list[dict[str, object]] = []

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        *,
        config: BCPushConfig = BCPushConfig(),
    ) -> "BCPushBackend":
        policy, _ = OneStepBCPolicy.load(checkpoint)
        return cls(policy, config=config)

    def _predict_actions(
        self,
        observation: np.ndarray,
        rollout_step: int,
    ) -> np.ndarray:
        del rollout_step
        prediction = np.asarray(self.policy.predict(observation), dtype=np.float64)
        return np.atleast_2d(prediction)

    def _reset_policy(self) -> None:
        """Reset optional stateful policy queues at the start of each rollout."""

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
        self.last_rollout_log = []
        self._reset_policy()
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
        previous_ee = context.primitives.current_pose.position.copy()
        ee_path_length = 0.0
        trace.append(SkillPhase.PUSH_LINEAR.value)
        clipped_predictions = 0
        rollout_step = 0
        while rollout_step < self.config.maximum_control_steps:
            observation = encode_push_state(
                context.primitives.current_pose,
                context.world,
                request.object_name,
                request.target_name,
            )
            predictions = self._predict_actions(observation, rollout_step)
            if (
                predictions.ndim != 2
                or predictions.shape[1] != 3
                or not np.all(np.isfinite(predictions))
            ):
                return self._failure(
                    SkillFailure.POLICY_ACTION_NONFINITE,
                    initial,
                    control_steps,
                    trace,
                )
            for chunk_offset, raw_prediction in enumerate(predictions):
                if rollout_step >= self.config.maximum_control_steps:
                    break
                current = context.primitives.current_pose.position
                predicted = raw_prediction.copy()
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
                updated_ee = context.primitives.current_pose.position.copy()
                ee_path_length += float(np.linalg.norm(updated_ee - previous_ee))
                previous_ee = updated_ee
                displacement = float(np.linalg.norm(final[:2] - initial[:2]))
                self.last_rollout_log.append(
                    {
                        "rollout_step": rollout_step,
                        "chunk_offset": chunk_offset,
                        "ee_xyz": updated_ee.tolist(),
                        "cube_xyz": final.tolist(),
                        "predicted_xyz": raw_prediction.tolist(),
                        "executed_xyz": predicted.tolist(),
                        "predicted_step_m": distance,
                        "cube_displacement_m": displacement,
                        "ee_path_length_m": ee_path_length,
                    }
                )
                rollout_step += 1
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
