"""Scripted contact-grasp experiment assembled from manipulation primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from primitives import ManipulationPrimitives, PrimitiveResult
from robot import PandaRobot, Pose
from world import WorldModel


@dataclass(frozen=True)
class GraspParameters:
    pregrasp_height: float = 0.14
    grasp_z_offset: float = 0.005
    lift_height: float = 0.15
    minimum_lift: float = 0.08
    maximum_gripper_distance: float = 0.12
    opening_steps: int = 40
    closing_steps: int = 50


@dataclass(frozen=True)
class GraspTrialResult:
    success: bool
    failure_reason: str | None
    initial_cube_pose: Pose
    final_cube_pose: Pose
    cube_lift: float
    contact_grasped: bool
    eef_cube_distance: float
    primitive_results: dict[str, PrimitiveResult]

    def as_log_record(self, trial: int) -> dict[str, object]:
        return {
            "trial": trial,
            "cube_initial_position": self.initial_cube_pose.position.round(5).tolist(),
            "success": self.success,
            "failure_reason": self.failure_reason,
            "contact_grasped": self.contact_grasped,
            "final_cube_height": round(float(self.final_cube_pose.position[2]), 5),
            "cube_lift": round(self.cube_lift, 5),
            "eef_cube_distance": round(self.eef_cube_distance, 5),
            "primitives": {
                name: {
                    "success": result.success,
                    "reason": result.reason.value if result.reason else None,
                    "steps": result.steps,
                    "position_error": result.final_position_error,
                    "orientation_error": result.final_orientation_error,
                }
                for name, result in self.primitive_results.items()
            },
        }


def _primitive_failure(results: dict[str, PrimitiveResult]) -> str | None:
    for name, result in results.items():
        if not result.success:
            reason = result.reason.value if result.reason else "UNKNOWN"
            return f"{name}:{reason}"
    return None


def run_grasp_trial(
    robot: PandaRobot,
    world: WorldModel,
    primitives: ManipulationPrimitives,
    *,
    parameters: GraspParameters = GraspParameters(),
) -> GraspTrialResult:
    """Execute an explicit approach-descend-close-lift sequence."""

    initial_cube = world.pose("red_cube")
    downward_orientation = robot.pose.quaternion
    results: dict[str, PrimitiveResult] = {}

    results["open"] = primitives.open_gripper(steps=parameters.opening_steps)
    pregrasp = Pose(
        initial_cube.position + np.array([0.0, 0.0, parameters.pregrasp_height]),
        downward_orientation,
    )
    results["pregrasp"] = primitives.move_to_pose(pregrasp)

    if results["pregrasp"].success:
        grasp_pose = Pose(
            initial_cube.position + np.array([0.0, 0.0, parameters.grasp_z_offset]),
            downward_orientation,
        )
        results["descend"] = primitives.move_linear(
            grasp_pose,
            max_cartesian_step=0.015,
        )

    if results.get("descend", PrimitiveResult(False, None, 0, None, None)).success:
        results["close"] = primitives.close_gripper(steps=parameters.closing_steps)

    if results.get("close", PrimitiveResult(False, None, 0, None, None)).success:
        current = robot.pose
        lift_pose = Pose(
            current.position + np.array([0.0, 0.0, parameters.lift_height]),
            current.quaternion,
        )
        results["lift"] = primitives.move_linear(
            lift_pose,
            max_cartesian_step=0.015,
        )

    final_cube = world.pose("red_cube")
    contact_grasped = world.is_grasped("red_cube")
    cube_lift = float(final_cube.position[2] - initial_cube.position[2])
    eef_cube_distance = float(np.linalg.norm(robot.pose.position - final_cube.position))

    failure_reason = _primitive_failure(results)
    if failure_reason is None and not contact_grasped:
        failure_reason = "NO_CONTACT_GRASP"
    if failure_reason is None and cube_lift < parameters.minimum_lift:
        failure_reason = "CUBE_NOT_LIFTED"
    if failure_reason is None and eef_cube_distance > parameters.maximum_gripper_distance:
        failure_reason = "CUBE_NOT_WITH_GRIPPER"

    return GraspTrialResult(
        success=failure_reason is None,
        failure_reason=failure_reason,
        initial_cube_pose=initial_cube,
        final_cube_pose=final_cube,
        cube_lift=cube_lift,
        contact_grasped=contact_grasped,
        eef_cube_distance=eef_cube_distance,
        primitive_results=results,
    )
