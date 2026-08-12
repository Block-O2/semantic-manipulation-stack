"""Run the full semantic Agent path with a selected Push backend."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from planner import Goal, PlanValidator, RuleBasedPlanner, SkillRegistry
from playground import WorldController
from primitives import ManipulationPrimitives
from robot import PandaRobot, Pose
from runtime import AgentRuntime, SemanticSkillFactory, create_push_backend
from sim import make_environment
from world import WorldModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("right_side",), default="right_side")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--inspect-seconds", type=float, default=8.0)
    parser.add_argument("--cube-x", type=float)
    parser.add_argument("--cube-y", type=float)
    parser.add_argument(
        "--push-backend",
        choices=("classical", "bc", "progress_bc", "chunk_bc", "act"),
        default="classical",
        help=(
            "physical backend; classical and act are current paths, while "
            "bc/progress_bc/chunk_bc are retained diagnostics"
        ),
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--execution-horizon", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--act-execution-mode",
        choices=("queue", "temporal_ensemble"),
        default="queue",
        help=(
            "ACT inference mode; queue is retained for historical comparison, "
            "while temporal_ensemble is the validated learned mode"
        ),
    )
    args = parser.parse_args()

    env = make_environment(render=not args.no_render, seed=113)
    try:
        env.reset()
        robot = PandaRobot(env)
        world = WorldModel(env)
        primitives = ManipulationPrimitives(robot)
        if (args.cube_x is None) != (args.cube_y is None):
            parser.error("--cube-x and --cube-y must be provided together")
        if args.cube_x is not None and args.cube_y is not None:
            standby = Pose(
                np.array([-0.25, -0.25, 1.10]),
                robot.pose.quaternion,
            )
            if not primitives.move_to_pose(standby).success:
                raise RuntimeError("could not reach randomized-demo standby pose")
            primitives.open_gripper()
            controller = WorldController(env, world)
            controller.remove_object("green_cube")
            controller.remove_object("blue_cube")
            controller.move_object("red_cube", args.cube_x, args.cube_y)
            requested = np.array([args.cube_x, args.cube_y])
            primitives.wait(steps=10)
            if (
                np.linalg.norm(world.pose("red_cube").position[:2] - requested)
                > 0.003
                or not world.is_on_table("red_cube")
            ):
                raise ValueError(
                    "requested cube position intersects the initialized robot geometry"
                )
        registry = SkillRegistry.standard()
        push_backend = create_push_backend(
            args.push_backend,
            checkpoint=args.checkpoint,
            execution_horizon=args.execution_horizon,
            device=args.device,
            act_execution_mode=args.act_execution_mode,
        )
        initial = world.pose("red_cube").position.copy()
        goal = Goal.push_to_region(
            f"Push the red cube toward {args.target}.",
            "red_cube",
            args.target,
        )
        result = AgentRuntime(
            planner=RuleBasedPlanner(),
            registry=registry,
            validator=PlanValidator(registry),
            observer=world.semantic_state,
            skill_factory=SemanticSkillFactory(
                world,
                primitives,
                push_backend=push_backend,
            ),
            max_replans=0,
            logger=print,
        ).run(goal)
        final = world.pose("red_cube").position.copy()
        skill_result = (
            result.executed_steps[-1].task_result.skill_results[-1]
            if result.executed_steps
            else None
        )
        print(
            json.dumps(
                {
                    "success": result.success,
                    "requested_region": args.target,
                    "initial_position": initial.tolist(),
                    "final_position": final.tolist(),
                    "displacement": float(np.linalg.norm(final[:2] - initial[:2])),
                    "region_satisfied": world.is_inside_push_region(
                        "red_cube", args.target
                    ),
                    "attempts": skill_result.attempts if skill_result else 0,
                    "failure_reason": (
                        skill_result.reason.value
                        if skill_result and skill_result.reason
                        else result.failure_detail
                    ),
                    "skill_trace": list(skill_result.trace) if skill_result else [],
                    "agent_trace": list(result.trace),
                },
                indent=2,
                sort_keys=True,
            )
        )

        inspect_steps = round(max(0.0, args.inspect_seconds) * env.control_freq)
        for _ in range(inspect_steps):
            robot.hold(1)
            if not args.no_render:
                time.sleep(1.0 / env.control_freq)
        if not result.success:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
