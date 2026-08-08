"""Modest randomized physical validation for the classical PushSkill."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import numpy as np

from planner import Goal, PlanValidator, RuleBasedPlanner, SkillRegistry
from playground import WorldController
from primitives import ManipulationPrimitives
from robot import PandaRobot, Pose
from runtime import AgentRuntime, SemanticSkillFactory
from sim import make_environment
from world import WorldModel


EVALUATION_STANDBY_POSITION = np.array([-0.25, -0.25, 1.10])


def run_evaluation(*, trials: int, seed: int) -> dict[str, object]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(seed)
    successes = 0
    failures: Counter[str] = Counter()
    attempts: list[int] = []
    placement_rejections = 0
    for trial in range(1, trials + 1):
        # Each trial owns a fresh simulator instance. A soft robosuite reset can
        # retain OSC controller state from the preceding contact-rich episode,
        # which makes trials order-dependent rather than independently seeded.
        env = make_environment(
            render=False,
            randomize_cube=False,
            seed=seed + trial,
        )
        try:
            env.reset()
            world = WorldModel(env)
            robot = PandaRobot(env)
            primitives = ManipulationPrimitives(robot)
            standby = Pose(
                EVALUATION_STANDBY_POSITION,
                robot.pose.quaternion,
            )
            standby_result = primitives.move_to_pose(standby)
            if not standby_result.success:
                raise RuntimeError("could not move robot to evaluation standby pose")
            open_result = primitives.open_gripper()
            if not open_result.success:
                raise RuntimeError("could not open gripper during evaluation setup")
            controller = WorldController(env, world)
            controller.remove_object("green_cube")
            controller.remove_object("blue_cube")
            for _ in range(100):
                sampled = np.array(
                    [
                        rng.uniform(-0.10, 0.08),
                        rng.uniform(-0.15, 0.12),
                    ]
                )
                controller.move_object(
                    "red_cube",
                    float(sampled[0]),
                    float(sampled[1]),
                )
                primitives.wait(steps=10)
                settled = world.pose("red_cube").position
                if (
                    np.linalg.norm(settled[:2] - sampled) <= 0.003
                    and world.is_on_table("red_cube")
                ):
                    break
                placement_rejections += 1
            else:
                raise RuntimeError("could not sample a physically valid cube position")
            initial = world.pose("red_cube").position.copy()
            registry = SkillRegistry.standard()
            result = AgentRuntime(
                planner=RuleBasedPlanner(),
                registry=registry,
                validator=PlanValidator(registry),
                observer=world.semantic_state,
                skill_factory=SemanticSkillFactory(world, primitives),
                max_replans=0,
            ).run(
                Goal.push_to_region(
                    "Push the red cube toward the right side of the table.",
                    "red_cube",
                    "right_side",
                )
            )
            final = world.pose("red_cube").position.copy()
            skill_result = (
                result.executed_steps[-1].task_result.skill_results[-1]
                if result.executed_steps
                else None
            )
            reason = (
                skill_result.reason.value
                if skill_result is not None and skill_result.reason is not None
                else result.failure_detail
            )
            successes += int(result.success)
            if not result.success:
                failures[reason or "UNKNOWN"] += 1
            attempts.append(skill_result.attempts if skill_result else 0)
            print(
                json.dumps(
                    {
                        "trial": trial,
                        "initial_position": initial.tolist(),
                        "final_position": final.tolist(),
                        "displacement": float(
                            np.linalg.norm(final[:2] - initial[:2])
                        ),
                        "success": result.success,
                        "attempts": attempts[-1],
                        "failure_reason": reason,
                        "region_satisfied": world.is_inside_push_region(
                            "red_cube", "right_side"
                        ),
                    },
                    sort_keys=True,
                )
            )
        finally:
            env.close()

    required = math.ceil(0.9 * trials)
    summary: dict[str, object] = {
        "trials": trials,
        "successes": successes,
        "success_rate": successes / trials,
        "required_successes": required,
        "target_met": successes >= required,
        "failure_reasons": dict(sorted(failures.items())),
        "average_attempts": sum(attempts) / trials,
        "placement_rejections": placement_rejections,
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=127)
    args = parser.parse_args()
    summary = run_evaluation(trials=args.trials, seed=args.seed)
    if not summary["target_met"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
