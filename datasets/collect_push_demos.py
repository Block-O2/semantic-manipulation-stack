"""Collect successful classical Push demonstrations at 20 Hz control resolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from datasets import (
    PushDataset,
    PushTrajectoryRecorder,
    save_push_dataset,
    validate_push_dataset,
)
from evaluation.push_sampling import prepare_push_scene, sample_valid_cube_position
from planner import Goal, PlanValidator, RuleBasedPlanner, SkillRegistry
from playground import WorldController
from primitives import ManipulationPrimitives
from robot import PandaRobot
from runtime import AgentRuntime, SemanticSkillFactory
from sim import make_environment
from world import WorldModel


def collect_push_demonstrations(
    *,
    episodes: int,
    seed: int,
    output: str | Path,
) -> PushDataset:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    rng = np.random.default_rng(seed)
    collected = []
    rejected_executions = 0
    placement_rejections = 0
    trial = 0
    while len(collected) < episodes:
        trial += 1
        if trial > episodes * 4:
            raise RuntimeError("too many failed expert executions during collection")
        episode_seed = seed + trial
        env = make_environment(render=False, randomize_cube=False, seed=episode_seed)
        try:
            env.reset()
            robot = PandaRobot(env)
            world = WorldModel(env)
            primitives = ManipulationPrimitives(robot)
            controller = WorldController(env, world)
            prepare_push_scene(world, robot, primitives, controller)
            initial, rejected = sample_valid_cube_position(
                world,
                primitives,
                controller,
                rng,
            )
            placement_rejections += rejected

            recorder = PushTrajectoryRecorder(world, "red_cube", "right_side")
            recorder.attach(robot)
            try:
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
            finally:
                recorder.detach()
            final = world.pose("red_cube").position.copy()
            skill_result = (
                result.executed_steps[-1].task_result.skill_results[-1]
                if result.executed_steps
                else None
            )
            displacement = float(np.linalg.norm(final[:2] - initial[:2]))
            metadata = {
                "episode_id": f"push-{seed}-{len(collected):04d}",
                "random_seed": episode_seed,
                "initial_cube_position": initial.tolist(),
                "target": "right_side",
                "success": bool(result.success),
                "attempt_count": skill_result.attempts if skill_result else 0,
                "episode_length": recorder.length,
                "final_cube_position": final.tolist(),
                "total_displacement": displacement,
                "target_reached": world.is_inside_push_region(
                    "red_cube", "right_side"
                ),
                "failure_reason": (
                    skill_result.reason.value
                    if skill_result is not None and skill_result.reason is not None
                    else result.failure_detail
                ),
                "control_frequency_hz": env.control_freq,
            }
            if result.success:
                episode = recorder.episode(metadata)
                collected.append(episode)
                print(
                    json.dumps(
                        {
                            "collected": len(collected),
                            "episode_length": episode.length,
                            "attempt_count": metadata["attempt_count"],
                            "initial_cube_position": metadata[
                                "initial_cube_position"
                            ],
                            "displacement": displacement,
                        },
                        sort_keys=True,
                    )
                )
            else:
                rejected_executions += 1
                print(json.dumps({"rejected_expert_execution": metadata}, sort_keys=True))
        finally:
            env.close()

    dataset = PushDataset(tuple(collected))
    report = validate_push_dataset(dataset)
    if not report.valid:
        raise RuntimeError(f"collected dataset is invalid: {report.errors}")
    save_push_dataset(output, dataset)
    print(
        json.dumps(
            {
                "output": str(output),
                "rejected_executions": rejected_executions,
                "placement_rejections": placement_rejections,
                "validation": {
                    "valid": report.valid,
                    "errors": list(report.errors),
                    "statistics": report.statistics,
                },
            },
            sort_keys=True,
        )
    )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="data/push_demos.npz")
    args = parser.parse_args()
    collect_push_demonstrations(
        episodes=args.episodes,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
