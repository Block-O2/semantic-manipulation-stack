"""Compare classical and one-step BC Push on identical valid initial states."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from datasets import load_push_dataset
from evaluation.push_sampling import prepare_push_scene
from planner import Goal, PlanValidator, RuleBasedPlanner, SkillRegistry
from playground import WorldController
from primitives import ManipulationPrimitives
from robot import PandaRobot
from runtime import AgentRuntime, SemanticSkillFactory, create_push_backend
from sim import make_environment
from skills import PushBackend
from world import WorldModel


def run_backend_trial(
    *,
    backend: PushBackend,
    initial_xy: np.ndarray,
    seed: int,
) -> dict[str, object]:
    env = make_environment(render=False, randomize_cube=False, seed=seed)
    command_steps = 0
    try:
        env.reset()
        robot = PandaRobot(env)
        world = WorldModel(env)
        primitives = ManipulationPrimitives(robot)
        controller = WorldController(env, world)
        prepare_push_scene(world, robot, primitives, controller)
        controller.move_object(
            "red_cube",
            float(initial_xy[0]),
            float(initial_xy[1]),
        )
        primitives.wait(steps=10)
        initial = world.pose("red_cube").position.copy()
        if np.linalg.norm(initial[:2] - initial_xy) > 0.003:
            raise RuntimeError("matched initial state is not physically stable")

        def count_command(event) -> None:
            nonlocal command_steps
            command_steps += 1

        robot.add_command_observer(count_command)
        registry = SkillRegistry.standard()
        result = AgentRuntime(
            planner=RuleBasedPlanner(),
            registry=registry,
            validator=PlanValidator(registry),
            observer=world.semantic_state,
            skill_factory=SemanticSkillFactory(
                world,
                primitives,
                push_backend=backend,
            ),
            max_replans=0,
        ).run(
            Goal.push_to_region(
                "Push the red cube toward the right side of the table.",
                "red_cube",
                "right_side",
            )
        )
        robot.remove_command_observer(count_command)
        final = world.pose("red_cube").position.copy()
        target_satisfied = bool(
            world.is_inside_push_region("red_cube", "right_side")
            and world.is_on_table("red_cube")
        )
        skill_result = (
            result.executed_steps[-1].task_result.skill_results[-1]
            if result.executed_steps
            else None
        )
        return {
            "backend": backend.name,
            "success": result.success,
            "initial_position": initial.tolist(),
            "final_position": final.tolist(),
            "displacement": float(np.linalg.norm(final[:2] - initial[:2])),
            "target_satisfied": target_satisfied,
            "control_steps": command_steps,
            "attempts": skill_result.attempts if skill_result else 0,
            "failure_reason": (
                skill_result.reason.value
                if skill_result is not None and skill_result.reason is not None
                else result.failure_detail
            ),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    dataset = load_push_dataset(args.dataset)
    if args.trials <= 0 or args.trials > len(dataset.episodes):
        raise ValueError("trials must fit within the supplied dataset")
    backends = (
        create_push_backend("classical"),
        create_push_backend("bc", checkpoint=args.checkpoint),
    )
    records: list[dict[str, object]] = []
    skipped_states = 0
    for dataset_index, episode in enumerate(dataset.episodes):
        if len(records) // len(backends) >= args.trials:
            break
        initial_xy = np.asarray(
            episode.metadata["initial_cube_position"][:2],
            dtype=np.float64,
        )
        episode_seed = int(episode.metadata["random_seed"])
        pair: list[dict[str, object]] = []
        try:
            for backend in backends:
                pair.append(
                    run_backend_trial(
                        backend=backend,
                        initial_xy=initial_xy,
                        seed=episode_seed,
                    )
                )
        except RuntimeError as exc:
            skipped_states += 1
            print(
                json.dumps(
                    {
                        "skipped_dataset_index": dataset_index,
                        "reason": str(exc),
                    },
                    sort_keys=True,
                )
            )
            continue
        trial = len(records) // len(backends) + 1
        for record in pair:
            record["trial"] = trial
            record["dataset_index"] = dataset_index
            records.append(record)
            print(json.dumps(record, sort_keys=True))
    if len(records) // len(backends) != args.trials:
        raise RuntimeError("dataset did not contain enough replay-stable matched states")

    summary = {}
    for backend in backends:
        selected = [item for item in records if item["backend"] == backend.name]
        failures = Counter(
            str(item["failure_reason"])
            for item in selected
            if not item["success"]
        )
        summary[backend.name] = {
            "trials": len(selected),
            "successes": sum(bool(item["success"]) for item in selected),
            "failure_reasons": dict(sorted(failures.items())),
            "average_control_steps": float(
                np.mean([int(item["control_steps"]) for item in selected])
            ),
        }
    print(
        json.dumps(
            {"summary": summary, "skipped_initial_states": skipped_states},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
