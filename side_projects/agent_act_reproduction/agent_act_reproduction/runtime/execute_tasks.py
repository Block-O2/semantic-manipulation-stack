"""Three-task command -> MockAgent -> Router -> ACT -> Panda runtime."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from agent_act_reproduction.agent import MockAgent
from agent_act_reproduction.policies import ACTMotorPolicy
from agent_act_reproduction.robot import PandaCartesianAdapter
from agent_act_reproduction.runtime.router import PolicyRouter
from agent_act_reproduction.sim import make_bottle_env, make_draw_env, make_tissue_env
from agent_act_reproduction.tasks import (
    BottleObservation,
    BottleSuccessChecker,
    DrawObservation,
    DrawSuccessChecker,
    TissueObservation,
    TissueSuccessChecker,
)


TASKS = {
    "place_bottle_on_shelf": (make_bottle_env, BottleObservation, BottleSuccessChecker, 320),
    "pull_tissue_from_box": (make_tissue_env, TissueObservation, TissueSuccessChecker, 280),
    "draw_horizontal_line": (make_draw_env, DrawObservation, DrawSuccessChecker, 430),
}


def execute_command(
    *,
    command: str,
    checkpoints: dict[str, Path],
    seed: int,
    perturbation_m: float,
    device: str,
    render: bool = False,
    offscreen: bool = False,
    max_steps: int | None = None,
    frame_callback: Callable[[object, int], None] | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    response = MockAgent().select(command)
    if response.status != "OK" or response.request is None:
        return response.to_dict()
    skill = response.request.skill
    router = PolicyRouter(checkpoints)
    checkpoint = router.checkpoint_for(skill)
    env_factory, observation_type, checker_type, default_steps = TASKS[skill]
    motor_policy = ACTMotorPolicy(checkpoint, device=device)
    env = env_factory(render=render, offscreen=offscreen, seed=seed)
    checker = checker_type()
    stable_success_steps = 0
    clips: Counter[str] = Counter()
    try:
        env.configure_episode(seed=seed, perturbation_m=perturbation_m)
        env.reset()
        adapter = PandaCartesianAdapter(env)
        motor_policy.reset()
        if frame_callback:
            frame_callback(env, 0)
        for step in range(1, (max_steps or default_steps) + 1):
            raw_action = motor_policy.select_action(observation_type.read(env))
            executed = adapter.execute(raw_action)
            for axis, raw, clipped in zip(("x", "y", "z", "gripper"), raw_action, executed):
                if not np.isclose(raw, clipped):
                    clips[axis] += 1
            if frame_callback:
                frame_callback(env, step)
            status = checker.check(env)
            stable_success_steps = stable_success_steps + 1 if status.success else 0
            if stable_success_steps >= 8:
                break
        final_status = checker.check(env)
        result = {
            "command": command,
            "agent": response.to_dict(),
            "selected_skill": skill,
            "checkpoint": str(checkpoint),
            "policy_class": "lerobot.policies.act.modeling_act.ACTPolicy",
            "device": str(motor_policy.device),
            "seed": seed,
            "perturbation_m": perturbation_m,
            "episode_length": step,
            "success": stable_success_steps >= 8 and final_status.success,
            "success_details": asdict(final_status),
            "action_clip_counts": dict(sorted(clips.items())),
            "hidden_scripted_assistance": False,
        }
    finally:
        env.close()
    if verbose:
        print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command")
    parser.add_argument("--bottle", type=Path, default=Path("checkpoints/bottle_act.pt"))
    parser.add_argument("--tissue", type=Path, default=Path("checkpoints/tissue_act.pt"))
    parser.add_argument("--draw", type=Path, default=Path("checkpoints/draw_act.pt"))
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--perturbation-m", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    result = execute_command(
        command=args.command,
        checkpoints={
            "place_bottle_on_shelf": args.bottle,
            "pull_tissue_from_box": args.tissue,
            "draw_horizontal_line": args.draw,
        },
        seed=args.seed,
        perturbation_m=args.perturbation_m,
        device=args.device,
        render=args.render,
    )
    if not result.get("success", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
