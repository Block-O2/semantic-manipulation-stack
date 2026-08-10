"""Unified command -> MockAgent -> ACT -> Panda -> physical result runtime."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from agent_act_reproduction.agent import MockAgent
from agent_act_reproduction.policies import ACTMotorPolicy
from agent_act_reproduction.robot import PandaCartesianAdapter
from agent_act_reproduction.runtime import PolicyRouter
from agent_act_reproduction.sim import make_bottle_env
from agent_act_reproduction.tasks import BottleObservation, BottleSuccessChecker


def execute_command(
    *,
    command: str,
    checkpoint: Path,
    seed: int,
    perturbation_m: float,
    device: str,
    render: bool,
    max_steps: int = 320,
    verbose: bool = True,
) -> dict[str, object]:
    agent_response = MockAgent().select(command)
    if verbose:
        print("USER")
        print(command)
        print("\nAGENT")
        print(json.dumps(agent_response.to_dict(), sort_keys=True))
    if agent_response.status != "OK" or agent_response.request is None:
        return agent_response.to_dict()

    router = PolicyRouter({"place_bottle_on_shelf": checkpoint})
    selected_checkpoint = router.checkpoint_for(agent_response.request.skill)
    if verbose:
        print("\nPOLICY")
        print(f"selected_skill = {agent_response.request.skill}")
        print(f"checkpoint = {selected_checkpoint}")
        print("implementation = Hugging Face LeRobot ACTPolicy 0.4.4")

    # Runtime intentionally imports no expert module and has no recovery FSM.
    motor_policy = ACTMotorPolicy(selected_checkpoint, device=device)
    env = make_bottle_env(render=render, seed=seed)
    stable_success_steps = 0
    checker = BottleSuccessChecker()
    action_clip_counts: Counter[str] = Counter()
    try:
        env.configure_episode(seed=seed, perturbation_m=perturbation_m)
        env.reset()
        adapter = PandaCartesianAdapter(env)
        motor_policy.reset()
        initial_bottle_position = env.bottle_position().tolist()
        for step in range(1, max_steps + 1):
            observation = BottleObservation.read(env)
            raw_action = motor_policy.select_action(observation)
            executed = adapter.execute(raw_action)
            for axis, raw, clipped in zip(("x", "y", "z", "gripper"), raw_action, executed):
                if not np.isclose(raw, clipped):
                    action_clip_counts[axis] += 1
            status = checker.check(env)
            stable_success_steps = stable_success_steps + 1 if status.success else 0
            if verbose and (step == 1 or step % 25 == 0 or stable_success_steps == 1):
                print(
                    json.dumps(
                        {
                            "execution_step": step,
                            "bottle_position": env.bottle_position().round(5).tolist(),
                            "eef_position": env.end_effector_position().round(5).tolist(),
                            "grasped": env.is_grasping_bottle(),
                            "candidate_success": status.success,
                        },
                        sort_keys=True,
                    )
                )
            if stable_success_steps >= 10:
                break
        final_status = checker.check(env)
        result = {
            "command": command,
            "agent": agent_response.to_dict(),
            "selected_skill": agent_response.request.skill,
            "checkpoint": str(selected_checkpoint),
            "policy_class": "lerobot.policies.act.modeling_act.ACTPolicy",
            "device": str(motor_policy.device),
            "seed": seed,
            "perturbation_m": perturbation_m,
            "initial_bottle_position": initial_bottle_position,
            "final_bottle_position": env.bottle_position().tolist(),
            "episode_length": step,
            "success": stable_success_steps >= 10 and final_status.success,
            "success_details": asdict(final_status),
            "action_clip_counts": dict(sorted(action_clip_counts.items())),
            "hidden_scripted_assistance": False,
        }
    finally:
        env.close()
    if verbose:
        print("\nRESULT")
        print("SUCCESS" if result["success"] else "FAILURE")
        print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/bottle_act.pt"))
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--perturbation-m", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=320)
    args = parser.parse_args()
    result = execute_command(
        command=args.command,
        checkpoint=args.checkpoint,
        seed=args.seed,
        perturbation_m=args.perturbation_m,
        device=args.device,
        render=args.render,
        max_steps=args.max_steps,
    )
    if result.get("status") == "CANNOT_EXECUTE":
        raise SystemExit(2)
    if not result.get("success", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
