"""Interactive semantic-step playground for changing the MuJoCo world."""

from __future__ import annotations

import argparse
import json
import time

from planner import (
    Goal,
    OpenAICompatiblePlanner,
    PlanValidator,
    RuleBasedPlanner,
    SkillRegistry,
)
from playground import WorldController
from primitives import ManipulationPrimitives
from robot import PandaRobot
from runtime import AgentBoundaryEvent, AgentRuntime, SemanticSkillFactory
from sim import make_environment
from world import WorldModel, WorldState


HELP = """Commands:
  help
  state
  move object <name> <x> <y>
  move target <name> <x> <y>
  drop
  occupy <target> <object>
  remove <object>
  restore <object>
  continue
  quit
"""


class PlaygroundQuit(Exception):
    pass


def _compact_state(state: WorldState) -> dict[str, object]:
    return {
        "holding": state.robot.holding,
        "objects": {
            name: {
                "exists": value.exists,
                "position": value.pose.position if value.pose else None,
                "grasped": value.grasped,
                "reachable": value.reachable,
            }
            for name, value in state.objects.items()
        },
        "targets": {
            name: {
                "position": value.pose.position if value.pose else None,
                "reachable": value.reachable,
                "occupied": value.occupied,
                "occupied_by": list(value.occupied_by),
                "role": value.role,
            }
            for name, value in state.targets.items()
        },
        "push_regions": {
            name: {
                "center": value.center,
                "lower_xy": value.lower_xy,
                "upper_xy": value.upper_xy,
                "reachable": value.reachable,
            }
            for name, value in state.push_regions.items()
        },
        "true_relations": [
            name for name, value in sorted(state.relations.items()) if value
        ],
    }


def _handle_command(
    command: str,
    controller: WorldController,
    world: WorldModel,
) -> bool:
    parts = command.split()
    if not parts:
        return False
    if parts == ["help"]:
        print(HELP)
    elif parts == ["state"]:
        print(json.dumps(_compact_state(world.semantic_state()), indent=2))
    elif parts == ["continue"]:
        return True
    elif parts == ["quit"]:
        raise PlaygroundQuit
    elif len(parts) == 5 and parts[:2] == ["move", "object"]:
        update = controller.move_object(parts[2], float(parts[3]), float(parts[4]))
        print(f"WORLD UPDATE: {update}")
    elif len(parts) == 5 and parts[:2] == ["move", "target"]:
        update = controller.move_target(parts[2], float(parts[3]), float(parts[4]))
        print(f"WORLD UPDATE: {update}")
    elif parts == ["drop"]:
        print(f"WORLD UPDATE: {controller.drop_held_object()}")
    elif len(parts) == 3 and parts[0] == "occupy":
        print(
            "WORLD UPDATE: "
            f"{controller.place_object_in_target(parts[2], parts[1])}"
        )
    elif len(parts) == 2 and parts[0] == "remove":
        print(f"WORLD UPDATE: {controller.remove_object(parts[1])}")
    elif len(parts) == 2 and parts[0] == "restore":
        print(f"WORLD UPDATE: {controller.restore_object(parts[1])}")
    else:
        raise ValueError("Unknown command; enter 'help' for syntax")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", choices=("rule", "llm"), default="rule")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--inspect-seconds", type=float, default=8.0)
    args = parser.parse_args()

    env = make_environment(
        render=not args.no_render,
        randomize_cube=args.randomize,
        seed=71,
    )
    try:
        env.reset()
        robot = PandaRobot(env)
        world = WorldModel(env)
        controller = WorldController(env, world)
        primitives = ManipulationPrimitives(robot)
        registry = SkillRegistry.standard()
        planner_backend = (
            OpenAICompatiblePlanner() if args.planner == "llm" else RuleBasedPlanner()
        )

        def pause(event: AgentBoundaryEvent) -> None:
            print(
                f"\nREADY TO EXECUTE: {event.step.skill}({event.step.args})\n"
                "Enter world commands, then 'continue'."
            )
            print(json.dumps(_compact_state(event.world_state), indent=2))
            while True:
                try:
                    if _handle_command(input("> ").strip(), controller, world):
                        return
                except ValueError as exc:
                    print(f"ERROR: {exc}")

        runtime = AgentRuntime(
            planner=planner_backend,
            registry=registry,
            validator=PlanValidator(registry),
            observer=world.semantic_state,
            skill_factory=SemanticSkillFactory(world, primitives),
            logger=print,
        )
        goal = Goal.put_inside(
            "Put the red cube inside the blue target.",
            "red_cube",
            "blue_target",
        )
        print(HELP)
        try:
            result = runtime.run(goal, before_step=pause)
        except PlaygroundQuit:
            print("PLAYGROUND QUIT")
            return
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

        inspect_steps = round(max(0.0, args.inspect_seconds) * env.control_freq)
        for _ in range(inspect_steps):
            robot.hold(1)
            if not args.no_render:
                time.sleep(1.0 / env.control_freq)
    finally:
        env.close()


if __name__ == "__main__":
    main()
