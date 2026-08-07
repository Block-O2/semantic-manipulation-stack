"""Move the Panda through safe Cartesian targets and exercise its gripper."""

from __future__ import annotations

import argparse
import time

import numpy as np

from robot import PandaRobot, Pose
from sim import make_environment
from world import WorldModel


def _format_pose(pose: Pose) -> str:
    position = np.array2string(pose.position, precision=4, suppress_small=True)
    quaternion = np.array2string(pose.quaternion, precision=4, suppress_small=True)
    return f"position={position} m, quaternion_xyzw={quaternion}"


def run_demo(*, render: bool, inspect_seconds: float, randomize_cube: bool) -> None:
    env = make_environment(render=render, randomize_cube=randomize_cube, seed=7)
    try:
        env.reset()
        robot = PandaRobot(env)
        world = WorldModel(env)

        initial = robot.pose
        print(f"Initial end-effector pose: {_format_pose(initial)}")
        for name, pose in world.snapshot().items():
            print(f"Initial {name} pose: {_format_pose(pose)}")

        offsets = (
            np.array([0.00, 0.00, 0.06]),
            np.array([0.08, 0.00, 0.06]),
            np.array([0.04, 0.08, 0.08]),
            np.array([0.00, 0.00, 0.03]),
        )
        for index, offset in enumerate(offsets, start=1):
            target = Pose(initial.position + offset, initial.quaternion)
            result = robot.move_end_effector(target)
            print(
                f"Target {index}: reached={result.reached}, steps={result.steps}, "
                f"position_error={result.position_error:.4f} m, "
                f"orientation_error={result.orientation_error:.4f} rad"
            )
            if not result.reached:
                raise RuntimeError(f"Cartesian target {index} did not converge")

        print("Opening gripper...")
        robot.open_gripper()
        print("Closing gripper...")
        robot.close_gripper()
        print("Opening gripper for inspection...")
        robot.open_gripper()

        inspect_steps = round(max(0.0, inspect_seconds) * env.control_freq)
        if inspect_steps:
            print(f"Holding viewer for {inspect_seconds:.1f} seconds...")
            for _ in range(inspect_steps):
                robot.hold(1)
                if render:
                    time.sleep(1.0 / env.control_freq)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="run headlessly (useful for CI and smoke tests)",
    )
    parser.add_argument(
        "--inspect-seconds",
        type=float,
        default=8.0,
        help="simulation seconds to keep stepping after the motion",
    )
    parser.add_argument(
        "--randomize-cube",
        action="store_true",
        help="sample the cube position within a small reachable range",
    )
    args = parser.parse_args()
    run_demo(
        render=not args.no_render,
        inspect_seconds=args.inspect_seconds,
        randomize_cube=args.randomize_cube,
    )


if __name__ == "__main__":
    main()
