import numpy as np

from robot import PandaRobot
from sim import make_environment
from world import WorldModel


def test_headless_environment_exposes_expected_state() -> None:
    env = make_environment(render=False, seed=3)
    try:
        env.reset()
        robot = PandaRobot(env)
        world = WorldModel(env)

        assert env.action_spec[0].shape == (7,)
        assert robot.pose.position.shape == (3,)
        assert world.object_names == (
            "red_cube",
            "green_cube",
            "blue_cube",
            "red_target",
            "blue_target",
            "temporary_area",
        )
        assert set(world.snapshot()) == set(world.object_names)
        assert np.isfinite(world.pose("red_cube").position).all()
        assert world.linear_velocity("red_cube").shape == (3,)
        assert world.target_half_extents("blue_target").shape == (3,)
        assert not world.is_inside_target("red_cube", "blue_target")
        assert world.is_stable("red_cube")
        assert not world.is_grasped("red_cube")
        semantic = world.semantic_state()
        assert semantic.robot.holding is None
        assert semantic.objects["red_cube"].reachable
        assert semantic.objects["green_cube"].reachable
        assert semantic.objects["blue_cube"].reachable
        assert semantic.targets["blue_target"].reachable
        assert semantic.targets["red_target"].reachable
        assert semantic.targets["temporary_area"].reachable
        assert semantic.targets["temporary_area"].role == "temporary"
        assert semantic.targets["temporary_area"].occupied_by == ()
        assert not semantic.relations["red_cube_inside_blue_target"]
    finally:
        env.close()
