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
        assert world.object_names == ("red_cube", "blue_target")
        assert set(world.snapshot()) == {"red_cube", "blue_target"}
        assert np.isfinite(world.pose("red_cube").position).all()
        assert world.linear_velocity("red_cube").shape == (3,)
        assert world.target_half_extents("blue_target").shape == (3,)
        assert not world.is_inside_target("red_cube", "blue_target")
        assert world.is_stable("red_cube")
        assert not world.is_grasped("red_cube")
        semantic = world.semantic_state()
        assert semantic.robot.holding is None
        assert semantic.objects["red_cube"].reachable
        assert semantic.targets["blue_target"].reachable
        assert not semantic.relations["red_cube_inside_blue_target"]
    finally:
        env.close()
