"""Shared, explicit schemas and Bottle task constants."""

from __future__ import annotations

from dataclasses import asdict, dataclass


ROBOT_STATE_FEATURES = (
    "eef_position_x_m",
    "eef_position_y_m",
    "eef_position_z_m",
    "gripper_width_m",
)

ENV_STATE_FEATURES = (
    "bottle_position_x_m",
    "bottle_position_y_m",
    "bottle_position_z_m",
    "bottle_velocity_x_mps",
    "bottle_velocity_y_mps",
    "bottle_velocity_z_mps",
    "shelf_target_x_m",
    "shelf_target_y_m",
    "shelf_target_z_m",
    "bottle_grasped_bool",
    "episode_progress_fraction",
)

ACTION_FEATURES = (
    "desired_eef_x_m",
    "desired_eef_y_m",
    "desired_eef_z_m",
    "gripper_command_open_minus1_close_plus1",
)


@dataclass(frozen=True)
class BottleSceneConfig:
    """Geometry in MuJoCo world coordinates; lengths are metres."""

    table_size: tuple[float, float, float] = (0.8, 0.8, 0.05)
    table_offset: tuple[float, float, float] = (0.0, 0.0, 0.8)
    bottle_radius: float = 0.022
    bottle_half_height: float = 0.055
    bottle_xy: tuple[float, float] = (0.05, -0.10)
    shelf_xy: tuple[float, float] = (0.10, 0.20)
    shelf_deck_half_size: tuple[float, float, float] = (0.09, 0.08, 0.012)
    shelf_deck_center_z: float = 0.90
    shelf_post_half_size: tuple[float, float, float] = (0.012, 0.07, 0.05)
    random_xy_m: float = 0.01
    random_yaw_rad: float = 0.10
    control_freq_hz: int = 20
    max_episode_steps: int = 420

    @property
    def table_top_z(self) -> float:
        return self.table_offset[2]

    @property
    def bottle_initial_z(self) -> float:
        return self.table_top_z + self.bottle_half_height

    @property
    def shelf_top_z(self) -> float:
        return self.shelf_deck_center_z + self.shelf_deck_half_size[2]

    @property
    def shelf_target(self) -> tuple[float, float, float]:
        return (
            self.shelf_xy[0],
            self.shelf_xy[1],
            self.shelf_top_z + self.bottle_half_height,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_SCENE = BottleSceneConfig()
