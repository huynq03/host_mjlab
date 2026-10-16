"""Typed observation settings owned by the native environment configuration.

original_config.py is historical reference data, not an active configuration.
"""

from dataclasses import dataclass

GROUPS = ("task", "regu", "style", "target")


@dataclass(kw_only=True)
class HostObservationCfg:
    history_length: int = 6
    angular_velocity_scale: float = 0.25
    joint_position_scale: float = 1.0
    joint_velocity_scale: float = 0.05
    angular_velocity_noise: float = 0.05
    gravity_noise: float = 0.05
    joint_position_noise: float = 0.01
    joint_velocity_noise: float = 0.075
    action_scale_noise: float = 0.05
    enable_noise: bool = True
    clip: float = 100.0
