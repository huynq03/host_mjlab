"""Registry-visible original HoST settings; custom runner owns four-critic PPO."""

from copy import deepcopy
from dataclasses import dataclass, field
from mjlab.rl import RslRlBaseRunnerCfg
from host_mjlab.host.settings import DATA


@dataclass
class HostRunnerCfg(RslRlBaseRunnerCfg):
    policy: dict = field(default_factory=lambda: deepcopy(DATA["ppo"]["policy"]))
    algorithm: dict = field(default_factory=lambda: deepcopy(DATA["ppo"]["algorithm"]))
    reward_group_weights: tuple[float, ...] = (2.5, 0.1, 1.0, 1.0)
    num_critics: int = 4


def runner_cfg():
    return HostRunnerCfg(
        seed=1,
        num_steps_per_env=50,
        max_iterations=12000,
        save_interval=100,
        obs_groups={"actor": ("actor",), "critic": ("actor",)},
    )
