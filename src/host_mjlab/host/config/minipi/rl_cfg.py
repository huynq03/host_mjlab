"""Native MJLab runner configuration for the original four-critic HoST baseline."""

from dataclasses import dataclass, field
from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class HostPolicyCfg:
    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    init_noise_std: float = 0.8


@dataclass
class HostPpoCfg:
    clip_param: float = 0.2
    desired_kl: float = 0.01
    entropy_coef: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 0.001
    max_grad_norm: float = 1.0
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    schedule: str = "adaptive"
    smoothness_lower_bound: float = 0.1
    smoothness_upper_bound: float = 1.0
    use_clipped_value_loss: bool = True
    value_loss_coef: float = 1.0
    value_smoothness_coef: float = 0.1


@dataclass
class HostRunnerCfg(RslRlBaseRunnerCfg):
    policy: HostPolicyCfg = field(default_factory=HostPolicyCfg)
    algorithm: HostPpoCfg = field(default_factory=HostPpoCfg)
    reward_group_weights: tuple[float, ...] = (2.5, 0.1, 1.0, 1.0)
    num_critics: int = 4


def runner_cfg():
    return HostRunnerCfg(
        seed=1,
        logger="tensorboard",
        num_steps_per_env=50,
        max_iterations=12000,
        save_interval=100,
        obs_groups={"actor": ("actor",), "critic": ("actor",)},
    )
