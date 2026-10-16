"""MJLab 1.6 reward-manager extension for original HoST group reductions."""

import torch
from mjlab.managers.reward_manager import RewardManager
from .settings import GROUPS


class HostRewardManager(RewardManager):
    """Evaluate arbitrary native reward callables once, with no term-side cache.

    MJLab 1.6 constructs RewardManager directly. The startup installer reuses
    its publicly resolved term configs so class-based terms are not constructed
    a second time. Only this subclass owns the inherited bookkeeping buffers.
    """

    def __init__(self, env, resolved_manager):
        super().__init__({}, env, scale_by_dt=False)
        self.cfg = {
            name: resolved_manager.get_term_cfg(name) for name in resolved_manager.active_terms
        }
        self._term_names = list(self.cfg)
        self._term_cfgs = list(self.cfg.values())
        self._class_term_cfgs = [
            cfg for cfg in self._term_cfgs if callable(getattr(cfg.func, "reset", None))
        ]
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in self._term_names
        }
        self._step_reward = torch.zeros(self.num_envs, len(self._term_names), device=self.device)
        self.group_rewards = torch.zeros(self.num_envs, len(GROUPS), device=self.device)
        self.step_counter = -1
        self._groups = {}
        for name in self._term_names:
            group = env.cfg.reward_term_groups.get(name, name.split("_", 1)[0])
            if group not in GROUPS:
                raise ValueError(f"Assign reward {name!r} to one of {GROUPS}")
            self._groups[name] = group
        if any(group not in GROUPS[1:] for group in env.cfg.reward_group_positive):
            raise ValueError("Positive clipping applies only to additive HoST groups")

    def compute(self, dt):
        self.group_rewards.zero_()
        self.group_rewards[:, GROUPS.index("task")] = 1.0
        for column, name in enumerate(self.active_terms):
            cfg = self.get_term_cfg(name)
            self._step_reward[:, column] = 0
            if cfg.weight == 0:
                continue
            raw = cfg.func(self._env, **cfg.params)
            self._check_term_shape(name, raw)
            if raw.device != self.group_rewards.device:
                raise ValueError(f"Reward {name!r} returned data on the wrong device")
            group = self._groups[name]
            coefficient = cfg.weight if group == "task" else cfg.weight * dt
            value = raw * coefficient
            index = GROUPS.index(group)
            if group == "task":
                self.group_rewards[:, index] *= value
            else:
                self.group_rewards[:, index] += value
                if group in self._env.cfg.reward_group_positive:
                    self.group_rewards[:, index].clamp_(min=0)
            # Keep original HoST numerical behavior in group rewards. Only
            # MJLab's diagnostic term logs use its usual NaN sanitization.
            logged = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            self._step_reward[:, column] = logged
            self._episode_sums[name] += logged
        self._reward_buf.copy_(self.group_rewards.sum(dim=-1))
        self.step_counter = self._env.common_step_counter
        return self._reward_buf


def install_reward_manager(env, env_ids):
    """Startup-only extension point; no MJLab core or task registry patching."""
    if env.cfg.scale_rewards_by_dt:
        raise ValueError("HoST applies group-specific dt scaling; disable global scaling")
    if not isinstance(env.reward_manager, HostRewardManager):
        env.reward_manager = HostRewardManager(env, env.reward_manager)


def reward_step(env, env_ids=None):
    """Log already-reduced groups and capture terminal data before auto-reset."""
    from .mdp.state import state

    manager = env.reward_manager
    if (
        not isinstance(manager, HostRewardManager)
        or manager.step_counter != env.common_step_counter
    ):
        raise RuntimeError("HoST reward manager was not installed or evaluated")
    for index, group in enumerate(GROUPS):
        env.extras["log"]["HoST/" + group] = manager.group_rewards[:, index].mean()
    state(env).capture_terminal_observations(env.reset_buf.nonzero(as_tuple=False).flatten())
