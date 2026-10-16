"""MJLab five-tuple -> original HoST four-group PPO contract."""


class HostAdapter:
    def __init__(self, env):
        self.env = env.unwrapped
        if not self.env.cfg.auto_reset:
            raise ValueError("Original HoST PPO adapter requires auto_reset=True")

    def reset(self, seed=None):
        import torch

        self.env.reset(seed=seed)
        # Original BaseTask.reset() performs one zero-action simulation step.
        obs, critic, _, _, _ = self.step(torch.zeros(self.env.num_envs, 12, device=self.env.device))
        self._check_observations(obs, critic)
        return obs, critic

    def step(self, actions):
        obs, _scalar, terminated, truncated, info = self.env.step(actions)
        s = self.env.host_state
        if s.reward_step != self.env.common_step_counter:
            raise RuntimeError("HoST aggregate_rewards step event is missing or out of order")
        self._check_observations(obs["actor"], obs.get("critic", obs["actor"]))
        info = dict(info)
        info["time_outs"] = truncated
        info["terminal_observations"] = s.terminal_observations.clone()
        return (
            obs["actor"],
            obs.get("critic", obs["actor"]),
            s.group_rewards.clone(),
            terminated | truncated,
            info,
        )

    def _check_observations(self, actor, critic):
        expected = (self.env.num_envs, 258)
        if actor.shape != expected or critic.shape != expected:
            raise ValueError(
                "Original HoST networks and mixed-observation smoothness require "
                f"actor and critic observations of shape {expected}"
            )
