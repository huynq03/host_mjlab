"""MJLab five-tuple -> original HoST four-group PPO contract."""


class HostAdapter:
    def __init__(self, env):
        self.env = env

    def reset(self, seed=None):
        import torch
        self.env.reset(seed=seed)
        # Original BaseTask.reset() performs one zero-action simulation step.
        obs, critic, _, _, _ = self.step(
            torch.zeros(self.env.num_envs, 12, device=self.env.device)
        )
        return obs, critic

    def step(self, actions):
        obs, _scalar, terminated, truncated, info = self.env.step(actions)
        s = self.env.host_state
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
