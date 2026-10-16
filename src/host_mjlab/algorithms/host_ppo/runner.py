"""Small standalone runner retaining HoST's rollout and optimization schedule."""

import json
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from .actor_critic import ActorCritic
from .ppo import PPO
from .adapter import HostAdapter
from host_mjlab.host.settings import DATA


class HostRunner:
    def __init__(self, env, train_cfg=None, log_dir=None, device=None):
        # Never route this task through MJLab's single-critic default runner.
        self.env = env.unwrapped
        self.device = device or self.env.device
        self.adapter = HostAdapter(self.env)
        cfg = DATA["ppo"]
        self.policy = ActorCritic(258, 258, 12, 4, **cfg["policy"]).to(self.device)
        self.alg = PPO(
            self.policy,
            DATA["env"]["rewards"]["reward_group_weights"],
            device=self.device,
            **cfg["algorithm"],
        )
        self.alg.init_storage(self.env.num_envs, 50, [258], [None], [12], 4)
        self.log_dir = Path(log_dir or "logs/host/default")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self.iteration = 0
        (self.log_dir / "config.json").write_text(json.dumps(DATA, indent=2))

    def learn(self, num_learning_iterations, init_at_random_ep_len=True):
        obs, critic = self.adapter.reset(seed=self.env.cfg.seed)
        if init_at_random_ep_len:
            self.env.episode_length_buf.copy_(
                torch.randint_like(self.env.episode_length_buf, high=500)
            )
        self.policy.train()
        for _ in range(num_learning_iterations):
            with torch.inference_mode():
                for _ in range(50):
                    action = self.alg.act(obs, critic)
                    obs, critic, rewards, dones, infos = self.adapter.step(action)
                    self.alg.process_env_step(rewards, dones, infos)
                filled = self.alg.storage.step
                self.alg.compute_returns(critic)
            value, policy = self.alg.update()
            self.iteration += 1
            self.writer.add_scalar("Loss/value", value, self.iteration)
            self.writer.add_scalar("Loss/surrogate", policy, self.iteration)
            self.writer.add_scalar("Rollout/stored_steps", filled, self.iteration)
            for i, name in enumerate(DATA["env"]["rewards"]["reward_groups"]):
                self.writer.add_scalar(
                    "Reward/" + name, rewards[:, i].mean().item(), self.iteration
                )
            print(
                f"iteration={self.iteration} value_loss={value:.6g} surrogate_loss={policy:.6g} stored={filled}/50",
                flush=True,
            )
            if self.iteration % 100 == 0:
                self.save(self.log_dir / f"model_{self.iteration}.pt")
        self.save(self.log_dir / f"model_{self.iteration}.pt")
        self.writer.flush()

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.iteration,
                "learning_rate": self.alg.learning_rate,
                "host_config": DATA,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(data["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.iteration = data["iter"]
        self.alg.learning_rate = data.get("learning_rate", self.alg.learning_rate)

    def get_inference_policy(self, device=None):
        self.policy.eval()
        if device is not None:
            self.policy.to(device)
        return self.policy.act_inference
