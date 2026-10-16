"""MJLab-compatible runner around the unchanged original HoST Multi-Critic PPO."""

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json
import math
import os
from pathlib import Path
import subprocess

import torch
from torch.utils.tensorboard import SummaryWriter

from .actor_critic import ActorCritic
from .ppo import PPO
from host_mjlab.host.settings import GROUPS


class HostAdapter:
    def __init__(self, env, obs_groups=None):
        self.env = env.unwrapped
        groups = obs_groups or {
            "actor": ("actor",),
            "critic": (("critic",) if "critic" in self.env.cfg.observations else ("actor",)),
        }
        self.actor_groups = tuple(groups["actor"])
        self.critic_groups = tuple(groups["critic"])
        dimensions = self.env.observation_manager.group_obs_dim

        def size(names):
            if not names or any(not isinstance(dimensions[name], tuple) for name in names):
                raise ValueError("HoST requires concatenated tensor observation groups")
            return sum(math.prod(dimensions[name]) for name in names)

        self.actor_dim, self.critic_dim = size(self.actor_groups), size(self.critic_groups)
        self.action_dim = self.env.action_manager.total_action_dim
        if self.actor_dim != self.critic_dim:
            raise ValueError(
                "Original HoST smoothness loss requires equal actor/critic input widths"
            )

        if not self.env.cfg.auto_reset:
            raise ValueError("Original HoST PPO adapter requires auto_reset=True")

    def reset(self, seed=None):
        import torch

        self.env.reset(seed=seed)
        # Original BaseTask.reset() performs one zero-action simulation step.
        obs, critic, _, _, _ = self.step(
            torch.zeros(self.env.num_envs, self.action_dim, device=self.env.device)
        )
        self._check_observations(obs, critic)
        return obs, critic

    def step(self, actions):
        obs, _scalar, terminated, truncated, info = self.env.step(actions)
        s = self.env.host_state
        manager = self.env.reward_manager
        if manager.step_counter != self.env.common_step_counter:
            raise RuntimeError("HoST reward manager did not evaluate this step")
        actor = self.select_observations(obs, self.actor_groups)
        critic = self.select_observations(obs, self.critic_groups)
        self._check_observations(actor, critic)
        info = dict(info)
        info["time_outs"] = truncated
        info["terminal_observations"] = s.terminal_observations.clone()
        return (
            actor,
            critic,
            manager.group_rewards.clone(),
            terminated | truncated,
            info,
        )

    @staticmethod
    def select_observations(observations, names):
        tensors = [observations[name].reshape(observations[name].shape[0], -1) for name in names]
        return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=-1)

    def _check_observations(self, actor, critic):
        if actor.shape != (self.env.num_envs, self.actor_dim) or critic.shape != (
            self.env.num_envs,
            self.critic_dim,
        ):
            raise ValueError("Observation shape changed after HoST network initialization")


class HostRunner:
    def __init__(self, env, train_cfg, log_dir=None, device=None):
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("Original HoST PPO has no distributed gradient synchronization")
        # Native MJLab CLI passes RslRlVecEnvWrapper; training must use our
        # four-group adapter rather than that wrapper's scalar reward.
        self.env = env.unwrapped
        self.device = torch.device(device or self.env.device)
        if self.device != torch.device(self.env.device):
            raise ValueError("HoST policy, storage and environment must share a device")
        if train_cfg is None:
            raise ValueError("Pass the task's native HostRunnerCfg to HostRunner")
        cfg = asdict(train_cfg) if is_dataclass(train_cfg) else deepcopy(train_cfg)
        self.adapter = HostAdapter(self.env, cfg.get("obs_groups"))
        if cfg.get("num_critics", len(GROUPS)) != len(GROUPS):
            raise ValueError("HoST requires four independent critics")
        self.reward_group_weights = list(cfg["reward_group_weights"])
        if len(self.reward_group_weights) != len(GROUPS):
            raise ValueError("Expected one advantage weight per HoST reward group")
        self.num_steps_per_env = cfg["num_steps_per_env"]
        self.save_interval = cfg["save_interval"]
        if self.num_steps_per_env < 2 or self.save_interval < 1:
            raise ValueError(
                "HoST requires at least two rollout steps and a positive save interval"
            )
        batch_size = self.env.num_envs * (self.num_steps_per_env - 1)
        if batch_size < cfg["algorithm"]["num_mini_batches"]:
            raise ValueError("Too few rollout transitions for the configured PPO minibatches")
        self.policy = ActorCritic(
            self.adapter.actor_dim,
            self.adapter.critic_dim,
            self.adapter.action_dim,
            len(GROUPS),
            **cfg["policy"],
        ).to(self.device)
        self.alg = PPO(
            self.policy, self.reward_group_weights, device=self.device, **cfg["algorithm"]
        )
        critic_shape = (
            [self.adapter.critic_dim]
            if self.adapter.critic_groups != self.adapter.actor_groups
            else [None]
        )
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.adapter.actor_dim],
            critic_shape,
            [self.adapter.action_dim],
            len(GROUPS),
        )
        self.cfg = cfg
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.writer = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir))
            (self.log_dir / "config.json").write_text(
                json.dumps(
                    {
                        "observations": asdict(self.env.cfg.host_observations),
                        "runner": cfg,
                        "reward_group_weights": self.reward_group_weights,
                        "num_steps_per_env": self.num_steps_per_env,
                        "reward_weights": {
                            name: self.env.reward_manager.get_term_cfg(name).weight
                            for name in self.env.reward_manager.active_terms
                        },
                    },
                    indent=2,
                )
            )
        self.iteration = 0

    def learn(self, num_learning_iterations, init_at_random_ep_len=True):
        obs, critic = self.adapter.reset(seed=self.env.cfg.seed)
        if init_at_random_ep_len:
            self.env.episode_length_buf.copy_(
                torch.randint_like(
                    self.env.episode_length_buf, high=int(self.env.max_episode_length)
                )
            )
        self.policy.train()
        for _ in range(num_learning_iterations):
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    action = self.alg.act(obs, critic)
                    obs, critic, rewards, dones, infos = self.adapter.step(action)
                    self.alg.process_env_step(rewards, dones, infos)
                filled = self.alg.storage.step
                self.alg.compute_returns(critic)
            value, policy = self.alg.update()
            self.iteration += 1
            if self.writer is not None:
                self.writer.add_scalar("Loss/value", value, self.iteration)
                self.writer.add_scalar("Loss/surrogate", policy, self.iteration)
                self.writer.add_scalar("Rollout/stored_steps", filled, self.iteration)
                self.writer.add_scalar(
                    "Learning/learning_rate", self.alg.learning_rate, self.iteration
                )
                for i, name in enumerate(GROUPS):
                    self.writer.add_scalar(
                        "Reward/" + name, rewards[:, i].mean().item(), self.iteration
                    )
            print(
                f"iteration={self.iteration} value_loss={value:.6g} "
                f"surrogate_loss={policy:.6g} stored={filled}/{self.num_steps_per_env}",
                flush=True,
            )
            if self.log_dir is not None and self.iteration % self.save_interval == 0:
                self.save(self.log_dir / f"model_{self.iteration}.pt")
        if self.log_dir is not None:
            self.save(self.log_dir / f"model_{self.iteration}.pt")
            self.writer.flush()

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.iteration,
                "learning_rate": self.alg.learning_rate,
                "host_config": self.cfg,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True, *, load_cfg=None, strict=True, map_location=None):
        """Accept local CLI and MJLab 1.6.0 train/play checkpoint signatures.

        Original checkpoints lack learning_rate: recover it from saved Adam
        groups. New checkpoints restore the explicit adaptive rate to BOTH PPO
        and Adam. Actor-only playback leaves all optimization state untouched.
        """
        data = torch.load(path, map_location=map_location or self.device, weights_only=False)
        selection = (
            {"actor": True, "critic": True, "optimizer": load_optimizer, "iteration": True}
            if load_cfg is None
            else dict(load_cfg)
        )
        selection["optimizer"] = selection.get("optimizer", False) and load_optimizer
        model = data["model_state_dict"]
        if selection.get("actor") and selection.get("critic"):
            self.policy.load_state_dict(model, strict=strict)
        else:
            if selection.get("actor"):
                self.policy.actor.load_state_dict(
                    {
                        key.removeprefix("actor."): value
                        for key, value in model.items()
                        if key.startswith("actor.")
                    },
                    strict=strict,
                )
                if "std" in model:
                    with torch.no_grad():
                        self.policy.std.copy_(model["std"].to(self.device))
                elif strict:
                    raise KeyError("HoST checkpoint is missing the actor's learned std")
            if selection.get("critic"):
                self.policy.critics.load_state_dict(
                    {
                        key.removeprefix("critics."): value
                        for key, value in model.items()
                        if key.startswith("critics.")
                    },
                    strict=strict,
                )
        if selection["optimizer"]:
            optimizer = data["optimizer_state_dict"]
            rate = float(data.get("learning_rate", optimizer["param_groups"][0]["lr"]))
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("Checkpoint learning rate must be positive and finite")
            self.alg.optimizer.load_state_dict(optimizer)
            self.alg.learning_rate = rate
            for group in self.alg.optimizer.param_groups:
                group["lr"] = rate
        if selection.get("iteration"):
            self.iteration = int(data["iter"])
        return data.get("infos")

    def get_inference_policy(self, device=None):
        self.policy.eval()
        target = torch.device(device) if device is not None else self.device
        self.policy.to(target)

        def policy(observations):
            # MJLab viewer supplies TensorDict; the local CLI supplies a tensor.
            actor_obs = (
                observations
                if isinstance(observations, torch.Tensor)
                else self.adapter.select_observations(observations, self.adapter.actor_groups)
            )
            return self.policy.act_inference(actor_obs.to(target))

        return policy

    def add_git_repo_to_log(self, path):
        """Hook invoked by native MJLab training; log locally without changing Git."""
        if self.log_dir is None:
            return
        source = Path(path).resolve()
        cwd = source if source.is_dir() else source.parent
        try:
            result = subprocess.run(
                ["git", "-C", str(cwd), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            revision = result.stdout.strip() or "unavailable"
        except OSError:
            revision = "unavailable"
        with (self.log_dir / "git_sources.txt").open("a") as stream:
            stream.write(f"{source}: {revision}\n")
