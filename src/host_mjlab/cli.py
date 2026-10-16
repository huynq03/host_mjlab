"""Train, headless playback/evaluation, and finite simulation smoke checks."""

import argparse
from datetime import datetime
from pathlib import Path
import json
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "play", "evaluate", "smoke"])
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--log-dir", type=Path)
    p.add_argument("--video", type=Path)
    args = p.parse_args()
    if args.num_envs <= 0 or args.steps <= 0 or args.iterations <= 0:
        p.error("--num-envs, --steps and --iterations must be positive")
    if args.mode in ("play", "evaluate") and args.checkpoint is None:
        p.error("--checkpoint is required for play/evaluate")
    from mjlab.envs import ManagerBasedRlEnv
    from host_mjlab.host.host_env_cfg import make_env_cfg
    from host_mjlab.algorithms.host_ppo.runner import HostRunner
    from host_mjlab.algorithms.host_ppo.adapter import HostAdapter

    torch.manual_seed(args.seed)
    cfg = make_env_cfg(args.num_envs)
    cfg.seed = args.seed
    env = ManagerBasedRlEnv(
        cfg, device=args.device, render_mode="rgb_array" if args.video else None
    )
    log_dir = args.log_dir or Path("logs/host") / datetime.now().strftime("%Y%m%d_%H%M%S")
    runner = None
    try:
        if args.mode == "train":
            runner = HostRunner(env, log_dir=log_dir)
            if args.checkpoint:
                runner.load(args.checkpoint)
            runner.learn(args.iterations)
            return
        adapter = HostAdapter(env)
        obs, critic = adapter.reset(seed=args.seed)
        runner = None
        if args.checkpoint:
            runner = HostRunner(env, log_dir=log_dir)
            runner.load(args.checkpoint, load_optimizer=False)
        frames = []
        total = torch.zeros(4, device=env.device)
        done_count = 0
        with torch.inference_mode():
            for _ in range(args.steps):
                action = (
                    runner.policy.act_inference(obs)
                    if runner
                    else torch.zeros(args.num_envs, 12, device=env.device)
                )
                obs, critic, rewards, dones, infos = adapter.step(action)
                if not torch.isfinite(obs).all() or not torch.isfinite(rewards).all():
                    raise RuntimeError("Non-finite rollout")
                total += rewards.mean(0)
                done_count += int(dones.sum())
                if args.video:
                    frames.append(env.render())
        result = {
            "steps": args.steps,
            "num_envs": args.num_envs,
            "reward_group_mean": (total / args.steps).tolist(),
            "terminations": done_count,
            "seed": args.seed,
        }
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "evaluation.json").write_text(json.dumps(result, indent=2))
        print(result)
        if args.video:
            import imageio.v3 as iio

            args.video.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(args.video, frames, fps=50)
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.close()
        env.close()


if __name__ == "__main__":
    main()
