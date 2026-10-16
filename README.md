# HoST Mini-Pi training on MJLab

Standalone original Mini-Pi ground recovery with native MJLab managers and HoST four-critic PPO (`task`, `regu`, `style`, `target`). No runtime dependency on either source repository.

## Install and train

Python 3.13 and `uv` are required. GPU training requires a driver compatible with the locked CUDA 13 stack.

```bash
uv sync --locked
uv run host-mjlab train --device cuda:0 --num-envs 4096 --iterations 12000
```

Resume actor/critics and optimizer from a checkpoint:

```bash
uv run host-mjlab train --device cuda:0 --num-envs 4096 --iterations 1000 --checkpoint logs/host/RUN/model_100.pt
```

Checkpoints and TensorBoard logs are written under `logs/host/<timestamp>/`. Simulation state resets on resume.

```bash
uv run tensorboard --logdir logs/host
uv run host-mjlab play --checkpoint logs/host/RUN/model_100.pt --num-envs 1 --steps 500 --video logs/play.mp4
uv run host-mjlab evaluate --checkpoint logs/host/RUN/model_100.pt --num-envs 16 --steps 1000
```

## Code

- `src/host_mjlab/host/mdp/`: task observations, control, rewards, resets, randomization and curriculum.
- `src/host_mjlab/host/original_config.py`: original HoST configuration.
- `src/host_mjlab/algorithms/host_ppo/`: four-critic PPO, storage, adapter and runner.
- `src/host_mjlab/assets/minipi/`: HoST-derived robot MJCF, URDF and joint limits; active meshes copied from `mjlab_playground`’s `hightorque_minipi` asset zoo.
- `src/host_mjlab/host/config/minipi/`: MJLab task registration.

Tests, audit JSON reports, conversion/comparison scripts and previous generated run artifacts have been removed. Runtime configuration is Python; training still writes its resolved configuration and evaluation outputs to the run directory.

The original reward equations and PPO are retained. PhysX and MuJoCo are not physically identical; see [the implementation audit](docs/AUDIT.md) for explicit differences and preserved source quirks. No long training or learned recovery-success claim is made.

HoST code/assets retain the OpenRobotLab MIT license. Original RSL-RL algorithm files retain their NVIDIA/ETH BSD notices.
