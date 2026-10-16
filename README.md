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
- `src/host_mjlab/asset_zoo/robots/minipi/`: robot configuration, joint limits and `xmls/minipi.xml`; its 13 active meshes are in `xmls/assets/`, copied from the reference asset zoo.
- `src/host_mjlab/host/config/minipi/`: MJLab task registration.

Tests, audit JSON reports, conversion/comparison scripts and previous generated run artifacts have been removed. Runtime configuration is Python; training still writes its resolved configuration and evaluation outputs to the run directory.

The original reward equations and PPO are retained. PhysX and MuJoCo are not physically identical; see [the implementation audit](docs/AUDIT.md) for explicit differences and preserved source quirks. No long training or learned recovery-success claim is made.

HoST code/assets retain the OpenRobotLab MIT license. Original RSL-RL algorithm files retain their NVIDIA/ETH BSD notices.

## Configure individual rewards

`make_env_cfg()` creates one native MJLab reward entry for each of the 24 active
original terms. Edit coefficients before creating the environment, for example:

```python
from host_mjlab.host.host_env_cfg import make_env_cfg

cfg = make_env_cfg()
cfg.rewards["regu_action_rate"].weight = -0.01  # Original coefficient.
# A weight of 0.0 disables a term; removing an entry also disables it.
```

Task coefficients are dimensionless factors in the original product. Other
coefficients are multiplied by the unchanged 0.02 s control timestep. Their
per-term values are logged by MJLab; `aggregate_rewards` reduces the original
task product and three sums before resets and sends those four rewards to PPO.
The scalar sum returned by native MJLab is diagnostic. Training must use the
registered `HostRunner`/`HostAdapter` to preserve Multi-Critic PPO.

The latest changes were reviewed using **static analysis only**. See
[the static review](docs/AUDIT.md#static-review-of-host-mjlab-port) for API coverage and remaining risks.

Task discovery uses the `host_mjlab` package entry point, which imports
`host/config/minipi/` directly, following mjlab_playground. Unused source URDF
and `base_link.STL` are omitted; the runtime robot is defined by its MJCF.

The `algorithms/host_ppo/` folder has four implementation modules:
`actor_critic.py` (networks), `ppo.py` (optimization), `rollout_storage.py`
(storage and GAE), and `runner.py` (MJLab adapter, collection and checkpoints).
The adapter is imported as `from host_mjlab.algorithms.host_ppo.runner import HostAdapter`.
Unused normalizer/AMP utilities are omitted; the original trajectory-splitting
helper lives beside the storage that uses it.
