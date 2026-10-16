from mjlab.tasks.registry import register_mjlab_task
from .rl_cfg import runner_cfg
from host_mjlab.host.host_env_cfg import make_env_cfg
from host_mjlab.algorithms.host_ppo.runner import HostRunner

register_mjlab_task(
    task_id="Mjlab-HoST-Ground-MiniPi",
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(1),
    rl_cfg=runner_cfg(),
    runner_cls=HostRunner,
)
