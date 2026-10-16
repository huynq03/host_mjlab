import torch
from .state import state
from .curricula import update_recovery_metrics


def time_out(env):
    s = state(env)
    if s.cached_step != env.common_step_counter:
        # MJLab normally evaluates derived poses one substep behind qpos.
        env.sim.forward()
        s.refresh()
        update_recovery_metrics(env)
        s.real_episode_length_buf += 1
        s.cached_step = env.common_step_counter
    return env.episode_length_buf > env.max_episode_length


def velocity_limit(env, joint_limit=300.0, base_limit=20.0):
    s = state(env)
    # Preserve abs(max(v)), including its asymmetric treatment of negative speeds.
    return (
        (s.dof_vel.max(dim=1).values.abs() > joint_limit)
        | (torch.linalg.vector_norm(s.base_lin_vel, dim=-1) > base_limit)
    ) & (s.real_episode_length_buf > s.control.settle_steps)
