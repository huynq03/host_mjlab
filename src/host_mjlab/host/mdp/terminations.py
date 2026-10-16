import torch
from .state import state


def time_out(env):
    s = state(env)
    if s.cached_step != env.common_step_counter:
        # MJLab normally evaluates derived poses one substep behind qpos.
        env.sim.forward()
        s.refresh()
        s.real_episode_length_buf += 1
        s.cached_step = env.common_step_counter
    return env.episode_length_buf > 500


def velocity_limit(env):
    s = state(env)
    # Preserve abs(max(v)), including its asymmetric treatment of negative speeds.
    return (
        (s.dof_vel.max(dim=1).values.abs() > 300)
        | (torch.linalg.vector_norm(s.base_lin_vel, dim=-1) > 20)
    ) & (s.real_episode_length_buf > 30)
