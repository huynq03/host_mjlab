"""Six-frame HoST history, preserving its original carry-over across resets."""

from .state import state


def history(env):
    s = state(env)
    key = env.common_step_counter
    if s.observation_step != key:
        # A physics step advances every env exactly once, including auto-resets.
        s.refresh()
        s.history.copy_(s.history.roll(-s.frame_dim, dims=1))
        s.history[:, -s.frame_dim :] = s.frame()
        s.history.clamp_(-s.observation_cfg.clip, s.observation_cfg.clip)
        s.observation_step = key
        s.history_reset_ids = None
    elif s.history_reset_ids is not None:
        # Explicit partial reset without stepping: only selected rows advance.
        # MJLab passes env_ids to its buffers, not to observation functions.
        ids = s.history_reset_ids
        s.refresh()
        s.history[ids] = s.history[ids].roll(-s.frame_dim, dims=1)
        s.history[ids, -s.frame_dim :] = s.frame(ids)
        s.history.clamp_(-s.observation_cfg.clip, s.observation_cfg.clip)
        s.history_reset_ids = None
    # An independent tensor prevents subsequent history writes from mutating
    # observations already held by PPO or its rollout storage.
    return s.history.clone()
