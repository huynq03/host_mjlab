"""Six-frame HoST history, preserving its original carry-over across resets."""

from .state import state


def history(env):
    s = state(env)
    key = env.common_step_counter
    if s.observation_step != key:
        # A physics step advances every env exactly once, including auto-resets.
        s.refresh()
        s.history.copy_(s.history.roll(-43, dims=1))
        s.history[:, -43:] = s.frame()
        s.history.clamp_(-100, 100)
        s.observation_step = key
        s.history_reset_ids = None
    elif s.history_reset_ids is not None:
        # Explicit partial reset without stepping: only selected rows advance.
        # MJLab passes env_ids to its buffers, not to observation functions.
        ids = s.history_reset_ids
        s.refresh()
        s.history[ids] = s.history[ids].roll(-43, dims=1)
        s.history[ids, -43:] = s.frame(ids)
        s.history.clamp_(-100, 100)
        s.history_reset_ids = None
    # An independent tensor prevents subsequent history writes from mutating
    # observations already held by PPO or its rollout storage.
    return s.history.clone()
