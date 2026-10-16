from .state import state


def history(env):
    s = state(env)
    # The manager calls terms once to infer shapes before training starts.
    key = env.common_step_counter  # Reset events explicitly invalidate this cache.
    if s.observation_step != key:
        s.refresh()
        frame = s.frame()
        s.history = s.history.roll(-43, dims=1)
        s.history[:, -43:] = frame
        # HoST clips the stored obs_buf itself, including history.
        s.history.clamp_(-100, 100)
        s.observation_step = key
    return s.history.clamp(-100, 100)
