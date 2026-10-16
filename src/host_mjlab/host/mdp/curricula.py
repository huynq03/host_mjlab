from .state import state


def recovery(env, env_ids):
    s = state(env)
    if env_ids is None:
        return
    if len(env_ids) and s.old_headheight[env_ids].mean() > 0.37:
        s.force[env_ids] = (s.force[env_ids] - 20).clamp(min=0)
        s.action_rescale[env_ids] = (s.action_rescale[env_ids] - 0.02).clamp(min=0.25)
    return {"pull_force": s.force.mean(), "action_scale": s.action_rescale.mean()}
