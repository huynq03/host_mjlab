from .state import state


def update_recovery_metrics(env):
    """Track physical recovery once before rewards, independent of reward enablement."""
    import torch

    s = state(env)
    head = s.rigid_body_states[:, s.head_indices, 2].clone()
    head -= s.rigid_body_states[:, s.feet_indices, 2].mean(-1, keepdim=True)
    s.max_headheight.copy_(torch.cat((head, s.old_headheight), dim=1).max(dim=1).values[:, None])
    s.old_headheight.copy_(head)


def recovery(
    env,
    env_ids,
    threshold_height=0.37,
    force_decrement=20.0,
    action_decrement=0.02,
    minimum_action_scale=0.25,
):
    # MJLab _reset_idx calls this BEFORE sim.reset, scene.reset and reset events.
    # old_headheight must still describe the terminal pose, not the reset pose.
    s = state(env)
    if env_ids is None:
        return
    if len(env_ids) and s.old_headheight[env_ids].mean() > threshold_height:
        s.force[env_ids] = (s.force[env_ids] - force_decrement).clamp(min=0)
        s.action_rescale[env_ids] = (s.action_rescale[env_ids] - action_decrement).clamp(
            min=minimum_action_scale
        )
    return {"pull_force": s.force.mean(), "action_scale": s.action_rescale.mean()}
