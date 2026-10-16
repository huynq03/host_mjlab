import math
import torch
from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel
from .state import state


def reset_robot(
    env,
    env_ids,
    joint_position_range=(-0.1, 0.1),
    kp_range=(0.85, 1.15),
    kd_range=(0.85, 1.15),
    motor_strength_range=(0.9, 1.1),
    actuation_offset_range=(-0.05, 0.05),
):
    s = state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if n == 0:
        return
    # CurriculumManager already consumed terminal old_headheight before this
    # reset event. Never sample a terminal frame after scene.reset().
    s.pending_pull[env_ids] = 0
    s.torques[env_ids] = 0
    s.joint_pos_target[env_ids] = 0
    zeros = torch.zeros(n, s.robot.num_bodies, 3, device=env.device)
    s.robot.write_external_wrench_to_sim(zeros, zeros, env_ids=env_ids)
    s.robot.set_joint_effort_target(
        torch.zeros(n, s.num_joints, device=env.device), joint_ids=s.joint_ids, env_ids=env_ids
    )
    q = (
        torch.rand(n, s.num_joints, device=env.device)
        * (joint_position_range[1] - joint_position_range[0])
        + joint_position_range[0]
    ).clamp(s.dof_pos_limits[:, 0], s.dof_pos_limits[:, 1])
    s.robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=s.joint_ids, env_ids=env_ids)
    root = s.robot.data.default_root_state[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    s.robot.write_root_state_to_sim(root, env_ids=env_ids)
    for name in [
        "last_actions",
        "last_last_actions",
        "last_dof_vel",
        "old_headheight",
        "max_headheight",
        "real_episode_length_buf",
    ]:
        getattr(s, name)[env_ids] = 0
    s.delay_buffer[:, env_ids] = 0
    s.delay_idx[env_ids] = torch.randint(s.control.delay_steps, (n,), device=env.device)
    for name, lo, hi in [
        ("Kp_factors", *kp_range),
        ("Kd_factors", *kd_range),
        ("motor_strength", *motor_strength_range),
        ("actuation_offset", *actuation_offset_range),
    ]:
        getattr(s, name)[env_ids] = torch.rand(n, s.num_joints, device=env.device) * (hi - lo) + lo
    s.actuation_offset[env_ids] *= s.torque_limits
    # History is intentionally not cleared: original HoST shifts in zero frames.
    if s.history_reset_ids is None:
        s.history_reset_ids = env_ids.clone()
    else:
        s.history_reset_ids = torch.unique(torch.cat((s.history_reset_ids, env_ids)))


@requires_model_fields(
    "body_mass",
    "body_inertia",
    "body_ipos",
    "geom_friction",
    "geom_solref",
    "pair_friction",
    "pair_solref",
    recompute=RecomputeLevel.set_const,
)
def randomize_physics(
    env,
    env_ids,
    link_mass_range=(0.8, 1.2),
    com_range=(-0.03, 0.03),
    friction_range=(0.1, 1.0),
    restitution_range=(0.0, 1.0),
    ground_friction=0.8,
    ground_restitution=0.3,
    contact_time_constant=0.02,
):
    s = state(env)
    model = env.sim.model
    ids = s.robot.indexing.body_ids.long()
    gids = s.robot.indexing.geom_ids.long()
    n = env.num_envs
    # HoST overwrites payload mass with per-link default mass * U(.8,1.2).
    scale = (
        torch.rand(n, len(ids), device=env.device) * (link_mass_range[1] - link_mass_range[0])
        + link_mass_range[0]
    )
    model.body_mass[:, ids] *= scale
    model.body_inertia[:, ids] *= scale[:, :, None]
    base = ids[s.base_indices]
    model.body_ipos[:, base] += (
        torch.rand(n, len(base), 3, device=env.device) * (com_range[1] - com_range[0])
        + com_range[0]
    )
    friction = (
        torch.rand(n, 1, device=env.device) * (friction_range[1] - friction_range[0])
        + friction_range[0]
    )
    model.geom_friction[:, gids, 0] = friction
    s.friction = friction
    # All explicit pairs are robot-ground. PhysX average material mixing.
    model.pair_friction[:, :, 0] = 0.5 * (friction + ground_friction)
    model.pair_friction[:, :, 1] = 0.5 * (friction + ground_friction)
    # MuJoCo has no restitution coefficient. Match an ideal damped impact's
    # restitution via damping ratio, retaining the sampled coefficient for audit.
    s.restitution = (
        torch.rand(n, 1, device=env.device) * (restitution_range[1] - restitution_range[0])
        + restitution_range[0]
    )
    log_e = s.restitution.clamp(1e-6, 1 - 1e-6).log()
    damping = -log_e / torch.sqrt(math.pi**2 + log_e**2)
    model.geom_solref[:, gids, 0] = contact_time_constant
    model.geom_solref[:, gids, 1] = damping
    ground_log = (0.5 * (s.restitution + ground_restitution)).log()
    model.pair_solref[:, :, 0] = contact_time_constant
    model.pair_solref[:, :, 1] = -ground_log / torch.sqrt(math.pi**2 + ground_log**2)


def ground_contact_pairs(spec):
    """Explicit ground pairs allow average material mixing instead of MuJoCo max."""
    for geom in list(spec.geoms):
        if geom.name.startswith("robot/") and "_collision_" in geom.name:
            spec.add_pair(
                name="host_ground_" + geom.name.replace("/", "_"),
                geomname1=geom.name,
                geomname2="terrain",
                condim=3,
                friction=[0.8, 0.8, 0, 0, 0],
                margin=0.01,
                gap=0.01,
                solref=[0.02, 1],
            )
