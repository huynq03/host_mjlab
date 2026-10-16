"""Tensor state shared by HoST manager terms; no simulation loop lives here."""

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from host_mjlab.asset_zoo.robots.minipi.minipi_constants import JOINTS, JOINT_NAMES


class HostState:
    def __init__(self, env):
        self.env, self.robot = env, env.scene["robot"]
        self.device, self.num_envs = env.device, env.num_envs
        self.control = env.cfg.actions["host"]
        self.observation_cfg = env.cfg.host_observations
        self.dt = env.step_dt
        if abs(self.dt - 0.02) > 1e-12:
            raise ValueError("Original HoST requires 0.005 s physics dt and decimation 4")
        self.joint_ids, names = self.robot.find_joints(JOINT_NAMES, preserve_order=True)
        if tuple(names) != JOINT_NAMES:
            raise ValueError("Mini-Pi joint mapping differs from the original HoST order")
        # MJLab entity target writers require tensor indices for Cartesian
        # indexing together with a tensor of reset environment IDs.
        self.joint_ids = torch.tensor(self.joint_ids, device=env.device, dtype=torch.long)

        def z(*shape):
            return torch.zeros(env.num_envs, *shape, device=env.device)

        self.num_joints = len(JOINT_NAMES)
        self.frame_dim = 7 + 3 * self.num_joints
        self.history_dim = self.frame_dim * self.observation_cfg.history_length
        self.actions = z(self.num_joints)
        self.last_actions = z(self.num_joints)
        self.last_last_actions = z(self.num_joints)
        self.last_dof_vel = z(self.num_joints)
        self.torques = z(self.num_joints)
        self.joint_pos_target = z(self.num_joints)
        self.action_rescale = z(1) + self.control.action_scale
        self.force = z(1) + self.control.initial_pull_force
        self.pending_pull = z(1)
        self.old_headheight = z(1)
        self.max_headheight = z(1)
        self.real_episode_length_buf = z().long()
        self.history = z(self.history_dim)
        self.delay_buffer = torch.zeros(
            self.control.delay_steps, env.num_envs, self.num_joints, device=env.device
        )
        self.delay_idx = torch.randint(self.control.delay_steps, (env.num_envs,), device=env.device)
        self.Kp_factors = z(self.num_joints) + 1
        self.Kd_factors = z(self.num_joints) + 1
        self.motor_strength = z(self.num_joints) + 1
        self.actuation_offset = z(self.num_joints)
        self.p_gains = torch.tensor(
            [next(v for k, v in self.control.stiffness.items() if k in n) for n in JOINT_NAMES],
            device=env.device,
        )
        self.d_gains = torch.full((self.num_joints,), self.control.damping, device=env.device)
        self.torque_limits = torch.tensor([j["effort"] for j in JOINTS], device=env.device)
        self.dof_vel_limits = torch.tensor([j["velocity"] for j in JOINTS], device=env.device)
        self.dof_pos_limits = torch.tensor(
            [
                [
                    j["lower"] * self.robot.cfg.articulation.soft_joint_pos_limit_factor,
                    j["upper"] * self.robot.cfg.articulation.soft_joint_pos_limit_factor,
                ]
                for j in JOINTS
            ],
            device=env.device,
        )
        self.target_dof_pos = z(self.num_joints)
        obs = self.observation_cfg
        self.noise_scale_vec = torch.tensor(
            [obs.angular_velocity_noise] * 3
            + [obs.gravity_noise] * 3
            + [obs.joint_position_noise] * self.num_joints
            + [obs.joint_velocity_noise] * self.num_joints
            + [0.0] * (self.num_joints + 1),
            device=env.device,
        )
        self.terminal_observations = z(self.history_dim)
        self.cached_step = -1
        self.observation_step = -1
        self.history_reset_ids = None
        bodies = self.robot.body_names

        def body_ids(match, exclude=()):
            return torch.tensor(
                [
                    i
                    for i, n in enumerate(bodies)
                    if match in n and not any(x in n for x in exclude)
                ],
                device=env.device,
                dtype=torch.long,
            )

        self.base_indices = body_ids("base_link")
        self.head_indices = body_ids("keyframe_head")
        self.feet_indices = body_ids("ankle_roll", ("auxiliary",))
        for side in ("left", "right"):
            prefix = side[0]
            setattr(
                self,
                side + "_foot_indices",
                body_ids(prefix + "_ankle_pitch", ("auxiliary", "keyframe")),
            )
            setattr(self, side + "_knee_indices", body_ids(prefix + "_calf", ("keyframe",)))
            setattr(self, side + "_ankle_indices", body_ids(prefix + "_ankle_roll", ("keyframe",)))
            setattr(
                self,
                side + "_leg_joints_indices",
                [
                    JOINT_NAMES.index(n)
                    for n in [n for n in JOINT_NAMES if n.startswith(prefix + "_")]
                ],
            )
        self.hip_joint_indices = [JOINT_NAMES.index(n) for n in ["l_thigh_joint", "r_thigh_joint"]]
        self.hip_roll_joint_indices = [
            JOINT_NAMES.index(n) for n in ["l_hip_roll_joint", "r_hip_roll_joint"]
        ]
        self.knee_joint_indices = [JOINT_NAMES.index(n) for n in ["l_calf_joint", "r_calf_joint"]]
        self.refresh()

    def refresh(self):
        d = self.robot.data
        self.dof_pos = d.joint_pos[:, self.joint_ids]
        self.dof_vel = d.joint_vel[:, self.joint_ids]
        self.root_states = torch.cat((d.root_link_pose_w, d.root_link_vel_w), dim=-1)
        self.rigid_body_states = torch.cat((d.body_link_pose_w, d.body_link_vel_w), dim=-1)
        self.base_ang_vel = quat_apply_inverse(d.root_link_quat_w, d.root_link_ang_vel_w)
        self.base_lin_vel = quat_apply_inverse(d.root_link_quat_w, d.root_link_lin_vel_w)
        gravity = torch.zeros_like(self.base_ang_vel)
        gravity[:, 2] = -1
        self.projected_gravity = quat_apply_inverse(d.root_link_quat_w, gravity)

    def capture_terminal_observations(self, env_ids):
        """Capture before MJLab resets simulation/entity buffers; no RNG side effects."""
        if not len(env_ids):
            return
        device = torch.device(self.device)
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            frame = self.frame(env_ids)
            self.terminal_observations[env_ids] = torch.cat(
                (self.history[env_ids, self.frame_dim :], frame), dim=1
            ).clamp(-self.observation_cfg.clip, self.observation_cfg.clip)

    def frame(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        obs = torch.cat(
            (
                self.base_ang_vel[ids] * self.observation_cfg.angular_velocity_scale,
                self.projected_gravity[ids],
                self.dof_pos[ids] * self.observation_cfg.joint_position_scale,
                self.dof_vel[ids] * self.observation_cfg.joint_velocity_scale,
                self.actions[ids],
                self.action_rescale[ids]
                + (torch.rand_like(self.action_rescale[ids]) - 0.5)
                * self.observation_cfg.action_scale_noise,
            ),
            dim=-1,
        )
        if self.observation_cfg.enable_noise:
            obs += (2 * torch.rand_like(obs) - 1) * self.noise_scale_vec
        return obs * (self.real_episode_length_buf[ids, None] > self.control.settle_steps)


def state(env):
    if not hasattr(env, "host_state"):
        env.host_state = HostState(env)
    return env.host_state
