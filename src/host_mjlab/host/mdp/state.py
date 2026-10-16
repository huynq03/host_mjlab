"""Tensor state shared by HoST manager terms; no simulation loop lives here."""

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from host_mjlab.asset_zoo.robots.minipi.minipi_constants import JOINTS, JOINT_NAMES
from host_mjlab.host.settings import CFG
from .rewards import OriginalRewards


class HostState(OriginalRewards):
    def __init__(self, env):
        self.env, self.robot = env, env.scene["robot"]
        self.cfg, self.device, self.num_envs = CFG, env.device, env.num_envs
        self.dt, self.is_gaussian = env.step_dt, True
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

        self.actions = z(12)
        self.last_actions = z(12)
        self.last_last_actions = z(12)
        self.last_dof_vel = z(12)
        self.torques = z(12)
        self.joint_pos_target = z(12)
        self.action_rescale = z(1) + 1
        self.force = z(1) + 15
        self.pending_pull = z(1)
        self.old_headheight = z(1)
        self.max_headheight = z(1)
        self.real_episode_length_buf = z().long()
        self.history = z(258)
        self.delay_buffer = torch.zeros(5, env.num_envs, 12, device=env.device)
        self.delay_idx = torch.randint(5, (env.num_envs,), device=env.device)
        self.Kp_factors = z(12) + 1
        self.Kd_factors = z(12) + 1
        self.motor_strength = z(12) + 1
        self.actuation_offset = z(12)
        self.p_gains = torch.tensor(
            [
                next(v for k, v in CFG.control.stiffness.__dict__.items() if k in n)
                for n in JOINT_NAMES
            ],
            device=env.device,
        )
        self.d_gains = torch.full((12,), 0.2, device=env.device)
        self.torque_limits = torch.tensor([j["effort"] for j in JOINTS], device=env.device)
        self.dof_vel_limits = torch.tensor([j["velocity"] for j in JOINTS], device=env.device)
        self.dof_pos_limits = torch.tensor(
            [[j["lower"] * 0.9, j["upper"] * 0.9] for j in JOINTS], device=env.device
        )
        self.target_dof_pos = z(12)
        self.noise_scale_vec = torch.tensor(
            [0.05] * 3 + [0.05] * 3 + [0.01] * 12 + [0.075] * 12 + [0.0] * 13, device=env.device
        )
        self.group_rewards = z(4)
        self.terminal_observations = z(258)
        self.cached_step = -1
        self.observation_step = -1
        self.history_reset_ids = None
        self.reward_values = {}
        self.reward_step = -1
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
                [JOINT_NAMES.index(n) for n in getattr(CFG.asset, side + "_leg_joints")],
            )
        self.hip_joint_indices = [
            JOINT_NAMES.index(n) for n in CFG.asset.left_hip_joints + CFG.asset.right_hip_joints
        ]
        self.hip_roll_joint_indices = [
            JOINT_NAMES.index(n)
            for n in CFG.asset.left_hip_roll_joints + CFG.asset.right_hip_roll_joints
        ]
        self.knee_joint_indices = [JOINT_NAMES.index(n) for n in CFG.asset.knee_joints]
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
                (self.history[env_ids, 43:], frame), dim=1
            ).clamp(-100, 100)

    def frame(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        obs = torch.cat(
            (
                self.base_ang_vel[ids] * 0.25,
                self.projected_gravity[ids],
                self.dof_pos[ids],
                self.dof_vel[ids] * 0.05,
                self.actions[ids],
                self.action_rescale[ids] + (torch.rand_like(self.action_rescale[ids]) - 0.5) * 0.05,
            ),
            dim=-1,
        )
        if CFG.noise.add_noise:
            obs += (2 * torch.rand_like(obs) - 1) * self.noise_scale_vec
        return obs * (self.real_episode_length_buf[ids, None] > 30)


def state(env):
    if not hasattr(env, "host_state"):
        env.host_state = HostState(env)
    return env.host_state
