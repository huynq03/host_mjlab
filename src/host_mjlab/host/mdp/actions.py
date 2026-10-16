from dataclasses import dataclass, field
import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from .state import state


@dataclass(kw_only=True)
class HostActionCfg(ActionTermCfg):
    settle_steps: int = 30
    action_scale: float = 1.0
    action_clip: float = 100.0
    initial_pull_force: float = 15.0
    pull_orientation_threshold: float = -0.8
    delay_steps: int = 5
    stiffness: dict[str, float] = field(
        default_factory=lambda: {
            "hip_pitch": 30.0,
            "hip_roll": 15.0,
            "thigh": 15.0,
            "calf": 30.0,
            "ankle_pitch": 12.0,
            "ankle_roll": 5.0,
        }
    )
    damping: float = 0.2
    body_linear_damping: float = 0.01
    body_angular_damping: float = 0.01

    def build(self, env):
        return HostAction(self, env)


class HostAction(ActionTerm):
    @property
    def action_dim(self):
        return self._entity.num_joints

    @property
    def raw_action(self):
        return state(self._env).actions

    def process_actions(self, actions):
        s = state(self._env)
        s.last_last_actions.copy_(s.last_actions)
        s.last_actions.copy_(s.actions)
        s.last_dof_vel.copy_(s.robot.data.joint_vel[:, s.joint_ids])
        s.actions.copy_(actions.clamp(-self.cfg.action_clip, self.cfg.action_clip))
        s.actions.mul_(s.real_episode_length_buf[:, None] > self.cfg.settle_steps)

    def apply_actions(self):
        s = state(self._env)
        d = s.robot.data
        q = d.joint_pos[:, s.joint_ids]
        v = d.joint_vel[:, s.joint_ids]
        scaled = s.actions * s.action_rescale
        s.delay_buffer.copy_(torch.cat((s.delay_buffer[1:], scaled[None]), dim=0))
        delayed = s.delay_buffer[s.delay_idx, torch.arange(s.num_envs, device=s.device)]
        s.joint_pos_target.copy_(q + delayed)
        torque = s.p_gains * s.Kp_factors * (s.joint_pos_target - q) - s.d_gains * s.Kd_factors * v
        s.torques.copy_(
            torch.clamp(
                s.motor_strength * torque + s.actuation_offset, -s.torque_limits, s.torque_limits
            )
        )
        s.robot.set_joint_effort_target(s.torques, joint_ids=s.joint_ids)
        # PhysX damping is a body velocity decay, not joint damping.
        force, torque = body_damping(self._env, s)
        # Isaac defaults to ENV_SPACE: upward is world +Z. HoST submits the
        # pull after simulate(), so retain the one-substep submission delay.
        force[:, s.base_indices, 2] += s.pending_pull
        s.pending_pull.copy_(
            s.force
            * (s.real_episode_length_buf[:, None] > self.cfg.settle_steps)
            * (s.projected_gravity[:, 2:3] < self.cfg.pull_orientation_threshold)
        )
        s.robot.write_external_wrench_to_sim(force, torque)

    def reset(self, env_ids=None):
        # Original reset clears past actions, but preserves the current action.
        pass


def body_damping(env, s):
    """Map PhysX 0.01/s linear/angular velocity damping to body wrenches."""
    from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

    d = s.robot.data
    ids = s.robot.indexing.body_ids
    mass = env.sim.model.body_mass[:, ids]
    inertia = env.sim.model.body_inertia[:, ids]
    force = -s.control.body_linear_damping * mass[..., None] * d.body_com_lin_vel_w
    omega = quat_apply_inverse(d.body_com_quat_w, d.body_com_ang_vel_w)
    torque = quat_apply(d.body_com_quat_w, -s.control.body_angular_damping * inertia * omega)
    return force, torque
