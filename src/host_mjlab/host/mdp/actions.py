from dataclasses import dataclass
import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from .state import state


@dataclass(kw_only=True)
class HostActionCfg(ActionTermCfg):
    def build(self, env):
        return HostAction(self, env)


class HostAction(ActionTerm):
    @property
    def action_dim(self):
        return 12

    @property
    def raw_action(self):
        return state(self._env).actions

    def process_actions(self, actions):
        s = state(self._env)
        s.last_last_actions.copy_(s.last_actions)
        s.last_actions.copy_(s.actions)
        s.last_dof_vel.copy_(s.robot.data.joint_vel[:, s.joint_ids])
        s.actions.copy_(actions.clamp(-100, 100))
        s.actions.mul_(s.real_episode_length_buf[:, None] > 30)

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
            * (s.real_episode_length_buf[:, None] > 30)
            * (s.projected_gravity[:, 2:3] < -0.8)
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
    force = -0.01 * mass[..., None] * d.body_com_lin_vel_w
    omega = quat_apply_inverse(d.body_com_quat_w, d.body_com_ang_vel_w)
    torque = quat_apply(d.body_com_quat_w, -0.01 * inertia * omega)
    return force, torque
