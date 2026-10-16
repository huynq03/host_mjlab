"""Original HoST Mini-Pi reward equations (MIT, OpenRobotLab 2025)."""

import torch
import numpy as np
from .math import tolerance


class OriginalRewards:
    def _reward_orientation(self):
        if not self.is_gaussian:
            mse_error = torch.sum(
                torch.square(self.projected_gravity - torch.tensor([0, 0, 1], device=self.device)),
                dim=-1,
            )
            reward = torch.exp(mse_error / self.cfg.rewards.orientation_sigma) * (
                self.root_states[:, 2] > 0.4
            )
        else:
            base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase1
            reward = tolerance(
                -self.projected_gravity[:, 2],
                [self.cfg.rewards.orientation_threshold, np.inf],
                1.0,
                0.05,
            )  # -1
        return reward

    def _reward_head_height(self):
        if not self.is_gaussian:
            head_height = self.rigid_body_states[:, self.head_indices, 2].clone()
            return head_height.squeeze(1).clamp(0, 1)
        else:
            head_height = self.rigid_body_states[:, self.head_indices, 2].clone()
            feet_height = (
                self.rigid_body_states[:, self.feet_indices, 2].clone().mean(-1).unsqueeze(-1)
            )
            head_height -= feet_height
            reward = tolerance(
                head_height,
                (self.cfg.rewards.target_head_height, np.inf),
                self.cfg.rewards.target_head_margin,
                0.1,
            )
            delta_max_headheight = head_height - self.max_headheight
            delta_headheight = head_height - self.old_headheight
            self.max_headheight.copy_(
                torch.max(torch.cat((head_height, self.old_headheight), dim=1), dim=1)[0].unsqueeze(
                    -1
                )
            )
            self.old_headheight.copy_(head_height)
            return reward

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(
            torch.square(
                self.actions - self.last_actions - self.last_actions + self.last_last_actions
            ),
            dim=1,
        )

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_joint_power(self):
        # Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_joint_tracking_error(self):
        return torch.sum(torch.square(self.joint_pos_target - self.dof_pos), dim=-1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        return torch.sum(
            (
                torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit
            ).clip(min=0.0, max=1.0),
            dim=1,
        )

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (
                torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_waist_deviation(self):
        wrist_dof = self.dof_pos[:, self.waist_joint_indices]
        reward = (torch.abs(wrist_dof) > 1.4).float()
        return reward.squeeze(1)

    def _reward_hip_yaw_deviation(self):
        hip_yaw_dof = self.dof_pos[:, self.hip_joint_indices]
        reward = (
            torch.max(torch.abs(self.dof_pos[:, self.hip_joint_indices]), dim=-1)[0] > 1.4
        ) | (torch.min(torch.abs(self.dof_pos[:, self.hip_joint_indices]), dim=-1)[0] > 0.9)
        return reward

    def _reward_hip_roll_deviation(self):
        hip_roll_dof = self.dof_pos[:, self.hip_roll_joint_indices]
        reward = (
            torch.max(torch.abs(self.dof_pos[:, self.hip_roll_joint_indices]), dim=-1)[0] > 1.4
        ) | (torch.min(torch.abs(self.dof_pos[:, self.hip_roll_joint_indices]), dim=-1)[0] > 0.9)
        return reward

    def _reward_left_foot_displacement(self):
        base_xy = self.root_states[:, :2].clone()
        left_foot_xy = self.rigid_body_states[:, self.left_foot_indices, :2].squeeze(1)
        mse = torch.sum(torch.square(base_xy - left_foot_xy), dim=-1)
        mse_error = mse.clamp(0.3, np.inf)
        # mse_error = mse.clamp(0.025, np.inf) #better standing style
        # mse_error -=0.02 #better standing style
        reward = torch.exp(mse_error * self.cfg.rewards.left_foot_displacement_sigma) * (
            self.rigid_body_states[:, self.left_foot_indices, 2] < 0.15
        ).squeeze(1)

        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        # standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase2 #better standing style
        return reward * standup

    def _reward_right_foot_displacement(self):
        base_xy = self.root_states[:, :2].clone()
        right_foot_xy = self.rigid_body_states[:, self.right_foot_indices, :2].squeeze(1)
        mse = torch.sum(torch.square(base_xy - right_foot_xy), dim=-1)
        mse_error = mse.clamp(0.3, np.inf)
        # mse_error = mse.clamp(0.025, np.inf) #better standing style
        # mse_error -=0.02 #better standing style
        reward = torch.exp(mse_error * self.cfg.rewards.right_foot_displacement_sigma) * (
            self.rigid_body_states[:, self.right_foot_indices, 2] < 0.15
        ).squeeze(1)

        # standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase2 #better standing style
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return reward * standup

    def _reward_knee_deviation(self):
        hip_roll_dof = self.dof_pos[:, self.knee_joint_indices]
        reward = (
            torch.max(torch.abs(self.dof_pos[:, self.knee_joint_indices]), dim=-1)[0] > 1.65
        ) | (torch.min(self.dof_pos[:, self.knee_joint_indices], dim=-1)[0] < -0.6)
        return reward

    def _reward_shank_orientation(self):
        left_knee_pos = self.rigid_body_states[:, self.left_knee_indices, :3].clone()
        right_knee_pos = self.rigid_body_states[:, self.right_knee_indices, :3].clone()
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :3].clone()

        left_feet_orientation = (left_knee_pos - left_foot_pos)[:, :, 2] / torch.norm(
            left_knee_pos - left_foot_pos, dim=-1
        )
        right_feet_orientation = (right_knee_pos - right_foot_pos)[:, :, 2] / torch.norm(
            right_knee_pos - right_foot_pos, dim=-1
        )

        feet_orientation = torch.mean(
            torch.concat([left_feet_orientation, right_feet_orientation], dim=-1), dim=-1
        )

        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase1
        reward = tolerance(feet_orientation, [0.8, np.inf], 1, 0.1) * base_height  # .unsqueeze(1)

        if self.cfg.constraints.post_task:
            standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
            reward = reward * ~standup + torch.ones_like(reward) * standup
        return reward

    def _reward_ground_parallel(self):
        left_ankle_pos = self.rigid_body_states[:, self.left_ankle_indices, 2].clone() * 10
        right_ankle_pos = self.rigid_body_states[:, self.right_ankle_indices, 2].clone() * 10
        var = left_ankle_pos.var(1) + right_ankle_pos.var(1)
        var = torch.mean(
            torch.concat(
                [left_ankle_pos.var(1).view(-1, 1), right_ankle_pos.var(1).view(-1, 1)], dim=-1
            ),
            dim=-1,
        )
        reward = var < 0.05

        if self.cfg.constraints.post_task:
            standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
            reward = reward * ~standup + torch.ones_like(reward) * standup
        return reward

    def _reward_feet_distance(self):
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :3].clone()
        feet_distances = torch.norm(left_foot_pos - right_foot_pos, dim=-1)
        reward = tolerance(feet_distances, [0, 0.4], 0.3, 0.05)
        # return (feet_distances > 0.33).squeeze(1) #better standing style
        return (feet_distances > 0.45).squeeze(1)

    def _reward_style_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase1
        return (
            torch.exp(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -2) * base_height
        )

    def _reward_soft_symmetry_action(self):
        left_body_action = self.actions[:, self.left_leg_joints_indices]  # [num_envs, 6]
        right_body_action = self.actions[:, self.right_leg_joints_indices]  # [num_envs, 6]
        # negative_indices = torch.tensor([1, 2, 5, 16, 17], device=self.device, dtype=torch.int64)
        negative_indices = torch.tensor([1, 2, 5], device=self.device, dtype=torch.int64)
        left_body_action[:, negative_indices] *= -1
        body_symmetry = torch.norm(left_body_action - right_body_action, dim=-1)
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        body_symmetry[~standup] *= 0
        body_symmetry = body_symmetry * torch.clamp(-self.projected_gravity[:, 2], 0, 0.9) / 0.9

        return body_symmetry

    def _reward_soft_symmetry_body(self):
        left_body = self.dof_pos[:, self.left_leg_joints_indices]  # [num_envs, 6]
        right_body = self.dof_pos[:, self.right_leg_joints_indices]  # [num_envs, 6]
        # negative_indices = torch.tensor([1, 2, 5, 16, 17], device=self.device, dtype=torch.int64)
        negative_indices = torch.tensor([1, 2, 5], device=self.device, dtype=torch.int64)
        left_body[:, negative_indices] *= -1
        error = (torch.abs(left_body - right_body) - 0.08).clip(min=0)
        body_symmetry = torch.norm(error, dim=-1)
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        body_symmetry[~standup] *= 0
        reward = torch.exp(-body_symmetry / 0.025)

        reward[~standup] *= 0
        reward = reward * torch.clamp(-self.projected_gravity[:, 2], 0, 0.9) / 0.9

        return reward

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        reward = (
            torch.exp(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -2) * base_height
        )

        return reward

    def _reward_lin_vel_xy(self):
        # Penalize z axis base linear velocity
        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        reward = (
            torch.exp(torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1) * -5) * base_height
        )

        return reward

    def _reward_feet_height_var(self):
        left_foot_height = self.rigid_body_states[:, self.left_foot_indices, 2].clone() * 10
        right_foot_height = self.rigid_body_states[:, self.right_foot_indices, 2].clone() * 10
        feet_distance = (
            torch.abs(left_foot_height - right_foot_height).squeeze(1).clamp(0.2, np.inf)
        )
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        reward = torch.exp(feet_distance * -2) * standup
        return reward

    def _reward_target_orientation(self):
        # Penalize non flat base orientation
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return (
            torch.exp(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * -5) * standup
        )

    def _reward_target_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return (
            torch.exp(torch.abs(base_height - self.cfg.rewards.base_height_target) * -20) * standup
        )

    def _reward_target_lower_dof_pos(self):
        mse = torch.sum(torch.square(self.dof_pos[:, :] - self.target_dof_pos[:, :]), dim=-1)
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        reward = torch.exp(mse * self.cfg.rewards.target_dof_pos_sigma)
        reward = reward * standup
        return reward

    def _reward_target_upper_dof_pos(self):
        # if self.is_gaussian:
        mse = torch.sum(
            torch.square(
                self.dof_pos[:, self.upper_body_joint_indices]
                - self.target_dof_pos[:, self.upper_body_joint_indices]
            ),
            dim=-1,
        )
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        reward = torch.exp(mse * self.cfg.rewards.target_dof_pos_sigma)
        reward = reward * standup
        return reward

    def _reward_target_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        reward = torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 3 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )
        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase1

        reward = reward * base_height
        return reward

    def _reward_target_knee_angle(self):
        knee_angles = self.dof_pos[:, self.knee_joint_indices]
        reward = torch.all(knee_angles > 0, dim=1).float()
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return reward * standup


def _evaluate(env, name):
    """Expose one original equation as a native MJLab reward term.

    Cache the *unweighted* value for the HoST product/sum reduction. MJLab
    applies the independently configurable weight for its own term logs.
    """
    from .state import state

    s = state(env)
    value = getattr(s, "_reward_" + name.split("_", 1)[1])().reshape(s.num_envs)
    s.reward_values[name] = (env.common_step_counter, value)
    return value if name.startswith("task_") else value * env.step_dt


def task_head_height(env):
    return _evaluate(env, "task_head_height")


def task_orientation(env):
    return _evaluate(env, "task_orientation")


def regu_action_rate(env):
    return _evaluate(env, "regu_action_rate")


def regu_dof_acc(env):
    return _evaluate(env, "regu_dof_acc")


def regu_dof_pos_limits(env):
    return _evaluate(env, "regu_dof_pos_limits")


def regu_dof_vel(env):
    return _evaluate(env, "regu_dof_vel")


def regu_dof_vel_limits(env):
    return _evaluate(env, "regu_dof_vel_limits")


def regu_joint_power(env):
    return _evaluate(env, "regu_joint_power")


def regu_joint_tracking_error(env):
    return _evaluate(env, "regu_joint_tracking_error")


def regu_smoothness(env):
    return _evaluate(env, "regu_smoothness")


def regu_torques(env):
    return _evaluate(env, "regu_torques")


def style_feet_distance(env):
    return _evaluate(env, "style_feet_distance")


def style_ground_parallel(env):
    return _evaluate(env, "style_ground_parallel")


def style_hip_roll_deviation(env):
    return _evaluate(env, "style_hip_roll_deviation")


def style_hip_yaw_deviation(env):
    return _evaluate(env, "style_hip_yaw_deviation")


def style_knee_deviation(env):
    return _evaluate(env, "style_knee_deviation")


def style_left_foot_displacement(env):
    return _evaluate(env, "style_left_foot_displacement")


def style_right_foot_displacement(env):
    return _evaluate(env, "style_right_foot_displacement")


def style_style_ang_vel_xy(env):
    return _evaluate(env, "style_style_ang_vel_xy")


def target_ang_vel_xy(env):
    return _evaluate(env, "target_ang_vel_xy")


def target_feet_height_var(env):
    return _evaluate(env, "target_feet_height_var")


def target_lin_vel_xy(env):
    return _evaluate(env, "target_lin_vel_xy")


def target_target_base_height(env):
    return _evaluate(env, "target_target_base_height")


def target_target_orientation(env):
    return _evaluate(env, "target_target_orientation")


def make_reward_cfg():
    """Each original coefficient is a manager weight, independently editable."""
    from mjlab.managers.reward_manager import RewardTermCfg
    from host_mjlab.host.settings import DATA

    terms = {}
    for section in ("rewards", "constraints"):
        for name, coefficient in sorted(DATA["env"][section]["scales"].items()):
            if coefficient:
                terms[name] = RewardTermCfg(func=globals()[name], weight=coefficient)
    return terms


def aggregate_rewards(env, env_ids=None):
    """Step event: reduce terms after RewardManager.compute, before any reset.

    Use public manager configuration so edits to term weights/enablement affect
    both MJLab logging and the four values delivered to original HoST PPO.
    MJLab's scalar reward remains a diagnostic sum, not the training reward.
    """
    from .state import state
    from host_mjlab.host.settings import GROUPS

    s = state(env)
    s.group_rewards.zero_()
    s.group_rewards[:, 0] = 1.0
    head_evaluated = False
    for name in env.reward_manager.active_terms:
        group = name.split("_", 1)[0]
        if group not in GROUPS:
            raise ValueError(f"HoST reward {name!r} must belong to {GROUPS}")
        cfg = env.reward_manager.get_term_cfg(name)
        if cfg.weight == 0:
            continue  # As in original HoST, a disabled task factor is omitted.
        stamp, raw = s.reward_values[name]
        if stamp != env.common_step_counter:
            raise RuntimeError(f"HoST reward {name!r} was not evaluated this step")
        coefficient = cfg.weight if group == "task" else cfg.weight * env.step_dt
        value = raw * coefficient
        index = GROUPS.index(group)
        if group == "task":
            s.group_rewards[:, index] *= value
        else:
            s.group_rewards[:, index] += value
            if s.cfg.constraints.only_positive_rewards:
                s.group_rewards[:, index].clamp_(min=0)
        head_evaluated |= name == "task_head_height"
    # Curriculum measures physical recovery even when its reward is disabled.
    if not head_evaluated:
        s._reward_head_height()
    for index, group in enumerate(GROUPS):
        env.extras["log"]["HoST/" + group] = s.group_rewards[:, index].mean()
    s.reward_step = env.common_step_counter
    s.capture_terminal_observations(env.reset_buf.nonzero(as_tuple=False).flatten())
