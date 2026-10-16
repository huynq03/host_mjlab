"""Standalone original HoST reward equations; functions return unscaled vectors."""

import torch
import numpy as np
from .math import tolerance
from .state import state


def task_head_height(env, *, is_gaussian=True, target_head_margin=0.37, target_head_height=0.37):
    s = state(env)
    if not is_gaussian:
        head_height = s.rigid_body_states[:, s.head_indices, 2].clone()
        return head_height.squeeze(1).clamp(0, 1)
    else:
        head_height = s.rigid_body_states[:, s.head_indices, 2].clone()
        feet_height = s.rigid_body_states[:, s.feet_indices, 2].clone().mean(-1).unsqueeze(-1)
        head_height -= feet_height
        reward = tolerance(head_height, (target_head_height, np.inf), target_head_margin, 0.1)
        return reward.squeeze(1)


def task_orientation(
    env,
    *,
    is_gaussian=True,
    target_base_height_phase1=0.25,
    orientation_threshold=0.99,
    orientation_sigma=1,
):
    s = state(env)
    if not is_gaussian:
        mse_error = torch.sum(
            torch.square(s.projected_gravity - torch.tensor([0, 0, 1], device=env.device)), dim=-1
        )
        reward = torch.exp(mse_error / orientation_sigma) * (s.root_states[:, 2] > 0.4)
    else:
        base_height = s.root_states[:, 2] > target_base_height_phase1
        reward = tolerance(-s.projected_gravity[:, 2], [orientation_threshold, np.inf], 1.0, 0.05)
    return reward


def regu_action_rate(env):
    s = state(env)
    return torch.sum(torch.square(s.last_actions - s.actions), dim=1)


def regu_dof_acc(env):
    s = state(env)
    return torch.sum(torch.square((s.last_dof_vel - s.dof_vel) / env.step_dt), dim=1)


def regu_dof_pos_limits(env):
    s = state(env)
    out_of_limits = -(s.dof_pos - s.dof_pos_limits[:, 0]).clip(max=0.0)
    out_of_limits += (s.dof_pos - s.dof_pos_limits[:, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def regu_dof_vel(env):
    s = state(env)
    return torch.sum(torch.square(s.dof_vel), dim=1)


def regu_dof_vel_limits(env, *, soft_dof_vel_limit=0.9):
    s = state(env)
    return torch.sum(
        (torch.abs(s.dof_vel) - s.dof_vel_limits * soft_dof_vel_limit).clip(min=0.0, max=1.0), dim=1
    )


def regu_joint_power(env):
    s = state(env)
    return torch.sum(torch.abs(s.dof_vel) * torch.abs(s.torques), dim=1)


def regu_joint_tracking_error(env):
    s = state(env)
    return torch.sum(torch.square(s.joint_pos_target - s.dof_pos), dim=-1)


def regu_smoothness(env):
    s = state(env)
    return torch.sum(
        torch.square(s.actions - s.last_actions - s.last_actions + s.last_last_actions), dim=1
    )


def regu_torques(env):
    s = state(env)
    return torch.sum(torch.square(s.torques), dim=1)


def style_feet_distance(env):
    s = state(env)
    left_foot_pos = s.rigid_body_states[:, s.left_foot_indices, :3].clone()
    right_foot_pos = s.rigid_body_states[:, s.right_foot_indices, :3].clone()
    feet_distances = torch.norm(left_foot_pos - right_foot_pos, dim=-1)
    reward = tolerance(feet_distances, [0, 0.4], 0.3, 0.05)
    return (feet_distances > 0.45).squeeze(1)


def style_ground_parallel(env, *, post_task=False, target_base_height_phase3=0.34):
    s = state(env)
    left_ankle_pos = s.rigid_body_states[:, s.left_ankle_indices, 2].clone() * 10
    right_ankle_pos = s.rigid_body_states[:, s.right_ankle_indices, 2].clone() * 10
    var = left_ankle_pos.var(1) + right_ankle_pos.var(1)
    var = torch.mean(
        torch.concat(
            [left_ankle_pos.var(1).view(-1, 1), right_ankle_pos.var(1).view(-1, 1)], dim=-1
        ),
        dim=-1,
    )
    reward = var < 0.05
    if post_task:
        standup = s.root_states[:, 2] > target_base_height_phase3
        reward = reward * ~standup + torch.ones_like(reward) * standup
    return reward


def style_hip_roll_deviation(env):
    s = state(env)
    hip_roll_dof = s.dof_pos[:, s.hip_roll_joint_indices]
    reward = (torch.max(torch.abs(s.dof_pos[:, s.hip_roll_joint_indices]), dim=-1)[0] > 1.4) | (
        torch.min(torch.abs(s.dof_pos[:, s.hip_roll_joint_indices]), dim=-1)[0] > 0.9
    )
    return reward


def style_hip_yaw_deviation(env):
    s = state(env)
    hip_yaw_dof = s.dof_pos[:, s.hip_joint_indices]
    reward = (torch.max(torch.abs(s.dof_pos[:, s.hip_joint_indices]), dim=-1)[0] > 1.4) | (
        torch.min(torch.abs(s.dof_pos[:, s.hip_joint_indices]), dim=-1)[0] > 0.9
    )
    return reward


def style_knee_deviation(env):
    s = state(env)
    hip_roll_dof = s.dof_pos[:, s.knee_joint_indices]
    reward = (torch.max(torch.abs(s.dof_pos[:, s.knee_joint_indices]), dim=-1)[0] > 1.65) | (
        torch.min(s.dof_pos[:, s.knee_joint_indices], dim=-1)[0] < -0.6
    )
    return reward


def style_left_foot_displacement(
    env, *, target_base_height_phase3=0.34, left_foot_displacement_sigma=-2
):
    s = state(env)
    base_xy = s.root_states[:, :2].clone()
    left_foot_xy = s.rigid_body_states[:, s.left_foot_indices, :2].squeeze(1)
    mse = torch.sum(torch.square(base_xy - left_foot_xy), dim=-1)
    mse_error = mse.clamp(0.3, np.inf)
    reward = torch.exp(mse_error * left_foot_displacement_sigma) * (
        s.rigid_body_states[:, s.left_foot_indices, 2] < 0.15
    ).squeeze(1)
    standup = s.root_states[:, 2] > target_base_height_phase3
    return reward * standup


def style_right_foot_displacement(
    env, *, target_base_height_phase3=0.34, right_foot_displacement_sigma=-2
):
    s = state(env)
    base_xy = s.root_states[:, :2].clone()
    right_foot_xy = s.rigid_body_states[:, s.right_foot_indices, :2].squeeze(1)
    mse = torch.sum(torch.square(base_xy - right_foot_xy), dim=-1)
    mse_error = mse.clamp(0.3, np.inf)
    reward = torch.exp(mse_error * right_foot_displacement_sigma) * (
        s.rigid_body_states[:, s.right_foot_indices, 2] < 0.15
    ).squeeze(1)
    standup = s.root_states[:, 2] > target_base_height_phase3
    return reward * standup


def style_style_ang_vel_xy(env, *, target_base_height_phase1=0.25):
    s = state(env)
    base_height = s.root_states[:, 2] > target_base_height_phase1
    return torch.exp(torch.sum(torch.square(s.base_ang_vel[:, :2]), dim=1) * -2) * base_height


def target_ang_vel_xy(env, *, target_base_height_phase3=0.34):
    s = state(env)
    base_height = s.root_states[:, 2] > target_base_height_phase3
    reward = torch.exp(torch.sum(torch.square(s.base_ang_vel[:, :2]), dim=1) * -2) * base_height
    return reward


def target_feet_height_var(env, *, target_base_height_phase3=0.34):
    s = state(env)
    left_foot_height = s.rigid_body_states[:, s.left_foot_indices, 2].clone() * 10
    right_foot_height = s.rigid_body_states[:, s.right_foot_indices, 2].clone() * 10
    feet_distance = torch.abs(left_foot_height - right_foot_height).squeeze(1).clamp(0.2, np.inf)
    standup = s.root_states[:, 2] > target_base_height_phase3
    reward = torch.exp(feet_distance * -2) * standup
    return reward


def target_lin_vel_xy(env, *, target_base_height_phase3=0.34):
    s = state(env)
    base_height = s.root_states[:, 2] > target_base_height_phase3
    reward = torch.exp(torch.sum(torch.square(s.base_lin_vel[:, :2]), dim=1) * -5) * base_height
    return reward


def target_target_base_height(env, *, target_base_height_phase3=0.34, base_height_target=0.34):
    s = state(env)
    base_height = s.root_states[:, 2]
    standup = s.root_states[:, 2] > target_base_height_phase3
    return torch.exp(torch.abs(base_height - base_height_target) * -20) * standup


def target_target_orientation(env, *, target_base_height_phase3=0.34):
    s = state(env)
    standup = s.root_states[:, 2] > target_base_height_phase3
    return torch.exp(torch.sum(torch.square(s.projected_gravity[:, :2]), dim=1) * -5) * standup
