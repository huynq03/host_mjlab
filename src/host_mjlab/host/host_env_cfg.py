"""Native MJLab manager-based configuration for original HoST Mini-Pi ground."""

from dataclasses import dataclass, field
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.utils.spec_config import CollisionCfg
from host_mjlab.asset_zoo.robots.minipi.minipi_constants import robot_cfg
from .mdp.actions import HostActionCfg
from .mdp.observations import history
from .mdp import rewards as mdp_rewards
from .reward_manager import install_reward_manager, reward_step
from .settings import HostObservationCfg
from .mdp.events import reset_robot, randomize_physics, ground_contact_pairs
from .mdp.terminations import time_out, velocity_limit
from .mdp.curricula import recovery


@dataclass(kw_only=True)
class HostEnvCfg(ManagerBasedRlEnvCfg):
    host_observations: HostObservationCfg = field(default_factory=HostObservationCfg)
    reward_term_groups: dict[str, str] = field(default_factory=dict)
    reward_group_positive: tuple[str, ...] = ()


def make_env_cfg(num_envs=4096):
    return HostEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            spec_fn=ground_contact_pairs,
            entities={"robot": robot_cfg()},
            terrain=TerrainEntityCfg(
                terrain_type="plane",
                env_spacing=3.0,
                collisions=(
                    CollisionCfg(
                        geom_names_expr=(".*",),
                        friction=(0.8, 0.0, 0.0),
                        condim=3,
                        contype=1,
                        conaffinity=1,
                        priority=0,
                    ),
                ),
            ),
            sensors=(
                ContactSensorCfg(
                    name="contacts",
                    primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
                    fields=("force", "found"),
                    reduce="netforce",
                ),
            ),
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=8,
                integrator="euler",
                cone="pyramidal",
                disableflags=("multiccd", "nativeccd"),
            ),
            njmax=512,
        ),
        seed=1,
        decimation=4,
        episode_length_s=10.0,
        scale_rewards_by_dt=False,
        observations={
            "actor": ObservationGroupCfg(
                terms={"history": ObservationTermCfg(func=history)},
                concatenate_terms=True,
                enable_corruption=False,
            )
        },
        actions={"host": HostActionCfg(entity_name="robot")},
        rewards={
            "task_head_height": RewardTermCfg(
                func=mdp_rewards.task_head_height,
                weight=1,
                params={
                    "is_gaussian": True,
                    "target_head_margin": 0.37,
                    "target_head_height": 0.37,
                },
            ),
            "task_orientation": RewardTermCfg(
                func=mdp_rewards.task_orientation,
                weight=1,
                params={
                    "is_gaussian": True,
                    "target_base_height_phase1": 0.25,
                    "orientation_threshold": 0.99,
                    "orientation_sigma": 1,
                },
            ),
            "regu_action_rate": RewardTermCfg(
                func=mdp_rewards.regu_action_rate,
                weight=-0.01,
                params={},
            ),
            "regu_dof_acc": RewardTermCfg(
                func=mdp_rewards.regu_dof_acc,
                weight=-2.5e-07,
                params={},
            ),
            "regu_dof_pos_limits": RewardTermCfg(
                func=mdp_rewards.regu_dof_pos_limits,
                weight=-100.0,
                params={},
            ),
            "regu_dof_vel": RewardTermCfg(
                func=mdp_rewards.regu_dof_vel,
                weight=-0.001,
                params={},
            ),
            "regu_dof_vel_limits": RewardTermCfg(
                func=mdp_rewards.regu_dof_vel_limits,
                weight=-1,
                params={"soft_dof_vel_limit": 0.9},
            ),
            "regu_joint_power": RewardTermCfg(
                func=mdp_rewards.regu_joint_power,
                weight=-2.5e-05,
                params={},
            ),
            "regu_joint_tracking_error": RewardTermCfg(
                func=mdp_rewards.regu_joint_tracking_error,
                weight=-0.00025,
                params={},
            ),
            "regu_smoothness": RewardTermCfg(
                func=mdp_rewards.regu_smoothness,
                weight=-0.01,
                params={},
            ),
            "regu_torques": RewardTermCfg(
                func=mdp_rewards.regu_torques,
                weight=-2.5e-06,
                params={},
            ),
            "style_feet_distance": RewardTermCfg(
                func=mdp_rewards.style_feet_distance,
                weight=-10,
                params={},
            ),
            "style_ground_parallel": RewardTermCfg(
                func=mdp_rewards.style_ground_parallel,
                weight=20,
                params={"post_task": False, "target_base_height_phase3": 0.34},
            ),
            "style_hip_roll_deviation": RewardTermCfg(
                func=mdp_rewards.style_hip_roll_deviation,
                weight=-10,
                params={},
            ),
            "style_hip_yaw_deviation": RewardTermCfg(
                func=mdp_rewards.style_hip_yaw_deviation,
                weight=-10,
                params={},
            ),
            "style_knee_deviation": RewardTermCfg(
                func=mdp_rewards.style_knee_deviation,
                weight=-0.25,
                params={},
            ),
            "style_left_foot_displacement": RewardTermCfg(
                func=mdp_rewards.style_left_foot_displacement,
                weight=2.5,
                params={"target_base_height_phase3": 0.34, "left_foot_displacement_sigma": -2},
            ),
            "style_right_foot_displacement": RewardTermCfg(
                func=mdp_rewards.style_right_foot_displacement,
                weight=2.5,
                params={"target_base_height_phase3": 0.34, "right_foot_displacement_sigma": -2},
            ),
            "style_style_ang_vel_xy": RewardTermCfg(
                func=mdp_rewards.style_style_ang_vel_xy,
                weight=1,
                params={"target_base_height_phase1": 0.25},
            ),
            "target_ang_vel_xy": RewardTermCfg(
                func=mdp_rewards.target_ang_vel_xy,
                weight=10,
                params={"target_base_height_phase3": 0.34},
            ),
            "target_feet_height_var": RewardTermCfg(
                func=mdp_rewards.target_feet_height_var,
                weight=2.5,
                params={"target_base_height_phase3": 0.34},
            ),
            "target_lin_vel_xy": RewardTermCfg(
                func=mdp_rewards.target_lin_vel_xy,
                weight=10,
                params={"target_base_height_phase3": 0.34},
            ),
            "target_target_base_height": RewardTermCfg(
                func=mdp_rewards.target_target_base_height,
                weight=10,
                params={"target_base_height_phase3": 0.34, "base_height_target": 0.34},
            ),
            "target_target_orientation": RewardTermCfg(
                func=mdp_rewards.target_target_orientation,
                weight=10,
                params={"target_base_height_phase3": 0.34},
            ),
        },
        terminations={
            "timeout": TerminationTermCfg(func=time_out, time_out=True),
            "velocity": TerminationTermCfg(
                func=velocity_limit, params={"joint_limit": 300.0, "base_limit": 20.0}
            ),
        },
        events={
            "reward_manager": EventTermCfg(func=install_reward_manager, mode="startup"),
            "physics": EventTermCfg(
                func=randomize_physics,
                mode="startup",
                params={
                    "link_mass_range": (0.8, 1.2),
                    "com_range": (-0.03, 0.03),
                    "friction_range": (0.1, 1.0),
                    "restitution_range": (0.0, 1.0),
                    "ground_friction": 0.8,
                    "ground_restitution": 0.3,
                    "contact_time_constant": 0.02,
                },
            ),
            "reward_step": EventTermCfg(func=reward_step, mode="step"),
            "reset": EventTermCfg(
                func=reset_robot,
                mode="reset",
                params={
                    "joint_position_range": (-0.1, 0.1),
                    "kp_range": (0.85, 1.15),
                    "kd_range": (0.85, 1.15),
                    "motor_strength_range": (0.9, 1.1),
                    "actuation_offset_range": (-0.05, 0.05),
                },
            ),
        },
        curriculum={
            "recovery": CurriculumTermCfg(
                func=recovery,
                params={
                    "threshold_height": 0.37,
                    "force_decrement": 20.0,
                    "action_decrement": 0.02,
                    "minimum_action_scale": 0.25,
                },
            )
        },
    )
