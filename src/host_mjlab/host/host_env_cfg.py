"""Native MJLab manager-based configuration for original HoST Mini-Pi ground."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.utils.spec_config import CollisionCfg
from host_mjlab.assets.minipi.robot import robot_cfg
from .settings import GROUPS
from .mdp.actions import HostActionCfg
from .mdp.observations import history
from .mdp.rewards import reward_group
from .mdp.events import reset_robot, randomize_physics, ground_contact_pairs
from .mdp.terminations import time_out, velocity_limit
from .mdp.curricula import recovery


def make_env_cfg(num_envs=4096):
    return ManagerBasedRlEnvCfg(
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
            g: RewardTermCfg(func=reward_group, weight=1.0, params={"group": g}) for g in GROUPS
        },
        terminations={
            "timeout": TerminationTermCfg(func=time_out, time_out=True),
            "velocity": TerminationTermCfg(func=velocity_limit),
        },
        events={
            "physics": EventTermCfg(func=randomize_physics, mode="startup"),
            "reset": EventTermCfg(func=reset_robot, mode="reset"),
        },
        curriculum={"recovery": CurriculumTermCfg(func=recovery)},
    )
