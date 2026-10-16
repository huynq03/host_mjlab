"""Original Mini-Pi asset; torque control remains in the HoST action term."""

from pathlib import Path
import mujoco
from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

ROOT = Path(__file__).parent
from .joints import JOINTS
JOINT_NAMES = tuple(j["name"] for j in JOINTS)


def get_spec():
    return mujoco.MjSpec.from_file(str(ROOT / "minipi.xml"))


def entity_spec():
    spec = get_spec()
    defaults = mujoco.MjSpec().option
    spec.option.timestep = defaults.timestep
    spec.option.iterations = defaults.iterations
    return spec


def robot_cfg():
    return EntityCfg(
        spec_fn=entity_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0, 0, 0.351),
            rot=(2**-0.5, 0, -(2**-0.5), 0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(
                BuiltinMotorActuatorCfg(
                    target_names_expr=(j["name"],), effort_limit=j["effort"], armature=0.01
                )
                for j in JOINTS
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
