"""Original URDF joint order and physical limits."""

JOINTS = [{'name': 'r_hip_pitch_joint', 'lower': -3.0, 'upper': 3.0, 'effort': 20.0, 'velocity': 5.0},
 {'name': 'r_hip_roll_joint', 'lower': -3.0, 'upper': 0.5, 'effort': 20.0, 'velocity': 5.0},
 {'name': 'r_thigh_joint', 'lower': -3.0, 'upper': 3.0, 'effort': 20.0, 'velocity': 5.0},
 {'name': 'r_calf_joint', 'lower': -0.65, 'upper': 1.65, 'effort': 20.0, 'velocity': 5.0},
 {'name': 'r_ankle_pitch_joint', 'lower': -0.5, 'upper': 1.3, 'effort': 20.0, 'velocity': 5.0},
 {'name': 'r_ankle_roll_joint', 'lower': -0.15, 'upper': 0.15, 'effort': 20.0, 'velocity': 5.0},
 {'name': 'l_hip_pitch_joint', 'lower': -3.0, 'upper': 3.0, 'effort': 20.0, 'velocity': 5.0},
 {'name': 'l_hip_roll_joint', 'lower': -0.5, 'upper': 3.0, 'effort': 20.0, 'velocity': 5.0},
 {'name': 'l_thigh_joint', 'lower': -3.0, 'upper': 3.0, 'effort': 20.0, 'velocity': 5.0},
 {'name': 'l_calf_joint', 'lower': -0.65, 'upper': 1.65, 'effort': 20.0, 'velocity': 5.0},
 {'name': 'l_ankle_pitch_joint', 'lower': -0.5, 'upper': 1.3, 'effort': 20.0, 'velocity': 5.0},
 {'name': 'l_ankle_roll_joint', 'lower': -0.15, 'upper': 0.15, 'effort': 20.0, 'velocity': 5.0}]
