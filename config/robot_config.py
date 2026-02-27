"""
G1 机器人配置 - 基础版 (双臂共10自由度)

官方关节序列参考: https://support.unitree.com/home/zh/G1_developer/joint_motor_sequence

单臂自由度分布 (5DOF):
- 肩部: 3个 (pitch, roll, yaw)
- 肘部: 1个 (elbow)
- 手腕: 1个 (wrist_roll)
- 总计: 5个/臂 × 2臂 = 10个手臂自由度

左臂关节索引: 11-15
右臂关节索引: 16-20
"""

# G1 基础版双臂关节索引 (10个手臂自由度)
G1_ARM_JOINTS = {
    # 左臂 5DOF (索引 11-15)
    'left_shoulder_pitch': 11,
    'left_shoulder_roll': 12,
    'left_shoulder_yaw': 13,
    'left_elbow': 14,
    'left_wrist_roll': 15,  # 手腕: 1自由度 (roll)
    # 右臂 5DOF (索引 16-20)
    'right_shoulder_pitch': 16,
    'right_shoulder_roll': 17,
    'right_shoulder_yaw': 18,
    'right_elbow': 19,
    'right_wrist_roll': 20,  # 手腕: 1自由度 (roll)
}

# 关节限位 (弧度) - 基于官方文档
JOINT_LIMITS = {
    'shoulder_pitch': (-2.5, 2.5),
    'shoulder_roll': (-1.5, 1.5),
    'shoulder_yaw': (-2.0, 2.0),
    'elbow': (-2.0, 0.0),
    'wrist_roll': (-1.0, 1.0),  # 手腕翻滚限位
}

# 关节名列表 (LeRobot 格式用)
JOINT_NAMES = list(G1_ARM_JOINTS.keys())

# 单臂关节名称 (用于遍历)
LEFT_ARM_JOINTS = ['left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow', 'left_wrist_roll']
RIGHT_ARM_JOINTS = ['right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow', 'right_wrist_roll']

# LeRobot 数据集到 G1 的映射 (一对一映射)
LEROBOT_TO_G1_MAPPING = {name: name for name in JOINT_NAMES}

# 控制参数
CONTROL_FREQ = 100  # Hz
SIM_FREQ = 1000     # MuJoCo Hz

# 关节数量
NUM_ARM_JOINTS = 10  # 双臂共10自由度
NUM_JOINTS_PER_ARM = 5  # 单臂5自由度
