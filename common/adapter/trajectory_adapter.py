"""
轨迹适配器 - 共享模块
处理不同格式间的转换
"""

import numpy as np
from typing import Dict, List
from config.robot_config import JOINT_NAMES, CONTROL_FREQ


class TrajectoryAdapter:
    """轨迹格式适配器"""
    
    @staticmethod
    def lerobot_to_dict(lerobot_batch: Dict) -> Dict[str, np.ndarray]:
        """
        LeRobot 批次数据转换为字典格式
        
        lerobot_batch: {
            'observation.state': [B, T, num_joints],
            'action': [B, T, num_joints],
            ...
        }
        """
        return {
            'positions': lerobot_batch['action'].cpu().numpy(),
            'observation': lerobot_batch['observation.state'].cpu().numpy(),
        }
        
    @staticmethod
    def dict_to_g1_commands(trajectory: np.ndarray, 
                           joint_names: List[str] = None) -> List[Dict[str, float]]:
        """
        将轨迹数组转换为 G1 指令列表
        
        Args:
            trajectory: [T, num_joints] 位置数组
            joint_names: 关节名列表，默认使用 config 中的
            
        Returns:
            commands: 指令列表，每个元素是 {joint_name: value}
        """
        if joint_names is None:
            joint_names = JOINT_NAMES[:trajectory.shape[1]]
            
        commands = []
        for t in range(len(trajectory)):
            cmd = {}
            for i, joint in enumerate(joint_names):
                if i < trajectory.shape[1]:
                    cmd[joint] = float(trajectory[t, i])
            commands.append(cmd)
        return commands
        
    @staticmethod
    def resample(trajectory: np.ndarray, 
                 original_fps: int, 
                 target_fps: int = CONTROL_FREQ) -> np.ndarray:
        """重采样轨迹到目标频率"""
        from scipy.interpolate import interp1d
        
        T, J = trajectory.shape
        duration = T / original_fps
        
        x_old = np.linspace(0, duration, T)
        x_new = np.linspace(0, duration, int(duration * target_fps))
        
        resampled = np.zeros((len(x_new), J))
        for j in range(J):
            f = interp1d(x_old, trajectory[:, j], kind='linear')
            resampled[:, j] = f(x_new)
            
        return resampled
