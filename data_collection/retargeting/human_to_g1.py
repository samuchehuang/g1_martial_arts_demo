"""
人体姿态到 G1 机器人的重定向

将 MediaPipe 提取的人体关键点映射到 G1 的关节角度
手腕: 1自由度 (wrist_roll)
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class G1JointAngles:
    """G1 关节角度 - 单臂5DOF (手腕1自由度)"""
    # 左臂 5DOF
    left_shoulder_pitch: float = 0.0
    left_shoulder_roll: float = 0.0
    left_shoulder_yaw: float = 0.0
    left_elbow: float = 0.0
    left_wrist_roll: float = 0.0  # 手腕: 1自由度
    # 右臂 5DOF
    right_shoulder_pitch: float = 0.0
    right_shoulder_roll: float = 0.0
    right_shoulder_yaw: float = 0.0
    right_elbow: float = 0.0
    right_wrist_roll: float = 0.0  # 手腕: 1自由度
    
    def to_array(self) -> np.ndarray:
        """转换为 numpy 数组 [10]"""
        return np.array([
            self.left_shoulder_pitch,
            self.left_shoulder_roll,
            self.left_shoulder_yaw,
            self.left_elbow,
            self.left_wrist_roll,
            self.right_shoulder_pitch,
            self.right_shoulder_roll,
            self.right_shoulder_yaw,
            self.right_elbow,
            self.right_wrist_roll,
        ])
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'left_shoulder_pitch': self.left_shoulder_pitch,
            'left_shoulder_roll': self.left_shoulder_roll,
            'left_shoulder_yaw': self.left_shoulder_yaw,
            'left_elbow': self.left_elbow,
            'left_wrist_roll': self.left_wrist_roll,
            'right_shoulder_pitch': self.right_shoulder_pitch,
            'right_shoulder_roll': self.right_shoulder_roll,
            'right_shoulder_yaw': self.right_shoulder_yaw,
            'right_elbow': self.right_elbow,
            'right_wrist_roll': self.right_wrist_roll,
        }


class HumanToG1Retargeter:
    """
    人体到 G1 的重定向器
    
    将 MediaPipe 的 3D 关键点映射到 G1 的关节角度
    手腕: 1自由度 (仅 roll)
    """
    
    def __init__(self):
        """初始化重定向器"""
        # G1 与人体的比例因子
        self.scale_factor = 0.6  # G1 臂展约为人体的 60%
        
        # 关节限位 (弧度) - 单臂5DOF
        self.joint_limits = {
            'shoulder_pitch': (-2.5, 2.5),
            'shoulder_roll': (-1.5, 1.5),
            'shoulder_yaw': (-2.0, 2.0),
            'elbow': (-2.0, 0.0),
            'wrist_roll': (-1.0, 1.0),  # 手腕1自由度限位
        }
    
    def retarget_pose(self, 
                     human_landmarks: np.ndarray,
                     visibility: np.ndarray) -> Optional[G1JointAngles]:
        """
        将人体姿态重定向到 G1
        
        Args:
            human_landmarks: [33, 3] MediaPipe 关键点
            visibility: [33] 可见性
            
        Returns:
            g1_angles: G1 关节角度 (10自由度)
        """
        g1 = G1JointAngles()
        
        # 提取关键点
        lm = human_landmarks
        
        # ========== 左臂 (5DOF, 手腕1自由度) ==========
        if visibility[11] > 0.5 and visibility[13] > 0.5:  # 肩膀和手肘可见
            # 肩部位置
            left_shoulder = lm[11]
            left_elbow = lm[13]
            left_wrist = lm[15] if visibility[15] > 0.5 else None
            
            # 计算肩关节角度
            shoulder_angles = self._calculate_shoulder_angles(
                left_shoulder, left_elbow, is_left=True
            )
            g1.left_shoulder_pitch = shoulder_angles['pitch']
            g1.left_shoulder_roll = shoulder_angles['roll']
            g1.left_shoulder_yaw = shoulder_angles['yaw']
            
            # 计算肘关节角度
            if left_wrist is not None:
                g1.left_elbow = self._calculate_elbow_angle(
                    left_shoulder, left_elbow, left_wrist
                )
                # 计算手腕角度 (1自由度 roll)
                g1.left_wrist_roll = self._calculate_wrist_roll(
                    left_elbow, left_wrist, is_left=True
                )
        
        # ========== 右臂 (5DOF, 手腕1自由度) ==========
        if visibility[12] > 0.5 and visibility[14] > 0.5:
            right_shoulder = lm[12]
            right_elbow = lm[14]
            right_wrist = lm[16] if visibility[16] > 0.5 else None
            
            shoulder_angles = self._calculate_shoulder_angles(
                right_shoulder, right_elbow, is_left=False
            )
            g1.right_shoulder_pitch = shoulder_angles['pitch']
            g1.right_shoulder_roll = shoulder_angles['roll']
            g1.right_shoulder_yaw = shoulder_angles['yaw']
            
            if right_wrist is not None:
                g1.right_elbow = self._calculate_elbow_angle(
                    right_shoulder, right_elbow, right_wrist
                )
                # 计算手腕角度 (1自由度 roll)
                g1.right_wrist_roll = self._calculate_wrist_roll(
                    right_elbow, right_wrist, is_left=False
                )
        
        # 应用限位
        g1 = self._apply_joint_limits(g1)
        
        return g1
    
    def _calculate_shoulder_angles(self,
                                  shoulder: np.ndarray,
                                  elbow: np.ndarray,
                                  is_left: bool) -> Dict[str, float]:
        """
        计算肩关节角度
        
        Args:
            shoulder: 肩膀位置
            elbow: 手肘位置
            is_left: 是否为左臂
            
        Returns:
            angles: {'pitch', 'roll', 'yaw'}
        """
        # 上臂向量
        arm_vector = elbow - shoulder
        
        # 归一化
        arm_length = np.linalg.norm(arm_vector)
        if arm_length < 1e-6:
            return {'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0}
        
        arm_vector = arm_vector / arm_length
        
        # Pitch: 前后摆动 (绕 Y 轴)
        pitch = np.arcsin(-arm_vector[2])  # 负 Z 方向为前
        
        # Roll: 内外展 (绕 X 轴)
        if is_left:
            roll = np.arcsin(arm_vector[1])  # 左臂
        else:
            roll = -np.arcsin(arm_vector[1])  # 右臂
        
        # Yaw: 旋转 (近似计算)
        yaw = np.arctan2(arm_vector[0], -arm_vector[2])
        
        # 缩放以适应 G1 的比例
        pitch *= self.scale_factor
        roll *= self.scale_factor
        yaw *= self.scale_factor
        
        return {
            'pitch': pitch,
            'roll': roll,
            'yaw': yaw,
        }
    
    def _calculate_elbow_angle(self,
                              shoulder: np.ndarray,
                              elbow: np.ndarray,
                              wrist: np.ndarray) -> float:
        """
        计算肘关节角度
        
        Args:
            shoulder: 肩膀
            elbow: 手肘
            wrist: 手腕
            
        Returns:
            angle: 肘关节角度 (弧度)
        """
        # 上臂和前臂向量
        upper_arm = shoulder - elbow
        forearm = wrist - elbow
        
        # 归一化
        upper_arm_norm = np.linalg.norm(upper_arm)
        forearm_norm = np.linalg.norm(forearm)
        
        if upper_arm_norm < 1e-6 or forearm_norm < 1e-6:
            return 0.0
        
        upper_arm = upper_arm / upper_arm_norm
        forearm = forearm / forearm_norm
        
        # 计算夹角
        cosine = np.dot(upper_arm, forearm)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        # G1 的肘关节为负值（弯曲）
        return -angle * self.scale_factor
    
    def _calculate_wrist_roll(self,
                             elbow: np.ndarray,
                             wrist: np.ndarray,
                             is_left: bool) -> float:
        """
        计算手腕翻滚角度 (1自由度)
        
        Args:
            elbow: 手肘位置
            wrist: 手腕位置
            is_left: 是否为左臂
            
        Returns:
            roll_angle: 手腕翻滚角度 (弧度)
        """
        # 前臂方向
        forearm = wrist - elbow
        
        # 基于前臂方向估算手腕翻滚
        # 简化计算: 使用前臂的旋转来推断手腕翻滚
        forearm_norm = np.linalg.norm(forearm)
        if forearm_norm < 1e-6:
            return 0.0
        
        forearm = forearm / forearm_norm
        
        # 根据前臂的 XY 平面投影计算翻滚
        # 当手臂平举时，roll 为 0
        # 当手臂向内/向外旋转时，产生 roll
        roll = np.arctan2(forearm[1], forearm[0])
        
        # 左右臂对称处理
        if not is_left:
            roll = -roll
        
        # 缩放并限幅
        roll *= self.scale_factor * 0.5  # 手腕动作幅度较小
        
        return np.clip(roll, *self.joint_limits['wrist_roll'])
    
    def _apply_joint_limits(self, g1: G1JointAngles) -> G1JointAngles:
        """应用关节限位 (单臂5DOF, 手腕1自由度)"""
        # 左臂
        g1.left_shoulder_pitch = np.clip(
            g1.left_shoulder_pitch, *self.joint_limits['shoulder_pitch']
        )
        g1.left_shoulder_roll = np.clip(
            g1.left_shoulder_roll, *self.joint_limits['shoulder_roll']
        )
        g1.left_shoulder_yaw = np.clip(
            g1.left_shoulder_yaw, *self.joint_limits['shoulder_yaw']
        )
        g1.left_elbow = np.clip(
            g1.left_elbow, *self.joint_limits['elbow']
        )
        g1.left_wrist_roll = np.clip(
            g1.left_wrist_roll, *self.joint_limits['wrist_roll']
        )
        
        # 右臂
        g1.right_shoulder_pitch = np.clip(
            g1.right_shoulder_pitch, *self.joint_limits['shoulder_pitch']
        )
        g1.right_shoulder_roll = np.clip(
            g1.right_shoulder_roll, *self.joint_limits['shoulder_roll']
        )
        g1.right_shoulder_yaw = np.clip(
            g1.right_shoulder_yaw, *self.joint_limits['shoulder_yaw']
        )
        g1.right_elbow = np.clip(
            g1.right_elbow, *self.joint_limits['elbow']
        )
        g1.right_wrist_roll = np.clip(
            g1.right_wrist_roll, *self.joint_limits['wrist_roll']
        )
        
        return g1
    
    def smooth_trajectory(self,
                         trajectory: List[G1JointAngles],
                         window_size: int = 5) -> List[G1JointAngles]:
        """
        平滑轨迹
        
        Args:
            trajectory: G1 关节角度序列 (10自由度)
            window_size: 平滑窗口大小
            
        Returns:
            smoothed: 平滑后的轨迹
        """
        if len(trajectory) < window_size:
            return trajectory
        
        smoothed = []
        half_window = window_size // 2
        
        # 所有关节名称 (10个)
        joint_names = [
            'left_shoulder_pitch', 'left_shoulder_roll', 
            'left_shoulder_yaw', 'left_elbow', 'left_wrist_roll',
            'right_shoulder_pitch', 'right_shoulder_roll',
            'right_shoulder_yaw', 'right_elbow', 'right_wrist_roll'
        ]
        
        for i in range(len(trajectory)):
            # 获取窗口
            start = max(0, i - half_window)
            end = min(len(trajectory), i + half_window + 1)
            window = trajectory[start:end]
            
            # 平均 - 所有10个关节
            avg = G1JointAngles()
            for joint_name in joint_names:
                values = [getattr(p, joint_name) for p in window]
                setattr(avg, joint_name, np.mean(values))
            
            smoothed.append(avg)
        
        return smoothed


if __name__ == "__main__":
    retargeter = HumanToG1Retargeter()
    print("G1 Retargeter initialized (wrist: 1 DOF)")
