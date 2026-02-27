"""
安全检查工具
"""

import numpy as np
from typing import Dict
from config.robot_config import JOINT_LIMITS


class SafetyChecker:
    """安全检查器"""
    
    @staticmethod
    def check_joint_limits(positions: Dict[str, float]) -> bool:
        """检查关节限位"""
        for joint, pos in positions.items():
            for joint_type, limits in JOINT_LIMITS.items():
                if joint_type in joint:
                    if not (limits[0] <= pos <= limits[1]):
                        return False
        return True
        
    @staticmethod
    def check_velocity_limit(positions_prev: Dict[str, float],
                            positions_curr: Dict[str, float],
                            dt: float,
                            limit: float = 5.0) -> bool:
        """检查速度限制"""
        for joint in positions_prev.keys():
            vel = abs(positions_curr[joint] - positions_prev[joint]) / dt
            if vel > limit:
                return False
        return True
