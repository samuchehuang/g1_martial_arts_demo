"""
阶段2 Sim-to-Real: Gym 微调后的实机部署
"""

import time
import pickle
import numpy as np
from pathlib import Path

from common.control.g1_controller import G1Controller
from common.adapter.trajectory_adapter import TrajectoryAdapter
from common.utils.safety import SafetyChecker


class Stage2Deployer:
    """
    阶段2 实机部署器
    
    部署经过 Gym 微调的轨迹
    支持更激进的参数（因为已经过优化）
    """
    
    def __init__(self,
                 controller: G1Controller,
                 safety_checker: SafetyChecker = None):
        self.controller = controller
        self.safety = safety_checker or SafetyChecker()
        self.dt = 1.0 / 100
        
    def deploy(self,
              final_trajectory_path: str,
              speed: float = 1.0,
              use_optimized_gains: bool = True) -> bool:
        """
        部署最终优化的轨迹
        
        Args:
            final_trajectory_path: 最终轨迹路径
            speed: 播放速度
            use_optimized_gains: 使用优化的 PD 参数
            
        Returns:
            success: 是否成功
        """
        print(f"\n[S2 Deploy] 部署 Gym 优化轨迹")
        print(f"  文件: {final_trajectory_path}")
        
        with open(final_trajectory_path, 'rb') as f:
            data = pickle.load(f)
            
        trajectory = data['q']
        validation = data.get('validation', {})
        
        print(f"  轨迹长度: {len(trajectory)}")
        print(f"  改进分数: {validation.get('improvement_score', 'N/A')}")
        
        # 根据改进分数调整增益
        if use_optimized_gains and 'improvement_score' in validation:
            improvement = validation['improvement_score']
            kp = 60 + improvement * 20  # 改进越好，增益可以越高
            kd = 1.5 + improvement * 0.5
            print(f"  优化增益: kp={kp:.1f}, kd={kd:.1f}")
        else:
            kp, kd = 60.0, 1.5
            
        # 转换为指令
        commands = TrajectoryAdapter.dict_to_g1_commands(trajectory)
        
        # 执行
        prev_positions = self.controller.get_positions()
        
        for i, cmd in enumerate(commands):
            # 安全检查（阶段2可以放宽一些）
            if not self.safety.check_joint_limits(cmd):
                print(f"[S2 Safety] 第 {i} 帧超出限位")
                self.controller.emergency_stop()
                return False
                
            # 发送指令
            self.controller.send_positions(cmd, kp=kp, kd=kd)
            prev_positions = cmd
            
            time.sleep(self.dt / speed)
            
        print("[S2 Deploy] 部署完成")
        return True
        
    def compare_with_stage1(self,
                           stage1_trajectory_path: str,
                           stage2_trajectory_path: str):
        """
        对比阶段1和阶段2的实际执行效果
        
        Args:
            stage1_trajectory_path: 阶段1轨迹
            stage2_trajectory_path: 阶段2轨迹
        """
        print("\n[S2 Compare] 对比 S1 vs S2 执行效果")
        
        # 先执行阶段1
        print("\n  执行阶段1轨迹...")
        self.deploy(stage1_trajectory_path, speed=0.5)
        time.sleep(2)
        
        # 再执行阶段2
        print("\n  执行阶段2轨迹...")
        self.deploy(stage2_trajectory_path, speed=0.5)
        
        print("\n  对比完成 - 观察平滑度和稳定性差异")
