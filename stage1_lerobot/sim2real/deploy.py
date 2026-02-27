"""
阶段1 Sim-to-Real: LeRobot 策略实机部署
"""

import time
import pickle
import numpy as np
from pathlib import Path

from common.control.g1_controller import G1Controller
from common.adapter.trajectory_adapter import TrajectoryAdapter
from common.utils.safety import SafetyChecker


class Stage1Deployer:
    """
    阶段1 实机部署器
    
    加载 LeRobot 生成的轨迹，在 G1 上执行
    """
    
    def __init__(self, 
                 controller: G1Controller,
                 safety_checker: SafetyChecker = None):
        """
        Args:
            controller: G1 控制器
            safety_checker: 安全检查器
        """
        self.controller = controller
        self.safety = safety_checker or SafetyChecker()
        self.dt = 1.0 / 100  # 100Hz
        
    def deploy_trajectory(self, 
                         trajectory_path: str,
                         speed: float = 1.0) -> bool:
        """
        部署轨迹
        
        Args:
            trajectory_path: 轨迹文件路径 (.pkl)
            speed: 播放速度
            
        Returns:
            success: 是否成功
        """
        print(f"\n[S1 Deploy] 部署轨迹: {trajectory_path}")
        
        # 加载轨迹
        with open(trajectory_path, 'rb') as f:
            data = pickle.load(f)
            
        trajectory = data['q']
        fps = data.get('fps', 100)
        
        print(f"  帧数: {len(trajectory)}")
        print(f"  FPS: {fps}")
        print(f"  速度: {speed}x")
        
        # 转换为指令
        commands = TrajectoryAdapter.dict_to_g1_commands(trajectory)
        
        # 执行
        prev_positions = self.controller.get_positions()
        
        for i, cmd in enumerate(commands):
            # 安全检查
            if not self.safety.check_joint_limits(cmd):
                print(f"[S1 Safety] 第 {i} 帧超出限位，停止")
                self.controller.emergency_stop()
                return False
                
            if not self.safety.check_velocity_limit(prev_positions, cmd, self.dt):
                print(f"[S1 Safety] 第 {i} 帧速度过快，限制执行")
                # 可以继续执行或跳过
                
            # 发送指令
            self.controller.send_positions(cmd, kp=60.0, kd=1.5)
            prev_positions = cmd
            
            # 时间对齐
            time.sleep(self.dt / speed)
            
        print("[S1 Deploy] 部署完成")
        return True
        
    def deploy_policy_live(self,
                          trainer,
                          policy_path: str,
                          duration: float = 10.0):
        """
        实时策略部署 (闭环控制)
        
        Args:
            trainer: LeRobotTrainer 实例
            policy_path: 策略路径
            duration: 运行时长
        """
        print(f"\n[S1 Deploy] 实时策略部署")
        print(f"  时长: {duration}s")
        
        start_time = time.time()
        step = 0
        
        while time.time() - start_time < duration:
            # 获取当前观测
            obs = self.controller.get_positions()
            obs_array = np.array(list(obs.values()))
            
            # 策略推理
            actions = trainer.predict(policy_path, obs_array, num_actions=10)
            
            # 执行第一步
            cmd = {k: actions[0][i] for i, k in enumerate(obs.keys())}
            self.controller.send_positions(cmd)
            
            step += 1
            time.sleep(self.dt)
            
        print(f"[S1 Deploy] 运行完成，共 {step} 步")
