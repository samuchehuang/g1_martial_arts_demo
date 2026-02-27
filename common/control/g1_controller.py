"""
G1 控制器 - 共享模块
用于 Sim-to-Real 阶段
"""

import time
import numpy as np
from typing import Dict, Optional

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.utils.crc import CRC
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

from config.robot_config import G1_ARM_JOINTS, JOINT_LIMITS


class G1Controller:
    """G1 手臂控制器"""
    
    def __init__(self, network_interface: str = "eth0", simulation: bool = False):
        self.simulation = simulation or not SDK_AVAILABLE
        self.network_interface = network_interface
        self.current_positions = {name: 0.0 for name in G1_ARM_JOINTS.keys()}
        
        if not self.simulation:
            self._init_sdk()
        else:
            print("[Controller] 模拟模式")
            
    def _init_sdk(self):
        ChannelFactoryInitialize(0, self.network_interface)
        self.arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_pub.Init()
        self.low_state = LowState_()
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self._state_callback)
        self.crc = CRC()
        self.low_cmd = LowCmd_()
        self.low_cmd.level_flag = 0xFF
        
    def _state_callback(self, msg):
        self.low_state = msg
        for name, idx in G1_ARM_JOINTS.items():
            if idx < len(msg.motor_state):
                self.current_positions[name] = msg.motor_state[idx].q
                
    def enable_sdk(self):
        """启用 SDK 模式"""
        if self.simulation:
            return
        self.low_cmd.motor_cmd[9].q = 1
        self._send()
        
    def send_positions(self, positions: Dict[str, float], kp: float = 60.0, kd: float = 1.5):
        """发送位置指令"""
        if self.simulation:
            for joint, pos in positions.items():
                if joint in self.current_positions:
                    self.current_positions[joint] = pos
            return
            
        for joint_name, pos in positions.items():
            if joint_name in G1_ARM_JOINTS:
                idx = G1_ARM_JOINTS[joint_name]
                pos = self._clamp(joint_name, pos)
                self.low_cmd.motor_cmd[idx].q = pos
                self.low_cmd.motor_cmd[idx].dq = 0.0
                self.low_cmd.motor_cmd[idx].kp = kp
                self.low_cmd.motor_cmd[idx].kd = kd
                self.low_cmd.motor_cmd[idx].tau = 0.0
        self._send()
        
    def _send(self):
        if not self.simulation:
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.arm_pub.Write(self.low_cmd)
            
    def _clamp(self, joint_name: str, value: float) -> float:
        for joint_type, limits in JOINT_LIMITS.items():
            if joint_type in joint_name:
                return np.clip(value, limits[0], limits[1])
        return value
        
    def get_positions(self) -> Dict[str, float]:
        if self.simulation:
            return self.current_positions.copy()
        positions = {}
        for name, idx in G1_ARM_JOINTS.items():
            if idx < len(self.low_state.motor_state):
                positions[name] = self.low_state.motor_state[idx].q
            else:
                positions[name] = 0.0
        return positions
        
    def emergency_stop(self):
        print("[Emergency] 紧急停止")
        if self.simulation:
            return
        current = self.get_positions()
        for name, pos in current.items():
            idx = G1_ARM_JOINTS[name]
            self.low_cmd.motor_cmd[idx].q = pos
            self.low_cmd.motor_cmd[idx].dq = 0.0
            self.low_cmd.motor_cmd[idx].kp = 80.0
            self.low_cmd.motor_cmd[idx].kd = 2.0
        self._send()
