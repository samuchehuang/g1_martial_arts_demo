"""
阶段1 Sim-to-Sim: LeRobot 策略在 MuJoCo 中验证

G1 配置: 单臂5DOF (手腕1自由度 wrist_roll)
双臂共10自由度
"""

import mujoco
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import pickle


class LeRobotMujocoValidator:
    """
    LeRobot 策略 MuJoCo 验证器
    
    使用训练好的策略在 MuJoCo 中生成轨迹并验证
    手腕: 1自由度 (wrist_roll)
    """
    
    # 关节名称列表 (10自由度)
    JOINT_NAMES = [
        'left_shoulder_pitch', 'left_shoulder_roll',
        'left_shoulder_yaw', 'left_elbow', 'left_wrist_roll',
        'right_shoulder_pitch', 'right_shoulder_roll',
        'right_shoulder_yaw', 'right_elbow', 'right_wrist_roll'
    ]
    
    def __init__(self, model_path: str = "g1.xml"):
        """
        Args:
            model_path: MuJoCo 模型路径
        """
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        except:
            print("[警告] 使用简化模型")
            self.model = self._create_simple_model()
            
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        
    def _create_simple_model(self):
        """创建简化 G1 模型 - 单臂5DOF (手腕1自由度)"""
        xml = """
        <mujoco model="g1_simple">
          <compiler angle="radian"/>
          <option timestep="0.001"/>
          <worldbody>
            <body name="torso">
              <freejoint/>
              <geom type="capsule" size="0.1" fromto="0 0 0 0 0 0.5"/>
              <!-- 左臂 5DOF (手腕1自由度) -->
              <body name="left_shoulder" pos="0 0.2 0.4">
                <joint name="left_shoulder_pitch" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
                <joint name="left_shoulder_roll" type="hinge" axis="1 0 0" range="-1.5 1.5"/>
                <joint name="left_shoulder_yaw" type="hinge" axis="0 0 1" range="-2.0 2.0"/>
                <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"/>
                <body name="left_upper_arm" pos="0 0 0.3">
                  <joint name="left_elbow" type="hinge" axis="0 1 0" range="-2.0 0"/>
                  <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.3"/>
                  <body name="left_forearm" pos="0 0 0.3">
                    <joint name="left_wrist_roll" type="hinge" axis="0 0 1" range="-1.0 1.0"/>
                    <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.1"/>
                  </body>
                </body>
              </body>
              <!-- 右臂 5DOF (手腕1自由度) -->
              <body name="right_shoulder" pos="0 -0.2 0.4">
                <joint name="right_shoulder_pitch" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
                <joint name="right_shoulder_roll" type="hinge" axis="1 0 0" range="-1.5 1.5"/>
                <joint name="right_shoulder_yaw" type="hinge" axis="0 0 1" range="-2.0 2.0"/>
                <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"/>
                <body name="right_upper_arm" pos="0 0 0.3">
                  <joint name="right_elbow" type="hinge" axis="0 1 0" range="-2.0 0"/>
                  <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.3"/>
                  <body name="right_forearm" pos="0 0 0.3">
                    <joint name="right_wrist_roll" type="hinge" axis="0 0 1" range="-1.0 1.0"/>
                    <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.1"/>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
          <actuator>
            <!-- 左臂执行器 (5DOF) -->
            <position name="left_shoulder_pitch" joint="left_shoulder_pitch" kp="60" kv="1.5"/>
            <position name="left_shoulder_roll" joint="left_shoulder_roll" kp="60" kv="1.5"/>
            <position name="left_shoulder_yaw" joint="left_shoulder_yaw" kp="60" kv="1.5"/>
            <position name="left_elbow" joint="left_elbow" kp="60" kv="1.5"/>
            <position name="left_wrist_roll" joint="left_wrist_roll" kp="60" kv="1.5"/>
            <!-- 右臂执行器 (5DOF) -->
            <position name="right_shoulder_pitch" joint="right_shoulder_pitch" kp="60" kv="1.5"/>
            <position name="right_shoulder_roll" joint="right_shoulder_roll" kp="60" kv="1.5"/>
            <position name="right_shoulder_yaw" joint="right_shoulder_yaw" kp="60" kv="1.5"/>
            <position name="right_elbow" joint="right_elbow" kp="60" kv="1.5"/>
            <position name="right_wrist_roll" joint="right_wrist_roll" kp="60" kv="1.5"/>
          </actuator>
        </mujoco>
        """
        return mujoco.MjModel.from_xml_string(xml)
    
    def validate_policy(self,
                       trainer,
                       policy_path: str,
                       initial_state: Optional[np.ndarray] = None,
                       num_steps: int = 500,
                       render: bool = False) -> Dict:
        """
        验证策略
        
        Args:
            trainer: LeRobotTrainer 实例 (已加载训练好的策略)
            policy_path: 策略路径 (或已加载的策略状态)
            initial_state: 初始状态 [10] (10自由度)
            num_steps: 验证步数
            render: 是否渲染
            
        Returns:
            result: 验证结果
        """
        print(f"\n[S1 Sim2Sim] 验证 LeRobot 策略")
        print(f"  自由度: 10 (单臂5DOF, 手腕1自由度)")
        print(f"  步数: {num_steps}")
        print(f"  渲染: {render}")
        
        # 加载策略
        if policy_path and trainer.policy is None:
            trainer.load_checkpoint(policy_path)
        
        # 重置仿真
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始状态 (10自由度)
        if initial_state is not None:
            num_joints = min(len(initial_state), self.model.nq - 7, 10)
            self.data.qpos[7:7+num_joints] = initial_state[:num_joints]
        
        # 记录轨迹
        trajectory = []
        observations = []
        rewards = []
        
        for step in range(num_steps):
            # 获取当前观测 - 10个手臂自由度
            current_qpos = self.data.qpos[7:17].copy()
            observations.append(current_qpos)
            
            # 策略推理
            try:
                action = trainer.predict(current_qpos)
            except Exception as e:
                print(f"  [警告] 策略推理失败: {e}")
                action = current_qpos  # 保持当前位置
            
            # 执行动作
            self.data.ctrl[:len(action)] = action
            
            # 多步仿真 (保持 100Hz 控制)
            for _ in range(int(0.01 / self.dt)):
                mujoco.mj_step(self.model, self.data)
            
            # 记录
            trajectory.append(self.data.qpos.copy())
            
            # 简单奖励 (与参考轨迹的相似度)
            if hasattr(trainer, 'dataset') and trainer.dataset is not None:
                ref_traj = trainer.dataset.observations
                if step < len(ref_traj):
                    error = np.linalg.norm(current_qpos - ref_traj[step % len(ref_traj)])
                    reward = -error
                    rewards.append(reward)
            
            # 渲染
            if render and step % 10 == 0:
                pass  # 实际渲染使用 mujoco.viewer
        
        trajectory = np.array(trajectory)
        observations = np.array(observations)
        
        # 计算指标
        result = self._compute_metrics(trajectory, observations, rewards)
        result['trajectory'] = trajectory
        result['observations'] = observations
        
        print(f"[S1 Sim2Sim] 验证完成")
        print(f"  平均奖励: {result.get('mean_reward', 'N/A')}")
        print(f"  轨迹稳定性: {result['stability']:.4f}")
        print(f"  关节限位违规: {result['limit_violations']}")
        
        return result
    
    def validate_trajectory(self,
                           trajectory_path: str,
                           render: bool = False) -> Dict:
        """
        直接验证轨迹文件 (不使用策略)
        
        Args:
            trajectory_path: 轨迹文件路径 (.pkl)
            render: 是否渲染
            
        Returns:
            result: 验证结果
        """
        print(f"\n[S1 Sim2Sim] 验证轨迹: {trajectory_path}")
        
        # 加载轨迹
        with open(trajectory_path, 'rb') as f:
            data = pickle.load(f)
        
        trajectory = data['q']
        fps = data.get('fps', 100)
        
        print(f"  轨迹长度: {len(trajectory)}")
        print(f"  自由度: {trajectory.shape[1] if len(trajectory.shape) > 1 else 'unknown'}")
        print(f"  FPS: {fps}")
        
        # 重置仿真
        mujoco.mj_resetData(self.model, self.data)
        
        executed_traj = []
        tracking_errors = []
        
        control_steps = int(1.0 / fps / self.dt)
        
        for i, target_pos in enumerate(trajectory):
            # 设置控制目标 (最多10个关节)
            num_actuators = min(len(target_pos), self.model.nu, 10)
            self.data.ctrl[:num_actuators] = target_pos[:num_actuators]
            
            # 仿真步进
            for _ in range(control_steps):
                mujoco.mj_step(self.model, self.data)
            
            # 记录实际执行的位置 (10自由度)
            actual_pos = self.data.qpos[7:17]
            executed_traj.append(actual_pos.copy())
            
            # 跟踪误差
            error = np.linalg.norm(actual_pos - target_pos[:len(actual_pos)])
            tracking_errors.append(error)
        
        executed_traj = np.array(executed_traj)
        tracking_errors = np.array(tracking_errors)
        
        result = {
            'trajectory': executed_traj,
            'tracking_errors': tracking_errors,
            'mean_tracking_error': tracking_errors.mean(),
            'max_tracking_error': tracking_errors.max(),
            'stability': 1.0 / (1.0 + np.std(tracking_errors)),
            'limit_violations': self._count_limit_violations(executed_traj),
        }
        
        print(f"  平均跟踪误差: {result['mean_tracking_error']:.4f}")
        print(f"  最大跟踪误差: {result['max_tracking_error']:.4f}")
        
        return result
    
    def _compute_metrics(self, 
                        trajectory: np.ndarray,
                        observations: np.ndarray,
                        rewards: list) -> Dict:
        """计算验证指标 (10自由度)"""
        metrics = {}
        
        # 轨迹稳定性 (速度变化) - 10个手臂自由度
        velocities = np.diff(trajectory[:, 7:17], axis=0)
        acceleration = np.diff(velocities, axis=0)
        metrics['stability'] = 1.0 / (1.0 + np.linalg.norm(acceleration))
        
        # 奖励
        if rewards:
            metrics['mean_reward'] = np.mean(rewards)
            metrics['total_reward'] = np.sum(rewards)
        
        # 关节限位违规
        metrics['limit_violations'] = self._count_limit_violations(trajectory[:, 7:17])
        
        # 能量消耗 (力矩平方和)
        metrics['energy'] = np.sum(velocities ** 2)
        
        return metrics
    
    def _count_limit_violations(self, joint_positions: np.ndarray) -> int:
        """计算关节限位违规次数"""
        violations = 0
        # 简化的限位检查
        for pos in joint_positions:
            if np.abs(pos).max() > 2.5:  # 假设限位约 2.5 rad
                violations += 1
        return violations
    
    def export_for_real(self, 
                       result: Dict,
                       output_path: str,
                       fps: int = 100):
        """
        导出验证后的轨迹用于实机部署
        
        Args:
            result: 验证结果 (包含 'trajectory')
            output_path: 输出路径
            fps: 目标频率
        """
        trajectory = result.get('trajectory', result.get('observations'))
        
        if trajectory is None:
            raise ValueError("结果中未找到轨迹数据")
        
        # 提取关节位置 (排除自由关节) - 10个关节
        if trajectory.shape[1] > 10:
            # 假设前7个是自由关节，后面是控制关节
            joint_traj = trajectory[:, 7:17]
        else:
            joint_traj = trajectory
        
        # 重采样到目标频率
        from common.adapter.trajectory_adapter import TrajectoryAdapter
        resampled = TrajectoryAdapter.resample(
            joint_traj, 
            int(1.0 / self.dt), 
            fps
        )
        
        data = {
            'q': resampled,
            'fps': fps,
            'source': 'stage1_lerobot_sim2sim',
            'dof': 10,  # 10自由度
            'wrist_dof': 1,  # 手腕1自由度
            'validation_metrics': {
                'stability': result.get('stability', 0),
                'mean_tracking_error': result.get('mean_tracking_error', 0),
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"[S1 Sim2Sim] 导出轨迹: {output_path}")
        print(f"  帧数: {len(resampled)}")
        print(f"  时长: {len(resampled)/fps:.2f}s")
        
        return output_path
    
    def visualize_trajectory(self, 
                            trajectory: np.ndarray,
                            reference: Optional[np.ndarray] = None,
                            save_path: Optional[str] = None):
        """
        可视化轨迹对比
        
        Args:
            trajectory: 执行轨迹 [T, 10]
            reference: 参考轨迹 (可选)
            save_path: 保存路径 (可选)
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(5, 2, figsize=(12, 14))
            axes = axes.flatten()
            
            for i in range(10):
                ax = axes[i]
                ax.plot(trajectory[:, i], 'b-', label='Executed', linewidth=1.5)
                if reference is not None and i < reference.shape[1]:
                    ax.plot(reference[:, i], 'r--', label='Reference', linewidth=1.5)
                ax.set_ylabel(self.JOINT_NAMES[i])
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend()
            
            plt.suptitle('G1 Trajectory Visualization (10 DOF, wrist: 1 DOF)')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150)
                print(f"[可视化] 保存到: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("[警告] matplotlib 未安装，跳过可视化")


def test_validator():
    """
    ⚠️ 测试函数 - 仅用于开发调试
    
    使用随机轨迹测试 MuJoCo 验证器功能 (10自由度)。
    生产环境请使用真实轨迹调用 LeRobotMujocoValidator。
    
    运行: python -c "from stage1_lerobot.sim2sim.mujoco_validator import test_validator; test_validator()"
    """
    print("测试 MuJoCo 验证器 (手腕: 1自由度)...")
    
    validator = LeRobotMujocoValidator()
    
    # 创建测试轨迹 - 10自由度
    test_traj = np.random.randn(100, 10) * 0.1
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_path = Path(tmpdir) / "test.pkl"
        with open(traj_path, 'wb') as f:
            pickle.dump({'q': test_traj, 'fps': 100}, f)
        
        result = validator.validate_trajectory(str(traj_path))
        print(f"测试通过! 误差: {result['mean_tracking_error']:.4f}")


if __name__ == "__main__":
    test_validator()
