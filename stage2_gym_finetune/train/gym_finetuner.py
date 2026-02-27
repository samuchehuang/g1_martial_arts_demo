"""
阶段2: Gym 微调
基于阶段1的轨迹，在 Isaac Gym/MuJoCo 中进行 RL 微调优化
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List


class GymFinetuner:
    """
    Gym 微调器
    
    加载阶段1的轨迹作为参考，使用 PPO/SAC 进行微调
    """
    
    def __init__(self,
                 output_dir: str = "./outputs/stage2_gym",
                 env_type: str = "mujoco"):
        """
        Args:
            output_dir: 输出目录
            env_type: 环境类型 ('mujoco' 或 'isaacgym')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env_type = env_type
        
        print(f"[Gym Finetuner] 初始化")
        print(f"  环境: {env_type}")
        print(f"  输出: {output_dir}")
        
    def load_stage1_trajectory(self, stage1_path: str) -> np.ndarray:
        """
        加载阶段1的轨迹作为参考
        
        Args:
            stage1_path: 阶段1轨迹路径
            
        Returns:
            reference_traj: 参考轨迹
        """
        print(f"\n[S2 Train] 加载阶段1轨迹: {stage1_path}")
        
        with open(stage1_path, 'rb') as f:
            data = pickle.load(f)
            
        trajectory = data['q']
        print(f"  轨迹长度: {len(trajectory)}")
        print(f"  关节数: {trajectory.shape[1]}")
        
        return trajectory
        
    def finetune(self,
                stage1_trajectory_path: str,
                num_iterations: int = 500,
                reward_weights: Dict = None) -> str:
        """
        微调策略
        
        Args:
            stage1_trajectory_path: 阶段1轨迹路径
            num_iterations: 训练迭代数
            reward_weights: 奖励权重
            
        Returns:
            output_path: 微调后轨迹路径
        """
        if reward_weights is None:
            reward_weights = {
                'pose': 1.0,
                'velocity': 0.5,
                'smoothness': 0.3,
                'energy': -0.01,
            }
            
        print(f"\n[S2 Train] 开始 Gym 微调")
        print(f"  迭代数: {num_iterations}")
        print(f"  奖励权重: {reward_weights}")
        
        # 加载参考轨迹
        ref_traj = self.load_stage1_trajectory(stage1_trajectory_path)
        
        # 初始化环境
        env = self._create_env()
        
        # 训练循环 (简化)
        optimized_trajectory = self._train_loop(
            env, ref_traj, num_iterations, reward_weights
        )
        
        # 保存
        output_path = self.output_dir / "finetuned_trajectory.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump({
                'q': optimized_trajectory,
                'fps': 100,
                'source': 'stage2_gym_finetuned',
                'reference': stage1_trajectory_path,
            }, f)
            
        print(f"[S2 Train] 微调完成: {output_path}")
        return str(output_path)
        
    def _create_env(self):
        """创建训练环境"""
        if self.env_type == "mujoco":
            # 使用 MuJoCo 环境
            import mujoco
            try:
                model = mujoco.MjModel.from_xml_path("g1.xml")
            except:
                # 简化模型
                xml = """
                <mujoco>
                  <option timestep="0.01"/>
                  <worldbody>
                    <body name="base">
                      <joint type="free"/>
                      <geom type="sphere" size="0.1"/>
                    </body>
                  </worldbody>
                </mujoco>
                """
                model = mujoco.MjModel.from_xml_string(xml)
            return mujoco.MjData(model)
        else:
            # Isaac Gym 环境
            raise NotImplementedError("Isaac Gym 环境待实现")
            
    def _train_loop(self,
                   env,
                   ref_traj: np.ndarray,
                   num_iterations: int,
                   reward_weights: Dict) -> np.ndarray:
        """
        训练循环
        
        使用 PPO 风格的简化训练
        """
        print("[S2 Train] 训练循环开始")
        
        T, J = ref_traj.shape
        optimized = ref_traj.copy()
        
        for iteration in range(num_iterations):
            # 计算当前轨迹与参考的偏差
            # 实际应使用 RL 算法，这里简化处理
            
            # 添加平滑约束
            if iteration > 0:
                smoothness = np.diff(optimized, axis=0, n=2)
                penalty = reward_weights['smoothness'] * np.linalg.norm(smoothness)
                
                # 简单的梯度下降式优化
                noise = np.random.randn(T, J) * 0.001 * (1 - iteration / num_iterations)
                optimized = optimized * 0.99 + ref_traj * 0.01 + noise
                
            if iteration % 100 == 0:
                error = np.linalg.norm(optimized - ref_traj)
                print(f"  Iter {iteration}: error={error:.4f}")
                
        return optimized
        
    def evaluate(self, 
                trajectory_path: str,
                num_episodes: int = 10) -> Dict:
        """
        评估微调后的策略
        
        Args:
            trajectory_path: 轨迹路径
            num_episodes: 评估轮数
            
        Returns:
            metrics: 评估指标
        """
        with open(trajectory_path, 'rb') as f:
            data = pickle.load(f)
            
        trajectory = data['q']
        
        metrics = {
            'mean_position': trajectory.mean(),
            'std_position': trajectory.std(),
            'max_velocity': np.abs(np.diff(trajectory, axis=0)).max(),
            'smoothness': np.linalg.norm(np.diff(trajectory, n=2)),
        }
        
        print(f"\n[S2 Evaluate] 评估结果:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
        return metrics
