"""
LeRobot 格式数据集构建器

将视频处理后的轨迹保存为 LeRobot 可用的数据集格式
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class DatasetConfig:
    """数据集配置 - 手腕1自由度"""
    dataset_name: str = "g1_martial_arts"
    robot_type: str = "g1"
    fps: int = 30
    num_joints: int = 10  # G1基础版: 单臂5DOF (手腕1自由度) × 2臂 = 10
    wrist_dof: int = 1    # 手腕自由度: 1 (仅 roll)
    joint_names: List[str] = None
    
    def __post_init__(self):
        if self.joint_names is None:
            # 单臂5DOF: 肩部3 + 肘部1 + 手腕1(roll)
            self.joint_names = [
                'left_shoulder_pitch', 'left_shoulder_roll',
                'left_shoulder_yaw', 'left_elbow', 'left_wrist_roll',
                'right_shoulder_pitch', 'right_shoulder_roll',
                'right_shoulder_yaw', 'right_elbow', 'right_wrist_roll',
            ]


class LeRobotDatasetBuilder:
    """
    LeRobot 数据集构建器
    
    构建标准格式的数据集，可用于:
    - LeRobot 训练 (ACT, Diffusion Policy)
    - 直接加载为 numpy 数组
    """
    
    def __init__(self, config: DatasetConfig, output_dir: str = "./datasets"):
        """
        Args:
            config: 数据集配置
            output_dir: 输出目录
        """
        self.config = config
        self.output_dir = Path(output_dir) / config.dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.frames = []
        self.observations = []
        self.actions = []
        self.timestamps = []
        
    def add_frame(self,
                 observation: np.ndarray,
                 action: np.ndarray,
                 timestamp: float = 0.0,
                 metadata: Optional[dict] = None):
        """
        添加一帧数据
        
        Args:
            observation: 观测值 (当前关节角度) [num_joints]
            action: 动作值 (下一帧目标角度) [num_joints]
            timestamp: 时间戳
            metadata: 元数据
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.timestamps.append(timestamp)
        
        if metadata:
            self.frames.append({
                'timestamp': timestamp,
                'metadata': metadata,
            })
    
    def add_trajectory(self,
                      trajectory: np.ndarray,
                      episode_id: int = 0):
        """
        添加完整轨迹
        
        Args:
            trajectory: [T, num_joints] 关节角度序列
            episode_id:  episode ID
        """
        T = len(trajectory)
        dt = 1.0 / self.config.fps
        
        for t in range(T):
            obs = trajectory[t]
            # 动作是下一帧的目标（用于训练）
            action = trajectory[min(t + 1, T - 1)]
            
            self.add_frame(
                observation=obs,
                action=action,
                timestamp=t * dt,
                metadata={'episode_id': episode_id, 'frame_idx': t}
            )
        
        print(f"[Dataset] 添加轨迹: {T} 帧 (episode {episode_id})")
    
    def build(self) -> dict:
        """
        构建数据集
        
        Returns:
            dataset: 数据集字典
        """
        if len(self.observations) == 0:
            raise ValueError("数据集为空，请先添加数据")
        
        observations = np.array(self.observations)
        actions = np.array(self.actions)
        timestamps = np.array(self.timestamps)
        
        dataset = {
            'observation.state': observations,
            'action': actions,
            'timestamp': timestamps,
            'config': asdict(self.config),
            'num_frames': len(observations),
            'num_episodes': len(set(f['metadata']['episode_id'] for f in self.frames)),
        }
        
        return dataset
    
    def save(self, format: str = 'lerobot'):
        """
        保存数据集
        
        Args:
            format: 格式 ('lerobot', 'numpy', 'json')
        """
        dataset = self.build()
        
        if format == 'lerobot':
            self._save_lerobot_format(dataset)
        elif format == 'numpy':
            self._save_numpy_format(dataset)
        elif format == 'json':
            self._save_json_format(dataset)
        else:
            raise ValueError(f"未知格式: {format}")
    
    def _save_lerobot_format(self, dataset: dict):
        """保存为 LeRobot 格式 (parquet)"""
        try:
            import pandas as pd
            
            # 创建 DataFrame
            df_data = {
                'observation.state': list(dataset['observation.state']),
                'action': list(dataset['action']),
                'timestamp': dataset['timestamp'],
            }
            
            df = pd.DataFrame(df_data)
            
            # 保存 parquet
            parquet_path = self.output_dir / "data.parquet"
            df.to_parquet(parquet_path, index=False)
            
            # 保存元数据
            meta_path = self.output_dir / "meta.json"
            with open(meta_path, 'w') as f:
                json.dump({
                    'config': dataset['config'],
                    'num_frames': dataset['num_frames'],
                    'num_episodes': dataset['num_episodes'],
                }, f, indent=2)
            
            print(f"[Dataset] LeRobot 格式已保存:")
            print(f"  数据: {parquet_path}")
            print(f"  元数据: {meta_path}")
            
        except ImportError:
            print("[警告] pandas 未安装，使用 numpy 格式保存")
            self._save_numpy_format(dataset)
    
    def _save_numpy_format(self, dataset: dict):
        """保存为 numpy 格式 (.npz)"""
        npz_path = self.output_dir / "data.npz"
        
        np.savez(
            npz_path,
            observation_state=dataset['observation.state'],
            action=dataset['action'],
            timestamp=dataset['timestamp'],
        )
        
        # 保存配置
        meta_path = self.output_dir / "meta.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'config': dataset['config'],
                'num_frames': dataset['num_frames'],
                'num_episodes': dataset['num_episodes'],
            }, f, indent=2)
        
        print(f"[Dataset] NumPy 格式已保存: {npz_path}")
    
    def _save_json_format(self, dataset: dict):
        """保存为 JSON 格式（仅用于调试）"""
        json_path = self.output_dir / "data.json"
        
        # 转换为列表（numpy 数组不能直接序列化）
        data = {
            'observation.state': dataset['observation.state'].tolist(),
            'action': dataset['action'].tolist(),
            'timestamp': dataset['timestamp'].tolist(),
            'config': dataset['config'],
            'num_frames': dataset['num_frames'],
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        print(f"[Dataset] JSON 格式已保存: {json_path}")
    
    def save_raw(self, trajectory: np.ndarray, filename: str = "trajectory.pkl"):
        """
        保存原始轨迹（用于 stage1_lerobot）
        
        Args:
            trajectory: [T, num_joints] 轨迹
            filename: 文件名
        """
        pkl_path = self.output_dir / filename
        
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'q': trajectory,
                'fps': self.config.fps,
                'joint_names': self.config.joint_names,
            }, f)
        
        print(f"[Dataset] 原始轨迹已保存: {pkl_path}")
    
    @staticmethod
    def load_dataset(dataset_dir: str) -> dict:
        """
        加载数据集
        
        Args:
            dataset_dir: 数据集目录
            
        Returns:
            dataset: 数据集字典
        """
        dataset_dir = Path(dataset_dir)
        
        # 尝试加载不同格式
        parquet_path = dataset_dir / "data.parquet"
        npz_path = dataset_dir / "data.npz"
        pkl_path = dataset_dir / "trajectory.pkl"
        
        if parquet_path.exists():
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            return {
                'observation.state': np.array(df['observation.state'].tolist()),
                'action': np.array(df['action'].tolist()),
                'timestamp': df['timestamp'].values,
            }
        
        elif npz_path.exists():
            data = np.load(npz_path)
            return {
                'observation.state': data['observation_state'],
                'action': data['action'],
                'timestamp': data['timestamp'],
            }
        
        elif pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            return {
                'q': data['q'],
                'fps': data['fps'],
            }
        
        else:
            raise FileNotFoundError(f"数据集未找到: {dataset_dir}")


if __name__ == "__main__":
    # 测试
    config = DatasetConfig(dataset_name="test")
    builder = LeRobotDatasetBuilder(config)
