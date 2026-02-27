"""
阶段1: LeRobot 模仿学习训练
支持 ACT (Action Chunking with Transformers) 和 Diffusion Policy

使用示例:
    # 从视频创建的数据集训练
    trainer = LeRobotTrainer(policy_type="act", chunk_size=100)
    trainer.load_dataset("./datasets/punch")
    trainer.train(num_epochs=1000, batch_size=64)
    trainer.export_policy("./outputs/policy.pt")
"""

import os
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """训练配置"""
    num_epochs: int = 1000
    batch_size: int = 64
    learning_rate: float = 1e-5
    weight_decay: float = 1e-6
    grad_clip_norm: float = 10.0
    save_freq: int = 100  # 每多少 epoch 保存
    eval_freq: int = 50   # 每多少 epoch 评估
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # ACT 专用
    chunk_size: int = 100          # 预测未来多少步
    num_queries: int = 100         # Transformer queries
    dim_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    
    # Diffusion Policy 专用
    num_inference_steps: int = 10
    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"


class LeRobotDatasetAdapter:
    """
    LeRobot 数据集适配器
    兼容多种数据格式: parquet, npz, pkl
    """
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.data = None
        self.meta = None
        self._load()
    
    def _load(self):
        """加载数据集"""
        # 尝试多种格式
        if (self.dataset_path / "data.parquet").exists():
            self._load_parquet()
        elif (self.dataset_path / "data.npz").exists():
            self._load_npz()
        elif (self.dataset_path / "trajectory.pkl").exists():
            self._load_pkl()
        elif self.dataset_path.suffix == ".pkl":
            self._load_single_pkl()
        else:
            raise FileNotFoundError(f"未找到支持的数据格式: {self.dataset_path}")
    
    def _load_parquet(self):
        """加载 Parquet 格式 (LeRobot 标准)"""
        try:
            import pandas as pd
            df = pd.read_parquet(self.dataset_path / "data.parquet")
            
            # 解析 observation.state 列 (可能是字符串存储的列表)
            if df['observation.state'].dtype == object:
                self.observations = np.array(df['observation.state'].tolist())
                self.actions = np.array(df['action'].tolist())
            else:
                self.observations = df['observation.state'].values
                self.actions = df['action'].values
            
            self.timestamps = df.get('timestamp', np.arange(len(df))).values
            
            # 加载元数据
            meta_path = self.dataset_path / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    self.meta = json.load(f)
            
            print(f"[Dataset] 加载 Parquet: {len(df)} 帧")
            
        except ImportError:
            raise ImportError("请安装 pandas 和 pyarrow: pip install pandas pyarrow")
    
    def _load_npz(self):
        """加载 NumPy 格式"""
        data = np.load(self.dataset_path / "data.npz")
        self.observations = data['observation_state']
        self.actions = data['action']
        self.timestamps = data.get('timestamp', np.arange(len(self.observations)))
        
        print(f"[Dataset] 加载 NPZ: {len(self.observations)} 帧")
    
    def _load_pkl(self):
        """加载 trajectory.pkl"""
        with open(self.dataset_path / "trajectory.pkl", 'rb') as f:
            data = pickle.load(f)
        
        # trajectory.pkl 格式: {'q': [T, 8], 'fps': 30}
        trajectory = data['q']
        
        # 构建 observation-action 对
        # observation: 当前状态, action: 下一状态 (用于预测)
        self.observations = trajectory[:-1]  # 除了最后一帧
        self.actions = trajectory[1:]        # 除了第一帧
        self.timestamps = np.arange(len(self.observations)) / data.get('fps', 30)
        
        self.meta = {
            'fps': data.get('fps', 30),
            'joint_names': data.get('joint_names', []),
        }
        
        print(f"[Dataset] 加载 PKL: {len(self.observations)} 帧")
    
    def _load_single_pkl(self):
        """直接加载单个 pkl 文件"""
        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        trajectory = data['q']
        self.observations = trajectory[:-1]
        self.actions = trajectory[1:]
        self.timestamps = np.arange(len(self.observations)) / data.get('fps', 30)
        
        self.meta = {
            'fps': data.get('fps', 30),
            'joint_names': data.get('joint_names', []),
        }
        
        print(f"[Dataset] 加载 PKL: {len(self.observations)} 帧")
    
    def get_stats(self) -> Dict:
        """获取数据统计信息"""
        return {
            'num_frames': len(self.observations),
            'obs_mean': self.observations.mean(axis=0).tolist(),
            'obs_std': self.observations.std(axis=0).tolist(),
            'action_mean': self.actions.mean(axis=0).tolist(),
            'action_std': self.actions.std(axis=0).tolist(),
            'obs_min': self.observations.min(axis=0).tolist(),
            'obs_max': self.observations.max(axis=0).tolist(),
        }
    
    def normalize(self, normalize_stats: Optional[Dict] = None) -> Dict:
        """标准化数据"""
        if normalize_stats is None:
            # 计算统计信息
            self.obs_mean = self.observations.mean(axis=0)
            self.obs_std = self.observations.std(axis=0) + 1e-8
            self.action_mean = self.actions.mean(axis=0)
            self.action_std = self.actions.std(axis=0) + 1e-8
        else:
            # 使用提供的统计信息
            self.obs_mean = np.array(normalize_stats['obs_mean'])
            self.obs_std = np.array(normalize_stats['obs_std'])
            self.action_mean = np.array(normalize_stats['action_mean'])
            self.action_std = np.array(normalize_stats['action_std'])
        
        # 标准化
        self.observations = (self.observations - self.obs_mean) / self.obs_std
        self.actions = (self.actions - self.action_mean) / self.action_std
        
        return {
            'obs_mean': self.obs_mean.tolist(),
            'obs_std': self.obs_std.tolist(),
            'action_mean': self.action_mean.tolist(),
            'action_std': self.action_std.tolist(),
        }
    
    def create_dataloader(self, batch_size: int, shuffle: bool = True):
        """创建 PyTorch DataLoader"""
        from torch.utils.data import Dataset, DataLoader
        
        class TrajectoryDataset(Dataset):
            def __init__(self, observations, actions):
                self.observations = torch.FloatTensor(observations)
                self.actions = torch.FloatTensor(actions)
            
            def __len__(self):
                return len(self.observations)
            
            def __getitem__(self, idx):
                return {
                    'observation.state': self.observations[idx],
                    'action': self.actions[idx],
                }
        
        dataset = TrajectoryDataset(self.observations, self.actions)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class ACTPolicy(torch.nn.Module):
    """
    ACT (Action Chunking with Transformers) 策略
    简化的实现，完整版应使用 LeRobot 库
    """
    
    def __init__(self, obs_dim: int, action_dim: int, config: TrainingConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.chunk_size = config.chunk_size
        
        # 简化的 Transformer 架构
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=config.dim_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_model * 4,
                batch_first=True
            ),
            num_layers=config.n_encoder_layers
        )
        
        self.decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=config.dim_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_model * 4,
                batch_first=True
            ),
            num_layers=config.n_decoder_layers
        )
        
        # 输入/输出投影
        self.obs_proj = torch.nn.Linear(obs_dim, config.dim_model)
        self.action_proj = torch.nn.Linear(action_dim, config.dim_model)
        self.output_proj = torch.nn.Linear(config.dim_model, action_dim)
        
        # 可学习的 queries
        self.query_embed = torch.nn.Embedding(config.num_queries, config.dim_model)
    
    def forward(self, obs, actions=None):
        batch_size = obs.shape[0]
        
        # 编码观测
        obs_tokens = self.obs_proj(obs).unsqueeze(1)  # [B, 1, D]
        memory = self.encoder(obs_tokens)  # [B, 1, D]
        
        # 解码动作
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        if actions is not None and self.training:
            # 训练时: Teacher forcing
            action_tokens = self.action_proj(actions)
            decoder_output = self.decoder(action_tokens, memory)
        else:
            # 推理时: 自回归生成
            decoder_output = self.decoder(queries, memory)
        
        actions_pred = self.output_proj(decoder_output)
        return actions_pred


class DiffusionPolicy(torch.nn.Module):
    """
    Diffusion Policy 策略 (简化版)
    """
    
    def __init__(self, obs_dim: int, action_dim: int, config: TrainingConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # 简化的 U-Net 风格网络
        self.obs_encoder = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        )
        
        self.noise_pred_net = torch.nn.Sequential(
            torch.nn.Linear(action_dim + 128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim)
        )
    
    def forward(self, noisy_actions, obs, timesteps=None):
        """预测噪声"""
        obs_feat = self.obs_encoder(obs)
        input_feat = torch.cat([noisy_actions, obs_feat], dim=-1)
        noise_pred = self.noise_pred_net(input_feat)
        return noise_pred


class LeRobotTrainer:
    """
    LeRobot 训练器
    支持 ACT 和 Diffusion Policy
    """
    
    def __init__(self,
                 policy_type: str = "act",
                 output_dir: str = "./outputs/stage1_lerobot",
                 config: Optional[TrainingConfig] = None):
        """
        Args:
            policy_type: "act" 或 "diffusion"
            output_dir: 输出目录
            config: 训练配置
        """
        self.policy_type = policy_type.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or TrainingConfig()
        
        # 设置随机种子
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        # 初始化
        self.dataset = None
        self.policy = None
        self.optimizer = None
        self.normalize_stats = None
        
        print(f"[LeRobot Trainer] 初始化")
        print(f"  策略: {policy_type}")
        print(f"  设备: {self.config.device}")
        print(f"  输出: {output_dir}")
    
    def load_dataset(self, dataset_path: str) -> Dict:
        """
        加载数据集
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            stats: 数据统计信息
        """
        print(f"\n[Dataset] 加载: {dataset_path}")
        
        self.dataset = LeRobotDatasetAdapter(dataset_path)
        
        # 标准化
        self.normalize_stats = self.dataset.normalize()
        
        # 获取统计信息
        stats = self.dataset.get_stats()
        stats['normalize'] = self.normalize_stats
        
        print(f"  观测维度: {self.dataset.observations.shape[1]}")
        print(f"  动作维度: {self.dataset.actions.shape[1]}")
        print(f"  数据帧数: {stats['num_frames']}")
        
        # 保存统计信息
        stats_path = self.output_dir / "normalize_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def create_policy(self) -> torch.nn.Module:
        """创建策略网络"""
        if self.dataset is None:
            raise ValueError("请先调用 load_dataset()")
        
        obs_dim = self.dataset.observations.shape[1]
        action_dim = self.dataset.actions.shape[1]
        
        print(f"\n[Policy] 创建 {self.policy_type.upper()} 策略")
        
        if self.policy_type == "act":
            self.policy = ACTPolicy(obs_dim, action_dim, self.config)
        elif self.policy_type == "diffusion":
            self.policy = DiffusionPolicy(obs_dim, action_dim, self.config)
        else:
            raise ValueError(f"未知策略类型: {self.policy_type}")
        
        self.policy.to(self.config.device)
        
        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.num_epochs
        )
        
        print(f"  参数量: {sum(p.numel() for p in self.policy.parameters()):,}")
        
        return self.policy
    
    def train(self, num_epochs: Optional[int] = None) -> Dict:
        """
        训练策略
        
        Args:
            num_epochs: 训练轮数 (默认使用 config 中的值)
            
        Returns:
            history: 训练历史
        """
        if self.policy is None:
            self.create_policy()
        
        num_epochs = num_epochs or self.config.num_epochs
        
        # 创建 DataLoader
        dataloader = self.dataset.create_dataloader(
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        print(f"\n[Train] 开始训练")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch: {self.config.batch_size}")
        print(f"  Steps per epoch: {len(dataloader)}")
        
        history = {'loss': [], 'epoch': []}
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                obs = batch['observation.state'].to(self.config.device)
                action = batch['action'].to(self.config.device)
                
                loss = self._train_step(obs, action)
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            history['loss'].append(avg_loss)
            history['epoch'].append(epoch)
            
            self.scheduler.step()
            
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, LR: {lr:.2e}")
            
            # 保存检查点
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(epoch + 1)
        
        # 保存最终模型
        self.save_checkpoint("final")
        
        print(f"[Train] 训练完成")
        return history
    
    def _train_step(self, obs: torch.Tensor, action: torch.Tensor) -> float:
        """单步训练"""
        self.policy.train()
        self.optimizer.zero_grad()
        
        if self.policy_type == "act":
            # ACT: 预测动作序列
            # 扩展 action 为 chunk_size
            batch_size = action.shape[0]
            action_seq = action.unsqueeze(1).expand(-1, self.config.chunk_size, -1)
            
            pred_actions = self.policy(obs, action_seq)
            loss = torch.nn.functional.mse_loss(pred_actions, action_seq)
            
        elif self.policy_type == "diffusion":
            # Diffusion: 去噪训练
            batch_size = action.shape[0]
            
            # 添加噪声
            noise = torch.randn_like(action)
            timesteps = torch.randint(0, self.config.num_train_timesteps, (batch_size,))
            noisy_action = action + noise * 0.1  # 简化的噪声调度
            
            # 预测噪声
            pred_noise = self.policy(noisy_action, obs, timesteps)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.config.grad_clip_norm
        )
        
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        使用策略预测动作
        
        Args:
            observation: 当前观测 [obs_dim]
            
        Returns:
            action: 预测动作 [action_dim] 或 [chunk_size, action_dim]
        """
        if self.policy is None:
            raise ValueError("请先训练或加载策略")
        
        self.policy.eval()
        
        # 标准化
        obs_norm = (observation - self.normalize_stats['obs_mean']) / self.normalize_stats['obs_std']
        obs_tensor = torch.FloatTensor(obs_norm).unsqueeze(0).to(self.config.device)
        
        if self.policy_type == "act":
            pred_action = self.policy(obs_tensor)
            # 取第一个动作
            action = pred_action[0, 0].cpu().numpy()
        elif self.policy_type == "diffusion":
            # 简化的 DDPM 采样
            action = torch.randn(1, self.policy.action_dim).to(self.config.device)
            for _ in range(self.config.num_inference_steps):
                noise_pred = self.policy(action, obs_tensor)
                action = action - noise_pred * 0.1
            action = action[0].cpu().numpy()
        
        # 反标准化
        action = action * self.normalize_stats['action_std'] + self.normalize_stats['action_mean']
        
        return action
    
    def generate_trajectory(self, 
                           initial_obs: np.ndarray,
                           num_steps: int = 500) -> np.ndarray:
        """
        生成完整轨迹
        
        Args:
            initial_obs: 初始观测
            num_steps: 步数
            
        Returns:
            trajectory: [num_steps, action_dim]
        """
        trajectory = []
        obs = initial_obs.copy()
        
        for _ in range(num_steps):
            action = self.predict(obs)
            trajectory.append(action)
            
            # 简单的状态转移 (假设动作为下一状态)
            obs = action
        
        return np.array(trajectory)
    
    def save_checkpoint(self, epoch: Union[int, str]):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'policy_type': self.policy_type,
            'policy_state': self.policy.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'normalize_stats': self.normalize_stats,
            'config': {k: v for k, v in self.config.__dict__.items()},
        }
        
        if isinstance(epoch, int):
            path = self.output_dir / f"checkpoint_{epoch:04d}.pt"
        else:
            path = self.output_dir / f"checkpoint_{epoch}.pt"
        
        torch.save(checkpoint, path)
        print(f"  保存检查点: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        print(f"[Load] 加载检查点: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.policy_type = checkpoint['policy_type']
        self.normalize_stats = checkpoint['normalize_stats']
        
        # 恢复配置
        for k, v in checkpoint['config'].items():
            setattr(self.config, k, v)
        
        # 创建并加载策略
        self.create_policy()
        self.policy.load_state_dict(checkpoint['policy_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        print(f"  已加载 epoch {checkpoint['epoch']}")
    
    def export_policy(self, output_path: Optional[str] = None) -> str:
        """
        导出策略为推理格式
        
        Args:
            output_path: 输出路径 (默认: output_dir/policy.pt)
            
        Returns:
            path: 导出路径
        """
        if output_path is None:
            output_path = self.output_dir / "policy.pt"
        else:
            output_path = Path(output_path)
        
        export_dict = {
            'policy_type': self.policy_type,
            'policy_state': self.policy.state_dict(),
            'normalize_stats': self.normalize_stats,
            'config': {k: v for k, v in self.config.__dict__.items()},
        }
        
        torch.save(export_dict, output_path)
        
        # 同时保存为 ONNX (可选)
        try:
            onnx_path = output_path.with_suffix('.onnx')
            self._export_onnx(onnx_path)
            print(f"[Export] ONNX 导出: {onnx_path}")
        except Exception as e:
            print(f"[Export] ONNX 导出失败: {e}")
        
        print(f"[Export] 策略已导出: {output_path}")
        return str(output_path)
    
    def _export_onnx(self, onnx_path: str):
        """导出 ONNX 格式"""
        dummy_input = torch.randn(1, self.policy.obs_dim).to(self.config.device)
        torch.onnx.export(
            self.policy,
            dummy_input,
            onnx_path,
            input_names=['observation'],
            output_names=['action'],
            dynamic_axes={
                'observation': {0: 'batch_size'},
                'action': {0: 'batch_size'}
            }
        )


def test_trainer():
    """
    ⚠️ 测试函数 - 仅用于开发调试
    
    使用随机生成的数据测试训练器功能。
    生产环境请使用真实数据集调用 LeRobotTrainer。
    
    运行: python -c "from stage1_lerobot.train.lerobot_trainer import test_trainer; test_trainer()"
    """
    # 创建模拟数据 - G1基础版10个手臂自由度
    print("创建测试数据...")
    test_data = {
        'q': np.random.randn(1000, 10).cumsum(axis=0) * 0.1,  # 随机游走
        'fps': 30,
        'joint_names': [
            'left_shoulder_pitch', 'left_shoulder_roll',
            'left_shoulder_yaw', 'left_elbow', 'left_wrist_roll',
            'right_shoulder_pitch', 'right_shoulder_roll',
            'right_shoulder_yaw', 'right_elbow', 'right_wrist_roll'
        ]
    }
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path.mkdir()
        
        with open(dataset_path / "trajectory.pkl", 'wb') as f:
            pickle.dump(test_data, f)
        
        # 测试 ACT
        print("\n测试 ACT 策略...")
        trainer = LeRobotTrainer(
            policy_type="act",
            output_dir=Path(tmpdir) / "act_output",
            config=TrainingConfig(num_epochs=10, batch_size=32)
        )
        
        stats = trainer.load_dataset(str(dataset_path))
        trainer.train()
        
        # 测试预测
        obs = np.zeros(10)  # G1基础版10个手臂自由度
        action = trainer.predict(obs)
        print(f"预测动作: {action}")
        
        print("\n测试通过!")


if __name__ == "__main__":
    test_trainer()
