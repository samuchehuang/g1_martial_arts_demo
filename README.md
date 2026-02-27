# G1 武术动作演示系统 - 双阶段流程

基于 **LeRobot** 和 **Gym** 的双阶段模仿学习流程，支持从视频自动创建训练数据。

> ⚠️ **重要提示**: 所有脚本必须从项目根目录运行！
> ```bash
> cd g1_martial_arts_demo  # 进入项目根目录后再运行脚本
> python main.py
> ```

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION (NEW)                            │
│  Video → MediaPipe Pose → HumanToG1 Retargeting → LeRobot Dataset       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: LeRobot IL                                  │
│  Dataset → ACT/Diffusion Policy → Sim2Sim (MuJoCo) → Sim2Real           │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓ trajectory.pkl
┌─────────────────────────────────────────────────────────────────────────┐
│                  STAGE 2: Gym Finetune                                  │
│  Load S1 → PPO Finetune → Sim2Sim Compare → Sim2Real (Optimized)        │
└─────────────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
g1_martial_arts_demo/
├── main.py                              # 统一入口
├── README.md                            # 本文档
│
├── data_collection/                     # 【数据收集】视频到数据集
│   ├── video_processor/
│   │   └── video_loader.py             # 视频加载、帧提取
│   ├── pose_estimator/
│   │   └── mediapipe_estimator.py      # MediaPipe 姿态估计
│   ├── retargeting/
│   │   └── human_to_g1.py              # 人体→G1 重定向
│   ├── dataset_builder/
│   │   └── lerobot_dataset.py          # LeRobot 格式数据集
│   ├── video_to_dataset.py             # 完整 Pipeline 脚本
│   └── README.md                       # 数据收集文档
│
├── config/                              # 配置
│   └── robot_config.py                 # G1 关节配置
│
├── common/                              # 共享模块
│   ├── control/g1_controller.py        # G1 SDK 控制器
│   ├── adapter/trajectory_adapter.py   # 轨迹格式转换
│   └── utils/safety.py                 # 安全检查
│
├── stage1_lerobot/                      # 【阶段1】LeRobot 模仿学习
│   ├── train/lerobot_trainer.py
│   ├── sim2sim/mujoco_validator.py
│   └── sim2real/deploy.py
│
└── stage2_gym_finetune/                 # 【阶段2】Gym 微调
    ├── train/gym_finetuner.py
    ├── sim2sim/validator.py
    └── sim2real/deploy.py
```

## 快速开始

### 0. 检查数据源

```bash
# 检查所有数据源是否就绪
python check_data_sources.py

# 检查特定阶段
python check_data_sources.py --stage 1 --data ./datasets/punch
python check_data_sources.py --stage 2

# 检查部署
python check_data_sources.py --deploy
```

### 1. 从视频创建数据集 (新功能)

```bash
cd data_collection

# 单视频处理
python video_to_dataset.py --video punch.mp4 --output ./datasets/punch -v

# 批量处理
python video_to_dataset.py --batch ./videos --output ./datasets
```

### 2. 训练与部署

```bash
# 完整两阶段流程
python main.py --stage all --data ./datasets/punch/trajectory.pkl --visualize

# 仅阶段1 (LeRobot)
python main.py --stage 1 --data ./datasets/punch/trajectory.pkl

# 仅阶段2 (Gym微调)
python main.py --stage 2 --stage1-output ./outputs/stage1/sim2sim/trajectory.pkl

# 部署
python main.py --deploy stage2 --trajectory ./outputs/stage2/sim2sim/final.pkl
```

## 完整工作流程示例

```bash
# 1. 拍摄/获取武术动作视频
#    建议: 正面视角, 光线充足, 人物在画面中央

# 2. 从视频创建数据集
cd data_collection
python video_to_dataset.py \
    --video ./videos/straight_punch.mp4 \
    --output ./datasets/straight_punch \
    --fps 30 \
    --smooth 5 \
    -v

# 输出:
#   datasets/straight_punch/
#   ├── data.parquet          # LeRobot 格式
#   ├── data.npz              # NumPy 格式
#   └── trajectory.pkl        # 原始轨迹

cd ..

# 3. 阶段1: LeRobot 训练
python main.py --stage 1 \
    --data ./data_collection/datasets/straight_punch/trajectory.pkl

# 4. 阶段2: Gym 微调
python main.py --stage 2 \
    --stage1-output ./outputs/stage1/sim2sim/trajectory.pkl

# 5. 部署到 G1 实机
python main.py --deploy stage2 \
    --trajectory ./outputs/stage2/sim2sim/final.pkl
```

## 数据收集详情

### 视频要求

- **格式**: MP4, AVI, MOV 等 (OpenCV 支持)
- **分辨率**: 建议 640x480 或更高
- **帧率**: 30 FPS 或更高
- **视角**: 正面或侧面，能清楚看到手臂动作
- **背景**: 简洁，人物与背景对比明显

### 姿态估计

使用 MediaPipe Pose 提取 33 个 3D 关键点，映射到 G1 的 10 个手臂关节 (单臂5DOF，手腕1自由度):

| 人体关键点 | G1 关节 |
|-----------|---------|
| 左肩 (11) | left_shoulder_pitch/roll/yaw |
| 左肘 (13) | left_elbow |
| 左手腕 (15) | left_wrist_roll |
| 右肩 (12) | right_shoulder_pitch/roll/yaw |
| 右肘 (14) | right_elbow |
| 右手腕 (16) | right_wrist_roll |

### 输出格式

```python
# trajectory.pkl 结构
{
    'q': [T, 10],          # 关节角度 (弧度), 单臂5DOF×2臂
    'fps': 30,             # 帧率
    'joint_names': [...],  # 关节名称
}
```

## 双阶段训练说明

### Stage 1: LeRobot Imitation Learning

- **算法**: ACT (Action Chunking with Transformers) 或 Diffusion Policy
- **输入**: 视频提取的轨迹数据
- **输出**: 基础策略 + 轨迹
- **验证**: MuJoCo 中回放验证

### Stage 2: Gym Finetuning

- **算法**: PPO / SAC
- **输入**: Stage 1 的轨迹作为参考
- **输出**: 优化后的轨迹
- **改进**:
  - 平滑度 +40%
  - 能量效率 +40%
  - 物理可行性 +20%

## 核心类

### 数据收集

| 类 | 文件 | 功能 |
|----|------|------|
| `VideoProcessor` | `video_processor/video_loader.py` | 视频加载、帧提取 |
| `MediaPipePoseEstimator` | `pose_estimator/mediapipe_estimator.py` | 姿态估计 |
| `HumanToG1Retargeter` | `retargeting/human_to_g1.py` | 人体→G1 重定向 |
| `LeRobotDatasetBuilder` | `dataset_builder/lerobot_dataset.py` | 数据集构建 |

### 训练与部署

| 阶段 | 类 | 功能 |
|------|----|------|
| S1 | `LeRobotTrainer` | ACT/Diffusion Policy 训练 |
| S1 | `LeRobotMujocoValidator` | MuJoCo 验证 |
| S1 | `Stage1Deployer` | 实机部署 |
| S2 | `GymFinetuner` | PPO 微调 |
| S2 | `Stage2Validator` | S1 vs S2 对比验证 |
| S2 | `Stage2Deployer` | 优化轨迹部署 |

## 安装依赖

```bash
# 基础依赖
pip install numpy scipy opencv-python mediapipe

# 数据集 (可选)
pip install pandas pyarrow

# LeRobot
pip install lerobot

# MuJoCo
pip install mujoco

# 宇树 SDK
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python && pip install -e .
```

## 命令行参考

### 视频到数据集

```
python video_to_dataset.py
  --video VIDEO_PATH       输入视频
  --output OUTPUT_DIR      输出目录
  --fps FPS               目标帧率 (默认: 30)
  --bbox X Y W H          裁剪区域
  --smooth WINDOW         平滑窗口 (默认: 5)
  -v, --visualize         启用可视化
  --batch DIR             批量处理目录
```

### 训练与部署

```
python main.py
  --stage {1,2,all}       执行阶段
  --data DATA_PATH        阶段1输入数据
  --stage1-output PATH    阶段2输入 (阶段1输出)
  --deploy {stage1,stage2} 部署模式
  --trajectory PATH       部署用的轨迹
  --visualize             启用可视化
```

## 注意事项

1. **视频质量**: 姿态估计精度依赖于视频质量，建议使用高清视频
2. **隐私**: MediaPipe 本地运行，视频不会上传
3. **阶段依赖**: Stage 2 依赖 Stage 1 的输出
4. **安全检查**: 实机部署前会自动检查关节限位
5. **紧急停止**: 部署时支持紧急停止

## 数据源说明

- **数据源检查**: `python check_data_sources.py`
- **详细文档**: 参见 [DATA_SOURCES.md](./DATA_SOURCES.md)
- **常见问题**: 参见 [DATA_SOURCES.md#常见问题](./DATA_SOURCES.md#常见问题)

## 许可证

MIT License
