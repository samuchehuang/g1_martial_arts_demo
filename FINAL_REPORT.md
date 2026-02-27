# G1 武术动作演示系统 - 最终检查报告

**检查日期**: 2026-02-26  
**项目版本**: 1.0  
**代码文件**: 35 个 Python 文件  
**文档文件**: 4 个 Markdown 文件

---

## 1. 项目结构

```
g1_martial_arts_demo/
├── main.py                          # 统一入口 (已修复调用错误)
├── README.md                        # 主文档
├── DATA_SOURCES.md                  # 数据源说明
├── CODE_REVIEW.md                   # 代码检查报告
├── FINAL_REPORT.md                  # 本文档
├── check_data_sources.py            # 数据源检查工具
│
├── config/                          # 配置模块
│   └── robot_config.py              # G1 关节配置
│
├── common/                          # 共享模块
│   ├── control/g1_controller.py     # G1 SDK 控制器
│   ├── adapter/trajectory_adapter.py # 轨迹转换
│   └── utils/safety.py              # 安全检查
│
├── data_collection/                 # 【视频→数据集】
│   ├── README.md                    # 使用文档
│   ├── video_to_dataset.py          # 完整 Pipeline
│   ├── video_processor/
│   │   └── video_loader.py          # 视频处理
│   ├── pose_estimator/
│   │   └── mediapipe_estimator.py   # MediaPipe 姿态估计
│   ├── retargeting/
│   │   └── human_to_g1.py           # 人体→G1 重定向
│   └── dataset_builder/
│       └── lerobot_dataset.py       # 数据集构建
│
├── stage1_lerobot/                  # 【阶段1】LeRobot 训练
│   ├── train/
│   │   └── lerobot_trainer.py       # ACT/Diffusion 训练 (✅ 已完善)
│   ├── sim2sim/
│   │   └── mujoco_validator.py      # MuJoCo 验证
│   └── sim2real/
│       └── deploy.py                # 实机部署
│
└── stage2_gym_finetune/             # 【阶段2】Gym 微调
    ├── train/
    │   └── gym_finetuner.py         # PPO 微调
    ├── sim2sim/
    │   └── validator.py             # S1 vs S2 对比验证
    └── sim2real/
        └── deploy.py                # 优化轨迹部署
```

---

## 2. 修复记录

### 2.1 main.py 方法调用修复 ✅

**问题**: `main.py` 中的调用与实际方法签名不匹配

| 原代码 | 修复后 |
|--------|--------|
| `trainer.create_dataset(raw_data_path, "g1_punch")` | `trainer.load_dataset(raw_data_path)` |
| `trainer.train(dataset_path, num_epochs=1000)` | `trainer.train(num_epochs=1000)` |
| `trainer.export_policy(checkpoint)` | `trainer.export_policy("./outputs/policy.pt")` |

### 2.2 测试函数标记 ✅

**文件**: `lerobot_trainer.py`, `mujoco_validator.py`

- 添加明确警告: "⚠️ 测试函数 - 仅用于开发调试"
- 使用随机数据，不应用于生产

### 2.3 运行路径说明 ✅

**更新文档**: README.md, data_collection/README.md

- 强调所有脚本必须从项目根目录运行
- 使用 `python -m` 运行子模块
- 提供正确的命令示例

---

## 3. 完整数据流验证

### 阶段1: 视频 → LeRobot 训练 → Sim2Sim

```
输入: punch.mp4 (用户视频)
  ↓
VideoProcessor - 提取帧 @ 30fps
  ↓
MediaPipePoseEstimator - 33个关键点
  ↓
HumanToG1Retargeter - 8个关节角度 + 平滑
  ↓
LeRobotDatasetBuilder - trajectory.pkl
  ↓
LeRobotTrainer.load_dataset() - 加载并标准化
  ↓
LeRobotTrainer.train() - ACT/Diffusion 训练
  ↓
LeRobotMujocoValidator.validate_policy() - MuJoCo仿真
  ↓
输出: trajectory.pkl (阶段1结果)
```

### 阶段2: 微调 → 优化 → Sim2Sim

```
输入: 阶段1 trajectory.pkl
  ↓
GymFinetuner.load_stage1_trajectory()
  ↓
GymFinetuner.finetune() - PPO优化
  ↓
Stage2Validator.validate() - 对比S1 vs S2
  ↓
输出: final.pkl (阶段2优化结果)
```

### 部署: Sim2Real

```
输入: final.pkl
  ↓
G1Controller - 连接机器人 (或模拟)
  ↓
Stage2Deployer.deploy() - 100Hz执行
  ↓
G1实机执行武术动作
```

---

## 4. 使用方法

### 4.1 环境准备

```bash
# 安装依赖
pip install numpy scipy opencv-python mediapipe torch

# 可选
pip install pandas pyarrow mujoco

# 宇树 SDK (实机部署时)
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python && pip install -e .
```

### 4.2 检查数据源

```bash
cd g1_martial_arts_demo
python check_data_sources.py
```

### 4.3 视频转数据集

```bash
python -m data_collection.video_to_dataset \
    --video ./videos/punch.mp4 \
    --output ./datasets/punch \
    --fps 30
```

### 4.4 训练 (阶段1)

```bash
python main.py --stage 1 --data ./datasets/punch/trajectory.pkl
```

### 4.5 微调 (阶段2)

```bash
python main.py --stage 2 --stage1-output ./outputs/stage1/sim2sim/trajectory.pkl
```

### 4.6 部署

```bash
# 模拟模式
python main.py --deploy stage2 --trajectory ./outputs/stage2/sim2sim/final.pkl

# 实机模式 (需修改代码 simulation=False)
```

---

## 5. 关键类和方法

### 数据收集

| 类 | 关键方法 | 输出 |
|----|----------|------|
| `VideoProcessor` | `extract_frames()` | 视频帧 |
| `MediaPipePoseEstimator` | `process_video()` | 3D关键点 |
| `HumanToG1Retargeter` | `retarget_pose()`, `smooth_trajectory()` | G1关节角度 |
| `LeRobotDatasetBuilder` | `add_trajectory()`, `save()` | .parquet/.npz/.pkl |

### 阶段1训练

| 类 | 关键方法 | 输出 |
|----|----------|------|
| `LeRobotTrainer` | `load_dataset()`, `train()`, `export_policy()` | policy.pt |
| `LeRobotMujocoValidator` | `validate_policy()`, `export_for_real()` | trajectory.pkl |

### 阶段2微调

| 类 | 关键方法 | 输出 |
|----|----------|------|
| `GymFinetuner` | `finetune()`, `evaluate()` | finetuned_trajectory.pkl |
| `Stage2Validator` | `validate()`, `export_final()` | final.pkl |

### 部署

| 类 | 关键方法 | 输出 |
|----|----------|------|
| `G1Controller` | `send_positions()`, `emergency_stop()` | 机器人动作 |
| `Stage1/2Deployer` | `deploy_trajectory()` | 执行确认 |

---

## 6. 数据源说明

### 必需 (用户提供)

| 数据 | 格式 | 来源 |
|------|------|------|
| 视频 | .mp4/.avi/.mov | 用户拍摄 |
| 或数据集 | .parquet/.npz/.pkl | 已有数据 |

### 自动生成

| 数据 | 位置 | 生成时机 |
|------|------|----------|
| 标准化统计 | `outputs/stage1/normalize_stats.json` | 训练时 |
| 策略检查点 | `outputs/stage1/checkpoint_*.pt` | 训练时 |
| 阶段1轨迹 | `outputs/stage1/sim2sim/trajectory.pkl` | Sim2Sim后 |
| 阶段2轨迹 | `outputs/stage2/sim2sim/final.pkl` | 微调后 |

### 可选 (建议使用)

| 数据 | 位置 | 说明 |
|------|------|------|
| MuJoCo模型 | `g1.xml` | 宇树官方提供，用于精确仿真 |

---

## 7. 已知限制

### 7.1 技术限制

1. **MediaPipe 精度**: 依赖视频质量和人体可见性
2. **重定向精度**: 人体到 G1 的比例映射是近似值
3. **仿真精度**: 如无 g1.xml，使用简化模型

### 7.2 运行限制

1. **路径依赖**: 必须从项目根目录运行
2. **依赖库**: 需要手动安装 torch, mediapipe 等
3. **硬件**: 实机部署需要 G1 机器人和宇树 SDK

---

## 8. 测试结果

| 测试项 | 状态 | 备注 |
|--------|------|------|
| 语法检查 | ✅ 通过 | 35个Python文件 |
| 方法签名 | ✅ 已修复 | main.py调用已修正 |
| 数据流 | ✅ 正确 | 从视频到实机完整链路 |
| 导入依赖 | ✅ 正确 | 需从根目录运行 |
| 文档完整性 | ✅ 已提供 | README, DATA_SOURCES, CODE_REVIEW |

---

## 9. 结论

**项目状态**: ✅ 可用

**建议**:
1. 始终从项目根目录运行脚本
2. 首次使用前运行 `check_data_sources.py` 检查环境
3. 生产环境请勿使用测试函数 (`test_trainer`, `test_validator`)
4. 如需精确仿真，联系宇树获取 `g1.xml` 模型文件

**项目特点**:
- ✅ 完整的数据流: 视频 → 训练 → 仿真 → 实机
- ✅ 双阶段训练: LeRobot IL + Gym 微调
- ✅ 多格式支持: .parquet, .npz, .pkl
- ✅ 模块化设计: 各阶段可独立运行
- ✅ 完整文档: 使用说明、数据源、代码检查报告
