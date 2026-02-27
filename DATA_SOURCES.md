# 项目数据源说明

本文档说明项目中所有数据来源及其获取方式。

## 1. 输入数据源

### 1.1 视频文件 (用户提供)

**来源**: 用户自己拍摄或收集的武术动作视频

**要求**:
- 格式: MP4, AVI, MOV (OpenCV 支持)
- 分辨率: 建议 640x480 或更高
- 帧率: 30 FPS 或更高
- 内容: 人物在画面中央，光线充足

**示例**:
```bash
python data_collection/video_to_dataset.py --video ./videos/punch.mp4 --output ./datasets/punch
```

### 1.2 现有数据集 (可选)

如果你已有轨迹数据，可以直接使用:

**格式 1: LeRobot Parquet**
```
datasets/my_data/
├── data.parquet      # 包含 observation.state 和 action 列
└── meta.json         # 元数据
```

**格式 2: NumPy NPZ**
```
datasets/my_data/
├── data.npz          # 包含 observation_state, action 数组
└── meta.json
```

**格式 3: PKL 轨迹 (视频转换后的格式)**
```
datasets/my_data/
└── trajectory.pkl    # {'q': [T, 8], 'fps': 30, 'joint_names': [...]}
```

## 2. 模型文件

### 2.1 MuJoCo 模型 (可选)

**文件**: `g1.xml`

**来源**: 
- 宇树官方提供的 MuJoCo 模型
- 或使用代码中内置的简化模型

**说明**: 如果 `g1.xml` 不存在，代码会自动使用简化模型。
如需精确仿真，请从宇树官方获取完整模型。

### 2.2 训练检查点 (自动生成)

**路径**: `outputs/stage1/checkpoint_*.pt`

**包含**:
- 策略网络权重
- 优化器状态
- 标准化统计信息
- 训练配置

## 3. 中间数据流

```
Stage 1: 视频/数据集 → LeRobot 训练 → Sim2Sim 验证 → 轨迹文件
                    ↓
            outputs/stage1/
            ├── checkpoint_final.pt    # 训练好的策略
            ├── normalize_stats.json   # 标准化统计
            └── sim2sim/
                └── trajectory.pkl     # 阶段1输出

Stage 2: 阶段1轨迹 → Gym 微调 → Sim2Sim 验证 → 最终轨迹
                    ↓
            outputs/stage2/
            └── sim2sim/
                └── final.pkl          # 阶段2输出
```

## 4. 测试数据 (仅用于开发)

**注意**: 以下数据仅用于代码测试，不应在生产中使用

| 函数 | 位置 | 数据类型 |
|------|------|----------|
| `test_trainer()` | `lerobot_trainer.py` | `np.random.randn` 生成的随机游走 |
| `test_validator()` | `mujoco_validator.py` | 随机轨迹数据 |

使用方式:
```bash
# 单独测试训练器
python -c "from stage1_lerobot.train.lerobot_trainer import test_trainer; test_trainer()"

# 单独测试验证器
python -c "from stage1_lerobot.sim2sim.mujoco_validator import test_validator; test_validator()"
```

## 5. 运行时生成的数据

### 5.1 标准化统计
训练时自动生成，用于推理时的数据标准化:
```json
{
  "obs_mean": [...],
  "obs_std": [...],
  "action_mean": [...],
  "action_std": [...]
}
```

### 5.2 验证指标
Sim2Sim 验证时生成:
```json
{
  "mean_tracking_error": 0.05,
  "stability": 0.95,
  "limit_violations": 0
}
```

## 6. 数据源检查清单

在运行项目前，请确认:

- [ ] **阶段 1 训练**: 已准备视频文件或数据集
- [ ] **阶段 2 微调**: 已完成阶段 1，且 `outputs/stage1/sim2sim/trajectory.pkl` 存在
- [ ] **实机部署**: 已连接 G1 机器人，或添加 `--simulation` 参数
- [ ] **MuJoCo 模型**: (可选) 如需精确仿真，准备 `g1.xml`

## 7. 常见问题

**Q: 提示找不到 g1.xml**
A: 代码会自动使用简化模型，不影响基本功能。如需完整模型，请联系宇树获取。

**Q: 提示找不到数据集**
A: 确保先运行视频转换: `python video_to_dataset.py --video YOUR_VIDEO.mp4 --output ./datasets/punch`

**Q: 能否使用多个视频**
A: 可以，使用批量处理: `python video_to_dataset.py --batch ./videos --output ./datasets`
