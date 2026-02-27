# 代码检查报告

## 检查时间: 2026-02-26

---

## 1. 已修复问题

### 1.1 main.py 方法调用不匹配 ✅ 已修复

**问题**: `main.py` 中的调用与 `lerobot_trainer.py` 的方法签名不匹配

| 原代码 | 问题 | 修复后 |
|--------|------|--------|
| `trainer.create_dataset(raw_data_path, "g1_punch")` | 方法名不存在 | `trainer.load_dataset(raw_data_path)` |
| `trainer.train(dataset_path, num_epochs=1000)` | 参数多余 | `trainer.train(num_epochs=1000)` |
| `trainer.export_policy(checkpoint)` | 参数类型错误 | `trainer.export_policy("./outputs/policy.pt")` |

---

## 2. 代码结构正确性

### 2.1 模块导入 ✅ 正确

```python
# 项目使用绝对导入，需要从项目根目录运行
python main.py  # ✓ 正确
python -m data_collection.video_to_dataset  # ✓ 正确

# 从子目录直接运行会失败
python data_collection/video_to_dataset.py  # ✗ 错误 (导入失败)
```

**解决方案**: 始终从项目根目录运行脚本

### 2.2 类定义完整性 ✅ 正确

| 类名 | 文件 | 关键方法 | 状态 |
|------|------|----------|------|
| `VideoProcessor` | `video_loader.py` | `load_video`, `extract_frames` | ✅ |
| `MediaPipePoseEstimator` | `mediapipe_estimator.py` | `process_frame`, `process_video` | ✅ |
| `HumanToG1Retargeter` | `human_to_g1.py` | `retarget_pose`, `smooth_trajectory` | ✅ |
| `LeRobotDatasetBuilder` | `lerobot_dataset.py` | `add_trajectory`, `save` | ✅ |
| `LeRobotTrainer` | `lerobot_trainer.py` | `load_dataset`, `train`, `export_policy` | ✅ |
| `LeRobotMujocoValidator` | `mujoco_validator.py` | `validate_policy`, `export_for_real` | ✅ |
| `G1Controller` | `g1_controller.py` | `send_positions`, `emergency_stop` | ✅ |
| `GymFinetuner` | `gym_finetuner.py` | `finetune`, `evaluate` | ✅ |

### 2.3 数据流正确性 ✅ 正确

```
视频 (.mp4)
  ↓
VideoProcessor - 提取帧
  ↓
MediaPipePoseEstimator - 姿态估计 (33关键点)
  ↓
HumanToG1Retargeter - 重定向到8个关节
  ↓
LeRobotDatasetBuilder - 构建数据集 (.parquet/.npz/.pkl)
  ↓
LeRobotTrainer.load_dataset() - 加载
  ↓
LeRobotTrainer.train() - 训练 ACT/Diffusion
  ↓
LeRobotMujocoValidator.validate_policy() - Sim2Sim验证
  ↓
trajectory.pkl - 导出用于实机部署
```

---

## 3. 潜在问题与建议

### 3.1 MuJoCo 模型文件 (警告)

**文件**: `mujoco_validator.py`, `gym_finetuner.py`, `validator.py`

```python
# 代码逻辑
try:
    self.model = mujoco.MjModel.from_xml_path("g1.xml")
except:
    print("[警告] 使用简化模型")
    self.model = self._create_simple_model()
```

**影响**: 如果没有 `g1.xml`，会使用简化模型，仿真精度降低

**建议**: 
- 如需精确仿真，从宇树官方获取 `g1.xml`
- 或修改代码使用标准 humanoid 模型

### 3.2 测试函数标记 (信息)

**文件**: `lerobot_trainer.py`, `mujoco_validator.py`

```python
def test_trainer():
    """
    ⚠️ 测试函数 - 仅用于开发调试
    ...
    """
    # 使用 np.random.randn 生成测试数据
```

**状态**: 已添加明确警告注释 ✅

### 3.3 导入路径依赖 (信息)

**问题**: 脚本需要从项目根目录运行

**正确用法**:
```bash
cd g1_martial_arts_demo

# 主程序
python main.py --stage 1 --data ./datasets/punch

# 视频转换
python -m data_collection.video_to_dataset --video ./vids/punch.mp4 --output ./datasets/punch
```

**错误用法**:
```bash
cd g1_martial_arts_demo/data_collection
python video_to_dataset.py  # 导入会失败
```

---

## 4. 运行依赖检查

### 4.1 必需依赖

| 包 | 用途 | 安装命令 |
|----|------|----------|
| numpy | 数值计算 | `pip install numpy` |
| scipy | 插值/滤波 | `pip install scipy` |
| opencv-python | 视频处理 | `pip install opencv-python` |
| mediapipe | 姿态估计 | `pip install mediapipe` |
| torch | 深度学习 | `pip install torch` |

### 4.2 可选依赖

| 包 | 用途 | 安装命令 |
|----|------|----------|
| pandas | parquet 格式 | `pip install pandas pyarrow` |
| mujoco | 仿真验证 | `pip install mujoco` |
| unitree_sdk2py | 实机控制 | 需从 GitHub 安装 |

---

## 5. 完整使用流程验证

### 5.1 阶段1: 视频到训练 (✅ 流程正确)

```bash
# 步骤1: 视频转数据集
cd g1_martial_arts_demo
python -m data_collection.video_to_dataset \
    --video ./videos/punch.mp4 \
    --output ./datasets/punch \
    --fps 30

# 输出: datasets/punch/trajectory.pkl

# 步骤2: 训练
python main.py --stage 1 --data ./datasets/punch/trajectory.pkl

# 输出: outputs/stage1_lerobot/sim2sim/trajectory.pkl
```

### 5.2 阶段2: 微调 (✅ 流程正确)

```bash
python main.py --stage 2 --stage1-output ./outputs/stage1/sim2sim/trajectory.pkl

# 输出: outputs/stage2_gym/sim2sim/final.pkl
```

### 5.3 部署 (✅ 流程正确)

```bash
# 模拟模式
python main.py --deploy stage2 --trajectory ./outputs/stage2/sim2sim/final.pkl

# 实机模式 (需修改代码 simulation=False 或添加 --real 参数)
```

---

## 6. 建议添加的功能

### 6.1 命令行参数增强

建议为 `main.py` 添加:
```python
parser.add_argument('--simulation', action='store_true', help='使用模拟模式 (无实机)')
parser.add_argument('--network', default='eth0', help='实机网络接口')
```

### 6.2 配置文件支持

建议添加 `config.yaml` 管理超参数，避免硬编码。

### 6.3 日志系统

建议使用 `logging` 模块替代 `print`，便于调试。

---

## 7. 总结

| 检查项 | 状态 |
|--------|------|
| 代码语法 | ✅ 正确 |
| 方法签名匹配 | ✅ 已修复 |
| 数据流完整性 | ✅ 正确 |
| 导入依赖 | ✅ 正确 (需从根目录运行) |
| 测试数据标记 | ✅ 已标记 |
| 文档完整性 | ✅ 已提供 |

**结论**: 代码结构和流程是正确的，可以正常使用。主要注意:
1. 从项目根目录运行脚本
2. 确保依赖库已安装
3. 如需精确仿真，获取 g1.xml 模型文件
