# 视频到数据集转换模块

从视频文件自动提取人体姿态并转换为 G1 机器人训练数据集的完整 Pipeline。

## 流程概览

```
视频文件 (.mp4)
      │
      ▼
┌─────────────────────┐
│ 1. VideoProcessor   │  加载视频、提取帧、调整分辨率
│    (OpenCV)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 2. PoseEstimator    │  人体姿态估计
│    (MediaPipe)      │  提取 33 个 3D 关键点
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 3. HumanToG1        │  重定向到 G1 关节
│    Retargeter       │  计算关节角度、平滑处理
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 4. DatasetBuilder   │  构建数据集
│    (LeRobot格式)    │  保存为 parquet/npz/pkl
└─────────────────────┘
```

## 安装依赖

```bash
pip install opencv-python mediapipe numpy pandas pyarrow

# 可选 (用于 LeRobot 格式)
pip install lerobot
```

## 快速开始

### 单视频处理

```bash
# ⚠️ 必须在项目根目录运行！
cd g1_martial_arts_demo

# 基础用法
python -m data_collection.video_to_dataset --video punch.mp4 --output ./datasets/punch

# 带可视化
python -m data_collection.video_to_dataset --video punch.mp4 --output ./datasets/punch -v

# 裁剪区域 (只处理视频的一部分)
python -m data_collection.video_to_dataset --video punch.mp4 \
                                           --output ./datasets/punch \
                                           --bbox 100 100 400 400

# 调整帧率和平滑度
python -m data_collection.video_to_dataset --video punch.mp4 \
                                           --output ./datasets/punch \
                                           --fps 30 \
                                           --smooth 5
```

### 批量处理

```bash
python -m data_collection.video_to_dataset --batch ./videos --output ./datasets
```

## 模块说明

### 1. VideoProcessor

```python
from video_processor.video_loader import VideoProcessor

processor = VideoProcessor(target_fps=30, target_resolution=(640, 480))
info = processor.load_video("punch.mp4")

for frame in processor.extract_frames("punch.mp4"):
    # 处理每一帧
    pass
```

### 2. MediaPipePoseEstimator

```python
from pose_estimator.mediapipe_estimator import MediaPipePoseEstimator

estimator = MediaPipePoseEstimator(
    model_complexity=2,  # 0=轻量, 1=完整, 2=重型
    min_detection_confidence=0.5
)

pose = estimator.process_frame(frame, timestamp=0.0)
# pose.landmarks: [33, 3] 3D关键点
# pose.visibility: [33] 可见性分数
```

### 3. HumanToG1Retargeter

```python
from retargeting.human_to_g1 import HumanToG1Retargeter

retargeter = HumanToG1Retargeter()

# 单帧重定向
g1_angles = retargeter.retarget_pose(pose.landmarks, pose.visibility)
# g1_angles.left_shoulder_pitch
# g1_angles.right_elbow
# ...

# 平滑轨迹
smoothed = retargeter.smooth_trajectory(trajectory, window_size=5)
```

### 4. LeRobotDatasetBuilder

```python
from dataset_builder.lerobot_dataset import LeRobotDatasetBuilder, DatasetConfig

config = DatasetConfig(
    dataset_name="g1_punch",
    fps=30,
    num_joints=8
)

builder = LeRobotDatasetBuilder(config, output_dir="./datasets")
builder.add_trajectory(trajectory_array, episode_id=0)
builder.save(format='lerobot')  # 或 'numpy', 'json'
```

## 完整 Pipeline 代码

```python
from data_collection.video_to_dataset import video_to_dataset

dataset_path = video_to_dataset(
    video_path="punch.mp4",
    output_dir="./datasets/punch",
    target_fps=30,
    smooth_window=5,
    visualize=True
)
```

## 输出格式

### LeRobot 格式 (推荐)

```
datasets/punch/
├── data.parquet          # 主数据文件
└── meta.json             # 元数据

数据结构:
- observation.state: [N, 8] 当前关节角度
- action: [N, 8] 下一帧目标角度
- timestamp: [N] 时间戳
```

### NumPy 格式

```
datasets/punch/
├── data.npz              # 压缩的 numpy 数组
│   ├── observation_state
│   ├── action
│   └── timestamp
└── meta.json
```

### 原始轨迹 (用于直接训练)

```python
import pickle

with open("datasets/punch/trajectory.pkl", 'rb') as f:
    data = pickle.load(f)
    
# data['q']: [T, 8] 关节角度
# data['fps']: 30
# data['joint_names']: 关节名称列表
```

## 关键关节映射

| MediaPipe 关键点 | G1 关节 | 说明 |
|------------------|---------|------|
| 11 (left_shoulder) | left_shoulder_pitch/roll/yaw | 左肩 |
| 13 (left_elbow) | left_elbow | 左肘 |
| 15 (left_wrist) | - | 用于计算肘角度 |
| 12 (right_shoulder) | right_shoulder_pitch/roll/yaw | 右肩 |
| 14 (right_elbow) | right_elbow | 右肘 |
| 16 (right_wrist) | - | 用于计算肘角度 |

## 参数调优

### 姿态估计精度

```python
# 高精度 (慢但准)
MediaPipePoseEstimator(model_complexity=2, min_detection_confidence=0.7)

# 快速 (适合实时)
MediaPipePoseEstimator(model_complexity=0, min_detection_confidence=0.5)
```

### 平滑度

```python
# 高平滑 (动作更流畅)
video_to_dataset(..., smooth_window=10)

# 低平滑 (保留更多细节)
video_to_dataset(..., smooth_window=3)
```

## 常见问题

**Q: 检测不到姿态?**
- 确保人物在画面中央
- 调整 `--bbox` 裁剪到人物区域
- 提高 `min_detection_confidence` 阈值

**Q: 关节角度跳动很大?**
- 增加 `--smooth` 参数
- 降低视频帧率 `--fps`

**Q: 重定向后动作不像?**
- 调整 `HumanToG1Retargeter.scale_factor` (默认 0.6)
- 检查关节限位是否正确

## 注意事项

1. **隐私**: MediaPipe 在本地运行，视频不会上传到云端
2. **性能**: model_complexity=2 需要较好的 GPU
3. **格式**: 支持 .mp4, .avi, .mov 等 OpenCV 支持的格式
