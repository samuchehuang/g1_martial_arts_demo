"""
数据收集模块

从视频创建训练数据集的完整流程:
1. 视频预处理 (VideoProcessor)
2. 姿态估计 (PoseEstimator - MediaPipe/OpenPose)
3. 重定向到 G1 (HumanoidRetargeter)
4. 构建数据集 (DatasetBuilder)
"""
