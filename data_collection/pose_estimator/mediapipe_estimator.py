"""
使用 MediaPipe 进行姿态估计
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Pose3D:
    """3D 姿态数据结构"""
    landmarks: np.ndarray  # [num_joints, 3] (x, y, z)
    visibility: np.ndarray  # [num_joints] 可见性分数
    timestamp: float  # 时间戳
    
    def to_dict(self) -> dict:
        return {
            'landmarks': self.landmarks,
            'visibility': self.visibility,
            'timestamp': self.timestamp,
        }


class MediaPipePoseEstimator:
    """
    MediaPipe 姿态估计器
    
    提取人体 3D 关键点，用于后续重定向到 G1
    """
    
    # MediaPipe 33 个关键点索引
    KEYPOINT_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky', 'right_pinky',
        'left_index', 'right_index',
        'left_thumb', 'right_thumb',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    # 我们关心的关键关节（上半身）
    UPPER_BODY_INDICES = [
        11, 12,  # 肩膀
        13, 14,  # 手肘
        15, 16,  # 手腕
        23, 24,  # 髋部
    ]
    
    def __init__(self,
                 static_image_mode: bool = False,
                 model_complexity: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Args:
            static_image_mode: 是否为静态图片模式
            model_complexity: 模型复杂度 (0, 1, 2)
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.pose = None
        self._init_mediapipe()
    
    def _init_mediapipe(self):
        """初始化 MediaPipe"""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self.mp_drawing = mp.solutions.drawing_utils
            print("[MediaPipe] 初始化成功")
        except ImportError:
            print("[错误] 请先安装: pip install mediapipe")
            raise
    
    def process_frame(self, 
                     frame: np.ndarray,
                     timestamp: float = 0.0) -> Optional[Pose3D]:
        """
        处理单帧图像
        
        Args:
            frame: BGR 图像
            timestamp: 时间戳
            
        Returns:
            pose: 3D 姿态数据
        """
        # 转换颜色空间 (BGR -> RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 处理
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # 提取关键点
        landmarks = []
        visibility = []
        
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
            visibility.append(landmark.visibility)
        
        return Pose3D(
            landmarks=np.array(landmarks),
            visibility=np.array(visibility),
            timestamp=timestamp
        )
    
    def process_video(self,
                     video_path: str,
                     fps: Optional[int] = None) -> List[Pose3D]:
        """
        处理整个视频
        
        Args:
            video_path: 视频路径
            fps: 目标帧率
            
        Returns:
            poses: 姿态序列
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps or original_fps
        
        poses = []
        frame_idx = 0
        
        print(f"[MediaPipe] 处理视频: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / original_fps
            pose = self.process_frame(frame, timestamp)
            
            if pose is not None:
                poses.append(pose)
            
            frame_idx += 1
            
            # 帧率控制
            if fps < original_fps:
                skip = int(original_fps / fps) - 1
                for _ in range(skip):
                    cap.read()
                    frame_idx += 1
        
        cap.release()
        
        print(f"[MediaPipe] 处理完成: {len(poses)} 帧姿态")
        return poses
    
    def visualize(self,
                 frame: np.ndarray,
                 pose: Pose3D,
                 show_upper_body_only: bool = True) -> np.ndarray:
        """
        可视化姿态
        
        Args:
            frame: 原始帧
            pose: 姿态数据
            show_upper_body_only: 只显示上半身
            
        Returns:
            vis_frame: 可视化后的帧
        """
        h, w = frame.shape[:2]
        
        # 绘制关键点
        indices = self.UPPER_BODY_INDICES if show_upper_body_only else range(33)
        
        for idx in indices:
            if pose.visibility[idx] > 0.5:
                x = int(pose.landmarks[idx, 0] * w)
                y = int(pose.landmarks[idx, 1] * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # 绘制骨骼连接
        connections = [
            (11, 13), (13, 15),  # 左臂
            (12, 14), (14, 16),  # 右臂
            (11, 12),            # 肩膀
            (11, 23), (12, 24),  # 躯干
        ]
        
        for start, end in connections:
            if pose.visibility[start] > 0.5 and pose.visibility[end] > 0.5:
                x1 = int(pose.landmarks[start, 0] * w)
                y1 = int(pose.landmarks[start, 1] * h)
                x2 = int(pose.landmarks[end, 0] * w)
                y2 = int(pose.landmarks[end, 1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        return frame
    
    def extract_arm_angles(self, pose: Pose3D) -> Dict[str, float]:
        """
        从姿态提取手臂关节角度
        
        Args:
            pose: 姿态数据
            
        Returns:
            angles: 关节角度字典
        """
        lm = pose.landmarks
        
        def calc_angle(a, b, c):
            """计算三点形成的角度"""
            ba = a - b
            bc = c - b
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            return angle
        
        angles = {}
        
        # 左臂
        if pose.visibility[11] > 0.5 and pose.visibility[13] > 0.5 and pose.visibility[15] > 0.5:
            angles['left_elbow'] = calc_angle(lm[11], lm[13], lm[15])
        
        # 右臂
        if pose.visibility[12] > 0.5 and pose.visibility[14] > 0.5 and pose.visibility[16] > 0.5:
            angles['right_elbow'] = calc_angle(lm[12], lm[14], lm[16])
        
        return angles


if __name__ == "__main__":
    # 测试
    estimator = MediaPipePoseEstimator()
