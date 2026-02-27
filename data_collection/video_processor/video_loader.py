"""
视频加载和预处理
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional


class VideoProcessor:
    """
    视频处理器
    
    功能:
    - 加载视频文件
    - 提取帧
    - 调整分辨率/裁剪
    - 同步多视角视频
    """
    
    def __init__(self, 
                 target_fps: Optional[int] = None,
                 target_resolution: Optional[Tuple[int, int]] = None):
        """
        Args:
            target_fps: 目标帧率 (None=保持原始)
            target_resolution: 目标分辨率 (width, height)
        """
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        
    def load_video(self, video_path: str) -> dict:
        """
        加载视频并获取基本信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            info: 视频信息字典
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        info = {
            'path': video_path,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
        }
        
        cap.release()
        
        print(f"[Video] 加载成功: {Path(video_path).name}")
        print(f"  分辨率: {info['width']}x{info['height']}")
        print(f"  帧率: {info['fps']:.2f} FPS")
        print(f"  总帧数: {info['frame_count']}")
        print(f"  时长: {info['duration']:.2f}s")
        
        return info
    
    def extract_frames(self, 
                      video_path: str,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None) -> Iterator[np.ndarray]:
        """
        提取视频帧
        
        Args:
            video_path: 视频路径
            start_time: 开始时间 (秒)
            end_time: 结束时间 (秒)
            
        Yields:
            frame: BGR 格式的 numpy 数组
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 计算起始/结束帧
        start_frame = int(start_time * fps) if start_time else 0
        end_frame = int(end_time * fps) if end_time else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 跳转到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_idx = start_frame
        
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 调整分辨率
            if self.target_resolution:
                frame = cv2.resize(frame, self.target_resolution)
            
            yield frame
            frame_idx += 1
            
            # 帧率控制
            if self.target_fps and self.target_fps < fps:
                skip = int(fps / self.target_fps) - 1
                for _ in range(skip):
                    cap.read()
                    frame_idx += 1
        
        cap.release()
    
    def crop_region(self, 
                   frame: np.ndarray,
                   bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        裁剪感兴趣区域
        
        Args:
            frame: 输入帧
            bbox: (x, y, width, height)
            
        Returns:
            cropped: 裁剪后的帧
        """
        x, y, w, h = bbox
        return frame[y:y+h, x:x+w]
    
    def save_frames(self,
                   video_path: str,
                   output_dir: str,
                   prefix: str = "frame"):
        """
        将视频帧保存为图片
        
        Args:
            video_path: 视频路径
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, frame in enumerate(self.extract_frames(video_path)):
            filename = f"{prefix}_{idx:06d}.jpg"
            cv2.imwrite(str(output_dir / filename), frame)
        
        print(f"[Video] 帧已保存到: {output_dir}")
    
    @staticmethod
    def sync_videos(video_paths: list,
                   offsets: list = None) -> list:
        """
        同步多视角视频
        
        Args:
            video_paths: 视频路径列表
            offsets: 时间偏移列表 (秒)
           
        Returns:
            synced_info: 同步后的信息
        """
        if offsets is None:
            offsets = [0] * len(video_paths)
        
        infos = []
        for path, offset in zip(video_paths, offsets):
            vp = VideoProcessor()
            info = vp.load_video(path)
            info['offset'] = offset
            infos.append(info)
        
        # 计算同步后的有效时间段
        max_start = max(info['offset'] for info in infos)
        min_end = min(info['offset'] + info['duration'] for info in infos)
        
        print(f"\n[Sync] 多视角同步结果:")
        print(f"  有效时间段: {max_start:.2f}s - {min_end:.2f}s")
        print(f"  有效时长: {min_end - max_start:.2f}s")
        
        return infos


if __name__ == "__main__":
    # ⚠️ 示例代码 - 需要替换为实际视频路径
    import sys
    if len(sys.argv) < 2:
        print("用法: python video_loader.py <视频文件路径>")
        print("示例: python video_loader.py ./videos/punch.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    processor = VideoProcessor(target_fps=30, target_resolution=(640, 480))
    info = processor.load_video(video_path)
