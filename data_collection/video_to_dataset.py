#!/usr/bin/env python3
"""
视频到数据集的完整 Pipeline

使用示例:
  # 基础用法
  python video_to_dataset.py --video punch_video.mp4 --output ./datasets/punch
  
  # 指定帧率和可视化
  python video_to_dataset.py --video punch_video.mp4 \
                             --output ./datasets/punch \
                             --fps 30 \
                             --visualize \
                             --smooth 5
  
  # 裁剪区域
  python video_to_dataset.py --video punch_video.mp4 \
                             --bbox 100 100 400 400 \
                             --output ./datasets/punch
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

# 导入各模块
from video_processor.video_loader import VideoProcessor
from pose_estimator.mediapipe_estimator import MediaPipePoseEstimator
from retargeting.human_to_g1 import HumanToG1Retargeter, G1JointAngles
from dataset_builder.lerobot_dataset import LeRobotDatasetBuilder, DatasetConfig


def video_to_dataset(video_path: str,
                    output_dir: str,
                    target_fps: int = 30,
                    bbox: tuple = None,
                    smooth_window: int = 5,
                    visualize: bool = False) -> str:
    """
    视频到数据集的完整流程
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        target_fps: 目标帧率
        bbox: 裁剪区域 (x, y, w, h)
        smooth_window: 平滑窗口大小
        visualize: 是否可视化处理过程
        
    Returns:
        dataset_path: 数据集路径
    """
    print("=" * 60)
    print("视频到数据集转换")
    print("=" * 60)
    
    # ========== Step 1: 视频处理 ==========
    print("\n[Step 1/4] 视频处理...")
    video_processor = VideoProcessor(target_fps=target_fps)
    video_info = video_processor.load_video(video_path)
    
    # ========== Step 2: 姿态估计 ==========
    print("\n[Step 2/4] 姿态估计 (MediaPipe)...")
    pose_estimator = MediaPipePoseEstimator(
        model_complexity=2,
        min_detection_confidence=0.5
    )
    
    g1_trajectory = []
    frame_count = 0
    
    # 创建重定向器
    retargeter = HumanToG1Retargeter()
    
    # 处理每一帧
    for frame in video_processor.extract_frames(video_path):
        # 裁剪 (如果需要)
        if bbox:
            frame = video_processor.crop_region(frame, bbox)
        
        # 姿态估计
        timestamp = frame_count / target_fps
        pose = pose_estimator.process_frame(frame, timestamp)
        
        if pose is None:
            print(f"  [警告] 第 {frame_count} 帧未检测到姿态")
            frame_count += 1
            continue
        
        # 重定向到 G1
        g1_angles = retargeter.retarget_pose(pose.landmarks, pose.visibility)
        
        if g1_angles:
            g1_trajectory.append(g1_angles)
        
        # 可视化
        if visualize:
            vis_frame = pose_estimator.visualize(frame.copy(), pose)
            cv2.imshow("Processing", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  已处理 {frame_count} 帧...")
    
    if visualize:
        cv2.destroyAllWindows()
    
    print(f"[姿态估计] 完成: {len(g1_trajectory)} 帧")
    
    # ========== Step 3: 平滑处理 ==========
    print(f"\n[Step 3/4] 平滑处理 (窗口={smooth_window})...")
    if smooth_window > 1:
        g1_trajectory = retargeter.smooth_trajectory(g1_trajectory, smooth_window)
    
    # 转换为 numpy 数组
    trajectory_array = np.array([g.to_array() for g in g1_trajectory])
    
    # ========== Step 4: 构建数据集 ==========
    print("\n[Step 4/4] 构建数据集...")
    
    config = DatasetConfig(
        dataset_name=Path(output_dir).name,
        fps=target_fps,
        num_joints=8
    )
    
    builder = LeRobotDatasetBuilder(config, output_dir=Path(output_dir).parent)
    
    # 添加轨迹
    builder.add_trajectory(trajectory_array, episode_id=0)
    
    # 保存为多种格式
    print("\n  保存为 LeRobot 格式...")
    builder.save(format='lerobot')
    
    print("  保存为 NumPy 格式...")
    builder.save(format='numpy')
    
    print("  保存原始轨迹...")
    builder.save_raw(trajectory_array, filename="trajectory.pkl")
    
    print("\n" + "=" * 60)
    print("数据集构建完成!")
    print(f"  输出目录: {output_dir}")
    print(f"  总帧数: {len(g1_trajectory)}")
    print(f"  时长: {len(g1_trajectory) / target_fps:.2f}s")
    print("=" * 60)
    
    return output_dir


def batch_process(video_dir: str,
                 output_base_dir: str,
                 target_fps: int = 30):
    """
    批量处理视频目录
    
    Args:
        video_dir: 视频文件目录
        output_base_dir: 输出基础目录
        target_fps: 目标帧率
    """
    video_dir = Path(video_dir)
    output_base_dir = Path(output_base_dir)
    
    video_files = list(video_dir.glob("*.mp4")) + \
                  list(video_dir.glob("*.avi")) + \
                  list(video_dir.glob("*.mov"))
    
    print(f"[批量处理] 发现 {len(video_files)} 个视频")
    
    for video_path in video_files:
        output_name = video_path.stem
        output_dir = output_base_dir / output_name
        
        try:
            video_to_dataset(
                str(video_path),
                str(output_dir),
                target_fps=target_fps
            )
        except Exception as e:
            print(f"[错误] 处理 {video_path.name} 失败: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="从视频创建 G1 训练数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法
  python video_to_dataset.py --video punch.mp4 --output ./datasets/punch
  
  # 带可视化
  python video_to_dataset.py --video punch.mp4 --output ./datasets/punch -v
  
  # 批量处理
  python video_to_dataset.py --batch ./videos --output ./datasets
        """
    )
    
    parser.add_argument('--video',
                       help='输入视频文件路径')
    parser.add_argument('--output',
                       help='输出数据集目录')
    parser.add_argument('--fps',
                       type=int,
                       default=30,
                       help='目标帧率 (默认: 30)')
    parser.add_argument('--bbox',
                       type=int,
                       nargs=4,
                       metavar=('X', 'Y', 'W', 'H'),
                       help='裁剪区域 (x y width height)')
    parser.add_argument('--smooth',
                       type=int,
                       default=5,
                       help='平滑窗口大小 (默认: 5)')
    parser.add_argument('-v', '--visualize',
                       action='store_true',
                       help='启用可视化')
    parser.add_argument('--batch',
                       help='批量处理视频目录')
    
    args = parser.parse_args()
    
    # 批量模式
    if args.batch:
        if not args.output:
            print("错误: 批量模式需要 --output 指定输出目录")
            return 1
        batch_process(args.batch, args.output, args.fps)
        return 0
    
    # 单文件模式
    if not args.video or not args.output:
        parser.print_help()
        return 1
    
    try:
        video_to_dataset(
            args.video,
            args.output,
            target_fps=args.fps,
            bbox=tuple(args.bbox) if args.bbox else None,
            smooth_window=args.smooth,
            visualize=args.visualize
        )
    except KeyboardInterrupt:
        print("\n[中断] 用户取消")
        cv2.destroyAllWindows()
        return 1
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
