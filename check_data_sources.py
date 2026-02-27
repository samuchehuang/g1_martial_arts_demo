#!/usr/bin/env python3
"""
数据源检查工具

检查项目所需的所有数据源是否就绪

用法:
    python check_data_sources.py
    python check_data_sources.py --stage 1 --data ./datasets/punch
    python check_data_sources.py --stage 2
"""

import os
import sys
import argparse
from pathlib import Path


def check_file(path: str, description: str, required: bool = True) -> bool:
    """检查文件是否存在"""
    exists = os.path.exists(path)
    status = "✓" if exists else ("✗" if required else "○")
    req_text = "[必需]" if required else "[可选]"
    print(f"  {status} {req_text} {description}: {path}")
    return exists


def check_stage1_requirements(data_path: str = None) -> bool:
    """检查阶段1所需数据源"""
    print("\n" + "="*60)
    print("Stage 1: LeRobot 训练 数据源检查")
    print("="*60)
    
    all_ok = True
    
    # 检查依赖库
    print("\n[依赖库检查]")
    try:
        import torch
        print("  ✓ PyTorch 已安装")
    except ImportError:
        print("  ✗ PyTorch 未安装: pip install torch")
        all_ok = False
    
    try:
        import mediapipe
        print("  ✓ MediaPipe 已安装 (用于视频转换)")
    except ImportError:
        print("  ○ MediaPipe 未安装 (视频转换时需要): pip install mediapipe")
    
    # 检查数据集
    if data_path:
        print(f"\n[数据集检查] {data_path}")
        data_path = Path(data_path)
        
        if data_path.is_file() and data_path.suffix == '.pkl':
            check_file(str(data_path), "PKL 轨迹文件")
        elif data_path.is_dir():
            has_parquet = (data_path / "data.parquet").exists()
            has_npz = (data_path / "data.npz").exists()
            has_pkl = (data_path / "trajectory.pkl").exists()
            
            if has_parquet:
                check_file(str(data_path / "data.parquet"), "Parquet 数据集")
            elif has_npz:
                check_file(str(data_path / "data.npz"), "NumPy 数据集")
            elif has_pkl:
                check_file(str(data_path / "trajectory.pkl"), "PKL 轨迹文件")
            else:
                print(f"  ✗ 未找到支持的数据格式")
                print(f"    期望: data.parquet, data.npz, 或 trajectory.pkl")
                all_ok = False
        else:
            print(f"  ✗ 路径不存在: {data_path}")
            all_ok = False
    else:
        print("\n[数据集检查]")
        print("  ○ 未指定数据集路径，请使用 --data 参数")
        print("    或运行视频转换: python data_collection/video_to_dataset.py")
    
    return all_ok


def check_stage2_requirements() -> bool:
    """检查阶段2所需数据源"""
    print("\n" + "="*60)
    print("Stage 2: Gym 微调 数据源检查")
    print("="*60)
    
    all_ok = True
    
    # 检查阶段1输出
    print("\n[阶段1输出检查]")
    stage1_output = "./outputs/stage1/sim2sim/trajectory.pkl"
    
    if not check_file(stage1_output, "阶段1轨迹输出", required=True):
        print(f"\n  ⚠️  未找到阶段1输出，请先完成阶段1训练:")
        print(f"      python main.py --stage 1 --data <your_data>")
        all_ok = False
    
    # 检查 MuJoCo
    print("\n[MuJoCo 检查]")
    try:
        import mujoco
        print("  ✓ MuJoCo 已安装")
        
        # 检查模型文件
        if os.path.exists("g1.xml"):
            print("  ✓ g1.xml 模型文件存在")
        else:
            print("  ○ g1.xml 不存在，将使用简化模型")
            print("    (如需精确仿真，请联系宇树获取完整模型)")
    except ImportError:
        print("  ✗ MuJoCo 未安装: pip install mujoco")
        all_ok = False
    
    return all_ok


def check_deploy_requirements(trajectory_path: str = None) -> bool:
    """检查部署所需数据源"""
    print("\n" + "="*60)
    print("部署阶段 数据源检查")
    print("="*60)
    
    all_ok = True
    
    # 检查轨迹文件
    print("\n[轨迹文件检查]")
    if trajectory_path:
        if not check_file(trajectory_path, "部署轨迹文件", required=True):
            all_ok = False
    else:
        # 检查默认路径
        stage1_traj = "./outputs/stage1/sim2sim/trajectory.pkl"
        stage2_traj = "./outputs/stage2/sim2sim/final.pkl"
        
        has_s1 = os.path.exists(stage1_traj)
        has_s2 = os.path.exists(stage2_traj)
        
        if has_s2:
            check_file(stage2_traj, "阶段2优化轨迹 (推荐)")
        elif has_s1:
            check_file(stage1_traj, "阶段1轨迹")
            print("  ○ 建议先完成阶段2优化以获得更好效果")
        else:
            print("  ✗ 未找到轨迹文件")
            print(f"    期望: {stage2_traj} 或 {stage1_traj}")
            all_ok = False
    
    # 检查 G1 SDK
    print("\n[G1 SDK 检查]")
    try:
        import unitree_sdk2py
        print("  ✓ Unitree SDK 已安装")
    except ImportError:
        print("  ○ Unitree SDK 未安装 (实机部署时需要)")
        print("    安装: git clone https://github.com/unitreerobotics/unitree_sdk2_python.git")
        print("    或使用模拟模式: python main.py --deploy stage1 --simulation")
    
    return all_ok


def check_video_to_dataset_requirements(video_path: str = None) -> bool:
    """检查视频转换所需数据源"""
    print("\n" + "="*60)
    print("视频转换 数据源检查")
    print("="*60)
    
    all_ok = True
    
    # 检查依赖
    print("\n[依赖库检查]")
    try:
        import cv2
        print("  ✓ OpenCV 已安装")
    except ImportError:
        print("  ✗ OpenCV 未安装: pip install opencv-python")
        all_ok = False
    
    try:
        import mediapipe
        print("  ✓ MediaPipe 已安装")
    except ImportError:
        print("  ✗ MediaPipe 未安装: pip install mediapipe")
        all_ok = False
    
    try:
        import pandas
        print("  ✓ Pandas 已安装 (用于 parquet 格式)")
    except ImportError:
        print("  ○ Pandas 未安装: pip install pandas pyarrow")
    
    # 检查视频文件
    print("\n[视频文件检查]")
    if video_path:
        if check_file(video_path, "输入视频文件", required=True):
            # 检查格式
            ext = Path(video_path).suffix.lower()
            if ext in ['.mp4', '.avi', '.mov']:
                print(f"  ✓ 支持的视频格式: {ext}")
            else:
                print(f"  ⚠️  可能不支持的视频格式: {ext}")
                print(f"      推荐: .mp4, .avi, .mov")
    else:
        print("  ○ 未指定视频文件，请使用 --video 参数")
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="检查项目数据源",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查所有阶段
  python check_data_sources.py
  
  # 检查特定阶段
  python check_data_sources.py --stage 1 --data ./datasets/punch
  python check_data_sources.py --stage 2
  
  # 检查部署
  python check_data_sources.py --deploy --trajectory ./outputs/stage2/sim2sim/final.pkl
  
  # 检查视频转换
  python check_data_sources.py --video ./videos/punch.mp4
        """
    )
    
    parser.add_argument('--stage', 
                       type=int,
                       choices=[1, 2],
                       help='检查特定阶段 (1 或 2)')
    parser.add_argument('--data',
                       help='阶段1数据集路径')
    parser.add_argument('--deploy',
                       action='store_true',
                       help='检查部署阶段')
    parser.add_argument('--trajectory',
                       help='部署用的轨迹路径')
    parser.add_argument('--video',
                       help='视频文件路径 (用于视频转换检查)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("G1 武术动作项目 - 数据源检查")
    print("="*60)
    
    all_ok = True
    
    if args.video or (not args.stage and not args.deploy):
        all_ok &= check_video_to_dataset_requirements(args.video)
    
    if args.stage == 1 or (not args.stage and not args.deploy and not args.video):
        all_ok &= check_stage1_requirements(args.data)
    
    if args.stage == 2 or (not args.stage and not args.deploy and not args.video):
        all_ok &= check_stage2_requirements()
    
    if args.deploy:
        all_ok &= check_deploy_requirements(args.trajectory)
    
    print("\n" + "="*60)
    if all_ok:
        print("✓ 所有必需数据源已就绪")
        print("="*60)
        return 0
    else:
        print("✗ 部分数据源缺失，请根据上述提示补充")
        print("="*60)
        print("\n详细说明请参见: DATA_SOURCES.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
