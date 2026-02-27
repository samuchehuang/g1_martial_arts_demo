#!/usr/bin/env python3
"""
G1 武术动作演示系统 - 双阶段流程

完整流程:
  Stage 1: LeRobot IL -> Sim-to-Sim -> Sim-to-Real
  Stage 2: Gym Finetune -> Sim-to-Sim -> Sim-to-Real

用法:
  # 完整两阶段流程
  python main.py --stage all --data ./data/punch_raw.pkl

  # 仅阶段1
  python main.py --stage 1 --data ./data/punch_raw.pkl

  # 阶段2（基于阶段1结果）
  python main.py --stage 2 --stage1-output ./outputs/stage1_lerobot/sim2sim/trajectory.pkl

  # 部署阶段1结果
  python main.py --deploy stage1 --trajectory ./outputs/stage1_lerobot/sim2sim/trajectory.pkl

  # 部署阶段2结果
  python main.py --deploy stage2 --trajectory ./outputs/stage2_gym/sim2sim/final.pkl
"""

import argparse
import sys
from pathlib import Path


def run_stage1(raw_data_path: str, visualize: bool = False):
    """
    阶段1: LeRobot 模仿学习
    
    流程: 训练 -> Sim2Sim验证 -> 导出
    """
    print("\n" + "="*60)
    print("STAGE 1: LeRobot Imitation Learning")
    print("="*60)
    
    from stage1_lerobot.train.lerobot_trainer import LeRobotTrainer
    from stage1_lerobot.sim2sim.mujoco_validator import LeRobotMujocoValidator
    
    # 1. 训练
    trainer = LeRobotTrainer(
        output_dir="./outputs/stage1_lerobot",
        policy_type="act"
    )
    
    # 加载数据集
    trainer.load_dataset(raw_data_path)
    
    # 训练
    history = trainer.train(num_epochs=1000)
    
    # 导出策略
    policy_path = trainer.export_policy("./outputs/stage1_lerobot/policy.pt")
    
    # 2. Sim-to-Sim 验证
    validator = LeRobotMujocoValidator()
    
    import numpy as np
    initial_state = np.zeros(10)  # 根据实际关节数调整
    
    result = validator.validate_policy(
        trainer, policy_path, initial_state,
        num_steps=500, render=visualize
    )
    
    # 3. 导出轨迹
    output_path = "./outputs/stage1_lerobot/sim2sim/trajectory.pkl"
    validator.export_for_real(result, output_path, fps=100)
    
    print("\n[Stage 1 完成]")
    print(f"  输出: {output_path}")
    
    return output_path


def run_stage2(stage1_trajectory: str, visualize: bool = False):
    """
    阶段2: Gym 微调
    
    流程: 加载S1轨迹 -> 微调 -> Sim2Sim验证对比 -> 导出
    """
    print("\n" + "="*60)
    print("STAGE 2: Gym Finetuning")
    print("="*60)
    
    from stage2_gym_finetune.train.gym_finetuner import GymFinetuner
    from stage2_gym_finetune.sim2sim.validator import Stage2Validator
    
    # 1. 微调
    finetuner = GymFinetuner(
        output_dir="./outputs/stage2_gym",
        env_type="mujoco"
    )
    
    finetuned_path = finetuner.finetune(
        stage1_trajectory,
        num_iterations=500
    )
    
    # 2. 评估
    metrics = finetuner.evaluate(finetuned_path)
    
    # 3. Sim-to-Sim 验证对比
    validator = Stage2Validator()
    
    validation_result = validator.validate(
        stage1_trajectory,
        finetuned_path,
        render=visualize
    )
    
    # 4. 导出最终版本
    final_path = "./outputs/stage2_gym/sim2sim/final.pkl"
    validator.export_final(finetuned_path, validation_result, final_path)
    
    print("\n[Stage 2 完成]")
    print(f"  输出: {final_path}")
    
    return final_path


def deploy_stage1(trajectory_path: str, live: bool = False):
    """部署阶段1结果"""
    print("\n" + "="*60)
    print("DEPLOY: Stage 1 (LeRobot)")
    print("="*60)
    
    from common.control.g1_controller import G1Controller
    from stage1_lerobot.sim2real.deploy import Stage1Deployer
    from stage1_lerobot.train.lerobot_trainer import LeRobotTrainer
    
    controller = G1Controller(simulation=True)  # 改为 False 实机部署
    controller.enable_sdk()
    
    deployer = Stage1Deployer(controller)
    
    if live:
        # 实时策略部署
        trainer = LeRobotTrainer()
        policy_path = "./outputs/stage1_lerobot/policy.pt"
        deployer.deploy_policy_live(trainer, policy_path, duration=10.0)
    else:
        # 轨迹部署
        deployer.deploy_trajectory(trajectory_path)


def deploy_stage2(trajectory_path: str):
    """部署阶段2结果"""
    print("\n" + "="*60)
    print("DEPLOY: Stage 2 (Gym Finetuned)")
    print("="*60)
    
    from common.control.g1_controller import G1Controller
    from stage2_gym_finetune.sim2real.deploy import Stage2Deployer
    
    controller = G1Controller(simulation=True)
    controller.enable_sdk()
    
    deployer = Stage2Deployer(controller)
    deployer.deploy(trajectory_path, use_optimized_gains=True)


def main():
    parser = argparse.ArgumentParser(
        description="G1 武术动作 - 双阶段流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程（两阶段）
  python main.py --stage all --data ./data/punch.pkl --visualize
  
  # 仅阶段1（LeRobot IL）
  python main.py --stage 1 --data ./data/punch.pkl
  
  # 仅阶段2（Gym微调，需先完成阶段1）
  python main.py --stage 2 --stage1-output ./outputs/stage1/sim2sim/trajectory.pkl
  
  # 部署
  python main.py --deploy stage1 --trajectory ./outputs/stage1/sim2sim/trajectory.pkl
  python main.py --deploy stage2 --trajectory ./outputs/stage2/sim2sim/final.pkl
        """
    )
    
    # 阶段选择
    parser.add_argument('--stage',
                       choices=['1', '2', 'all'],
                       help='执行阶段 (1=LeRobot, 2=Gym微调, all=全部)')
    
    # 输入数据
    parser.add_argument('--data',
                       help='原始数据路径（阶段1用）')
    parser.add_argument('--stage1-output',
                       help='阶段1输出路径（阶段2用）')
    
    # 部署选项
    parser.add_argument('--deploy',
                       choices=['stage1', 'stage2'],
                       help='部署模式')
    parser.add_argument('--trajectory',
                       help='部署用的轨迹路径')
    parser.add_argument('--live',
                       action='store_true',
                       help='实时策略模式（仅stage1）')
    
    # 其他选项
    parser.add_argument('--visualize',
                       action='store_true',
                       help='启用可视化')
    
    args = parser.parse_args()
    
    # 执行逻辑
    if args.deploy:
        # 部署模式
        if args.deploy == 'stage1':
            if not args.trajectory:
                print("错误: 部署 stage1 需要 --trajectory")
                return 1
            deploy_stage1(args.trajectory, args.live)
        else:
            if not args.trajectory:
                print("错误: 部署 stage2 需要 --trajectory")
                return 1
            deploy_stage2(args.trajectory)
            
    elif args.stage:
        # 训练模式
        if args.stage in ['1', 'all']:
            if not args.data:
                print("错误: 阶段1需要 --data")
                return 1
            stage1_output = run_stage1(args.data, args.visualize)
        else:
            stage1_output = args.stage1_output
            
        if args.stage in ['2', 'all']:
            if not stage1_output:
                if not args.stage1_output:
                    print("错误: 阶段2需要 --stage1-output 或先运行阶段1")
                    return 1
                stage1_output = args.stage1_output
            run_stage2(stage1_output, args.visualize)
    else:
        parser.print_help()
        return 1
    
    print("\n" + "="*60)
    print("全部完成")
    print("="*60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
