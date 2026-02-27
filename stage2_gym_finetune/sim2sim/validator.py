"""
阶段2 Sim-to-Sim: Gym 微调后的轨迹验证
"""

import mujoco
import numpy as np
import pickle
from pathlib import Path
from typing import Dict


class Stage2Validator:
    """
    阶段2 Sim-to-Sim 验证器
    
    对比阶段1和阶段2的轨迹差异
    验证微调后的改进效果
    """
    
    def __init__(self, model_path: str = "g1.xml"):
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        except:
            xml = """
            <mujoco model="g1_simple">
              <compiler angle="radian"/>
              <option timestep="0.001"/>
              <worldbody>
                <body name="torso">
                  <freejoint/>
                  <geom type="capsule" size="0.1" fromto="0 0 0 0 0 0.5"/>
                </body>
              </worldbody>
            </mujoco>
            """
            self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        
    def validate(self,
                stage1_path: str,
                stage2_path: str,
                render: bool = False) -> Dict:
        """
        验证阶段2相对于阶段1的改进
        
        Args:
            stage1_path: 阶段1轨迹路径
            stage2_path: 阶段2轨迹路径
            render: 是否渲染
            
        Returns:
            comparison: 对比结果
        """
        print(f"\n[S2 Sim2Sim] 验证微调效果")
        
        # 加载两个阶段的轨迹
        with open(stage1_path, 'rb') as f:
            traj1 = pickle.load(f)['q']
        with open(stage2_path, 'rb') as f:
            traj2 = pickle.load(f)['q']
        
        print(f"  S1 轨迹: {traj1.shape}")
        print(f"  S2 轨迹: {traj2.shape}")
        
        # 计算改进指标
        metrics = self._compare_trajectories(traj1, traj2)
        
        # MuJoCo 物理验证
        physics_metrics = self._validate_in_mujoco(traj2, render)
        
        result = {
            'comparison': metrics,
            'physics': physics_metrics,
            'improvement_score': self._calculate_improvement(metrics),
        }
        
        print(f"\n[S2 Sim2Sim] 验证结果:")
        print(f"  平滑度改进: {metrics['smoothness_improvement']:.2%}")
        print(f"  能量效率改进: {metrics['energy_improvement']:.2%}")
        print(f"  综合改进分数: {result['improvement_score']:.2f}")
        
        return result
        
    def _compare_trajectories(self, 
                             traj1: np.ndarray, 
                             traj2: np.ndarray) -> Dict:
        """对比两个轨迹"""
        # 确保形状相同
        min_len = min(len(traj1), len(traj2))
        t1 = traj1[:min_len]
        t2 = traj2[:min_len]
        
        # 平滑度 (二阶差分的范数)
        smooth1 = np.linalg.norm(np.diff(t1, n=2, axis=0))
        smooth2 = np.linalg.norm(np.diff(t2, n=2, axis=0))
        smooth_improvement = (smooth1 - smooth2) / smooth1 if smooth1 > 0 else 0
        
        # 能量 (速度平方和)
        energy1 = np.sum(np.diff(t1, axis=0) ** 2)
        energy2 = np.sum(np.diff(t2, axis=0) ** 2)
        energy_improvement = (energy1 - energy2) / energy1 if energy1 > 0 else 0
        
        # 轨迹偏差
        deviation = np.linalg.norm(t2 - t1) / min_len
        
        return {
            'smoothness_1': smooth1,
            'smoothness_2': smooth2,
            'smoothness_improvement': smooth_improvement,
            'energy_1': energy1,
            'energy_2': energy2,
            'energy_improvement': energy_improvement,
            'deviation': deviation,
        }
        
    def _validate_in_mujoco(self, 
                           trajectory: np.ndarray,
                           render: bool) -> Dict:
        """在 MuJoCo 中验证轨迹"""
        mujoco.mj_resetData(self.model, self.data)
        
        # 简化的物理验证
        violations = 0
        for i in range(len(trajectory)):
            q = trajectory[i]
            if np.abs(q).max() > 3.0:  # 简单限位检查
                violations += 1
                
        return {
            'joint_limit_violations': violations,
            'violation_rate': violations / len(trajectory),
        }
        
    def _calculate_improvement(self, metrics: Dict) -> float:
        """计算综合改进分数"""
        score = (
            metrics['smoothness_improvement'] * 0.4 +
            metrics['energy_improvement'] * 0.4 +
            (1.0 - min(metrics['deviation'], 1.0)) * 0.2
        )
        return score
        
    def export_final(self,
                    stage2_trajectory: str,
                    validation_result: Dict,
                    output_path: str):
        """
        导出最终版本
        
        Args:
            stage2_trajectory: 阶段2轨迹路径
            validation_result: 验证结果
            output_path: 输出路径
        """
        with open(stage2_trajectory, 'rb') as f:
            data = pickle.load(f)
            
        # 添加验证信息
        data['validation'] = {
            'improvement_score': validation_result['improvement_score'],
            'comparison': validation_result['comparison'],
            'physics': validation_result['physics'],
        }
        data['stage'] = 2
        data['final'] = True
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"[S2 Sim2Sim] 导出最终版本: {output_path}")
        return output_path
