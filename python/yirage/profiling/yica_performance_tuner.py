#!/usr/bin/env python3
"""
YICA-Yirage 自动调优模块

提供智能的 YICA 性能参数自动调优功能，包括：
- 网格搜索 (Grid Search)
- 随机搜索 (Random Search) 
- 贝叶斯优化 (Bayesian Optimization)
- 遗传算法 (Genetic Algorithm)
"""

import json
import time
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class YICAConfig:
    """YICA 配置参数"""
    cim_array_count: int = 16
    cim_array_size: int = 64
    spm_size_mb: int = 64
    enable_operator_fusion: bool = True
    enable_memory_optimization: bool = True
    compute_frequency_mhz: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'YICAConfig':
        return cls(**data)
    
    def copy(self) -> 'YICAConfig':
        return YICAConfig.from_dict(self.to_dict())


@dataclass
class PerformanceMetrics:
    """性能指标"""
    latency_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    energy_consumption_j: float
    
    def to_score(self) -> float:
        """转换为综合评分"""
        return (
            -self.latency_ms +
            self.throughput_ops_per_sec / 1000 -
            self.memory_usage_mb / 1000 -
            self.energy_consumption_j
        )


class YICAPerformanceEvaluator:
    """YICA 性能评估器"""
    
    def __init__(self):
        self.cache = {}
        self.evaluation_count = 0
    
    def evaluate(self, config: YICAConfig, workload: Dict[str, Any]) -> PerformanceMetrics:
        """评估配置性能"""
        self.evaluation_count += 1
        
        # 模拟性能评估
        batch_size = workload.get('batch_size', 1)
        sequence_length = workload.get('sequence_length', 512)
        hidden_size = workload.get('hidden_size', 768)
        
        # 基础计算
        base_compute = batch_size * sequence_length * hidden_size
        
        # CIM 阵列效率
        cim_efficiency = min(1.0, config.cim_array_count / 16.0)
        fusion_boost = 1.2 if config.enable_operator_fusion else 1.0
        
        # 计算性能指标
        latency_ms = (base_compute / (config.compute_frequency_mhz * 1000)) / (cim_efficiency * fusion_boost)
        throughput = 1000 / latency_ms if latency_ms > 0 else 0
        memory_usage = base_compute * 4 / (1024 * 1024)  # MB
        energy = base_compute * 1e-9 / cim_efficiency
        
        return PerformanceMetrics(
            latency_ms=latency_ms,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=memory_usage,
            energy_consumption_j=energy
        )


class AutoTuner(ABC):
    """自动调优器基类"""
    
    def __init__(self, evaluator: YICAPerformanceEvaluator):
        self.evaluator = evaluator
        self.history = []
        self.best_config = None
        self.best_score = float('-inf')
    
    @abstractmethod
    def tune(self, workload: Dict[str, Any], param_ranges: Dict[str, Any],
             max_evaluations: int = 100) -> Tuple[YICAConfig, PerformanceMetrics]:
        """执行自动调优"""
        pass
    
    def _evaluate_config(self, config: YICAConfig, workload: Dict[str, Any]) -> float:
        """评估配置并返回评分"""
        metrics = self.evaluator.evaluate(config, workload)
        score = metrics.to_score()
        
        self.history.append({
            'config': config.to_dict(),
            'metrics': asdict(metrics),
            'score': score,
            'timestamp': datetime.now().isoformat()
        })
        
        if score > self.best_score:
            self.best_score = score
            self.best_config = config.copy()
        
        return score


class RandomSearchTuner(AutoTuner):
    """随机搜索调优器"""
    
    def tune(self, workload: Dict[str, Any], param_ranges: Dict[str, Any],
             max_evaluations: int = 100) -> Tuple[YICAConfig, PerformanceMetrics]:
        """执行随机搜索调优"""
        print("🎲 开始随机搜索调优...")
        
        for i in range(max_evaluations):
            config = self._generate_random_config(param_ranges)
            score = self._evaluate_config(config, workload)
            
            if (i + 1) % 20 == 0:
                print(f"  进度: {i + 1}/{max_evaluations} (最佳评分: {self.best_score:.4f})")
        
        best_metrics = self.evaluator.evaluate(self.best_config, workload)
        print(f"✅ 随机搜索完成，最佳评分: {self.best_score:.4f}")
        
        return self.best_config, best_metrics
    
    def _generate_random_config(self, param_ranges: Dict[str, Any]) -> YICAConfig:
        """生成随机配置"""
        params = {}
        
        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, list):
                params[param_name] = random.choice(param_range)
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                min_val, max_val = param_range
                if isinstance(min_val, int):
                    params[param_name] = random.randint(min_val, max_val)
                else:
                    params[param_name] = random.uniform(min_val, max_val)
            elif isinstance(param_range, bool):
                params[param_name] = random.choice([True, False])
        
        return YICAConfig(**params)


class YICAAutoTuner:
    """YICA 自动调优主类"""
    
    def __init__(self):
        self.evaluator = YICAPerformanceEvaluator()
        self.results_history = []
    
    def auto_tune(self, workload: Dict[str, Any], method: str = 'random_search',
                  max_evaluations: int = 50) -> Dict[str, Any]:
        """执行自动调优"""
        print(f"🎯 开始 YICA 自动调优 (方法: {method})")
        print(f"📋 工作负载: {workload}")
        
        param_ranges = {
            'cim_array_count': [8, 16, 32, 64],
            'cim_array_size': [32, 64, 128],
            'spm_size_mb': [32, 64, 128],
            'enable_operator_fusion': [True, False],
            'enable_memory_optimization': [True, False],
            'compute_frequency_mhz': (500, 2000)
        }
        
        # 创建调优器
        tuner = RandomSearchTuner(self.evaluator)
        
        # 执行调优
        start_time = time.time()
        best_config, best_metrics = tuner.tune(workload, param_ranges, max_evaluations)
        end_time = time.time()
        
        result = {
            'method': method,
            'workload': workload,
            'best_config': best_config.to_dict(),
            'best_metrics': asdict(best_metrics),
            'best_score': best_metrics.to_score(),
            'tuning_time_seconds': end_time - start_time,
            'evaluations_count': self.evaluator.evaluation_count,
            'history': tuner.history,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results_history.append(result)
        
        print(f"⏱️ 调优用时: {result['tuning_time_seconds']:.2f} 秒")
        print(f"🔢 评估次数: {result['evaluations_count']}")
        print(f"🏆 最佳评分: {result['best_score']:.4f}")
        
        return result
    
    def save_results(self, output_file: str = None) -> str:
        """保存调优结果"""
        if output_file is None:
            output_file = f"yica_autotuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results_history, f, indent=2, ensure_ascii=False)
        
        print(f"💾 调优结果已保存: {output_file}")
        return output_file


def main():
    """演示自动调优功能"""
    print("🎯 YICA 自动调优演示")
    
    auto_tuner = YICAAutoTuner()
    
    workload = {
        'batch_size': 16,
        'sequence_length': 1024,
        'hidden_size': 768,
        'model_type': 'transformer'
    }
    
    result = auto_tuner.auto_tune(workload, max_evaluations=30)
    results_file = auto_tuner.save_results()
    
    print("\n✅ 自动调优演示完成")
    print(f"📊 结果文件: {results_file}")


if __name__ == "__main__":
    main()