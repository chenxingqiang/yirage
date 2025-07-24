#!/usr/bin/env python3
"""
YICA 性能监控和自动调优演示

这个演示展示了 YICA 性能监控和自动调优系统的完整功能：
1. 实时性能监控和异常检测
2. 多种自动调优算法和策略
3. 性能瓶颈分析和优化建议
4. 可视化仪表板和报告生成
5. 完整的性能基准测试套件
"""

import os
import sys
import time
import json
import threading
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import pandas as pd
import seaborn as sns

# 添加 Yirage 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from yirage.yica.config import YICAConfig


@dataclass
class PerformanceMetric:
    """性能指标数据"""
    type: str
    name: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, str]


@dataclass
class PerformanceAnomaly:
    """性能异常报告"""
    type: str
    description: str
    severity_score: float
    detection_time: float
    related_metrics: List[PerformanceMetric]
    suggested_actions: List[str]


@dataclass
class TuningRecommendation:
    """调优建议"""
    parameter_name: str
    current_value: str
    recommended_value: str
    expected_improvement: float
    justification: str
    priority: int


@dataclass
class AutoTuningConfig:
    """自动调优配置"""
    enable_aggressive_tuning: bool = False
    improvement_threshold: float = 0.05
    max_tuning_iterations: int = 50
    tuning_interval: int = 300  # seconds
    tunable_parameters: List[str] = None
    parameter_bounds: Dict[str, Tuple[float, float]] = None


class YICAPerformanceMonitorDemo:
    """YICA 性能监控演示类"""
    
    def __init__(self):
        # YICA 配置
        self.yica_config = YICAConfig(
            num_cim_arrays=32,
            spm_size_per_die=256 * 1024 * 1024,  # 256MB
            dram_size_per_cluster=16 * 1024 * 1024 * 1024,  # 16GB
            enable_quantization=True,
            target_precision="fp16"
        )
        
        # 监控配置
        self.monitor_config = {
            'enabled_counters': [
                'CIM_UTILIZATION', 'SPM_HIT_RATE', 'DRAM_BANDWIDTH',
                'INSTRUCTION_THROUGHPUT', 'ENERGY_CONSUMPTION', 'TEMPERATURE',
                'MEMORY_USAGE', 'COMPUTE_EFFICIENCY', 'COMMUNICATION_LATENCY'
            ],
            'sampling_interval': 100,  # ms
            'max_history_size': 10000,
            'enable_real_time_analysis': True,
            'enable_anomaly_detection': True,
            'enable_auto_tuning': True
        }
        
        # 自动调优配置
        self.auto_tuning_config = AutoTuningConfig(
            enable_aggressive_tuning=False,
            improvement_threshold=0.05,
            max_tuning_iterations=50,
            tuning_interval=300,
            tunable_parameters=[
                'cim_array_count', 'spm_allocation_strategy', 'cache_line_size',
                'prefetch_distance', 'pipeline_depth', 'batch_size'
            ],
            parameter_bounds={
                'cim_array_count': (8, 64),
                'cache_line_size': (32, 128),
                'prefetch_distance': (1, 16),
                'pipeline_depth': (2, 8),
                'batch_size': (1, 128)
            }
        )
        
        # 数据存储
        self.performance_history = defaultdict(deque)
        self.anomaly_history = []
        self.tuning_history = []
        
        # 监控状态
        self.monitoring_active = False
        self.monitoring_thread = None
        self.auto_tuning_active = False
        self.auto_tuning_thread = None
        
        # 性能基线
        self.performance_baseline = {}
        
    def demonstrate_real_time_monitoring(self):
        """演示实时性能监控"""
        print("📊 YICA 实时性能监控演示")
        print("=" * 60)
        
        print("🚀 启动性能监控...")
        self.start_monitoring()
        
        # 模拟工作负载运行
        print("⚡ 模拟工作负载运行...")
        self._simulate_workload_execution()
        
        # 显示实时监控结果
        print("\n📈 实时性能指标:")
        current_metrics = self._get_current_metrics()
        
        for metric in current_metrics:
            status_icon = self._get_metric_status_icon(metric.value, metric.type)
            print(f"  {status_icon} {metric.name}: {metric.value:.2f} {metric.unit}")
        
        # 检测异常
        anomalies = self._detect_performance_anomalies(current_metrics)
        if anomalies:
            print(f"\n⚠️  检测到 {len(anomalies)} 个性能异常:")
            for anomaly in anomalies:
                print(f"    🔴 {anomaly.type}: {anomaly.description}")
                print(f"        严重程度: {anomaly.severity_score:.2f}")
                print(f"        建议措施: {', '.join(anomaly.suggested_actions)}")
        else:
            print("\n✅ 未检测到性能异常")
        
        time.sleep(2)
        self.stop_monitoring()
        
        print("✅ 实时监控演示完成")
    
    def demonstrate_anomaly_detection(self):
        """演示异常检测功能"""
        print("\n🔍 性能异常检测演示")
        print("=" * 60)
        
        # 生成包含异常的模拟数据
        print("📊 生成包含异常的性能数据...")
        
        anomaly_scenarios = [
            {
                'name': '高延迟异常',
                'type': 'HIGH_LATENCY',
                'description': '通信延迟异常升高',
                'severity': 0.8,
                'affected_metrics': ['COMMUNICATION_LATENCY'],
                'trigger_condition': lambda x: x > 50.0  # ms
            },
            {
                'name': '低利用率异常',
                'type': 'LOW_UTILIZATION',
                'description': 'CIM 阵列利用率异常降低',
                'severity': 0.6,
                'affected_metrics': ['CIM_UTILIZATION'],
                'trigger_condition': lambda x: x < 0.3  # 30%
            },
            {
                'name': '内存泄漏异常',
                'type': 'MEMORY_LEAK',
                'description': '内存使用量持续增长',
                'severity': 0.9,
                'affected_metrics': ['MEMORY_USAGE'],
                'trigger_condition': lambda x: x > 0.9  # 90%
            },
            {
                'name': '温度限制异常',
                'type': 'THERMAL_THROTTLING',
                'description': '芯片温度过高导致性能下降',
                'severity': 0.7,
                'affected_metrics': ['TEMPERATURE'],
                'trigger_condition': lambda x: x > 85.0  # °C
            }
        ]
        
        # 模拟异常检测
        detected_anomalies = []
        
        for scenario in anomaly_scenarios:
            print(f"\n🔬 检测场景: {scenario['name']}")
            
            # 生成模拟数据
            normal_data = np.random.normal(50, 10, 100)  # 正常数据
            anomaly_data = self._inject_anomaly(normal_data, scenario)
            
            # 执行异常检测算法
            anomaly_detected = self._run_anomaly_detection(anomaly_data, scenario)
            
            if anomaly_detected:
                anomaly = PerformanceAnomaly(
                    type=scenario['type'],
                    description=scenario['description'],
                    severity_score=scenario['severity'],
                    detection_time=time.time(),
                    related_metrics=[],
                    suggested_actions=self._generate_anomaly_suggestions(scenario['type'])
                )
                detected_anomalies.append(anomaly)
                
                print(f"    🔴 异常检测: {scenario['description']}")
                print(f"    📊 严重程度: {scenario['severity']:.1f}/1.0")
                print(f"    💡 建议措施: {', '.join(anomaly.suggested_actions)}")
            else:
                print(f"    ✅ 未检测到异常")
        
        # 异常统计
        print(f"\n📈 异常检测统计:")
        print(f"  总检测场景: {len(anomaly_scenarios)}")
        print(f"  检测到异常: {len(detected_anomalies)}")
        print(f"  检测准确率: {len(detected_anomalies)/len(anomaly_scenarios)*100:.1f}%")
        
        self.anomaly_history.extend(detected_anomalies)
        
        print("✅ 异常检测演示完成")
    
    def demonstrate_auto_tuning(self):
        """演示自动调优功能"""
        print("\n🎯 自动调优演示")
        print("=" * 60)
        
        # 定义调优参数空间
        tuning_parameters = {
            'cim_array_count': {
                'current': 16,
                'candidates': [8, 12, 16, 20, 24, 32],
                'impact_weight': 0.4
            },
            'spm_allocation_strategy': {
                'current': 'locality_first',
                'candidates': ['locality_first', 'size_first', 'reuse_first', 'hybrid'],
                'impact_weight': 0.25
            },
            'cache_line_size': {
                'current': 64,
                'candidates': [32, 64, 128, 256],
                'impact_weight': 0.15
            },
            'prefetch_distance': {
                'current': 4,
                'candidates': [1, 2, 4, 8, 16],
                'impact_weight': 0.1
            },
            'pipeline_depth': {
                'current': 4,
                'candidates': [2, 3, 4, 5, 6, 8],
                'impact_weight': 0.1
            }
        }
        
        print("📊 当前参数配置:")
        for param_name, param_info in tuning_parameters.items():
            print(f"  {param_name}: {param_info['current']}")
        
        # 测量基线性能
        print("\n📈 测量基线性能...")
        baseline_performance = self._measure_performance_score(tuning_parameters)
        print(f"  基线性能得分: {baseline_performance:.3f}")
        
        # 演示不同调优算法
        tuning_algorithms = [
            ('网格搜索', self._grid_search_tuning),
            ('随机搜索', self._random_search_tuning),
            ('贝叶斯优化', self._bayesian_optimization_tuning),
            ('遗传算法', self._genetic_algorithm_tuning)
        ]
        
        best_overall_config = None
        best_overall_performance = baseline_performance
        
        for algorithm_name, algorithm_func in tuning_algorithms:
            print(f"\n🔧 运行 {algorithm_name}...")
            
            best_config, best_performance, tuning_steps = algorithm_func(
                tuning_parameters, baseline_performance
            )
            
            improvement = (best_performance - baseline_performance) / baseline_performance * 100
            
            print(f"    最佳配置: {best_config}")
            print(f"    性能得分: {best_performance:.3f}")
            print(f"    性能提升: {improvement:.1f}%")
            print(f"    调优步数: {tuning_steps}")
            
            if best_performance > best_overall_performance:
                best_overall_performance = best_performance
                best_overall_config = best_config
            
            # 记录调优历史
            self._record_tuning_attempt(algorithm_name, best_config, 
                                      baseline_performance, best_performance)
        
        # 应用最佳配置
        if best_overall_config:
            print(f"\n🏆 最佳整体配置:")
            for param, value in best_overall_config.items():
                print(f"  {param}: {tuning_parameters[param]['current']} → {value}")
            
            overall_improvement = (best_overall_performance - baseline_performance) / baseline_performance * 100
            print(f"  整体性能提升: {overall_improvement:.1f}%")
            
            # 模拟应用配置
            print("⚙️  应用最佳配置...")
            self._apply_tuning_configuration(best_overall_config)
            
        print("✅ 自动调优演示完成")
    
    def demonstrate_performance_analysis(self):
        """演示性能分析功能"""
        print("\n📊 性能分析演示")
        print("=" * 60)
        
        # 生成分析数据
        print("📈 生成性能分析数据...")
        analysis_data = self._generate_analysis_data()
        
        # 瓶颈分析
        print("\n🔍 瓶颈分析:")
        bottlenecks = self._identify_bottlenecks(analysis_data)
        
        for i, bottleneck in enumerate(bottlenecks, 1):
            print(f"  {i}. {bottleneck['component']}")
            print(f"     利用率: {bottleneck['utilization']:.1%}")
            print(f"     影响得分: {bottleneck['impact_score']:.2f}")
            print(f"     根本原因: {', '.join(bottleneck['causes'])}")
            print(f"     解决方案: {', '.join(bottleneck['solutions'])}")
        
        # 效率分析
        print("\n⚡ 效率分析:")
        efficiency_analysis = self._analyze_efficiency(analysis_data)
        
        print(f"  计算效率: {efficiency_analysis['compute_efficiency']:.1%}")
        print(f"  内存效率: {efficiency_analysis['memory_efficiency']:.1%}")
        print(f"  能效: {efficiency_analysis['energy_efficiency']:.1%}")
        print(f"  通信效率: {efficiency_analysis['communication_efficiency']:.1%}")
        print(f"  限制因子: {efficiency_analysis['limiting_factor']}")
        
        # 趋势分析
        print("\n📈 性能趋势分析:")
        trend_analysis = self._analyze_performance_trends(analysis_data)
        
        for trend in trend_analysis:
            direction_icon = "📈" if trend['slope'] > 0 else "📉" if trend['slope'] < 0 else "➡️"
            print(f"  {direction_icon} {trend['metric']}: {trend['direction']}")
            print(f"     趋势斜率: {trend['slope']:.4f}")
            print(f"     相关系数: {trend['correlation']:.3f}")
            print(f"     预测准确度: {trend['accuracy']:.1%}")
        
        # 优化建议
        print("\n💡 优化建议:")
        optimization_suggestions = self._generate_optimization_suggestions(
            bottlenecks, efficiency_analysis
        )
        
        for i, suggestion in enumerate(optimization_suggestions, 1):
            priority_icon = "🔴" if suggestion['priority'] >= 8 else "🟡" if suggestion['priority'] >= 5 else "🟢"
            print(f"  {priority_icon} {suggestion['title']}")
            print(f"     预期提升: {suggestion['expected_improvement']:.1f}%")
            print(f"     实施难度: {suggestion['implementation_difficulty']}")
            print(f"     详细说明: {suggestion['description']}")
        
        print("✅ 性能分析演示完成")
    
    def demonstrate_visualization_dashboard(self):
        """演示可视化仪表板"""
        print("\n📊 性能可视化仪表板演示")
        print("=" * 60)
        
        # 生成可视化数据
        print("📈 生成可视化数据...")
        viz_data = self._generate_visualization_data()
        
        # 创建性能仪表板
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('YICA 性能监控仪表板', fontsize=16, fontweight='bold')
        
        # 1. CIM 阵列利用率时间序列
        ax1 = axes[0, 0]
        times = viz_data['timestamps']
        cim_util = viz_data['cim_utilization']
        ax1.plot(times, cim_util, color='#2E86C1', linewidth=2)
        ax1.fill_between(times, cim_util, alpha=0.3, color='#2E86C1')
        ax1.set_title('CIM 阵列利用率')
        ax1.set_ylabel('利用率 (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # 2. 内存使用情况
        ax2 = axes[0, 1]
        memory_data = viz_data['memory_usage']
        labels = ['SPM', 'DRAM', '未使用']
        colors = ['#E74C3C', '#F39C12', '#95A5A6']
        ax2.pie(memory_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('内存使用分布')
        
        # 3. 吞吐量对比
        ax3 = axes[0, 2]
        categories = ['MatMul', 'Conv2D', 'Attention', 'MLP']
        baseline_throughput = viz_data['baseline_throughput']
        optimized_throughput = viz_data['optimized_throughput']
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax3.bar(x - width/2, baseline_throughput, width, label='基线', color='#95A5A6')
        ax3.bar(x + width/2, optimized_throughput, width, label='YICA优化', color='#27AE60')
        ax3.set_title('吞吐量对比')
        ax3.set_ylabel('GFLOPS')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 能耗效率热图
        ax4 = axes[1, 0]
        workloads = ['Small', 'Medium', 'Large', 'XLarge']
        precisions = ['FP32', 'FP16', 'INT8']
        energy_efficiency = viz_data['energy_efficiency_matrix']
        
        im = ax4.imshow(energy_efficiency, cmap='RdYlGn', aspect='auto')
        ax4.set_title('能耗效率热图')
        ax4.set_xticks(range(len(precisions)))
        ax4.set_xticklabels(precisions)
        ax4.set_yticks(range(len(workloads)))
        ax4.set_yticklabels(workloads)
        
        # 添加数值标注
        for i in range(len(workloads)):
            for j in range(len(precisions)):
                ax4.text(j, i, f'{energy_efficiency[i, j]:.1f}', 
                        ha='center', va='center', color='black', fontweight='bold')
        
        # 5. 异常检测时间线
        ax5 = axes[1, 1]
        anomaly_times = viz_data['anomaly_timestamps']
        anomaly_types = viz_data['anomaly_types']
        anomaly_severities = viz_data['anomaly_severities']
        
        colors_map = {'HIGH_LATENCY': '#E74C3C', 'LOW_UTILIZATION': '#F39C12', 
                     'MEMORY_LEAK': '#8E44AD', 'THERMAL_THROTTLING': '#E67E22'}
        
        for i, (time, atype, severity) in enumerate(zip(anomaly_times, anomaly_types, anomaly_severities)):
            color = colors_map.get(atype, '#95A5A6')
            ax5.scatter(time, severity, c=color, s=100, alpha=0.7, label=atype if i == 0 else "")
        
        ax5.set_title('异常检测时间线')
        ax5.set_xlabel('时间')
        ax5.set_ylabel('严重程度')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 1)
        
        # 6. 调优历史
        ax6 = axes[1, 2]
        tuning_iterations = viz_data['tuning_iterations']
        performance_scores = viz_data['performance_scores']
        
        ax6.plot(tuning_iterations, performance_scores, marker='o', linewidth=2, markersize=6)
        ax6.set_title('自动调优历史')
        ax6.set_xlabel('调优迭代')
        ax6.set_ylabel('性能得分')
        ax6.grid(True, alpha=0.3)
        
        # 标注最佳点
        best_idx = np.argmax(performance_scores)
        ax6.annotate(f'最佳: {performance_scores[best_idx]:.3f}', 
                    xy=(tuning_iterations[best_idx], performance_scores[best_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.savefig('yica_performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 仪表板已生成: yica_performance_dashboard.png")
        
        # 生成实时监控报告
        self._generate_performance_report(viz_data)
        
        print("✅ 可视化仪表板演示完成")
    
    def run_comprehensive_benchmark(self):
        """运行综合性能基准测试"""
        print("\n🏁 综合性能基准测试")
        print("=" * 60)
        
        # 定义基准测试套件
        benchmark_suite = {
            'matrix_multiplication': {
                'description': '矩阵乘法性能测试',
                'test_cases': [
                    {'size': (512, 512), 'precision': 'fp16'},
                    {'size': (1024, 1024), 'precision': 'fp16'},
                    {'size': (2048, 2048), 'precision': 'fp16'},
                    {'size': (1024, 1024), 'precision': 'fp32'},
                ]
            },
            'convolution_2d': {
                'description': '2D卷积性能测试',
                'test_cases': [
                    {'input_shape': (32, 3, 224, 224), 'kernel_size': 3, 'precision': 'fp16'},
                    {'input_shape': (64, 64, 56, 56), 'kernel_size': 3, 'precision': 'fp16'},
                    {'input_shape': (32, 128, 28, 28), 'kernel_size': 1, 'precision': 'fp16'},
                ]
            },
            'attention_mechanism': {
                'description': '注意力机制性能测试',
                'test_cases': [
                    {'batch_size': 32, 'seq_length': 512, 'hidden_size': 768, 'num_heads': 12},
                    {'batch_size': 16, 'seq_length': 1024, 'hidden_size': 1024, 'num_heads': 16},
                    {'batch_size': 8, 'seq_length': 2048, 'hidden_size': 1024, 'num_heads': 16},
                ]
            },
            'fused_mlp': {
                'description': '融合MLP性能测试',
                'test_cases': [
                    {'input_size': 768, 'hidden_size': 3072, 'batch_size': 32},
                    {'input_size': 1024, 'hidden_size': 4096, 'batch_size': 64},
                    {'input_size': 2048, 'hidden_size': 8192, 'batch_size': 16},
                ]
            }
        }
        
        benchmark_results = {}
        
        for benchmark_name, benchmark_config in benchmark_suite.items():
            print(f"\n🔬 运行 {benchmark_config['description']}...")
            
            test_results = []
            
            for i, test_case in enumerate(benchmark_config['test_cases']):
                print(f"  测试用例 {i+1}/{len(benchmark_config['test_cases'])}: {test_case}")
                
                # 模拟基准测试执行
                result = self._run_single_benchmark(benchmark_name, test_case)
                test_results.append(result)
                
                print(f"    延迟: {result['latency']:.3f} ms")
                print(f"    吞吐量: {result['throughput']:.1f} GFLOPS")
                print(f"    能耗: {result['energy']:.2f} W")
                print(f"    效率: {result['efficiency']:.1%}")
            
            benchmark_results[benchmark_name] = test_results
        
        # 生成基准测试报告
        self._generate_benchmark_report(benchmark_results)
        
        print("\n📊 基准测试汇总:")
        self._print_benchmark_summary(benchmark_results)
        
        print("✅ 综合性能基准测试完成")
        
        return benchmark_results
    
    # ===========================================
    # 内部辅助方法
    # ===========================================
    
    def start_monitoring(self):
        """启动性能监控"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            # 收集性能指标
            metrics = self._collect_performance_metrics()
            
            # 存储历史数据
            for metric in metrics:
                self.performance_history[metric.type].append(metric)
                
                # 限制历史数据大小
                if len(self.performance_history[metric.type]) > self.monitor_config['max_history_size']:
                    self.performance_history[metric.type].popleft()
            
            time.sleep(self.monitor_config['sampling_interval'] / 1000.0)
    
    def _simulate_workload_execution(self):
        """模拟工作负载执行"""
        workloads = ['矩阵乘法', '卷积计算', '注意力机制', 'MLP前向']
        
        for i, workload in enumerate(workloads):
            print(f"  执行 {workload}... ({i+1}/{len(workloads)})")
            time.sleep(0.5)  # 模拟执行时间
    
    def _get_current_metrics(self) -> List[PerformanceMetric]:
        """获取当前性能指标"""
        return self._collect_performance_metrics()
    
    def _collect_performance_metrics(self) -> List[PerformanceMetric]:
        """收集性能指标"""
        current_time = time.time()
        
        # 模拟性能指标采集
        metrics = [
            PerformanceMetric(
                type='CIM_UTILIZATION',
                name='CIM阵列利用率',
                value=np.random.normal(75, 10),
                unit='%',
                timestamp=current_time,
                metadata={'array_count': '32'}
            ),
            PerformanceMetric(
                type='SPM_HIT_RATE',
                name='SPM命中率',
                value=np.random.normal(85, 5),
                unit='%',
                timestamp=current_time,
                metadata={'cache_size': '256MB'}
            ),
            PerformanceMetric(
                type='DRAM_BANDWIDTH',
                name='DRAM带宽利用率',
                value=np.random.normal(60, 15),
                unit='%',
                timestamp=current_time,
                metadata={'peak_bandwidth': '1TB/s'}
            ),
            PerformanceMetric(
                type='INSTRUCTION_THROUGHPUT',
                name='指令吞吐量',
                value=np.random.normal(1200, 200),
                unit='GOPS',
                timestamp=current_time,
                metadata={'pipeline_depth': '4'}
            ),
            PerformanceMetric(
                type='ENERGY_CONSUMPTION',
                name='能耗',
                value=np.random.normal(150, 30),
                unit='W',
                timestamp=current_time,
                metadata={'voltage': '1.0V'}
            ),
            PerformanceMetric(
                type='TEMPERATURE',
                name='温度',
                value=np.random.normal(65, 8),
                unit='°C',
                timestamp=current_time,
                metadata={'sensor_location': 'core'}
            ),
            PerformanceMetric(
                type='MEMORY_USAGE',
                name='内存使用率',
                value=np.random.normal(70, 12),
                unit='%',
                timestamp=current_time,
                metadata={'total_memory': '16GB'}
            ),
            PerformanceMetric(
                type='COMMUNICATION_LATENCY',
                name='通信延迟',
                value=np.random.normal(25, 8),
                unit='ms',
                timestamp=current_time,
                metadata={'protocol': 'YCCL'}
            )
        ]
        
        # 确保值在合理范围内
        for metric in metrics:
            if metric.unit == '%':
                metric.value = max(0, min(100, metric.value))
            elif metric.unit == '°C':
                metric.value = max(20, min(100, metric.value))
            elif metric.unit == 'W':
                metric.value = max(50, min(300, metric.value))
            else:
                metric.value = max(0, metric.value)
        
        return metrics
    
    def _get_metric_status_icon(self, value: float, metric_type: str) -> str:
        """获取指标状态图标"""
        if metric_type in ['CIM_UTILIZATION', 'SPM_HIT_RATE', 'DRAM_BANDWIDTH']:
            if value >= 80:
                return "🟢"
            elif value >= 60:
                return "🟡"
            else:
                return "🔴"
        elif metric_type == 'TEMPERATURE':
            if value <= 70:
                return "🟢"
            elif value <= 85:
                return "🟡"
            else:
                return "🔴"
        elif metric_type == 'COMMUNICATION_LATENCY':
            if value <= 20:
                return "🟢"
            elif value <= 40:
                return "🟡"
            else:
                return "🔴"
        else:
            return "📊"
    
    def _detect_performance_anomalies(self, metrics: List[PerformanceMetric]) -> List[PerformanceAnomaly]:
        """检测性能异常"""
        anomalies = []
        
        for metric in metrics:
            anomaly = None
            
            if metric.type == 'CIM_UTILIZATION' and metric.value < 30:
                anomaly = PerformanceAnomaly(
                    type='LOW_UTILIZATION',
                    description=f'CIM阵列利用率异常低: {metric.value:.1f}%',
                    severity_score=0.7,
                    detection_time=metric.timestamp,
                    related_metrics=[metric],
                    suggested_actions=['检查工作负载分布', '优化数据布局', '调整并行度']
                )
            elif metric.type == 'TEMPERATURE' and metric.value > 85:
                anomaly = PerformanceAnomaly(
                    type='THERMAL_THROTTLING',
                    description=f'温度过高: {metric.value:.1f}°C',
                    severity_score=0.8,
                    detection_time=metric.timestamp,
                    related_metrics=[metric],
                    suggested_actions=['检查散热系统', '降低工作频率', '优化功耗管理']
                )
            elif metric.type == 'COMMUNICATION_LATENCY' and metric.value > 50:
                anomaly = PerformanceAnomaly(
                    type='HIGH_LATENCY',
                    description=f'通信延迟过高: {metric.value:.1f}ms',
                    severity_score=0.6,
                    detection_time=metric.timestamp,
                    related_metrics=[metric],
                    suggested_actions=['检查网络连接', '优化通信协议', '调整批次大小']
                )
            elif metric.type == 'MEMORY_USAGE' and metric.value > 90:
                anomaly = PerformanceAnomaly(
                    type='MEMORY_LEAK',
                    description=f'内存使用率过高: {metric.value:.1f}%',
                    severity_score=0.9,
                    detection_time=metric.timestamp,
                    related_metrics=[metric],
                    suggested_actions=['检查内存泄漏', '优化内存管理', '增加内存容量']
                )
            
            if anomaly:
                anomalies.append(anomaly)
        
        return anomalies
    
    def _inject_anomaly(self, normal_data: np.ndarray, scenario: Dict[str, Any]) -> np.ndarray:
        """向正常数据中注入异常"""
        anomaly_data = normal_data.copy()
        
        # 在后半部分注入异常
        anomaly_start = len(anomaly_data) // 2
        
        if scenario['type'] == 'HIGH_LATENCY':
            anomaly_data[anomaly_start:] += 30  # 增加延迟
        elif scenario['type'] == 'LOW_UTILIZATION':
            anomaly_data[anomaly_start:] *= 0.3  # 降低利用率
        elif scenario['type'] == 'MEMORY_LEAK':
            # 模拟内存泄漏的线性增长
            leak_growth = np.linspace(0, 40, len(anomaly_data) - anomaly_start)
            anomaly_data[anomaly_start:] += leak_growth
        elif scenario['type'] == 'THERMAL_THROTTLING':
            anomaly_data[anomaly_start:] += 25  # 增加温度
        
        return anomaly_data
    
    def _run_anomaly_detection(self, data: np.ndarray, scenario: Dict[str, Any]) -> bool:
        """运行异常检测算法"""
        # 简化的异常检测：基于统计阈值
        recent_data = data[-20:]  # 最近20个数据点
        
        if len(recent_data) < 5:
            return False
        
        mean_recent = np.mean(recent_data)
        
        # 根据场景类型检测异常
        return scenario['trigger_condition'](mean_recent)
    
    def _generate_anomaly_suggestions(self, anomaly_type: str) -> List[str]:
        """生成异常处理建议"""
        suggestions_map = {
            'HIGH_LATENCY': [
                '检查网络连接质量',
                '优化数据传输协议',
                '调整批次大小',
                '启用数据压缩'
            ],
            'LOW_UTILIZATION': [
                '增加工作负载并行度',
                '优化数据布局',
                '调整CIM阵列配置',
                '检查资源分配策略'
            ],
            'MEMORY_LEAK': [
                '检查内存分配和释放',
                '优化数据结构',
                '启用内存池',
                '增加垃圾回收频率'
            ],
            'THERMAL_THROTTLING': [
                '检查散热系统',
                '降低工作频率',
                '优化功耗管理',
                '调整环境温度'
            ]
        }
        
        return suggestions_map.get(anomaly_type, ['联系技术支持'])
    
    def _measure_performance_score(self, parameters: Dict[str, Any]) -> float:
        """测量性能得分"""
        # 模拟性能测量
        base_score = 0.7
        
        # 根据参数配置计算性能得分
        for param_name, param_info in parameters.items():
            current_value = param_info['current']
            weight = param_info['impact_weight']
            
            # 简化的性能模型
            if param_name == 'cim_array_count':
                # CIM阵列数量的影响
                optimal_count = 24
                efficiency = 1.0 - abs(current_value - optimal_count) / optimal_count * 0.3
                base_score += weight * efficiency * 0.5
            elif param_name == 'cache_line_size':
                # 缓存行大小的影响
                if current_value == 64:
                    base_score += weight * 0.1
                elif current_value == 128:
                    base_score += weight * 0.05
            # 其他参数的简化影响...
        
        # 添加随机噪声模拟真实测量的变化
        noise = np.random.normal(0, 0.02)
        return max(0.1, min(1.0, base_score + noise))
    
    def _grid_search_tuning(self, parameters: Dict[str, Any], baseline: float) -> Tuple[Dict[str, Any], float, int]:
        """网格搜索调优"""
        best_config = {}
        best_performance = baseline
        steps = 0
        
        # 简化的网格搜索：只搜索前两个最重要的参数
        important_params = sorted(parameters.items(), key=lambda x: x[1]['impact_weight'], reverse=True)[:2]
        
        for param1_name, param1_info in [important_params[0]]:
            for param1_value in param1_info['candidates'][:3]:  # 限制搜索空间
                for param2_name, param2_info in [important_params[1]]:
                    for param2_value in param2_info['candidates'][:3]:
                        # 创建测试配置
                        test_config = {param1_name: param1_value, param2_name: param2_value}
                        
                        # 模拟性能测试
                        performance = self._simulate_performance_test(test_config, parameters)
                        steps += 1
                        
                        if performance > best_performance:
                            best_performance = performance
                            best_config = test_config
        
        return best_config, best_performance, steps
    
    def _random_search_tuning(self, parameters: Dict[str, Any], baseline: float) -> Tuple[Dict[str, Any], float, int]:
        """随机搜索调优"""
        best_config = {}
        best_performance = baseline
        steps = 20  # 随机搜索20次
        
        for _ in range(steps):
            # 随机选择参数值
            test_config = {}
            for param_name, param_info in parameters.items():
                test_config[param_name] = np.random.choice(param_info['candidates'])
            
            # 模拟性能测试
            performance = self._simulate_performance_test(test_config, parameters)
            
            if performance > best_performance:
                best_performance = performance
                best_config = test_config
        
        return best_config, best_performance, steps
    
    def _bayesian_optimization_tuning(self, parameters: Dict[str, Any], baseline: float) -> Tuple[Dict[str, Any], float, int]:
        """贝叶斯优化调优（简化版）"""
        best_config = {}
        best_performance = baseline
        steps = 15
        
        # 简化的贝叶斯优化：基于历史最佳结果进行智能搜索
        for i in range(steps):
            if i < 5:
                # 前几次随机探索
                test_config = {}
                for param_name, param_info in parameters.items():
                    test_config[param_name] = np.random.choice(param_info['candidates'])
            else:
                # 后续基于已知最佳配置进行局部搜索
                test_config = best_config.copy()
                # 随机改变一个参数
                param_to_change = np.random.choice(list(parameters.keys()))
                param_info = parameters[param_to_change]
                test_config[param_to_change] = np.random.choice(param_info['candidates'])
            
            performance = self._simulate_performance_test(test_config, parameters)
            
            if performance > best_performance:
                best_performance = performance
                best_config = test_config
        
        return best_config, best_performance, steps
    
    def _genetic_algorithm_tuning(self, parameters: Dict[str, Any], baseline: float) -> Tuple[Dict[str, Any], float, int]:
        """遗传算法调优（简化版）"""
        population_size = 10
        generations = 5
        steps = population_size * generations
        
        # 初始化种群
        population = []
        for _ in range(population_size):
            individual = {}
            for param_name, param_info in parameters.items():
                individual[param_name] = np.random.choice(param_info['candidates'])
            population.append(individual)
        
        best_config = {}
        best_performance = baseline
        
        for generation in range(generations):
            # 评估种群
            fitness_scores = []
            for individual in population:
                performance = self._simulate_performance_test(individual, parameters)
                fitness_scores.append(performance)
                
                if performance > best_performance:
                    best_performance = performance
                    best_config = individual.copy()
            
            # 选择和交叉（简化版）
            # 保留最好的一半
            sorted_indices = np.argsort(fitness_scores)[-population_size//2:]
            new_population = [population[i] for i in sorted_indices]
            
            # 生成新个体（交叉和变异）
            while len(new_population) < population_size:
                parent1, parent2 = np.random.choice(new_population, 2, replace=False)
                child = {}
                for param_name in parameters.keys():
                    # 随机选择父母之一的基因
                    child[param_name] = np.random.choice([parent1[param_name], parent2[param_name]])
                
                # 变异
                if np.random.random() < 0.1:  # 10% 变异概率
                    param_to_mutate = np.random.choice(list(parameters.keys()))
                    param_info = parameters[param_to_mutate]
                    child[param_to_mutate] = np.random.choice(param_info['candidates'])
                
                new_population.append(child)
            
            population = new_population
        
        return best_config, best_performance, steps
    
    def _simulate_performance_test(self, config: Dict[str, Any], parameters: Dict[str, Any]) -> float:
        """模拟性能测试"""
        base_score = 0.7
        
        for param_name, param_value in config.items():
            if param_name in parameters:
                weight = parameters[param_name]['impact_weight']
                
                # 简化的参数影响模型
                if param_name == 'cim_array_count':
                    optimal_value = 24
                    efficiency = 1.0 - abs(param_value - optimal_value) / optimal_value * 0.3
                    base_score += weight * efficiency * 0.3
                elif param_name == 'spm_allocation_strategy':
                    if param_value == 'hybrid':
                        base_score += weight * 0.15
                    elif param_value == 'locality_first':
                        base_score += weight * 0.1
                # 其他参数...
        
        # 添加噪声
        noise = np.random.normal(0, 0.01)
        return max(0.1, min(1.0, base_score + noise))
    
    def _record_tuning_attempt(self, algorithm: str, config: Dict[str, Any], 
                             before: float, after: float):
        """记录调优尝试"""
        self.tuning_history.append({
            'timestamp': time.time(),
            'algorithm': algorithm,
            'config': config,
            'performance_before': before,
            'performance_after': after,
            'improvement': (after - before) / before * 100
        })
    
    def _apply_tuning_configuration(self, config: Dict[str, Any]):
        """应用调优配置"""
        print("    配置已应用到YICA系统")
        # 在实际实现中，这里会调用系统API来应用配置
        time.sleep(0.5)
    
    def _generate_analysis_data(self) -> Dict[str, Any]:
        """生成分析数据"""
        return {
            'cim_utilization': np.random.normal(75, 10, 100),
            'spm_hit_rate': np.random.normal(85, 5, 100),
            'memory_usage': np.random.normal(70, 15, 100),
            'energy_consumption': np.random.normal(150, 20, 100),
            'throughput': np.random.normal(1200, 200, 100),
            'latency': np.random.normal(25, 8, 100)
        }
    
    def _identify_bottlenecks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        # 分析各组件利用率
        if np.mean(data['cim_utilization']) < 60:
            bottlenecks.append({
                'component': 'CIM阵列',
                'utilization': np.mean(data['cim_utilization']) / 100,
                'impact_score': 0.8,
                'causes': ['数据依赖', '负载不均衡', '同步开销'],
                'solutions': ['优化数据布局', '增加并行度', '减少同步点']
            })
        
        if np.mean(data['spm_hit_rate']) < 80:
            bottlenecks.append({
                'component': 'SPM缓存',
                'utilization': np.mean(data['spm_hit_rate']) / 100,
                'impact_score': 0.6,
                'causes': ['缓存容量不足', '访问模式不规律', '预取策略不当'],
                'solutions': ['增加缓存容量', '优化访问模式', '改进预取算法']
            })
        
        if np.mean(data['memory_usage']) > 85:
            bottlenecks.append({
                'component': '内存子系统',
                'utilization': np.mean(data['memory_usage']) / 100,
                'impact_score': 0.7,
                'causes': ['内存带宽限制', '内存碎片', '数据局部性差'],
                'solutions': ['优化内存访问模式', '使用内存池', '提高数据局部性']
            })
        
        return sorted(bottlenecks, key=lambda x: x['impact_score'], reverse=True)
    
    def _analyze_efficiency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析效率"""
        return {
            'compute_efficiency': min(1.0, np.mean(data['cim_utilization']) / 100),
            'memory_efficiency': min(1.0, np.mean(data['spm_hit_rate']) / 100),
            'energy_efficiency': min(1.0, np.mean(data['throughput']) / np.mean(data['energy_consumption']) * 10),
            'communication_efficiency': min(1.0, max(0.1, 1.0 - np.mean(data['latency']) / 100)),
            'limiting_factor': '内存带宽' if np.mean(data['memory_usage']) > 80 else 'CIM利用率'
        }
    
    def _analyze_performance_trends(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析性能趋势"""
        trends = []
        
        for metric_name, values in data.items():
            if len(values) > 10:
                # 计算趋势斜率
                x = np.arange(len(values))
                slope, _ = np.polyfit(x, values, 1)
                
                # 计算相关系数
                correlation = np.corrcoef(x, values)[0, 1]
                
                # 确定趋势方向
                if abs(slope) < 0.1:
                    direction = "稳定"
                elif slope > 0:
                    direction = "上升"
                else:
                    direction = "下降"
                
                trends.append({
                    'metric': metric_name,
                    'slope': slope,
                    'correlation': correlation,
                    'direction': direction,
                    'accuracy': abs(correlation)
                })
        
        return trends
    
    def _generate_optimization_suggestions(self, bottlenecks: List[Dict[str, Any]], 
                                         efficiency: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成优化建议"""
        suggestions = []
        
        # 基于瓶颈的建议
        for bottleneck in bottlenecks:
            if bottleneck['component'] == 'CIM阵列':
                suggestions.append({
                    'title': '优化CIM阵列利用率',
                    'expected_improvement': 25.0,
                    'implementation_difficulty': '中等',
                    'priority': 8,
                    'description': '通过优化数据布局和增加并行度来提高CIM阵列利用率'
                })
            elif bottleneck['component'] == 'SPM缓存':
                suggestions.append({
                    'title': '改进SPM缓存策略',
                    'expected_improvement': 15.0,
                    'implementation_difficulty': '简单',
                    'priority': 6,
                    'description': '优化缓存预取策略和替换算法'
                })
        
        # 基于效率的建议
        if efficiency['energy_efficiency'] < 0.7:
            suggestions.append({
                'title': '优化能耗效率',
                'expected_improvement': 20.0,
                'implementation_difficulty': '困难',
                'priority': 7,
                'description': '通过动态电压频率调节和功耗管理来提高能效'
            })
        
        return sorted(suggestions, key=lambda x: x['priority'], reverse=True)
    
    def _generate_visualization_data(self) -> Dict[str, Any]:
        """生成可视化数据"""
        # 时间序列数据
        timestamps = np.arange(100)
        
        return {
            'timestamps': timestamps,
            'cim_utilization': np.random.normal(75, 10, 100).clip(0, 100),
            'memory_usage': [60, 25, 15],  # SPM, DRAM, 未使用
            'baseline_throughput': [800, 1200, 900, 1100],
            'optimized_throughput': [1200, 1800, 1350, 1650],
            'energy_efficiency_matrix': np.random.uniform(5, 15, (4, 3)),
            'anomaly_timestamps': np.random.choice(timestamps, 5),
            'anomaly_types': ['HIGH_LATENCY', 'LOW_UTILIZATION', 'MEMORY_LEAK', 'THERMAL_THROTTLING', 'HIGH_LATENCY'],
            'anomaly_severities': np.random.uniform(0.3, 0.9, 5),
            'tuning_iterations': np.arange(1, 21),
            'performance_scores': np.random.normal(0.75, 0.05, 20).cummax()  # 单调递增
        }
    
    def _generate_performance_report(self, viz_data: Dict[str, Any]):
        """生成性能报告"""
        report_content = f"""
# YICA 性能监控报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 系统配置
- CIM 阵列数量: {self.yica_config.num_cim_arrays}
- SPM 容量: {self.yica_config.spm_size_per_die // (1024*1024)} MB
- DRAM 容量: {self.yica_config.dram_size_per_cluster // (1024*1024*1024)} GB
- 目标精度: {self.yica_config.target_precision}

## 性能摘要
- 平均CIM利用率: {np.mean(viz_data['cim_utilization']):.1f}%
- 基线吞吐量: {np.mean(viz_data['baseline_throughput']):.0f} GFLOPS
- 优化后吞吐量: {np.mean(viz_data['optimized_throughput']):.0f} GFLOPS
- 性能提升: {(np.mean(viz_data['optimized_throughput']) / np.mean(viz_data['baseline_throughput']) - 1) * 100:.1f}%

## 异常检测
- 检测到异常: {len(viz_data['anomaly_timestamps'])} 个
- 平均严重程度: {np.mean(viz_data['anomaly_severities']):.2f}

## 调优历史
- 调优迭代次数: {len(viz_data['tuning_iterations'])}
- 最终性能得分: {viz_data['performance_scores'][-1]:.3f}
- 调优提升: {(viz_data['performance_scores'][-1] / viz_data['performance_scores'][0] - 1) * 100:.1f}%

## 建议
1. 继续优化CIM阵列利用率
2. 改进内存访问模式
3. 启用更激进的调优策略
4. 定期监控系统性能指标
"""
        
        with open('yica_performance_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print("📄 性能报告已生成: yica_performance_report.md")
    
    def _run_single_benchmark(self, benchmark_name: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个基准测试"""
        # 模拟基准测试执行
        base_latency = 5.0  # ms
        base_throughput = 1000.0  # GFLOPS
        base_energy = 100.0  # W
        
        # 根据测试用例调整性能
        if 'size' in test_case:
            size_factor = np.prod(test_case['size']) / (1024 * 1024)
            base_latency *= size_factor * 0.1
            base_throughput /= size_factor * 0.05
        
        if 'precision' in test_case:
            if test_case['precision'] == 'fp16':
                base_latency *= 0.7
                base_throughput *= 1.4
                base_energy *= 0.8
        
        # 添加随机变化
        latency = base_latency * (1 + np.random.normal(0, 0.1))
        throughput = base_throughput * (1 + np.random.normal(0, 0.1))
        energy = base_energy * (1 + np.random.normal(0, 0.1))
        
        efficiency = throughput / (throughput * 1.2)  # 相对于理论峰值
        
        return {
            'latency': latency,
            'throughput': throughput,
            'energy': energy,
            'efficiency': efficiency
        }
    
    def _generate_benchmark_report(self, results: Dict[str, Any]):
        """生成基准测试报告"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'yica_config': asdict(self.yica_config),
            'benchmark_results': results,
            'summary': {}
        }
        
        # 计算汇总统计
        all_latencies = []
        all_throughputs = []
        all_efficiencies = []
        
        for benchmark_name, test_results in results.items():
            for result in test_results:
                all_latencies.append(result['latency'])
                all_throughputs.append(result['throughput'])
                all_efficiencies.append(result['efficiency'])
        
        report_data['summary'] = {
            'avg_latency': np.mean(all_latencies),
            'avg_throughput': np.mean(all_throughputs),
            'avg_efficiency': np.mean(all_efficiencies),
            'total_tests': sum(len(results) for results in results.values())
        }
        
        with open('yica_benchmark_results.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print("📊 基准测试报告已保存: yica_benchmark_results.json")
    
    def _print_benchmark_summary(self, results: Dict[str, Any]):
        """打印基准测试汇总"""
        print("  基准测试类型 | 平均延迟(ms) | 平均吞吐量(GFLOPS) | 平均效率")
        print("  " + "-" * 60)
        
        for benchmark_name, test_results in results.items():
            avg_latency = np.mean([r['latency'] for r in test_results])
            avg_throughput = np.mean([r['throughput'] for r in test_results])
            avg_efficiency = np.mean([r['efficiency'] for r in test_results])
            
            print(f"  {benchmark_name:<15} | {avg_latency:>10.2f} | {avg_throughput:>15.1f} | {avg_efficiency:>8.1%}")


def main():
    """主函数"""
    print("🚀 YICA 性能监控和自动调优演示启动")
    print("=" * 80)
    
    # 创建演示实例
    demo = YICAPerformanceMonitorDemo()
    
    try:
        # 运行各种演示
        demo.demonstrate_real_time_monitoring()
        demo.demonstrate_anomaly_detection()
        demo.demonstrate_auto_tuning()
        demo.demonstrate_performance_analysis()
        demo.demonstrate_visualization_dashboard()
        benchmark_results = demo.run_comprehensive_benchmark()
        
        print("\n" + "=" * 80)
        print("🎉 YICA 性能监控和自动调优演示完成!")
        print(f"📊 完成 {sum(len(results) for results in benchmark_results.values())} 项基准测试")
        print(f"🔍 检测到 {len(demo.anomaly_history)} 个性能异常")
        print(f"🎯 执行了 {len(demo.tuning_history)} 次调优尝试")
        print("💾 所有结果已保存到文件")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  演示被用户中断")
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        if demo.monitoring_active:
            demo.stop_monitoring()


if __name__ == "__main__":
    main() 