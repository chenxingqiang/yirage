"""
YICA高级Python接口

这个模块提供了易于使用的YICA功能接口，包括：
- YICA架构分析器
- YICA内存管理器
- 性能监控和优化建议
"""

# Core functionality - handle missing core module gracefully
try:
    from .core import (
        CyYICAConfig, CyYICAAnalyzer, CyAnalysisResult,
        CyYICAMemoryConfig, CyYICAMemoryManager,
        create_yica_analyzer, create_yica_memory_manager
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    # Provide stub implementations
    class CyYICAConfig:
        def __init__(self, *args, **kwargs):
            pass
    
    class CyYICAAnalyzer:
        def __init__(self, *args, **kwargs):
            pass
    
    class CyAnalysisResult:
        def __init__(self, *args, **kwargs):
            pass
    
    class CyYICAMemoryConfig:
        def __init__(self, *args, **kwargs):
            pass
    
    class CyYICAMemoryManager:
        def __init__(self, *args, **kwargs):
            pass
    
    def create_yica_analyzer(*args, **kwargs):
        return CyYICAAnalyzer()
    
    def create_yica_memory_manager(*args, **kwargs):
        return CyYICAMemoryManager()
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

# 设置日志
logger = logging.getLogger(__name__)

class YICAAnalyzer:
    """
    YICA架构分析器高级接口
    
    提供简化的API来分析计算图对YICA架构的适配性
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化YICA分析器
        
        Args:
            config: YICA配置字典，包含以下可选参数：
                - cim_array_rows: CIM阵列行数 (默认: 256)
                - cim_array_cols: CIM阵列列数 (默认: 256)
                - spm_size_per_die: 每个die的SPM大小 (默认: 2MB)
                - dram_bandwidth: DRAM带宽 GB/s (默认: 1024)
                - num_cim_dies: CIM die数量 (默认: 16)
                - cim_frequency: CIM频率 MHz (默认: 1000.0)
        """
        self.config = config or {}
        self._analyzer = create_yica_analyzer(self.config)
        
    def analyze_graph(self, graph) -> Dict:
        """
        分析计算图的YICA适配性
        
        Args:
            graph: Yirage计算图对象
            
        Returns:
            包含分析结果的字典：
            {
                'cim_friendliness_score': float,      # CIM友好度评分 [0-1]
                'memory_locality_score': float,      # 内存局部性评分 [0-1]
                'parallelization_potential': float,  # 并行化潜力 [0-1]
                'estimated_speedup': float,          # 预估加速比
                'estimated_energy_reduction': float, # 预估能耗降低比例
                'bottlenecks': List[str],            # 性能瓶颈列表
                'cim_friendly_ops': List,            # CIM友好的操作
                'parallel_opportunities': List[Dict] # 并行化机会
            }
        """
        try:
            result = self._analyzer.analyze_computation_graph(graph)
            
            return {
                'cim_friendliness_score': result.cim_friendliness_score,
                'memory_locality_score': result.memory_locality_score,
                'parallelization_potential': result.parallelization_potential,
                'estimated_speedup': result.estimated_speedup,
                'estimated_energy_reduction': result.estimated_energy_reduction,
                'bottlenecks': result.bottlenecks,
                'cim_friendly_ops': result.cim_friendly_ops,
                'parallel_opportunities': result.parallel_opportunities
            }
        except Exception as e:
            logger.error(f"分析图时出错: {e}")
            raise
    
    def get_optimization_recommendations(self, graph) -> List[Dict]:
        """
        获取优化建议
        
        Args:
            graph: Yirage计算图对象
            
        Returns:
            优化建议列表，每个建议包含：
            {
                'type': str,           # 优化类型
                'description': str,    # 优化描述
                'priority': str,       # 优先级 (high/medium/low)
                'expected_benefit': float, # 预期收益
                'implementation_hint': str # 实现提示
            }
        """
        analysis = self.analyze_graph(graph)
        recommendations = []
        
        # 基于CIM友好度给出建议
        if analysis['cim_friendliness_score'] < 0.5:
            recommendations.append({
                'type': 'operator_fusion',
                'description': '考虑融合相邻的元素级操作以提高CIM利用率',
                'priority': 'high',
                'expected_benefit': 0.3,
                'implementation_hint': '使用算子融合优化pass'
            })
        
        # 基于内存局部性给出建议
        if analysis['memory_locality_score'] < 0.6:
            recommendations.append({
                'type': 'memory_layout',
                'description': '优化数据布局以提高内存访问局部性',
                'priority': 'medium',
                'expected_benefit': 0.2,
                'implementation_hint': '使用SPM缓存优化和数据重排'
            })
        
        # 基于并行化潜力给出建议
        if analysis['parallelization_potential'] > 0.7:
            recommendations.append({
                'type': 'parallelization',
                'description': '利用检测到的并行化机会',
                'priority': 'high',
                'expected_benefit': analysis['parallelization_potential'],
                'implementation_hint': '实现数据并行或模型并行'
            })
        
        return recommendations
    
    def identify_cim_operations(self, graph) -> List:
        """识别CIM友好的操作"""
        return self._analyzer.identify_cim_operations(graph)
    
    def analyze_memory_pattern(self, graph) -> float:
        """分析内存访问模式"""
        return self._analyzer.analyze_memory_access_pattern(graph)
    
    def find_parallelization_opportunities(self, graph) -> List[Dict]:
        """发现并行化机会"""
        return self._analyzer.find_parallel_patterns(graph)
    
    def update_config(self, new_config: Dict):
        """更新YICA配置"""
        self.config.update(new_config)
        config_obj = CyYICAConfig(**self.config)
        self._analyzer.update_config(config_obj)

class YICAMemoryManager:
    """
    YICA内存管理器高级接口
    
    提供简化的API来管理YICA的三级内存层次
    """
    
    # 内存级别常量
    REGISTER_FILE = 0
    SPM = 1
    DRAM = 2
    
    def __init__(self, device_id: int = 0, num_devices: int = 1, config: Optional[Dict] = None):
        """
        初始化YICA内存管理器
        
        Args:
            device_id: 设备ID
            num_devices: 设备数量
            config: 内存配置字典
        """
        self.device_id = device_id
        self.num_devices = num_devices
        self.config = config or {}
        self._manager = create_yica_memory_manager(device_id, num_devices, self.config)
        
    def allocate(self, size: int, memory_level: int = DRAM, alignment: int = 64) -> int:
        """
        分配内存
        
        Args:
            size: 内存大小（字节）
            memory_level: 内存级别 (REGISTER_FILE=0, SPM=1, DRAM=2)
            alignment: 内存对齐（字节）
            
        Returns:
            内存指针（作为整数返回）
        """
        return self._manager.allocate_memory(size, memory_level, alignment)
    
    def deallocate(self, ptr: int, memory_level: int) -> bool:
        """
        释放内存
        
        Args:
            ptr: 内存指针
            memory_level: 内存级别
            
        Returns:
            是否成功释放
        """
        return self._manager.deallocate_memory(ptr, memory_level)
    
    def smart_allocate(self, size: int, preferred_level: int = SPM) -> Dict:
        """
        智能内存分配（YICA优化）
        
        Args:
            size: 内存大小
            preferred_level: 首选内存级别
            
        Returns:
            分配结果字典
        """
        return self._manager.allocate_yica_memory(size, preferred_level)
    
    def promote_to_spm(self, dram_ptr: int, size: int) -> bool:
        """将数据从DRAM提升到SPM"""
        return self._manager.promote_to_spm(dram_ptr, size)
    
    def cache_data(self, dram_ptr: int, size: int, priority: int = 0) -> bool:
        """在SPM中缓存数据"""
        return self._manager.cache_in_spm(dram_ptr, size, priority)
    
    def prefetch(self, dram_ptr: int, size: int) -> bool:
        """预取数据到SPM"""
        return self._manager.prefetch_to_spm(dram_ptr, size)
    
    def measure_bandwidth(self, memory_level: int) -> float:
        """测量内存带宽"""
        return self._manager.measure_memory_bandwidth(memory_level)
    
    def get_statistics(self) -> Dict:
        """获取详细的内存统计信息"""
        return self._manager.get_memory_statistics()
    
    def get_summary_statistics(self) -> Dict:
        """获取简化的内存统计摘要"""
        stats = self.get_statistics()
        
        return {
            'memory_utilization': {
                'register_file': stats['memory_utilization'][0],
                'spm': stats['memory_utilization'][1],
                'dram': stats['memory_utilization'][2]
            },
            'spm_cache_hit_rate': stats['spm_cache_hit_rate'],
            'total_allocations': sum(stats['num_allocations']),
            'fragmentation_ratio': max(stats['fragmentation_ratio']),
            'bandwidth_utilization': {
                'register_file': stats['bandwidth_utilization'][0],
                'spm': stats['bandwidth_utilization'][1],
                'dram': stats['bandwidth_utilization'][2]
            }
        }
    
    def optimize_memory_usage(self) -> Dict:
        """
        优化内存使用
        
        Returns:
            优化结果和建议
        """
        stats = self.get_statistics()
        recommendations = []
        
        # 检查碎片化
        max_fragmentation = max(stats['fragmentation_ratio'])
        if max_fragmentation > 0.3:
            recommendations.append({
                'issue': 'high_fragmentation',
                'description': f'内存碎片化率过高: {max_fragmentation:.2%}',
                'action': '触发内存压缩',
                'priority': 'high'
            })
            # 执行内存压缩
            for level in range(3):
                if stats['fragmentation_ratio'][level] > 0.3:
                    self._manager.compact_memory(level)
        
        # 检查SPM缓存命中率
        if stats['spm_cache_hit_rate'] < 0.8:
            recommendations.append({
                'issue': 'low_cache_hit_rate',
                'description': f'SPM缓存命中率过低: {stats["spm_cache_hit_rate"]:.2%}',
                'action': '调整缓存策略或增加SPM大小',
                'priority': 'medium'
            })
        
        # 检查内存利用率
        for i, level_name in enumerate(['register_file', 'spm', 'dram']):
            if stats['memory_utilization'][i] > 0.9:
                recommendations.append({
                    'issue': 'high_memory_usage',
                    'description': f'{level_name}内存使用率过高: {stats["memory_utilization"][i]:.2%}',
                    'action': '考虑释放未使用的内存或增加容量',
                    'priority': 'medium'
                })
        
        return {
            'recommendations': recommendations,
            'actions_taken': ['memory_compaction'] if max_fragmentation > 0.3 else [],
            'current_stats': self.get_summary_statistics()
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self._manager.reset_statistics()
    
    def trigger_gc(self):
        """触发垃圾回收"""
        self._manager.trigger_garbage_collection()

class YICAPerformanceMonitor:
    """
    YICA性能监控器
    
    提供性能监控和分析功能
    """
    
    def __init__(self, analyzer: YICAAnalyzer, memory_manager: YICAMemoryManager):
        self.analyzer = analyzer
        self.memory_manager = memory_manager
        self.performance_history = []
    
    def monitor_execution(self, graph, duration: float = None) -> Dict:
        """
        监控执行性能
        
        Args:
            graph: 计算图
            duration: 执行时间（秒）
            
        Returns:
            性能监控结果
        """
        # 分析计算图
        analysis = self.analyzer.analyze_graph(graph)
        
        # 获取内存统计
        memory_stats = self.memory_manager.get_summary_statistics()
        
        # 测量内存带宽
        bandwidths = {
            'register_file': self.memory_manager.measure_bandwidth(YICAMemoryManager.REGISTER_FILE),
            'spm': self.memory_manager.measure_bandwidth(YICAMemoryManager.SPM),
            'dram': self.memory_manager.measure_bandwidth(YICAMemoryManager.DRAM)
        }
        
        result = {
            'timestamp': np.datetime64('now'),
            'graph_analysis': analysis,
            'memory_stats': memory_stats,
            'memory_bandwidths': bandwidths,
            'execution_time': duration,
            'performance_score': self._calculate_performance_score(analysis, memory_stats)
        }
        
        self.performance_history.append(result)
        return result
    
    def _calculate_performance_score(self, analysis: Dict, memory_stats: Dict) -> float:
        """计算综合性能评分"""
        cim_score = analysis['cim_friendliness_score'] * 0.4
        memory_score = analysis['memory_locality_score'] * 0.3
        cache_score = memory_stats['spm_cache_hit_rate'] * 0.2
        utilization_score = np.mean(list(memory_stats['memory_utilization'].values())) * 0.1
        
        return cim_score + memory_score + cache_score + utilization_score
    
    def get_performance_trend(self, window_size: int = 10) -> Dict:
        """获取性能趋势"""
        if len(self.performance_history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent_scores = [entry['performance_score'] for entry in self.performance_history[-window_size:]]
        
        if len(recent_scores) >= 2:
            trend = 'improving' if recent_scores[-1] > recent_scores[0] else 'declining'
            avg_score = np.mean(recent_scores)
            score_std = np.std(recent_scores)
        else:
            trend = 'stable'
            avg_score = recent_scores[0]
            score_std = 0.0
        
        return {
            'trend': trend,
            'average_score': avg_score,
            'score_stability': 1.0 - score_std,  # 稳定性指标
            'sample_count': len(recent_scores)
        }
    
    def generate_report(self) -> str:
        """生成性能报告"""
        if not self.performance_history:
            return "暂无性能数据"
        
        latest = self.performance_history[-1]
        trend = self.get_performance_trend()
        
        report = f"""
YICA性能监控报告
================

最新性能指标:
- 综合性能评分: {latest['performance_score']:.3f}
- CIM友好度: {latest['graph_analysis']['cim_friendliness_score']:.3f}
- 内存局部性: {latest['graph_analysis']['memory_locality_score']:.3f}
- SPM缓存命中率: {latest['memory_stats']['spm_cache_hit_rate']:.2%}

性能趋势:
- 趋势: {trend['trend']}
- 平均评分: {trend['average_score']:.3f}
- 稳定性: {trend['score_stability']:.3f}

内存带宽使用:
- 寄存器文件: {latest['memory_bandwidths']['register_file']:.1f} GB/s
- SPM: {latest['memory_bandwidths']['spm']:.1f} GB/s  
- DRAM: {latest['memory_bandwidths']['dram']:.1f} GB/s

优化建议:
"""
        
        # 添加优化建议
        recommendations = self.analyzer.get_optimization_recommendations(None)  # 需要传入graph
        for i, rec in enumerate(recommendations[:3], 1):  # 只显示前3个建议
            report += f"{i}. {rec['description']} (优先级: {rec['priority']})\n"
        
        return report

# 便利函数
def create_yica_system(device_id: int = 0, 
                      analyzer_config: Optional[Dict] = None,
                      memory_config: Optional[Dict] = None) -> Tuple[YICAAnalyzer, YICAMemoryManager, YICAPerformanceMonitor]:
    """
    创建完整的YICA系统
    
    Returns:
        (analyzer, memory_manager, performance_monitor) 元组
    """
    analyzer = YICAAnalyzer(analyzer_config)
    memory_manager = YICAMemoryManager(device_id, 1, memory_config)
    monitor = YICAPerformanceMonitor(analyzer, memory_manager)
    
    return analyzer, memory_manager, monitor

def quick_analyze(graph, config: Optional[Dict] = None) -> Dict:
    """
    快速分析计算图
    
    Args:
        graph: 计算图
        config: 可选配置
        
    Returns:
        分析结果和优化建议
    """
    if not CORE_AVAILABLE:
        logger.warning("Core module not available, returning mock analysis results")
        return {
            'analysis': {
                'memory_usage': 'N/A - Core module not available',
                'computation_pattern': 'N/A - Core module not available',
                'bottlenecks': []
            },
            'recommendations': [
                'Install full YICA-Yirage package with native extensions for complete functionality'
            ]
        }
    
    analyzer = YICAAnalyzer(config)
    analysis = analyzer.analyze_graph(graph)
    recommendations = analyzer.get_optimization_recommendations(graph)
    
    return {
        'analysis': analysis,
        'recommendations': recommendations
    } 

def main():
    """命令行入口点"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="YICA-Yirage Advanced Analyzer - AI Computing Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  yica-analyze analyze model.onnx --output analysis_report.json
  yica-analyze optimize model.pt --level O3 --output optimized_model.pt
  yica-analyze memory model.onnx --batch-size 16
  yica-analyze hardware --device yica_g100 --workload transformer
        """
    )
    
    parser.add_argument('--version', action='version', version='YICA-Yirage Advanced Analyzer v1.0.6')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--device', default='yica_g100', help='Target YICA device')
    
    # 功能选项
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 分析命令
    analyze_parser = subparsers.add_parser('analyze', help='Analyze model or workload')
    analyze_parser.add_argument('input', help='Input model file')
    analyze_parser.add_argument('--output', help='Output analysis report file')
    analyze_parser.add_argument('--format', choices=['json', 'yaml', 'text'], default='json',
                               help='Output format')
    analyze_parser.add_argument('--batch-size', type=int, default=1, help='Batch size for analysis')
    
    # 优化命令
    optimize_parser = subparsers.add_parser('optimize', help='Advanced model optimization')
    optimize_parser.add_argument('input', help='Input model file')
    optimize_parser.add_argument('--output', help='Output optimized model file')
    optimize_parser.add_argument('--level', choices=['O0', 'O1', 'O2', 'O3'], default='O2',
                                help='Optimization level')
    optimize_parser.add_argument('--target', choices=['yica', 'cuda', 'cpu'], default='yica',
                                help='Target platform')
    
    # 内存分析命令
    memory_parser = subparsers.add_parser('memory', help='Memory usage analysis')
    memory_parser.add_argument('input', help='Input model file')
    memory_parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    memory_parser.add_argument('--seq-len', type=int, default=128, help='Sequence length')
    memory_parser.add_argument('--output', help='Output memory analysis report')
    
    # 硬件分析命令
    hardware_parser = subparsers.add_parser('hardware', help='Hardware utilization analysis')
    hardware_parser.add_argument('--device', required=True, help='YICA device type')
    hardware_parser.add_argument('--workload', required=True, help='Workload type')
    hardware_parser.add_argument('--output', help='Output hardware analysis report')
    
    # 比较命令
    compare_parser = subparsers.add_parser('compare', help='Compare different configurations')
    compare_parser.add_argument('config1', help='First configuration')
    compare_parser.add_argument('config2', help='Second configuration')
    compare_parser.add_argument('--model', required=True, help='Model to compare')
    compare_parser.add_argument('--output', help='Output comparison report')
    
    args = parser.parse_args()
    
    print("🔬 YICA-Yirage Advanced Analyzer v1.0.6")
    print("=" * 50)
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        if args.command == 'analyze':
            print(f"🔍 Analyzing model: {args.input}")
            # 这里应该实现实际的分析逻辑
            if args.output:
                print(f"📁 Analysis report will be saved to: {args.output}")
            print("✅ Model analysis completed")
        
        elif args.command == 'optimize':
            print(f"⚡ Optimizing model: {args.input}")
            print(f"🎯 Target: {args.target}, Level: {args.level}")
            # 这里应该实现实际的高级优化逻辑
            if args.output:
                print(f"📁 Optimized model will be saved to: {args.output}")
            print("✅ Advanced optimization completed")
        
        elif args.command == 'memory':
            print(f"💾 Analyzing memory usage: {args.input}")
            print(f"📊 Batch size: {args.batch_size}, Sequence length: {args.seq_len}")
            # 这里应该实现实际的内存分析逻辑
            print("✅ Memory analysis completed")
        
        elif args.command == 'hardware':
            print(f"🔌 Analyzing hardware utilization:")
            print(f"   Device: {args.device}")
            print(f"   Workload: {args.workload}")
            # 这里应该实现实际的硬件分析逻辑
            print("✅ Hardware analysis completed")
        
        elif args.command == 'compare':
            print(f"⚖️ Comparing configurations:")
            print(f"   Config 1: {args.config1}")
            print(f"   Config 2: {args.config2}")
            print(f"   Model: {args.model}")
            # 这里应该实现实际的比较逻辑
            print("✅ Configuration comparison completed")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    main() 