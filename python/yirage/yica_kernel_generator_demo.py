#!/usr/bin/env python3
"""
YICA 内核生成器演示

这个演示展示了 YICA 内核生成器的各种功能：
1. 不同类型内核模板的生成
2. CIM 阵列优化和 SPM 内存优化
3. 内核融合和性能分析
4. 自动调优和性能基准测试
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

# 添加 Yirage 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from yirage.yica.config import YICAConfig


@dataclass
class KernelBenchmarkConfig:
    """内核基准测试配置"""
    kernel_types: List[str]
    input_shapes: List[Tuple[int, ...]]
    batch_sizes: List[int]
    precisions: List[str]
    num_iterations: int = 100
    warmup_iterations: int = 10


@dataclass
class KernelGenerationResult:
    """内核生成结果"""
    kernel_name: str
    kernel_type: str
    yis_code: str
    triton_code: str
    
    # 性能预测
    estimated_latency: float
    estimated_throughput: float
    memory_footprint: int
    cim_utilization: float
    spm_utilization: float
    
    # 资源使用
    cim_arrays_used: int
    spm_memory_used: int
    register_count: int
    instruction_count: int
    
    # 优化信息
    optimization_log: List[str]
    generation_successful: bool
    error_message: str = ""


class YICAKernelGeneratorDemo:
    """YICA 内核生成器演示类"""
    
    def __init__(self):
        # YICA 硬件配置
        self.yica_config = YICAConfig(
            num_cim_arrays=32,
            spm_size_per_die=256 * 1024 * 1024,  # 256MB
            dram_size_per_cluster=16 * 1024 * 1024 * 1024,  # 16GB
            enable_quantization=True,
            target_precision="fp16"
        )
        
        # 内核生成器（模拟）
        self.kernel_generator = None  # 在实际实现中会初始化真实的生成器
        
        # 结果存储
        self.generated_kernels = []
        self.benchmark_results = []
        
    def demonstrate_kernel_templates(self):
        """演示各种内核模板"""
        print("🔧 YICA 内核模板演示")
        print("=" * 60)
        
        # 定义测试用的内核配置
        kernel_configs = {
            "CIM_MATMUL": {
                "template_type": "CIM_MATMUL",
                "compute_mode": "CIM_PARALLEL",
                "input_shapes": [(1024, 512), (512, 2048)],
                "cim_arrays": 16,
                "spm_size": 64 * 1024 * 1024,  # 64MB
                "optimizations": ["loop_unroll", "instruction_fusion"]
            },
            
            "CIM_CONV2D": {
                "template_type": "CIM_CONV2D",
                "compute_mode": "CIM_PARALLEL",
                "input_shapes": [(32, 3, 224, 224), (64, 3, 7, 7)],
                "cim_arrays": 24,
                "spm_size": 128 * 1024 * 1024,  # 128MB
                "optimizations": ["register_tiling", "vectorization"]
            },
            
            "CIM_ATTENTION": {
                "template_type": "CIM_ATTENTION",
                "compute_mode": "CIM_PARALLEL",
                "input_shapes": [(32, 512, 768)],  # [batch, seq_len, hidden]
                "num_heads": 12,
                "head_dim": 64,
                "cim_arrays": 32,
                "spm_size": 256 * 1024 * 1024,  # 256MB
                "optimizations": ["instruction_fusion", "prefetch"]
            },
            
            "FUSED_MLP": {
                "template_type": "FUSED_MLP",
                "compute_mode": "PIPELINE_FUSION",
                "input_shapes": [(1024, 768), (768, 3072), (3072, 768)],
                "cim_arrays": 20,
                "spm_size": 128 * 1024 * 1024,  # 128MB
                "optimizations": ["vertical_fusion", "double_buffer"]
            },
            
            "SPM_LAYERNORM": {
                "template_type": "SPM_LAYERNORM",
                "compute_mode": "SPM_OPTIMIZED",
                "input_shapes": [(32, 512, 768)],
                "cim_arrays": 8,
                "spm_size": 32 * 1024 * 1024,  # 32MB
                "optimizations": ["cache_locality", "prefetch"]
            }
        }
        
        # 生成各种类型的内核
        for kernel_name, config in kernel_configs.items():
            print(f"\n🚀 生成 {kernel_name} 内核...")
            
            result = self._generate_kernel_mock(kernel_name, config)
            self.generated_kernels.append(result)
            
            self._print_kernel_info(result)
            
        print(f"\n✅ 成功生成 {len(self.generated_kernels)} 个内核")
    
    def _generate_kernel_mock(self, kernel_name: str, config: Dict[str, Any]) -> KernelGenerationResult:
        """模拟内核生成（实际实现中会调用真实的生成器）"""
        
        # 模拟 YIS 代码生成
        yis_code = self._generate_mock_yis_code(kernel_name, config)
        
        # 模拟 Triton 代码生成
        triton_code = self._generate_mock_triton_code(kernel_name, config)
        
        # 模拟性能预测
        performance = self._predict_mock_performance(config)
        
        # 模拟资源分析
        resources = self._analyze_mock_resources(config)
        
        return KernelGenerationResult(
            kernel_name=f"yica_{kernel_name.lower()}_kernel",
            kernel_type=kernel_name,
            yis_code=yis_code,
            triton_code=triton_code,
            estimated_latency=performance["latency"],
            estimated_throughput=performance["throughput"],
            memory_footprint=performance["memory"],
            cim_utilization=performance["cim_util"],
            spm_utilization=performance["spm_util"],
            cim_arrays_used=resources["cim_arrays"],
            spm_memory_used=resources["spm_memory"],
            register_count=resources["registers"],
            instruction_count=resources["instructions"],
            optimization_log=config.get("optimizations", []),
            generation_successful=True
        )
    
    def _generate_mock_yis_code(self, kernel_name: str, config: Dict[str, Any]) -> str:
        """生成模拟的 YIS 代码"""
        
        yis_templates = {
            "CIM_MATMUL": f"""
// YICA Generated Kernel: {kernel_name}
kernel yica_cim_matmul_kernel {{
  // CIM Array Setup
  cim_init {config['cim_arrays']}
  cim_config array_size 256 256
  cim_enable_pipeline
  cim_set_utilization 0.9
  
  // SPM Memory Setup
  spm_alloc {config['smp_size']}
  spm_strategy locality_first
  spm_enable_prefetch
  spm_enable_double_buffer
  
  // Matrix multiplication loop
  for (int tile_m = 0; tile_m < {config['input_shapes'][0][0]}; tile_m += 32) {{
    for (int tile_n = 0; tile_n < {config['input_shapes'][1][1]}; tile_n += 32) {{
      for (int tile_k = 0; tile_k < {config['input_shapes'][0][1]}; tile_k += 32) {{
        cim_load_tile a_tile, tile_m, tile_k, 32, 32
        cim_load_tile b_tile, tile_k, tile_n, 32, 32
        cim_matmul_tile a_tile, b_tile, c_tile
        cim_accumulate c_result, c_tile
      }}
    }}
  }}
  
  yis_sync
}}""",
            
            "CIM_ATTENTION": f"""
// YICA Generated Kernel: {kernel_name}
kernel yica_cim_attention_kernel {{
  cim_init {config['cim_arrays']}
  spm_alloc {config['spm_size']}
  
  // Q, K, V projection using CIM arrays
  cim_parallel_begin {config.get('num_heads', 12)}
    cim_matmul input_tensor, q_weight, q_projection
    cim_matmul input_tensor, k_weight, k_projection
    cim_matmul input_tensor, v_weight, v_projection
  cim_parallel_end
  
  // Attention score computation
  cim_matmul q_projection, k_projection, attention_scores
  cim_softmax attention_scores
  
  // Output computation
  cim_matmul attention_scores, v_projection, output
  
  yis_sync
}}""",
            
            "FUSED_MLP": f"""
// YICA Generated Kernel: {kernel_name}
kernel yica_fused_mlp_kernel {{
  cim_init {config['cim_arrays']}
  spm_alloc {config['spm_size']}
  
  // Fused MLP computation: Linear -> Activation -> Linear
  spm_load input_tensor, spm_addr_0
  spm_load weight1_tensor, spm_addr_1
  cim_matmul input_tensor, weight1_tensor, hidden_output
  cim_gelu hidden_output  // Fused activation
  spm_load weight2_tensor, spm_addr_2
  cim_matmul hidden_output, weight2_tensor, final_output
  spm_store final_output, spm_addr_3
  
  yis_sync
}}""",
        }
        
        return yis_templates.get(kernel_name, f"// Mock YIS code for {kernel_name}")
    
    def _generate_mock_triton_code(self, kernel_name: str, config: Dict[str, Any]) -> str:
        """生成模拟的 Triton 代码"""
        
        return f"""
import triton
import triton.language as tl

@triton.jit
def yica_{kernel_name.lower()}_kernel(
    input_ptr, output_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 128,
    BLOCK_SIZE_N: tl.constexpr = 128,
    BLOCK_SIZE_K: tl.constexpr = 32,
):
    \"\"\"YICA-optimized Triton kernel for {kernel_name}\"\"\"
    
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Simulate YICA CIM array computation
    # Using {config['cim_arrays']} CIM arrays
    # SPM size: {config['spm_size'] // (1024*1024)}MB
    
    # Compute offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Load and compute (YICA CIM simulation)
    input_ptrs = input_ptr + offs_m[:, None] * N + offs_n[None, :]
    output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    input_data = tl.load(input_ptrs, mask=mask)
    
    # YICA-specific optimizations: {', '.join(config.get('optimizations', []))}
    result = input_data  # Placeholder computation
    
    tl.store(output_ptrs, result, mask=mask)
"""
    
    def _predict_mock_performance(self, config: Dict[str, Any]) -> Dict[str, float]:
        """模拟性能预测"""
        
        # 基于配置参数的简化性能模型
        base_latency = 1.0  # ms
        cim_speedup = min(config['cim_arrays'] / 8.0, 4.0)  # 最多4倍加速
        spm_speedup = (config['spm_size'] / (64 * 1024 * 1024)) * 0.2 + 0.8  # SPM 大小影响
        
        # 优化带来的加速
        opt_speedup = 1.0
        for opt in config.get('optimizations', []):
            if opt in ['loop_unroll', 'instruction_fusion']:
                opt_speedup *= 1.15
            elif opt in ['register_tiling', 'vectorization']:
                opt_speedup *= 1.25
            elif opt in ['prefetch', 'double_buffer']:
                opt_speedup *= 1.1
        
        total_speedup = cim_speedup * spm_speedup * opt_speedup
        estimated_latency = base_latency / total_speedup
        
        return {
            "latency": estimated_latency,
            "throughput": 1000.0 / estimated_latency,  # GFLOPS
            "memory": config['spm_size'],
            "cim_util": min(0.95, 0.6 + config['cim_arrays'] * 0.01),
            "smp_util": min(0.90, 0.5 + (config['spm_size'] / (128 * 1024 * 1024)) * 0.3)
        }
    
    def _analyze_mock_resources(self, config: Dict[str, Any]) -> Dict[str, int]:
        """模拟资源分析"""
        
        return {
            "cim_arrays": config['cim_arrays'],
            "spm_memory": config['spm_size'],
            "registers": config['cim_arrays'] * 32,  # 每个阵列 32 个寄存器
            "instructions": len(config.get('optimizations', [])) * 50 + 100
        }
    
    def _print_kernel_info(self, result: KernelGenerationResult):
        """打印内核信息"""
        
        print(f"  📊 内核: {result.kernel_name}")
        print(f"     类型: {result.kernel_type}")
        print(f"     预估延迟: {result.estimated_latency:.3f} ms")
        print(f"     预估吞吐量: {result.estimated_throughput:.1f} GFLOPS")
        print(f"     内存占用: {result.memory_footprint // (1024*1024)} MB")
        print(f"     CIM 利用率: {result.cim_utilization:.1%}")
        print(f"     SPM 利用率: {result.spm_utilization:.1%}")
        print(f"     使用 CIM 阵列: {result.cim_arrays_used}")
        print(f"     指令数量: {result.instruction_count}")
        print(f"     优化策略: {', '.join(result.optimization_log)}")
    
    def demonstrate_kernel_fusion(self):
        """演示内核融合功能"""
        print("\n🔀 内核融合演示")
        print("=" * 60)
        
        # 选择几个内核进行融合
        if len(self.generated_kernels) < 2:
            print("❌ 需要至少2个内核才能演示融合")
            return
        
        # 融合场景1：MLP 层融合
        print("🧩 场景1: MLP 层融合 (Linear + Activation + Linear)")
        
        mlp_kernels = [k for k in self.generated_kernels if 'MLP' in k.kernel_type or 'MATMUL' in k.kernel_type]
        if len(mlp_kernels) >= 2:
            fused_result = self._simulate_kernel_fusion(mlp_kernels[:2], "VERTICAL_FUSION")
            print(f"  融合前延迟: {sum(k.estimated_latency for k in mlp_kernels[:2]):.3f} ms")
            print(f"  融合后延迟: {fused_result['latency']:.3f} ms")
            print(f"  性能提升: {fused_result['speedup']:.2f}x")
            print(f"  内存节省: {fused_result['memory_savings'] // (1024*1024)} MB")
        
        # 融合场景2：注意力机制融合
        print("\n🧩 场景2: 注意力机制融合 (Q/K/V + Attention + Output)")
        
        attention_kernels = [k for k in self.generated_kernels if 'ATTENTION' in k.kernel_type]
        if attention_kernels:
            fused_result = self._simulate_kernel_fusion([attention_kernels[0]], "HORIZONTAL_FUSION")
            print(f"  融合策略: 水平融合 (并行计算 Q/K/V)")
            print(f"  预估加速比: {fused_result['speedup']:.2f}x")
            print(f"  CIM 阵列利用率提升: {fused_result['utilization_improvement']:.1%}")
    
    def _simulate_kernel_fusion(self, kernels: List[KernelGenerationResult], 
                               fusion_type: str) -> Dict[str, float]:
        """模拟内核融合"""
        
        total_latency = sum(k.estimated_latency for k in kernels)
        total_memory = sum(k.memory_footprint for k in kernels)
        
        # 融合带来的优化
        if fusion_type == "VERTICAL_FUSION":
            # 垂直融合：减少中间结果存储
            fusion_speedup = 1.3 + len(kernels) * 0.1
            memory_savings = total_memory * 0.25  # 节省25%内存
            
        elif fusion_type == "HORIZONTAL_FUSION":
            # 水平融合：提高并行度
            fusion_speedup = 1.2 + len(kernels) * 0.05
            memory_savings = total_memory * 0.15  # 节省15%内存
            
        else:
            fusion_speedup = 1.1
            memory_savings = total_memory * 0.1
        
        fused_latency = total_latency / fusion_speedup
        utilization_improvement = (fusion_speedup - 1.0) * 0.2  # 利用率提升
        
        return {
            "latency": fused_latency,
            "speedup": fusion_speedup,
            "memory_savings": memory_savings,
            "utilization_improvement": utilization_improvement
        }
    
    def run_performance_benchmark(self):
        """运行性能基准测试"""
        print("\n📈 性能基准测试")
        print("=" * 60)
        
        # 定义基准测试配置
        benchmark_config = KernelBenchmarkConfig(
            kernel_types=["CIM_MATMUL", "CIM_ATTENTION", "FUSED_MLP"],
            input_shapes=[(512, 512), (1024, 1024), (2048, 2048)],
            batch_sizes=[1, 8, 32],
            precisions=["fp16", "fp32"],
            num_iterations=50,
            warmup_iterations=5
        )
        
        # 运行基准测试
        for kernel_type in benchmark_config.kernel_types:
            print(f"\n🔬 测试 {kernel_type} 内核...")
            
            kernels = [k for k in self.generated_kernels if k.kernel_type == kernel_type]
            if not kernels:
                print(f"  ⚠️  未找到 {kernel_type} 类型的内核")
                continue
            
            kernel = kernels[0]
            
            # 不同输入大小的性能测试
            for shape in benchmark_config.input_shapes:
                for batch_size in benchmark_config.batch_sizes:
                    for precision in benchmark_config.precisions:
                        
                        # 模拟基准测试
                        result = self._run_mock_benchmark(kernel, shape, batch_size, precision)
                        self.benchmark_results.append(result)
                        
                        print(f"    📊 {shape} x {batch_size} ({precision}): "
                              f"{result['latency']:.3f}ms, "
                              f"{result['throughput']:.1f} GFLOPS, "
                              f"效率: {result['efficiency']:.1%}")
        
        # 生成性能报告
        self._generate_performance_report()
    
    def _run_mock_benchmark(self, kernel: KernelGenerationResult, 
                           shape: Tuple[int, int], batch_size: int, 
                           precision: str) -> Dict[str, float]:
        """模拟基准测试运行"""
        
        # 基于参数的性能模拟
        base_latency = kernel.estimated_latency
        
        # 输入大小影响
        size_factor = (shape[0] * shape[1]) / (1024 * 1024)
        size_latency = base_latency * size_factor
        
        # 批次大小影响
        batch_latency = size_latency * (1.0 + (batch_size - 1) * 0.8)  # 批次并行效率
        
        # 精度影响
        precision_factor = 1.0 if precision == "fp16" else 1.3  # fp32 更慢
        final_latency = batch_latency * precision_factor
        
        # 计算吞吐量
        flops = 2 * shape[0] * shape[1] * batch_size  # 简化的 FLOPS 计算
        throughput = flops / (final_latency / 1000) / 1e9  # GFLOPS
        
        # 效率计算
        theoretical_peak = kernel.estimated_throughput * 2  # 假设的理论峰值
        efficiency = throughput / theoretical_peak
        
        return {
            "kernel_type": kernel.kernel_type,
            "shape": shape,
            "batch_size": batch_size,
            "precision": precision,
            "latency": final_latency,
            "throughput": throughput,
            "efficiency": efficiency,
            "flops": flops
        }
    
    def _generate_performance_report(self):
        """生成性能报告"""
        print("\n📊 性能基准测试报告")
        print("=" * 60)
        
        if not self.benchmark_results:
            print("❌ 没有基准测试结果")
            return
        
        # 按内核类型分组统计
        kernel_stats = {}
        for result in self.benchmark_results:
            kernel_type = result['kernel_type']
            if kernel_type not in kernel_stats:
                kernel_stats[kernel_type] = {
                    'latencies': [],
                    'throughputs': [],
                    'efficiencies': []
                }
            
            kernel_stats[kernel_type]['latencies'].append(result['latency'])
            kernel_stats[kernel_type]['throughputs'].append(result['throughput'])
            kernel_stats[kernel_type]['efficiencies'].append(result['efficiency'])
        
        # 打印统计结果
        for kernel_type, stats in kernel_stats.items():
            print(f"\n🔧 {kernel_type} 统计:")
            print(f"  平均延迟: {np.mean(stats['latencies']):.3f} ± {np.std(stats['latencies']):.3f} ms")
            print(f"  平均吞吐量: {np.mean(stats['throughputs']):.1f} ± {np.std(stats['throughputs']):.1f} GFLOPS")
            print(f"  平均效率: {np.mean(stats['efficiencies']):.1%} ± {np.std(stats['efficiencies']):.1%}")
            print(f"  最佳性能: {np.max(stats['throughputs']):.1f} GFLOPS")
        
        # 保存详细结果
        self._save_benchmark_results()
    
    def _save_benchmark_results(self):
        """保存基准测试结果"""
        
        results_data = {
            'yica_config': asdict(self.yica_config),
            'generated_kernels': [asdict(k) for k in self.generated_kernels],
            'benchmark_results': self.benchmark_results,
            'summary': {
                'total_kernels': len(self.generated_kernels),
                'total_benchmarks': len(self.benchmark_results),
                'avg_cim_utilization': np.mean([k.cim_utilization for k in self.generated_kernels]),
                'avg_spm_utilization': np.mean([k.spm_utilization for k in self.generated_kernels])
            }
        }
        
        with open('yica_kernel_generator_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n💾 详细结果已保存到: yica_kernel_generator_results.json")
    
    def demonstrate_auto_tuning(self):
        """演示自动调优功能"""
        print("\n🎯 自动调优演示")
        print("=" * 60)
        
        # 选择一个内核进行调优
        if not self.generated_kernels:
            print("❌ 没有可调优的内核")
            return
        
        kernel = self.generated_kernels[0]
        print(f"🔧 调优内核: {kernel.kernel_name}")
        
        # 定义调优参数空间
        tuning_space = {
            'cim_arrays': [8, 16, 24, 32],
            'tile_size': [16, 32, 64, 128],
            'spm_allocation': [32, 64, 128, 256],  # MB
            'optimization_level': [1, 2, 3]
        }
        
        print("📊 调优参数空间:")
        for param, values in tuning_space.items():
            print(f"  {param}: {values}")
        
        # 模拟自动调优过程
        best_config = None
        best_performance = 0
        
        print("\n🔍 调优过程:")
        
        for i, cim_arrays in enumerate([8, 16, 24, 32]):
            for j, tile_size in enumerate([32, 64, 128]):
                # 模拟性能测试
                config = {
                    'cim_arrays': cim_arrays,
                    'tile_size': tile_size,
                    'smp_allocation': 64,
                    'optimization_level': 2
                }
                
                # 简化的性能模拟
                performance = self._simulate_tuning_performance(config)
                
                print(f"  配置 {i*3+j+1}: CIM={cim_arrays}, Tile={tile_size} -> "
                      f"{performance:.1f} GFLOPS")
                
                if performance > best_performance:
                    best_performance = performance
                    best_config = config
        
        print(f"\n🏆 最佳配置:")
        for param, value in best_config.items():
            print(f"  {param}: {value}")
        print(f"  最佳性能: {best_performance:.1f} GFLOPS")
        print(f"  相比默认配置提升: {(best_performance / kernel.estimated_throughput - 1) * 100:.1f}%")
    
    def _simulate_tuning_performance(self, config: Dict[str, Any]) -> float:
        """模拟调优性能测试"""
        
        base_performance = 100.0  # GFLOPS
        
        # CIM 阵列数影响
        cim_factor = min(config['cim_arrays'] / 16.0, 2.0)
        
        # 分块大小影响
        tile_factor = 1.0
        if config['tile_size'] == 32:
            tile_factor = 1.1
        elif config['tile_size'] == 64:
            tile_factor = 1.2
        elif config['tile_size'] == 128:
            tile_factor = 1.0
        
        # SPM 分配影响
        spm_factor = 1.0 + (config['smp_allocation'] - 32) / 128 * 0.2
        
        # 优化级别影响
        opt_factor = 1.0 + config['optimization_level'] * 0.1
        
        # 添加一些随机性模拟实际测试的变化
        noise = 1.0 + (np.random.random() - 0.5) * 0.1
        
        return base_performance * cim_factor * tile_factor * spm_factor * opt_factor * noise
    
    def generate_visualization_report(self):
        """生成可视化报告"""
        print("\n📈 生成可视化报告")
        print("=" * 60)
        
        if not self.generated_kernels or not self.benchmark_results:
            print("❌ 没有足够的数据生成可视化报告")
            return
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('YICA 内核生成器性能分析报告', fontsize=16, fontweight='bold')
        
        # 图1: 内核类型性能对比
        kernel_types = [k.kernel_type for k in self.generated_kernels]
        throughputs = [k.estimated_throughput for k in self.generated_kernels]
        
        ax1.bar(kernel_types, throughputs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax1.set_title('各内核类型性能对比')
        ax1.set_ylabel('吞吐量 (GFLOPS)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 图2: CIM 阵列利用率
        cim_utils = [k.cim_utilization for k in self.generated_kernels]
        ax2.hist(cim_utils, bins=10, color='#74B9FF', alpha=0.7, edgecolor='black')
        ax2.set_title('CIM 阵列利用率分布')
        ax2.set_xlabel('利用率')
        ax2.set_ylabel('内核数量')
        
        # 图3: 内存占用分析
        memory_usage = [k.memory_footprint / (1024*1024) for k in self.generated_kernels]  # MB
        ax3.scatter(throughputs, memory_usage, c=cim_utils, cmap='viridis', s=100, alpha=0.7)
        ax3.set_title('性能 vs 内存占用')
        ax3.set_xlabel('吞吐量 (GFLOPS)')
        ax3.set_ylabel('内存占用 (MB)')
        cbar = plt.colorbar(ax3.collections[0], ax=ax3)
        cbar.set_label('CIM 利用率')
        
        # 图4: 基准测试结果趋势
        if self.benchmark_results:
            # 按输入大小分组
            size_groups = {}
            for result in self.benchmark_results:
                size = result['shape'][0] * result['shape'][1]
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append(result['throughput'])
            
            sizes = sorted(size_groups.keys())
            avg_throughputs = [np.mean(size_groups[size]) for size in sizes]
            std_throughputs = [np.std(size_groups[size]) for size in sizes]
            
            ax4.errorbar(sizes, avg_throughputs, yerr=std_throughputs, 
                        marker='o', capsize=5, capthick=2, linewidth=2)
            ax4.set_title('不同输入大小的性能表现')
            ax4.set_xlabel('输入大小 (元素数)')
            ax4.set_ylabel('平均吞吐量 (GFLOPS)')
            ax4.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig('yica_kernel_generator_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 可视化报告已生成: yica_kernel_generator_analysis.png")


def main():
    """主函数"""
    print("🚀 YICA 内核生成器演示启动")
    print("=" * 80)
    
    # 创建演示实例
    demo = YICAKernelGeneratorDemo()
    
    try:
        # 演示各种功能
        demo.demonstrate_kernel_templates()
        demo.demonstrate_kernel_fusion()
        demo.run_performance_benchmark()
        demo.demonstrate_auto_tuning()
        demo.generate_visualization_report()
        
        print("\n" + "=" * 80)
        print("🎉 YICA 内核生成器演示完成!")
        print(f"📊 总计生成 {len(demo.generated_kernels)} 个内核")
        print(f"📈 完成 {len(demo.benchmark_results)} 项基准测试")
        print("💾 详细结果已保存到文件")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  演示被用户中断")
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 