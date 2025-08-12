#!/usr/bin/env python3
"""
YICA 强化学习优化器演示程序

这个示例展示了如何使用 YICA 强化学习优化器来优化深度学习计算图：
1. 创建测试计算图
2. 训练强化学习优化器
3. 应用强化学习优化
4. 比较优化效果
5. 保存和加载模型

使用方法:
    python yica_rl_optimization_demo.py [--train] [--model-path MODEL_PATH]
"""

import argparse
import json
import time
import numpy as np
from typing import List, Dict, Any

# 模拟 YICA Python 绑定接口
class MockYICAConfig:
    def __init__(self):
        self.num_cim_arrays = 16
        self.spm_size_kb = 1024
        self.dram_bandwidth_gbps = 1600.0
        self.peak_tops = 100.0

class MockKernelGraph:
    def __init__(self, name: str = "test_graph"):
        self.name = name
        self.operators = []
        self.tensors = []

    def add_matmul(self, m: int, n: int, k: int):
        """添加矩阵乘法操作"""
        op = {
            "type": "matmul",
            "input_shapes": [(m, k), (k, n)],
            "output_shape": (m, n),
            "flops": 2 * m * n * k
        }
        self.operators.append(op)
        return self

    def add_elementwise(self, shape: tuple, op_type: str = "add"):
        """添加逐元素操作"""
        op = {
            "type": f"elementwise_{op_type}",
            "input_shapes": [shape, shape],
            "output_shape": shape,
            "flops": np.prod(shape)
        }
        self.operators.append(op)
        return self

    def add_reduction(self, input_shape: tuple, axis: int):
        """添加归约操作"""
        output_shape = list(input_shape)
        output_shape.pop(axis)
        
        op = {
            "type": "reduction",
            "input_shapes": [input_shape],
            "output_shape": tuple(output_shape),
            "flops": np.prod(input_shape)
        }
        self.operators.append(op)
        return self

    def add_normalization(self, shape: tuple):
        """添加归一化操作"""
        op = {
            "type": "rmsnorm",
            "input_shapes": [shape],
            "output_shape": shape,
            "flops": np.prod(shape) * 5  # 归一化大约需要5个操作每元素
        }
        self.operators.append(op)
        return self

class MockYICABackend:
    def __init__(self, config: MockYICAConfig):
        self.config = config
        self.rl_model_trained = False
        self.optimization_history = []

    def optimize_for_yica(self, graph: MockKernelGraph) -> Dict[str, Any]:
        """传统 YICA 优化"""
        start_time = time.time()
        
        # 模拟优化过程
        total_flops = sum(op["flops"] for op in graph.operators)
        memory_usage = self._estimate_memory_usage(graph)
        
        # 基础优化策略
        speedup = 1.0
        
        # 算子融合优化
        fusion_opportunities = self._count_fusion_opportunities(graph)
        speedup += fusion_opportunities * 0.15
        
        # CIM 并行优化
        cim_utilization = min(len(graph.operators) / self.config.num_cim_arrays, 1.0)
        speedup += cim_utilization * 0.3
        
        # SPM 缓存优化
        spm_hit_rate = min(memory_usage / (self.config.spm_size_kb * 1024), 1.0)
        speedup += (1.0 - spm_hit_rate) * 0.2
        
        optimization_time = time.time() - start_time
        
        result = {
            "used_rl_optimization": False,
            "estimated_speedup": speedup,
            "memory_footprint": memory_usage,
            "optimization_time": optimization_time,
            "optimization_log": [
                "Traditional YICA optimization applied",
                f"Fusion opportunities: {fusion_opportunities}",
                f"CIM utilization: {cim_utilization:.2f}",
                f"SPM hit rate: {spm_hit_rate:.2f}",
                f"Final speedup: {speedup:.2f}x"
            ],
            "yis_kernel_code": self._generate_yis_code(graph),
            "triton_kernel_code": self._generate_triton_code(graph)
        }
        
        self.optimization_history.append(result)
        return result

    def optimize_with_reinforcement_learning(self, graph: MockKernelGraph) -> Dict[str, Any]:
        """强化学习优化"""
        start_time = time.time()
        
        # 提取状态特征
        state_features = self._extract_state_features(graph)
        
        # 生成可能的动作
        possible_actions = self._generate_possible_actions(graph, state_features)
        
        # 选择最优动作序列
        selected_actions = self._select_optimal_actions(state_features, possible_actions)
        
        # 应用动作并计算奖励
        optimized_graph = self._apply_actions(graph, selected_actions)
        reward = self._calculate_reward(graph, optimized_graph)
        
        # 获取传统优化结果作为基准
        traditional_result = self.optimize_for_yica(optimized_graph)
        
        # RL 增强因子
        rl_enhancement = 1.0
        if self.rl_model_trained:
            rl_enhancement = 1.0 + reward * 0.3  # 基于奖励的增强
        
        optimization_time = time.time() - start_time
        
        result = {
            "used_rl_optimization": True,
            "estimated_speedup": traditional_result["estimated_speedup"] * rl_enhancement,
            "memory_footprint": traditional_result["memory_footprint"],
            "optimization_time": optimization_time,
            "optimization_log": [
                "RL optimization applied",
                f"State features: {len(state_features)} dimensions",
                f"Possible actions: {len(possible_actions)}",
                f"Selected actions: {len(selected_actions)}",
                f"RL reward: {reward:.3f}",
                f"RL enhancement factor: {rl_enhancement:.2f}",
            ] + traditional_result["optimization_log"],
            "yis_kernel_code": traditional_result["yis_kernel_code"],
            "triton_kernel_code": traditional_result["triton_kernel_code"],
            "rl_state_features": state_features,
            "rl_actions": selected_actions,
            "rl_reward": reward
        }
        
        self.optimization_history.append(result)
        return result

    def train_rl_optimizer(self, training_graphs: List[MockKernelGraph], episodes: int = 1000):
        """训练强化学习优化器"""
        print(f"Training RL optimizer with {len(training_graphs)} graphs for {episodes} episodes...")
        
        total_reward = 0.0
        best_reward = -float('inf')
        
        for episode in range(episodes):
            # 随机选择训练图
            graph = np.random.choice(training_graphs)
            
            # 执行一个训练 episode
            episode_reward = self._train_episode(graph)
            total_reward += episode_reward
            
            if episode_reward > best_reward:
                best_reward = episode_reward
            
            # 输出训练进度
            if episode % 100 == 0:
                avg_reward = total_reward / (episode + 1)
                print(f"Episode {episode}/{episodes}, "
                      f"Avg Reward: {avg_reward:.3f}, "
                      f"Best Reward: {best_reward:.3f}")
        
        self.rl_model_trained = True
        print(f"Training completed! Final average reward: {total_reward / episodes:.3f}")

    def save_rl_model(self, path: str):
        """保存 RL 模型"""
        model_data = {
            "trained": self.rl_model_trained,
            "config": {
                "num_cim_arrays": self.config.num_cim_arrays,
                "spm_size_kb": self.config.smp_size_kb,
                "dram_bandwidth_gbps": self.config.dram_bandwidth_gbps,
                "peak_tops": self.config.peak_tops
            },
            "optimization_history": self.optimization_history[-100:]  # 保存最近100次优化历史
        }
        
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"RL model saved to {path}")

    def load_rl_model(self, path: str):
        """加载 RL 模型"""
        try:
            with open(path, 'r') as f:
                model_data = json.load(f)
            
            self.rl_model_trained = model_data.get("trained", False)
            self.optimization_history = model_data.get("optimization_history", [])
            
            print(f"RL model loaded from {path}")
            print(f"Model trained: {self.rl_model_trained}")
            print(f"Optimization history: {len(self.optimization_history)} entries")
            
        except Exception as e:
            print(f"Failed to load RL model: {e}")
            raise

    # 私有辅助方法
    def _estimate_memory_usage(self, graph: MockKernelGraph) -> int:
        """估算内存使用量"""
        total_memory = 0
        for op in graph.operators:
            for shape in op["input_shapes"]:
                total_memory += np.prod(shape) * 4  # 假设 FP32
            total_memory += np.prod(op["output_shape"]) * 4
        return total_memory

    def _count_fusion_opportunities(self, graph: MockKernelGraph) -> int:
        """计算融合机会"""
        fusion_count = 0
        for i in range(len(graph.operators) - 1):
            op1, op2 = graph.operators[i], graph.operators[i + 1]
            if self._can_fuse(op1, op2):
                fusion_count += 1
        return fusion_count

    def _can_fuse(self, op1: Dict, op2: Dict) -> bool:
        """判断两个操作是否可以融合"""
        fusible_pairs = [
            ("matmul", "elementwise_add"),
            ("matmul", "elementwise_relu"),
            ("elementwise_add", "elementwise_relu"),
            ("matmul", "rmsnorm")
        ]
        return (op1["type"], op2["type"]) in fusible_pairs

    def _extract_state_features(self, graph: MockKernelGraph) -> List[float]:
        """提取状态特征"""
        features = []
        
        # 基本图统计
        features.append(len(graph.operators))  # 图大小
        
        # 操作类型分布
        op_types = ["matmul", "elementwise_add", "elementwise_relu", "reduction", "rmsnorm"]
        for op_type in op_types:
            count = sum(1 for op in graph.operators if op["type"] == op_type)
            features.append(count / len(graph.operators))
        
        # 计算强度
        total_flops = sum(op["flops"] for op in graph.operators)
        memory_ops = self._estimate_memory_usage(graph) / 4  # 转换为元素数量
        compute_intensity = total_flops / max(memory_ops, 1)
        features.append(min(compute_intensity, 10.0))  # 限制范围
        
        # 内存压力
        memory_pressure = memory_ops / (self.config.spm_size_kb * 256)  # 归一化
        features.append(min(memory_pressure, 5.0))
        
        # 并行度潜力
        parallelizable_ops = sum(1 for op in graph.operators 
                               if op["type"] in ["matmul", "elementwise_add", "elementwise_relu"])
        parallel_potential = parallelizable_ops / len(graph.operators)
        features.append(parallel_potential)
        
        # 融合机会
        fusion_opportunities = self._count_fusion_opportunities(graph)
        fusion_density = fusion_opportunities / max(len(graph.operators) - 1, 1)
        features.append(fusion_density)
        
        # 硬件特征
        features.append(self.config.num_cim_arrays / 16.0)  # 归一化
        features.append(self.config.smp_size_kb / 1024.0)
        
        return features

    def _generate_possible_actions(self, graph: MockKernelGraph, state_features: List[float]) -> List[Dict]:
        """生成可能的动作"""
        actions = []
        
        # 算子融合动作
        for i in range(len(graph.operators) - 1):
            if self._can_fuse(graph.operators[i], graph.operators[i + 1]):
                actions.append({
                    "type": "fusion",
                    "target": f"ops_{i}_{i+1}",
                    "value": 1.0
                })
        
        # 数据重用动作
        if state_features[6] > 2.0:  # 高内存压力
            actions.append({
                "type": "data_reuse",
                "target": "global",
                "value": state_features[6] / 5.0
            })
        
        # 并行化动作
        if state_features[8] > 0.5:  # 高并行潜力
            actions.append({
                "type": "parallelization",
                "target": "global",
                "value": state_features[8]
            })
        
        # 内存布局优化动作
        if state_features[7] > 1.0:  # 内存压力大
            actions.append({
                "type": "layout_transform",
                "target": "tensors",
                "value": min(state_features[7] / 3.0, 1.0)
            })
        
        return actions

    def _select_optimal_actions(self, state_features: List[float], possible_actions: List[Dict]) -> List[Dict]:
        """选择最优动作"""
        if not possible_actions:
            return []
        
        # 简化的动作选择策略
        scored_actions = []
        for action in possible_actions:
            score = self._evaluate_action(state_features, action)
            scored_actions.append((score, action))
        
        # 按分数排序并选择前几个
        scored_actions.sort(reverse=True)
        max_actions = min(3, len(scored_actions))  # 最多选择3个动作
        
        return [action for _, action in scored_actions[:max_actions]]

    def _evaluate_action(self, state_features: List[float], action: Dict) -> float:
        """评估动作价值"""
        base_score = action["value"]
        
        # 根据状态调整分数
        if action["type"] == "fusion":
            # 融合在高计算强度时更有价值
            base_score *= (1.0 + state_features[6] * 0.2)
        elif action["type"] == "parallelization":
            # 并行化在低CIM利用率时更有价值
            cim_utilization = min(state_features[0] / state_features[10], 1.0)
            base_score *= (1.0 + (1.0 - cim_utilization) * 0.3)
        elif action["type"] == "data_reuse":
            # 数据重用在高内存压力时更有价值
            base_score *= (1.0 + state_features[7] * 0.25)
        
        return base_score

    def _apply_actions(self, graph: MockKernelGraph, actions: List[Dict]) -> MockKernelGraph:
        """应用动作到图上"""
        # 创建图的副本
        optimized_graph = MockKernelGraph(graph.name + "_optimized")
        optimized_graph.operators = graph.operators.copy()
        
        # 应用每个动作
        for action in actions:
            if action["type"] == "fusion":
                optimized_graph = self._apply_fusion(optimized_graph, action)
            elif action["type"] == "data_reuse":
                optimized_graph = self._apply_data_reuse(optimized_graph, action)
            elif action["type"] == "parallelization":
                optimized_graph = self._apply_parallelization(optimized_graph, action)
            elif action["type"] == "layout_transform":
                optimized_graph = self._apply_layout_transform(optimized_graph, action)
        
        return optimized_graph

    def _apply_fusion(self, graph: MockKernelGraph, action: Dict) -> MockKernelGraph:
        """应用融合动作"""
        # 简化实现：标记融合
        for op in graph.operators:
            if not hasattr(op, 'optimizations'):
                op['optimizations'] = []
            op['optimizations'].append(f"fusion_{action['target']}")
        return graph

    def _apply_data_reuse(self, graph: MockKernelGraph, action: Dict) -> MockKernelGraph:
        """应用数据重用动作"""
        for op in graph.operators:
            if not hasattr(op, 'optimizations'):
                op['optimizations'] = []
            op['optimizations'].append("data_reuse")
        return graph

    def _apply_parallelization(self, graph: MockKernelGraph, action: Dict) -> MockKernelGraph:
        """应用并行化动作"""
        for op in graph.operators:
            if op["type"] in ["matmul", "elementwise_add", "elementwise_relu"]:
                if not hasattr(op, 'optimizations'):
                    op['optimizations'] = []
                op['optimizations'].append("parallelization")
        return graph

    def _apply_layout_transform(self, graph: MockKernelGraph, action: Dict) -> MockKernelGraph:
        """应用布局转换动作"""
        for op in graph.operators:
            if not hasattr(op, 'optimizations'):
                op['optimizations'] = []
            op['optimizations'].append("layout_transform")
        return graph

    def _calculate_reward(self, original_graph: MockKernelGraph, optimized_graph: MockKernelGraph) -> float:
        """计算奖励"""
        # 计算优化前后的性能指标
        original_perf = self._estimate_performance(original_graph)
        optimized_perf = self._estimate_performance(optimized_graph)
        
        # 性能提升奖励
        perf_reward = (optimized_perf - original_perf) / max(original_perf, 1.0)
        
        # 内存效率奖励
        original_memory = self._estimate_memory_usage(original_graph)
        optimized_memory = self._estimate_memory_usage(optimized_graph)
        memory_reward = (original_memory - optimized_memory) / max(original_memory, 1.0)
        
        # 总奖励
        total_reward = 0.7 * perf_reward + 0.3 * memory_reward
        
        return total_reward

    def _estimate_performance(self, graph: MockKernelGraph) -> float:
        """估算图性能"""
        base_perf = sum(op["flops"] for op in graph.operators)
        
        # 考虑优化效果
        optimization_bonus = 0.0
        for op in graph.operators:
            if 'optimizations' in op:
                optimization_bonus += len(op['optimizations']) * 0.1 * base_perf / len(graph.operators)
        
        return base_perf + optimization_bonus

    def _train_episode(self, graph: MockKernelGraph) -> float:
        """训练一个 episode"""
        # 简化的训练逻辑
        state_features = self._extract_state_features(graph)
        possible_actions = self._generate_possible_actions(graph, state_features)
        
        if not possible_actions:
            return 0.0
        
        # 随机选择动作（探索）
        action = np.random.choice(possible_actions)
        optimized_graph = self._apply_actions(graph, [action])
        reward = self._calculate_reward(graph, optimized_graph)
        
        return reward

    def _generate_yis_code(self, graph: MockKernelGraph) -> str:
        """生成 YIS 汇编代码"""
        code = f"// YIS Kernel for {graph.name}\n"
        code += ".kernel yica_optimized_kernel {\n"
        
        for i, op in enumerate(graph.operators):
            code += f"    // Operation {i}: {op['type']}\n"
            if op["type"] == "matmul":
                code += "    yis.cim.matmul(input_a, input_b, output);\n"
            elif "elementwise" in op["type"]:
                code += f"    yis.cim.elementwise_{op['type'].split('_')[1]}(input, output);\n"
            elif op["type"] == "reduction":
                code += "    yis.cim.reduce(input, output);\n"
            elif op["type"] == "rmsnorm":
                code += "    yis.cim.rmsnorm(input, output);\n"
            
            # 添加优化标记
            if 'optimizations' in op:
                for opt in op['optimizations']:
                    code += f"    // Optimization: {opt}\n"
        
        code += "    yis.control.end();\n"
        code += "}\n"
        
        return code

    def _generate_triton_code(self, graph: MockKernelGraph) -> str:
        """生成 Triton 内核代码"""
        code = "import triton\nimport triton.language as tl\n\n"
        code += "@triton.jit\n"
        code += "def yica_optimized_kernel(\n"
        code += "    input_ptr, output_ptr,\n"
        code += "    M, N, K,\n"
        code += f"    CIM_ARRAYS: tl.constexpr = {self.config.num_cim_arrays},\n"
        code += f"    SPM_SIZE: tl.constexpr = {self.config.smp_size_kb}\n"
        code += "):\n"
        code += "    # YICA-optimized computation\n"
        code += "    pid = tl.program_id(axis=0)\n"
        code += "    cim_id = pid % CIM_ARRAYS\n"
        code += "    \n"
        code += "    # Optimized computation logic\n"
        
        for i, op in enumerate(graph.operators):
            code += f"    # {op['type']} operation\n"
            if 'optimizations' in op:
                code += f"    # Applied optimizations: {', '.join(op['optimizations'])}\n"
        
        code += "    # Store results\n"
        code += "    tl.store(output_ptr, result)\n"
        
        return code

def create_test_graphs() -> List[MockKernelGraph]:
    """创建测试图集合"""
    graphs = []
    
    # 1. 简单的 MatMul + ReLU 图
    graph1 = MockKernelGraph("matmul_relu") \
        .add_matmul(1024, 512, 256) \
        .add_elementwise((1024, 512), "relu")
    graphs.append(graph1)
    
    # 2. 复杂的 Transformer 层
    graph2 = MockKernelGraph("transformer_layer") \
        .add_matmul(512, 2048, 512) \
        .add_elementwise((512, 2048), "add") \
        .add_normalization((512, 2048)) \
        .add_matmul(512, 512, 2048) \
        .add_elementwise((512, 512), "relu") \
        .add_matmul(512, 2048, 512)
    graphs.append(graph2)
    
    # 3. 卷积后处理图
    graph3 = MockKernelGraph("conv_postprocess") \
        .add_elementwise((64, 64, 256), "add") \
        .add_elementwise((64, 64, 256), "relu") \
        .add_reduction((64, 64, 256), 2) \
        .add_normalization((64, 64))
    graphs.append(graph3)
    
    # 4. 大规模矩阵计算图
    graph4 = MockKernelGraph("large_matmul_chain")
    for i in range(5):
        graph4.add_matmul(2048, 2048, 2048)
        if i < 4:  # 不在最后一个操作后添加激活
            graph4.add_elementwise((2048, 2048), "relu")
    graphs.append(graph4)
    
    return graphs

def benchmark_optimization(backend: MockYICABackend, graphs: List[MockKernelGraph]):
    """基准测试优化性能"""
    print("\n" + "="*60)
    print("OPTIMIZATION PERFORMANCE BENCHMARK")
    print("="*60)
    
    traditional_times = []
    rl_times = []
    traditional_speedups = []
    rl_speedups = []
    
    for graph in graphs:
        print(f"\nTesting graph: {graph.name} ({len(graph.operators)} operators)")
        
        # 测试传统优化
        start_time = time.time()
        trad_result = backend.optimize_for_yica(graph)
        trad_time = time.time() - start_time
        traditional_times.append(trad_time)
        traditional_speedups.append(trad_result["estimated_speedup"])
        
        # 测试 RL 优化
        start_time = time.time()
        rl_result = backend.optimize_with_reinforcement_learning(graph)
        rl_time = time.time() - start_time
        rl_times.append(rl_time)
        rl_speedups.append(rl_result["estimated_speedup"])
        
        print(f"  Traditional: {trad_time:.3f}s, Speedup: {trad_result['estimated_speedup']:.2f}x")
        print(f"  RL:          {rl_time:.3f}s, Speedup: {rl_result['estimated_speedup']:.2f}x")
        print(f"  RL vs Trad:  {rl_result['estimated_speedup']/trad_result['estimated_speedup']:.2f}x improvement")
    
    # 总结统计
    print(f"\n{'SUMMARY':<20} {'Traditional':<15} {'RL':<15} {'Improvement':<15}")
    print("-" * 65)
    print(f"{'Avg Time (s)':<20} {np.mean(traditional_times):<15.3f} {np.mean(rl_times):<15.3f} {np.mean(rl_times)/np.mean(traditional_times):<15.2f}")
    print(f"{'Avg Speedup (x)':<20} {np.mean(traditional_speedups):<15.2f} {np.mean(rl_speedups):<15.2f} {np.mean(rl_speedups)/np.mean(traditional_speedups):<15.2f}")
    print(f"{'Max Speedup (x)':<20} {np.max(traditional_speedups):<15.2f} {np.max(rl_speedups):<15.2f} {np.max(rl_speedups)/np.max(traditional_speedups):<15.2f}")

def main():
    parser = argparse.ArgumentParser(description="YICA RL Optimization Demo")
    parser.add_argument("--train", action="store_true", help="Train the RL optimizer")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--model-path", type=str, default="yica_rl_model.json", help="Path to save/load RL model")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    
    args = parser.parse_args()
    
    print("YICA Reinforcement Learning Optimization Demo")
    print("=" * 50)
    
    # 初始化配置和后端
    config = MockYICAConfig()
    backend = MockYICABackend(config)
    
    print(f"YICA Configuration:")
    print(f"  CIM Arrays: {config.num_cim_arrays}")
    print(f"  SPM Size: {config.smp_size_kb} KB")
    print(f"  DRAM Bandwidth: {config.dram_bandwidth_gbps} GB/s")
    print(f"  Peak Performance: {config.peak_tops} TOPS")
    
    # 创建测试图
    test_graphs = create_test_graphs()
    print(f"\nCreated {len(test_graphs)} test graphs")
    
    # 训练模式
    if args.train:
        print(f"\nTraining RL optimizer for {args.episodes} episodes...")
        backend.train_rl_optimizer(test_graphs, args.episodes)
        backend.save_rl_model(args.model_path)
    else:
        # 尝试加载已有模型
        try:
            backend.load_rl_model(args.model_path)
        except:
            print(f"No existing model found at {args.model_path}, using untrained model")
    
    # 基准测试
    if args.benchmark:
        benchmark_optimization(backend, test_graphs)
    else:
        # 简单演示
        print(f"\nDemonstrating optimization on {test_graphs[0].name}...")
        
        # 传统优化
        trad_result = backend.optimize_for_yica(test_graphs[0])
        print(f"\nTraditional Optimization Result:")
        print(f"  Speedup: {trad_result['estimated_speedup']:.2f}x")
        print(f"  Memory: {trad_result['memory_footprint']} bytes")
        print(f"  Time: {trad_result['optimization_time']:.3f}s")
        
        # RL 优化
        rl_result = backend.optimize_with_reinforcement_learning(test_graphs[0])
        print(f"\nRL Optimization Result:")
        print(f"  Speedup: {rl_result['estimated_speedup']:.2f}x")
        print(f"  Memory: {rl_result['memory_footprint']} bytes")
        print(f"  Time: {rl_result['optimization_time']:.3f}s")
        print(f"  RL Enhancement: {rl_result['estimated_speedup']/trad_result['estimated_speedup']:.2f}x")
        
        if 'rl_state_features' in rl_result:
            print(f"  State Features: {len(rl_result['rl_state_features'])} dimensions")
            print(f"  Actions Applied: {len(rl_result['rl_actions'])}")
            print(f"  RL Reward: {rl_result['rl_reward']:.3f}")
    
    print(f"\nDemo completed! Model saved to {args.model_path}")

if __name__ == "__main__":
    main()
