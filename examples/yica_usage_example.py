#!/usr/bin/env python3
"""
YICA功能使用示例

展示如何使用新添加的YICA Python绑定功能
"""

import yirage
import numpy as np

def basic_yica_analyzer_example():
    """基础YICA分析器使用示例"""
    print("🔍 基础YICA分析器使用示例")
    print("-" * 40)
    
    # 创建YICA分析器
    analyzer = yirage.YICAAnalyzer({
        'cim_array_rows': 256,
        'cim_array_cols': 256,
        'spm_size_per_die': 4 * 1024 * 1024,  # 4MB
        'num_cim_dies': 16,
        'cim_frequency': 1200.0  # 1.2GHz
    })
    
    # 创建一个简单的计算图
    graph = yirage.new_kernel_graph()
    
    # 添加输入张量
    input1 = graph.new_input(dims=(1024, 1024), dtype=yirage.float16)
    input2 = graph.new_input(dims=(1024, 1024), dtype=yirage.float16)
    
    # 添加矩阵乘法操作
    matmul_out = graph.matmul(input1, input2)
    
    # 添加激活函数
    relu_out = graph.relu(matmul_out)
    
    # 标记输出
    graph.mark_output(relu_out)
    
    try:
        # 分析计算图
        analysis = analyzer.analyze_graph(graph)
        
        print(f"CIM友好度评分: {analysis['cim_friendliness_score']:.3f}")
        print(f"内存局部性评分: {analysis['memory_locality_score']:.3f}")
        print(f"并行化潜力: {analysis['parallelization_potential']:.3f}")
        print(f"预估加速比: {analysis['estimated_speedup']:.2f}x")
        print(f"预估能耗降低: {analysis['estimated_energy_reduction']:.1%}")
        
        if analysis['bottlenecks']:
            print("性能瓶颈:")
            for bottleneck in analysis['bottlenecks']:
                print(f"  - {bottleneck}")
        
        # 获取优化建议
        recommendations = analyzer.get_optimization_recommendations(graph)
        if recommendations:
            print("\n优化建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['description']}")
                print(f"     优先级: {rec['priority']}, 预期收益: {rec['expected_benefit']:.1%}")
                print(f"     实现提示: {rec['implementation_hint']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return False

def yica_memory_manager_example():
    """YICA内存管理器使用示例"""
    print("\n💾 YICA内存管理器使用示例")
    print("-" * 40)
    
    # 创建内存管理器
    memory_manager = yirage.YICAMemoryManager(
        device_id=0,
        num_devices=1,
        config={
            'register_file_size': 64 * 1024,  # 64KB
            'spm_size_per_die': 256 * 1024 * 1024,  # 256MB
            'dram_total_size': 16 * 1024 * 1024 * 1024,  # 16GB
            'allocation_strategy': 6,  # YICA_OPTIMIZED
            'enable_memory_coalescing': True,
            'enable_prefetching': True,
            'enable_spm_caching': True
        }
    )
    
    try:
        # 测试内存分配
        print("测试内存分配:")
        
        # 在DRAM中分配内存
        dram_ptr = memory_manager.allocate(1024 * 1024, memory_manager.DRAM)  # 1MB
        print(f"  DRAM分配: 0x{dram_ptr:x}")
        
        # 在SPM中分配内存
        spm_ptr = memory_manager.allocate(64 * 1024, memory_manager.SPM)  # 64KB
        print(f"  SPM分配: 0x{spm_ptr:x}")
        
        # 智能分配（YICA优化）
        smart_result = memory_manager.smart_allocate(128 * 1024, memory_manager.SPM)
        if smart_result['allocation_successful']:
            print(f"  智能分配成功: 0x{smart_result['ptr']:x}")
            print(f"    实际分配级别: {smart_result['allocated_level']}")
            print(f"    分配效率: {smart_result['allocation_efficiency']:.2%}")
        
        # 测试数据提升
        print("\n测试数据提升:")
        success = memory_manager.promote_to_spm(dram_ptr, 1024)
        print(f"  数据提升到SPM: {'成功' if success else '失败'}")
        
        # 测试预取
        success = memory_manager.prefetch(dram_ptr, 2048)
        print(f"  数据预取: {'成功' if success else '失败'}")
        
        # 测试带宽测量
        print("\n内存带宽测量:")
        for level, name in [(0, 'Register'), (1, 'SPM'), (2, 'DRAM')]:
            bandwidth = memory_manager.measure_bandwidth(level)
            print(f"  {name}: {bandwidth:.1f} GB/s")
        
        # 获取内存统计
        print("\n内存统计摘要:")
        stats = memory_manager.get_summary_statistics()
        print(f"  总分配次数: {stats['total_allocations']}")
        print(f"  SPM缓存命中率: {stats['spm_cache_hit_rate']:.2%}")
        print(f"  最大碎片化率: {stats['fragmentation_ratio']:.2%}")
        
        # 内存优化
        print("\n执行内存优化:")
        optimization = memory_manager.optimize_memory_usage()
        if optimization['recommendations']:
            print("  优化建议:")
            for rec in optimization['recommendations']:
                print(f"    - {rec['description']} (优先级: {rec['priority']})")
        
        # 释放内存
        memory_manager.deallocate(dram_ptr, memory_manager.DRAM)
        memory_manager.deallocate(spm_ptr, memory_manager.SPM)
        print("\n内存释放完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 内存管理测试失败: {e}")
        return False

def complete_yica_system_example():
    """完整YICA系统使用示例"""
    print("\n🌟 完整YICA系统使用示例")
    print("-" * 40)
    
    try:
        # 创建完整的YICA系统
        analyzer, memory_manager, monitor = yirage.create_yica_system(
            device_id=0,
            analyzer_config={
                'cim_array_rows': 512,
                'cim_array_cols': 512,
                'spm_size_per_die': 8 * 1024 * 1024,  # 8MB
                'num_cim_dies': 32
            },
            memory_config={
                'register_file_size': 128 * 1024,  # 128KB
                'spm_size_per_die': 512 * 1024 * 1024,  # 512MB
                'enable_spm_caching': True,
                'enable_prefetching': True
            }
        )
        
        print("✅ 完整YICA系统创建成功")
        
        # 创建一个更复杂的计算图
        graph = yirage.new_kernel_graph()
        
        # 多层神经网络示例
        input_tensor = graph.new_input(dims=(1024, 512), dtype=yirage.float16)
        weight1 = graph.new_input(dims=(512, 256), dtype=yirage.float16)
        weight2 = graph.new_input(dims=(256, 128), dtype=yirage.float16)
        
        # 第一层
        mm1 = graph.matmul(input_tensor, weight1)
        relu1 = graph.relu(mm1)
        
        # 第二层
        mm2 = graph.matmul(relu1, weight2)
        output = graph.relu(mm2)
        
        graph.mark_output(output)
        
        # 性能监控
        print("\n执行性能监控:")
        monitoring_result = monitor.monitor_execution(graph, duration=0.1)  # 假设执行时间100ms
        
        print(f"  综合性能评分: {monitoring_result['performance_score']:.3f}")
        print(f"  CIM友好度: {monitoring_result['graph_analysis']['cim_friendliness_score']:.3f}")
        print(f"  内存局部性: {monitoring_result['graph_analysis']['memory_locality_score']:.3f}")
        
        # 生成性能报告
        print("\n性能报告:")
        report = monitor.generate_report()
        print(report)
        
        return True
        
    except Exception as e:
        print(f"❌ 完整系统测试失败: {e}")
        return False

def quick_analysis_example():
    """快速分析示例"""
    print("\n⚡ 快速分析示例")
    print("-" * 40)
    
    try:
        # 创建简单计算图
        graph = yirage.new_kernel_graph()
        input1 = graph.new_input(dims=(512, 512), dtype=yirage.float32)
        input2 = graph.new_input(dims=(512, 512), dtype=yirage.float32)
        
        # 复杂的计算序列
        add_result = graph.add(input1, input2)
        mul_result = graph.mul(add_result, input1)
        sqrt_result = graph.sqrt(mul_result)
        output = graph.exp(sqrt_result)
        
        graph.mark_output(output)
        
        # 快速分析
        result = yirage.quick_analyze(graph, {
            'cim_array_rows': 256,
            'spm_size_per_die': 2 * 1024 * 1024
        })
        
        print("快速分析结果:")
        analysis = result['analysis']
        print(f"  CIM友好度: {analysis['cim_friendliness_score']:.3f}")
        print(f"  预估加速比: {analysis['estimated_speedup']:.2f}x")
        
        print("\n优化建议:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec['description']} (优先级: {rec['priority']})")
        
        return True
        
    except Exception as e:
        print(f"❌ 快速分析失败: {e}")
        return False

def main():
    """主函数"""
    print("🎯 YICA功能使用示例")
    print("=" * 50)
    
    examples = [
        ("基础分析器", basic_yica_analyzer_example),
        ("内存管理器", yica_memory_manager_example),
        ("完整系统", complete_yica_system_example),
        ("快速分析", quick_analysis_example)
    ]
    
    success_count = 0
    
    for name, example_func in examples:
        print(f"\n🚀 运行{name}示例...")
        try:
            if example_func():
                success_count += 1
                print(f"✅ {name}示例运行成功")
            else:
                print(f"❌ {name}示例运行失败")
        except Exception as e:
            print(f"❌ {name}示例异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 示例完成: {success_count}/{len(examples)} 成功")
    
    if success_count == len(examples):
        print("🎉 所有YICA功能示例运行成功！")
        print("\n📚 更多用法请参考:")
        print("  - yirage.YICAAnalyzer: YICA架构分析")
        print("  - yirage.YICAMemoryManager: 三级内存管理")
        print("  - yirage.YICAPerformanceMonitor: 性能监控")
        print("  - yirage.create_yica_system(): 创建完整系统")
        print("  - yirage.quick_analyze(): 快速分析")
    else:
        print("⚠️  部分示例失败，可能需要先构建C++库")

if __name__ == "__main__":
    main() 