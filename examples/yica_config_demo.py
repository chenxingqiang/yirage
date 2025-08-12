#!/usr/bin/env python3
"""
YICA-Yirage 配置系统演示

展示如何使用增强的全局配置系统来管理YICA设备和各种优化选项
"""

import os
import sys

# 添加yirage到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import yirage
from yirage.core.global_config import (
    global_config, 
    set_yica_device, 
    set_backend,
    enable_verbose,
    print_config,
    YICADeviceType,
    BackendType,
    OptimizationLevel
)


def demo_basic_config():
    """基础配置演示"""
    print("=" * 60)
    print("1. 基础配置演示")
    print("=" * 60)
    
    # 启用详细输出
    enable_verbose()
    
    # 设置YICA设备
    set_yica_device("yica_g100", 0)
    
    # 设置后端
    set_backend("yica")
    
    # 打印当前配置
    print_config()


def demo_yica_device_types():
    """YICA设备类型演示"""
    print("\n" + "=" * 60)
    print("2. YICA设备类型演示")
    print("=" * 60)
    
    device_types = [
        ("yica_g100", "YICA G100 存算一体芯片"),
        ("yica_g200", "YICA G200 升级版"),
        ("yica_cluster", "YICA集群"),
        ("yica_simulator", "YICA模拟器"),
    ]
    
    for device_type, description in device_types:
        print(f"\n设置设备类型: {device_type} ({description})")
        global_config.set_yica_device(device_type, 0)
        
        device_info = global_config.get_yica_device_info()
        effective_backend = global_config.get_effective_backend()
        
        print(f"  设备信息: {device_info['device_type']}")
        print(f"  CIM阵列: {device_info['cim_arrays']}")
        print(f"  SPM大小: {device_info['spm_size_mb']} MB")
        print(f"  带宽: {device_info['bandwidth_gbps']} Gbps")
        print(f"  有效后端: {effective_backend.value}")
        print(f"  YICA可用: {'✓' if global_config.is_yica_available() else '✗'}")


def demo_cluster_configuration():
    """集群配置演示"""
    print("\n" + "=" * 60)
    print("3. YICA集群配置演示")
    print("=" * 60)
    
    # 启用YICA集群
    global_config.enable_yica_cluster(cluster_size=4, node_id=0)
    
    print("集群配置:")
    device_info = global_config.get_yica_device_info()
    print(f"  集群大小: {device_info['cluster_size']}")
    print(f"  节点ID: {device_info['node_id']}")
    print(f"  设备类型: {device_info['device_type']}")


def demo_environment_variables():
    """环境变量配置演示"""
    print("\n" + "=" * 60)
    print("4. 环境变量配置演示")
    print("=" * 60)
    
    print("支持的环境变量:")
    env_vars = [
        ("YICA_DEVICE_TYPE", "YICA设备类型 (yica_g100, yica_g200, etc.)"),
        ("YICA_DEVICE_ID", "YICA设备ID (默认: 0)"),
        ("YICA_CIM_ARRAYS", "CIM阵列数量 (默认: 64)"),
        ("YICA_SPM_SIZE_MB", "SPM大小MB (默认: 256)"),
        ("YICA_BANDWIDTH_GBPS", "带宽Gbps (默认: 400.0)"),
        ("YIRAGE_BACKEND", "默认后端 (yica, cuda, cpu, triton, auto)"),
        ("YIRAGE_OPT_LEVEL", "优化级别 (O0, O1, O2, O3)"),
        ("YIRAGE_VERBOSE", "详细输出 (true/false)"),
        ("YIRAGE_DEBUG", "调试模式 (true/false)"),
        ("YIRAGE_PROFILING", "性能分析 (true/false)"),
    ]
    
    for var, description in env_vars:
        current_value = os.getenv(var, "未设置")
        print(f"  {var:<25} : {description}")
        print(f"  {' ' * 25}   当前值: {current_value}")


def demo_performance_settings():
    """性能设置演示"""
    print("\n" + "=" * 60)
    print("5. 性能设置演示")
    print("=" * 60)
    
    # 启用性能监控
    global_config.enable_profiling = True
    global_config.enable_auto_tuning = True
    global_config.enable_mixed_precision = True
    global_config.enable_graph_fusion = True
    
    print("性能设置:")
    print(f"  性能监控: {'✓' if global_config.enable_profiling else '✗'}")
    print(f"  自动调优: {'✓' if global_config.enable_auto_tuning else '✗'}")
    print(f"  混合精度: {'✓' if global_config.enable_mixed_precision else '✗'}")
    print(f"  图融合: {'✓' if global_config.enable_graph_fusion else '✗'}")
    print(f"  JIT编译: {'✓' if global_config.enable_jit_compilation else '✗'}")
    print(f"  内存池: {'✓' if global_config.enable_memory_pool else '✗'}")
    
    print(f"\n性能配置:")
    print(f"  内存池大小: {global_config.memory_pool_size_mb} MB")
    print(f"  最大并行图: {global_config.max_parallel_graphs}")
    print(f"  编译线程: {global_config.max_compilation_threads}")
    print(f"  缓存目录: {global_config.compilation_cache_dir}")


def demo_distributed_settings():
    """分布式设置演示"""
    print("\n" + "=" * 60)
    print("6. 分布式训练设置演示")
    print("=" * 60)
    
    # 模拟分布式环境
    global_config.distributed_backend = "yccl"
    global_config.distributed_world_size = 4
    global_config.distributed_rank = 0
    global_config.distributed_local_rank = 0
    
    print("分布式配置:")
    print(f"  后端: {global_config.distributed_backend}")
    print(f"  世界大小: {global_config.distributed_world_size}")
    print(f"  全局排名: {global_config.distributed_rank}")
    print(f"  本地排名: {global_config.distributed_local_rank}")


def demo_config_summary():
    """配置摘要演示"""
    print("\n" + "=" * 60)
    print("7. 完整配置摘要")
    print("=" * 60)
    
    # 获取配置摘要
    config_summary = global_config.get_config_summary()
    
    import json
    print("配置摘要 (JSON格式):")
    print(json.dumps(config_summary, indent=2, ensure_ascii=False))


def demo_practical_usage():
    """实际使用场景演示"""
    print("\n" + "=" * 60)
    print("8. 实际使用场景演示")
    print("=" * 60)
    
    # 场景1: 单机YICA G100开发
    print("场景1: 单机YICA G100开发")
    global_config.set_yica_device("yica_g100", 0)
    global_config.set_backend("yica")
    global_config.set_optimization_level("O2")
    global_config.enable_profiling = True
    print(f"  设备: {global_config.yica_device_type.value}")
    print(f"  后端: {global_config.get_effective_backend().value}")
    print(f"  优化: {global_config.optimization_level.value}")
    
    print("\n场景2: YICA集群训练")
    global_config.enable_yica_cluster(cluster_size=8, node_id=0)
    global_config.distributed_backend = "yccl"
    global_config.distributed_world_size = 8
    global_config.enable_auto_tuning = True
    print(f"  集群大小: {global_config.yica_cluster_size}")
    print(f"  分布式后端: {global_config.distributed_backend}")
    print(f"  自动调优: {'✓' if global_config.enable_auto_tuning else '✗'}")
    
    print("\n场景3: 模拟器调试")
    global_config.set_yica_device("yica_simulator", 0)
    global_config.debug = True
    global_config.verbose = True
    global_config.bypass_compile_errors = True
    print(f"  设备: {global_config.yica_device_type.value}")
    print(f"  调试模式: {'✓' if global_config.debug else '✗'}")
    print(f"  忽略编译错误: {'✓' if global_config.bypass_compile_errors else '✗'}")


def main():
    """主演示函数"""
    print("YICA-Yirage 配置系统演示")
    print(f"版本: {yirage.__version__}")
    
    # 运行各个演示
    demo_basic_config()
    demo_yica_device_types()
    demo_cluster_configuration()
    demo_environment_variables()
    demo_performance_settings()
    demo_distributed_settings()
    demo_config_summary()
    demo_practical_usage()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    
    print("\n使用建议:")
    print("1. 根据硬件环境设置正确的YICA设备类型")
    print("2. 使用环境变量进行批量配置")
    print("3. 启用性能监控来优化模型性能")
    print("4. 在集群环境中正确配置分布式参数")
    print("5. 使用调试模式来排查问题")


if __name__ == "__main__":
    main()
