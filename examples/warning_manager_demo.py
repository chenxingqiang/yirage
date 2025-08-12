#!/usr/bin/env python3
"""
YICA-Yirage 警告管理器演示

展示如何使用新的警告管理系统来控制和管理警告信息
"""

import os
import sys

# 添加yirage到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from yirage.utils.warning_manager import (
    warning_manager, 
    warn_dependency_missing,
    warn_cython_unavailable,
    warn_import_failed,
    warn_performance_issue,
    warn_hardware_issue,
    print_dependency_summary,
    set_warning_level,
    WarningLevel
)


def demo_warning_levels():
    """演示不同的警告级别"""
    print("=" * 60)
    print("警告级别演示")
    print("=" * 60)
    
    levels = ["silent", "essential", "normal", "verbose"]
    
    for level in levels:
        print(f"\n🔧 设置警告级别为: {level}")
        set_warning_level(level)
        
        # 重置警告状态以便重新显示
        warning_manager.reset_warnings()
        
        # 测试不同类型的警告
        warn_dependency_missing("test-package", "testing functionality")
        warn_cython_unavailable("test-module")
        warn_import_failed("test-import", True)
        warn_performance_issue("Test performance issue", "Use optimization")
        warn_hardware_issue("Test hardware issue", "Check device")
        
        print(f"   警告级别 {level} 演示完成\n")


def demo_dependency_warnings():
    """演示依赖包警告"""
    print("=" * 60)
    print("依赖包警告演示")
    print("=" * 60)
    
    set_warning_level("normal")
    warning_manager.reset_warnings()
    
    # 模拟各种依赖包缺失情况
    dependencies = [
        ("graphviz", "graph visualization"),
        ("tg4perfetto", "performance tracing"),
        ("torch", "deep learning functionality"),
        ("numpy", "numerical computing"),
        ("z3-solver", "constraint solving"),
    ]
    
    for package, feature in dependencies:
        warn_dependency_missing(package, feature)


def demo_performance_warnings():
    """演示性能相关警告"""
    print("\n" + "=" * 60)
    print("性能警告演示")
    print("=" * 60)
    
    set_warning_level("essential")
    warning_manager.reset_warnings()
    
    performance_issues = [
        ("YICA device not found, falling back to CPU", "Install YICA drivers"),
        ("Using unoptimized matrix multiplication", "Enable YICA CIM acceleration"),
        ("Memory fragmentation detected", "Consider using memory pooling"),
        ("Suboptimal batch size detected", "Use batch size multiple of 64"),
    ]
    
    for issue, suggestion in performance_issues:
        warn_performance_issue(issue, suggestion)


def demo_hardware_warnings():
    """演示硬件相关警告"""
    print("\n" + "=" * 60)
    print("硬件警告演示")
    print("=" * 60)
    
    set_warning_level("essential")
    warning_manager.reset_warnings()
    
    hardware_issues = [
        ("YICA G100 device not detected", "Check device connection"),
        ("SPM memory insufficient", "Reduce model size or increase SPM"),
        ("CIM array utilization low", "Optimize tensor shapes"),
        ("YCCL communication timeout", "Check network configuration"),
    ]
    
    for issue, suggestion in hardware_issues:
        warn_hardware_issue(issue, suggestion)


def demo_cython_warnings():
    """演示Cython模块警告"""
    print("\n" + "=" * 60)
    print("Cython模块警告演示")
    print("=" * 60)
    
    set_warning_level("verbose")
    warning_manager.reset_warnings()
    
    cython_modules = [
        "yica_core",
        "yica_kernels", 
        "yica_optimizer",
        "yica_memory_manager",
    ]
    
    for module in cython_modules:
        warn_cython_unavailable(module)


def demo_import_warnings():
    """演示导入失败警告"""
    print("\n" + "=" * 60)
    print("导入失败警告演示")
    print("=" * 60)
    
    set_warning_level("verbose")
    warning_manager.reset_warnings()
    
    import_failures = [
        ("yica.backend", True),
        ("yica.optimizer", True),
        ("yica.profiler", False),
        ("yica.visualizer", True),
    ]
    
    for module, has_fallback in import_failures:
        warn_import_failed(module, has_fallback)


def demo_dependency_summary():
    """演示依赖包摘要"""
    print("\n" + "=" * 60)
    print("依赖包摘要演示")
    print("=" * 60)
    
    set_warning_level("normal")
    
    print("当前系统的依赖包状态:")
    print_dependency_summary()


def demo_environment_variables():
    """演示环境变量控制"""
    print("\n" + "=" * 60)
    print("环境变量控制演示")
    print("=" * 60)
    
    print("支持的环境变量:")
    env_vars = [
        ("YIRAGE_WARNING_LEVEL", "警告级别 (silent, essential, normal, verbose)"),
        ("YIRAGE_SHOW_DEPENDENCY_SUMMARY", "启动时显示依赖摘要 (true/false)"),
        ("YIRAGE_VERBOSE", "详细输出模式 (true/false)"),
    ]
    
    for var, description in env_vars:
        current_value = os.getenv(var, "未设置")
        print(f"  {var:<30} : {description}")
        print(f"  {' ' * 30}   当前值: {current_value}")
    
    print("\n使用示例:")
    print("  export YIRAGE_WARNING_LEVEL=verbose")
    print("  export YIRAGE_SHOW_DEPENDENCY_SUMMARY=true")
    print("  python your_script.py")


def demo_warning_suppression():
    """演示警告抑制功能"""
    print("\n" + "=" * 60)
    print("警告抑制功能演示")
    print("=" * 60)
    
    set_warning_level("normal")
    warning_manager.reset_warnings()
    
    print("第一次调用 - 显示警告:")
    warn_dependency_missing("example-package", "example functionality")
    
    print("\n第二次调用 - 警告被抑制:")
    warn_dependency_missing("example-package", "example functionality")
    
    print("\n第三次调用 - 仍然被抑制:")
    warn_dependency_missing("example-package", "example functionality")
    
    print("\n不同包的警告 - 会显示:")
    warn_dependency_missing("another-package", "another functionality")


def main():
    """主演示函数"""
    print("YICA-Yirage 警告管理器演示")
    print(f"当前警告级别: {warning_manager._warning_level.name}")
    
    # 运行各个演示
    demo_warning_levels()
    demo_dependency_warnings()
    demo_performance_warnings()
    demo_hardware_warnings()
    demo_cython_warnings()
    demo_import_warnings()
    demo_dependency_summary()
    demo_environment_variables()
    demo_warning_suppression()
    
    print("\n" + "=" * 60)
    print("警告管理器演示完成！")
    print("=" * 60)
    
    print("\n使用建议:")
    print("1. 在生产环境中使用 'essential' 级别，只显示关键警告")
    print("2. 在开发环境中使用 'verbose' 级别，获取详细信息")
    print("3. 在CI/CD中使用 'silent' 级别，避免日志噪音")
    print("4. 使用环境变量进行批量配置")
    print("5. 定期检查依赖包摘要，确保最佳性能")


if __name__ == "__main__":
    main()
