#!/usr/bin/env python3
"""
增强算子测试 - 验证yirage扩展算子功能

此测试脚本验证新增的算子是否正常工作，包括：
- 逐元素操作：add, mul, sub, div
- 激活函数：relu, gelu, silu, exp, sqrt
- 归约操作：reduction
- 矩阵操作：matmul
- 规范化：rms_norm
"""

import sys
import os
import time
from datetime import datetime

# 添加yirage路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

def test_enhanced_operators():
    """测试增强的算子功能"""
    print("🧪 yirage增强算子测试")
    print("=" * 60)
    print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        import yirage
        print(f"✅ yirage版本: {yirage.__version__}")
    except ImportError as e:
        print(f"❌ yirage导入失败: {e}")
        return False
    
    # 创建计算图
    graph = yirage.new_kernel_graph()
    print("✅ 计算图创建成功")
    print()
    
    # 创建测试输入
    print("📊 创建测试输入:")
    try:
        A = graph.new_input(dims=(64, 64), dtype="float32")
        B = graph.new_input(dims=(64, 64), dtype="float32")
        print(f"  ✅ 输入A: {A.dims if hasattr(A, 'dims') else 'unknown'}")
        print(f"  ✅ 输入B: {B.dims if hasattr(B, 'dims') else 'unknown'}")
    except Exception as e:
        print(f"  ❌ 输入创建失败: {e}")
        return False
    
    print()
    
    # 测试算子列表
    test_cases = [
        # 二元算子
        ("矩阵乘法", lambda: graph.matmul(A, B)),
        ("逐元素加法", lambda: graph.add(A, B)),
        ("逐元素乘法", lambda: graph.mul(A, B)),
        ("逐元素减法", lambda: graph.sub(A, B)),
        ("逐元素除法", lambda: graph.div(A, B)),
        
        # 一元算子
        ("ReLU激活", lambda: graph.relu(A)),
        ("GELU激活", lambda: graph.gelu(A)),
        ("SiLU激活", lambda: graph.silu(A)),
        ("指数函数", lambda: graph.exp(A)),
        ("平方根", lambda: graph.sqrt(A)),
        
        # 归约操作
        ("归约求和", lambda: graph.reduction(A, dim=0)),
        
        # 规范化
        ("RMS归一化", lambda: graph.rms_norm(A, normalized_shape=[64])),
    ]
    
    print("🧪 算子功能测试:")
    successful_tests = 0
    total_tests = len(test_cases)
    
    for test_name, test_func in test_cases:
        print(f"  🔸 {test_name}...", end=" ")
        start_time = time.time()
        
        try:
            result = test_func()
            duration = (time.time() - start_time) * 1000
            
            # 验证结果
            if result is not None:
                if hasattr(result, 'dims'):
                    print(f"✅ ({duration:.2f}ms) 输出维度: {result.dims}")
                else:
                    print(f"✅ ({duration:.2f}ms)")
                successful_tests += 1
            else:
                print(f"❌ ({duration:.2f}ms) - 返回None")
                
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            print(f"❌ ({duration:.2f}ms) - {str(e)[:50]}...")
    
    print()
    
    # 测试复合操作
    print("🔗 复合操作测试:")
    try:
        print("  🔸 复杂计算图...", end=" ")
        start_time = time.time()
        
        # 构建复杂计算图: (A @ B) + (A * B) - ReLU(A)
        matmul_result = graph.matmul(A, B)
        mul_result = graph.mul(A, B)
        relu_result = graph.relu(A)
        
        add_result = graph.add(matmul_result, mul_result)
        final_result = graph.sub(add_result, relu_result)
        
        graph.mark_output(final_result)
        
        duration = (time.time() - start_time) * 1000
        print(f"✅ ({duration:.2f}ms) 复合操作成功")
        successful_tests += 1
        total_tests += 1
        
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        print(f"❌ ({duration:.2f}ms) - {str(e)[:50]}...")
        total_tests += 1
    
    print()
    
    # 测试图优化
    print("⚡ 图优化测试:")
    try:
        print("  🔸 超优化...", end=" ")
        start_time = time.time()
        
        optimized = graph.superoptimize(backend="cpu")
        
        duration = (time.time() - start_time) * 1000
        print(f"✅ ({duration:.2f}ms) 优化成功")
        successful_tests += 1
        total_tests += 1
        
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        print(f"❌ ({duration:.2f}ms) - {str(e)[:50]}...")
        total_tests += 1
    
    print()
    
    # 测试统计
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    print("📋 测试统计:")
    print(f"  📊 成功率: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
    print(f"  ⏱️  总测试数: {total_tests}")
    
    if success_rate >= 80:
        print("  🎉 测试结果: 优秀")
    elif success_rate >= 60:
        print("  ✅ 测试结果: 良好") 
    else:
        print("  ⚠️  测试结果: 需要改进")
    
    print()
    
    # 算子覆盖率分析
    print("📈 算子覆盖率分析:")
    
    # 基础算子类别
    categories = {
        "矩阵操作": ["matmul"],
        "逐元素二元": ["add", "mul", "sub", "div"],
        "激活函数": ["relu", "gelu", "silu"],
        "数学函数": ["exp", "sqrt"],
        "归约操作": ["reduction"],
        "规范化": ["rms_norm"]
    }
    
    for category, ops in categories.items():
        available_ops = []
        for op in ops:
            if hasattr(graph, op):
                available_ops.append(op)
        
        coverage = len(available_ops) / len(ops) * 100
        print(f"  🔹 {category}: {len(available_ops)}/{len(ops)} ({coverage:.0f}%)")
    
    print()
    
    # 改进建议
    print("💡 改进建议:")
    if success_rate < 100:
        print("  • 完善错误处理机制")
        print("  • 优化算子实现的健壮性")
    
    print("  • 添加更多激活函数 (sigmoid, tanh)")
    print("  • 实现卷积操作 (conv2d)")
    print("  • 添加注意力机制算子")
    print("  • 支持更多数据类型")
    
    print()
    print("🎯 结论: yirage算子扩展测试完成!")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = test_enhanced_operators()
    sys.exit(0 if success else 1)
