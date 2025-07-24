#!/usr/bin/env python3
"""
YICA (存算一体芯片架构) 综合演示

本脚本展示基于Yirage已有例子改进的YICA优化版本，包括：
1. Gated MLP
2. Group Query Attention  
3. RMS Normalization
4. LoRA (Low-Rank Adaptation)

保留原有的非YICA版本，便于性能对比
"""

import argparse
import sys
import os
import time
import torch
import traceback
from typing import Dict, List, Any

# 添加demo目录到Python路径
demo_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, demo_path)

# 导入YICA优化模块
try:
    from demo_yica_gated_mlp import YICAGatedMLP, run_yica_vs_yirage_gqa_comparison as run_gated_mlp
    from demo_yica_group_query_attention import YICAGroupQueryAttention, run_yica_vs_yirage_gqa_comparison
    from demo_yica_rms_norm import YICARMSNorm, run_yica_vs_yirage_rmsnorm_comparison
    from demo_yica_lora import YICALoRA, run_yica_vs_yirage_lora_comparison
except ImportError as e:
    print(f"⚠️  警告: 无法导入部分YICA模块: {e}")
    print("请确保所有YICA演示文件都已创建")

# YICA全局配置
YICA_GLOBAL_CONFIG = {
    'device': 'cuda:0',
    'dtype': torch.float16,
    'enable_profiling': True,
    'num_warmup_runs': 16,
    'num_test_runs': 1000,
    'enable_memory_analysis': True,
    'enable_power_analysis': False,  # 需要特殊硬件支持
}

def print_yica_banner():
    """打印YICA横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                       🧠 YICA 存算一体架构 演示平台 🧠                        ║
    ║                                                                              ║
    ║  基于Yirage已有例子的YICA优化版本                                              ║
    ║  - 保留原始Yirage版本以便对比                                                  ║
    ║  - 展示YICA的CIM阵列并行、SPM内存优化、存算一体计算特性                         ║
    ║  - 支持Gated MLP、Group Query Attention、RMS Norm、LoRA等模块                ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_yica_environment():
    """检查YICA运行环境"""
    print("🔍 检查YICA运行环境...")
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，YICA演示需要CUDA支持")
        return False
    
    print(f"✅ CUDA设备: {torch.cuda.get_device_name()}")
    print(f"✅ CUDA版本: {torch.version.cuda}")
    print(f"✅ PyTorch版本: {torch.__version__}")
    
    # 检查内存
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU内存: {total_memory:.1f}GB")
    
    if total_memory < 8.0:
        print("⚠️  警告: GPU内存较小，某些大矩阵测试可能失败")
    
    # 检查Triton
    try:
        import triton
        print(f"✅ Triton版本: {triton.__version__}")
    except ImportError:
        print("❌ Triton不可用，YICA内核需要Triton支持")
        return False
    
    return True

def run_yica_module_test(module_name: str, test_func, *args, **kwargs) -> Dict[str, Any]:
    """运行单个YICA模块测试"""
    print(f"\n{'='*80}")
    print(f"🧪 测试YICA模块: {module_name}")
    print(f"{'='*80}")
    
    try:
        start_time = time.time()
        results = test_func(*args, **kwargs)
        end_time = time.time()
        
        if results:
            results['test_duration_seconds'] = end_time - start_time
            results['module_name'] = module_name
            results['status'] = 'success'
        else:
            results = {
                'module_name': module_name,
                'status': 'failed',
                'test_duration_seconds': end_time - start_time
            }
        
        print(f"✅ {module_name} 测试完成 ({end_time - start_time:.1f}秒)")
        return results
        
    except Exception as e:
        print(f"❌ {module_name} 测试失败: {e}")
        print(f"错误详情:\n{traceback.format_exc()}")
        return {
            'module_name': module_name,
            'status': 'error',
            'error': str(e),
            'test_duration_seconds': 0
        }

def analyze_yica_results(all_results: List[Dict[str, Any]]):
    """分析YICA测试结果"""
    print(f"\n{'='*80}")
    print("📊 YICA综合性能分析")
    print(f"{'='*80}")
    
    successful_tests = [r for r in all_results if r.get('status') == 'success']
    failed_tests = [r for r in all_results if r.get('status') != 'success']
    
    print(f"\n📈 测试概况:")
    print(f"   ✅ 成功: {len(successful_tests)}")
    print(f"   ❌ 失败: {len(failed_tests)}")
    print(f"   📊 成功率: {len(successful_tests)/(len(all_results))*100:.1f}%")
    
    if successful_tests:
        print(f"\n🚀 YICA加速比分析:")
        total_speedup = 0
        speedup_count = 0
        
        for result in successful_tests:
            module_name = result['module_name']
            
            # 检查不同类型的加速比
            if 'speedup' in result:
                speedup = result['speedup']
                print(f"   {module_name}: {speedup:.2f}x")
                total_speedup += speedup
                speedup_count += 1
            elif 'speedup_fused' in result:
                speedup = result['speedup_fused']
                print(f"   {module_name} (融合): {speedup:.2f}x")
                total_speedup += speedup
                speedup_count += 1
            elif 'speedup_adaptive' in result:
                speedup = result['speedup_adaptive']
                print(f"   {module_name} (自适应): {speedup:.2f}x")
                total_speedup += speedup
                speedup_count += 1
        
        if speedup_count > 0:
            avg_speedup = total_speedup / speedup_count
            print(f"\n🎯 平均YICA加速比: {avg_speedup:.2f}x")
    
    print(f"\n💾 YICA架构特性验证:")
    cim_arrays_used = set()
    spm_sizes_used = set()
    
    for result in successful_tests:
        # 从结果中提取YICA特性信息（需要各个模块提供）
        if hasattr(result, 'yica_config'):
            config = result['yica_config']
            cim_arrays_used.add(config.get('num_cim_arrays', 'unknown'))
            spm_sizes_used.add(config.get('spm_size_kb', 'unknown'))
    
    print(f"   🧠 CIM阵列使用情况: {sorted(cim_arrays_used)}")
    print(f"   💿 SPM大小使用情况: {sorted(spm_sizes_used)}KB")
    
    # 计算总体性能指标
    total_test_time = sum(r.get('test_duration_seconds', 0) for r in all_results)
    print(f"\n⏱️  总测试时间: {total_test_time:.1f}秒")
    
    if failed_tests:
        print(f"\n❌ 失败的测试:")
        for result in failed_tests:
            print(f"   {result['module_name']}: {result.get('error', '未知错误')}")

def export_yica_results(all_results: List[Dict[str, Any]], output_file: str = "yica_results.txt"):
    """导出YICA测试结果"""
    print(f"\n💾 导出结果到 {output_file}...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("YICA (存算一体芯片架构) 测试结果报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"设备信息: {torch.cuda.get_device_name()}\n")
            f.write(f"CUDA版本: {torch.version.cuda}\n")
            f.write(f"PyTorch版本: {torch.__version__}\n\n")
            
            for result in all_results:
                f.write(f"模块: {result['module_name']}\n")
                f.write(f"状态: {result['status']}\n")
                
                if result['status'] == 'success':
                    for key, value in result.items():
                        if key not in ['module_name', 'status']:
                            f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  错误: {result.get('error', '未知')}\n")
                
                f.write("\n" + "-" * 40 + "\n\n")
        
        print(f"✅ 结果已导出到 {output_file}")
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")

def main():
    """YICA综合演示主函数"""
    parser = argparse.ArgumentParser(description="YICA存算一体架构综合演示")
    parser.add_argument('--modules', nargs='+', 
                       choices=['gated_mlp', 'attention', 'rms_norm', 'lora', 'all'],
                       default=['all'],
                       help='要测试的YICA模块')
    parser.add_argument('--export', type=str, default=None,
                       help='导出结果文件名')
    parser.add_argument('--skip-env-check', action='store_true',
                       help='跳过环境检查')
    
    args = parser.parse_args()
    
    # 打印横幅
    print_yica_banner()
    
    # 环境检查
    if not args.skip_env_check:
        if not check_yica_environment():
            print("❌ 环境检查失败，退出")
            sys.exit(1)
    
    # 确定要测试的模块
    if 'all' in args.modules:
        test_modules = ['gated_mlp', 'attention', 'rms_norm', 'lora']
    else:
        test_modules = args.modules
    
    print(f"\n🎯 将测试以下YICA模块: {', '.join(test_modules)}")
    
    # 运行测试
    all_results = []
    
    for module in test_modules:
        if module == 'gated_mlp':
            try:
                result = run_yica_module_test("Gated MLP", run_gated_mlp)
                all_results.append(result)
            except NameError:
                print("⚠️  跳过Gated MLP测试（模块未导入）")
        
        elif module == 'attention':
            try:
                result = run_yica_module_test("Group Query Attention", run_yica_vs_yirage_gqa_comparison)
                all_results.append(result)
            except NameError:
                print("⚠️  跳过Group Query Attention测试（模块未导入）")
        
        elif module == 'rms_norm':
            try:
                result = run_yica_module_test("RMS Normalization", run_yica_vs_yirage_rmsnorm_comparison)
                all_results.append(result)
            except NameError:
                print("⚠️  跳过RMS Norm测试（模块未导入）")
        
        elif module == 'lora':
            try:
                result = run_yica_module_test("LoRA", run_yica_vs_yirage_lora_comparison)
                all_results.append(result)
            except NameError:
                print("⚠️  跳过LoRA测试（模块未导入）")
    
    # 分析结果
    if all_results:
        analyze_yica_results(all_results)
        
        # 导出结果
        if args.export:
            export_yica_results(all_results, args.export)
        
        print(f"\n🎉 YICA综合演示完成！")
        print(f"📚 查看各个模块的详细实现:")
        print(f"   - demo_yica_gated_mlp.py")
        print(f"   - demo_yica_group_query_attention.py") 
        print(f"   - demo_yica_rms_norm.py")
        print(f"   - demo_yica_lora.py")
        print(f"\n💡 这些文件保留了原有的Yirage版本，便于性能对比和学习")
        
    else:
        print("❌ 没有成功的测试结果")
        sys.exit(1)

if __name__ == "__main__":
    main() 