# YICA (存算一体芯片架构) 演示文档

## 概述

本目录包含基于Yirage已有例子改进的YICA优化版本。YICA是一种存算一体芯片架构，通过以下特性提升深度学习计算性能：

- **CIM阵列并行**: 利用多个存算一体(CIM)阵列并行处理不同的计算任务
- **SPM内存优化**: 分层内存管理，减少数据移动开销  
- **存算一体计算**: 直接在存储单元中执行计算，避免数据传输瓶颈
- **智能负载均衡**: 动态分配计算资源，优化硬件利用率

## 文件结构

### YICA优化版本
- `demo_yica_gated_mlp.py` - YICA优化的Gated MLP
- `demo_yica_group_query_attention.py` - YICA优化的Group Query Attention
- `demo_yica_rms_norm.py` - YICA优化的RMS Normalization
- `demo_yica_lora.py` - YICA优化的LoRA (Low-Rank Adaptation)
- `demo_yica_comprehensive.py` - 综合演示脚本

### 原始Yirage版本 (保留用于对比)
- `demo_gated_mlp.py` - 原始Gated MLP
- `demo_group_query_attention.py` - 原始Group Query Attention  
- `demo_rms_norm.py` - 原始RMS Norm
- `demo_lora.py` - 原始LoRA

## 快速开始

### 环境要求
```bash
# 基础依赖
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- Triton 2.0+

# Yirage框架
cd yirage && pip install -e .
```

### 运行单个模块
```bash
# Gated MLP
python demo_yica_gated_mlp.py

# Group Query Attention
python demo_yica_group_query_attention.py

# RMS Normalization
python demo_yica_rms_norm.py

# LoRA
python demo_yica_lora.py
```

### 运行综合演示
```bash
# 测试所有模块
python demo_yica_comprehensive.py

# 测试特定模块
python demo_yica_comprehensive.py --modules gated_mlp attention

# 导出结果
python demo_yica_comprehensive.py --export yica_results.txt
```

## 模块详解

### 1. Gated MLP (`demo_yica_gated_mlp.py`)

**原理**: Gated MLP包含Gate分支和Up分支，通过SiLU激活函数进行门控

**YICA优化**:
- 4个CIM阵列并行处理Gate和Up分支
- SPM内存层次优化减少数据移动
- 存算一体SiLU激活函数计算
- 智能负载均衡

**配置参数**:
```python
YICA_CONFIG = {
    'num_cim_arrays': 4,
    'spm_size_kb': 512,
    'memory_bandwidth_gbps': 1000.0,
    'enable_yica_optimization': True
}
```

**使用示例**:
```python
from demo_yica_gated_mlp import YICAGatedMLP

yica_mlp = YICAGatedMLP()
result = yica_mlp([X, W1, W2])  # X: 输入, W1: Gate权重, W2: Up权重
```

### 2. Group Query Attention (`demo_yica_group_query_attention.py`)

**原理**: Group Query Attention让多个Query头共享Key-Value头，减少内存使用

**YICA优化**:
- 8个CIM阵列并行处理不同的注意力头
- SPM优化Q、K、V数据访问
- 存算一体Softmax计算
- 在线注意力算法减少内存

**配置参数**:
```python
YICA_CONFIG = {
    'num_cim_arrays': 8,
    'spm_size_kb': 1024,
    'num_heads': 32,
    'num_kv_heads': 8  # Group Query
}
```

**使用示例**:
```python
from demo_yica_group_query_attention import YICAGroupQueryAttention

yica_attention = YICAGroupQueryAttention()
result = yica_attention([Q, K, V])  # Q: Query, K: Key, V: Value
```

### 3. RMS Normalization (`demo_yica_rms_norm.py`)

**原理**: RMS Norm通过均方根进行归一化: `x / sqrt(mean(x^2) + eps) * w`

**YICA优化**:
- 2个CIM阵列并行处理不同序列
- SPM优化数据加载和存储
- 存算一体平方根计算
- 残差连接融合版本

**配置参数**:
```python
YICA_CONFIG = {
    'num_cim_arrays': 2,
    'spm_size_kb': 256,
    'enable_vectorization': True,
    'eps': 1e-6
}
```

**使用示例**:
```python
from demo_yica_rms_norm import YICARMSNorm

yica_norm = YICARMSNorm()
result = yica_norm([X, W])  # X: 输入, W: 权重
# 融合残差连接
result = yica_norm([X, W], residual=R)
```

### 4. LoRA (`demo_yica_lora.py`)

**原理**: Low-Rank Adaptation通过低秩矩阵进行参数高效微调: `O = X @ W + alpha * X @ A @ B`

**YICA优化**:
- 6个CIM阵列分别处理主分支和LoRA分支
- SPM优化低秩矩阵计算
- 自适应秩调整机制
- 存算一体缩放融合

**配置参数**:
```python
YICA_CONFIG = {
    'num_cim_arrays': 6,
    'spm_size_kb': 512,
    'low_rank': 64,
    'alpha': 16.0,
    'enable_adaptive_rank': True
}
```

**使用示例**:
```python
from demo_yica_lora import YICALoRA

yica_lora = YICALoRA()
result = yica_lora([X, W, A, B])  # X: 输入, W: 主权重, A&B: LoRA矩阵
# 自适应秩
adaptive_rank = torch.tensor(32, dtype=torch.int32, device='cuda:0')
result = yica_lora([X, W, A, B], adaptive_rank)
```

## 性能对比

运行演示后，您将看到类似以下的性能对比结果：

```
📈 性能对比结果:
   📊 Yirage运行时间: 2.345ms
   ⚡ YICA运行时间: 1.234ms  
   🚀 YICA加速比: 1.90x

🧠 YICA资源利用率:
   💾 CIM阵列数量: 4
   📊 实际TOPS: 12.5
   📈 CIM利用率: 50.0%
   💿 SPM大小: 512KB
```

## YICA架构特性

### 1. CIM阵列并行
- 将不同的计算任务分配到不同的CIM阵列
- 实现指令级并行和数据级并行
- 动态负载均衡优化资源利用

### 2. SPM内存层次
- 分层内存管理：SPM -> HBM -> DDR
- 数据预取和缓存优化
- 减少内存访问延迟

### 3. 存算一体计算
- 直接在存储单元执行计算
- 避免数据在计算单元和存储单元间传输
- 降低功耗和延迟

### 4. 智能优化策略
- 自适应参数调整（如LoRA秩）
- 融合计算减少内存访问
- 向量化处理提升并行度

## 扩展开发

### 添加新的YICA模块

1. **创建新模块文件**:
```python
# demo_yica_your_module.py
import triton
import triton.language as tl

@triton.jit
def yica_your_kernel(...):
    # 实现YICA优化的内核
    pass

class YICAYourModule:
    def __init__(self, config=None):
        self.config = config
    
    def forward(self, inputs):
        # YICA优化的前向传播
        pass
```

2. **添加到综合演示**:
在`demo_yica_comprehensive.py`中添加新模块的导入和测试逻辑

3. **配置YICA参数**:
根据模块特性调整CIM阵列数量、SPM大小等参数

### 优化建议

1. **CIM阵列数量**: 根据并行度需求调整，通常2-8个
2. **SPM大小**: 根据数据重用模式调整，通常256KB-2MB
3. **块大小**: 优化Triton内核的块大小以匹配硬件特性
4. **内存访问模式**: 尽量使用连续访问和数据重用

## 故障排除

### 常见问题

1. **CUDA内存不足**:
   - 减小批次大小或矩阵维度
   - 调整SPM大小配置

2. **Triton编译错误**:
   - 检查Triton版本兼容性
   - 验证内核参数类型

3. **性能不如预期**:
   - 调整CIM阵列配置
   - 优化内存访问模式
   - 检查负载均衡策略

### 调试技巧

1. **使用较小的测试数据**进行快速验证
2. **逐步增加YICA特性**，分别测试效果
3. **对比原始Yirage版本**，确认正确性
4. **使用Profiler工具**分析性能瓶颈

## 参考资料

- [Yirage官方文档](https://yirage.readthedocs.io/)
- [Triton教程](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [YICA架构论文](link-to-yica-paper)
- [存算一体计算综述](link-to-cim-survey)

## 贡献指南

欢迎贡献新的YICA优化模块或改进现有实现：

1. Fork本仓库
2. 创建特性分支
3. 添加新模块或优化现有代码  
4. 添加相应的测试和文档
5. 提交Pull Request

## 许可证

本代码遵循与Yirage相同的许可证。 