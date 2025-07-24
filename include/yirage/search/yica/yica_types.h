#pragma once

#include <vector>
#include <string>
#include <map>

namespace yirage {
namespace kernel {
class DTensor;
}
}

namespace yirage {
namespace search {
namespace yica {

/**
 * YICA架构配置
 */
struct YICAConfig {
    size_t cim_array_rows = 256;
    size_t cim_array_cols = 256;
    size_t spm_size_per_die = 2 * 1024 * 1024;  // 2MB
    size_t dram_bandwidth = 1024;  // GB/s
    size_t num_cim_dies = 16;
    float cim_frequency = 1000.0f;  // MHz
    
    // 能耗参数
    float cim_energy_per_op = 0.1f;     // pJ per operation
    float spm_energy_per_access = 1.0f; // pJ per access  
    float dram_energy_per_access = 100.0f; // pJ per access
    float communication_latency = 10.0f; // ns
};

/**
 * 并行化机会类型
 */
struct ParallelizationOpportunity {
    enum Type {
        DATA_PARALLEL,     // 数据并行
        MODEL_PARALLEL,    // 模型并行
        PIPELINE_PARALLEL  // 流水线并行
    };
    
    Type type;
    std::vector<kernel::DTensor*> involved_tensors;
    float efficiency_score;  // 并行效率评分 [0-1]
    size_t recommended_parallelism;  // 推荐并行度
    std::string description; // 并行化描述
};

/**
 * YICA分析结果
 */
struct AnalysisResult {
    float cim_friendliness_score;      // CIM操作友好度 [0-1]
    float memory_locality_score;      // 内存局部性评分 [0-1]
    float parallelization_potential;  // 并行化潜力 [0-1]
    std::vector<std::string> bottlenecks;  // 性能瓶颈列表
    std::map<std::string, float> optimization_suggestions;  // 优化建议
    
    // 详细分析数据
    std::vector<ParallelizationOpportunity> parallel_opportunities;
    std::vector<kernel::DTensor*> cim_friendly_ops;
    float estimated_speedup = 1.0f;    // 预估加速比
    float estimated_energy_reduction = 0.0f;  // 预估能耗降低比例
};

/**
 * 操作类型枚举（用于CIM友好度分析）
 */
enum class OpType {
    MATMUL,        // 矩阵乘法
    ELEMENTWISE,   // 元素级运算
    REDUCTION,     // 归约运算
    CONVOLUTION,   // 卷积运算
    POOLING,       // 池化运算
    NORMALIZATION, // 归一化运算
    ACTIVATION,    // 激活函数
    OTHER          // 其他操作
};

/**
 * 数据类型枚举
 */
enum class DataType {
    FP32,    // 32位浮点
    FP16,    // 16位浮点
    BF16,    // Brain Float 16
    INT32,   // 32位整数
    INT16,   // 16位整数
    INT8,    // 8位整数
    INT4,    // 4位整数
    UNKNOWN  // 未知类型
};

/**
 * 内存访问模式
 */
struct MemoryAccessPattern {
    float reuse_distance;      // 重用距离
    float spatial_locality;    // 空间局部性
    float temporal_locality;   // 时间局部性
    size_t working_set_size;   // 工作集大小
    bool spm_friendly;         // 是否适合SPM
};

} // namespace yica
} // namespace search
} // namespace yirage 