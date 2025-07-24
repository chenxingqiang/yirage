#pragma once

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <functional>
#include "yirage/kernel/graph.h"
#include "yirage/kernel/operator.h"
#include "yirage/transpiler/transpiler.h"
#include "yirage/yica/config.h"
#include "yirage/yica/yis_instruction_set.h"

namespace yirage {
namespace yica {

// YICA 内核模板类型
enum class YICAKernelTemplate {
    CIM_MATMUL,           // CIM 阵列矩阵乘法
    CIM_CONV2D,           // CIM 阵列卷积
    CIM_ATTENTION,        // CIM 阵列注意力机制
    SPM_REDUCTION,        // SPM 优化的归约操作
    SPM_ELEMENTWISE,      // SPM 优化的逐元素操作
    SPM_LAYERNORM,        // SPM 优化的层归一化
    FUSED_MLP,            // 融合 MLP 计算
    FUSED_ATTENTION,      // 融合注意力计算
    FUSED_CONV_RELU,      // 融合卷积+ReLU
    FUSED_MATMUL_BIAS,    // 融合矩阵乘法+偏置
    CUSTOM_YICA           // 自定义 YICA 内核
};

// YICA 计算模式
enum class YICAComputeMode {
    CIM_PARALLEL,         // CIM 阵列并行模式
    SPM_OPTIMIZED,        // SPM 内存优化模式
    PIPELINE_FUSION,      // 流水线融合模式
    TENSOR_CORE_LIKE,     // 类 Tensor Core 模式
    MIXED_PRECISION,      // 混合精度模式
    QUANTIZED_COMPUTE     // 量化计算模式
};

// YICA 内核配置
struct YICAKernelConfig {
    YICAKernelTemplate template_type;
    YICAComputeMode compute_mode;
    
    // CIM 阵列配置
    struct CIMConfig {
        int num_arrays;               // 使用的 CIM 阵列数量
        int array_size_x;             // 阵列 X 维度
        int array_size_y;             // 阵列 Y 维度
        bool enable_pipelining;       // 启用流水线
        float utilization_target;     // 目标利用率
    } cim_config;
    
    // SPM 内存配置
    struct SPMConfig {
        size_t allocation_size;       // SPM 分配大小
        std::string allocation_strategy; // 分配策略
        bool enable_prefetch;         // 启用预取
        bool enable_double_buffer;    // 启用双缓冲
        int cache_line_size;          // 缓存行大小
    } spm_config;
    
    // 优化配置
    struct OptimizationConfig {
        bool enable_loop_unroll;      // 启用循环展开
        bool enable_vectorization;    // 启用向量化
        bool enable_instruction_fusion; // 启用指令融合
        bool enable_register_tiling;  // 启用寄存器分块
        int tile_size_x;              // X 方向分块大小
        int tile_size_y;              // Y 方向分块大小
    } opt_config;
    
    // 数据类型和精度
    std::string input_precision;     // 输入精度
    std::string compute_precision;   // 计算精度
    std::string output_precision;    // 输出精度
    
    // 调试和性能
    bool enable_debug_output;        // 启用调试输出
    bool enable_performance_counters; // 启用性能计数器
};

// YICA 内核生成结果
struct YICAKernelGenerationResult {
    std::string kernel_name;         // 内核名称
    std::string yis_code;            // 生成的 YIS 代码
    std::string triton_code;         // 生成的 Triton 代码
    std::string cuda_code;           // 生成的 CUDA 代码 (可选)
    
    // 性能预测
    struct PerformancePrediction {
        float estimated_latency;      // 预估延迟 (ms)
        float estimated_throughput;   // 预估吞吐量 (GFLOPS)
        size_t memory_footprint;      // 内存占用 (bytes)
        float cim_utilization;        // CIM 利用率
        float spm_utilization;        // SPM 利用率
    } performance;
    
    // 资源使用情况
    struct ResourceUsage {
        int cim_arrays_used;          // 使用的 CIM 阵列数
        size_t spm_memory_used;       // 使用的 SPM 内存
        int register_count;           // 寄存器使用数量
        int instruction_count;        // 指令数量
    } resources;
    
    // 元数据
    std::map<std::string, std::string> metadata;
    std::vector<std::string> optimization_log;
    bool generation_successful;
    std::string error_message;
};

// YICA 内核生成器
class YICAKernelGenerator {
public:
    explicit YICAKernelGenerator(const YICAConfig& config);
    ~YICAKernelGenerator() = default;
    
    // 主要接口：生成 YICA 优化内核
    YICAKernelGenerationResult generate_kernel(
        const kernel::KNOperator* op,
        const YICAKernelConfig& kernel_config
    );
    
    // 批量生成内核
    std::vector<YICAKernelGenerationResult> generate_kernels(
        const kernel::Graph& graph,
        const std::map<std::string, YICAKernelConfig>& configs
    );
    
    // 模板内核生成
    YICAKernelGenerationResult generate_template_kernel(
        YICAKernelTemplate template_type,
        const std::vector<int>& input_shapes,
        const YICAKernelConfig& config
    );
    
    // 自定义内核生成
    YICAKernelGenerationResult generate_custom_kernel(
        const std::string& kernel_spec,
        const YICAKernelConfig& config
    );
    
    // 内核优化
    YICAKernelGenerationResult optimize_existing_kernel(
        const std::string& existing_kernel,
        const YICAKernelConfig& optimization_config
    );
    
    // 内核融合
    YICAKernelGenerationResult fuse_kernels(
        const std::vector<YICAKernelGenerationResult>& kernels,
        const YICAKernelConfig& fusion_config
    );
    
    // 性能分析和调优
    struct KernelAnalysis {
        float theoretical_peak_performance;
        float actual_performance;
        float efficiency_ratio;
        std::vector<std::string> bottlenecks;
        std::vector<std::string> optimization_suggestions;
    };
    
    KernelAnalysis analyze_kernel_performance(
        const YICAKernelGenerationResult& kernel
    );
    
    // 自动调优
    YICAKernelConfig auto_tune_kernel_config(
        const kernel::KNOperator* op,
        const std::vector<std::vector<int>>& input_shapes
    );
    
    // 模板管理
    void register_custom_template(
        const std::string& template_name,
        const std::string& template_code
    );
    
    std::vector<std::string> get_available_templates() const;
    
    // 缓存管理
    void enable_kernel_cache(bool enable) { cache_enabled_ = enable; }
    void clear_kernel_cache();
    size_t get_cache_size() const;

private:
    YICAConfig config_;
    bool cache_enabled_;
    
    // 内核缓存
    std::map<std::string, YICAKernelGenerationResult> kernel_cache_;
    
    // 模板存储
    std::map<std::string, std::string> custom_templates_;
    
    // 内核生成器组件
    std::unique_ptr<YISInstructionSet> yis_generator_;
    std::unique_ptr<CIMArrayCodeGenerator> cim_code_gen_;
    std::unique_ptr<SPMOptimizer> spm_optimizer_;
    std::unique_ptr<KernelFusionEngine> fusion_engine_;
    
    // 内部生成方法
    YICAKernelGenerationResult generate_cim_matmul_kernel(
        const kernel::KNOperator* op, const YICAKernelConfig& config
    );
    
    YICAKernelGenerationResult generate_cim_conv2d_kernel(
        const kernel::KNOperator* op, const YICAKernelConfig& config
    );
    
    YICAKernelGenerationResult generate_cim_attention_kernel(
        const kernel::KNOperator* op, const YICAKernelConfig& config
    );
    
    YICAKernelGenerationResult generate_spm_reduction_kernel(
        const kernel::KNOperator* op, const YICAKernelConfig& config
    );
    
    YICAKernelGenerationResult generate_spm_elementwise_kernel(
        const kernel::KNOperator* op, const YICAKernelConfig& config
    );
    
    YICAKernelGenerationResult generate_fused_mlp_kernel(
        const kernel::KNOperator* op, const YICAKernelConfig& config
    );
    
    // 代码生成辅助方法
    std::string generate_kernel_header(const std::string& kernel_name);
    std::string generate_kernel_footer();
    std::string generate_memory_allocation_code(const YICAKernelConfig& config);
    std::string generate_cim_setup_code(const YICAKernelConfig::CIMConfig& cim_config);
    std::string generate_spm_setup_code(const YICAKernelConfig::SPMConfig& spm_config);
    
    // 优化方法
    std::string apply_loop_optimizations(const std::string& code, 
                                       const YICAKernelConfig& config);
    std::string apply_instruction_fusion(const std::string& code);
    std::string apply_register_tiling(const std::string& code, 
                                    const YICAKernelConfig& config);
    
    // 性能预测
    YICAKernelGenerationResult::PerformancePrediction predict_performance(
        const std::string& kernel_code, const YICAKernelConfig& config
    );
    
    // 资源分析
    YICAKernelGenerationResult::ResourceUsage analyze_resource_usage(
        const std::string& kernel_code, const YICAKernelConfig& config
    );
    
    // 缓存管理
    std::string generate_cache_key(const kernel::KNOperator* op, 
                                  const YICAKernelConfig& config);
    void cache_kernel(const std::string& key, 
                     const YICAKernelGenerationResult& result);
    bool get_cached_kernel(const std::string& key, 
                          YICAKernelGenerationResult& result);
};

// CIM 阵列代码生成器
class CIMArrayCodeGenerator {
public:
    explicit CIMArrayCodeGenerator(const YICAConfig& config);
    
    // 生成 CIM 阵列计算代码
    std::string generate_cim_matmul(
        const std::vector<int>& a_shape,
        const std::vector<int>& b_shape,
        const YICAKernelConfig::CIMConfig& cim_config
    );
    
    std::string generate_cim_conv2d(
        const std::vector<int>& input_shape,
        const std::vector<int>& weight_shape,
        const YICAKernelConfig::CIMConfig& cim_config
    );
    
    std::string generate_cim_attention_qkv(
        const std::vector<int>& input_shape,
        int num_heads, int head_dim,
        const YICAKernelConfig::CIMConfig& cim_config
    );
    
    // CIM 阵列配置优化
    YICAKernelConfig::CIMConfig optimize_cim_config(
        const std::vector<int>& computation_shape,
        const YICAConfig& hardware_config
    );
    
private:
    YICAConfig config_;
    
    // 内部方法
    std::string generate_cim_array_allocation(int num_arrays);
    std::string generate_cim_data_loading(const std::vector<int>& shape);
    std::string generate_cim_computation_loop(const std::vector<int>& shape);
    std::string generate_cim_result_collection();
    
    // 优化方法
    int calculate_optimal_array_count(const std::vector<int>& shape);
    std::vector<int> calculate_optimal_tiling(const std::vector<int>& shape, int num_arrays);
};

// SPM 优化器
class SPMOptimizer {
public:
    explicit SPMOptimizer(const YICAConfig& config);
    
    // SPM 内存布局优化
    std::string generate_spm_allocation_code(
        const std::vector<std::pair<std::string, size_t>>& tensors,
        const YICAKernelConfig::SPMConfig& spm_config
    );
    
    std::string generate_spm_data_movement_code(
        const std::string& tensor_name,
        const std::vector<int>& shape,
        bool is_input
    );
    
    // SPM 缓存策略
    enum class SPMCacheStrategy {
        LOCALITY_FIRST,    // 局部性优先
        REUSE_FIRST,       // 重用优先
        SIZE_FIRST,        // 大小优先
        HYBRID             // 混合策略
    };
    
    SPMCacheStrategy select_optimal_cache_strategy(
        const std::vector<std::pair<std::string, size_t>>& tensors
    );
    
    // 预取优化
    std::string generate_prefetch_code(
        const std::string& tensor_name,
        const std::vector<int>& access_pattern
    );
    
private:
    YICAConfig config_;
    
    // 内部方法
    size_t calculate_spm_requirement(const std::vector<int>& shape);
    std::vector<std::string> generate_memory_layout_plan(
        const std::vector<std::pair<std::string, size_t>>& tensors
    );
};

// 内核融合引擎
class KernelFusionEngine {
public:
    explicit KernelFusionEngine(const YICAConfig& config);
    
    // 融合模式
    enum class FusionPattern {
        VERTICAL_FUSION,    // 垂直融合 (串行操作)
        HORIZONTAL_FUSION,  // 水平融合 (并行操作)
        LOOP_FUSION,        // 循环融合
        DATA_LAYOUT_FUSION  // 数据布局融合
    };
    
    // 融合分析
    struct FusionAnalysis {
        std::vector<FusionPattern> applicable_patterns;
        float estimated_speedup;
        size_t memory_savings;
        std::vector<std::string> fusion_barriers;
    };
    
    FusionAnalysis analyze_fusion_opportunity(
        const std::vector<YICAKernelGenerationResult>& kernels
    );
    
    // 执行融合
    YICAKernelGenerationResult execute_fusion(
        const std::vector<YICAKernelGenerationResult>& kernels,
        FusionPattern pattern,
        const YICAKernelConfig& config
    );
    
private:
    YICAConfig config_;
    
    // 融合实现方法
    std::string fuse_vertical_kernels(
        const std::vector<std::string>& kernel_codes
    );
    
    std::string fuse_horizontal_kernels(
        const std::vector<std::string>& kernel_codes
    );
    
    std::string fuse_loops(
        const std::vector<std::string>& kernel_codes
    );
    
    // 融合优化
    std::string optimize_fused_kernel(const std::string& fused_code);
    bool check_fusion_legality(const std::vector<YICAKernelGenerationResult>& kernels);
};

// 内核模板库
namespace kernel_templates {
    
    // 预定义模板
    extern const std::string CIM_MATMUL_TEMPLATE;
    extern const std::string CIM_CONV2D_TEMPLATE;
    extern const std::string CIM_ATTENTION_TEMPLATE;
    extern const std::string SPM_REDUCTION_TEMPLATE;
    extern const std::string SPM_ELEMENTWISE_TEMPLATE;
    extern const std::string FUSED_MLP_TEMPLATE;
    
    // 模板工具函数
    std::string instantiate_template(
        const std::string& template_code,
        const std::map<std::string, std::string>& parameters
    );
    
    std::vector<std::string> extract_template_parameters(
        const std::string& template_code
    );
    
    bool validate_template(const std::string& template_code);
    
} // namespace kernel_templates

// 内核生成工具函数
namespace kernel_utils {
    
    // 形状分析
    std::vector<int> infer_output_shape(
        const std::vector<std::vector<int>>& input_shapes,
        type::KNOperatorType op_type
    );
    
    // 计算复杂度分析
    size_t calculate_flops(
        const std::vector<int>& shape,
        type::KNOperatorType op_type
    );
    
    // 内存访问分析
    size_t calculate_memory_access(
        const std::vector<std::vector<int>>& shapes
    );
    
    // 最优配置推荐
    YICAKernelConfig recommend_kernel_config(
        const kernel::KNOperator* op,
        const YICAConfig& hardware_config
    );
    
    // 性能基准测试
    float benchmark_kernel_performance(
        const YICAKernelGenerationResult& kernel,
        const std::vector<std::vector<int>>& test_shapes
    );
    
} // namespace kernel_utils

} // namespace yica
} // namespace yirage 