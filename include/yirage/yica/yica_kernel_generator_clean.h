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
#include "yirage/yica/cim_resource_manager.h"
#include "yirage/yica/spm_memory_manager.h"
#include "yirage/yica/kernel_fusion_engine.h"

namespace yirage {
namespace yica {

// YICA 内核模板类型
enum class YICAKernelTemplate {
    CIM_MATMUL,           // CIM 阵列矩阵乘法
    CIM_CONV2D,           // CIM 阵列卷积
    CIM_ATTENTION,        // CIM 阵列注意力机制
    SPM_REDUCTION,        // SPM 优化的归约操作
    SPM_ELEMENTWISE,      // SPM 优化的逐元素操作
    FUSED_MLP,            // 融合的 MLP 层
    FUSED_ATTENTION_BLOCK,// 融合的注意力块
    CUSTOM                // 自定义模板
};

// YICA 计算模式
enum class YICAComputeMode {
    CIM_ONLY,             // 仅使用 CIM
    SPM_ONLY,             // 仅使用 SPM
    CIM_PARALLEL,         // CIM 并行模式
    CIM_SPM_HYBRID,       // CIM-SPM 混合模式
    AUTO                  // 自动选择
};

// YICA 内核配置
struct YICAKernelConfig {
    // 基本配置
    YICAKernelTemplate template_type = YICAKernelTemplate::CUSTOM;
    YICAComputeMode compute_mode = YICAComputeMode::AUTO;
    std::string kernel_name;
    
    // CIM 配置
    struct CIMConfig {
        uint32_t num_arrays = 1;
        uint32_t array_size_x = 256;
        uint32_t array_size_y = 256;
        bool enable_pipelining = true;
        float utilization_target = 0.8f;
    } cim_config;
    
    // SPM 配置
    struct SPMConfig {
        size_t allocation_size = 0;
        bool enable_double_buffer = false;
        bool enable_prefetch = true;
        uint32_t cache_line_size = 64;
    } spm_config;
    
    // 优化配置
    bool enable_loop_unrolling = true;
    bool enable_instruction_fusion = true;
    bool enable_register_tiling = true;
    uint32_t unroll_factor = 4;
    
    // 数据类型配置
    std::string input_dtype = "float32";
    std::string output_dtype = "float32";
    std::string compute_dtype = "float32";
};

// YICA 内核生成结果
struct YICAKernelGenerationResult {
    // 生成的代码
    std::string yis_code;          // YIS 指令代码
    std::string triton_code;        // Triton 包装代码
    std::string cuda_code;          // CUDA 代码（可选）
    std::string kernel_name;        // 内核名称
    
    // 生成状态
    bool generation_successful = false;
    std::string error_message;
    std::vector<std::string> warnings;
    
    // 性能预测
    struct PerformancePrediction {
        float estimated_latency_ms = 0.0f;
        float estimated_throughput_gops = 0.0f;
        float cim_utilization = 0.0f;
        float spm_utilization = 0.0f;
        float memory_bandwidth_utilization = 0.0f;
    } performance;
    
    // 资源使用
    struct ResourceUsage {
        uint32_t cim_arrays_used = 0;
        size_t spm_bytes_used = 0;
        size_t register_count = 0;
        size_t instruction_count = 0;
    } resource_usage;
    
    // 优化信息
    std::vector<std::string> applied_optimizations;
    std::map<std::string, float> optimization_metrics;
};

// YICA 内核生成器主类
class YICAKernelGenerator {
public:
    explicit YICAKernelGenerator(const YICAConfig& config);
    ~YICAKernelGenerator();
    
    // 生成内核
    YICAKernelGenerationResult generate_kernel(
        const kernel::KNOperator* op,
        const YICAKernelConfig& kernel_config = YICAKernelConfig{}
    );
    
    // 批量生成融合内核
    YICAKernelGenerationResult generate_fused_kernel(
        const std::vector<kernel::KNOperator*>& ops,
        const YICAKernelConfig& kernel_config = YICAKernelConfig{}
    );
    
    // 从图生成优化内核
    YICAKernelGenerationResult generate_from_graph(
        const kernel::Graph& graph,
        const YICAKernelConfig& kernel_config = YICAKernelConfig{}
    );
    
    // 配置管理
    void set_default_config(const YICAKernelConfig& config);
    YICAKernelConfig get_default_config() const;
    YICAKernelConfig recommend_config(const kernel::KNOperator* op) const;
    
    // 模板管理
    void register_custom_template(const std::string& name, 
                                const std::string& template_code);
    bool has_template(const std::string& name) const;
    std::vector<std::string> get_available_templates() const;
    
    // 缓存管理
    void enable_cache(bool enable) { cache_enabled_ = enable; }
    void clear_cache() { kernel_cache_.clear(); }
    size_t get_cache_size() const { return kernel_cache_.size(); }
    
    // 性能分析
    YICAKernelGenerationResult::PerformancePrediction 
        predict_performance(const kernel::KNOperator* op,
                          const YICAKernelConfig& config) const;
    
    // 验证配置
    bool validate_config(const YICAKernelConfig& config) const;
    std::vector<std::string> get_config_warnings(const YICAKernelConfig& config) const;
    
private:
    // 配置
    YICAConfig config_;
    YICAKernelConfig default_kernel_config_;
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
    
    // Helper function for Triton wrapper generation
    std::string generate_triton_wrapper(const std::string& yis_code, const YICAKernelConfig& config);
    
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

} // namespace yica
} // namespace yirage
