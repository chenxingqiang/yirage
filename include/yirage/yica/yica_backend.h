#pragma once

#include "yirage/transpiler/transpiler.h"
#include "yirage/kernel/graph.h"
#include "yirage/yica/config.h"
#include "yirage/yica/yis_instruction_set.h"
#include "yirage/yica/cim_resource_manager.h"
#include "yirage/yica/spm_memory_manager.h"

namespace yirage {
namespace yica {

// YICA 专用后端实现
class YICABackend : public transpiler::Backend {
public:
    explicit YICABackend(const YICAConfig& config);
    ~YICABackend() override;

    // 主要接口：将 Yirage 计算图转译为 YICA 优化代码
    transpiler::TranspileResult transpile(kernel::Graph const* graph) override;
    
    // YICA 特定优化接口
    struct YICAOptimizationResult {
        std::string yis_kernel_code;           // 生成的 YIS 内核代码
        std::string triton_kernel_code;        // 生成的 Triton 内核代码
        CIMResourceAllocation cim_allocation;  // CIM 资源分配方案
        SPMMemoryPlan spm_memory_plan;        // SPM 内存分配计划
        float estimated_speedup;              // 预估加速比
        size_t memory_footprint;              // 内存占用
        std::vector<std::string> optimization_log; // 优化日志
    };
    
    // 执行 YICA 专用优化
    YICAOptimizationResult optimize_for_yica(kernel::Graph const* graph);
    
    // 性能分析接口
    struct PerformanceAnalysis {
        float compute_intensity;              // 计算密度
        float memory_bandwidth_requirement;   // 内存带宽需求
        float cim_friendliness_score;         // CIM 友好度评分
        std::vector<BottleneckAnalysis> bottlenecks; // 瓶颈分析
    };
    
    PerformanceAnalysis analyze_performance(kernel::Graph const* graph);

private:
    // 核心组件
    YICAConfig config_;
    std::unique_ptr<YISInstructionSet> yis_generator_;
    std::unique_ptr<CIMResourceManager> cim_manager_;
    std::unique_ptr<SPMMemoryManager> spm_manager_;
    std::unique_ptr<YICAGraphOptimizer> graph_optimizer_;
    
    // 内部优化方法
    kernel::Graph apply_yica_graph_optimizations(const kernel::Graph& graph);
    std::string generate_yis_code(const kernel::Graph& optimized_graph);
    std::string generate_triton_wrapper(const std::string& yis_code);
    CIMResourceAllocation allocate_cim_resources(const kernel::Graph& graph);
    SPMMemoryPlan plan_spm_memory(const kernel::Graph& graph);
    
    // 优化策略集合
    std::vector<std::unique_ptr<YICAOptimizationPass>> optimization_passes_;
    
    void initialize_optimization_passes();
};

// YICA 优化 Pass 基类
class YICAOptimizationPass {
public:
    virtual ~YICAOptimizationPass() = default;
    
    struct PassResult {
        kernel::Graph transformed_graph;
        bool applied;
        std::string description;
        float estimated_benefit;
    };
    
    virtual PassResult apply(const kernel::Graph& graph, const YICAConfig& config) = 0;
    virtual std::string get_pass_name() const = 0;
    virtual bool is_applicable(const kernel::Graph& graph) const = 0;
};

// CIM 数据重用优化 Pass
class CIMDataReuseOptimizationPass : public YICAOptimizationPass {
public:
    PassResult apply(const kernel::Graph& graph, const YICAConfig& config) override;
    std::string get_pass_name() const override { return "CIM Data Reuse Optimization"; }
    bool is_applicable(const kernel::Graph& graph) const override;

private:
    struct DataReusePattern {
        kernel::DTensor* tensor;
        std::vector<kernel::KNOperator*> consumers;
        float reuse_factor;
        size_t spm_cache_requirement;
        float estimated_speedup;
    };
    
    std::vector<DataReusePattern> identify_reuse_patterns(const kernel::Graph& graph);
    kernel::Graph implement_data_reuse(const kernel::Graph& graph, 
                                      const std::vector<DataReusePattern>& patterns);
};

// CIM 算子融合优化 Pass
class CIMOperatorFusionPass : public YICAOptimizationPass {
public:
    PassResult apply(const kernel::Graph& graph, const YICAConfig& config) override;
    std::string get_pass_name() const override { return "CIM Operator Fusion"; }
    bool is_applicable(const kernel::Graph& graph) const override;

private:
    struct FusionCandidate {
        std::vector<kernel::KNOperator*> operators;
        std::string fusion_type;
        float cim_efficiency_gain;
        size_t spm_requirement;
        std::string yis_template;
    };
    
    std::vector<FusionCandidate> identify_fusion_opportunities(const kernel::Graph& graph);
    kernel::Graph apply_operator_fusion(const kernel::Graph& graph, 
                                       const std::vector<FusionCandidate>& candidates);
};

// SPM 内存布局优化 Pass
class SPMMemoryLayoutOptimizationPass : public YICAOptimizationPass {
public:
    PassResult apply(const kernel::Graph& graph, const YICAConfig& config) override;
    std::string get_pass_name() const override { return "SPM Memory Layout Optimization"; }
    bool is_applicable(const kernel::Graph& graph) const override;

private:
    struct LayoutOptimization {
        kernel::DTensor* tensor;
        layout::DmemLayout original_layout;
        layout::DmemLayout optimized_layout;
        float access_efficiency_gain;
        size_t spm_footprint;
    };
    
    std::vector<LayoutOptimization> analyze_memory_layouts(const kernel::Graph& graph);
    kernel::Graph apply_layout_optimizations(const kernel::Graph& graph, 
                                            const std::vector<LayoutOptimization>& optimizations);
};

// YICA 性能分析器
class YICAPerformanceAnalyzer {
public:
    explicit YICAPerformanceAnalyzer(const YICAConfig& config);
    
    struct DetailedAnalysis {
        // 计算分析
        struct ComputeAnalysis {
            float total_flops;
            float peak_flops_utilization;
            std::map<std::string, float> op_type_distribution;
            float cim_compute_efficiency;
        } compute;
        
        // 内存分析
        struct MemoryAnalysis {
            size_t total_memory_access;
            float spm_hit_rate;
            float dram_bandwidth_utilization;
            std::vector<MemoryHotspot> hotspots;
        } memory;
        
        // CIM 阵列分析
        struct CIMAnalysis {
            std::vector<float> array_utilization;
            float load_balance_score;
            float parallel_efficiency;
        } cim;
        
        // 瓶颈分析
        std::vector<PerformanceBottleneck> bottlenecks;
        std::vector<OptimizationRecommendation> recommendations;
    };
    
    DetailedAnalysis analyze(const kernel::Graph& graph);
    
private:
    YICAConfig config_;
    std::unique_ptr<ComputeIntensityAnalyzer> compute_analyzer_;
    std::unique_ptr<MemoryAccessAnalyzer> memory_analyzer_;
    std::unique_ptr<CIMUtilizationAnalyzer> cim_analyzer_;
};

} // namespace yica
} // namespace yirage 