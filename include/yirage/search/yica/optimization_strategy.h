#pragma once

#include "yirage/search/yica/yica_types.h"
#include "yirage/kernel/graph.h"
#include <memory>
#include <vector>
#include <map>
#include <string>

namespace yirage {
namespace search {
namespace yica {

/**
 * 优化策略基类
 */
class OptimizationStrategy {
public:
    enum StrategyType {
        CIM_DATA_REUSE,           // CIM数据重用优化
        SPM_ALLOCATION,           // SPM分配优化
        CROSS_CIM_COMMUNICATION,  // 跨CIM通信优化
        MEMORY_ACCESS_PATTERN,    // 内存访问模式优化
        OPERATOR_FUSION,          // 算子融合优化
        PARALLELIZATION          // 并行化优化
    };
    
    struct OptimizationResult {
        bool success = false;
        float improvement_score = 0.0f;      // 改进评分 [0-1]
        std::string description;             // 优化描述
        std::map<std::string, float> metrics;  // 具体指标改进
        
        // 性能改进指标
        float latency_improvement = 0.0f;    // 延迟改进
        float energy_reduction = 0.0f;       // 能耗降低
        float memory_efficiency_gain = 0.0f; // 内存效率提升
    };
    
    virtual ~OptimizationStrategy() = default;
    
    /**
     * 判断策略是否适用于给定的分析结果
     */
    virtual bool is_applicable(const AnalysisResult& analysis) const = 0;
    
    /**
     * 应用优化策略到计算图
     */
    virtual OptimizationResult apply(kernel::Graph& graph, const YICAConfig& config) = 0;
    
    /**
     * 获取策略类型
     */
    virtual StrategyType get_type() const = 0;
    
    /**
     * 获取策略名称
     */
    virtual std::string get_name() const = 0;
    
    /**
     * 获取策略描述
     */
    virtual std::string get_description() const = 0;
    
    /**
     * 估算策略的预期收益
     */
    virtual float estimate_benefit(const AnalysisResult& analysis) const = 0;
};

/**
 * CIM数据重用优化策略
 */
class CIMDataReuseStrategy : public OptimizationStrategy {
public:
    struct ReusePattern {
        std::vector<kernel::DTensor*> reusable_tensors;
        float reuse_factor = 1.0f;           // 重用因子
        size_t memory_saving = 0;            // 内存节省量（字节）
        std::string pattern_type;            // 重用模式类型
    };
    
    CIMDataReuseStrategy();
    virtual ~CIMDataReuseStrategy() = default;
    
    bool is_applicable(const AnalysisResult& analysis) const override;
    OptimizationResult apply(kernel::Graph& graph, const YICAConfig& config) override;
    StrategyType get_type() const override { return CIM_DATA_REUSE; }
    std::string get_name() const override { return "CIM Data Reuse Optimization"; }
    std::string get_description() const override;
    float estimate_benefit(const AnalysisResult& analysis) const override;

private:
    std::vector<ReusePattern> identify_reuse_opportunities(const kernel::Graph& graph);
    void implement_data_reuse(kernel::Graph& graph, const ReusePattern& pattern);
    float calculate_reuse_benefit(const ReusePattern& pattern, const YICAConfig& config) const;
};

/**
 * SPM分配优化策略
 */
class SPMAllocationStrategy : public OptimizationStrategy {
public:
    struct AllocationPlan {
        std::map<kernel::DTensor*, size_t> tensor_allocation;  // 张量->SPM位置映射
        float spm_utilization = 0.0f;       // SPM利用率
        float access_efficiency = 0.0f;     // 访问效率提升
        size_t total_allocated = 0;         // 总分配大小
    };
    
    SPMAllocationStrategy();
    virtual ~SPMAllocationStrategy() = default;
    
    bool is_applicable(const AnalysisResult& analysis) const override;
    OptimizationResult apply(kernel::Graph& graph, const YICAConfig& config) override;
    StrategyType get_type() const override { return SPM_ALLOCATION; }
    std::string get_name() const override { return "SPM Allocation Optimization"; }
    std::string get_description() const override;
    float estimate_benefit(const AnalysisResult& analysis) const override;

private:
    AllocationPlan generate_allocation_plan(const kernel::Graph& graph, const YICAConfig& config);
    void implement_spm_allocation(kernel::Graph& graph, const AllocationPlan& plan);
    float calculate_allocation_benefit(const AllocationPlan& plan, const YICAConfig& config) const;
};

/**
 * 算子融合优化策略
 */
class OperatorFusionStrategy : public OptimizationStrategy {
public:
    struct FusionGroup {
        std::vector<kernel::KNOperator*> operators;
        float fusion_benefit = 0.0f;        // 融合收益评分
        std::string fusion_type;            // 融合类型描述
        size_t estimated_memory_saving = 0; // 预估内存节省
    };
    
    OperatorFusionStrategy();
    virtual ~OperatorFusionStrategy() = default;
    
    bool is_applicable(const AnalysisResult& analysis) const override;
    OptimizationResult apply(kernel::Graph& graph, const YICAConfig& config) override;
    StrategyType get_type() const override { return OPERATOR_FUSION; }
    std::string get_name() const override { return "Operator Fusion Optimization"; }
    std::string get_description() const override;
    float estimate_benefit(const AnalysisResult& analysis) const override;

private:
    std::vector<FusionGroup> identify_fusion_opportunities(const kernel::Graph& graph);
    void implement_operator_fusion(kernel::Graph& graph, const FusionGroup& group);
    float calculate_fusion_benefit(const FusionGroup& group, const YICAConfig& config) const;
    bool can_fuse_operators(kernel::KNOperator* op1, kernel::KNOperator* op2) const;
};

} // namespace yica
} // namespace search
} // namespace yirage 