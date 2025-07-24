#pragma once

#include "yirage/search/yica/optimization_strategy.h"
#include "yirage/search/yica/yica_types.h"
#include <memory>
#include <vector>
#include <map>

namespace yirage {
namespace search {
namespace yica {

/**
 * YICA优化策略库
 * 
 * 负责管理和应用各种YICA架构优化策略，
 * 包括策略选择、组合优化和执行控制。
 */
class YICAOptimizationStrategyLibrary {
public:
    using StrategyPtr = std::unique_ptr<OptimizationStrategy>;
    
    struct StrategySelection {
        std::vector<OptimizationStrategy*> selected_strategies;
        float expected_improvement = 0.0f;   // 预期改进程度
        std::string selection_rationale;     // 选择理由
        float confidence_score = 0.0f;       // 置信度评分
    };
    
    struct StrategyApplicationResult {
        std::vector<OptimizationStrategy::OptimizationResult> individual_results;
        float overall_improvement = 0.0f;    // 总体改进程度
        float total_latency_improvement = 0.0f;
        float total_energy_reduction = 0.0f;
        float total_memory_efficiency_gain = 0.0f;
        bool success = false;
        std::string summary;                  // 结果摘要
    };
    
    /**
     * 构造函数
     */
    YICAOptimizationStrategyLibrary();
    
    /**
     * 析构函数
     */
    ~YICAOptimizationStrategyLibrary();
    
    // 禁用拷贝构造和赋值
    YICAOptimizationStrategyLibrary(const YICAOptimizationStrategyLibrary&) = delete;
    YICAOptimizationStrategyLibrary& operator=(const YICAOptimizationStrategyLibrary&) = delete;
    
    /**
     * 策略管理
     */
    void register_strategy(StrategyPtr strategy);
    void unregister_strategy(OptimizationStrategy::StrategyType type);
    std::vector<OptimizationStrategy*> get_all_strategies() const;
    OptimizationStrategy* get_strategy(OptimizationStrategy::StrategyType type) const;
    
    /**
     * 策略选择
     */
    StrategySelection select_strategies(const AnalysisResult& analysis) const;
    std::vector<OptimizationStrategy*> get_applicable_strategies(const AnalysisResult& analysis) const;
    
    /**
     * 策略应用
     */
    StrategyApplicationResult apply_strategies(
        kernel::Graph& graph, 
        const YICAConfig& config,
        const std::vector<OptimizationStrategy*>& strategies) const;
    
    StrategyApplicationResult apply_selected_strategies(
        kernel::Graph& graph,
        const YICAConfig& config,
        const StrategySelection& selection) const;
    
    /**
     * 端到端优化
     */
    StrategyApplicationResult optimize_graph(
        kernel::Graph& graph,
        const YICAConfig& config,
        const AnalysisResult& analysis) const;
    
    /**
     * 策略组合优化
     */
    StrategySelection optimize_strategy_combination(const AnalysisResult& analysis) const;

private:
    std::map<OptimizationStrategy::StrategyType, StrategyPtr> strategies_;
    
    // 策略评估和选择算法
    float evaluate_strategy_combination(
        const std::vector<OptimizationStrategy*>& strategies,
        const AnalysisResult& analysis) const;
    
    bool are_strategies_compatible(
        OptimizationStrategy* s1, 
        OptimizationStrategy* s2) const;
    
    std::vector<OptimizationStrategy*> select_greedy_strategies(
        const AnalysisResult& analysis) const;
    
    std::vector<OptimizationStrategy*> select_optimal_strategies(
        const AnalysisResult& analysis) const;
    
    float estimate_strategy_benefit(
        OptimizationStrategy* strategy,
        const AnalysisResult& analysis) const;
    
    /**
     * 初始化默认策略
     */
    void initialize_default_strategies();
    
    /**
     * 策略兼容性检查
     */
    bool check_strategy_dependencies(
        const std::vector<OptimizationStrategy*>& strategies) const;
    
    /**
     * 生成策略应用摘要
     */
    std::string generate_application_summary(
        const StrategyApplicationResult& result) const;
};

} // namespace yica
} // namespace search
} // namespace yirage 