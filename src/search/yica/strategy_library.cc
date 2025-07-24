#include "yirage/search/yica/strategy_library.h"
#include "yirage/search/yica/optimization_strategy.h"
#include <algorithm>
#include <sstream>
#include <cmath>

namespace yirage {
namespace search {
namespace yica {

YICAOptimizationStrategyLibrary::YICAOptimizationStrategyLibrary() {
    initialize_default_strategies();
}

YICAOptimizationStrategyLibrary::~YICAOptimizationStrategyLibrary() {
    // unique_ptr会自动清理内存
}

void YICAOptimizationStrategyLibrary::initialize_default_strategies() {
    // 注册默认策略
    register_strategy(std::make_unique<CIMDataReuseStrategy>());
    register_strategy(std::make_unique<SPMAllocationStrategy>());
    register_strategy(std::make_unique<OperatorFusionStrategy>());
}

void YICAOptimizationStrategyLibrary::register_strategy(StrategyPtr strategy) {
    if (strategy) {
        auto type = strategy->get_type();
        strategies_[type] = std::move(strategy);
    }
}

void YICAOptimizationStrategyLibrary::unregister_strategy(OptimizationStrategy::StrategyType type) {
    strategies_.erase(type);
}

std::vector<OptimizationStrategy*> YICAOptimizationStrategyLibrary::get_all_strategies() const {
    std::vector<OptimizationStrategy*> result;
    result.reserve(strategies_.size());
    
    for (const auto& [type, strategy] : strategies_) {
        result.push_back(strategy.get());
    }
    
    return result;
}

OptimizationStrategy* YICAOptimizationStrategyLibrary::get_strategy(
    OptimizationStrategy::StrategyType type) const {
    auto it = strategies_.find(type);
    return (it != strategies_.end()) ? it->second.get() : nullptr;
}

std::vector<OptimizationStrategy*> YICAOptimizationStrategyLibrary::get_applicable_strategies(
    const AnalysisResult& analysis) const {
    
    std::vector<OptimizationStrategy*> applicable;
    
    for (const auto& [type, strategy] : strategies_) {
        if (strategy->is_applicable(analysis)) {
            applicable.push_back(strategy.get());
        }
    }
    
    return applicable;
}

YICAOptimizationStrategyLibrary::StrategySelection 
YICAOptimizationStrategyLibrary::select_strategies(const AnalysisResult& analysis) const {
    
    StrategySelection selection;
    
    // 使用贪心算法选择策略
    auto greedy_strategies = select_greedy_strategies(analysis);
    
    if (!greedy_strategies.empty()) {
        selection.selected_strategies = greedy_strategies;
        selection.expected_improvement = evaluate_strategy_combination(greedy_strategies, analysis);
        selection.confidence_score = std::min(selection.expected_improvement * 1.2f, 1.0f);
        
        // 生成选择理由
        std::stringstream ss;
        ss << "Selected " << greedy_strategies.size() << " strategies based on greedy optimization: ";
        for (size_t i = 0; i < greedy_strategies.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << greedy_strategies[i]->get_name();
        }
        selection.selection_rationale = ss.str();
    } else {
        selection.selection_rationale = "No applicable strategies found for the given analysis";
    }
    
    return selection;
}

std::vector<OptimizationStrategy*> YICAOptimizationStrategyLibrary::select_greedy_strategies(
    const AnalysisResult& analysis) const {
    
    std::vector<OptimizationStrategy*> applicable = get_applicable_strategies(analysis);
    std::vector<OptimizationStrategy*> selected;
    
    if (applicable.empty()) {
        return selected;
    }
    
    // 按预期收益排序
    std::sort(applicable.begin(), applicable.end(),
              [&](OptimizationStrategy* a, OptimizationStrategy* b) {
                  return a->estimate_benefit(analysis) > b->estimate_benefit(analysis);
              });
    
    // 贪心选择：按收益从高到低添加兼容的策略
    for (auto* strategy : applicable) {
        bool compatible = true;
        
        for (auto* selected_strategy : selected) {
            if (!are_strategies_compatible(strategy, selected_strategy)) {
                compatible = false;
                break;
            }
        }
        
        if (compatible) {
            selected.push_back(strategy);
        }
    }
    
    return selected;
}

std::vector<OptimizationStrategy*> YICAOptimizationStrategyLibrary::select_optimal_strategies(
    const AnalysisResult& analysis) const {
    
    // 简化的最优策略选择：目前使用贪心算法
    // 在实际实现中，这里可以使用动态规划或其他优化算法
    return select_greedy_strategies(analysis);
}

float YICAOptimizationStrategyLibrary::evaluate_strategy_combination(
    const std::vector<OptimizationStrategy*>& strategies,
    const AnalysisResult& analysis) const {
    
    if (strategies.empty()) {
        return 0.0f;
    }
    
    float total_benefit = 0.0f;
    
    // 简化的组合评估：累加各策略收益，考虑递减效应
    for (size_t i = 0; i < strategies.size(); ++i) {
        float individual_benefit = strategies[i]->estimate_benefit(analysis);
        float decay_factor = 1.0f / (1.0f + i * 0.1f); // 递减因子
        total_benefit += individual_benefit * decay_factor;
    }
    
    return std::min(total_benefit, 1.0f);
}

bool YICAOptimizationStrategyLibrary::are_strategies_compatible(
    OptimizationStrategy* s1, OptimizationStrategy* s2) const {
    
    if (!s1 || !s2) {
        return false;
    }
    
    // 简化的兼容性检查
    auto type1 = s1->get_type();
    auto type2 = s2->get_type();
    
    // 某些策略组合可能不兼容
    if (type1 == OptimizationStrategy::SPM_ALLOCATION && 
        type2 == OptimizationStrategy::CIM_DATA_REUSE) {
        // SPM分配和CIM数据重用可能有冲突
        return false;
    }
    
    // 默认情况下，大多数策略是兼容的
    return true;
}

YICAOptimizationStrategyLibrary::StrategyApplicationResult 
YICAOptimizationStrategyLibrary::apply_strategies(
    kernel::Graph& graph, 
    const YICAConfig& config,
    const std::vector<OptimizationStrategy*>& strategies) const {
    
    StrategyApplicationResult result;
    
    if (strategies.empty()) {
        result.success = false;
        result.summary = "No strategies to apply";
        return result;
    }
    
    // 检查策略依赖关系
    if (!check_strategy_dependencies(strategies)) {
        result.success = false;
        result.summary = "Strategy dependencies not satisfied";
        return result;
    }
    
    // 依次应用每个策略
    float total_improvement = 0.0f;
    bool all_succeeded = true;
    
    for (auto* strategy : strategies) {
        try {
            auto individual_result = strategy->apply(graph, config);
            result.individual_results.push_back(individual_result);
            
            if (individual_result.success) {
                total_improvement += individual_result.improvement_score;
                result.total_latency_improvement += individual_result.latency_improvement;
                result.total_energy_reduction += individual_result.energy_reduction;
                result.total_memory_efficiency_gain += individual_result.memory_efficiency_gain;
            } else {
                all_succeeded = false;
            }
            
        } catch (const std::exception& e) {
            all_succeeded = false;
            OptimizationStrategy::OptimizationResult error_result;
            error_result.success = false;
            error_result.description = "Error applying " + strategy->get_name() + ": " + e.what();
            result.individual_results.push_back(error_result);
        }
    }
    
    result.success = all_succeeded;
    result.overall_improvement = std::min(total_improvement, 1.0f);
    result.summary = generate_application_summary(result);
    
    return result;
}

YICAOptimizationStrategyLibrary::StrategyApplicationResult 
YICAOptimizationStrategyLibrary::apply_selected_strategies(
    kernel::Graph& graph,
    const YICAConfig& config,
    const StrategySelection& selection) const {
    
    return apply_strategies(graph, config, selection.selected_strategies);
}

YICAOptimizationStrategyLibrary::StrategyApplicationResult 
YICAOptimizationStrategyLibrary::optimize_graph(
    kernel::Graph& graph,
    const YICAConfig& config,
    const AnalysisResult& analysis) const {
    
    // 端到端优化：自动选择和应用策略
    auto selection = select_strategies(analysis);
    return apply_selected_strategies(graph, config, selection);
}

YICAOptimizationStrategyLibrary::StrategySelection 
YICAOptimizationStrategyLibrary::optimize_strategy_combination(const AnalysisResult& analysis) const {
    
    // 使用更sophisticated的算法优化策略组合
    // 目前使用贪心算法，未来可以扩展为遗传算法或模拟退火
    
    auto greedy_selection = select_strategies(analysis);
    
    // 尝试优化策略顺序
    if (greedy_selection.selected_strategies.size() > 1) {
        auto& strategies = greedy_selection.selected_strategies;
        
        // 简单的顺序优化：按预期收益重新排序
        std::sort(strategies.begin(), strategies.end(),
                  [&](OptimizationStrategy* a, OptimizationStrategy* b) {
                      return a->estimate_benefit(analysis) > b->estimate_benefit(analysis);
                  });
        
        // 重新评估组合收益
        greedy_selection.expected_improvement = evaluate_strategy_combination(strategies, analysis);
    }
    
    return greedy_selection;
}

bool YICAOptimizationStrategyLibrary::check_strategy_dependencies(
    const std::vector<OptimizationStrategy*>& strategies) const {
    
    // 简化的依赖关系检查
    for (size_t i = 0; i < strategies.size(); ++i) {
        for (size_t j = i + 1; j < strategies.size(); ++j) {
            if (!are_strategies_compatible(strategies[i], strategies[j])) {
                return false;
            }
        }
    }
    
    return true;
}

std::string YICAOptimizationStrategyLibrary::generate_application_summary(
    const StrategyApplicationResult& result) const {
    
    std::stringstream ss;
    
    if (result.success) {
        ss << "Successfully applied " << result.individual_results.size() << " optimization strategies. ";
        ss << "Overall improvement: " << (result.overall_improvement * 100) << "%. ";
        ss << "Latency improvement: " << (result.total_latency_improvement * 100) << "%, ";
        ss << "Energy reduction: " << (result.total_energy_reduction * 100) << "%, ";
        ss << "Memory efficiency gain: " << (result.total_memory_efficiency_gain * 100) << "%.";
    } else {
        size_t failed_count = 0;
        for (const auto& individual_result : result.individual_results) {
            if (!individual_result.success) {
                failed_count++;
            }
        }
        ss << "Optimization partially failed. " << failed_count << " out of " 
           << result.individual_results.size() << " strategies failed to apply.";
    }
    
    return ss.str();
}

float YICAOptimizationStrategyLibrary::estimate_strategy_benefit(
    OptimizationStrategy* strategy,
    const AnalysisResult& analysis) const {
    
    if (!strategy) {
        return 0.0f;
    }
    
    return strategy->estimate_benefit(analysis);
}

} // namespace yica
} // namespace search
} // namespace yirage
} 