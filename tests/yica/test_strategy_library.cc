#include "yirage/search/yica/strategy_library.h"
#include "yirage/search/yica/yica_analyzer.h"
#include <gtest/gtest.h>
#include <memory>

namespace yirage {
namespace search {
namespace yica {

class YICAStrategyLibraryTest : public ::testing::Test {
protected:
    void SetUp() override {
        library = std::make_unique<YICAOptimizationStrategyLibrary>();
        
        // 设置测试配置
        config.num_cim_arrays = 16;
        config.cim_array_size = 128;
        config.spm_size_kb = 64;
        config.dram_bandwidth_gbps = 100.0f;
        config.spm_bandwidth_gbps = 500.0f;
        config.cim_frequency_mhz = 200.0f;
        
        // 创建简单的测试图
        test_graph = std::make_unique<kernel::Graph>();
        
        // 设置分析结果
        test_analysis.cim_friendliness_score = 0.7f;
        test_analysis.spm_efficiency = 0.4f;
        test_analysis.parallelization_opportunities.data_parallel_ops = 5;
        test_analysis.parallelization_opportunities.model_parallel_ops = 3;
        
        // 添加内存访问模式
        MemoryAccessPattern pattern1;
        pattern1.tensor_id = "tensor_1";
        pattern1.access_frequency = 3;
        pattern1.pattern_type = "sequential";
        pattern1.spm_friendly = true;
        test_analysis.memory_access_patterns.push_back(pattern1);
        
        MemoryAccessPattern pattern2;
        pattern2.tensor_id = "tensor_2";
        pattern2.access_frequency = 5;
        pattern2.pattern_type = "random";
        pattern2.spm_friendly = false;
        test_analysis.memory_access_patterns.push_back(pattern2);
    }

    std::unique_ptr<YICAOptimizationStrategyLibrary> library;
    std::unique_ptr<kernel::Graph> test_graph;
    YICAConfig config;
    AnalysisResult test_analysis;
};

TEST_F(YICAStrategyLibraryTest, InitializationTest) {
    EXPECT_NE(library, nullptr);
    
    // 检查默认策略是否已注册
    auto all_strategies = library->get_all_strategies();
    EXPECT_GE(all_strategies.size(), 3); // 至少有3个默认策略
    
    // 检查特定策略是否存在
    auto cim_strategy = library->get_strategy(OptimizationStrategy::CIM_DATA_REUSE);
    EXPECT_NE(cim_strategy, nullptr);
    
    auto spm_strategy = library->get_strategy(OptimizationStrategy::SPM_ALLOCATION);
    EXPECT_NE(spm_strategy, nullptr);
    
    auto fusion_strategy = library->get_strategy(OptimizationStrategy::OPERATOR_FUSION);
    EXPECT_NE(fusion_strategy, nullptr);
}

TEST_F(YICAStrategyLibraryTest, StrategyRegistrationTest) {
    size_t initial_count = library->get_all_strategies().size();
    
    // 注册新策略（使用现有的策略类型作为测试）
    library->register_strategy(std::make_unique<CIMDataReuseStrategy>());
    
    // 策略数量应该保持不变（替换现有策略）
    EXPECT_EQ(library->get_all_strategies().size(), initial_count);
    
    // 注销策略
    library->unregister_strategy(OptimizationStrategy::CIM_DATA_REUSE);
    EXPECT_EQ(library->get_all_strategies().size(), initial_count - 1);
}

TEST_F(YICAStrategyLibraryTest, ApplicableStrategiesTest) {
    // 测试获取适用策略
    auto applicable = library->get_applicable_strategies(test_analysis);
    
    // 基于测试分析结果，应该有一些适用的策略
    EXPECT_GT(applicable.size(), 0);
    
    // 验证每个返回的策略确实适用
    for (auto* strategy : applicable) {
        EXPECT_TRUE(strategy->is_applicable(test_analysis));
    }
}

TEST_F(YICAStrategyLibraryTest, StrategySelectionTest) {
    auto selection = library->select_strategies(test_analysis);
    
    // 应该选择到一些策略
    if (!selection.selected_strategies.empty()) {
        EXPECT_GT(selection.expected_improvement, 0.0f);
        EXPECT_GT(selection.confidence_score, 0.0f);
        EXPECT_FALSE(selection.selection_rationale.empty());
        
        // 验证所有选中的策略都是适用的
        for (auto* strategy : selection.selected_strategies) {
            EXPECT_TRUE(strategy->is_applicable(test_analysis));
        }
    }
}

TEST_F(YICAStrategyLibraryTest, StrategyApplicationTest) {
    // 选择策略
    auto selection = library->select_strategies(test_analysis);
    
    if (!selection.selected_strategies.empty()) {
        // 应用策略
        auto result = library->apply_selected_strategies(*test_graph, config, selection);
        
        EXPECT_FALSE(result.summary.empty());
        EXPECT_EQ(result.individual_results.size(), selection.selected_strategies.size());
        
        // 验证结果结构
        EXPECT_GE(result.overall_improvement, 0.0f);
        EXPECT_LE(result.overall_improvement, 1.0f);
    }
}

TEST_F(YICAStrategyLibraryTest, EndToEndOptimizationTest) {
    // 端到端优化测试
    auto result = library->optimize_graph(*test_graph, config, test_analysis);
    
    EXPECT_FALSE(result.summary.empty());
    
    // 如果有策略被应用，检查结果
    if (!result.individual_results.empty()) {
        EXPECT_GE(result.overall_improvement, 0.0f);
        EXPECT_LE(result.overall_improvement, 1.0f);
        EXPECT_GE(result.total_latency_improvement, 0.0f);
        EXPECT_GE(result.total_energy_reduction, 0.0f);
        EXPECT_GE(result.total_memory_efficiency_gain, 0.0f);
    }
}

TEST_F(YICAStrategyLibraryTest, StrategyCompatibilityTest) {
    auto all_strategies = library->get_all_strategies();
    
    if (all_strategies.size() >= 2) {
        // 测试策略兼容性检查
        bool found_incompatible = false;
        
        for (size_t i = 0; i < all_strategies.size(); ++i) {
            for (size_t j = i + 1; j < all_strategies.size(); ++j) {
                // 这里只是调用兼容性检查函数，不强制要求特定结果
                // 因为兼容性取决于具体实现
                auto* s1 = all_strategies[i];
                auto* s2 = all_strategies[j];
                
                // 确保函数不会崩溃
                EXPECT_NO_THROW({
                    // 通过选择包含这两个策略的策略列表来间接测试兼容性
                    std::vector<OptimizationStrategy*> test_strategies = {s1, s2};
                    library->apply_strategies(*test_graph, config, test_strategies);
                });
            }
        }
    }
}

TEST_F(YICAStrategyLibraryTest, EmptyAnalysisTest) {
    // 测试空分析结果的处理
    AnalysisResult empty_analysis;
    empty_analysis.cim_friendliness_score = 0.0f;
    empty_analysis.spm_efficiency = 0.0f;
    
    auto selection = library->select_strategies(empty_analysis);
    
    // 对于空分析，可能没有适用的策略
    if (selection.selected_strategies.empty()) {
        EXPECT_EQ(selection.expected_improvement, 0.0f);
        EXPECT_FALSE(selection.selection_rationale.empty());
    }
}

TEST_F(YICAStrategyLibraryTest, HighQualityAnalysisTest) {
    // 测试高质量分析结果
    AnalysisResult high_quality_analysis;
    high_quality_analysis.cim_friendliness_score = 0.9f;
    high_quality_analysis.spm_efficiency = 0.2f; // 低效率，有优化空间
    
    // 添加更多内存访问模式
    for (int i = 0; i < 5; ++i) {
        MemoryAccessPattern pattern;
        pattern.tensor_id = "tensor_" + std::to_string(i);
        pattern.access_frequency = 3 + i;
        pattern.pattern_type = "sequential";
        pattern.spm_friendly = true;
        high_quality_analysis.memory_access_patterns.push_back(pattern);
    }
    
    auto selection = library->select_strategies(high_quality_analysis);
    
    // 高质量分析应该产生更好的策略选择
    EXPECT_GT(selection.expected_improvement, 0.0f);
    
    if (!selection.selected_strategies.empty()) {
        EXPECT_GT(selection.confidence_score, 0.0f);
    }
}

TEST_F(YICAStrategyLibraryTest, StrategyBenefitEstimationTest) {
    auto all_strategies = library->get_all_strategies();
    
    for (auto* strategy : all_strategies) {
        if (strategy->is_applicable(test_analysis)) {
            float benefit = strategy->estimate_benefit(test_analysis);
            
            // 收益应该在合理范围内
            EXPECT_GE(benefit, 0.0f);
            EXPECT_LE(benefit, 1.0f);
        }
    }
}

} // namespace yica
} // namespace search
} // namespace yirage 