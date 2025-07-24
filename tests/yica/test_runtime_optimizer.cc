#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include "yirage/search/yica/runtime_types.h"
#include "yirage/search/yica/performance_monitor.h"
#include "yirage/search/yica/runtime_optimizer.h"
#include "yirage/search/yica/ml_optimizer.h"
#include "yirage/search/yica/yica_runtime.h"

namespace yirage {
namespace search {
namespace yica {

class RuntimeOptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试配置
        test_config.num_cim_arrays = 16;
        test_config.cim_frequency_mhz = 1000;
        test_config.spm_size_kb = 1024;
        test_config.enable_data_reuse = true;
        test_config.enable_operator_fusion = true;
        test_config.enable_parallel_execution = true;
        test_config.memory_hierarchy.l1_cache_size_kb = 64;
        test_config.memory_hierarchy.l2_cache_size_kb = 512;
        test_config.memory_hierarchy.main_memory_size_mb = 8192;
        
        // 创建测试性能指标
        test_metrics.cim_utilization = 0.75f;
        test_metrics.spm_hit_rate = 0.85f;
        test_metrics.memory_bandwidth_usage = 0.6f;
        test_metrics.compute_throughput = 80.0f;
        test_metrics.power_consumption = 200.0f;
        test_metrics.latency_ms = 25.0f;
        test_metrics.per_array_util = std::vector<float>(16, 0.75f);
        test_metrics.timestamp = std::chrono::steady_clock::now();
        
        // 创建测试工作负载特征
        test_workload.compute_intensity = 0.8f;
        test_workload.memory_intensity = 0.4f;
        test_workload.data_reuse_factor = 0.7f;
        test_workload.op_distribution = {0.3f, 0.2f, 0.25f, 0.15f, 0.1f};
        test_workload.batch_size_variability = 0.2f;
    }
    
    YICAConfig test_config;
    PerformanceMetrics test_metrics;
    WorkloadCharacteristics test_workload;
};

// 测试性能指标计算
TEST_F(RuntimeOptimizerTest, TestPerformanceMetricsCalculation) {
    float score = test_metrics.compute_performance_score();
    EXPECT_GT(score, 0.0f);
    EXPECT_LE(score, 1.0f);
    
    // 测试边界情况
    PerformanceMetrics perfect_metrics;
    perfect_metrics.cim_utilization = 1.0f;
    perfect_metrics.spm_hit_rate = 1.0f;
    perfect_metrics.memory_bandwidth_usage = 0.0f;
    perfect_metrics.compute_throughput = 100.0f;
    perfect_metrics.latency_ms = 0.0f;
    
    float perfect_score = perfect_metrics.compute_performance_score();
    EXPECT_NEAR(perfect_score, 1.0f, 0.1f);
}

// 测试工作负载特征分析
TEST_F(RuntimeOptimizerTest, TestWorkloadCharacteristics) {
    float complexity = test_workload.compute_complexity();
    EXPECT_GT(complexity, 0.0f);
    EXPECT_LE(complexity, 1.0f);
    
    // 测试高复杂度工作负载
    WorkloadCharacteristics complex_workload;
    complex_workload.compute_intensity = 1.0f;
    complex_workload.memory_intensity = 1.0f;
    complex_workload.data_reuse_factor = 0.0f;
    complex_workload.batch_size_variability = 1.0f;
    
    float high_complexity = complex_workload.compute_complexity();
    EXPECT_GT(high_complexity, complexity);
}

// 测试优化目标验证
TEST_F(RuntimeOptimizerTest, TestOptimizationObjective) {
    OptimizationObjective objective;
    EXPECT_TRUE(objective.validate_weights());
    
    float obj_value = objective.compute_objective_value(test_metrics);
    EXPECT_GT(obj_value, 0.0f);
    EXPECT_LE(obj_value, 1.0f);
    
    // 测试无效权重
    OptimizationObjective invalid_objective;
    invalid_objective.performance_weight = 0.5f;
    invalid_objective.energy_weight = 0.3f;
    invalid_objective.latency_weight = 0.3f;  // 总和不为1
    EXPECT_FALSE(invalid_objective.validate_weights());
}

// 测试运行时状态管理
TEST_F(RuntimeOptimizerTest, TestRuntimeState) {
    RuntimeState state;
    
    // 添加性能指标到历史
    for (int i = 0; i < 10; ++i) {
        PerformanceMetrics metrics = test_metrics;
        metrics.cim_utilization += i * 0.01f;
        state.add_metrics_to_history(metrics);
    }
    
    EXPECT_EQ(state.metrics_history.size(), 10);
    
    // 获取最近指标
    auto recent = state.get_recent_metrics(5);
    EXPECT_EQ(recent.size(), 5);
    
    // 测试历史清理
    state.cleanup_old_history();
    EXPECT_LE(state.metrics_history.size(), RuntimeState::MAX_HISTORY_SIZE);
}

// 测试时序特征构建
TEST_F(RuntimeOptimizerTest, TestTimeSeriesFeatures) {
    std::vector<PerformanceMetrics> history;
    for (int i = 0; i < 20; ++i) {
        PerformanceMetrics metrics = test_metrics;
        metrics.cim_utilization += i * 0.01f;
        metrics.memory_bandwidth_usage += i * 0.005f;
        history.push_back(metrics);
    }
    
    TimeSeriesFeatures features;
    features.build_from_metrics_history(history);
    
    EXPECT_TRUE(features.is_valid());
    EXPECT_EQ(features.cim_utilization_history.size(), 20);
    EXPECT_EQ(features.memory_access_pattern.size(), 20);
    EXPECT_EQ(features.workload_characteristics.size(), 20);
    
    // 测试特征向量转换
    auto feature_vector = features.to_feature_vector();
    EXPECT_EQ(feature_vector.size(), 60);  // 20 * 3
}

// 测试Q学习状态
TEST_F(RuntimeOptimizerTest, TestQLearningState) {
    QLearningState state1;
    state1.current_metrics = test_metrics;
    state1.current_config = test_config;
    state1.workload = test_workload;
    
    QLearningState state2 = state1;
    EXPECT_TRUE(state1 == state2);
    
    // 修改状态并测试不等性
    state2.current_metrics.cim_utilization += 0.1f;
    EXPECT_FALSE(state1 == state2);
    
    // 测试哈希计算
    size_t hash1 = state1.compute_hash();
    size_t hash2 = state2.compute_hash();
    EXPECT_NE(hash1, hash2);
}

// 测试配置变更请求
TEST_F(RuntimeOptimizerTest, TestConfigChangeRequest) {
    ConfigChangeRequest request;
    request.new_config = test_config;
    request.action = OptimizationAction::INCREASE_CIM_FREQUENCY;
    request.priority = 1.0f;
    request.reason = "Performance optimization";
    request.request_time = std::chrono::steady_clock::now();
    
    EXPECT_FALSE(request.is_expired(5000));  // 5秒超时
    
    // 模拟过期
    request.request_time = std::chrono::steady_clock::now() - std::chrono::milliseconds(6000);
    EXPECT_TRUE(request.is_expired(5000));
}

// 测试性能异常分类
TEST_F(RuntimeOptimizerTest, TestPerformanceAnomaly) {
    PerformanceAnomaly anomaly;
    
    anomaly.severity = 0.9f;
    EXPECT_EQ(anomaly.get_severity_level(), PerformanceAnomaly::Severity::CRITICAL);
    
    anomaly.severity = 0.7f;
    EXPECT_EQ(anomaly.get_severity_level(), PerformanceAnomaly::Severity::HIGH);
    
    anomaly.severity = 0.5f;
    EXPECT_EQ(anomaly.get_severity_level(), PerformanceAnomaly::Severity::MEDIUM);
    
    anomaly.severity = 0.2f;
    EXPECT_EQ(anomaly.get_severity_level(), PerformanceAnomaly::Severity::LOW);
}

// 测试优化结果创建
TEST_F(RuntimeOptimizerTest, TestOptimizationResult) {
    // 测试成功结果
    auto success_result = OptimizationResult::create_success(
        test_config, OptimizationAction::INCREASE_CIM_FREQUENCY, 0.15f, "Performance boost");
    
    EXPECT_TRUE(success_result.success);
    EXPECT_EQ(success_result.recommended_action, OptimizationAction::INCREASE_CIM_FREQUENCY);
    EXPECT_NEAR(success_result.expected_improvement, 0.15f, 1e-6f);
    
    // 测试失败结果
    auto failure_result = OptimizationResult::create_failure("Invalid configuration");
    
    EXPECT_FALSE(failure_result.success);
    EXPECT_EQ(failure_result.recommended_action, OptimizationAction::NO_ACTION);
    EXPECT_NEAR(failure_result.expected_improvement, 0.0f, 1e-6f);
}

// 测试模拟硬件计数器
TEST_F(RuntimeOptimizerTest, TestSimulatedHardwareCounters) {
    auto hw_counters = std::make_unique<SimulatedHardwareCounters>(16);
    
    EXPECT_TRUE(hw_counters->is_available());
    
    auto cim_util = hw_counters->read_cim_utilization();
    EXPECT_EQ(cim_util.size(), 16);
    
    for (float util : cim_util) {
        EXPECT_GE(util, 0.0f);
        EXPECT_LE(util, 1.0f);
    }
    
    float spm_hit = hw_counters->read_spm_hit_rate();
    EXPECT_GE(spm_hit, 0.0f);
    EXPECT_LE(spm_hit, 1.0f);
    
    float memory_bw = hw_counters->read_memory_bandwidth_usage();
    EXPECT_GE(memory_bw, 0.0f);
    EXPECT_LE(memory_bw, 1.0f);
    
    float throughput = hw_counters->read_compute_throughput();
    EXPECT_GE(throughput, 0.0f);
    
    float power = hw_counters->read_power_consumption();
    EXPECT_GE(power, 0.0f);
    
    float latency = hw_counters->read_latency();
    EXPECT_GE(latency, 0.0f);
}

// 测试性能监控器
TEST_F(RuntimeOptimizerTest, TestPerformanceMonitor) {
    PerformanceMonitor monitor;
    
    auto hw_counters = std::make_unique<SimulatedHardwareCounters>(16);
    EXPECT_TRUE(monitor.initialize(std::move(hw_counters)));
    
    EXPECT_TRUE(monitor.start_monitoring());
    EXPECT_TRUE(monitor.is_monitoring());
    
    // 等待一些数据收集
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto current_metrics = monitor.get_current_metrics();
    EXPECT_TRUE(current_metrics.is_valid());
    
    auto average_metrics = monitor.get_average_metrics(50);
    EXPECT_TRUE(average_metrics.is_valid());
    
    monitor.stop_monitoring();
    EXPECT_FALSE(monitor.is_monitoring());
}

// 测试工作负载剖析器
TEST_F(RuntimeOptimizerTest, TestWorkloadProfiler) {
    WorkloadProfiler profiler;
    
    // 添加性能数据
    for (int i = 0; i < 50; ++i) {
        PerformanceMetrics metrics = test_metrics;
        metrics.cim_utilization += (i % 10) * 0.01f;
        metrics.compute_throughput += (i % 5) * 2.0f;
        profiler.add_performance_data(metrics);
    }
    
    EXPECT_EQ(profiler.get_profile_data_size(), 50);
    
    auto characteristics = profiler.analyze_workload_characteristics();
    EXPECT_TRUE(profiler.are_characteristics_valid());
    
    auto cached_characteristics = profiler.get_cached_characteristics();
    EXPECT_NEAR(characteristics.compute_intensity, cached_characteristics.compute_intensity, 1e-6f);
}

// 测试运行时配置管理器
TEST_F(RuntimeOptimizerTest, TestRuntimeConfigManager) {
    RuntimeConfigManager config_manager;
    
    EXPECT_TRUE(config_manager.validate_config(test_config));
    
    EXPECT_TRUE(config_manager.apply_config(test_config, "Initial configuration"));
    
    auto current_config = config_manager.get_current_config();
    EXPECT_EQ(current_config.num_cim_arrays, test_config.num_cim_arrays);
    
    // 测试配置历史
    YICAConfig modified_config = test_config;
    modified_config.cim_frequency_mhz = 1200;
    EXPECT_TRUE(config_manager.apply_config(modified_config, "Frequency boost"));
    
    auto history = config_manager.get_config_history(5);
    EXPECT_EQ(history.size(), 2);
    
    // 测试回滚
    EXPECT_TRUE(config_manager.rollback_to_previous_config());
    
    auto rolled_back_config = config_manager.get_current_config();
    EXPECT_EQ(rolled_back_config.cim_frequency_mhz, test_config.cim_frequency_mhz);
    
    // 测试检查点
    config_manager.save_config_checkpoint("test_checkpoint");
    
    modified_config.spm_size_kb = 2048;
    EXPECT_TRUE(config_manager.apply_config(modified_config, "SPM expansion"));
    
    EXPECT_TRUE(config_manager.restore_config_checkpoint("test_checkpoint"));
    
    auto restored_config = config_manager.get_current_config();
    EXPECT_EQ(restored_config.spm_size_kb, test_config.spm_size_kb);
}

// 测试优化上下文验证
TEST_F(RuntimeOptimizerTest, TestOptimizationContext) {
    OptimizationContext context;
    context.runtime_state.current_metrics = test_metrics;
    context.runtime_state.active_config = test_config;
    context.workload = test_workload;
    context.objective = OptimizationObjective();
    context.candidate_configs = {test_config};
    context.optimization_budget_ms = 100.0f;
    context.enable_ml_optimization = true;
    
    EXPECT_TRUE(context.validate());
    
    // 测试无效上下文
    context.candidate_configs.clear();
    EXPECT_FALSE(context.validate());
    
    context.candidate_configs = {test_config};
    context.optimization_budget_ms = -1.0f;
    EXPECT_FALSE(context.validate());
}

}  // namespace yica
}  // namespace search
}  // namespace yirage 