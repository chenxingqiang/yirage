#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include "yica_types.h"

namespace yirage {
namespace search {
namespace yica {

// 性能指标结构
struct PerformanceMetrics {
    float cim_utilization = 0.0f;           // CIM阵列利用率 [0,1]
    float spm_hit_rate = 0.0f;              // SPM命中率 [0,1]
    float memory_bandwidth_usage = 0.0f;     // 内存带宽使用率 [0,1]
    float compute_throughput = 0.0f;         // 计算吞吐量 TOPS
    float power_consumption = 0.0f;          // 功耗 W
    float latency_ms = 0.0f;                // 延迟 ms
    std::vector<float> per_array_util;       // 每个CIM阵列利用率
    
    // 时间戳
    std::chrono::time_point<std::chrono::steady_clock> timestamp;
    
    // 计算综合性能得分
    float compute_performance_score() const;
    
    // 检查是否有效
    bool is_valid() const;
};

// 工作负载特征
struct WorkloadCharacteristics {
    float compute_intensity = 0.0f;          // 计算密集度
    float memory_intensity = 0.0f;           // 内存密集度
    float data_reuse_factor = 0.0f;         // 数据重用因子
    std::vector<float> op_distribution;      // 操作分布
    float batch_size_variability = 0.0f;    // 批大小变化
    
    // 计算工作负载复杂度
    float compute_complexity() const;
};

// 优化动作枚举
enum class OptimizationAction {
    INCREASE_CIM_FREQUENCY,    // 提高CIM频率
    DECREASE_CIM_FREQUENCY,    // 降低CIM频率
    REDISTRIBUTE_SPM,          // 重新分配SPM
    CHANGE_FUSION_STRATEGY,    // 改变融合策略
    ADJUST_PARALLELISM,        // 调整并行度
    ENABLE_PREFETCH,           // 启用预取
    DISABLE_PREFETCH,          // 禁用预取
    NO_ACTION                  // 无动作
};

// 优化目标
struct OptimizationObjective {
    float performance_weight = 0.6f;  // 性能权重
    float energy_weight = 0.3f;      // 能效权重
    float latency_weight = 0.1f;     // 延迟权重
    
    // 验证权重和为1
    bool validate_weights() const;
    
    // 计算目标函数值
    float compute_objective_value(const PerformanceMetrics& metrics) const;
};

// 运行时状态
struct RuntimeState {
    PerformanceMetrics current_metrics;     // 当前性能指标
    YICAConfig active_config;              // 当前配置
    std::vector<OptimizationAction> action_history;  // 动作历史
    std::chrono::time_point<std::chrono::steady_clock> last_update; // 最后更新时间
    
    // 性能历史记录
    std::deque<PerformanceMetrics> metrics_history;
    static constexpr size_t MAX_HISTORY_SIZE = 1000;
    
    // 添加性能指标到历史
    void add_metrics_to_history(const PerformanceMetrics& metrics);
    
    // 获取历史指标
    std::vector<PerformanceMetrics> get_recent_metrics(size_t count) const;
    
    // 清理旧的历史数据
    void cleanup_old_history();
};

// 优化上下文
struct OptimizationContext {
    RuntimeState runtime_state;             // 运行时状态
    WorkloadCharacteristics workload;       // 工作负载特征
    OptimizationObjective objective;        // 优化目标
    std::vector<YICAConfig> candidate_configs; // 候选配置
    float optimization_budget_ms = 100.0f;  // 优化时间预算
    bool enable_ml_optimization = true;      // 是否启用ML优化
    
    // 验证上下文有效性
    bool validate() const;
};

// 优化结果
struct OptimizationResult {
    bool success = false;                   // 优化是否成功
    YICAConfig recommended_config;          // 推荐配置
    OptimizationAction recommended_action;   // 推荐动作
    float expected_improvement = 0.0f;       // 预期改进幅度
    std::string optimization_reason;        // 优化原因
    float optimization_time_ms = 0.0f;      // 优化耗时
    
    // 创建成功结果
    static OptimizationResult create_success(
        const YICAConfig& config,
        OptimizationAction action,
        float improvement,
        const std::string& reason);
    
    // 创建失败结果
    static OptimizationResult create_failure(const std::string& reason);
};

// 时序特征（用于机器学习）
struct TimeSeriesFeatures {
    std::vector<float> cim_utilization_history;
    std::vector<float> memory_access_pattern;
    std::vector<float> workload_characteristics;
    int sequence_length = 50;  // 历史窗口长度
    
    // 从性能指标历史构建特征
    void build_from_metrics_history(const std::vector<PerformanceMetrics>& history);
    
    // 验证特征有效性
    bool is_valid() const;
    
    // 获取特征向量
    std::vector<float> to_feature_vector() const;
};

// Q-learning状态表示
struct QLearningState {
    PerformanceMetrics current_metrics;
    YICAConfig current_config;
    WorkloadCharacteristics workload;
    
    // 状态哈希（用于Q表索引）
    size_t compute_hash() const;
    
    // 状态比较
    bool operator==(const QLearningState& other) const;
};

// 配置变更请求
struct ConfigChangeRequest {
    YICAConfig new_config;
    OptimizationAction action;
    float priority = 1.0f;
    std::string reason;
    std::chrono::time_point<std::chrono::steady_clock> request_time;
    
    // 检查请求是否过期
    bool is_expired(int timeout_ms = 5000) const;
};

// 性能异常
struct PerformanceAnomaly {
    std::string type;           // 异常类型
    std::string description;    // 异常描述
    float severity = 0.0f;      // 严重程度 [0,1]
    PerformanceMetrics metrics; // 异常时的性能指标
    std::chrono::time_point<std::chrono::steady_clock> detection_time;
    
    // 异常严重程度分类
    enum class Severity {
        LOW,      // 轻微异常
        MEDIUM,   // 中等异常
        HIGH,     // 严重异常
        CRITICAL  // 临界异常
    };
    
    Severity get_severity_level() const;
};

// 回调函数类型定义
using PerformanceCallback = std::function<void(const PerformanceMetrics&)>;
using AnomalyCallback = std::function<void(const PerformanceAnomaly&)>;
using ConfigChangeCallback = std::function<void(const YICAConfig&, const YICAConfig&)>;

}  // namespace yica
}  // namespace search
}  // namespace yirage 