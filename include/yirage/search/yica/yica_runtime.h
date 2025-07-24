#pragma once

#include <memory>
#include <atomic>
#include <thread>
#include <queue>
#include <condition_variable>
#include "runtime_types.h"
#include "performance_monitor.h"
#include "runtime_optimizer.h"
#include "ml_optimizer.h"

namespace yirage {
namespace search {
namespace yica {

// 配置管理器
class RuntimeConfigManager {
private:
    // 配置历史
    struct ConfigEntry {
        YICAConfig config;
        std::chrono::time_point<std::chrono::steady_clock> timestamp;
        std::string reason;
        bool active = false;
    };
    
    std::vector<ConfigEntry> config_history_;
    mutable std::mutex config_mutex_;
    
    // 配置检查点
    std::unordered_map<std::string, YICAConfig> config_checkpoints_;
    
    // 配置验证规则
    struct ValidationRule {
        std::string name;
        std::function<bool(const YICAConfig&)> validator;
        std::string error_message;
    };
    
    std::vector<ValidationRule> validation_rules_;
    
    // 配置依赖关系
    struct ConfigDependency {
        std::string param_name;
        std::vector<std::string> dependent_params;
        std::function<bool(const YICAConfig&)> dependency_check;
    };
    
    std::vector<ConfigDependency> dependencies_;
    
    // 初始化默认验证规则
    void initialize_default_validation_rules();
    
public:
    RuntimeConfigManager();
    
    // 配置验证
    bool validate_config(const YICAConfig& config) const;
    std::vector<std::string> get_validation_errors(const YICAConfig& config) const;
    
    // 配置转换
    YICAConfig merge_configs(const YICAConfig& base, const YICAConfig& update);
    
    // 配置应用
    bool apply_config(const YICAConfig& new_config, const std::string& reason);
    
    // 配置回滚
    bool rollback_to_previous_config();
    bool rollback_to_timestamp(std::chrono::time_point<std::chrono::steady_clock> timestamp);
    
    // 配置历史管理
    std::vector<YICAConfig> get_config_history(int max_count = 10) const;
    YICAConfig get_current_config() const;
    
    // 配置检查点
    void save_config_checkpoint(const std::string& name);
    bool restore_config_checkpoint(const std::string& name);
    std::vector<std::string> get_checkpoint_names() const;
    
    // 添加验证规则
    void add_validation_rule(const std::string& name,
                           std::function<bool(const YICAConfig&)> validator,
                           const std::string& error_message);
    
    // 添加配置依赖
    void add_config_dependency(const std::string& param_name,
                             const std::vector<std::string>& dependent_params,
                             std::function<bool(const YICAConfig&)> dependency_check);
    
    // 清理历史
    void cleanup_old_history(std::chrono::hours max_age = std::chrono::hours(24));
    
    // 重置管理器
    void reset();
};

// YICA运行时主类
class YICARuntime {
private:
    // 核心组件
    std::unique_ptr<PerformanceMonitor> performance_monitor_;
    std::unique_ptr<RuntimeOptimizer> runtime_optimizer_;
    std::unique_ptr<MLOptimizer> ml_optimizer_;
    std::unique_ptr<RuntimeConfigManager> config_manager_;
    std::unique_ptr<WorkloadProfiler> workload_profiler_;
    
    // 运行时状态
    std::atomic<bool> initialized_;
    std::atomic<bool> running_;
    RuntimeState runtime_state_;
    mutable std::mutex state_mutex_;
    
    // 主控制线程
    std::thread main_control_thread_;
    std::atomic<bool> control_thread_running_;
    
    // 优化任务队列
    std::queue<OptimizationContext> optimization_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread optimization_thread_;
    
    // 回调函数
    PerformanceCallback performance_callback_;
    AnomalyCallback anomaly_callback_;
    ConfigChangeCallback config_change_callback_;
    
    // 配置变更请求队列
    std::queue<ConfigChangeRequest> config_change_queue_;
    std::mutex config_queue_mutex_;
    std::condition_variable config_queue_cv_;
    
    // 统计信息
    struct RuntimeStats {
        std::chrono::time_point<std::chrono::steady_clock> start_time;
        size_t total_optimizations = 0;
        size_t successful_optimizations = 0;
        size_t config_changes = 0;
        size_t anomalies_detected = 0;
        float average_performance_improvement = 0.0f;
        float total_optimization_time_ms = 0.0f;
    };
    
    RuntimeStats stats_;
    mutable std::mutex stats_mutex_;
    
    // 内部方法
    void main_control_loop();
    void optimization_loop();
    void process_config_changes();
    void handle_performance_update(const PerformanceMetrics& metrics);
    void handle_anomaly(const PerformanceAnomaly& anomaly);
    bool apply_configuration_internal(const YICAConfig& new_config);
    
    // 自动优化触发条件
    bool should_trigger_optimization(const PerformanceMetrics& metrics) const;
    OptimizationContext create_optimization_context() const;
    
public:
    YICARuntime();
    ~YICARuntime();
    
    // 初始化运行时系统
    bool initialize(const YICAConfig& initial_config);
    
    // 启动和停止
    bool start();
    bool stop();
    bool is_running() const;
    
    // 监控控制
    bool start_monitoring();
    void stop_monitoring();
    bool is_monitoring() const;
    
    // 优化控制
    OptimizationResult optimize(const OptimizationContext& context);
    OptimizationResult optimize_async(const OptimizationContext& context);
    void enable_auto_optimization(bool enable = true);
    bool is_auto_optimization_enabled() const;
    
    // 配置管理
    bool apply_configuration(const YICAConfig& new_config);
    bool apply_configuration_async(const YICAConfig& new_config, const std::string& reason);
    YICAConfig get_current_configuration() const;
    std::vector<YICAConfig> get_configuration_history(int max_count = 10) const;
    
    // 状态查询
    RuntimeState get_current_state() const;
    PerformanceMetrics get_current_metrics() const;
    PerformanceMetrics get_average_metrics(int window_ms = 1000) const;
    
    // 异常和监控
    std::vector<PerformanceAnomaly> get_detected_anomalies() const;
    std::vector<float> get_performance_trend(const std::string& metric_name) const;
    
    // 回调设置
    void set_performance_callback(PerformanceCallback callback);
    void set_anomaly_callback(AnomalyCallback callback);
    void set_config_change_callback(ConfigChangeCallback callback);
    
    // 机器学习控制
    void enable_ml_optimization(bool enable = true);
    bool is_ml_optimization_enabled() const;
    void update_ml_models(const TimeSeriesFeatures& features,
                         const PerformanceMetrics& actual_performance);
    
    // 工作负载适应
    void adapt_to_workload(const std::string& workload_type);
    WorkloadCharacteristics get_current_workload_characteristics() const;
    
    // 统计和诊断
    struct DiagnosticInfo {
        RuntimeStats stats;
        PerformanceMonitor::MLStats ml_stats;
        RuntimeOptimizer::OptimizationStats opt_stats;
        size_t queue_size;
        bool auto_optimization_enabled;
        bool ml_optimization_enabled;
        std::vector<std::string> active_validation_rules;
        std::vector<std::string> recent_errors;
    };
    
    DiagnosticInfo get_diagnostic_info() const;
    
    // 持久化
    bool save_runtime_state(const std::string& filename) const;
    bool load_runtime_state(const std::string& filename);
    
    // 配置管理
    void save_config_checkpoint(const std::string& name);
    bool restore_config_checkpoint(const std::string& name);
    
    // 重置和清理
    void reset();
    void cleanup_resources();
    
    // 高级优化接口
    struct AdvancedOptimizationOptions {
        bool use_reinforcement_learning = true;
        bool use_multi_objective = true;
        bool use_ml_prediction = true;
        float optimization_budget_ms = 100.0f;
        OptimizationObjective objective;
        std::vector<YICAConfig> candidate_configs;
    };
    
    OptimizationResult optimize_advanced(const AdvancedOptimizationOptions& options);
    
    // 批量优化
    std::vector<OptimizationResult> optimize_batch(
        const std::vector<OptimizationContext>& contexts);
    
    // 预测接口
    PerformanceMetrics predict_performance(const YICAConfig& config) const;
    std::vector<YICAConfig> recommend_configurations(int top_k = 5) const;
    
    // 调试接口
    void set_debug_mode(bool enable);
    std::vector<std::string> get_debug_log() const;
    void dump_internal_state(const std::string& filename) const;
};

// 运行时工厂类
class YICARuntimeFactory {
public:
    // 创建标准运行时
    static std::unique_ptr<YICARuntime> create_standard_runtime(
        const YICAConfig& initial_config);
    
    // 创建高性能运行时
    static std::unique_ptr<YICARuntime> create_high_performance_runtime(
        const YICAConfig& initial_config);
    
    // 创建低功耗运行时
    static std::unique_ptr<YICARuntime> create_low_power_runtime(
        const YICAConfig& initial_config);
    
    // 创建调试运行时
    static std::unique_ptr<YICARuntime> create_debug_runtime(
        const YICAConfig& initial_config);
    
    // 创建自定义运行时
    static std::unique_ptr<YICARuntime> create_custom_runtime(
        const YICAConfig& initial_config,
        const std::vector<std::string>& enabled_features);
};

// 全局运行时管理器
class GlobalRuntimeManager {
private:
    static std::unique_ptr<YICARuntime> global_runtime_;
    static std::mutex global_mutex_;
    
public:
    // 初始化全局运行时
    static bool initialize_global_runtime(const YICAConfig& config);
    
    // 获取全局运行时
    static YICARuntime* get_global_runtime();
    
    // 关闭全局运行时
    static void shutdown_global_runtime();
    
    // 检查全局运行时是否可用
    static bool is_global_runtime_available();
};

}  // namespace yica
}  // namespace search
}  // namespace yirage 