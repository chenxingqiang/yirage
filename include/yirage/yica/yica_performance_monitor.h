#pragma once

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include "yirage/yica/config.h"
#include "yirage/yica/yica_kernel_generator.h"

namespace yirage {
namespace yica {

// 性能计数器类型
enum class PerformanceCounterType {
    CIM_UTILIZATION,        // CIM 阵列利用率
    SPM_HIT_RATE,          // SPM 命中率
    DRAM_BANDWIDTH,        // DRAM 带宽利用率
    INSTRUCTION_THROUGHPUT, // 指令吞吐量
    ENERGY_CONSUMPTION,     // 能耗
    TEMPERATURE,           // 温度
    MEMORY_USAGE,          // 内存使用量
    COMPUTE_EFFICIENCY,    // 计算效率
    COMMUNICATION_LATENCY, // 通信延迟
    CACHE_MISS_RATE       // 缓存缺失率
};

// 性能指标数据
struct PerformanceMetric {
    PerformanceCounterType type;
    std::string name;
    double value;
    std::string unit;
    std::chrono::high_resolution_clock::time_point timestamp;
    std::map<std::string, std::string> metadata;
};

// 性能监控配置
struct PerformanceMonitorConfig {
    std::vector<PerformanceCounterType> enabled_counters;
    std::chrono::milliseconds sampling_interval{100};  // 采样间隔
    size_t max_history_size{10000};                   // 最大历史记录数
    bool enable_real_time_analysis{true};             // 启用实时分析
    bool enable_anomaly_detection{true};              // 启用异常检测
    bool enable_auto_tuning{true};                    // 启用自动调优
    std::string log_file_path{"yica_performance.log"}; // 日志文件路径
};

// 性能异常类型
enum class PerformanceAnomalyType {
    HIGH_LATENCY,           // 高延迟
    LOW_UTILIZATION,        // 低利用率
    MEMORY_LEAK,           // 内存泄漏
    THERMAL_THROTTLING,    // 温度限制
    BANDWIDTH_SATURATION,  // 带宽饱和
    CACHE_THRASHING,       // 缓存抖动
    POWER_SPIKE,           // 功耗峰值
    DEADLOCK_DETECTION     // 死锁检测
};

// 性能异常报告
struct PerformanceAnomaly {
    PerformanceAnomalyType type;
    std::string description;
    double severity_score;  // 严重程度评分 (0-1)
    std::chrono::high_resolution_clock::time_point detection_time;
    std::vector<PerformanceMetric> related_metrics;
    std::vector<std::string> suggested_actions;
};

// 调优建议
struct TuningRecommendation {
    std::string parameter_name;
    std::string current_value;
    std::string recommended_value;
    double expected_improvement;  // 预期提升百分比
    std::string justification;    // 调优理由
    int priority;                // 优先级 (1-10)
};

// 自动调优配置
struct AutoTuningConfig {
    bool enable_aggressive_tuning{false};    // 启用激进调优
    double improvement_threshold{0.05};      // 最小改进阈值 (5%)
    int max_tuning_iterations{50};           // 最大调优迭代次数
    std::chrono::seconds tuning_interval{300}; // 调优间隔 (5分钟)
    std::vector<std::string> tunable_parameters; // 可调优参数列表
    std::map<std::string, std::pair<double, double>> parameter_bounds; // 参数边界
};

// YICA 性能监控器
class YICAPerformanceMonitor {
public:
    explicit YICAPerformanceMonitor(const YICAConfig& yica_config,
                                   const PerformanceMonitorConfig& monitor_config);
    ~YICAPerformanceMonitor();
    
    // 生命周期管理
    bool initialize();
    void start_monitoring();
    void stop_monitoring();
    void finalize();
    
    // 性能计数器管理
    void register_counter(PerformanceCounterType type, const std::string& name);
    void unregister_counter(PerformanceCounterType type);
    void update_counter(PerformanceCounterType type, double value);
    
    // 实时监控
    std::vector<PerformanceMetric> get_current_metrics() const;
    std::vector<PerformanceMetric> get_historical_metrics(
        PerformanceCounterType type,
        std::chrono::high_resolution_clock::time_point start_time,
        std::chrono::high_resolution_clock::time_point end_time
    ) const;
    
    // 性能分析
    struct PerformanceAnalysis {
        double average_cim_utilization;
        double peak_memory_usage;
        double average_throughput;
        double energy_efficiency;
        std::vector<std::string> bottlenecks;
        std::vector<std::string> optimization_opportunities;
    };
    
    PerformanceAnalysis analyze_performance(
        std::chrono::high_resolution_clock::time_point start_time,
        std::chrono::high_resolution_clock::time_point end_time
    ) const;
    
    // 异常检测
    void enable_anomaly_detection(bool enable);
    std::vector<PerformanceAnomaly> get_detected_anomalies() const;
    void register_anomaly_callback(std::function<void(const PerformanceAnomaly&)> callback);
    
    // 基准测试
    struct BenchmarkResult {
        std::string test_name;
        double execution_time;
        double throughput;
        double energy_consumption;
        std::map<PerformanceCounterType, double> counter_values;
    };
    
    BenchmarkResult run_benchmark(const std::string& test_name,
                                 std::function<void()> benchmark_function);
    
    std::vector<BenchmarkResult> run_comprehensive_benchmark();
    
    // 性能报告
    void generate_performance_report(const std::string& output_path) const;
    std::string get_performance_summary() const;
    
    // 配置管理
    void update_monitor_config(const PerformanceMonitorConfig& new_config);
    PerformanceMonitorConfig get_monitor_config() const { return monitor_config_; }

private:
    YICAConfig yica_config_;
    PerformanceMonitorConfig monitor_config_;
    
    // 监控状态
    bool initialized_;
    bool monitoring_active_;
    std::thread monitoring_thread_;
    mutable std::mutex metrics_mutex_;
    std::condition_variable monitoring_cv_;
    
    // 性能数据存储
    std::map<PerformanceCounterType, std::queue<PerformanceMetric>> metrics_history_;
    std::map<PerformanceCounterType, PerformanceMetric> current_metrics_;
    
    // 异常检测
    std::vector<PerformanceAnomaly> detected_anomalies_;
    std::vector<std::function<void(const PerformanceAnomaly&)>> anomaly_callbacks_;
    mutable std::mutex anomaly_mutex_;
    
    // 内部方法
    void monitoring_loop();
    void collect_system_metrics();
    void collect_cim_metrics();
    void collect_memory_metrics();
    void collect_energy_metrics();
    
    // 异常检测算法
    bool detect_high_latency_anomaly(const std::vector<PerformanceMetric>& metrics);
    bool detect_low_utilization_anomaly(const std::vector<PerformanceMetric>& metrics);
    bool detect_memory_leak_anomaly(const std::vector<PerformanceMetric>& metrics);
    bool detect_thermal_throttling_anomaly(const std::vector<PerformanceMetric>& metrics);
    
    void notify_anomaly(const PerformanceAnomaly& anomaly);
    
    // 数据管理
    void cleanup_old_metrics();
    void save_metrics_to_disk() const;
    void load_metrics_from_disk();
};

// YICA 自动调优器
class YICAAutoTuner {
public:
    explicit YICAAutoTuner(const YICAConfig& yica_config,
                          const AutoTuningConfig& tuning_config,
                          YICAPerformanceMonitor* performance_monitor);
    ~YICAAutoTuner();
    
    // 调优管理
    bool initialize();
    void start_auto_tuning();
    void stop_auto_tuning();
    void finalize();
    
    // 手动调优
    std::vector<TuningRecommendation> analyze_and_recommend();
    bool apply_tuning_recommendation(const TuningRecommendation& recommendation);
    void apply_all_recommendations(const std::vector<TuningRecommendation>& recommendations);
    
    // 调优历史
    struct TuningHistory {
        std::chrono::high_resolution_clock::time_point timestamp;
        std::string parameter_name;
        std::string old_value;
        std::string new_value;
        double performance_before;
        double performance_after;
        double improvement_ratio;
        bool successful;
    };
    
    std::vector<TuningHistory> get_tuning_history() const;
    void clear_tuning_history();
    
    // 调优策略
    enum class TuningStrategy {
        CONSERVATIVE,    // 保守策略
        BALANCED,       // 平衡策略
        AGGRESSIVE,     // 激进策略
        CUSTOM          // 自定义策略
    };
    
    void set_tuning_strategy(TuningStrategy strategy);
    TuningStrategy get_tuning_strategy() const { return tuning_strategy_; }
    
    // 参数空间搜索
    struct ParameterSpace {
        std::string name;
        std::vector<std::string> possible_values;
        std::string current_value;
        int search_priority;
    };
    
    void define_parameter_space(const std::vector<ParameterSpace>& spaces);
    std::vector<ParameterSpace> get_parameter_spaces() const;
    
    // 优化算法
    enum class OptimizationAlgorithm {
        GRID_SEARCH,        // 网格搜索
        RANDOM_SEARCH,      // 随机搜索
        BAYESIAN_OPTIMIZATION, // 贝叶斯优化
        GENETIC_ALGORITHM,  // 遗传算法
        SIMULATED_ANNEALING // 模拟退火
    };
    
    void set_optimization_algorithm(OptimizationAlgorithm algorithm);
    OptimizationAlgorithm get_optimization_algorithm() const { return optimization_algorithm_; }

private:
    YICAConfig yica_config_;
    AutoTuningConfig tuning_config_;
    YICAPerformanceMonitor* performance_monitor_;
    
    // 调优状态
    bool initialized_;
    bool tuning_active_;
    std::thread tuning_thread_;
    TuningStrategy tuning_strategy_;
    OptimizationAlgorithm optimization_algorithm_;
    
    // 参数管理
    std::vector<ParameterSpace> parameter_spaces_;
    std::vector<TuningHistory> tuning_history_;
    mutable std::mutex tuning_mutex_;
    
    // 内部方法
    void auto_tuning_loop();
    double evaluate_current_performance();
    std::vector<TuningRecommendation> generate_recommendations();
    
    // 调优算法实现
    std::vector<TuningRecommendation> grid_search_tuning();
    std::vector<TuningRecommendation> random_search_tuning();
    std::vector<TuningRecommendation> bayesian_optimization_tuning();
    std::vector<TuningRecommendation> genetic_algorithm_tuning();
    std::vector<TuningRecommendation> simulated_annealing_tuning();
    
    // 参数操作
    bool set_parameter_value(const std::string& parameter_name, const std::string& value);
    std::string get_parameter_value(const std::string& parameter_name) const;
    bool validate_parameter_value(const std::string& parameter_name, const std::string& value) const;
    
    // 性能评估
    double calculate_performance_score(const YICAPerformanceMonitor::PerformanceAnalysis& analysis);
    bool is_improvement_significant(double old_score, double new_score) const;
    
    void record_tuning_attempt(const TuningRecommendation& recommendation,
                              double performance_before, double performance_after, bool successful);
};

// 性能分析工具
class YICAPerformanceAnalyzer {
public:
    explicit YICAPerformanceAnalyzer(YICAPerformanceMonitor* monitor);
    
    // 瓶颈分析
    struct BottleneckAnalysis {
        std::string component_name;
        double utilization_ratio;
        double impact_score;
        std::vector<std::string> causes;
        std::vector<std::string> solutions;
    };
    
    std::vector<BottleneckAnalysis> identify_bottlenecks();
    
    // 效率分析
    struct EfficiencyAnalysis {
        double compute_efficiency;      // 计算效率
        double memory_efficiency;      // 内存效率
        double energy_efficiency;      // 能效
        double communication_efficiency; // 通信效率
        std::string limiting_factor;   // 限制因子
    };
    
    EfficiencyAnalysis analyze_efficiency();
    
    // 趋势分析
    struct TrendAnalysis {
        std::string metric_name;
        double trend_slope;            // 趋势斜率
        double correlation_coefficient; // 相关系数
        std::string trend_direction;   // 趋势方向
        double prediction_accuracy;    // 预测准确度
    };
    
    std::vector<TrendAnalysis> analyze_performance_trends();
    
    // 对比分析
    struct ComparisonResult {
        std::string baseline_name;
        std::string comparison_name;
        std::map<std::string, double> performance_ratios;
        std::vector<std::string> improvements;
        std::vector<std::string> regressions;
    };
    
    ComparisonResult compare_performance_profiles(const std::string& baseline_name,
                                                 const std::string& comparison_name);

private:
    YICAPerformanceMonitor* performance_monitor_;
    
    // 分析算法
    double calculate_utilization_score(const std::vector<PerformanceMetric>& metrics);
    std::vector<std::string> identify_root_causes(const std::vector<PerformanceMetric>& metrics);
    double predict_performance_trend(const std::vector<PerformanceMetric>& historical_data);
};

// 性能可视化工具
class YICAPerformanceVisualizer {
public:
    explicit YICAPerformanceVisualizer(YICAPerformanceMonitor* monitor);
    
    // 图表生成
    void generate_utilization_chart(const std::string& output_path);
    void generate_throughput_chart(const std::string& output_path);
    void generate_energy_efficiency_chart(const std::string& output_path);
    void generate_memory_usage_chart(const std::string& output_path);
    
    // 实时仪表板
    void start_real_time_dashboard(int port = 8080);
    void stop_real_time_dashboard();
    
    // 报告生成
    void generate_html_report(const std::string& output_path);
    void generate_pdf_report(const std::string& output_path);

private:
    YICAPerformanceMonitor* performance_monitor_;
    bool dashboard_active_;
    std::thread dashboard_thread_;
    
    void dashboard_server_loop(int port);
};

// 工具函数
namespace performance_utils {
    
    // 指标计算
    double calculate_efficiency_ratio(double actual_performance, double theoretical_peak);
    double calculate_energy_efficiency(double performance, double power_consumption);
    double calculate_memory_bandwidth_utilization(double actual_bandwidth, double peak_bandwidth);
    
    // 异常检测辅助函数
    bool is_outlier(double value, const std::vector<double>& baseline_values, double threshold = 2.0);
    double calculate_anomaly_score(const PerformanceMetric& metric, 
                                  const std::vector<PerformanceMetric>& historical_data);
    
    // 调优辅助函数
    std::vector<std::string> suggest_optimization_strategies(const std::vector<BottleneckAnalysis>& bottlenecks);
    double estimate_tuning_impact(const TuningRecommendation& recommendation,
                                 const YICAPerformanceMonitor::PerformanceAnalysis& current_analysis);
    
    // 数据处理
    std::vector<double> smooth_time_series(const std::vector<double>& data, int window_size = 5);
    double calculate_moving_average(const std::vector<double>& data, int window_size);
    std::pair<double, double> calculate_linear_trend(const std::vector<std::pair<double, double>>& data_points);
    
} // namespace performance_utils

} // namespace yica
} // namespace yirage 