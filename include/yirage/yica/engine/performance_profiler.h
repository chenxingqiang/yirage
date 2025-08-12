/**
 * @file performance_profiler.h
 * @brief 性能分析器 - YICA 综合性能监控和分析
 * 
 * 实现 YICA 的全面性能分析，包括：
 * - 实时性能监控
 * - 详细的性能分析
 * - 瓶颈识别和优化建议
 * - 性能报告生成
 */

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <atomic>
#include <mutex>
#include <thread>
#include <fstream>
#include <functional>

#include "yirage/yica/config.h"

namespace yirage {
namespace yica {

/**
 * @brief 性能事件类型
 */
enum class PerformanceEventType {
    INSTRUCTION_EXECUTION = 0,  ///< 指令执行
    MEMORY_ACCESS = 1,          ///< 内存访问
    CIM_COMPUTATION = 2,        ///< CIM计算
    DATA_TRANSFER = 3,          ///< 数据传输
    SYNCHRONIZATION = 4,        ///< 同步操作
    CACHE_OPERATION = 5,        ///< 缓存操作
    POWER_CONSUMPTION = 6,      ///< 功耗监控
    THERMAL_EVENT = 7           ///< 热事件
};

/**
 * @brief 性能计数器类型
 */
enum class PerformanceCounterType {
    CYCLES = 0,                 ///< 周期计数
    INSTRUCTIONS = 1,           ///< 指令计数
    CACHE_HITS = 2,             ///< 缓存命中
    CACHE_MISSES = 3,           ///< 缓存未命中
    MEMORY_READS = 4,           ///< 内存读取
    MEMORY_WRITES = 5,          ///< 内存写入
    BRANCH_TAKEN = 6,           ///< 分支跳转
    BRANCH_MISPREDICTED = 7,    ///< 分支预测错误
    FLOATING_POINT_OPS = 8,     ///< 浮点操作
    INTEGER_OPS = 9,            ///< 整数操作
    VECTOR_OPS = 10,            ///< 向量操作
    CIM_OPS = 11                ///< CIM操作
};

/**
 * @brief 性能事件
 */
struct PerformanceEvent {
    uint64_t event_id;                          ///< 事件ID
    PerformanceEventType event_type;            ///< 事件类型
    std::chrono::high_resolution_clock::time_point timestamp; ///< 时间戳
    uint64_t duration_ns;                       ///< 持续时间 (纳秒)
    std::string description;                    ///< 事件描述
    std::unordered_map<std::string, double> metrics; ///< 相关指标
    uint32_t thread_id;                         ///< 线程ID
    uint32_t cim_array_id;                      ///< CIM阵列ID
    
    PerformanceEvent() : event_id(0), event_type(PerformanceEventType::INSTRUCTION_EXECUTION),
                        duration_ns(0), thread_id(0), cim_array_id(0) {}
};

/**
 * @brief 性能计数器
 */
struct PerformanceCounter {
    PerformanceCounterType counter_type;        ///< 计数器类型
    std::atomic<uint64_t> count;               ///< 计数值
    std::atomic<uint64_t> peak_value;          ///< 峰值
    std::chrono::steady_clock::time_point last_update; ///< 最后更新时间
    double average_rate;                        ///< 平均速率
    std::string unit;                          ///< 单位
    
    PerformanceCounter() : counter_type(PerformanceCounterType::CYCLES),
                          count(0), peak_value(0), average_rate(0.0) {}
};

/**
 * @brief 性能分析结果
 */
struct PerformanceAnalysis {
    // 整体性能指标
    double total_execution_time_ms;             ///< 总执行时间 (毫秒)
    double cpu_utilization;                     ///< CPU利用率
    double cim_utilization;                     ///< CIM利用率
    double memory_utilization;                  ///< 内存利用率
    double cache_hit_rate;                      ///< 缓存命中率
    
    // 吞吐量指标
    double instructions_per_second;             ///< 每秒指令数
    double operations_per_second;               ///< 每秒操作数
    double memory_bandwidth_gbps;               ///< 内存带宽 (GB/s)
    double compute_throughput_gflops;           ///< 计算吞吐量 (GFLOPS)
    
    // 延迟指标
    double average_instruction_latency_ns;      ///< 平均指令延迟 (ns)
    double average_memory_latency_ns;           ///< 平均内存延迟 (ns)
    double average_cim_latency_ns;              ///< 平均CIM延迟 (ns)
    
    // 能效指标
    double total_energy_consumption_j;          ///< 总能耗 (焦耳)
    double average_power_consumption_w;         ///< 平均功耗 (瓦特)
    double energy_efficiency_gflops_w;          ///< 能效比 (GFLOPS/W)
    
    // 瓶颈分析
    std::vector<std::string> bottlenecks;       ///< 识别的瓶颈
    std::vector<std::string> optimization_suggestions; ///< 优化建议
    
    // 热点分析
    std::vector<std::pair<std::string, double>> hotspots; ///< 热点函数/操作
};

/**
 * @brief 性能分析器配置
 */
struct ProfilerConfig {
    bool enable_real_time_monitoring;          ///< 启用实时监控
    bool enable_detailed_tracing;              ///< 启用详细跟踪
    bool enable_power_monitoring;              ///< 启用功耗监控
    bool enable_thermal_monitoring;            ///< 启用热监控
    uint32_t sampling_interval_ms;             ///< 采样间隔 (毫秒)
    uint32_t max_events_in_memory;             ///< 内存中最大事件数
    std::string output_directory;              ///< 输出目录
    std::vector<PerformanceEventType> monitored_events; ///< 监控的事件类型
    
    ProfilerConfig() : enable_real_time_monitoring(true),
                      enable_detailed_tracing(false),
                      enable_power_monitoring(true),
                      enable_thermal_monitoring(true),
                      sampling_interval_ms(100),
                      max_events_in_memory(100000),
                      output_directory("./profiler_output") {}
};

/**
 * @brief YICA 性能分析器
 * 
 * 提供全面的性能监控和分析功能，包括：
 * - 实时性能数据收集
 * - 详细的性能分析报告
 * - 瓶颈识别和优化建议
 * - 可视化性能数据
 */
class PerformanceProfiler {
public:
    /**
     * @brief 构造函数
     * @param config YICA 配置
     * @param profiler_config 分析器配置
     */
    PerformanceProfiler(const YICAConfig& config, const ProfilerConfig& profiler_config);
    
    /**
     * @brief 析构函数
     */
    ~PerformanceProfiler();
    
    // 禁用拷贝和赋值
    PerformanceProfiler(const PerformanceProfiler&) = delete;
    PerformanceProfiler& operator=(const PerformanceProfiler&) = delete;
    
    /**
     * @brief 启动性能分析器
     * @return 成功返回 true
     */
    bool start();
    
    /**
     * @brief 停止性能分析器
     */
    void stop();
    
    /**
     * @brief 暂停性能分析
     */
    void pause();
    
    /**
     * @brief 恢复性能分析
     */
    void resume();
    
    /**
     * @brief 记录性能事件
     * @param event 性能事件
     */
    void record_event(const PerformanceEvent& event);
    
    /**
     * @brief 开始性能区间
     * @param name 区间名称
     * @param event_type 事件类型
     * @return 区间ID
     */
    uint64_t begin_region(const std::string& name, 
                         PerformanceEventType event_type = PerformanceEventType::INSTRUCTION_EXECUTION);
    
    /**
     * @brief 结束性能区间
     * @param region_id 区间ID
     */
    void end_region(uint64_t region_id);
    
    /**
     * @brief 增加计数器
     * @param counter_type 计数器类型
     * @param increment 增量
     */
    void increment_counter(PerformanceCounterType counter_type, uint64_t increment = 1);
    
    /**
     * @brief 设置计数器值
     * @param counter_type 计数器类型
     * @param value 新值
     */
    void set_counter(PerformanceCounterType counter_type, uint64_t value);
    
    /**
     * @brief 获取计数器值
     * @param counter_type 计数器类型
     * @return 计数器值
     */
    uint64_t get_counter(PerformanceCounterType counter_type) const;
    
    /**
     * @brief 记录功耗数据
     * @param power_w 功耗 (瓦特)
     * @param component 组件名称
     */
    void record_power_consumption(double power_w, const std::string& component = "total");
    
    /**
     * @brief 记录温度数据
     * @param temperature_c 温度 (摄氏度)
     * @param sensor_name 传感器名称
     */
    void record_temperature(double temperature_c, const std::string& sensor_name = "core");
    
    /**
     * @brief 生成性能分析报告
     * @return 性能分析结果
     */
    PerformanceAnalysis generate_analysis();
    
    /**
     * @brief 导出性能数据
     * @param filename 文件名
     * @param format 格式 ("json", "csv", "xml")
     * @return 成功返回 true
     */
    bool export_data(const std::string& filename, const std::string& format = "json");
    
    /**
     * @brief 生成性能报告
     * @param filename 报告文件名
     * @param format 报告格式 ("html", "pdf", "markdown")
     * @return 成功返回 true
     */
    bool generate_report(const std::string& filename, const std::string& format = "html");
    
    /**
     * @brief 获取实时性能指标
     * @return 当前性能指标映射
     */
    std::unordered_map<std::string, double> get_real_time_metrics() const;
    
    /**
     * @brief 设置性能回调函数
     * @param callback 回调函数
     */
    void set_performance_callback(std::function<void(const PerformanceAnalysis&)> callback);
    
    /**
     * @brief 重置所有统计数据
     */
    void reset_statistics();
    
    /**
     * @brief 获取分析器状态
     * @return 是否正在运行
     */
    bool is_running() const { return running_.load(); }
    
    /**
     * @brief 获取分析器配置
     * @return 配置信息
     */
    const ProfilerConfig& get_config() const { return profiler_config_; }
    
    /**
     * @brief 更新配置
     * @param new_config 新配置
     */
    void update_config(const ProfilerConfig& new_config);

private:
    // 配置和状态
    YICAConfig yica_config_;
    ProfilerConfig profiler_config_;
    std::atomic<bool> running_;
    std::atomic<bool> paused_;
    
    // 事件存储
    std::vector<PerformanceEvent> events_;
    std::mutex events_mutex_;
    uint64_t next_event_id_;
    
    // 性能计数器
    std::unordered_map<PerformanceCounterType, PerformanceCounter> counters_;
    std::mutex counters_mutex_;
    
    // 区间跟踪
    std::unordered_map<uint64_t, std::pair<std::string, std::chrono::high_resolution_clock::time_point>> 
        active_regions_;
    std::mutex regions_mutex_;
    uint64_t next_region_id_;
    
    // 功耗和温度数据
    std::vector<std::pair<std::chrono::steady_clock::time_point, double>> power_history_;
    std::vector<std::pair<std::chrono::steady_clock::time_point, double>> temperature_history_;
    std::mutex thermal_mutex_;
    
    // 监控线程
    std::thread monitoring_thread_;
    std::thread analysis_thread_;
    
    // 回调函数
    std::function<void(const PerformanceAnalysis&)> performance_callback_;
    std::mutex callback_mutex_;
    
    // 输出流
    std::ofstream trace_file_;
    std::mutex output_mutex_;
    
    // 私有方法：监控
    void monitoring_thread_function();
    void analysis_thread_function();
    void collect_system_metrics();
    void collect_hardware_counters();
    
    // 私有方法：分析
    PerformanceAnalysis analyze_events() const;
    std::vector<std::string> identify_bottlenecks(const PerformanceAnalysis& analysis) const;
    std::vector<std::string> generate_optimization_suggestions(const PerformanceAnalysis& analysis) const;
    std::vector<std::pair<std::string, double>> identify_hotspots() const;
    
    // 私有方法：计算
    double calculate_utilization(PerformanceCounterType counter_type) const;
    double calculate_throughput(PerformanceCounterType counter_type) const;
    double calculate_average_latency(PerformanceEventType event_type) const;
    double calculate_energy_efficiency() const;
    
    // 私有方法：输出
    bool export_to_json(const std::string& filename) const;
    bool export_to_csv(const std::string& filename) const;
    bool export_to_xml(const std::string& filename) const;
    bool generate_html_report(const std::string& filename, const PerformanceAnalysis& analysis) const;
    bool generate_markdown_report(const std::string& filename, const PerformanceAnalysis& analysis) const;
    
    // 私有方法：工具函数
    std::string event_type_to_string(PerformanceEventType type) const;
    std::string counter_type_to_string(PerformanceCounterType type) const;
    void log_debug(const std::string& message) const;
    void cleanup_old_events();
    
    // 常量
    static constexpr uint32_t DEFAULT_MONITORING_INTERVAL_MS = 100;
    static constexpr uint32_t DEFAULT_ANALYSIS_INTERVAL_MS = 1000;
    static constexpr size_t MAX_POWER_HISTORY_SIZE = 10000;
    static constexpr size_t MAX_TEMPERATURE_HISTORY_SIZE = 10000;
};

/**
 * @brief 性能分析器工厂类
 */
class PerformanceProfilerFactory {
public:
    /**
     * @brief 创建性能分析器
     * @param yica_config YICA 配置
     * @param profiler_config 分析器配置
     * @return 性能分析器实例
     */
    static std::unique_ptr<PerformanceProfiler> create(
        const YICAConfig& yica_config,
        const ProfilerConfig& profiler_config = ProfilerConfig()
    );
    
    /**
     * @brief 创建轻量级分析器
     * @param yica_config YICA 配置
     * @return 轻量级分析器实例
     */
    static std::unique_ptr<PerformanceProfiler> create_lightweight(
        const YICAConfig& yica_config
    );
    
    /**
     * @brief 创建详细分析器
     * @param yica_config YICA 配置
     * @return 详细分析器实例
     */
    static std::unique_ptr<PerformanceProfiler> create_detailed(
        const YICAConfig& yica_config
    );
};

/**
 * @brief 性能分析器辅助宏
 */
#define YICA_PROFILE_REGION(profiler, name) \
    auto __region_id = profiler->begin_region(name); \
    auto __region_guard = std::unique_ptr<void, std::function<void(void*)>>( \
        nullptr, [profiler, __region_id](void*) { profiler->end_region(__region_id); })

#define YICA_PROFILE_FUNCTION(profiler) \
    YICA_PROFILE_REGION(profiler, __FUNCTION__)

#define YICA_INCREMENT_COUNTER(profiler, counter_type) \
    profiler->increment_counter(counter_type)

#define YICA_RECORD_POWER(profiler, power_w) \
    profiler->record_power_consumption(power_w)

} // namespace yica
} // namespace yirage 