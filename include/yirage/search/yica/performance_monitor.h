#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include "runtime_types.h"

namespace yirage {
namespace search {
namespace yica {

// 硬件性能计数器接口
class HardwareCounters {
public:
    virtual ~HardwareCounters() = default;
    
    // 读取CIM阵列利用率
    virtual std::vector<float> read_cim_utilization() = 0;
    
    // 读取SPM命中率
    virtual float read_spm_hit_rate() = 0;
    
    // 读取内存带宽使用
    virtual float read_memory_bandwidth_usage() = 0;
    
    // 读取计算吞吐量
    virtual float read_compute_throughput() = 0;
    
    // 读取功耗
    virtual float read_power_consumption() = 0;
    
    // 读取延迟
    virtual float read_latency() = 0;
    
    // 检查计数器是否可用
    virtual bool is_available() const = 0;
};

// 模拟硬件计数器（用于测试）
class SimulatedHardwareCounters : public HardwareCounters {
private:
    int num_cim_arrays_;
    mutable std::mt19937 rng_;
    
public:
    explicit SimulatedHardwareCounters(int num_cim_arrays = 16);
    
    std::vector<float> read_cim_utilization() override;
    float read_spm_hit_rate() override;
    float read_memory_bandwidth_usage() override;
    float read_compute_throughput() override;
    float read_power_consumption() override;
    float read_latency() override;
    bool is_available() const override;
    
    // 设置模拟参数
    void set_base_utilization(float base_util);
    void set_noise_level(float noise);
};

// 指标收集器
class MetricsCollector {
private:
    std::unique_ptr<HardwareCounters> hw_counters_;
    std::atomic<bool> collecting_;
    std::thread collection_thread_;
    mutable std::mutex metrics_mutex_;
    PerformanceMetrics latest_metrics_;
    
    // 收集间隔（毫秒）
    int collection_interval_ms_ = 1;  // 1000Hz
    
    // 内部收集循环
    void collection_loop();
    
public:
    explicit MetricsCollector(std::unique_ptr<HardwareCounters> hw_counters);
    ~MetricsCollector();
    
    // 启动收集
    bool start_collection();
    
    // 停止收集
    void stop_collection();
    
    // 获取最新指标
    PerformanceMetrics get_latest_metrics() const;
    
    // 设置收集间隔
    void set_collection_interval(int interval_ms);
    
    // 检查是否正在收集
    bool is_collecting() const;
};

// 滑动窗口监控器
class SlidingWindowMonitor {
private:
    std::deque<PerformanceMetrics> window_;
    mutable std::mutex window_mutex_;
    size_t window_size_;
    std::chrono::milliseconds window_duration_;
    
    // 清理过期数据
    void cleanup_expired_metrics();
    
public:
    SlidingWindowMonitor(size_t window_size = 1000, 
                        std::chrono::milliseconds window_duration = std::chrono::seconds(10));
    
    // 更新指标
    void update_metrics(const PerformanceMetrics& metrics);
    
    // 获取平均指标
    PerformanceMetrics get_average_metrics(int window_size_ms = 1000) const;
    
    // 获取最近指标
    std::vector<PerformanceMetrics> get_recent_metrics(size_t count) const;
    
    // 检测性能异常
    bool detect_performance_anomaly() const;
    
    // 识别瓶颈
    std::vector<std::string> identify_bottlenecks() const;
    
    // 获取性能趋势
    std::vector<float> get_performance_trend(const std::string& metric_name) const;
    
    // 清空窗口
    void clear();
    
    // 获取窗口大小
    size_t get_window_size() const;
};

// 异常检测器
class AnomalyDetector {
private:
    struct AnomalyThresholds {
        float cim_util_low = 0.1f;      // CIM利用率过低阈值
        float cim_util_high = 0.95f;    // CIM利用率过高阈值
        float spm_hit_low = 0.6f;       // SPM命中率过低阈值
        float memory_bw_high = 0.9f;    // 内存带宽过高阈值
        float latency_high = 100.0f;    // 延迟过高阈值
        float power_high = 300.0f;      // 功耗过高阈值
    };
    
    AnomalyThresholds thresholds_;
    std::vector<PerformanceAnomaly> detected_anomalies_;
    mutable std::mutex anomalies_mutex_;
    
    // 检测特定类型异常
    std::vector<PerformanceAnomaly> check_utilization_anomalies(
        const PerformanceMetrics& metrics) const;
    std::vector<PerformanceAnomaly> check_memory_anomalies(
        const PerformanceMetrics& metrics) const;
    std::vector<PerformanceAnomaly> check_performance_anomalies(
        const PerformanceMetrics& metrics) const;
    
public:
    AnomalyDetector();
    
    // 检测异常
    std::vector<PerformanceAnomaly> detect_anomalies(
        const PerformanceMetrics& metrics);
    
    // 检测异常（基于历史数据）
    std::vector<PerformanceAnomaly> detect_anomalies_with_history(
        const std::vector<PerformanceMetrics>& metrics_history);
    
    // 设置阈值
    void set_thresholds(const AnomalyThresholds& thresholds);
    
    // 获取已检测异常
    std::vector<PerformanceAnomaly> get_detected_anomalies() const;
    
    // 清除异常历史
    void clear_anomaly_history();
    
    // 更新阈值（自适应）
    void update_adaptive_thresholds(const std::vector<PerformanceMetrics>& history);
};

// 性能监控器主类
class PerformanceMonitor {
private:
    std::unique_ptr<MetricsCollector> metrics_collector_;
    std::unique_ptr<SlidingWindowMonitor> window_monitor_;
    std::unique_ptr<AnomalyDetector> anomaly_detector_;
    
    // 回调函数
    PerformanceCallback performance_callback_;
    AnomalyCallback anomaly_callback_;
    
    // 监控状态
    std::atomic<bool> monitoring_;
    std::thread monitoring_thread_;
    
    // 监控循环
    void monitoring_loop();
    
public:
    PerformanceMonitor();
    ~PerformanceMonitor();
    
    // 初始化监控器
    bool initialize(std::unique_ptr<HardwareCounters> hw_counters);
    
    // 启动监控
    bool start_monitoring();
    
    // 停止监控
    void stop_monitoring();
    
    // 获取当前性能指标
    PerformanceMetrics get_current_metrics() const;
    
    // 获取平均性能指标
    PerformanceMetrics get_average_metrics(int window_ms = 1000) const;
    
    // 获取性能历史
    std::vector<PerformanceMetrics> get_metrics_history(size_t count) const;
    
    // 获取检测到的异常
    std::vector<PerformanceAnomaly> get_detected_anomalies() const;
    
    // 设置回调函数
    void set_performance_callback(PerformanceCallback callback);
    void set_anomaly_callback(AnomalyCallback callback);
    
    // 检查监控状态
    bool is_monitoring() const;
    
    // 手动触发异常检测
    std::vector<PerformanceAnomaly> detect_anomalies() const;
    
    // 获取性能趋势
    std::vector<float> get_performance_trend(const std::string& metric_name) const;
    
    // 重置监控器
    void reset();
};

// 工作负载剖析器
class WorkloadProfiler {
private:
    std::vector<PerformanceMetrics> profile_data_;
    WorkloadCharacteristics cached_characteristics_;
    bool characteristics_valid_;
    mutable std::mutex profile_mutex_;
    
    // 分析计算密集度
    float analyze_compute_intensity(const std::vector<PerformanceMetrics>& data) const;
    
    // 分析内存密集度
    float analyze_memory_intensity(const std::vector<PerformanceMetrics>& data) const;
    
    // 分析数据重用
    float analyze_data_reuse(const std::vector<PerformanceMetrics>& data) const;
    
    // 分析操作分布
    std::vector<float> analyze_operation_distribution(
        const std::vector<PerformanceMetrics>& data) const;
    
    // 分析批大小变化
    float analyze_batch_size_variability(
        const std::vector<PerformanceMetrics>& data) const;
    
public:
    WorkloadProfiler();
    
    // 添加性能数据
    void add_performance_data(const PerformanceMetrics& metrics);
    
    // 添加批量数据
    void add_performance_data_batch(const std::vector<PerformanceMetrics>& metrics);
    
    // 分析工作负载特征
    WorkloadCharacteristics analyze_workload_characteristics();
    
    // 获取缓存的特征
    WorkloadCharacteristics get_cached_characteristics() const;
    
    // 检查特征是否有效
    bool are_characteristics_valid() const;
    
    // 清除剖析数据
    void clear_profile_data();
    
    // 获取剖析数据大小
    size_t get_profile_data_size() const;
    
    // 设置最小数据量阈值
    void set_min_data_threshold(size_t threshold);
};

}  // namespace yica
}  // namespace search
}  // namespace yirage 