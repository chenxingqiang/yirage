/**
 * @file dram_interface.h
 * @brief DRAM 接口 - YICA 主内存访问接口
 * 
 * 实现 YICA DRAM 的高效访问接口，包括：
 * - UMA/NUMA 内存访问优化
 * - 批量数据传输
 * - 内存带宽管理
 * - 异步I/O操作
 */

#pragma once

#include <memory>
#include <vector>
#include <queue>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <future>

#include "yirage/yica/config.h"

namespace yirage {
namespace yica {

/**
 * @brief DRAM 访问类型
 */
enum class DRAMAccessType {
    READ = 0,           ///< 读取操作
    WRITE = 1,          ///< 写入操作
    READ_WRITE = 2,     ///< 读写操作
    PREFETCH = 3,       ///< 预取操作
    WRITEBACK = 4       ///< 回写操作
};

/**
 * @brief DRAM 内存类型
 */
enum class DRAMMemoryType {
    UMA = 0,            ///< 统一内存访问
    NUMA = 1,           ///< 非统一内存访问
    HBM = 2,            ///< 高带宽内存
    DDR4 = 3,           ///< DDR4内存
    DDR5 = 4            ///< DDR5内存
};

/**
 * @brief DRAM 访问优先级
 */
enum class DRAMPriority {
    LOW = 0,            ///< 低优先级
    NORMAL = 1,         ///< 正常优先级
    HIGH = 2,           ///< 高优先级
    CRITICAL = 3        ///< 关键优先级
};

/**
 * @brief DRAM 访问请求
 */
struct DRAMRequest {
    uint64_t request_id;            ///< 请求ID
    DRAMAccessType access_type;     ///< 访问类型
    uint64_t address;               ///< 内存地址
    size_t size;                    ///< 数据大小
    void* buffer;                   ///< 数据缓冲区
    DRAMPriority priority;          ///< 优先级
    uint32_t numa_node;             ///< NUMA节点ID
    std::chrono::steady_clock::time_point submit_time; ///< 提交时间
    
    // 回调函数
    std::function<void(bool, double)> completion_callback;
    
    DRAMRequest() : request_id(0), access_type(DRAMAccessType::READ),
                   address(0), size(0), buffer(nullptr),
                   priority(DRAMPriority::NORMAL), numa_node(0) {}
};

/**
 * @brief DRAM 通道配置
 */
struct DRAMChannelConfig {
    uint32_t channel_id;            ///< 通道ID
    DRAMMemoryType memory_type;     ///< 内存类型
    size_t channel_capacity;        ///< 通道容量
    uint32_t bus_width;             ///< 总线位宽
    uint32_t frequency_mhz;         ///< 频率 (MHz)
    double peak_bandwidth_gbps;     ///< 峰值带宽 (GB/s)
    uint32_t access_latency_ns;     ///< 访问延迟 (ns)
    uint32_t numa_node;             ///< 所属NUMA节点
    bool supports_ecc;              ///< 支持ECC
    bool supports_compression;      ///< 支持压缩
};

/**
 * @brief DRAM 性能统计
 */
struct DRAMMetrics {
    // 容量统计
    size_t total_capacity;          ///< 总容量
    size_t used_capacity;           ///< 已使用容量
    size_t free_capacity;           ///< 空闲容量
    double utilization_rate;        ///< 利用率
    
    // 访问统计
    uint64_t total_requests;        ///< 总请求数
    uint64_t read_requests;         ///< 读请求数
    uint64_t write_requests;        ///< 写请求数
    uint64_t completed_requests;    ///< 完成请求数
    uint64_t failed_requests;       ///< 失败请求数
    
    // 性能统计
    double average_latency_ns;      ///< 平均延迟 (ns)
    double peak_bandwidth_gbps;     ///< 峰值带宽 (GB/s)
    double sustained_bandwidth_gbps; ///< 持续带宽 (GB/s)
    double read_bandwidth_gbps;     ///< 读带宽 (GB/s)
    double write_bandwidth_gbps;    ///< 写带宽 (GB/s)
    
    // 队列统计
    uint32_t pending_requests;      ///< 待处理请求数
    double average_queue_depth;     ///< 平均队列深度
    uint32_t max_queue_depth;       ///< 最大队列深度
    
    // NUMA统计
    std::vector<double> numa_node_utilization; ///< 各NUMA节点利用率
    std::vector<double> numa_access_latency;   ///< 各NUMA节点访问延迟
};

/**
 * @brief DRAM 接口类
 * 
 * 负责管理 YICA DRAM 的访问，包括：
 * - 高效的内存访问调度
 * - NUMA感知的内存分配
 * - 批量传输优化
 * - 异步I/O操作
 */
class DRAMInterface {
public:
    /**
     * @brief 构造函数
     * @param config YICA 配置
     */
    explicit DRAMInterface(const YICAConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~DRAMInterface();
    
    // 禁用拷贝和赋值
    DRAMInterface(const DRAMInterface&) = delete;
    DRAMInterface& operator=(const DRAMInterface&) = delete;
    
    /**
     * @brief 初始化 DRAM 接口
     * @return 成功返回 true
     */
    bool initialize();
    
    /**
     * @brief 关闭 DRAM 接口
     */
    void shutdown();
    
    /**
     * @brief 同步读取数据
     * @param address 内存地址
     * @param size 读取大小
     * @param buffer 目标缓冲区
     * @param numa_node NUMA节点ID (-1表示自动选择)
     * @return 实际读取的字节数
     */
    size_t read_sync(uint64_t address, size_t size, void* buffer, 
                     int numa_node = -1);
    
    /**
     * @brief 同步写入数据
     * @param address 内存地址
     * @param size 写入大小
     * @param buffer 源缓冲区
     * @param numa_node NUMA节点ID (-1表示自动选择)
     * @return 实际写入的字节数
     */
    size_t write_sync(uint64_t address, size_t size, const void* buffer,
                      int numa_node = -1);
    
    /**
     * @brief 异步读取数据
     * @param address 内存地址
     * @param size 读取大小
     * @param buffer 目标缓冲区
     * @param priority 优先级
     * @param numa_node NUMA节点ID (-1表示自动选择)
     * @return 异步操作的future对象
     */
    std::future<size_t> read_async(uint64_t address, size_t size, void* buffer,
                                   DRAMPriority priority = DRAMPriority::NORMAL,
                                   int numa_node = -1);
    
    /**
     * @brief 异步写入数据
     * @param address 内存地址
     * @param size 写入大小
     * @param buffer 源缓冲区
     * @param priority 优先级
     * @param numa_node NUMA节点ID (-1表示自动选择)
     * @return 异步操作的future对象
     */
    std::future<size_t> write_async(uint64_t address, size_t size, const void* buffer,
                                    DRAMPriority priority = DRAMPriority::NORMAL,
                                    int numa_node = -1);
    
    /**
     * @brief 批量数据传输
     * @param requests 传输请求列表
     * @return 完成的传输数量
     */
    uint32_t batch_transfer(const std::vector<DRAMRequest>& requests);
    
    /**
     * @brief 内存拷贝 (DRAM到DRAM)
     * @param src_address 源地址
     * @param dst_address 目标地址
     * @param size 拷贝大小
     * @param async 是否异步执行
     * @return 成功返回 true
     */
    bool memory_copy(uint64_t src_address, uint64_t dst_address, size_t size,
                     bool async = false);
    
    /**
     * @brief 内存预取
     * @param address 预取地址
     * @param size 预取大小
     * @param priority 优先级
     * @return 预取请求ID
     */
    uint64_t prefetch(uint64_t address, size_t size, 
                      DRAMPriority priority = DRAMPriority::LOW);
    
    /**
     * @brief 等待所有异步操作完成
     * @param timeout_ms 超时时间 (毫秒)
     * @return 是否在超时前完成
     */
    bool wait_for_completion(uint32_t timeout_ms = 5000);
    
    /**
     * @brief 取消异步操作
     * @param request_id 请求ID
     * @return 成功返回 true
     */
    bool cancel_request(uint64_t request_id);
    
    /**
     * @brief 获取性能统计
     * @return DRAM 性能指标
     */
    DRAMMetrics get_metrics() const;
    
    /**
     * @brief 重置统计信息
     */
    void reset_metrics();
    
    /**
     * @brief 获取可用容量
     * @param numa_node NUMA节点ID (-1表示所有节点)
     * @return 可用字节数
     */
    size_t get_available_capacity(int numa_node = -1) const;
    
    /**
     * @brief 获取内存带宽利用率
     * @return 带宽利用率 (0.0-1.0)
     */
    double get_bandwidth_utilization() const;
    
    /**
     * @brief 获取平均访问延迟
     * @param numa_node NUMA节点ID (-1表示所有节点)
     * @return 平均延迟 (纳秒)
     */
    double get_average_latency(int numa_node = -1) const;
    
    /**
     * @brief 设置访问优化策略
     * @param strategy 策略名称 ("bandwidth", "latency", "balanced")
     */
    void set_optimization_strategy(const std::string& strategy);
    
    /**
     * @brief 启用/禁用自动预取
     * @param enable 是否启用
     */
    void set_auto_prefetch(bool enable);
    
    /**
     * @brief 设置NUMA亲和性
     * @param enable 是否启用NUMA亲和性
     */
    void set_numa_affinity(bool enable);
    
    /**
     * @brief 获取通道配置
     * @param channel_id 通道ID
     * @return 通道配置信息
     */
    DRAMChannelConfig get_channel_config(uint32_t channel_id) const;
    
    /**
     * @brief 获取最佳NUMA节点
     * @param address 内存地址
     * @param size 数据大小
     * @return 最佳NUMA节点ID
     */
    uint32_t get_optimal_numa_node(uint64_t address, size_t size) const;

private:
    // 配置和状态
    YICAConfig config_;
    std::atomic<bool> initialized_;
    std::atomic<bool> shutdown_requested_;
    
    // 通道配置
    std::vector<DRAMChannelConfig> channel_configs_;
    std::vector<std::mutex> channel_mutexes_;
    
    // 请求队列管理
    std::priority_queue<DRAMRequest, std::vector<DRAMRequest>,
                       std::function<bool(const DRAMRequest&, const DRAMRequest&)>> 
                       request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // 工作线程
    std::vector<std::thread> worker_threads_;
    std::vector<std::thread> channel_threads_;
    
    // 异步操作管理
    std::unordered_map<uint64_t, std::promise<size_t>> async_promises_;
    std::mutex promise_mutex_;
    std::atomic<uint64_t> next_request_id_;
    
    // 性能统计
    mutable std::mutex metrics_mutex_;
    DRAMMetrics metrics_;
    std::chrono::steady_clock::time_point start_time_;
    
    // 优化配置
    std::string optimization_strategy_;
    std::atomic<bool> auto_prefetch_enabled_;
    std::atomic<bool> numa_affinity_enabled_;
    
    // 预取管理
    std::queue<std::pair<uint64_t, size_t>> prefetch_queue_;
    std::mutex prefetch_mutex_;
    std::condition_variable prefetch_cv_;
    std::thread prefetch_thread_;
    
    // NUMA拓扑信息
    std::vector<std::vector<double>> numa_distance_matrix_;
    std::vector<uint32_t> numa_node_channels_;
    
    // 私有方法：请求处理
    void worker_thread_function();
    void channel_thread_function(uint32_t channel_id);
    void prefetch_thread_function();
    bool process_request(const DRAMRequest& request);
    
    // 私有方法：调度优化
    uint32_t select_optimal_channel(uint64_t address, size_t size, 
                                   DRAMAccessType access_type) const;
    bool should_batch_requests(const std::vector<DRAMRequest>& requests) const;
    std::vector<std::vector<DRAMRequest>> group_requests_by_locality(
        const std::vector<DRAMRequest>& requests) const;
    
    // 私有方法：NUMA优化
    void initialize_numa_topology();
    double calculate_numa_distance(uint32_t node1, uint32_t node2) const;
    uint32_t get_numa_node_for_address(uint64_t address) const;
    
    // 私有方法：性能监控
    void update_access_metrics(const DRAMRequest& request, double latency_ns, bool success);
    void update_bandwidth_metrics(size_t bytes_transferred, double time_ns);
    double calculate_access_latency(uint32_t channel_id, size_t size, 
                                   DRAMAccessType access_type) const;
    
    // 私有方法：预取优化
    void analyze_access_patterns();
    void schedule_intelligent_prefetch();
    bool should_prefetch(uint64_t address, size_t size) const;
    
    // 私有方法：工具函数
    bool is_address_aligned(uint64_t address, size_t alignment) const;
    size_t get_optimal_transfer_size(size_t requested_size) const;
    void log_debug(const std::string& message) const;
    
    // 常量
    static constexpr size_t MAX_QUEUE_SIZE = 10000;
    static constexpr size_t OPTIMAL_BATCH_SIZE = 64;
    static constexpr double BANDWIDTH_UTILIZATION_THRESHOLD = 0.8;
    static constexpr uint32_t DEFAULT_WORKER_THREADS = 8;
};

/**
 * @brief DRAM 接口工厂类
 */
class DRAMInterfaceFactory {
public:
    /**
     * @brief 创建 DRAM 接口
     * @param config YICA 配置
     * @return DRAM 接口实例
     */
    static std::unique_ptr<DRAMInterface> create(const YICAConfig& config);
    
    /**
     * @brief 创建优化的 DRAM 接口
     * @param config YICA 配置
     * @param optimization_profile 优化配置文件
     * @return 优化的 DRAM 接口实例
     */
    static std::unique_ptr<DRAMInterface> create_optimized(
        const YICAConfig& config,
        const std::string& optimization_profile = "balanced"
    );
};

} // namespace yica
} // namespace yirage 