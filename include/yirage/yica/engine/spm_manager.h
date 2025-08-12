/**
 * @file spm_manager.h
 * @brief SPM 内存管理器 - YICA 片上可编程存储器管理
 * 
 * 实现 YICA SPM (Scratchpad Memory) 的完整管理，包括：
 * - 分层内存分配和回收
 * - 多Bank并行访问
 * - 缓存策略和预取
 * - 数据布局优化
 */

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <queue>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <functional>

#include "yirage/yica/config.h"

namespace yirage {
namespace yica {

/**
 * @brief SPM 内存块状态
 */
enum class SPMBlockState {
    FREE = 0,           ///< 空闲状态
    ALLOCATED = 1,      ///< 已分配
    CACHED = 2,         ///< 缓存状态
    PREFETCHED = 3,     ///< 预取状态
    DIRTY = 4,          ///< 脏数据状态
    LOCKED = 5          ///< 锁定状态
};

/**
 * @brief SPM 数据布局类型
 */
enum class SPMLayoutType {
    ROW_MAJOR = 0,      ///< 行优先布局
    COL_MAJOR = 1,      ///< 列优先布局
    TILED_ROW = 2,      ///< 分块行优先
    TILED_COL = 3,      ///< 分块列优先
    COMPRESSED = 4,     ///< 压缩布局
    ADAPTIVE = 5        ///< 自适应布局
};

/**
 * @brief SPM 访问模式
 */
enum class SPMAccessPattern {
    SEQUENTIAL = 0,     ///< 顺序访问
    RANDOM = 1,         ///< 随机访问
    STRIDED = 2,        ///< 步长访问
    BLOCKED = 3,        ///< 分块访问
    BROADCAST = 4,      ///< 广播访问
    GATHER_SCATTER = 5  ///< 收集/分散访问
};

/**
 * @brief SPM 内存块描述符
 */
struct SPMBlock {
    uint64_t address;               ///< 物理地址
    size_t size;                    ///< 块大小
    SPMBlockState state;            ///< 块状态
    SPMLayoutType layout;           ///< 数据布局
    uint32_t bank_id;               ///< 所属Bank ID
    uint32_t ref_count;             ///< 引用计数
    std::chrono::steady_clock::time_point last_access; ///< 最后访问时间
    void* data_ptr;                 ///< 数据指针
    
    // 元数据
    std::unordered_map<std::string, std::string> metadata;
    
    SPMBlock() : address(0), size(0), state(SPMBlockState::FREE), 
                layout(SPMLayoutType::ROW_MAJOR), bank_id(0), 
                ref_count(0), data_ptr(nullptr) {}
};

/**
 * @brief SPM Bank 配置
 */
struct SPMBankConfig {
    uint32_t bank_id;               ///< Bank ID
    size_t bank_size;               ///< Bank 大小
    size_t bank_width;              ///< Bank 位宽
    uint32_t access_latency_cycles; ///< 访问延迟(周期)
    double bandwidth_gbps;          ///< 带宽 (GB/s)
    bool supports_dual_port;        ///< 支持双端口访问
    bool supports_ecc;              ///< 支持ECC
};

/**
 * @brief SPM 性能统计
 */
struct SPMMetrics {
    // 容量统计
    size_t total_capacity;          ///< 总容量
    size_t used_capacity;           ///< 已使用容量
    size_t free_capacity;           ///< 空闲容量
    double utilization_rate;        ///< 利用率
    
    // 访问统计
    uint64_t total_reads;           ///< 总读取次数
    uint64_t total_writes;          ///< 总写入次数
    uint64_t cache_hits;            ///< 缓存命中次数
    uint64_t cache_misses;          ///< 缓存未命中次数
    double hit_rate;                ///< 命中率
    
    // 性能统计
    double average_latency_ns;      ///< 平均延迟 (纳秒)
    double peak_bandwidth_gbps;     ///< 峰值带宽 (GB/s)
    double sustained_bandwidth_gbps; ///< 持续带宽 (GB/s)
    
    // 碎片统计
    uint32_t total_fragments;       ///< 总碎片数
    size_t largest_free_block;      ///< 最大空闲块
    double fragmentation_ratio;     ///< 碎片率
};

/**
 * @brief SPM 内存管理器
 * 
 * 负责管理 YICA SPM 的分配、回收、缓存和优化，
 * 支持多Bank并行访问和智能数据布局。
 */
class SPMMemoryManager {
public:
    /**
     * @brief 构造函数
     * @param config YICA 配置
     */
    explicit SPMMemoryManager(const YICAConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~SPMMemoryManager();
    
    // 禁用拷贝和赋值
    SPMMemoryManager(const SPMMemoryManager&) = delete;
    SPMMemoryManager& operator=(const SPMMemoryManager&) = delete;
    
    /**
     * @brief 初始化 SPM 管理器
     * @return 成功返回 true
     */
    bool initialize();
    
    /**
     * @brief 关闭 SPM 管理器
     */
    void shutdown();
    
    /**
     * @brief 分配 SPM 内存
     * @param size 请求大小
     * @param alignment 对齐要求
     * @param layout 数据布局类型
     * @param preferred_bank 首选Bank ID (-1表示自动选择)
     * @return 分配的内存块，nullptr表示失败
     */
    std::shared_ptr<SPMBlock> allocate(
        size_t size, 
        size_t alignment = 8,
        SPMLayoutType layout = SPMLayoutType::ROW_MAJOR,
        int preferred_bank = -1
    );
    
    /**
     * @brief 释放 SPM 内存
     * @param block 要释放的内存块
     * @return 成功返回 true
     */
    bool deallocate(std::shared_ptr<SPMBlock> block);
    
    /**
     * @brief 读取数据
     * @param block 内存块
     * @param offset 偏移量
     * @param size 读取大小
     * @param buffer 目标缓冲区
     * @return 实际读取的字节数
     */
    size_t read(const std::shared_ptr<SPMBlock>& block, 
               size_t offset, size_t size, void* buffer);
    
    /**
     * @brief 写入数据
     * @param block 内存块
     * @param offset 偏移量
     * @param size 写入大小
     * @param buffer 源缓冲区
     * @return 实际写入的字节数
     */
    size_t write(const std::shared_ptr<SPMBlock>& block,
                size_t offset, size_t size, const void* buffer);
    
    /**
     * @brief 数据预取
     * @param dram_address DRAM 地址
     * @param size 预取大小
     * @param layout 数据布局
     * @param priority 优先级 (0-7)
     * @return 预取的SPM块
     */
    std::shared_ptr<SPMBlock> prefetch(
        uint64_t dram_address, 
        size_t size,
        SPMLayoutType layout = SPMLayoutType::ROW_MAJOR,
        uint32_t priority = 4
    );
    
    /**
     * @brief 数据回写
     * @param block SPM 块
     * @param dram_address 目标 DRAM 地址
     * @param async 是否异步回写
     * @return 成功返回 true
     */
    bool writeback(const std::shared_ptr<SPMBlock>& block,
                   uint64_t dram_address, bool async = true);
    
    /**
     * @brief 内存整理
     * @param aggressive 是否激进整理
     * @return 整理后释放的字节数
     */
    size_t defragment(bool aggressive = false);
    
    /**
     * @brief 缓存刷新
     * @param force_all 是否强制刷新所有
     * @return 刷新的块数
     */
    uint32_t flush_cache(bool force_all = false);
    
    /**
     * @brief 获取性能统计
     * @return SPM 性能指标
     */
    SPMMetrics get_metrics() const;
    
    /**
     * @brief 重置统计信息
     */
    void reset_metrics();
    
    /**
     * @brief 获取可用容量
     * @return 可用字节数
     */
    size_t get_available_capacity() const;
    
    /**
     * @brief 获取最大连续空闲块大小
     * @return 最大空闲块字节数
     */
    size_t get_largest_free_block() const;
    
    /**
     * @brief 设置访问模式提示
     * @param pattern 访问模式
     */
    void set_access_pattern_hint(SPMAccessPattern pattern);
    
    /**
     * @brief 启用/禁用自动预取
     * @param enable 是否启用
     */
    void set_auto_prefetch(bool enable);
    
    /**
     * @brief 设置缓存策略
     * @param policy 缓存策略名称 ("LRU", "LFU", "FIFO", "ADAPTIVE")
     */
    void set_cache_policy(const std::string& policy);
    
    /**
     * @brief 获取Bank信息
     * @param bank_id Bank ID
     * @return Bank配置信息
     */
    SPMBankConfig get_bank_config(uint32_t bank_id) const;
    
    /**
     * @brief 获取内存布局建议
     * @param access_pattern 访问模式
     * @param data_size 数据大小
     * @return 建议的布局类型
     */
    SPMLayoutType suggest_layout(SPMAccessPattern access_pattern, 
                                size_t data_size) const;

private:
    // 配置和状态
    YICAConfig config_;
    std::atomic<bool> initialized_;
    std::atomic<bool> shutdown_requested_;
    
    // Bank 配置
    std::vector<SPMBankConfig> bank_configs_;
    std::vector<std::mutex> bank_mutexes_;
    
    // 内存管理
    std::vector<std::vector<SPMBlock>> bank_blocks_;  // 每个Bank的块列表
    std::unordered_map<uint64_t, std::shared_ptr<SPMBlock>> address_map_;
    std::mutex allocation_mutex_;
    
    // 空闲块管理
    std::vector<std::priority_queue<std::shared_ptr<SPMBlock>,
                                   std::vector<std::shared_ptr<SPMBlock>>,
                                   std::function<bool(const std::shared_ptr<SPMBlock>&,
                                                     const std::shared_ptr<SPMBlock>&)>>> 
                free_blocks_;  // 每个Bank的空闲块优先队列
    
    // 缓存管理
    std::unordered_map<uint64_t, std::shared_ptr<SPMBlock>> cache_map_;
    std::queue<std::shared_ptr<SPMBlock>> lru_queue_;
    std::mutex cache_mutex_;
    std::string cache_policy_;
    
    // 预取管理
    std::queue<std::pair<uint64_t, size_t>> prefetch_queue_;
    std::mutex prefetch_mutex_;
    std::condition_variable prefetch_cv_;
    std::vector<std::thread> prefetch_workers_;
    std::atomic<bool> auto_prefetch_enabled_;
    
    // 性能统计
    mutable std::mutex metrics_mutex_;
    SPMMetrics metrics_;
    std::chrono::steady_clock::time_point start_time_;
    
    // 访问模式
    std::atomic<SPMAccessPattern> current_access_pattern_;
    
    // 私有方法：内存分配
    std::shared_ptr<SPMBlock> allocate_from_bank(uint32_t bank_id, size_t size, 
                                                size_t alignment, SPMLayoutType layout);
    bool merge_free_blocks(uint32_t bank_id);
    uint32_t select_best_bank(size_t size, SPMLayoutType layout) const;
    
    // 私有方法：缓存管理
    void update_lru_cache(const std::shared_ptr<SPMBlock>& block);
    void evict_cache_blocks(size_t required_size);
    bool is_cache_hit(uint64_t address) const;
    
    // 私有方法：预取
    void prefetch_worker_function();
    void analyze_access_pattern();
    void schedule_intelligent_prefetch();
    
    // 私有方法：性能监控
    void update_access_metrics(bool is_read, size_t size, double latency_ns);
    void update_cache_metrics(bool hit);
    double calculate_access_latency(uint32_t bank_id, size_t size) const;
    
    // 私有方法：工具函数
    size_t align_size(size_t size, size_t alignment) const;
    uint64_t align_address(uint64_t address, size_t alignment) const;
    void log_debug(const std::string& message) const;
    
    // 常量
    static constexpr size_t DEFAULT_ALIGNMENT = 8;
    static constexpr size_t MIN_BLOCK_SIZE = 32;
    static constexpr size_t MAX_PREFETCH_QUEUE_SIZE = 1000;
    static constexpr double DEFRAG_THRESHOLD = 0.3;  // 30%碎片率触发整理
};

/**
 * @brief SPM 内存管理器工厂类
 */
class SPMManagerFactory {
public:
    /**
     * @brief 创建 SPM 管理器
     * @param config YICA 配置
     * @return SPM 管理器实例
     */
    static std::unique_ptr<SPMMemoryManager> create(const YICAConfig& config);
    
    /**
     * @brief 创建优化的 SPM 管理器
     * @param config YICA 配置
     * @param optimization_profile 优化配置文件
     * @return 优化的 SPM 管理器实例
     */
    static std::unique_ptr<SPMMemoryManager> create_optimized(
        const YICAConfig& config, 
        const std::string& optimization_profile = "default"
    );
};

} // namespace yica
} // namespace yirage 