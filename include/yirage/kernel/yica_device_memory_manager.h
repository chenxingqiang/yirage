/* Copyright 2023-2024 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "yirage/kernel/device_tensor.h"
#include "yirage/yica/yis_instruction_set.h"
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>

namespace yirage {
namespace kernel {

/**
 * @brief YICA专用设备内存管理器
 * 管理YICA架构的三级内存层次：寄存器、SPM、DRAM
 */
class YICADeviceMemoryManager {
public:
  static YICADeviceMemoryManager *singleton;
  YICADeviceMemoryManager(int device_id, int num_devices, const YICAMemoryConfig &config);
  ~YICADeviceMemoryManager();

  static YICADeviceMemoryManager *get_instance();
  static void set_device_id(int device_id);

  // 内存层次枚举
  enum class MemoryLevel {
    REGISTER_FILE = 0,    // 寄存器文件
    SPM = 1,             // 暂存器内存
    DRAM = 2             // 主内存
  };

  // 内存分配策略
  enum class AllocationStrategy {
    FIRST_FIT,           // 首次适应
    BEST_FIT,            // 最佳适应
    WORST_FIT,           // 最坏适应
    BUDDY_SYSTEM,        // 伙伴系统
    SLAB_ALLOCATOR,      // Slab分配器
    POOL_ALLOCATOR,      // 池分配器
    YICA_OPTIMIZED       // YICA优化分配
  };

  // 内存配置
  struct YICAMemoryConfig {
    // 寄存器文件配置
    size_t register_file_size = 32 * 1024;      // 32KB
    int num_register_banks = 16;
    
    // SPM配置
    size_t spm_size_per_die = 128 * 1024 * 1024; // 128MB per die
    int num_spm_banks = 8;
    size_t spm_cache_line_size = 64;             // 64B cache line
    
    // DRAM配置
    size_t dram_total_size = 8ULL * 1024 * 1024 * 1024; // 8GB
    size_t dram_bandwidth_gbps = 512;           // 512 GB/s
    
    // 分配策略
    AllocationStrategy allocation_strategy = AllocationStrategy::YICA_OPTIMIZED;
    bool enable_memory_coalescing = true;
    bool enable_prefetching = true;
    float fragmentation_threshold = 0.2f;
    
    // 缓存配置
    bool enable_spm_caching = true;
    size_t spm_cache_associativity = 8;
    std::string spm_replacement_policy = "LRU";
  };

  // 内存分配接口
  void* allocate_memory(size_t size, MemoryLevel level, size_t alignment = 64);
  bool deallocate_memory(void* ptr, MemoryLevel level);
  
  // YICA专用分配接口
  struct YICAAllocationResult {
    void* ptr;
    MemoryLevel allocated_level;
    size_t actual_size;
    size_t alignment_offset;
    bool allocation_successful;
    float allocation_efficiency;
  };
  
  YICAAllocationResult allocate_yica_memory(
    size_t size, 
    MemoryLevel preferred_level,
    const std::vector<MemoryLevel> &fallback_levels = {});

  // 层次化内存管理
  bool promote_to_spm(void* dram_ptr, size_t size);
  bool demote_to_dram(void* spm_ptr, size_t size);
  bool prefetch_to_spm(void* dram_ptr, size_t size);
  
  // SPM缓存管理
  struct SPMCacheEntry {
    void* dram_address;
    void* spm_address;
    size_t size;
    int access_count;
    std::chrono::time_point<std::chrono::steady_clock> last_access;
    bool is_dirty;
    int priority;
  };
  
  bool cache_in_spm(void* dram_ptr, size_t size, int priority = 0);
  bool evict_from_spm(void* spm_ptr);
  std::vector<SPMCacheEntry> get_spm_cache_status();
  
  // 内存布局优化
  enum class DataLayout {
    ROW_MAJOR,           // 行优先
    COLUMN_MAJOR,        // 列优先
    TILED_ROW,           // 分块行优先
    TILED_COLUMN,        // 分块列优先
    Z_ORDER,             // Z序
    HILBERT_CURVE,       // Hilbert曲线
    YICA_OPTIMIZED       // YICA优化布局
  };
  
  bool optimize_data_layout(void* ptr, size_t size, DataLayout target_layout);
  DataLayout recommend_layout(const std::vector<int> &access_pattern);
  
  // 内存带宽优化
  struct BandwidthOptimization {
    bool enable_memory_interleaving;
    int interleaving_factor;
    bool enable_burst_access;
    size_t burst_size;
    bool enable_bank_parallelism;
    int active_banks;
  };
  
  bool apply_bandwidth_optimization(const BandwidthOptimization &config);
  float measure_memory_bandwidth(MemoryLevel level);
  
  // 内存压缩
  enum class CompressionType {
    NONE,                // 无压缩
    LZ4,                 // LZ4压缩
    SNAPPY,              // Snappy压缩
    ZSTD,                // Zstandard压缩
    DELTA_ENCODING,      // 增量编码
    BIT_PACKING,         // 位打包
    YICA_CUSTOM          // YICA自定义压缩
  };
  
  struct CompressionResult {
    void* compressed_ptr;
    size_t compressed_size;
    size_t original_size;
    float compression_ratio;
    CompressionType compression_type;
    bool compression_successful;
  };
  
  CompressionResult compress_memory_block(void* ptr, size_t size, CompressionType type);
  bool decompress_memory_block(const CompressionResult &compressed);
  
  // 内存统计和监控
  struct MemoryStatistics {
    // 分配统计
    size_t total_allocated_bytes[3];     // 按内存层次统计
    size_t peak_allocated_bytes[3];
    size_t num_allocations[3];
    size_t num_deallocations[3];
    
    // 性能统计
    double average_allocation_time[3];
    double average_access_latency[3];
    float memory_utilization[3];
    float fragmentation_ratio[3];
    
    // SPM缓存统计
    size_t spm_cache_hits;
    size_t spm_cache_misses;
    float spm_cache_hit_rate;
    size_t spm_evictions;
    
    // 带宽统计
    float measured_bandwidth[3];
    float bandwidth_utilization[3];
    size_t total_memory_transactions;
  };
  
  MemoryStatistics get_memory_statistics() const;
  void reset_statistics();
  
  // 内存调试和诊断
  struct MemoryDiagnostics {
    std::vector<std::string> memory_leaks;
    std::vector<std::string> fragmentation_issues;
    std::vector<std::string> performance_warnings;
    float overall_health_score;
  };
  
  MemoryDiagnostics diagnose_memory_health();
  bool validate_memory_integrity();
  void dump_memory_layout(const std::string &filename);

  // 垃圾回收和内存整理
  struct GCConfig {
    bool enable_automatic_gc;
    float gc_trigger_threshold;        // 触发GC的内存使用阈值
    int gc_frequency_ms;               // GC频率（毫秒）
    bool enable_compaction;            // 启用内存压缩整理
    bool enable_background_gc;         // 启用后台GC
  };
  
  bool configure_garbage_collection(const GCConfig &config);
  void trigger_garbage_collection();
  void compact_memory(MemoryLevel level);

public:
  int num_devices_, device_id_;
  YICAMemoryConfig config_;
  
  // 内存池
  struct MemoryPool {
    void* base_ptr;
    size_t total_size;
    size_t used_size;
    std::vector<std::pair<size_t, size_t>> free_blocks; // (offset, size)
    std::mutex pool_mutex;
  };
  
  MemoryPool register_pool_;
  MemoryPool spm_pool_;
  MemoryPool dram_pool_;
  
  // SPM缓存
  std::unordered_map<void*, SPMCacheEntry> spm_cache_;
  std::mutex spm_cache_mutex_;
  
  // 统计信息
  mutable MemoryStatistics statistics_;
  mutable std::mutex statistics_mutex_;
  
  // 查找表和指针映射
  yirage::type::FPType *exp_lookup_table_;
  yirage::type::FPType *div_p_lookup_table_;
  yirage::type::FPType *div_q_lookup_table_;
  yirage::type::FPType *sqrt_p_lookup_table_;
  yirage::type::FPType *sqrt_q_lookup_table_;

private:
  // 内存分配算法
  void* allocate_first_fit(MemoryPool &pool, size_t size, size_t alignment);
  void* allocate_best_fit(MemoryPool &pool, size_t size, size_t alignment);
  void* allocate_yica_optimized(MemoryPool &pool, size_t size, size_t alignment);
  
  // 内存碎片整理
  void defragment_memory_pool(MemoryPool &pool);
  float calculate_fragmentation_ratio(const MemoryPool &pool);
  
  // SPM缓存算法
  bool lru_evict_spm_entry();
  bool lfu_evict_spm_entry();
  void update_spm_access_statistics(void* spm_ptr);
  
  // 性能监控
  void update_allocation_statistics(MemoryLevel level, size_t size, double time);
  void update_access_statistics(MemoryLevel level, double latency);
  
  // 后台任务
  void background_gc_thread();
  void background_statistics_thread();
  
  std::thread gc_thread_;
  std::thread stats_thread_;
  std::atomic<bool> shutdown_requested_;
};

/**
 * @brief YICA内存管理辅助函数
 */
namespace yica_memory_utils {
  
  /**
   * @brief 计算最优的内存对齐
   */
  size_t calculate_optimal_alignment(size_t size, YICADeviceMemoryManager::MemoryLevel level);
  
  /**
   * @brief 估算内存访问延迟
   */
  double estimate_access_latency(
    YICADeviceMemoryManager::MemoryLevel level,
    size_t access_size,
    bool sequential_access = true);
  
  /**
   * @brief 生成最优的内存访问模式
   */
  std::vector<yica::YISInstruction> generate_optimal_memory_access(
    void* src_ptr, void* dst_ptr,
    size_t size,
    YICADeviceMemoryManager::MemoryLevel src_level,
    YICADeviceMemoryManager::MemoryLevel dst_level);
  
  /**
   * @brief 分析内存访问模式
   */
  struct MemoryAccessPattern {
    bool is_sequential;
    size_t stride;
    float locality_factor;
    std::vector<size_t> hot_regions;
  };
  
  MemoryAccessPattern analyze_access_pattern(
    const std::vector<void*> &access_sequence);

} // namespace yica_memory_utils

// 全局接口函数
void yica_set_device_id(int device_id);

} // namespace kernel
} // namespace yirage 