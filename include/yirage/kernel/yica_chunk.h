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

#include "yirage/kernel/operator.h"
#include "yirage/yica/yis_instruction_set.h"
#include <vector>
#include <memory>

namespace yirage {
namespace kernel {

/**
 * @brief YICA专用Chunk操作算子
 * 优化张量分块和重组操作，充分利用YICA的内存层次和并行能力
 */
class YICAChunkOp : public yirage::kernel::KNOperator {
public:
  YICAChunkOp(Graph *_graph, 
              DTensor const &input, 
              int chunk_size, 
              int dim,
              const YICAChunkConfig &config = {});
  ~YICAChunkOp();
  
  bool profile(ProfileResult &profile) override;
  bool fingerprint(void) override;
  operator json() const override;

  // Chunk操作类型
  enum class ChunkType {
    SPLIT,           // 分割操作
    CONCAT,          // 连接操作
    RESHAPE,         // 重塑操作
    TRANSPOSE,       // 转置操作
    PERMUTE,         // 置换操作
    TILE,            // 平铺操作
    REPEAT,          // 重复操作
    INTERLEAVE       // 交错操作
  };

  // YICA特定功能
  bool optimize_for_cim_chunking(int num_cim_arrays);
  bool enable_spm_chunk_caching(size_t cache_size);
  bool use_parallel_chunk_processing();
  bool enable_zero_copy_chunking();
  
  // 分块策略
  enum class ChunkStrategy {
    SEQUENTIAL,      // 顺序分块
    PARALLEL,        // 并行分块
    ADAPTIVE,        // 自适应分块
    MEMORY_ALIGNED,  // 内存对齐分块
    CIM_OPTIMIZED,   // CIM优化分块
    BANDWIDTH_AWARE  // 带宽感知分块
  };
  
  bool set_chunk_strategy(ChunkStrategy strategy);
  ChunkStrategy recommend_chunk_strategy(const DTensor &input);

  // YICA性能分析
  struct YICAChunkMetrics {
    float cim_chunking_efficiency;       // CIM分块效率
    float spm_cache_hit_rate;           // SPM缓存命中率
    size_t yis_copy_instruction_count;  // YIS拷贝指令数
    float memory_bandwidth_utilization; // 内存带宽利用率
    float parallel_processing_efficiency; // 并行处理效率
    size_t total_chunk_operations;      // 总分块操作数
    float zero_copy_ratio;              // 零拷贝比率
    float chunk_alignment_efficiency;   // 分块对齐效率
  };
  
  YICAChunkMetrics get_yica_metrics() const;

public:
  // YICA Chunk配置
  struct YICAChunkConfig {
    // 分块策略配置
    ChunkStrategy chunk_strategy = ChunkStrategy::CIM_OPTIMIZED;
    ChunkType chunk_type = ChunkType::SPLIT;
    bool enable_adaptive_chunking = true;
    
    // CIM配置
    int num_cim_arrays = 4;              // CIM阵列数量
    bool enable_parallel_cim_processing = true; // 并行CIM处理
    int cim_chunk_size_hint = 1024;      // CIM分块大小提示
    
    // SPM配置
    size_t spm_chunk_cache_size = 32 * 1024 * 1024; // SPM分块缓存 (32MB)
    bool enable_spm_prefetching = true;   // 启用SPM预取
    bool enable_chunk_coalescing = true;  // 启用分块合并
    
    // 内存优化
    bool enable_zero_copy = true;         // 启用零拷贝
    bool enable_memory_alignment = true;  // 启用内存对齐
    size_t alignment_boundary = 64;      // 对齐边界 (64B)
    
    // 并行配置
    int max_parallel_chunks = 8;         // 最大并行分块数
    bool enable_dynamic_load_balancing = true; // 动态负载均衡
    float load_balance_threshold = 0.8f; // 负载均衡阈值
    
    // 性能优化
    bool enable_vectorized_copy = true;   // 启用向量化拷贝
    int vector_width = 32;               // 向量宽度
    bool enable_burst_transfer = true;    // 启用突发传输
    size_t burst_size = 256;             // 突发大小
    
    // 缓存策略
    enum class CachePolicy {
      LRU, LFU, FIFO, RANDOM, YICA_ADAPTIVE
    } cache_policy = CachePolicy::YICA_ADAPTIVE;
    
    float cache_eviction_threshold = 0.9f; // 缓存驱逐阈值
  } yica_config;

private:
  // YIS指令生成
  std::vector<yica::YISInstruction> generate_chunk_instructions();
  std::vector<yica::YISInstruction> generate_split_instructions();
  std::vector<yica::YISInstruction> generate_concat_instructions();
  std::vector<yica::YISInstruction> generate_reshape_instructions();
  
  // CIM分块优化
  struct CIMChunkPlan {
    std::vector<std::pair<int, int>> chunk_boundaries; // (start, end)
    std::vector<int> cim_array_assignment;
    int optimal_chunk_size;
    float expected_efficiency;
    size_t memory_footprint;
  };
  
  CIMChunkPlan plan_cim_chunking();
  bool optimize_chunk_boundaries();
  
  // SPM缓存管理
  struct SPMChunkCache {
    struct CacheEntry {
      DTensor* tensor_chunk;
      size_t chunk_id;
      size_t access_count;
      std::chrono::time_point<std::chrono::steady_clock> last_access;
      bool is_dirty;
    };
    
    std::vector<CacheEntry> cache_entries;
    size_t total_cache_size;
    size_t used_cache_size;
    CachePolicy policy;
  };
  
  SPMChunkCache spm_cache_;
  bool cache_chunk_in_spm(const DTensor &chunk, size_t chunk_id);
  bool evict_chunk_from_spm(size_t chunk_id);
  
  // 零拷贝优化
  struct ZeroCopyPlan {
    std::vector<bool> chunk_zero_copyable;
    std::vector<size_t> view_offsets;
    float zero_copy_ratio;
    size_t memory_savings;
  };
  
  ZeroCopyPlan analyze_zero_copy_opportunities();
  bool implement_zero_copy_chunking();
  
  // 并行处理
  struct ParallelChunkPlan {
    std::vector<std::vector<int>> parallel_groups;
    std::vector<float> group_workloads;
    float load_balance_factor;
    int optimal_parallelism;
  };
  
  ParallelChunkPlan plan_parallel_processing();
  bool implement_parallel_chunking();
  
  // 内存对齐优化
  bool align_chunk_boundaries();
  size_t calculate_aligned_size(size_t original_size);
  
  // 向量化拷贝
  std::vector<yica::YISInstruction> generate_vectorized_copy_instructions();
  bool optimize_copy_pattern();
  
  // 自适应分块
  struct AdaptiveChunkState {
    std::vector<float> chunk_performance_history;
    ChunkStrategy current_best_strategy;
    std::map<ChunkStrategy, float> strategy_performance;
    bool adaptation_needed;
  };
  
  AdaptiveChunkState adaptive_state_;
  bool perform_adaptive_chunking();
  
  // 性能监控
  void update_chunk_performance_metrics();
  void analyze_chunk_access_patterns();

  // 内部状态
  int chunk_size_;
  int chunk_dim_;
  ChunkType chunk_type_;
  YICAChunkMetrics performance_metrics_;
  std::vector<yica::YISInstruction> generated_instructions_;
};

/**
 * @brief YICA Chunk工厂函数
 */
YICAChunkOp* create_yica_chunk_op(
  Graph *graph,
  DTensor const &input,
  int chunk_size,
  int dim,
  const YICAChunkOp::YICAChunkConfig &config = {});

/**
 * @brief YICA多维Chunk操作
 */
class YICAMultiDimChunkOp : public YICAChunkOp {
public:
  YICAMultiDimChunkOp(Graph *_graph,
                      DTensor const &input,
                      const std::vector<int> &chunk_sizes,
                      const std::vector<int> &dims,
                      const YICAChunkConfig &config = {});
  
  // 多维分块特定功能
  bool optimize_multidim_chunking();
  bool enable_hierarchical_chunking();
  
  // 多维分块策略
  struct MultiDimChunkStrategy {
    std::vector<int> chunk_order;        // 分块顺序
    std::vector<ChunkStrategy> dim_strategies; // 每维度策略
    bool enable_dimension_interleaving;  // 启用维度交错
    float expected_efficiency;
  };
  
  MultiDimChunkStrategy plan_multidim_strategy();

private:
  std::vector<int> chunk_sizes_;
  std::vector<int> chunk_dims_;
  MultiDimChunkStrategy multidim_strategy_;
};

/**
 * @brief YICA Chunk辅助函数
 */
namespace yica_chunk_utils {
  
  /**
   * @brief 计算最优的分块大小
   */
  struct OptimalChunkSize {
    int chunk_size;
    float efficiency_score;
    size_t memory_usage;
    int parallel_degree;
  };
  
  OptimalChunkSize calculate_optimal_chunk_size(
    const DTensor &input,
    int dim,
    const YICAChunkOp::YICAChunkConfig &config);
  
  /**
   * @brief 分析分块访问模式
   */
  struct ChunkAccessPattern {
    std::vector<size_t> access_sequence;
    bool is_sequential;
    float locality_factor;
    std::vector<int> hot_chunks;
  };
  
  ChunkAccessPattern analyze_chunk_access_pattern(
    const std::vector<int> &chunk_access_history);
  
  /**
   * @brief 生成内存对齐的分块边界
   */
  std::vector<std::pair<int, int>> generate_aligned_chunk_boundaries(
    int total_size,
    int chunk_size,
    size_t alignment_boundary);
  
  /**
   * @brief 估算分块操作的性能
   */
  struct ChunkPerformanceEstimate {
    float estimated_latency_ms;
    float memory_bandwidth_requirement;
    size_t cache_requirement;
    float parallel_efficiency;
  };
  
  ChunkPerformanceEstimate estimate_chunk_performance(
    const DTensor &input,
    YICAChunkOp::ChunkType chunk_type,
    const YICAChunkOp::YICAChunkConfig &config);
  
  /**
   * @brief 优化分块的内存布局
   */
  enum class ChunkLayout {
    CONTIGUOUS,      // 连续布局
    STRIDED,         // 跨步布局
    BLOCKED,         // 分块布局
    INTERLEAVED      // 交错布局
  };
  
  ChunkLayout recommend_chunk_layout(
    const DTensor &input,
    YICAChunkOp::ChunkType chunk_type);

} // namespace yica_chunk_utils

void from_json(json const &j, YICAChunkOp &op);

} // namespace kernel
} // namespace yirage 