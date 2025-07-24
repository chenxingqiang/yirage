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
#include <unordered_map>
#include <functional>

namespace yirage {
namespace kernel {

/**
 * @brief YICA专用计算图管理器
 * 优化计算图的调度和执行，充分利用YICA的并行计算和内存特性
 */
class YICAGraphManager {
public:
  YICAGraphManager(const YICAGraphConfig &config = {});
  ~YICAGraphManager();

  // 图操作类型
  enum class GraphOperationType {
    OPERATOR_FUSION,     // 算子融合
    PIPELINE_PARALLEL,   // 流水线并行
    DATA_PARALLEL,       // 数据并行
    MODEL_PARALLEL,      // 模型并行
    HYBRID_PARALLEL,     // 混合并行
    MEMORY_OPTIMIZATION, // 内存优化
    LOAD_BALANCING       // 负载均衡
  };

  // YICA特定功能
  bool optimize_for_cim_execution(int num_cim_arrays);
  bool enable_spm_graph_caching();
  bool use_yis_instruction_scheduling();
  bool enable_operator_fusion();
  
  // 图优化策略
  enum class GraphOptimizationStrategy {
    LATENCY_OPTIMAL,     // 延迟最优
    THROUGHPUT_OPTIMAL,  // 吞吐量最优
    MEMORY_OPTIMAL,      // 内存最优
    ENERGY_OPTIMAL,      // 能耗最优
    BALANCED,            // 均衡优化
    ADAPTIVE             // 自适应优化
  };
  
  bool set_optimization_strategy(GraphOptimizationStrategy strategy);
  GraphOptimizationStrategy recommend_strategy(const Graph &graph);

  // 图分析和优化
  struct GraphAnalysis {
    size_t total_operators;
    size_t fusable_operators;
    float parallelization_potential;
    size_t memory_footprint;
    float cim_friendliness_score;
    std::vector<std::string> optimization_opportunities;
  };
  
  GraphAnalysis analyze_graph(const Graph &graph);
  bool optimize_graph(Graph &graph);

  // YICA性能分析
  struct YICAGraphMetrics {
    float operator_fusion_ratio;        // 算子融合比率
    float cim_utilization;              // CIM利用率
    float spm_cache_efficiency;         // SPM缓存效率
    size_t yis_instruction_count;       // YIS指令数
    float pipeline_efficiency;          // 流水线效率
    float memory_bandwidth_utilization; // 内存带宽利用率
    float load_balance_factor;          // 负载均衡因子
    float graph_optimization_speedup;   // 图优化加速比
  };
  
  YICAGraphMetrics get_yica_metrics() const;

public:
  // YICA Graph配置
  struct YICAGraphConfig {
    // 优化策略配置
    GraphOptimizationStrategy optimization_strategy = GraphOptimizationStrategy::BALANCED;
    bool enable_automatic_optimization = true;
    
    // CIM配置
    int num_cim_arrays = 16;             // CIM阵列数量
    bool enable_dynamic_cim_allocation = true; // 动态CIM分配
    float cim_utilization_target = 0.85f; // CIM利用率目标
    
    // SPM配置
    size_t spm_graph_cache_size = 128 * 1024 * 1024; // SPM图缓存 (128MB)
    bool enable_graph_prefetching = true;   // 启用图预取
    bool enable_intermediate_caching = true; // 启用中间结果缓存
    
    // 算子融合配置
    struct FusionConfig {
      bool enable_vertical_fusion = true;   // 垂直融合
      bool enable_horizontal_fusion = true; // 水平融合
      int max_fusion_depth = 8;            // 最大融合深度
      size_t max_fusion_memory = 64 * 1024 * 1024; // 最大融合内存
      float fusion_benefit_threshold = 1.2f; // 融合收益阈值
    } fusion_config;
    
    // 并行配置
    struct ParallelConfig {
      bool enable_pipeline_parallel = true;  // 流水线并行
      bool enable_data_parallel = true;      // 数据并行
      bool enable_model_parallel = true;     // 模型并行
      int max_pipeline_stages = 8;          // 最大流水线阶段
      int data_parallel_degree = 4;         // 数据并行度
      float parallel_overhead_threshold = 0.1f; // 并行开销阈值
    } parallel_config;
    
    // 调度配置
    struct SchedulingConfig {
      enum class SchedulingPolicy {
        FIFO, PRIORITY, SHORTEST_JOB_FIRST, ROUND_ROBIN, YICA_ADAPTIVE
      } scheduling_policy = SchedulingPolicy::YICA_ADAPTIVE;
      
      bool enable_dynamic_scheduling = true;
      float load_balance_threshold = 0.8f;
      int scheduling_quantum_ms = 10;
    } scheduling_config;
    
    // 内存优化配置
    struct MemoryConfig {
      bool enable_memory_reuse = true;      // 启用内存复用
      bool enable_gradient_checkpointing = true; // 梯度检查点
      bool enable_activation_compression = true; // 激活压缩
      float memory_pressure_threshold = 0.9f;    // 内存压力阈值
      size_t memory_pool_size = 1024 * 1024 * 1024; // 内存池大小 (1GB)
    } memory_config;
    
    // 性能监控配置
    bool enable_performance_profiling = true;
    bool enable_bottleneck_detection = true;
    int profiling_interval_ms = 100;
  } yica_config;

private:
  // 图分析组件
  struct GraphNode {
    KNOperator* operator_ptr;
    std::vector<int> input_nodes;
    std::vector<int> output_nodes;
    size_t memory_requirement;
    float computation_cost;
    bool is_fusable;
    bool is_cim_friendly;
  };
  
  std::vector<GraphNode> graph_nodes_;
  std::unordered_map<KNOperator*, int> operator_to_node_;
  
  // 算子融合
  struct FusionGroup {
    std::vector<int> node_indices;
    float fusion_benefit;
    size_t memory_footprint;
    std::vector<yica::YISInstruction> fused_instructions;
  };
  
  std::vector<FusionGroup> identify_fusion_opportunities();
  bool implement_operator_fusion(const FusionGroup &group);
  
  // 并行化分析
  struct ParallelizationPlan {
    std::vector<std::vector<int>> pipeline_stages;
    std::vector<std::vector<int>> data_parallel_groups;
    std::map<int, int> model_parallel_mapping;
    float expected_speedup;
  };
  
  ParallelizationPlan analyze_parallelization_opportunities();
  bool implement_parallelization(const ParallelizationPlan &plan);
  
  // CIM映射优化
  struct CIMMapping {
    std::map<int, int> node_to_cim_mapping;
    std::vector<std::vector<int>> cim_execution_schedule;
    float cim_utilization;
    size_t communication_overhead;
  };
  
  CIMMapping optimize_cim_mapping();
  bool implement_cim_mapping(const CIMMapping &mapping);
  
  // SPM缓存管理
  struct SPMGraphCache {
    struct CacheEntry {
      int node_id;
      DTensor* cached_result;
      size_t access_count;
      std::chrono::time_point<std::chrono::steady_clock> last_access;
      bool is_dirty;
    };
    
    std::vector<CacheEntry> cache_entries;
    size_t total_cache_size;
    size_t used_cache_size;
  };
  
  SPMGraphCache spm_cache_;
  bool cache_intermediate_result(int node_id, const DTensor &result);
  bool prefetch_graph_data(const std::vector<int> &node_sequence);
  
  // 调度器
  class YICAScheduler {
  public:
    struct Task {
      int node_id;
      int priority;
      float estimated_runtime;
      std::vector<int> dependencies;
      bool is_ready;
    };
    
    std::vector<Task> task_queue_;
    std::vector<int> ready_queue_;
    std::vector<int> running_tasks_;
    
    void schedule_task(const Task &task);
    int get_next_ready_task();
    void update_task_status(int task_id, bool completed);
    
  private:
    YICAGraphConfig::SchedulingConfig::SchedulingPolicy policy_;
  };
  
  YICAScheduler scheduler_;
  
  // 内存管理
  struct MemoryManager {
    struct MemoryBlock {
      void* ptr;
      size_t size;
      bool is_free;
      int owner_node_id;
    };
    
    std::vector<MemoryBlock> memory_blocks_;
    size_t total_memory_;
    size_t used_memory_;
    
    void* allocate(size_t size, int node_id);
    void deallocate(void* ptr);
    void optimize_memory_layout();
  };
  
  MemoryManager memory_manager_;
  
  // 性能分析
  struct PerformanceProfiler {
    std::map<int, float> node_execution_times_;
    std::map<int, size_t> node_memory_usage_;
    std::vector<float> cim_utilization_history_;
    std::vector<float> memory_bandwidth_history_;
    
    void record_node_execution(int node_id, float execution_time);
    void record_memory_usage(int node_id, size_t memory_usage);
    void analyze_bottlenecks();
  };
  
  PerformanceProfiler profiler_;
  
  // 自适应优化
  struct AdaptiveOptimizer {
    std::map<GraphOptimizationStrategy, float> strategy_performance_;
    GraphOptimizationStrategy current_best_strategy_;
    std::vector<float> optimization_history_;
    bool adaptation_needed_;
    
    void update_strategy_performance(GraphOptimizationStrategy strategy, float performance);
    GraphOptimizationStrategy select_best_strategy();
    void adapt_optimization_parameters();
  };
  
  AdaptiveOptimizer adaptive_optimizer_;
  
  // YIS指令生成
  std::vector<yica::YISInstruction> generate_graph_execution_instructions();
  std::vector<yica::YISInstruction> generate_fusion_instructions(const FusionGroup &group);
  std::vector<yica::YISInstruction> generate_parallel_instructions(const ParallelizationPlan &plan);
  
  // 图验证和调试
  bool validate_graph_optimization();
  void dump_graph_analysis(const std::string &filename);
  void trace_graph_execution();

  // 内部状态
  YICAGraphMetrics performance_metrics_;
  std::vector<yica::YISInstruction> generated_instructions_;
  bool is_optimized_;
};

/**
 * @brief YICA Graph工厂函数
 */
YICAGraphManager* create_yica_graph_manager(const YICAGraphManager::YICAGraphConfig &config = {});

/**
 * @brief YICA图优化操作
 */
class YICAGraphOptimizationOp : public yirage::kernel::KNOperator {
public:
  YICAGraphOptimizationOp(Graph *_graph,
                          const YICAGraphManager::YICAGraphConfig &config = {});
  
  bool profile(ProfileResult &profile) override;
  bool fingerprint(void) override;
  operator json() const override;
  
  // 特定优化操作
  bool perform_operator_fusion();
  bool perform_memory_optimization();
  bool perform_parallelization();
  bool perform_cim_mapping();

private:
  std::unique_ptr<YICAGraphManager> graph_manager_;
};

/**
 * @brief YICA Graph辅助函数
 */
namespace yica_graph_utils {
  
  /**
   * @brief 分析图的YICA适配性
   */
  struct YICACompatibilityAnalysis {
    float cim_compatibility_score;
    float spm_benefit_score;
    float parallelization_score;
    std::vector<std::string> compatibility_issues;
    std::vector<std::string> optimization_recommendations;
  };
  
  YICACompatibilityAnalysis analyze_yica_compatibility(const Graph &graph);
  
  /**
   * @brief 估算图优化的性能收益
   */
  struct OptimizationBenefit {
    float latency_improvement;
    float throughput_improvement;
    float memory_reduction;
    float energy_savings;
    float overall_speedup;
  };
  
  OptimizationBenefit estimate_optimization_benefit(
    const Graph &original_graph,
    const YICAGraphManager::YICAGraphConfig &config);
  
  /**
   * @brief 生成图执行计划
   */
  struct ExecutionPlan {
    std::vector<int> execution_order;
    std::map<int, int> operator_to_cim_mapping;
    std::vector<std::pair<int, int>> fusion_pairs;
    std::vector<std::vector<int>> parallel_groups;
  };
  
  ExecutionPlan generate_execution_plan(
    const Graph &graph,
    const YICAGraphManager::YICAGraphConfig &config);
  
  /**
   * @brief 验证图优化的正确性
   */
  bool validate_graph_optimization(
    const Graph &original_graph,
    const Graph &optimized_graph);
  
  /**
   * @brief 可视化图结构和优化结果
   */
  void visualize_graph_optimization(
    const Graph &graph,
    const YICAGraphManager::GraphAnalysis &analysis,
    const std::string &output_file);

} // namespace yica_graph_utils

void from_json(json const &j, YICAGraphManager &manager);
void from_json(json const &j, YICAGraphOptimizationOp &op);

} // namespace kernel
} // namespace yirage 