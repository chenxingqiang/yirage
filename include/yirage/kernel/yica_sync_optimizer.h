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

#include "yirage/kernel/graph.h"
#include "yirage/yica/yis_instruction_set.h"
#include <vector>
#include <memory>

namespace yirage {
namespace kernel {

/**
 * @brief YICA同步优化器
 * 利用YISSYNC指令实现高效的多层次同步机制
 */
class YICASyncOptimizer {
public:
  YICASyncOptimizer();
  ~YICASyncOptimizer();

  // 同步层次枚举
  enum class SyncLevel {
    THREAD_LEVEL,    // 线程级同步
    WARP_LEVEL,      // Warp级同步
    WORKGROUP_LEVEL, // 工作组级同步 (WG)
    CWG_LEVEL,       // 计算工作组级同步 (CWG)
    SWG_LEVEL,       // 超级工作组级同步 (SWG)
    GRID_LEVEL,      // 网格级同步
    DEVICE_LEVEL     // 设备级同步
  };

  // 同步模式
  enum class SyncMode {
    BARRIER,         // 栅栏同步 - YISSYNC_BAR
    BUFFER_OBJECT,   // 缓冲区对象同步 - YISSYNC_BO*
    PRODUCER_CONSUMER, // 生产者-消费者模式
    PIPELINE_SYNC,   // 流水线同步
    MEMORY_FENCE     // 内存屏障
  };

  // 同步优化计划
  struct SyncOptimizationPlan {
    std::vector<yica::YISInstruction> sync_instructions;
    std::vector<SyncLevel> sync_hierarchy;
    SyncMode primary_sync_mode;
    float estimated_sync_overhead_percent;
    size_t total_sync_points;
    bool optimization_successful;
  };

  /**
   * @brief 分析计算图的同步需求
   */
  std::vector<SyncLevel> analyze_sync_requirements(const Graph& graph);

  /**
   * @brief 生成同步优化计划
   */
  SyncOptimizationPlan optimize_synchronization(
    const Graph& graph,
    const std::vector<SyncLevel>& required_levels);

  /**
   * @brief 生成YISSYNC_BAR指令 (栅栏同步)
   */
  yica::YISInstruction generate_barrier_sync(
    SyncLevel level,
    const std::string& scope = "WG");

  /**
   * @brief 生成缓冲区对象同步指令序列
   */
  std::vector<yica::YISInstruction> generate_buffer_object_sync(
    int buffer_id,
    const std::vector<DTensor>& dependent_tensors);

  /**
   * @brief 优化生产者-消费者同步模式
   */
  struct ProducerConsumerSync {
    std::vector<yica::YISInstruction> producer_instructions;
    std::vector<yica::YISInstruction> consumer_instructions;
    std::vector<yica::YISInstruction> sync_instructions;
    float pipeline_efficiency;
  };

  ProducerConsumerSync optimize_producer_consumer_sync(
    const std::vector<KNOperator*>& producers,
    const std::vector<KNOperator*>& consumers);

  /**
   * @brief 生成流水线同步指令
   */
  std::vector<yica::YISInstruction> generate_pipeline_sync(
    const std::vector<std::vector<KNOperator*>>& pipeline_stages);

  /**
   * @brief 内存屏障优化
   */
  enum class MemoryFenceType {
    ACQUIRE,  // 获取屏障
    RELEASE,  // 释放屏障
    ACQ_REL,  // 获取-释放屏障
    SEQ_CST   // 顺序一致性屏障
  };

  yica::YISInstruction generate_memory_fence(
    MemoryFenceType fence_type,
    SyncLevel scope_level);

  /**
   * @brief 同步性能分析
   */
  struct SyncPerformanceAnalysis {
    float sync_overhead_cycles;
    float parallel_efficiency;
    size_t critical_path_length;
    std::vector<std::string> bottleneck_operations;
    float load_balance_factor;
  };

  SyncPerformanceAnalysis analyze_sync_performance(
    const SyncOptimizationPlan& plan);

  /**
   * @brief 自适应同步优化
   * 根据运行时性能动态调整同步策略
   */
  struct AdaptiveSyncConfig {
    bool enable_dynamic_adjustment = true;
    float performance_threshold = 0.8f;
    int adjustment_window_size = 100;
    bool prefer_low_latency = true;
  };

  bool enable_adaptive_sync(const AdaptiveSyncConfig& config);
  void update_sync_strategy(const SyncPerformanceAnalysis& analysis);

private:
  // 同步依赖分析
  std::vector<std::pair<KNOperator*, KNOperator*>> analyze_sync_dependencies(
    const Graph& graph);
  
  // 同步层次优化
  SyncLevel determine_optimal_sync_level(
    const std::vector<KNOperator*>& dependent_ops);
  
  // 同步指令调度
  void schedule_sync_instructions(
    std::vector<yica::YISInstruction>& instructions);
  
  // 性能模型
  float estimate_sync_cost(const yica::YISInstruction& sync_instruction);
  float calculate_parallel_efficiency(const SyncOptimizationPlan& plan);
  
  // 自适应优化状态
  struct AdaptiveState {
    std::vector<float> recent_performance_samples;
    SyncMode current_preferred_mode;
    float current_efficiency;
    bool adaptation_enabled;
  } adaptive_state_;

  // 内部配置
  AdaptiveSyncConfig adaptive_config_;
  std::vector<yica::YISInstruction> generated_sync_instructions_;
};

/**
 * @brief YICA同步优化的辅助函数
 */
namespace yica_sync_utils {
  
  /**
   * @brief 检查操作是否需要同步
   */
  bool requires_synchronization(const KNOperator* op1, const KNOperator* op2);
  
  /**
   * @brief 计算同步点的最优位置
   */
  std::vector<size_t> find_optimal_sync_points(
    const std::vector<KNOperator*>& operations);
  
  /**
   * @brief 估算同步开销
   */
  float estimate_sync_overhead(SyncLevel level, SyncMode mode);
  
  /**
   * @brief 生成同步指令的调试信息
   */
  std::string generate_sync_debug_info(const yica::YISInstruction& instruction);

} // namespace yica_sync_utils

} // namespace kernel
} // namespace yirage 