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
 * @brief YICA专用Reduction算子
 * 利用CIM阵列实现高效的并行归约操作
 */
class YICAReductionOp : public yirage::kernel::KNOperator {
public:
  YICAReductionOp(Graph *_graph, 
                  DTensor const &input, 
                  int dim, 
                  int size,
                  const YICAReductionConfig &config = {});
  ~YICAReductionOp();
  
  bool profile(ProfileResult &profile) override;
  bool fingerprint(void) override;
  operator json() const override;

  // YICA特定功能
  bool optimize_for_cim_reduction(int num_cim_arrays);
  bool enable_hierarchical_reduction();
  bool use_tree_reduction_pattern();
  bool enable_spm_intermediate_buffering(size_t buffer_size);
  
  // 归约操作类型
  enum class ReductionType {
    SUM,          // 求和
    MEAN,         // 平均值
    MAX,          // 最大值
    MIN,          // 最小值
    PROD,         // 乘积
    L1_NORM,      // L1范数
    L2_NORM,      // L2范数
    VARIANCE,     // 方差
    STD_DEV,      // 标准差
    CUSTOM        // 自定义归约
  };
  
  bool set_reduction_type(ReductionType type);
  
  // 归约模式
  enum class ReductionMode {
    FULL_REDUCTION,      // 完全归约
    PARTIAL_REDUCTION,   // 部分归约
    SLIDING_WINDOW,      // 滑动窗口归约
    STRIDED_REDUCTION,   // 跨步归约
    HIERARCHICAL         // 层次化归约
  };

  // YICA性能分析
  struct YICAReductionMetrics {
    float cim_reduction_efficiency;      // CIM归约效率
    float parallel_utilization;         // 并行利用率
    size_t yis_reduction_instruction_count; // YIS归约指令数
    float memory_access_efficiency;     // 内存访问效率
    float tree_reduction_depth;         // 树形归约深度
    size_t total_reduction_operations;  // 总归约操作数
    float load_balance_factor;          // 负载均衡因子
  };
  
  YICAReductionMetrics get_yica_metrics() const;

public:
  // YICA Reduction配置
  struct YICAReductionConfig {
    // CIM配置
    int num_cim_arrays = 8;              // CIM阵列数量
    bool enable_parallel_reduction = true; // 启用并行归约
    int reduction_tree_fanout = 4;       // 归约树扇出度
    
    // SPM配置
    size_t spm_buffer_size = 16 * 1024 * 1024; // SPM缓冲区大小 (16MB)
    bool enable_spm_staging = true;      // 启用SPM暂存
    
    // 归约配置
    ReductionType reduction_type = ReductionType::SUM;
    ReductionMode reduction_mode = ReductionMode::HIERARCHICAL;
    bool enable_numerical_stability = true; // 启用数值稳定性
    
    // 优化配置
    bool enable_vectorization = true;    // 启用向量化
    bool enable_loop_unrolling = true;   // 启用循环展开
    int unroll_factor = 4;               // 展开因子
    
    // 内存优化
    bool enable_memory_coalescing = true; // 启用内存合并
    bool enable_prefetching = true;      // 启用数据预取
    size_t prefetch_distance = 64;      // 预取距离
    
    // 负载均衡
    bool enable_dynamic_load_balancing = true; // 启用动态负载均衡
    float load_balance_threshold = 0.8f; // 负载均衡阈值
  } yica_config;

private:
  // YIS指令生成
  std::vector<yica::YISInstruction> generate_reduction_instructions();
  std::vector<yica::YISInstruction> generate_tree_reduction_instructions();
  std::vector<yica::YISInstruction> generate_hierarchical_reduction_instructions();
  
  // CIM归约优化
  struct CIMReductionPlan {
    std::vector<int> cim_array_assignment;
    std::vector<std::vector<int>> reduction_schedule;
    int tree_depth;
    float expected_efficiency;
    size_t total_operations;
  };
  
  CIMReductionPlan plan_cim_reduction();
  bool optimize_reduction_tree();
  
  // SPM缓冲区管理
  struct SPMReductionLayout {
    size_t input_staging_offset;
    size_t intermediate_result_offset;
    size_t final_result_offset;
    std::vector<size_t> level_offsets;
    size_t total_usage;
  };
  
  SPMReductionLayout plan_spm_reduction_layout();
  bool setup_hierarchical_buffering();
  
  // 归约算法实现
  std::vector<yica::YISInstruction> implement_sum_reduction();
  std::vector<yica::YISInstruction> implement_max_min_reduction();
  std::vector<yica::YISInstruction> implement_norm_reduction();
  std::vector<yica::YISInstruction> implement_statistical_reduction();
  
  // 数值稳定性
  bool apply_kahan_summation();
  bool apply_pairwise_summation();
  float calculate_numerical_error_bound();
  
  // 负载均衡
  struct LoadBalanceResult {
    std::vector<int> workload_distribution;
    float balance_factor;
    bool rebalancing_needed;
  };
  
  LoadBalanceResult analyze_load_balance();
  bool rebalance_workload();
  
  // 性能预测
  float predict_reduction_time();
  size_t estimate_memory_bandwidth_requirement();
  float calculate_parallel_efficiency();

  // 内部状态
  int reduction_dim_idx_;
  int reduction_dim_size_;
  ReductionType reduction_type_;
  YICAReductionMetrics performance_metrics_;
  std::vector<yica::YISInstruction> generated_instructions_;
};

/**
 * @brief YICA Reduction工厂函数
 */
YICAReductionOp* create_yica_reduction(
  Graph *graph,
  DTensor const &input,
  int dim,
  int size,
  const YICAReductionOp::YICAReductionConfig &config = {});

/**
 * @brief YICA Reduction辅助函数
 */
namespace yica_reduction_utils {
  
  /**
   * @brief 计算最优的归约树结构
   */
  struct ReductionTree {
    int depth;
    int fanout;
    std::vector<std::vector<int>> levels;
    float expected_efficiency;
  };
  
  ReductionTree calculate_optimal_reduction_tree(
    int input_size,
    int num_cim_arrays,
    YICAReductionOp::ReductionType type);
  
  /**
   * @brief 估算归约操作的数值误差
   */
  float estimate_numerical_error(
    YICAReductionOp::ReductionType type,
    size_t input_size,
    float input_range);
  
  /**
   * @brief 生成向量化归约指令
   */
  std::vector<yica::YISInstruction> generate_vectorized_reduction(
    const DTensor& input,
    YICAReductionOp::ReductionType type,
    int vector_width);
  
  /**
   * @brief 优化归约的内存访问模式
   */
  struct ReductionAccessPattern {
    std::vector<size_t> read_pattern;
    std::vector<size_t> write_pattern;
    size_t stride;
    bool coalesced;
    float cache_efficiency;
  };
  
  ReductionAccessPattern optimize_reduction_access_pattern(
    const std::vector<int>& tensor_shape,
    int reduction_dim,
    int num_cim_arrays);

} // namespace yica_reduction_utils

void from_json(json const &j, YICAReductionOp &op);

} // namespace kernel
} // namespace yirage 