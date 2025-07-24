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
#include "yirage/kernel/operator.h"
#include "yirage/yica/yis_instruction_set.h"
#include "yirage/threadblock/graph.h"
#include <vector>
#include <memory>
#include <functional>

namespace yirage {
namespace kernel {

/**
 * @brief YICA专用自定义操作算子
 * 支持用户定义的复杂计算模式，充分利用YICA的存算一体特性
 */
class YICACustomizedOp : public yirage::kernel::KNOperator {
public:
  YICACustomizedOp(Graph *_kgraph,
                   std::vector<DTensor> const &inputs,
                   yirage::threadblock::Graph const &_bgraph,
                   const YICACustomizedConfig &config = {});
  
  virtual ~YICACustomizedOp();
  
  bool profile(ProfileResult &profile) override;
  void run(void);
  bool fingerprint(void) override;
  size_t get_owner_independent_hash() const override;
  operator json() const override;

  // 自定义计算模式
  enum class CustomComputeMode {
    THREAD_BLOCK_FUSION,     // 线程块融合模式
    CIM_ARRAY_MAPPING,       // CIM阵列映射模式
    SPM_OPTIMIZED,           // SPM优化模式
    PIPELINE_PARALLEL,       // 流水线并行模式
    HYBRID_COMPUTE,          // 混合计算模式
    USER_DEFINED             // 用户自定义模式
  };

  // YICA特定功能
  bool optimize_for_cim_arrays(int num_arrays);
  bool enable_spm_staging(size_t staging_size);
  bool use_custom_yis_instructions();
  bool enable_threadblock_fusion();
  
  // 自定义计算图管理
  void set_threadblock_graph(const yirage::threadblock::Graph &bgraph);
  yirage::threadblock::Graph get_threadblock_graph() const;
  bool validate_threadblock_graph();
  
  // 用户自定义函数接口
  using CustomComputeFunction = std::function<void(
    const std::vector<DTensor>&,  // inputs
    std::vector<DTensor>&,        // outputs
    const YICACustomizedConfig&   // config
  )>;
  
  bool set_custom_compute_function(CustomComputeFunction func);
  bool register_yis_instruction_pattern(const std::string &pattern_name,
                                       const std::vector<yica::YISInstruction> &instructions);

  // YICA性能分析
  struct YICACustomizedMetrics {
    float threadblock_fusion_efficiency;    // 线程块融合效率
    float cim_array_utilization;           // CIM阵列利用率
    float spm_staging_efficiency;          // SPM暂存效率
    size_t custom_yis_instruction_count;   // 自定义YIS指令数
    float pipeline_parallel_efficiency;    // 流水线并行效率
    size_t total_custom_operations;        // 总自定义操作数
    float compute_memory_overlap;          // 计算内存重叠度
    float user_function_overhead;          // 用户函数开销
  };
  
  YICACustomizedMetrics get_yica_metrics() const;

public:
  // YICA自定义操作配置
  struct YICACustomizedConfig {
    // 计算模式配置
    CustomComputeMode compute_mode = CustomComputeMode::CIM_ARRAY_MAPPING;
    bool enable_auto_optimization = true;
    
    // CIM阵列配置
    int num_cim_arrays = 8;              // CIM阵列数量
    bool enable_dynamic_cim_allocation = true; // 动态CIM分配
    std::vector<int> cim_array_priorities;     // CIM阵列优先级
    
    // SPM配置
    size_t spm_staging_size = 64 * 1024 * 1024; // SPM暂存大小 (64MB)
    bool enable_spm_double_buffering = true;     // 启用SPM双缓冲
    bool enable_spm_prefetching = true;          // 启用SPM预取
    
    // 线程块配置
    struct ThreadBlockConfig {
      int block_size_x = 256;
      int block_size_y = 1;
      int block_size_z = 1;
      int grid_size_x = 1;
      int grid_size_y = 1;
      int grid_size_z = 1;
      bool enable_cooperative_groups = true;
    } threadblock_config;
    
    // 流水线配置
    struct PipelineConfig {
      int pipeline_stages = 4;
      bool enable_stage_overlapping = true;
      size_t stage_buffer_size = 8 * 1024 * 1024; // 8MB per stage
      bool enable_dynamic_scheduling = true;
    } pipeline_config;
    
    // 用户自定义配置
    std::map<std::string, std::string> user_parameters;
    std::vector<std::string> custom_yis_patterns;
    
    // 性能配置
    bool enable_performance_monitoring = true;
    bool enable_adaptive_optimization = true;
    float optimization_threshold = 0.8f;
    
    // 调试配置
    bool enable_debug_output = false;
    bool enable_instruction_tracing = false;
    std::string debug_output_file = "yica_custom_debug.log";
  } yica_config;

private:
  // 核心组件
  yirage::threadblock::Graph bgraph_;
  CustomComputeFunction custom_function_;
  std::map<std::string, std::vector<yica::YISInstruction>> yis_patterns_;
  
  // YIS指令生成
  std::vector<yica::YISInstruction> generate_custom_instructions();
  std::vector<yica::YISInstruction> generate_threadblock_fusion_instructions();
  std::vector<yica::YISInstruction> generate_cim_mapping_instructions();
  std::vector<yica::YISInstruction> generate_pipeline_instructions();
  
  // CIM阵列优化
  struct CIMArrayMapping {
    std::vector<int> tensor_to_cim_mapping;
    std::vector<std::vector<int>> cim_computation_schedule;
    float expected_utilization;
    size_t memory_footprint;
  };
  
  CIMArrayMapping plan_cim_array_mapping();
  bool optimize_cim_computation_schedule();
  
  // SPM暂存优化
  struct SPMStagingPlan {
    std::vector<std::pair<DTensor*, size_t>> staging_allocation;
    std::vector<size_t> prefetch_schedule;
    bool double_buffering_enabled;
    size_t total_staging_usage;
  };
  
  SPMStagingPlan plan_spm_staging();
  bool setup_spm_double_buffering();
  
  // 线程块融合
  struct ThreadBlockFusion {
    std::vector<yirage::threadblock::TBOperator*> fused_operators;
    std::vector<int> fusion_boundaries;
    float fusion_efficiency;
    size_t shared_memory_usage;
  };
  
  ThreadBlockFusion analyze_threadblock_fusion();
  bool implement_threadblock_fusion();
  
  // 流水线并行
  struct PipelineStage {
    std::vector<yirage::threadblock::TBOperator*> stage_operators;
    size_t stage_buffer_size;
    std::vector<yica::YISInstruction> stage_instructions;
    float stage_latency;
  };
  
  std::vector<PipelineStage> plan_pipeline_stages();
  bool implement_pipeline_parallel();
  
  // 自适应优化
  struct AdaptiveOptimization {
    std::vector<float> performance_history;
    CustomComputeMode current_best_mode;
    std::map<CustomComputeMode, float> mode_performance;
    bool optimization_needed;
  };
  
  AdaptiveOptimization adaptive_state_;
  bool perform_adaptive_optimization();
  
  // 性能监控
  void update_performance_metrics();
  void log_execution_statistics();
  
  // 调试和诊断
  void dump_threadblock_graph(const std::string &filename);
  void trace_yis_instruction_execution();
  bool validate_custom_computation();

  // 内部状态
  YICACustomizedMetrics performance_metrics_;
  std::vector<yica::YISInstruction> generated_instructions_;
  bool is_optimized_;
};

/**
 * @brief YICA自定义操作工厂函数
 */
YICACustomizedOp* create_yica_customized_op(
  Graph *graph,
  std::vector<DTensor> const &inputs,
  yirage::threadblock::Graph const &bgraph,
  const YICACustomizedOp::YICACustomizedConfig &config = {});

/**
 * @brief YICA自定义操作构建器
 */
class YICACustomizedOpBuilder {
public:
  YICACustomizedOpBuilder(Graph *graph);
  
  // 输入输出管理
  YICACustomizedOpBuilder& add_input(const DTensor &input);
  YICACustomizedOpBuilder& add_inputs(const std::vector<DTensor> &inputs);
  YICACustomizedOpBuilder& set_threadblock_graph(const yirage::threadblock::Graph &bgraph);
  
  // 配置设置
  YICACustomizedOpBuilder& set_compute_mode(YICACustomizedOp::CustomComputeMode mode);
  YICACustomizedOpBuilder& set_cim_arrays(int num_arrays);
  YICACustomizedOpBuilder& set_spm_staging_size(size_t size);
  YICACustomizedOpBuilder& enable_pipeline_parallel(int stages);
  
  // 用户自定义函数
  YICACustomizedOpBuilder& set_custom_function(YICACustomizedOp::CustomComputeFunction func);
  YICACustomizedOpBuilder& add_yis_pattern(const std::string &name, 
                                          const std::vector<yica::YISInstruction> &instructions);
  
  // 构建操作
  YICACustomizedOp* build();

private:
  Graph *graph_;
  std::vector<DTensor> inputs_;
  yirage::threadblock::Graph bgraph_;
  YICACustomizedOp::YICACustomizedConfig config_;
  YICACustomizedOp::CustomComputeFunction custom_function_;
  std::map<std::string, std::vector<yica::YISInstruction>> yis_patterns_;
};

/**
 * @brief YICA自定义操作辅助函数
 */
namespace yica_customized_utils {
  
  /**
   * @brief 分析线程块图的YICA优化潜力
   */
  struct OptimizationPotential {
    float cim_friendliness_score;
    float spm_benefit_score;
    float fusion_potential_score;
    std::vector<std::string> optimization_suggestions;
  };
  
  OptimizationPotential analyze_optimization_potential(
    const yirage::threadblock::Graph &bgraph);
  
  /**
   * @brief 生成自定义YIS指令模式
   */
  std::vector<yica::YISInstruction> generate_yis_pattern(
    const std::string &pattern_type,
    const std::map<std::string, std::string> &parameters);
  
  /**
   * @brief 优化线程块配置
   */
  YICACustomizedOp::YICACustomizedConfig::ThreadBlockConfig optimize_threadblock_config(
    const std::vector<DTensor> &inputs,
    const yirage::threadblock::Graph &bgraph);
  
  /**
   * @brief 估算自定义操作的性能
   */
  struct PerformanceEstimate {
    float estimated_latency_ms;
    float estimated_throughput_gops;
    size_t estimated_memory_usage;
    float cim_utilization_estimate;
  };
  
  PerformanceEstimate estimate_custom_operation_performance(
    const std::vector<DTensor> &inputs,
    const yirage::threadblock::Graph &bgraph,
    const YICACustomizedOp::YICACustomizedConfig &config);
  
  /**
   * @brief 验证自定义操作的正确性
   */
  bool validate_custom_operation(
    const std::vector<DTensor> &inputs,
    const std::vector<DTensor> &expected_outputs,
    YICACustomizedOp::CustomComputeFunction func);

} // namespace yica_customized_utils

void from_json(json const &j, YICACustomizedOp &op);

} // namespace kernel
} // namespace yirage 