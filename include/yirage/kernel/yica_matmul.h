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
 * @brief YICA专用矩阵乘法算子
 * 针对存算一体架构优化，充分利用CIM阵列的并行计算能力
 */
class YICAMatMulOp : public yirage::kernel::KNOperator {
public:
  YICAMatMulOp(Graph *_graph,
               DTensor const &A,
               DTensor const &B,
               DTensor const &C,
               const YICAMatMulConfig &config = {});
  ~YICAMatMulOp();
  
  bool profile(ProfileResult &profile) override;
  bool fingerprint(void) override;
  operator json() const override;

  // MatMul操作类型
  enum class MatMulType {
    GEMM,           // 通用矩阵乘法
    GEMV,           // 矩阵向量乘法
    BATCHED_GEMM,   // 批量矩阵乘法
    SPARSE_GEMM,    // 稀疏矩阵乘法
    QUANTIZED_GEMM, // 量化矩阵乘法
    FUSED_GEMM,     // 融合矩阵乘法
    TENSOR_CORE,    // 张量核心乘法
    CIM_NATIVE      // CIM原生乘法
  };

  // YICA特定功能
  bool optimize_for_cim_arrays(int num_cim_arrays);
  bool enable_spm_data_staging();
  bool use_yismma_instructions();
  bool enable_mixed_precision();
  
  // 计算策略
  enum class ComputeStrategy {
    THROUGHPUT_OPTIMAL,  // 吞吐量最优
    LATENCY_OPTIMAL,     // 延迟最优
    MEMORY_OPTIMAL,      // 内存最优
    ENERGY_OPTIMAL,      // 能耗最优
    ACCURACY_OPTIMAL,    // 精度最优
    ADAPTIVE            // 自适应策略
  };
  
  bool set_compute_strategy(ComputeStrategy strategy);
  ComputeStrategy recommend_strategy(const DTensor &A, const DTensor &B);

  // YICA性能分析
  struct YICAMatMulMetrics {
    float cim_utilization;              // CIM利用率
    float spm_hit_rate;                 // SPM命中率
    size_t yismma_instruction_count;    // YISMMA指令数
    float mixed_precision_ratio;        // 混合精度比率
    float memory_bandwidth_efficiency;  // 内存带宽效率
    size_t total_multiply_accumulates;  // 总乘累加操作数
    float compute_intensity;            // 计算强度
    float numerical_accuracy;           // 数值精度
  };
  
  YICAMatMulMetrics get_yica_metrics() const;

public:
  // YICA MatMul配置
  struct YICAMatMulConfig {
    // 计算策略配置
    ComputeStrategy compute_strategy = ComputeStrategy::THROUGHPUT_OPTIMAL;
    MatMulType matmul_type = MatMulType::CIM_NATIVE;
    bool enable_auto_tuning = true;
    
    // CIM配置
    int num_cim_arrays = 32;             // CIM阵列数量
    int cim_array_rows = 256;            // CIM阵列行数
    int cim_array_cols = 256;            // CIM阵列列数
    bool enable_dynamic_cim_mapping = true; // 动态CIM映射
    
    // SPM配置
    size_t spm_a_buffer_size = 64 * 1024 * 1024;  // SPM A矩阵缓冲 (64MB)
    size_t spm_b_buffer_size = 64 * 1024 * 1024;  // SPM B矩阵缓冲 (64MB)
    size_t spm_c_buffer_size = 32 * 1024 * 1024;  // SPM C矩阵缓冲 (32MB)
    bool enable_spm_double_buffering = true;       // SPM双缓冲
    
    // 数据精度配置
    struct PrecisionConfig {
      enum class DataType {
        FP32, FP16, BF16, INT8, INT4, MIXED
      } input_precision = DataType::FP16;
      
      DataType weight_precision = DataType::FP16;
      DataType accumulator_precision = DataType::FP32;
      DataType output_precision = DataType::FP16;
      bool enable_dynamic_quantization = true;
      float quantization_threshold = 0.1f;
    } precision_config;
    
    // 分块配置
    struct TilingConfig {
      int tile_m = 128;                  // M维度分块大小
      int tile_n = 128;                  // N维度分块大小
      int tile_k = 64;                   // K维度分块大小
      bool enable_adaptive_tiling = true; // 自适应分块
      bool enable_hierarchical_tiling = true; // 层次分块
    } tiling_config;
    
    // 内存访问优化
    struct MemoryConfig {
      bool enable_prefetching = true;     // 启用预取
      bool enable_data_reuse = true;      // 启用数据复用
      bool enable_weight_stationary = true; // 权重驻留
      int prefetch_distance = 2;          // 预取距离
      size_t alignment_requirement = 64;  // 对齐要求
    } memory_config;
    
    // 并行配置
    struct ParallelConfig {
      bool enable_data_parallel = true;   // 数据并行
      bool enable_model_parallel = true;  // 模型并行
      int parallel_degree = 4;            // 并行度
      bool enable_pipeline_parallel = true; // 流水线并行
      int pipeline_stages = 3;            // 流水线阶段数
    } parallel_config;
    
    // 融合操作配置
    struct FusionConfig {
      bool enable_bias_fusion = true;     // 偏置融合
      bool enable_activation_fusion = true; // 激活函数融合
      bool enable_batch_norm_fusion = true; // 批归一化融合
      bool enable_residual_fusion = true;  // 残差连接融合
      std::vector<std::string> fused_operations; // 融合操作列表
    } fusion_config;
    
    // 性能优化配置
    bool enable_loop_unrolling = true;   // 循环展开
    bool enable_vectorization = true;    // 向量化
    int vector_width = 8;                // 向量宽度
    bool enable_instruction_scheduling = true; // 指令调度
  } yica_config;

private:
  // YIS指令生成
  std::vector<yica::YISInstruction> generate_matmul_instructions();
  std::vector<yica::YISInstruction> generate_yismma_instructions();
  std::vector<yica::YISInstruction> generate_data_movement_instructions();
  std::vector<yica::YISInstruction> generate_fusion_instructions();
  
  // CIM映射优化
  struct CIMMapping {
    std::vector<std::vector<int>> cim_tile_mapping;  // CIM分块映射
    std::vector<int> cim_workload_distribution;     // CIM工作量分配
    float load_balance_factor;                      // 负载均衡因子
    size_t communication_overhead;                  // 通信开销
  };
  
  CIMMapping optimize_cim_mapping();
  bool implement_cim_parallel_computation();
  
  // SPM数据管理
  struct SPMDataPlan {
    struct BufferPlan {
      size_t offset;
      size_t size;
      bool is_double_buffered;
      std::vector<int> access_pattern;
    };
    
    BufferPlan a_buffer_plan;
    BufferPlan b_buffer_plan;
    BufferPlan c_buffer_plan;
    size_t total_spm_usage;
    float reuse_efficiency;
  };
  
  SPMDataPlan plan_spm_data_management();
  bool implement_spm_staging();
  
  // 分块优化
  struct TilingPlan {
    struct Tile {
      int m_start, m_end;
      int n_start, n_end;
      int k_start, k_end;
      int cim_array_id;
      float computation_cost;
    };
    
    std::vector<Tile> tiles;
    int optimal_tile_m, optimal_tile_n, optimal_tile_k;
    float tiling_efficiency;
    size_t memory_footprint;
  };
  
  TilingPlan optimize_matrix_tiling();
  bool implement_hierarchical_tiling();
  
  // 混合精度优化
  struct MixedPrecisionPlan {
    std::vector<bool> use_low_precision_tiles;
    std::map<int, YICAMatMulConfig::PrecisionConfig::DataType> tile_precision_map;
    float accuracy_loss_estimate;
    float performance_gain_estimate;
  };
  
  MixedPrecisionPlan analyze_mixed_precision_opportunities();
  bool implement_mixed_precision_computation();
  
  // 数据预取优化
  struct PrefetchPlan {
    std::vector<std::pair<int, int>> prefetch_schedule; // (tile_id, prefetch_time)
    std::vector<size_t> prefetch_addresses;
    int optimal_prefetch_distance;
    float prefetch_efficiency;
  };
  
  PrefetchPlan optimize_data_prefetching();
  bool implement_prefetch_strategy();
  
  // 融合操作实现
  struct FusionImplementation {
    std::vector<std::string> fused_op_sequence;
    std::vector<yica::YISInstruction> fused_instructions;
    float fusion_benefit;
    size_t memory_savings;
  };
  
  FusionImplementation implement_operation_fusion();
  
  // 自动调优
  struct AutoTuner {
    struct TuningParameter {
      std::string name;
      std::vector<int> candidate_values;
      int current_best_value;
      float best_performance;
    };
    
    std::vector<TuningParameter> tuning_parameters_;
    std::map<std::string, float> performance_history_;
    bool tuning_converged_;
    
    void add_tuning_parameter(const std::string &name, const std::vector<int> &candidates);
    bool perform_auto_tuning();
    void update_best_configuration();
  };
  
  AutoTuner auto_tuner_;
  
  // 数值稳定性
  struct NumericalStability {
    float condition_number_estimate;
    bool requires_extended_precision;
    std::vector<int> problematic_tiles;
    float accuracy_threshold;
  };
  
  NumericalStability analyze_numerical_stability();
  bool ensure_numerical_accuracy();
  
  // 性能建模
  struct PerformanceModel {
    float predicted_latency_ms;
    float predicted_throughput_gops;
    size_t predicted_memory_usage;
    float predicted_energy_consumption;
    
    void update_model_parameters();
    float predict_performance(const YICAMatMulConfig &config);
  };
  
  PerformanceModel performance_model_;
  
  // 调试和验证
  bool validate_matmul_correctness();
  void dump_computation_trace();
  void analyze_performance_bottlenecks();

  // 内部状态
  DTensor A_, B_, C_;
  MatMulType matmul_type_;
  YICAMatMulMetrics performance_metrics_;
  std::vector<yica::YISInstruction> generated_instructions_;
};

/**
 * @brief YICA MatMul工厂函数
 */
YICAMatMulOp* create_yica_matmul_op(
  Graph *graph,
  DTensor const &A,
  DTensor const &B,
  DTensor const &C,
  const YICAMatMulOp::YICAMatMulConfig &config = {});

/**
 * @brief YICA批量MatMul操作
 */
class YICABatchedMatMulOp : public YICAMatMulOp {
public:
  YICABatchedMatMulOp(Graph *_graph,
                      const std::vector<DTensor> &A_batch,
                      const std::vector<DTensor> &B_batch,
                      const std::vector<DTensor> &C_batch,
                      const YICAMatMulConfig &config = {});
  
  // 批量特定优化
  bool optimize_batch_scheduling();
  bool enable_batch_parallelism();
  
  struct BatchOptimization {
    std::vector<int> batch_execution_order;
    std::vector<std::vector<int>> parallel_batch_groups;
    float batch_efficiency;
    size_t shared_memory_usage;
  };
  
  BatchOptimization optimize_batch_execution();

private:
  std::vector<DTensor> A_batch_, B_batch_, C_batch_;
  BatchOptimization batch_optimization_;
};

/**
 * @brief YICA MatMul辅助函数
 */
namespace yica_matmul_utils {
  
  /**
   * @brief 计算最优的矩阵分块大小
   */
  struct OptimalTiling {
    int tile_m, tile_n, tile_k;
    float efficiency_score;
    size_t memory_requirement;
    int parallel_degree;
  };
  
  OptimalTiling calculate_optimal_tiling(
    const DTensor &A,
    const DTensor &B,
    const YICAMatMulOp::YICAMatMulConfig &config);
  
  /**
   * @brief 分析矩阵乘法的计算特性
   */
  struct ComputationCharacteristics {
    size_t total_flops;
    float arithmetic_intensity;
    size_t memory_footprint;
    bool is_memory_bound;
    bool is_compute_bound;
  };
  
  ComputationCharacteristics analyze_computation_characteristics(
    const DTensor &A,
    const DTensor &B);
  
  /**
   * @brief 估算MatMul性能
   */
  struct MatMulPerformanceEstimate {
    float estimated_latency_ms;
    float estimated_throughput_gops;
    float estimated_energy_mj;
    float cim_utilization_estimate;
  };
  
  MatMulPerformanceEstimate estimate_matmul_performance(
    const DTensor &A,
    const DTensor &B,
    const YICAMatMulOp::YICAMatMulConfig &config);
  
  /**
   * @brief 生成CIM友好的数据布局
   */
  enum class DataLayout {
    ROW_MAJOR,           // 行主序
    COLUMN_MAJOR,        // 列主序
    BLOCKED,             // 分块布局
    CIM_OPTIMIZED,       // CIM优化布局
    HIERARCHICAL         // 层次布局
  };
  
  DataLayout recommend_data_layout(
    const DTensor &tensor,
    YICAMatMulOp::MatMulType matmul_type);
  
  /**
   * @brief 验证矩阵乘法结果
   */
  bool validate_matmul_result(
    const DTensor &A,
    const DTensor &B,
    const DTensor &C_computed,
    const DTensor &C_reference,
    float tolerance = 1e-5f);

} // namespace yica_matmul_utils

void from_json(json const &j, YICAMatMulOp &op);

} // namespace kernel
} // namespace yirage 